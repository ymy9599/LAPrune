#!/usr/bin/env python3
"""Train post-RoPE LAPrune residual codebooks on mixed long contexts.

The model forward is chunked to keep calibration practical.  Chunks retain
their absolute position ids and keys are rotated with the model's own rotary
embedding, including Llama-3.1's scaled RoPE frequencies.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


@dataclass
class CorpusDoc:
    source: str
    line: int
    tokens: torch.Tensor


def log(message: str, log_file) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{stamp}] {message}"
    print(text, flush=True)
    print(text, file=log_file, flush=True)


def parse_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or min(lengths) <= 0:
        raise argparse.ArgumentTypeError("length schedule must contain positive integers")
    return lengths


def load_corpus(
    tokenizer,
    corpus_paths: Iterable[str],
    minimum_tokens: int,
    log_file,
) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    for corpus_path in corpus_paths:
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") if isinstance(obj, dict) else None
                if not isinstance(text, str) or not text.strip():
                    continue
                ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
                if len(ids) >= minimum_tokens:
                    docs.append(
                        CorpusDoc(
                            source=str(path),
                            line=line_idx,
                            tokens=torch.tensor(ids, dtype=torch.long),
                        )
                    )
    if not docs:
        raise RuntimeError(f"no corpus documents contain at least {minimum_tokens} tokens")
    lengths = sorted(int(doc.tokens.numel()) for doc in docs)
    log(
        f"corpus cached: docs={len(docs)} min={lengths[0]} "
        f"median={lengths[len(lengths) // 2]} max={lengths[-1]}",
        log_file,
    )
    return docs


def sample_segments(
    docs: list[CorpusDoc],
    schedule: list[int],
    rng: random.Random,
) -> list[tuple[torch.Tensor, int, int]]:
    """Return (token ids, document line, start offset) for each scheduled length."""
    result: list[tuple[torch.Tensor, int, int]] = []
    used: set[tuple[str, int]] = set()
    for length in schedule:
        eligible = [doc for doc in docs if doc.tokens.numel() >= length]
        fresh = [doc for doc in eligible if (doc.source, doc.line) not in used]
        doc = rng.choice(fresh or eligible)
        max_start = int(doc.tokens.numel()) - length
        start = rng.randint(0, max_start) if max_start else 0
        used.add((doc.source, doc.line))
        result.append((doc.tokens[start : start + length], doc.line, start))
    return result


@torch.no_grad()
def collect_post_rope_keys(
    model,
    token_ids: torch.Tensor,
    layer_idx: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    config = model.config
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(config.hidden_size // num_heads)
    num_kv_groups = num_heads // num_kv_heads
    pieces: list[torch.Tensor] = []
    total = int(token_ids.numel())

    class _StopAtTargetLayer(Exception):
        pass

    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module, args, kwargs):
        captured["hidden"] = kwargs.get("hidden_states", args[0] if args else None)
        if captured["hidden"] is None:
            raise RuntimeError("could not capture decoder-layer input")
        raise _StopAtTargetLayer

    hook = model.model.layers[layer_idx].register_forward_pre_hook(capture_input, with_kwargs=True)

    try:
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            ids = token_ids[start:end].unsqueeze(0).to(device, non_blocking=True)
            position_ids = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(ids)
            captured.clear()
            try:
                model.model(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
            except _StopAtTargetLayer:
                pass
            hidden = captured["hidden"]
            layer = model.model.layers[layer_idx]
            hidden = layer.input_layernorm(hidden)
            keys = layer.self_attn.k_proj(hidden)
            keys = keys.view(1, end - start, num_kv_heads, head_dim).transpose(1, 2)
            if num_kv_groups > 1:
                keys = repeat_kv(keys, num_kv_groups)
            cos, sin = model.model.rotary_emb(keys, position_ids)
            _, keys = apply_rotary_pos_emb(keys, keys, cos, sin)
            pieces.append(keys[0].transpose(0, 1).contiguous().cpu().float())
            del hidden, keys, ids, attention_mask, position_ids, cos, sin
    finally:
        hook.remove()

    return torch.cat(pieces, dim=0)


@torch.no_grad()
def nearest_center_indices(
    x: torch.Tensor,
    centers: torch.Tensor,
    batch_size: int,
) -> Iterable[tuple[int, int, torch.Tensor]]:
    center_norm = centers.square().sum(dim=1).unsqueeze(0)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        batch = x[start:end]
        distances = batch.square().sum(dim=1, keepdim=True)
        distances = distances + center_norm - 2.0 * (batch @ centers.T)
        yield start, end, distances.argmin(dim=1)


@torch.no_grad()
def fit_residual_stage(
    residual: torch.Tensor,
    k: int,
    sample_size: int,
    lloyd_iters: int,
    distance_batch: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    n, dim = residual.shape
    take = min(sample_size, n)
    sample_idx = torch.randint(n, (take,), generator=generator, device=residual.device)
    sample = residual[sample_idx].clone()
    init_idx = torch.randperm(take, generator=generator, device=residual.device)[:k]
    centers = sample[init_idx].clone()

    for _ in range(lloyd_iters):
        sums = torch.zeros((k, dim), dtype=torch.float32, device=residual.device)
        counts = torch.zeros(k, dtype=torch.float32, device=residual.device)
        for start, end, assignment in nearest_center_indices(sample, centers, distance_batch):
            batch = sample[start:end]
            sums.index_add_(0, assignment, batch)
            counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float32))
        populated = counts > 0
        centers[populated] = sums[populated] / counts[populated, None]
        empty = (~populated).nonzero(as_tuple=False).flatten()
        if empty.numel():
            replacement = torch.randint(
                take, (empty.numel(),), generator=generator, device=residual.device
            )
            centers[empty] = sample[replacement]

    squared_error = 0.0
    for start, end, assignment in nearest_center_indices(residual, centers, distance_batch):
        reconstruction = centers[assignment]
        residual[start:end].sub_(reconstruction)
        squared_error += float(residual[start:end].square().sum().item())
    mse = squared_error / float(n * dim)
    return centers, mse


def load_model(model_name: str, dtype: torch.dtype, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map={"": device.index or 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--corpus_paths", nargs="+", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--length_schedule", type=parse_lengths, default=parse_lengths("131072,65536,65536,32768,32768"))
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=262144)
    parser.add_argument("--lloyd_iters", type=int, default=6)
    parser.add_argument("--distance_batch", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--run_id", default="run")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.chunk_size <= 0 or args.M <= 0 or args.K <= 1:
        raise ValueError("chunk_size, M, and K must be positive")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    dtype = torch.float16 if args.model_dtype == "fp16" else torch.bfloat16
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{args.run_id}_train.log"

    with log_path.open("a", encoding="utf-8") as log_file:
        log(f"model={args.model_name} dtype={args.model_dtype} device={device}", log_file)
        log(f"layers={args.layers} schedule={args.length_schedule} chunk={args.chunk_size}", log_file)
        log(
            f"RVQ M={args.M} K={args.K} sample={args.sample_size} "
            f"lloyd_iters={args.lloyd_iters} distance_batch={args.distance_batch}",
            log_file,
        )
        tokenizer, model = load_model(args.model_name, dtype, device)
        docs = load_corpus(tokenizer, args.corpus_paths, min(args.length_schedule), log_file)
        rng = random.Random(args.seed)
        cuda_generator = torch.Generator(device=device)
        cuda_generator.manual_seed(args.seed)

        num_layers = len(model.model.layers)
        for layer_idx in args.layers:
            if not 0 <= layer_idx < num_layers:
                raise ValueError(f"invalid layer {layer_idx}; model has {num_layers} layers")
            output_path = save_dir / f"layer_{layer_idx}_group_0.pt"
            if args.skip_existing and output_path.exists():
                log(f"layer={layer_idx} already exists; skipped", log_file)
                continue

            layer_start = time.time()
            segments = sample_segments(docs, args.length_schedule, rng)
            key_parts: list[torch.Tensor] = []
            forward_bar = tqdm(segments, desc=f"layer {layer_idx} forward", leave=False)
            for token_ids, line_idx, offset in forward_bar:
                log(
                    f"layer={layer_idx} segment_len={token_ids.numel()} "
                    f"doc_line={line_idx} offset={offset}",
                    log_file,
                )
                keys = collect_post_rope_keys(
                    model=model,
                    token_ids=token_ids,
                    layer_idx=layer_idx,
                    chunk_size=args.chunk_size,
                    device=device,
                )
                key_parts.append(keys.reshape(-1, keys.shape[-1]))
                del keys
                torch.cuda.empty_cache()

            vector_count = sum(part.shape[0] for part in key_parts)
            residual = torch.cat(key_parts, dim=0).to(device=device, dtype=torch.float32)
            del key_parts
            gc.collect()
            log(
                f"layer={layer_idx} cached_vectors={vector_count} "
                f"resident_gb={residual.numel() * residual.element_size() / 1e9:.2f} "
                f"gpu_allocated_gb={torch.cuda.memory_allocated() / 1e9:.2f}",
                log_file,
            )

            codebooks: list[torch.Tensor] = []
            stage_bar = tqdm(range(args.M), desc=f"layer {layer_idx} RVQ", leave=False)
            for stage_idx in stage_bar:
                centers, mse = fit_residual_stage(
                    residual=residual,
                    k=args.K,
                    sample_size=args.sample_size,
                    lloyd_iters=args.lloyd_iters,
                    distance_batch=args.distance_batch,
                    generator=cuda_generator,
                )
                codebooks.append(centers.cpu())
                stage_bar.set_postfix(mse=f"{mse:.4e}")
                log(f"layer={layer_idx} stage={stage_idx} cumulative_mse={mse:.8e}", log_file)

            payload = {
                "M": args.M,
                "K": args.K,
                "d": residual.shape[1],
                "codebooks": torch.stack(codebooks, dim=0),
                "metadata": {
                    "model_name": args.model_name,
                    "target": "key_post_rope",
                    "rope_source": "model.model.rotary_emb",
                    "length_schedule": args.length_schedule,
                    "chunk_size": args.chunk_size,
                    "seed": args.seed,
                    "sample_size": args.sample_size,
                    "lloyd_iters": args.lloyd_iters,
                    "vector_count": vector_count,
                },
            }
            torch.save(payload, output_path)
            del residual, codebooks, payload
            gc.collect()
            torch.cuda.empty_cache()
            log(
                f"layer={layer_idx} saved={output_path} elapsed_min={(time.time() - layer_start) / 60:.2f}",
                log_file,
            )

        log("training complete", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
