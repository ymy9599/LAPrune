#!/usr/bin/env python
"""
Train LAPrune codebooks for an LLM by collecting RoPE-space keys per layer.

Preferred usage (from repo root):
  PYTHONPATH=./LAPrune python -m laprune.cli.train_codebooks --help
"""

import argparse
import json
import os
import random
from collections import deque
from datetime import datetime
from typing import Generator, Iterable, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import repeat_kv

from laprune.quantizer import AdditiveQuantizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def _get_torch_dtype(dtype: str) -> torch.dtype:
    s = str(dtype).strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown --model_dtype '{dtype}' (expected: fp16/bf16/fp32)")


def iter_random_segments(
    tokenizer: AutoTokenizer,
    corpus_paths: Iterable[str],
    max_len: int,
    device: torch.device,
    max_segments: int,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Randomly sample non-overlapping `max_len` segments from JSONL corpora.
    Each line must contain {"text": "..."}.
    """
    seg_used = 0
    paths = list(corpus_paths)
    random.shuffle(paths)
    for path in paths:
        if seg_used >= max_segments:
            break
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if seg_used >= max_segments:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue

                enc = tokenizer(text, return_tensors="pt", truncation=False)
                ids_all = enc["input_ids"][0]
                mask_all = enc["attention_mask"][0]
                t_total = int(ids_all.shape[0])
                if t_total < max_len:
                    continue

                num_segments = t_total // max_len
                if num_segments <= 0:
                    continue
                seg_idx = random.randint(0, num_segments - 1)
                start = seg_idx * max_len
                end = start + max_len

                ids = ids_all[start:end].unsqueeze(0).to(device)
                mask = mask_all[start:end].unsqueeze(0).to(device)
                yield ids, mask
                seg_used += 1


def _assert_layer_compatible(layer) -> None:
    if not hasattr(layer, "input_layernorm"):
        raise RuntimeError("decoder layer missing `input_layernorm` (update training script for this model)")
    if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "k_proj"):
        raise RuntimeError("decoder layer missing `self_attn.k_proj` (update training script for this model)")


@torch.no_grad()
def collect_layer_keys_rope(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: List[int],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> dict:
    """
    Run one forward and collect RoPE-space keys for requested layers.
    Returns: {layer_idx: keys[T, H, D]} on CPU fp32.
    """
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    num_layers = len(model.model.layers)
    assert len(hidden_states) == num_layers + 1

    config = model.config
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(config.hidden_size // num_heads)
    num_kv_group = int(num_heads // num_kv_heads)

    keys_per_layer = {}
    for layer_idx in layers:
        assert 0 <= layer_idx < num_layers
        h = hidden_states[layer_idx]  # [1,T,hidden]
        t = int(h.shape[1])

        layer = model.model.layers[layer_idx]
        _assert_layer_compatible(layer)
        self_attn = layer.self_attn

        # Match online semantics: pre-attn LN then k_proj
        h_norm = layer.input_layernorm(h)
        k_proj = self_attn.k_proj(h_norm)  # [1,T,num_kv_heads*D]
        k_proj = k_proj.view(1, t, num_kv_heads, head_dim).transpose(1, 2)  # [1,Kv,T,D]
        if num_kv_group > 1:
            k_proj = repeat_kv(k_proj, num_kv_group)  # [1,H,T,D]

        k_device = k_proj.device
        cos_local = cos.to(device=k_device)
        sin_local = sin.to(device=k_device)
        position_ids = torch.arange(t, device=k_device, dtype=torch.long).unsqueeze(0)
        k_rope = apply_rotary_pos_emb(
            mat=k_proj,
            cos=cos_local,
            sin=sin_local,
            position_ids=position_ids,
            unsqueeze_dim=1,
        )
        keys = k_rope[0].transpose(0, 1).contiguous()  # [T,H,D]
        keys_per_layer[layer_idx] = keys.to(device="cpu", dtype=torch.float32)

    return keys_per_layer


def build_model_and_tokenizer(
    *,
    model_name: str,
    model_dtype: torch.dtype,
    device_map: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    model.config.output_hidden_states = True
    return tokenizer, model


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--model_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--corpus_paths", type=str, nargs="+", required=True)
    p.add_argument("--train_max_len", type=int, default=4096)
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--all_layers", action="store_true", default=False)

    p.add_argument("--M", type=int, default=8)
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--heads_per_group", type=int, default=0)
    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--max_segments_per_layer", type=int, default=5)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging file under LAPrune/laprune/cli/log
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        print(msg, file=log_file, flush=True)

    log(f"[LAPrune-Train] log_path={log_path}")
    log(f"[LAPrune-Train] model_name={args.model_name} dtype={args.model_dtype} device_map={args.device_map}")
    log(f"[LAPrune-Train] corpus_paths={args.corpus_paths}")
    log(f"[LAPrune-Train] train_max_len={args.train_max_len} max_segments_per_layer={args.max_segments_per_layer}")

    model_dtype = _get_torch_dtype(args.model_dtype)
    tokenizer, model = build_model_and_tokenizer(
        model_name=args.model_name,
        model_dtype=model_dtype,
        device_map=args.device_map,
    )

    num_layers = len(model.model.layers)
    if args.all_layers or not args.layers:
        layer_list = list(range(num_layers))
    else:
        layer_list = list(args.layers)

    config = model.config
    num_heads = int(config.num_attention_heads)
    head_dim = int(config.hidden_size // num_heads)

    heads_per_group = int(args.heads_per_group)
    if heads_per_group <= 0 or heads_per_group > num_heads:
        heads_per_group = num_heads

    stages = max(1, int(args.stages))
    if int(args.M) % stages != 0:
        raise ValueError(f"--M={args.M} must be divisible by --stages={stages}")
    m_per_stage = int(args.M) // stages

    # RoPE tables (same as online; built once on `device`)
    rope_theta = float(getattr(config, "rope_theta", 10000.0))
    pos_ids = torch.arange(int(args.train_max_len), device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.einsum("i,j->ij", pos_ids, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    attn_dtype = next(model.parameters()).dtype
    cos = emb.cos().to(dtype=attn_dtype)
    sin = emb.sin().to(dtype=attn_dtype)

    os.makedirs(args.save_dir, exist_ok=True)

    log(
        f"[LAPrune-Train] layers={len(layer_list)} heads={num_heads} head_dim={head_dim} "
        f"heads_per_group={heads_per_group} stages={stages} M_per_stage={m_per_stage} K={args.K}"
    )

    for layer_idx in layer_list:
        log(f"[LAPrune-Train] ===== layer {layer_idx}/{layer_list[-1]} =====")
        num_groups = num_heads // heads_per_group
        for group_idx in range(num_groups):
            log(f"[LAPrune-Train] layer={layer_idx} group={group_idx}: train stages={stages}")

            stage_quantizers: list[AdditiveQuantizer] = []
            for s in range(stages):
                aq = AdditiveQuantizer(
                    M=m_per_stage,
                    K=int(args.K),
                    d=head_dim,
                    init="random",
                    reg=1e-4,
                    device=device,
                    dtype=torch.float32,
                )
                aq.log_prefix = f"layer{layer_idx}_group{group_idx}_stage{s}: "
                stage_quantizers.append(aq)

            for stage_idx in range(stages):
                seg_count = int(args.max_segments_per_layer)
                seg_count = max(1, seg_count)
                seg_iter = iter_random_segments(
                    tokenizer=tokenizer,
                    corpus_paths=args.corpus_paths,
                    max_len=int(args.train_max_len),
                    device=device,
                    max_segments=seg_count,
                )

                progress = tqdm(total=seg_count, desc=f"layer{layer_idx} group{group_idx} stage{stage_idx}", leave=True)
                step = 0
                recent_losses = deque(maxlen=int(args.log_interval))
                recent_rels = deque(maxlen=int(args.log_interval))

                for ids, mask in seg_iter:
                    step += 1
                    keys_per_layer = collect_layer_keys_rope(
                        model=model,
                        input_ids=ids,
                        attention_mask=mask,
                        layers=[layer_idx],
                        cos=cos,
                        sin=sin,
                    )
                    keys = keys_per_layer[layer_idx]  # [T,H,D] fp32 cpu
                    t = int(keys.shape[0])
                    h = int(keys.shape[1])
                    d = int(keys.shape[2])
                    assert h == num_heads and d == head_dim

                    keys_train = keys.to(device)
                    h_start = group_idx * heads_per_group
                    h_end = h_start + heads_per_group
                    x_base = keys_train[:, h_start:h_end, :].reshape(-1, head_dim)
                    if x_base.numel() == 0:
                        progress.update(1)
                        continue

                    if stage_idx > 0:
                        residual = x_base
                        for prev_s in range(stage_idx):
                            prev_q = stage_quantizers[prev_s]
                            codes_prev, _, _ = prev_q.encode(
                                residual,
                                method="icm",
                                icm_iters=2,
                                batch_size=3_000_000,
                            )
                            recon_prev = torch.zeros_like(residual)
                            for m in range(prev_q.M):
                                recon_prev += prev_q.codebooks[m][codes_prev[:, m]]
                            residual = residual - recon_prev
                        x_train = residual
                    else:
                        x_train = x_base

                    def log_train(msg: str, *, _step=step):
                        try:
                            parts = msg.strip().split()
                            loss_val = float(parts[3])
                            rel_val = float(parts[-1])
                        except Exception:
                            loss_val = None
                            rel_val = None
                        if loss_val is not None and rel_val is not None:
                            recent_losses.append(loss_val)
                            recent_rels.append(rel_val)
                            avg_loss = sum(recent_losses) / len(recent_losses)
                            avg_rel = sum(recent_rels) / len(recent_rels)
                            progress.set_postfix(loss=f"{avg_loss:.4f}", rel=f"{avg_rel:.2f}")
                            if int(args.log_interval) > 0 and (_step % int(args.log_interval) == 0):
                                log(msg)

                    progress.update(1)
                    stage_q = stage_quantizers[stage_idx]
                    stage_q.log_fn = log_train
                    stage_q.fit(
                        x_train,
                        n_iter=int(args.n_iter),
                        method="icm",
                        update_method="batchwise",
                        batch_size=3_000_000,
                        ema_decay=None,
                        codebook_save_path=None,
                    )

                log(f"[LAPrune-Train] layer={layer_idx} group={group_idx} stage={stage_idx}: processed={step}/{seg_count} segments")

            # Save concatenated codebooks
            m_total = stages * m_per_stage
            codebooks_total = torch.zeros(m_total, int(args.K), head_dim, dtype=torch.float32, device=device)
            for s in range(stages):
                q = stage_quantizers[s]
                start = s * m_per_stage
                end = start + m_per_stage
                codebooks_total[start:end] = q.codebooks.to(device=device, dtype=torch.float32)

            path = os.path.join(args.save_dir, f"layer_{layer_idx}_group_{group_idx}.pt")
            torch.save({"M": m_total, "K": int(args.K), "d": head_dim, "codebooks": codebooks_total.cpu()}, path)
            log(f"[LAPrune-Train] saved {path}")

    log(f"[LAPrune-Train] done. save_dir={args.save_dir}")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


