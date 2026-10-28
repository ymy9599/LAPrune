#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import gzip
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from laprune.loading import load_method


HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Generate LongBench predictions")
    p.add_argument("--model", default="llama31-8b-instruct")
    p.add_argument("--method", choices=["dense", "oracle", "laprune"], required=True)
    p.add_argument("--config")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--data-dir", help="Optional directory containing <task>.jsonl or saved datasets")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-max-length", type=int, default=None)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chat-template", choices=["none", "llama-3"], default=None)
    return p.parse_args()


def load_rows(task: str, data_dir: str | None):
    if data_dir:
        path = Path(data_dir) / f"{task}.jsonl"
        if path.exists():
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        saved = Path(data_dir) / task
        if saved.exists():
            return load_from_disk(str(saved))
    bundled = HERE / "data" / f"{task}.jsonl.gz"
    if bundled.exists():
        with gzip.open(bundled, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return load_dataset("THUDM/LongBench", task, split="test")


def maybe_chat(task: str, prompt: str, template: str | None) -> str:
    no_chat = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
    if task in no_chat or template != "llama-3":
        return prompt
    return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + prompt.strip() + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def model_device(model):
    return next(model.parameters()).device


@torch.inference_mode()
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    template = args.chat_template
    if template is None:
        template = 'llama-3'
    prompts = json.loads((HERE / "prompts.json").read_text())
    max_new = json.loads((HERE / "max_new_tokens.json").read_text())
    tasks = args.datasets or json.loads((HERE / "datasets.json").read_text())
    tokenizer, model = load_method(
        method=args.method, model_name=args.model, dtype=args.dtype,
        config=args.config, device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = model_device(model)
    if args.model_max_length is None:
        cfg = model.config if args.method == 'dense' else model.model.decoder.model.config
        args.model_max_length = int(cfg.max_position_embeddings)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        rows = load_rows(task, args.data_dir)
        if args.max_samples is not None:
            rows = list(rows)[: args.max_samples]
        output = args.output_dir / f"{task}.jsonl"
        if output.exists():
            raise FileExistsError(f"Refusing to append to existing output: {output}")
        with output.open("w", encoding="utf-8") as handle:
            for row in tqdm(rows, desc=task):
                if hasattr(model, "reset"):
                    model.reset()
                prompt = maybe_chat(task, prompts[task].format(**row), template)
                encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
                budget = args.model_max_length - int(max_new[task])
                input_ids = encoded.input_ids[:, -budget:].to(device)
                prompt_len = input_ids.shape[-1]
                output_ids = model.generate(
                    input_ids=input_ids, max_new_tokens=int(max_new[task]),
                    do_sample=False, num_beams=1, pad_token_id=tokenizer.pad_token_id,
                )
                if output_ids.ndim == 1:
                    output_ids = output_ids.unsqueeze(0)
                generated = output_ids[0, prompt_len:].tolist()
                if tokenizer.eos_token_id in generated:
                    generated = generated[:generated.index(tokenizer.eos_token_id)]
                record = {
                    "pred": tokenizer.decode(generated, skip_special_tokens=True),
                    "answers": row["answers"],
                    "all_classes": row.get("all_classes", []),
                    "length": row.get("length", prompt_len),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
