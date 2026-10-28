#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import gzip
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm

from laprune.loading import load_method


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RULER JSONL data")
    p.add_argument("--model", default="llama31-8b-instruct")
    p.add_argument("--method", choices=["dense", "oracle", "laprune"], required=True)
    p.add_argument("--config")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--context-lengths", nargs="+", type=int, required=True)
    p.add_argument("--samples-per-task", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--model-max-length", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def normalize(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def score(prediction: str, answers: list[str], partial: bool) -> float:
    prediction = normalize(prediction)
    answers = [normalize(x) for x in answers if str(x).strip()]
    if not answers:
        return 0.0
    return float(any(x in prediction for x in answers) if partial else all(x in prediction for x in answers))


def answer_stop(tokenizer, prompt_len: int, answers: list[str]):
    """Use the same all-answer stopping rule as the paper evaluation."""
    gold = [normalize(x) for x in answers if str(x).strip()]
    if not gold:
        return None
    from transformers import StoppingCriteria, StoppingCriteriaList

    class AllAnswersSeen(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            text = normalize(tokenizer.decode(
                input_ids[0, prompt_len:].tolist(), skip_special_tokens=True
            ))
            return all(answer in text for answer in gold)

    return StoppingCriteriaList([AllAnswersSeen()])


def generate(model, input_ids, tokenizer, answers, max_new_tokens):
    kwargs = {
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        "max_new_tokens": int(max_new_tokens), "do_sample": False,
        "num_beams": 1, "pad_token_id": int(tokenizer.pad_token_id),
    }
    stopping = answer_stop(tokenizer, int(input_ids.shape[-1]), answers)
    if stopping is not None:
        kwargs["stopping_criteria"] = stopping
    while True:
        try:
            return model.generate(input_ids=input_ids, **kwargs)
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if match is None or match.group(1) not in kwargs:
                raise
            kwargs.pop(match.group(1))


def task_length(category: str):
    tail = str(category).split("/")[-1]
    task, length = tail.rsplit("_", 1)
    return task, int(length)


def load_groups(path: Path, tasks: list[str], lengths: list[int], limit: int):
    groups = {}
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        lines = handle.readlines()
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            task, length = task_length(row.get("category", ""))
        except Exception:
            continue
        if task not in tasks or length not in lengths:
            continue
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            continue
        extra = row.get("extra_info", {}) or {}
        answers = extra.get("answers")
        if answers is None:
            answers = extra.get("answer", [])
        if answers is None:
            answers = []
        if not isinstance(answers, list):
            answers = [answers]
        metric = str(extra.get("metric_fn", "") or extra.get("metric", "")).lower()
        item = {
            "task": task,
            "prompt": row["prompt"], "answers": [str(x) for x in answers],
            "partial": "part" in metric or ("all" not in metric and task.startswith("qa_")),
        }
        key = f"{task}_{length}"
        if len(groups.setdefault(key, [])) < limit:
            groups[key].append(item)
    return groups


@torch.inference_mode()
def main():
    args = parse_args()
    if args.config is None and args.method != 'dense':
        if args.model != 'llama31-8b-instruct':
            raise ValueError('Specify --config when using a local model path')
        args.config = f'configs/quality/{args.model}/ruler_{args.method}.json'
    tokenizer, model = load_method(
        method=args.method, model_name=args.model, dtype=args.dtype,
        config=args.config, device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = next(model.parameters()).device
    if args.model_max_length is None:
        cfg = model.config if args.method == 'dense' else model.model.decoder.model.config
        args.model_max_length = int(cfg.max_position_embeddings)
    groups = load_groups(args.data, args.tasks, args.context_lengths, args.samples_per_task)
    results = {}
    total_correct = 0.0
    total_count = 0
    for key, rows in sorted(groups.items()):
        correct = 0.0
        elapsed = 0.0
        for row in tqdm(rows, desc=key):
            if hasattr(model, "reset"):
                model.reset()
            ids = tokenizer(row["prompt"], return_tensors="pt", truncation=False).input_ids
            budget = max(1, args.model_max_length - args.max_new_tokens)
            ids = ids[:, -budget:].to(device)
            prompt_len = ids.shape[-1]
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = generate(model, ids, tokenizer, row["answers"], args.max_new_tokens)
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            if generated.ndim == 1:
                generated = generated.unsqueeze(0)
            prediction = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
            correct += score(prediction, row["answers"], row["partial"])
        count = len(rows)
        results[key] = {"accuracy": correct / count if count else 0.0, "correct": correct, "count": count, "seconds": elapsed}
        total_correct += correct
        total_count += count
    payload = {
        "method": args.method,
        "model": args.model,
        "overall_accuracy": total_correct / total_count if total_count else 0.0,
        "correct": total_correct,
        "count": total_count,
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
