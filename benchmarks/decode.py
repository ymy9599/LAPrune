"""Paper-level decode benchmark for Llama-3.1-8B-Instruct."""
import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from laprune.loading import load_method
from cache_setup import build_exact_prompt, prefill, decode_one, staged_prefill, sync

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='llama31-8b-instruct', help='Model ID or local checkpoint path')
    parser.add_argument('--method', choices=['laprune', 'flashattention2', 'vanilla'], default='laprune')
    parser.add_argument('--config', default='configs/decode/llama31-8b-instruct.json')
    parser.add_argument('--prompt-tokens', type=int, default=16000)
    parser.add_argument('--decode-tokens', type=int, default=256)
    parser.add_argument('--warmup-tokens', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--prefill-microbatch', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.batch_size, args.prefill_microbatch) < 1:
        parser.error('Lengths and batch sizes must be positive')
    if not 0 <= args.warmup_tokens < args.decode_tokens:
        parser.error('warmup-tokens must be smaller than decode-tokens')
    if args.repeats < 1:
        parser.error('repeats must be positive')
    if args.repeats > 1:
        runs = []
        for index in range(1, args.repeats + 1):
            output = args.output.with_name(f'{args.output.stem}_r{index}{args.output.suffix}')
            command = [sys.executable, str(Path(__file__).resolve()),
                '--model', args.model, '--method', args.method, '--config', args.config,
                '--prompt-tokens', str(args.prompt_tokens), '--decode-tokens', str(args.decode_tokens),
                '--warmup-tokens', str(args.warmup_tokens), '--batch-size', str(args.batch_size),
                '--prefill-microbatch', str(args.prefill_microbatch), '--repeats', '1',
                '--output', str(output)]
            subprocess.run(command, check=True)
            runs.append(json.loads(output.read_text()))
        result = {'method': args.method, 'prompt_tokens': args.prompt_tokens,
            'decode_tokens': args.decode_tokens, 'warmup_tokens': args.warmup_tokens,
            'batch_size': args.batch_size, 'repeats': args.repeats,
            'tokens_per_second': statistics.mean(run['tokens_per_second'] for run in runs),
            'tokens_per_second_std': statistics.stdev(run['tokens_per_second'] for run in runs),
            'decode_seconds': statistics.mean(run['decode_seconds'] for run in runs),
            'runs': runs}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return
    torch.manual_seed(0)
    backend = {'vanilla': 'sdpa', 'flashattention2': 'flash_attention_2'}.get(args.method)
    tok, model = load_method(method='laprune' if args.method == 'laprune' else 'dense',
        model_name=args.model, config=args.config, device_map={'': 0}, dense_backend=backend)
    cfg = model.model.decoder.model.config if args.method == 'laprune' else model.config
    if cfg.model_type != 'llama' or cfg.num_hidden_layers != 32 or cfg.hidden_size != 4096 or cfg.num_key_value_heads != 8 or cfg.max_position_embeddings != 131072:
        raise ValueError('This benchmark supports Llama-3.1-8B-Instruct only')
    with torch.inference_mode():
        if args.batch_size > args.prefill_microbatch:
            logits, cache = staged_prefill(args.method, model, tok, args.prompt_tokens,
                args.batch_size, args.prefill_microbatch, args.decode_tokens + 128)
        else:
            device = next(model.parameters()).device
            ids = build_exact_prompt(tok, args.prompt_tokens).repeat(args.batch_size, 1).to(device)
            logits, cache = prefill(args.method, model, ids)
            del ids
        token = logits[:, -1].argmax(-1, keepdim=True)
        finite = bool(torch.isfinite(logits).all().item())
        del logits
    gc.collect()
    torch.cuda.empty_cache()
    if args.method == 'vanilla':
        import transformers.integrations.sdpa_attention as sdpa_attention
        sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
    sync()
    start = time.perf_counter()
    warmup_end = 0.0
    generated_tensors = []
    finite_tensors = []
    with torch.inference_mode():
        for step in range(args.decode_tokens):
            logits, cache = decode_one(args.method, model, token, cache)
            finite_tensors.append(torch.isfinite(logits).all())
            token = logits[:, -1].argmax(-1, keepdim=True)
            generated_tensors.append(token.detach())
            if step + 1 == args.warmup_tokens:
                sync()
                warmup_end = time.perf_counter() - start
    sync()
    seconds = time.perf_counter() - start - warmup_end
    if finite_tensors:
        finite = finite and bool(torch.stack(finite_tensors).all().item())
    if not finite:
        raise RuntimeError('Non-finite logits during decode benchmark')
    measured = args.decode_tokens - args.warmup_tokens
    result = {'method': args.method, 'prompt_tokens': args.prompt_tokens,
        'decode_tokens': args.decode_tokens, 'warmup_tokens': args.warmup_tokens,
        'batch_size': args.batch_size, 'decode_seconds': seconds,
        'tokens_per_second': args.batch_size * measured / seconds, 'finite_logits': finite}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
