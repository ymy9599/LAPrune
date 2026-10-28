import gc
import math
import torch
from transformers.cache_utils import DynamicCache

def sync() -> None:
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)

def model_layers(model, method: str):
    if method in {"laprune"}:
        return list(model.model.decoder.layers)
    return list(model.model.layers)

def build_exact_prompt(tokenizer, tokens: int) -> torch.Tensor:
    # Fixed, local and deterministic. Throughput depends on dimensions, while a
    # non-constant seed still avoids the degenerate one-token repetition case.
    seed = tokenizer(
        "The quick brown fox jumps over the lazy dog. Long-context retrieval benchmark. ",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    ids = seed.repeat(1, math.ceil(tokens / seed.shape[1]))[:, :tokens].contiguous()
    return ids

def prefill(method: str, model, input_ids: torch.Tensor):
    if method in {"vanilla", "flashattention2"}:
        base = model.model(input_ids=input_ids, use_cache=True, return_dict=True)
        hidden_last = base.last_hidden_state[:, -1:, :]
        logits = model.lm_head(hidden_last).float()
        cache = base.past_key_values
        del hidden_last, base
        return logits, cache

    out = model.model(input_ids=input_ids)
    cache = out.past_key_values
    logits = out.logits
    if method == "laprune" and hasattr(model, "_encode_prefill_cache"):
        # This is normally called by AQGeneration.generate. With the paper-faithful
        # config it is a no-op verification because each layer encoded in prefill.
        model._encode_prefill_cache()
    return logits, cache

def decode_one(method: str, model, token: torch.Tensor, cache):
    if method in {"vanilla", "flashattention2"}:
        out = model(input_ids=token, past_key_values=cache, use_cache=True, return_dict=True)
    else:
        out = model.model(input_ids=token, kv_cache=cache)
    return out.logits, out.past_key_values

def _legacy_cache_pairs(cache):
    if hasattr(cache, "to_legacy_cache"):
        return list(cache.to_legacy_cache())
    if hasattr(cache, "layers"):
        return [(x.keys, x.values) for x in cache.layers]
    return list(cache)

def staged_prefill(method: str, model, tokenizer, prompt_tokens: int, batch_size: int,
                   microbatch: int, decode_capacity: int):
    """Build a large-batch decode state without timing or retaining prefill peaks.

    Each microbatch is prefetched independently, copied to CPU, and released.
    The final caches are concatenated on the batch dimension on one GPU.  This
    affects setup time only; the subsequently timed decode path is identical to
    a monolithic prefill with the same final cache tensors.
    """
    if method not in {"vanilla", "flashattention2", "laprune"}:
        raise ValueError(f"staged prefill unsupported for {method}")
    device = next(model.parameters()).device
    chunks = []
    logits_chunks = []
    for start in range(0, batch_size, microbatch):
        mb = min(microbatch, batch_size - start)
        ids = build_exact_prompt(tokenizer, prompt_tokens).repeat(mb, 1).contiguous().to(device)
        with torch.inference_mode():
            logits, cache = prefill(method, model, ids)
        sync()
        logits_chunks.append(logits[:, -1:].detach().cpu())
        if method == "laprune":
            layer_state = []
            for layer in model_layers(model, method):
                attn = layer.self_attn
                n = int(attn.k_cache_len)
                aqc = getattr(attn, "_aq_cache", None)
                codes = None
                if isinstance(aqc, dict) and torch.is_tensor(aqc.get("codes")):
                    codes = aqc["codes"][:, :, :n, :].detach().cpu()
                layer_state.append((
                    attn.k_cache[:, :, :n, :].detach().cpu(),
                    attn.v_cache[:, :, :n, :].detach().cpu(),
                    codes,
                ))
            chunks.append(layer_state)
            model.reset()
        else:
            chunks.append([
                (k.detach().cpu(), v.detach().cpu())
                for k, v in _legacy_cache_pairs(cache)
            ])
        del ids, logits, cache
        gc.collect()
        torch.cuda.empty_cache()

    if method in {"vanilla", "flashattention2"}:
        # DynamicCache(pairs, ...) feeds every supplied tensor through
        # DynamicLayer.update(), which concatenates it with an empty GPU tensor.
        # At B=4/128K that unnecessary copy can consume another ~1 GiB per
        # K/V tensor and OOM even though the completed cache itself fits.  Bind
        # the already assembled tensors directly to each cache layer instead.
        cache = DynamicCache(config=model.config)
        for li, layer in enumerate(model.model.layers):
            dev = layer.self_attn.q_proj.weight.device
            k = torch.cat([x[li][0] for x in chunks], dim=0).to(dev)
            v = torch.cat([x[li][1] for x in chunks], dim=0).to(dev)
            cache_layer = cache.layers[li]
            cache_layer.keys = k
            cache_layer.values = v
            cache_layer.dtype = k.dtype
            cache_layer.device = k.device
            cache_layer.is_initialized = True
    else:
        cache = []
        layers = model_layers(model, method)
        for li, layer in enumerate(layers):
            attn = layer.self_attn
            dev = attn.q_proj.weight.device
            k_parts = [x[li][0] for x in chunks]
            v_parts = [x[li][1] for x in chunks]
            n = int(k_parts[0].shape[2])
            b = sum(int(x.shape[0]) for x in k_parts)
            h, d = int(k_parts[0].shape[1]), int(k_parts[0].shape[3])
            cap = n + max(2, int(decode_capacity))
            k = torch.empty((b, h, cap, d), dtype=k_parts[0].dtype, device=dev)
            v = torch.empty_like(k)
            off = 0
            for kp, vp in zip(k_parts, v_parts):
                nb = int(kp.shape[0])
                k[off:off+nb, :, :n, :].copy_(kp.to(dev))
                v[off:off+nb, :, :n, :].copy_(vp.to(dev))
                off += nb
            attn.k_cache, attn.v_cache = k, v
            attn.k_cache_len, attn.k_cache_capacity, attn.total_seq_len = n, cap, n
            code_parts = [x[li][2] for x in chunks]
            if code_parts[0] is not None:
                hc, mc = int(code_parts[0].shape[1]), int(code_parts[0].shape[3])
                codes = torch.empty((b, hc, cap, mc), dtype=code_parts[0].dtype, device=dev)
                off = 0
                for cp in code_parts:
                    nb = int(cp.shape[0])
                    codes[off:off+nb, :, :n, :].copy_(cp.to(dev))
                    off += nb
                attn._aq_cache = {
                    "seq_len": n, "encoded_len": n, "batch_size": b,
                    "codes": codes, "approx": None,
                }
            cache.append((k[:, :, :n, :].data, v[:, :, :n, :].data))
    logits = torch.cat(logits_chunks, dim=0).to(device)
    del chunks, logits_chunks
    gc.collect()
    torch.cuda.empty_cache()
    return logits, cache
