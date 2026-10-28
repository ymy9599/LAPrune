from __future__ import annotations

import torch


def _tail_causal_mask(num_queries: int, num_keys: int, dtype, device):
    """Causal mask for queries occupying the final positions of a KV cache."""
    if num_queries > num_keys:
        raise ValueError(f"num_queries={num_queries} exceeds num_keys={num_keys}")
    mask = torch.full(
        (1, 1, num_queries, num_keys),
        torch.finfo(dtype).min,
        dtype=torch.float32,
        device=device,
    )
    mask[..., -num_queries:].triu_(diagonal=1)
    mask[..., : num_keys - num_queries].fill_(0)
    return mask.to(dtype=dtype)


def do_sdpa_attn(
    query,
    key,
    value,
    cos=None,
    sin=None,
    out_proj: torch.nn.Linear | None = None,
    query_down_proj: torch.nn.Parameter | None = None,
    key_down_proj: torch.nn.Parameter | None = None,
    mask: torch.Tensor | None = None,
    already_rope: bool = False,
):
    """Apply SDPA to post-RoPE Q/K tensors while preserving native GQA caches."""
    if not already_rope:
        raise ValueError("LAPrune attention expects post-RoPE query and key tensors")

    q, k, v = query, key, value
    num_heads = int(q.shape[1])
    num_queries = int(q.shape[-2])

    if query_down_proj is not None:
        q = q.transpose(1, 2).flatten(2) @ query_down_proj
        q = q.unflatten(-1, (num_heads, -1)).transpose(1, 2)
    if key_down_proj is not None:
        k = k.transpose(1, 2).flatten(2) @ key_down_proj
        k = k.unflatten(-1, (num_heads, -1)).transpose(1, 2)

    enable_gqa = int(q.shape[1]) != int(k.shape[1])
    if enable_gqa and int(q.shape[1]) % int(k.shape[1]) != 0:
        raise ValueError(f"invalid GQA head counts: Hq={q.shape[1]}, Hkv={k.shape[1]}")

    if mask is not None:
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, enable_gqa=enable_gqa
        )
    else:
        num_keys = int(k.shape[-2])
        if num_keys == num_queries:
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True, enable_gqa=enable_gqa
            )
        elif num_queries == 1:
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=False, enable_gqa=enable_gqa
            )
        else:
            causal_mask = _tail_causal_mask(num_queries, num_keys, q.dtype, q.device)
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, is_causal=False, enable_gqa=enable_gqa
            )

    output = output.transpose(1, 2).flatten(2)
    return out_proj(output) if out_proj is not None else output
