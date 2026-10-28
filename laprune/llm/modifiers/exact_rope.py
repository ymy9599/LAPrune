import torch
from transformers.models.llama.modeling_llama import rotate_half

def _cached_contiguous_frequencies(rotary, tensor: torch.Tensor, positions: torch.LongTensor):
    start = int(positions[0, 0]) if positions.numel() else 0
    length = int(positions.shape[-1])
    key = (str(tensor.device), tensor.dtype, int(positions.shape[0]), start, length)
    cached = getattr(rotary, '_oracle_contiguous_rope', None)
    if cached is None or cached[0] != key:
        (cos, sin) = rotary(tensor, positions)
        rotary._oracle_contiguous_rope = (key, cos, sin)
    else:
        (cos, sin) = (cached[1], cached[2])
    return (cos, sin)

def apply_model_rope(attn, query: torch.Tensor, key: torch.Tensor):
    rotary = getattr(attn, 'rotary_emb', None)
    if rotary is None:
        raise RuntimeError('Exact model rotary_emb is not attached to attention')
    batch_size = int(query.shape[0])
    query_len = int(query.shape[-2])
    key_len = int(key.shape[-2])
    if query_len > key_len:
        raise ValueError(f'query_len={query_len} exceeds key_len={key_len}')
    key_pos = torch.arange(key_len, device=key.device, dtype=torch.long)
    key_pos = key_pos.unsqueeze(0).expand(batch_size, -1)
    query_pos = key_pos[:, key_len - query_len:]
    (cos_k, sin_k) = _cached_contiguous_frequencies(rotary, key, key_pos)
    cos_q = cos_k[:, key_len - query_len:]
    sin_q = sin_k[:, key_len - query_len:]
    q_rope = query * cos_q.unsqueeze(1) + rotate_half(query) * sin_q.unsqueeze(1)
    k_rope = key * cos_k.unsqueeze(1) + rotate_half(key) * sin_k.unsqueeze(1)
    return (q_rope, k_rope)
