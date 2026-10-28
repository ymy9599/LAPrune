import os
import types
from typing import Dict, List, Tuple
import torch
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, repeat_kv, apply_rotary_pos_emb
from ...attention import do_sdpa_attn
from ..modifier import Modifier
from ..aq_kv import AQLayerConfig, ShardedAQLayerCodebookManager
try:
    from aq_kernels import gqa_gather_kv as _gqa_gather_kv_cuda, gqa_gather_kv_with_tail as _gqa_gather_kv_tail_cuda
except Exception:
    _gqa_gather_kv_cuda = None
    _gqa_gather_kv_tail_cuda = None

def model_forward(self, input_ids: torch.LongTensor, labels: torch.Tensor=None, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]=None, return_full_logits: bool=False, **_):
    (hidden_states, kv_cache_out, draft_attn, true_attn) = self.model(input_ids=input_ids, kv_cache=kv_cache)
    logits_hidden_states = hidden_states if labels is not None or return_full_logits else hidden_states[..., -1:, :]
    logits = self.lm_head(logits_hidden_states)
    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous().view(-1)
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        loss = loss_fct(shift_logits, shift_labels.to(shift_logits.device))
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=kv_cache_out, attentions=(draft_attn, true_attn))

def model_model_forward(self, input_ids: torch.LongTensor=None, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]=None):
    hidden_states = self.embed_tokens(input_ids)
    if kv_cache is None:
        kv_cache = [None] * len(self.layers)
    (draft_attns, true_attns) = ([], [])
    for (layer_idx, (decoder_layer, kv_cache_layer)) in enumerate(zip(self.layers, kv_cache)):
        (hidden_states, kv_cache_layer, draft_attn, true_attn) = decoder_layer(hidden_states, kv_cache_layer)
        draft_attns.append(draft_attn)
        true_attns.append(true_attn)
        kv_cache[layer_idx] = kv_cache_layer
    hidden_states = self.norm(hidden_states)
    return (hidden_states, kv_cache, draft_attns, true_attns)

def layer_forward(self, hidden_states: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor]=None):
    device = self.self_attn.q_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    (hidden_states, kv_cache, draft_attn, true_attn) = self.self_attn(hidden_states, kv_cache)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return (hidden_states, kv_cache, draft_attn, true_attn)

def _cached_position_ids(owner, device: torch.device, end: int) -> torch.Tensor:
    """Return a shared device arange buffer, growing only when necessary."""
    cache = getattr(owner, '_laprune_position_ids', None)
    if cache is None or cache.device != device or int(cache.numel()) < int(end):
        capacity = max(int(end), 1024)
        cache = torch.arange(capacity, device=device, dtype=torch.long)
        owner._laprune_position_ids = cache
    return cache

def _kv_cache_append(self_attn, k_new, v_new, total_len, input_kv_cache):
    """Append (k_new, v_new) into a growable ring buffer on `self_attn`.

    Returns:
      K_view, V_view: full-prefix [B,H,total_len,D] views into the buffer.
      kv_cache_out:   (K_view.data, V_view.data) tuple for upstream consumers.
    """
    k_buf = getattr(self_attn, 'k_cache', None)
    v_buf = getattr(self_attn, 'v_cache', None)
    k_len = int(getattr(self_attn, 'k_cache_len', 0))
    k_cap = int(getattr(self_attn, 'k_cache_capacity', 0))
    (B, H, Tn, D) = k_new.shape
    conf = getattr(self_attn, 'draft_kwargs', {}) or {}
    extra_capacity = max(0, int(conf.get('aq_kv_extra_capacity', 0)))
    if k_buf is None or v_buf is None or k_cap <= 0:
        if input_kv_cache is not None:
            (k_prefix, v_prefix) = input_kv_cache
            k_len = int(k_prefix.shape[-2])
            k_cap = max(total_len, max(16, k_len) * 2)
            k_buf = k_prefix.new_empty((B, H, k_cap, D))
            v_buf = v_prefix.new_empty((B, H, k_cap, D))
            if k_len > 0:
                k_buf[:, :, :k_len, :] = k_prefix
                v_buf[:, :, :k_len, :] = v_prefix
        else:
            k_len = 0
            k_cap = max(total_len + extra_capacity, 16)
            k_buf = k_new.new_empty((B, H, k_cap, D))
            v_buf = v_new.new_empty((B, H, k_cap, D))
    if total_len > k_cap:
        new_cap = max(total_len, k_cap * 2)
        k_buf_new = k_buf.new_empty((B, H, new_cap, D))
        v_buf_new = v_buf.new_empty((B, H, new_cap, D))
        if k_len > 0:
            k_buf_new[:, :, :k_len, :] = k_buf[:, :, :k_len, :]
            v_buf_new[:, :, :k_len, :] = v_buf[:, :, :k_len, :]
        (k_buf, v_buf, k_cap) = (k_buf_new, v_buf_new, new_cap)
    end = k_len + int(Tn)
    k_buf[:, :, k_len:end, :] = k_new
    v_buf[:, :, k_len:end, :] = v_new
    k_len = end
    self_attn.k_cache = k_buf
    self_attn.v_cache = v_buf
    self_attn.k_cache_len = k_len
    self_attn.k_cache_capacity = k_cap
    k_view = k_buf[:, :, :k_len, :]
    v_view = v_buf[:, :, :k_len, :]
    return (k_view, v_view, (k_view.data, v_view.data))

def _expand_headwise_codes(codes: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Map native GQA KV-head codes onto query heads.

    Codes are compact (uint8 in normal LAPrune runs), so repeating this tensor is
    inexpensive compared with repeating the fp16 K/V cache itself.
    """
    num_kv_heads = int(codes.shape[1])
    num_query_heads = int(num_query_heads)
    if num_kv_heads == num_query_heads:
        return codes
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f'Invalid GQA code heads: Hq={num_query_heads} Hkv={num_kv_heads}')
    return codes.repeat_interleave(num_query_heads // num_kv_heads, dim=1)

def _gather_pruned_gqa(tensor: torch.Tensor, keep_idx: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Gather per-query-head token sets from a native GQA KV cache.

    Expanding the entire [B,Hkv,T,D] cache to Hq at 128K costs tens of GiB.
    This expands only the selected ~2% token set used by LAPrune decode.
    """
    (batch_size, num_kv_heads, _, head_dim) = tensor.shape
    num_query_heads = int(num_query_heads)
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(f'Invalid GQA gather heads: Hq={num_query_heads} Hkv={num_kv_heads}')
    group_size = num_query_heads // num_kv_heads
    keep_count = int(keep_idx.shape[-1])
    idx = keep_idx.reshape(batch_size, num_kv_heads, group_size, keep_count)
    idx = idx.unsqueeze(-1).expand(batch_size, num_kv_heads, group_size, keep_count, head_dim)
    source = tensor.unsqueeze(2).expand(batch_size, num_kv_heads, group_size, tensor.shape[2], head_dim)
    gathered = torch.take_along_dim(source, idx, dim=3)
    return gathered.reshape(batch_size, num_query_heads, keep_count, head_dim)

def _decode_attention(self_attn, manager, layer_idx, Q_rope, K_rope, vals_full, aq_cache, conf):
    (B, H, Tq, D) = Q_rope.shape
    cfg = manager.layer_cfgs.get(layer_idx, None)
    codes = aq_cache['codes']
    seq_len_codes = int(aq_cache.get('seq_len', codes.shape[2]))
    seq_len_codes = max(0, min(seq_len_codes, codes.shape[2]))
    codes = codes[:, :, :seq_len_codes, :]
    H_kv_codes = int(codes.shape[1])
    Tk_codes = int(codes.shape[2])
    Tk_total = int(getattr(self_attn, 'total_seq_len', Tk_codes))
    q_last = Q_rope[:, :, -1:, :]
    scoring_m = int(conf.get('aq_scoring_m', 32))
    if scoring_m != 32:
        raise ValueError(f'aq_scoring_m must be 32, got {scoring_m}')
    if cfg is not None and cfg.heads_per_group >= H:
        q_flat = q_last[..., 0, :].reshape(B * H, -1)
        codes_flat = codes.reshape(B * H_kv_codes, Tk_codes, -1)
        if H % H_kv_codes != 0:
            raise ValueError(f'Invalid GQA code heads: Hq={H}, Hkv={H_kv_codes}')
        scores_flat = manager.approx_inner_product_multihead_gqa(layer_idx, q_flat, codes_flat, H // H_kv_codes, scoring_m=scoring_m)
        scores_all = scores_flat.reshape(B, H, Tk_codes).to(device=vals_full.device, dtype=vals_full.dtype)
    else:
        if scoring_m != 32:
            raise RuntimeError('Only full-codebook AQ scoring is supported; refusing to silently fall back to exact M=32 scoring')
        codes = _expand_headwise_codes(codes, H)
        scores_all = torch.empty((B, H, Tk_codes), device=vals_full.device, dtype=vals_full.dtype)
        for b in range(B):
            for h in range(H):
                q_vec = q_last[b, h, 0, :].to(dtype=manager.dtype)
                head_codes = codes[b, h, :, :].reshape(Tk_codes, -1)
                scores_all[b, h, :] = manager.approx_inner_product(layer_idx, h, q_vec, head_codes).to(device=vals_full.device, dtype=vals_full.dtype)
    if scores_all.shape != (B, H, Tk_codes):
        raise RuntimeError(f'AQ score shape mismatch: expected {(B, H, Tk_codes)}, got {tuple(scores_all.shape)}')
    if conf.get('aq_validate_scoring_output', False) and (not getattr(self_attn, '_aq_scoring_validated', False)):
        if not bool(torch.isfinite(scores_all).all().item()):
            raise RuntimeError(f'Non-finite AQ scores at layer {layer_idx}')
        self_attn._aq_scoring_validated = True
    self_attn._aq_scoring_m_last = scoring_m
    self_attn._aq_scoring_m_calls = int(getattr(self_attn, '_aq_scoring_m_calls', 0)) + 1
    mask_out = conf.get('mask_out', 0.98)
    min_remain = int(conf.get('min_remain', 20))
    sink_size = int(conf.get('sink_size', 4))
    local_window = int(conf.get('local_window', 64))
    sink_end = min(sink_size, Tk_codes) if sink_size > 0 else 0
    local_start = max(Tk_total - max(local_window, 0), 0)
    cand_start = sink_end
    cand_end = min(Tk_codes, local_start) if local_window > 0 else Tk_codes
    num_candidate = max(0, cand_end - cand_start)
    if num_candidate > 0:
        num_remain_cand = num_candidate - int(num_candidate * mask_out)
        num_remain_cand = max(min(num_candidate, min_remain), num_remain_cand)
        num_remain_cand = max(1, min(num_remain_cand, num_candidate))
        scores_cand = scores_all[..., cand_start:cand_end]
        topk_idx = torch.topk(scores_cand, k=num_remain_cand, dim=-1, sorted=False).indices
        if cand_start:
            topk_idx = topk_idx + cand_start
        draft_indices = topk_idx.unsqueeze(2)
    else:
        draft_indices = torch.empty((B, H, 1, 0), dtype=torch.long, device=vals_full.device)
        num_remain_cand = 0
    use_pruned_attn_default = bool(conf.get('aq_use_pruned_attn', False))
    use_pruned_attn = bool(conf.get('aq_use_pruned_attn_decode', use_pruned_attn_default))
    decode_full_attn = bool(conf.get('aq_decode_full_attention', False))
    can_prune = use_pruned_attn and (not decode_full_attn) and (Tq == 1)
    if can_prune:
        sb = min(sink_size, Tk_total) if sink_size > 0 else 0
        start = max(Tk_total - local_window, 0) if local_window > 0 else Tk_total
        if start <= 0:
            sb_eff = 0
        else:
            sb_eff = sb
            if start < sb_eff:
                start = sb_eff
        implicit_tail_start = None
        if sb_eff == 0 and local_window > 0 and (start < Tk_total) and (draft_indices.numel() > 0) and (K_rope.shape[1] != H) and (K_rope.dtype == torch.float16) and (vals_full.dtype == torch.float16) and (draft_indices.dtype in (torch.long, torch.int32)) and (draft_indices.dtype == torch.long and _gqa_gather_kv_tail_cuda is not None):
            keep_idx = draft_indices[..., -1, :]
            implicit_tail_start = start
        else:
            parts = []
            if draft_indices.numel() > 0:
                parts.append(draft_indices[..., -1, :])
            if sb_eff > 0:
                sink_idx = torch.arange(sb_eff, device=vals_full.device, dtype=torch.long)
                parts.append(sink_idx.view(1, 1, -1).expand(B, H, -1))
            if local_window > 0 and start < Tk_total:
                position_ids = _cached_position_ids(manager, vals_full.device, Tk_total)
                local_idx = position_ids[start:Tk_total]
                parts.append(local_idx.view(1, 1, -1).expand(B, H, -1))
            if not parts:
                keep_idx = torch.tensor([Tk_total - 1], device=vals_full.device, dtype=torch.long).view(1, 1, -1).expand(B, H, -1)
            else:
                keep_idx = torch.cat(parts, dim=-1)
        num_kv_heads = int(K_rope.shape[1])
        if K_rope.shape[1] == H:
            gather_idx = keep_idx.unsqueeze(-1).expand(B, H, keep_idx.shape[-1], D)
            key_for_attn = torch.take_along_dim(K_rope, gather_idx, dim=-2).contiguous()
            val_for_attn = torch.take_along_dim(vals_full, gather_idx, dim=-2).contiguous()
        elif _gqa_gather_kv_cuda is not None and K_rope.dtype == torch.float16 and (vals_full.dtype == torch.float16) and (keep_idx.dtype == torch.long) and (D % 2 == 0):
            if implicit_tail_start is not None:
                (key_for_attn, val_for_attn) = _gqa_gather_kv_tail_cuda(K_rope, vals_full, keep_idx, H // num_kv_heads, implicit_tail_start, Tk_total - implicit_tail_start)
            else:
                (key_for_attn, val_for_attn) = _gqa_gather_kv_cuda(K_rope, vals_full, keep_idx, H // num_kv_heads)
        else:
            key_for_attn = _gather_pruned_gqa(K_rope, keep_idx, H).contiguous()
            val_for_attn = _gather_pruned_gqa(vals_full, keep_idx, H).contiguous()
        attn_heads = torch.nn.functional.scaled_dot_product_attention(query=Q_rope, key=key_for_attn, value=val_for_attn, is_causal=False, attn_mask=None)
        attn_output = attn_heads.transpose(1, 2).flatten(2)
        return self_attn.o_proj(attn_output)
    mask = torch.zeros((1, H, Tq, Tk_total), dtype=vals_full.dtype, device=vals_full.device)
    last_q = Tq - 1
    row_mask = torch.full((1, H, 1, Tk_total), torch.finfo(vals_full.dtype).min, dtype=vals_full.dtype, device=vals_full.device)
    if draft_indices.numel() > 0:
        last_indices = draft_indices[..., -1, :].unsqueeze(2)
        row_mask = row_mask.scatter(dim=-1, index=last_indices, value=0)
    mask[..., last_q:last_q + 1, :] = row_mask
    offset = Tk_total - Tq
    query_pos = torch.arange(Tq, device=mask.device) + offset
    key_pos = torch.arange(Tk_total, device=mask.device)
    causal_cond = (query_pos[:, None] < key_pos[None, :])[None, None, :, :].expand_as(mask)
    mask = torch.where(causal_cond, torch.finfo(mask.dtype).min, mask)
    if sink_size > 0 or local_window > 0:
        for q_idx in range(Tq):
            bound = offset + q_idx + 1
            if sink_size > 0:
                sink_bound = min(sink_size, bound)
                if sink_bound > 0:
                    mask[..., q_idx, :sink_bound] = 0
            if local_window > 0:
                start = max(0, bound - local_window)
                if start < bound:
                    mask[..., q_idx, start:bound] = 0
    return do_sdpa_attn(query=Q_rope, key=K_rope, value=vals_full, cos=None, sin=None, mask=mask, out_proj=self_attn.o_proj, already_rope=True)

def self_attn_forward(self, hidden_states: torch.Tensor, kv_cache=None):
    manager = self.aq_manager
    layer_idx = self.layer_idx
    conf = self.draft_kwargs
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = self.config.hidden_size // num_heads
    batch_size = hidden_states.shape[0]
    T_new = hidden_states.shape[-2]
    input_kv_cache = kv_cache
    is_prefill = T_new > 1 and input_kv_cache is None
    ques = self.q_proj(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1, 2)
    keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1, 2)
    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1, 2)
    past_len = int(getattr(self, 'total_seq_len', 0))
    total_len = past_len + T_new
    position_ids = _cached_position_ids(manager, ques.device, total_len)
    pos_ids_new = position_ids[past_len:total_len][None, :].expand(batch_size, -1)
    rope_key = (str(ques.device), int(past_len), int(T_new), int(batch_size), ques.dtype)
    cached_rope = getattr(manager, '_laprune_rope_step', None)
    if layer_idx == 0 or cached_rope is None or cached_rope[0] != rope_key:
        (cos, sin) = self.rotary_emb(ques, pos_ids_new)
        manager._laprune_rope_step = (rope_key, cos, sin)
    else:
        (cos, sin) = (cached_rope[1], cached_rope[2])
    (Q_rope, K_rope_new) = apply_rotary_pos_emb(ques, keys, cos, sin)
    if layer_idx + 1 == int(self.config.num_hidden_layers):
        manager._laprune_rope_step = None
    (K_rope, vals_full, kv_cache_out) = _kv_cache_append(self, K_rope_new, vals, total_len, input_kv_cache)
    self.total_seq_len = total_len
    aq_enabled = conf.get('enable', True) and (not conf.get('aq_disable_encode', False))
    layer_aq = aq_enabled and (not self.is_fix_layer) and (not getattr(self, 'full_kv_layer', False)) and (layer_idx in manager.layer_cfgs)
    if is_prefill and conf.get('aq_disable_prefill', False):
        layer_aq = False
    if not layer_aq:
        attn_output = do_sdpa_attn(query=Q_rope, key=K_rope, value=vals_full, cos=None, sin=None, out_proj=self.o_proj, already_rope=True)
        return (attn_output, kv_cache_out, None, None)
    aq_cache = _prepare_aq_cache(self, manager, layer_idx, K_rope, need_recon=not conf.get('aq_use_exact_true_score', True), decode_group_size=int(conf.get('aq_decode_group_size', 1)), is_prefill=is_prefill)
    if is_prefill:
        attn_output = do_sdpa_attn(query=Q_rope, key=K_rope, value=vals_full, cos=None, sin=None, out_proj=self.o_proj, already_rope=True)
        return (attn_output, kv_cache_out, None, None)
    attn_output = _decode_attention(self, manager, layer_idx, Q_rope, K_rope, vals_full, aq_cache, conf)
    return (attn_output, kv_cache_out, None, None)

def _prepare_aq_cache(self, manager, layer_idx, keys, need_recon: bool, decode_group_size: int=1, is_prefill: bool=False):
    """Cache AQ encoding results for the current layer:
    - always cache `codes` (per layer, per head or per head-group) for LUT-based q·k approximation;
    - when need_recon=True, also reconstruct approximate K (used for debug or approximate-teacher setups).

    Cache layout (codes/approx may have extra capacity; only the first seq_len/encoded_len entries are valid):
        self._aq_cache = {
            "seq_len": seq_len,                      # logical total length (T_total)
            "encoded_len": encoded_len,              # encoded prefix length (<= seq_len)
            "batch_size": B,
            "codes": Tensor[B, H, T_cap, M],         # on manager.device, T_cap >= seq_len
            "approx": Optional[Tensor[B, H, T_cap,D]]# on keys.device/keys.dtype, only first seq_len entries are valid
        }
    """
    cache = getattr(self, '_aq_cache', None)
    (batch_size, num_heads, seq_len_k, head_dim) = keys.shape
    seq_len = seq_len_k
    conf_local = getattr(self, 'draft_kwargs', {}) or {}
    if cache is not None and cache.get('batch_size') == batch_size:
        cached_seq = cache.get('seq_len', 0)
        encoded_len = cache.get('encoded_len', cached_seq)
        if cached_seq == seq_len:
            if need_recon and cache.get('approx') is None:
                approx = torch.zeros(batch_size, num_heads, seq_len_k, head_dim, device=keys.device, dtype=keys.dtype)
                codes = cache['codes']
                valid_T = int(encoded_len)
                if valid_T > 0:
                    for head_idx in range(num_heads):
                        head_codes = codes[:, head_idx, :valid_T, :].reshape(batch_size * valid_T, -1)
                        recon = manager.reconstruct(layer_idx, head_idx, head_codes)
                        approx[:, head_idx, :valid_T, :] = recon.reshape(batch_size, valid_T, head_dim).to(keys.device, dtype=keys.dtype)
                cache['approx'] = approx
            cache['seq_len'] = seq_len
            cache['encoded_len'] = encoded_len
            return cache
        if seq_len_k > cached_seq:
            old_codes = cache['codes']
            (_, _, T_cap_old, M) = old_codes.shape
            codes_tensor = old_codes
            if seq_len_k > T_cap_old:
                new_cap = max(T_cap_old * 2 if T_cap_old > 0 else seq_len_k, seq_len_k)
                codes_tensor = old_codes.new_empty(batch_size, num_heads, new_cap, M)
                if encoded_len > 0:
                    codes_tensor[:, :, :encoded_len, :] = old_codes[:, :, :encoded_len, :]
                cache['codes'] = codes_tensor
            tail_since_last_encode = seq_len_k - encoded_len
            if not is_prefill and (not need_recon) and (decode_group_size > 1) and (tail_since_last_encode < decode_group_size):
                cache['seq_len'] = seq_len
                cache['encoded_len'] = encoded_len
                self._aq_cache = cache
                return cache
            tail_start = int(encoded_len)
            tail_len = seq_len_k - tail_start
            if tail_len <= 0:
                cache['seq_len'] = seq_len
                cache['encoded_len'] = encoded_len
                self._aq_cache = cache
                return cache
            cfg = manager.layer_cfgs.get(layer_idx, None)
            icm_iters_decode = int(conf_local.get('aq_encode_icm_iters_decode', 0))
            if cfg is not None and cfg.heads_per_group >= num_heads:
                tail_all = keys[:, :, tail_start:seq_len_k, :]
                tail_flat = tail_all.reshape(batch_size * num_heads * tail_len, head_dim)
                tail_codes_flat = manager.encode_keys_batch(layer_idx, 0, tail_flat.to(dtype=manager.dtype), icm_iters=icm_iters_decode)
                tail_codes = tail_codes_flat.reshape(batch_size, num_heads, tail_len, M)
                if tail_codes.dtype != codes_tensor.dtype:
                    tail_codes = tail_codes.to(dtype=codes_tensor.dtype)
                codes_tensor[:, :, tail_start:seq_len_k, :] = tail_codes
            else:
                for head_idx in range(num_heads):
                    head_tail = keys[:, head_idx, tail_start:seq_len_k, :]
                    head_tail_flat = head_tail.reshape(batch_size * tail_len, head_dim)
                    tail_codes = manager.encode_keys(layer_idx, head_idx, head_tail_flat.to(dtype=manager.dtype), icm_iters=icm_iters_decode)
                    tail_codes = tail_codes.reshape(batch_size, tail_len, -1)
                    if tail_codes.dtype != codes_tensor.dtype:
                        tail_codes = tail_codes.to(dtype=codes_tensor.dtype)
                    codes_tensor[:, head_idx, tail_start:seq_len_k, :] = tail_codes
            approx = cache.get('approx', None)
            if need_recon:
                if approx is None:
                    approx = torch.zeros(batch_size, num_heads, seq_len_k, head_dim, device=keys.device, dtype=keys.dtype)
                else:
                    (_, _, T_cap_approx_old, _) = approx.shape
                    if seq_len_k > T_cap_approx_old:
                        new_cap = max(T_cap_approx_old * 2 if T_cap_approx_old > 0 else seq_len_k, seq_len_k)
                        new_approx = approx.new_zeros(batch_size, num_heads, new_cap, head_dim)
                        if encoded_len > 0:
                            new_approx[:, :, :encoded_len, :] = approx[:, :, :encoded_len, :]
                        approx = new_approx
                for head_idx in range(num_heads):
                    head_codes = codes_tensor[:, head_idx, tail_start:seq_len_k, :].reshape(batch_size * tail_len, -1)
                    recon = manager.reconstruct(layer_idx, head_idx, head_codes)
                    approx[:, head_idx, tail_start:seq_len_k, :] = recon.reshape(batch_size, tail_len, head_dim).to(keys.device, dtype=keys.dtype)
            cache['seq_len'] = seq_len
            cache['encoded_len'] = seq_len
            cache['codes'] = codes_tensor
            cache['approx'] = approx
            self._aq_cache = cache
            return cache
    cfg = manager.layer_cfgs.get(layer_idx, None)
    if is_prefill:
        icm_iters_prefill = int(conf_local.get('aq_encode_icm_iters_prefill', 0))
    else:
        icm_iters_prefill = int(conf_local.get('aq_encode_icm_iters_prefill', 2))
    prefill_group_size = int(conf_local.get('aq_prefill_group_size', 0))
    if prefill_group_size <= 0:
        prefill_group_size = seq_len_k
    if cfg is not None and cfg.heads_per_group >= num_heads:
        if prefill_group_size >= seq_len_k:
            all_keys = keys.reshape(batch_size * num_heads * seq_len_k, head_dim)
            codes_flat = manager.encode_keys_batch(layer_idx, 0, all_keys.to(dtype=manager.dtype), icm_iters=icm_iters_prefill)
            M_actual = int(codes_flat.shape[-1])
            codes_tensor = codes_flat.reshape(batch_size, num_heads, seq_len_k, M_actual)
        else:
            codes_tensor = None
            M_actual = None
            for t0 in range(0, seq_len_k, prefill_group_size):
                t1 = min(seq_len_k, t0 + prefill_group_size)
                chunk = keys[:, :, t0:t1, :].reshape(batch_size * num_heads * (t1 - t0), head_dim)
                chunk_codes_flat = manager.encode_keys_batch(layer_idx, 0, chunk.to(dtype=manager.dtype), icm_iters=icm_iters_prefill)
                if M_actual is None:
                    M_actual = int(chunk_codes_flat.shape[-1])
                if codes_tensor is None:
                    codes_tensor = chunk_codes_flat.new_empty((batch_size, num_heads, seq_len_k, M_actual))
                codes_tensor[:, :, t0:t1, :] = chunk_codes_flat.reshape(batch_size, num_heads, t1 - t0, M_actual)
    else:
        codes_all = []
        for head_idx in range(num_heads):
            if prefill_group_size >= seq_len_k:
                head_keys = keys[:, head_idx, :, :]
                head_flat = head_keys.reshape(batch_size * seq_len_k, head_dim)
                codes = manager.encode_keys(layer_idx, head_idx, head_flat.to(dtype=manager.dtype), icm_iters=icm_iters_prefill)
                M_actual = int(codes.shape[-1])
                head_codes_tensor = codes.reshape(batch_size, seq_len_k, M_actual)
            else:
                head_codes_tensor = None
                M_actual = None
                for t0 in range(0, seq_len_k, prefill_group_size):
                    t1 = min(seq_len_k, t0 + prefill_group_size)
                    head_keys = keys[:, head_idx, t0:t1, :]
                    head_flat = head_keys.reshape(batch_size * (t1 - t0), head_dim)
                    codes = manager.encode_keys(layer_idx, head_idx, head_flat.to(dtype=manager.dtype), icm_iters=icm_iters_prefill)
                    if M_actual is None:
                        M_actual = int(codes.shape[-1])
                    if head_codes_tensor is None:
                        head_codes_tensor = codes.new_empty((batch_size, seq_len_k, M_actual))
                    head_codes_tensor[:, t0:t1, :] = codes.reshape(batch_size, t1 - t0, M_actual)
            codes_all.append(head_codes_tensor)
        codes_tensor = torch.stack(codes_all, dim=1)
    if bool(conf_local.get('aq_codes_compress', True)):
        try:
            if cfg is not None and cfg.K <= 256:
                codes_tensor = codes_tensor.to(dtype=torch.uint8)
        except Exception:
            pass
    approx = None
    if need_recon:
        approx = torch.zeros(batch_size, num_heads, seq_len_k, head_dim, device=keys.device, dtype=keys.dtype)
        for head_idx in range(num_heads):
            head_codes = codes_tensor[:, head_idx, :, :].reshape(batch_size * seq_len_k, -1)
            recon = manager.reconstruct(layer_idx, head_idx, head_codes)
            approx[:, head_idx, :, :] = recon.reshape(batch_size, seq_len_k, head_dim).to(keys.device, dtype=keys.dtype)
    code_extra_capacity = max(0, int(conf_local.get('aq_code_extra_capacity', 0)))
    if is_prefill and code_extra_capacity > 0:
        (Bc, Hc, Tc, Mc) = codes_tensor.shape
        codes_with_capacity = codes_tensor.new_empty((Bc, Hc, Tc + code_extra_capacity, Mc))
        codes_with_capacity[:, :, :Tc, :] = codes_tensor
        codes_tensor = codes_with_capacity
    cache = {'seq_len': seq_len, 'encoded_len': seq_len, 'batch_size': batch_size, 'codes': codes_tensor, 'approx': approx}
    self._aq_cache = cache
    return cache

class Decoder(torch.nn.Module):

    @property
    def layers(self):
        return self.decoder.model.layers

    @property
    def model(self):
        return self.decoder

    def reset(self):
        for layer in self.layers:
            attn = layer.self_attn
            for name in ('k_cache', 'v_cache', 'k_cache_len', 'k_cache_capacity', '_aq_cache', 'total_seq_len'):
                if hasattr(attn, name):
                    delattr(attn, name)

    def __init__(self, decoder, draft_kwargs, aq_manager):
        super().__init__()
        self.decoder = decoder
        self.draft_kwargs = draft_kwargs
        self.aq_manager = aq_manager
        self.apply_layers = set(draft_kwargs.get('apply_layers', []))
        self.full_kv_layers = set(draft_kwargs.get('full_kv_layers', []))
        rotary_emb = getattr(self.model.model, 'rotary_emb', None)
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        for (layer_idx, layer) in enumerate(self.layers):
            attn = layer.self_attn
            attn.layer_idx = layer_idx
            attn.is_fix_layer = layer_idx not in self.apply_layers
            attn.full_kv_layer = layer_idx in self.full_kv_layers
            attn.draft_kwargs = draft_kwargs
            attn.aq_enabled = layer_idx in self.apply_layers
            attn.aq_manager = aq_manager
            attn._aq_cache = None
            attn.rotary_emb = rotary_emb or getattr(attn, 'rotary_emb', None)
            layer.forward = types.MethodType(layer_forward, layer)
            attn.forward = types.MethodType(self_attn_forward, attn)

    def ft_params(self, layer=None):
        return []

    def forward(self, input_ids, labels=None, kv_cache=None, return_full_logits=False):
        return self.decoder(input_ids=input_ids, labels=labels, kv_cache=kv_cache, return_full_logits=return_full_logits)

class Model(torch.nn.Module):

    def __init__(self, decoder: Decoder):
        super().__init__()
        self.decoder = decoder

    def ft_params(self):
        return []

    def reset(self):
        self.decoder.reset()

    def forward(self, input_ids, kv_cache=None, labels=None, return_full_logits=False):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]
            labels = torch.tensor(labels, dtype=torch.int64)[None, :]
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0, 1)
        if labels is not None and labels.ndim == 3:
            labels = labels.flatten(0, 1)
        device = next(iter(self.decoder.parameters())).device
        input_ids = input_ids.to(device)
        return self.decoder(input_ids, labels=labels, kv_cache=kv_cache, return_full_logits=return_full_logits)

class AQEvaluation(Modifier):

    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        draft_kwargs = self.conf['draft_kwargs']
        (aq_manager, available_layers) = self._build_aq_manager(model, draft_kwargs)
        draft_kwargs['apply_layers'] = available_layers
        decoder = Decoder(model, draft_kwargs=draft_kwargs, aq_manager=aq_manager)
        decoder = Model(decoder)
        super().__init__(decoder, save_ckp, load_ckp)

    def _build_aq_manager(self, model, draft_kwargs):
        apply_layers = draft_kwargs.get('apply_layers', [])
        aq_codebook_dir = os.environ.get('LAPRUNE_AQ_CODEBOOK_DIR') or draft_kwargs.get('aq_codebook_dir', 'aq_codebooks/llama3-8b')
        if isinstance(aq_codebook_dir, str) and aq_codebook_dir and (not os.path.isabs(aq_codebook_dir)):
            laprune_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            aq_codebook_dir = os.path.join(laprune_root, aq_codebook_dir)
        if not os.path.isdir(str(aq_codebook_dir)):
            raise FileNotFoundError(f'aq_codebook_dir not found: {aq_codebook_dir}\n- Train codebooks into that directory (see LAPrune/scripts/train_codebooks_*.sh), or\n- Copy/symlink an existing codebook folder into LAPrune/aq_codebooks/.')
        heads_per_group = draft_kwargs.get('aq_heads_per_group', 2)
        aq_M = draft_kwargs.get('aq_M', 8)
        aq_K = draft_kwargs.get('aq_K', 256)
        config = model.config
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        if heads_per_group <= 0 or heads_per_group > num_heads:
            heads_per_group = num_heads
        layer_cfgs: Dict[int, AQLayerConfig] = {}
        for layer_idx in apply_layers:
            layer_cfgs[layer_idx] = AQLayerConfig(M=aq_M, K=aq_K, d=head_dim, heads_per_group=heads_per_group, num_heads=num_heads)
        devices = []
        try:
            hf_map = getattr(model, 'hf_device_map', None)
            if isinstance(hf_map, dict):
                for v in hf_map.values():
                    if isinstance(v, int):
                        devices.append(torch.device(f'cuda:{v}'))
                    elif isinstance(v, str) and v.startswith('cuda'):
                        devices.append(torch.device(v))
        except Exception:
            pass
        if devices:
            devices = sorted({torch.device(d) for d in devices}, key=lambda d: str(d))
        if not devices:
            try:
                devices = sorted({p.device for p in model.parameters() if p.is_cuda}, key=lambda d: str(d))
            except Exception:
                devices = []
        if not devices:
            devices = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]

        def _parse_dtype(x):
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip().lower()
                if s in ('fp16', 'float16'):
                    return torch.float16
                if s in ('bf16', 'bfloat16'):
                    return torch.bfloat16
                if s in ('fp32', 'float32'):
                    return torch.float32
            if isinstance(x, torch.dtype):
                return x
            return None
        try:
            model_dtype = next(iter(model.parameters())).dtype
        except Exception:
            model_dtype = torch.float16
        if model_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            model_dtype = torch.float16
        aq_dtype = _parse_dtype(draft_kwargs.get('aq_manager_dtype'))
        if aq_dtype is None:
            aq_dtype = torch.float16 if model_dtype == torch.bfloat16 else model_dtype
        manager = ShardedAQLayerCodebookManager(layer_cfgs=layer_cfgs, aq_codebook_dir=aq_codebook_dir, devices=devices, dtype=aq_dtype)
        try:
            total_cb_bytes = 0
            for ((_, _), aq) in manager._aq_per_group.items():
                if hasattr(aq, 'codebooks') and aq.codebooks is not None:
                    total_cb_bytes += aq.codebooks.numel() * aq.codebooks.element_size()
            total_cb_mb = total_cb_bytes / 1024 ** 2
            ndev = len(getattr(manager, 'devices', [])) or 1
            print(f'[AQ-MEM] total_codebooks_per_device={total_cb_mb:.1f} MB, replicas={ndev}, total≈{total_cb_mb * ndev:.1f} MB', flush=True)
        except Exception:
            pass
        available_layers = sorted({layer for (layer, _) in manager._aq_per_group.keys()})
        manager.layer_cfgs = {layer: manager.layer_cfgs[layer] for layer in available_layers if layer in manager.layer_cfgs}
        return (manager, available_layers)

    def ft_params(self):
        return []

    def reset(self):
        self.model.reset()
