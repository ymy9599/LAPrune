import torch
import types
import time
from ...attention import do_sdpa_attn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import repeat_kv
from torch.nn import CrossEntropyLoss
from ..modifier import Modifier
from typing import List, Tuple
import json
from .exact_rope import apply_model_rope
from .aq_evaluation import _gather_pruned_gqa, _gqa_gather_kv_cuda

def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _profile_add(attn_module, key: str, value: float):
    if not bool(getattr(attn_module, '_profile_enabled', False)):
        return
    stats = getattr(attn_module, '_profile_stats', None)
    if not isinstance(stats, dict):
        stats = {}
        attn_module._profile_stats = stats
    stats[key] = float(stats.get(key, 0.0) + float(value))

def _timed(enabled: bool):
    if enabled:
        _sync_cuda()
        return time.perf_counter()
    return None

def _timed_end(attn_module, enabled: bool, t0, key: str):
    if not enabled or t0 is None:
        return
    _sync_cuda()
    _profile_add(attn_module, key, time.perf_counter() - t0)

def model_forward(self, input_ids: torch.LongTensor, labels: torch.Tensor=None, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]=None, **kwargs):
    (hidden_states, kv_cache) = self.model(input_ids=input_ids, kv_cache=kv_cache)
    logits = self.lm_head(hidden_states[..., -1:, :])
    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=kv_cache)

def model_model_forward(self, input_ids: torch.LongTensor=None, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]=None):
    inputs_embeds = self.embed_tokens(input_ids)
    hidden_states = inputs_embeds
    if kv_cache is None:
        kv_cache = [None] * len(self.layers)
    for (layer_idx, (decoder_layer, kv_cache_layer)) in enumerate(zip(self.layers, kv_cache)):
        layer_output = decoder_layer(hidden_states, kv_cache_layer)
        (hidden_states, kv_cache_layer) = layer_output
        kv_cache[layer_idx] = kv_cache_layer
    hidden_states = self.norm(hidden_states)
    return (hidden_states, kv_cache)

def layer_forward(self, hidden_states: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor]=None):
    device = self.self_attn.q_proj.weight.data.device
    if hidden_states.device != device:
        hidden_states = hidden_states.to(device)
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    (hidden_states, kv_cache) = self.self_attn(hidden_states, kv_cache)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return (hidden_states, kv_cache)

def self_attn_forward(self, hidden_states: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor]=None):
    profile_enabled = bool(getattr(self, '_profile_enabled', False))
    stage_prefix = 'prefill' if hidden_states.shape[-2] > 1 and kv_cache is None else 'decode'
    t_layer = _timed(profile_enabled)
    (num_heads, embed_dim) = (self.config.num_attention_heads, self.config.hidden_size)
    num_kv_heads = self.config.num_key_value_heads
    num_kv_group = num_heads // num_kv_heads
    head_dim = embed_dim // num_heads
    prefill_cond1 = hidden_states.shape[-2] > 1
    prefill_cond2 = kv_cache is None
    is_prefill = prefill_cond1 and prefill_cond2
    t = _timed(profile_enabled)
    ques = self.q_proj(hidden_states).unflatten(-1, (num_heads, head_dim)).transpose(1, 2)
    keys = self.k_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1, 2)
    vals = self.v_proj(hidden_states).unflatten(-1, (num_kv_heads, head_dim)).transpose(1, 2)
    _timed_end(self, profile_enabled, t, f'{stage_prefix}_qkv_proj_s')
    t = _timed(profile_enabled)
    if kv_cache is not None:
        (key_cache, val_cache) = kv_cache
        keys = torch.cat([key_cache, keys], dim=-2)
        vals = torch.cat([val_cache, vals], dim=-2)
    _timed_end(self, profile_enabled, t, f'{stage_prefix}_kv_cache_cat_s')
    kv_cache = (keys.data, vals.data)
    t = _timed(profile_enabled)
    (Q_rope, K_rope) = apply_model_rope(self, ques, keys)
    _timed_end(self, profile_enabled, t, f'{stage_prefix}_rope_s')
    cond1 = self.draft_kwargs['enable'] is True
    cond2 = not self.is_fix_layer
    cond3 = not is_prefill
    execution_backend = str(self.draft_kwargs.get('execution_backend', 'mask')).lower()
    if cond1 and cond2 and cond3:
        t = _timed(profile_enabled)
        K_hash_rope = repeat_kv(K_rope, num_kv_group)
        draft_score = Q_rope @ K_hash_rope.transpose(-1, -2)
        _timed_end(self, profile_enabled, t, 'decode_score_s')

        def aggregate_topk(x, k):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            (_, x_topk) = x.topk(k=k, dim=-1)
            return x_topk
        t = _timed(profile_enabled)
        num_kv_pair = draft_score.shape[-1]
        sink_size = int(self.draft_kwargs.get('sink_size', 0))
        local_window = int(self.draft_kwargs.get('local_window', 0))
        sink_end = min(max(sink_size, 0), num_kv_pair)
        local_start = max(num_kv_pair - max(local_window, 0), sink_end)
        cand_start = sink_end
        cand_end = local_start if local_window > 0 else num_kv_pair
        cand_len = max(0, cand_end - cand_start)
        if cand_len > 0:
            num_remain = cand_len - int(cand_len * self.draft_kwargs['mask_out'])
            num_remain = max(min(cand_len, self.draft_kwargs['min_remain']), num_remain)
            draft_indices = aggregate_topk(draft_score[..., cand_start:cand_end], num_remain) + cand_start
        else:
            draft_indices = torch.empty((*draft_score.shape[:-1], 0), dtype=torch.long, device=draft_score.device)
        _timed_end(self, profile_enabled, t, 'decode_recall_topk_s')
        if execution_backend == 'gather':
            if int(Q_rope.shape[-2]) != 1:
                raise RuntimeError('Oracle gather backend supports decode (Tq=1) only')
            keep_parts = []
            if sink_end > 0:
                sink_idx = torch.arange(sink_end, device=draft_indices.device, dtype=torch.long).view(1, 1, 1, -1).expand(*draft_indices.shape[:-1], -1)
                keep_parts.append(sink_idx)
            keep_parts.append(draft_indices)
            if local_window > 0 and local_start < num_kv_pair:
                local_idx = torch.arange(local_start, num_kv_pair, device=draft_indices.device, dtype=torch.long).view(1, 1, 1, -1).expand(*draft_indices.shape[:-1], -1)
                keep_parts.append(local_idx)
            selected_indices = torch.cat(keep_parts, dim=-1).sort(dim=-1).values
            t = _timed(profile_enabled)
            keep_idx = selected_indices.squeeze(2)
            if _gqa_gather_kv_cuda is not None and K_rope.dtype == torch.float16 and (vals.dtype == torch.float16) and (keep_idx.dtype == torch.long) and (head_dim % 2 == 0):
                (K_selected, V_selected) = _gqa_gather_kv_cuda(K_rope, vals, keep_idx, num_kv_group)
            else:
                K_selected = _gather_pruned_gqa(K_rope, keep_idx, num_heads)
                V_selected = _gather_pruned_gqa(vals, keep_idx, num_heads)
            _timed_end(self, profile_enabled, t, 'decode_gather_s')
            t = _timed(profile_enabled)
            attn_output = do_sdpa_attn(query=Q_rope, key=K_selected, value=V_selected, cos=None, sin=None, mask=None, out_proj=self.o_proj, already_rope=True)
            _timed_end(self, profile_enabled, t, 'decode_attn_s')
        else:
            t = _timed(profile_enabled)
            mask = torch.full((ques.shape[0], num_heads, ques.shape[-2], num_kv_pair), fill_value=torch.finfo(draft_score.dtype).min, dtype=draft_score.dtype, device=draft_score.device)
            mask = mask.scatter_(dim=-1, index=draft_indices, value=0)
            if sink_end > 0:
                mask[..., :sink_end] = 0
            if local_window > 0 and local_start < num_kv_pair:
                mask[..., local_start:num_kv_pair] = 0
            _timed_end(self, profile_enabled, t, 'decode_mask_build_s')
            t = _timed(profile_enabled)
            attn_output = do_sdpa_attn(query=Q_rope, key=K_rope, value=vals, cos=None, sin=None, mask=mask, out_proj=self.o_proj, already_rope=True)
            _timed_end(self, profile_enabled, t, 'decode_attn_s')
    else:
        t = _timed(profile_enabled)
        attn_output = do_sdpa_attn(query=Q_rope, key=K_rope, value=vals, cos=None, sin=None, out_proj=self.o_proj, already_rope=True)
        _timed_end(self, profile_enabled, t, f'{stage_prefix}_attn_s')
    _timed_end(self, profile_enabled, t_layer, f'{stage_prefix}_layer_total_s')
    return (attn_output, kv_cache)

class Decoder(torch.nn.Module):

    @property
    def layers(self):
        return self.decoder.model.layers

    @property
    def model(self):
        return self.decoder

    def reset(self):
        for layer in self.layers:
            if hasattr(layer.self_attn, 'k_cache'):
                del layer.self_attn.k_cache
                del layer.self_attn.v_cache

    def __init__(self, decoder, draft_kwargs):
        super().__init__()
        self.decoder = decoder
        fix_layers = draft_kwargs.get('fix_layers', [])
        self.fix_layers = fix_layers
        self.draft_kwargs = draft_kwargs
        self.profile_enabled = bool(draft_kwargs.get('profile_latency', False))
        self.model.forward = types.MethodType(model_forward, self.model)
        self.model.model.forward = types.MethodType(model_model_forward, self.model.model)
        rotary_emb = getattr(self.model.model, 'rotary_emb', None)
        for (layer_idx, layer) in enumerate(self.layers):
            layer.self_attn.is_fix_layer = layer_idx in fix_layers
            layer.self_attn.rotary_emb = rotary_emb or getattr(layer.self_attn, 'rotary_emb', None)
            layer.self_attn.draft_kwargs = draft_kwargs
            layer.forward = types.MethodType(layer_forward, layer)
            layer.self_attn.forward = types.MethodType(self_attn_forward, layer.self_attn)
            layer.self_attn._profile_enabled = bool(self.profile_enabled)
            layer.self_attn._profile_stats = {}

    def is_benchmark_mode(self):
        return self.draft_kwargs['bench_mark']

    def enable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = True

    def disable_benchmark_mode(self):
        self.draft_kwargs['bench_mark'] = False

    def layer_ft_params(self, layer):
        return []

    def ft_params(self, layer=None):
        return []

    def forward(self, input_ids, labels=None, kv_cache=None):
        outputs = self.decoder(input_ids=input_ids, labels=labels, kv_cache=kv_cache)
        return outputs

class Model(torch.nn.Module):

    def __init__(self, decoder: Decoder):
        super().__init__()
        self.decoder = decoder

    def ft_params(self):
        return []

    def reset(self):
        self.decoder.reset()

    def forward(self, input_ids, kv_cache=None, labels=None, local_rank=None, **kwargs):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)[None, :]
            labels = torch.tensor(labels, dtype=torch.int64)[None, :]
        label_exist = labels is not None
        rank_exist = local_rank is not None
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0, 1)
        if label_exist and labels.ndim == 3:
            labels = labels.flatten(0, 1)
        if rank_exist:
            device = torch.device(local_rank)
        else:
            device = next(iter(self.decoder.parameters())).device
        input_ids = input_ids.to(device)
        outputs = self.decoder(input_ids, labels=labels, kv_cache=kv_cache)
        return outputs

class UpperBound(Modifier):

    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        draft_kwargs = self.conf['draft_kwargs']
        decoder = Decoder(model, draft_kwargs=draft_kwargs)
        decoder = Model(decoder)
        super().__init__(decoder, save_ckp, load_ckp)

    def ft_params(self):
        return []

    def reset(self):
        self.model.reset()

    def is_benchmark_mode(self):
        return self.model.decoder.is_benchmark_mode()

    def enable_benchmark_mode(self):
        return self.model.decoder.enable_benchmark_mode()

    def disable_benchmark_mode(self):
        return self.model.decoder.disable_benchmark_mode()

    def _profile_enabled(self):
        try:
            return bool(getattr(self.model.decoder, 'profile_enabled', False))
        except Exception:
            return False

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=128, eos_token_id=[2], stopping_criteria=None, **kwargs):
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0, 1)
        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)
        kv_cache = None
        profile_on = self._profile_enabled()
        if profile_on:
            for lyr in self.model.decoder.layers:
                lyr.self_attn._profile_stats = {}
        prefill_t0 = time.perf_counter() if profile_on else None
        if input_ids.shape[1] > 1:
            prefill_ids = input_ids[:, :-1]
            output = self.model(input_ids=prefill_ids)
            kv_cache = output.past_key_values
            new_tok = input_ids[:, -1:]
        else:
            new_tok = input_ids
        prefill_s = time.perf_counter() - prefill_t0 if profile_on else None
        if eos_token_id is None:
            eos_ids = []
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids = [int(x) for x in eos_token_id]
        else:
            eos_ids = [int(eos_token_id)]
        pad_id = eos_ids[0] if len(eos_ids) > 0 else 0
        bsz = int(input_ids.shape[0])
        new_ids = [[] for _ in range(bsz)]
        finished = [False] * bsz
        decode_steps = 0
        decode_t0 = time.perf_counter() if profile_on else None
        for _ in range(int(max_new_tokens)):
            output = self.model(input_ids=new_tok, kv_cache=kv_cache)
            (logits, kv_cache) = (output.logits, output.past_key_values)
            next_tok = logits.argmax(dim=-1)
            decode_steps += 1
            step_tokens = next_tok.squeeze(-1).tolist()
            all_finished = True
            for i in range(bsz):
                if finished[i]:
                    continue
                tok = int(step_tokens[i])
                if tok in eos_ids:
                    finished[i] = True
                    continue
                new_ids[i].append(tok)
                all_finished = False
            if all_finished:
                break
            if stopping_criteria is not None and bsz == 1 and new_ids[0]:
                generated = torch.tensor(new_ids[0], dtype=input_ids.dtype, device=input_ids.device)[None, :]
                if bool(stopping_criteria(torch.cat([input_ids, generated], dim=-1), None)):
                    break
            new_tok = next_tok
        decode_s = time.perf_counter() - decode_t0 if profile_on else None
        self.model.reset()
        if profile_on:
            prof = {}
            for lyr in self.model.decoder.layers:
                st = getattr(lyr.self_attn, '_profile_stats', None)
                if isinstance(st, dict):
                    for (k, v) in st.items():
                        if isinstance(v, (int, float)):
                            prof[k] = float(prof.get(k, 0.0) + float(v))
            prof['prefill_total_s'] = float(prefill_s if prefill_s is not None else 0.0)
            prof['decode_total_s'] = float(decode_s if decode_s is not None else 0.0)
            prof['decode_steps'] = float(decode_steps)
            self.last_profile_stats = prof
        max_gen = max((len(x) for x in new_ids), default=0)
        if max_gen == 0:
            return input_ids
        new_ids_t = torch.full((bsz, max_gen), int(pad_id), dtype=input_ids.dtype, device=input_ids.device)
        for (i, toks) in enumerate(new_ids):
            if len(toks) > 0:
                new_ids_t[i, :len(toks)] = torch.tensor(toks, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, new_ids_t], dim=-1)
