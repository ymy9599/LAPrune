import os
import time
import torch
from typing import Dict
from ..modifier import Modifier
from ..aq_kv import AQLayerCodebookManager, ShardedAQLayerCodebookManager, AQLayerConfig
from .aq_evaluation import Decoder, Model

def _build_aq_manager(model, draft_kwargs):
    """
    Copied from aq_evaluation.AQEvaluation._build_aq_manager.
    Builds the AQ codebook manager so generation and evaluation share identical AQ configs and loading logic.
    """
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

class AQGeneration(Modifier):
    """
    Token-by-token generation wrapper for AQ:
      - run one prefill over the full prompt: standard causal attention, and also AQ-encode
        each layer's keys so decode has codes available;
      - decode token-by-token using kv_cache, with AQ top-k pruning applied in decode.
    """

    def __init__(self, model, save_ckp, load_ckp, config):
        self.get_conf(config)
        assert isinstance(self.conf, dict)
        draft_kwargs = dict(self.conf['draft_kwargs'])
        (aq_manager, available_layers) = _build_aq_manager(model, draft_kwargs)
        draft_kwargs['apply_layers'] = available_layers
        decoder = Decoder(model, draft_kwargs=draft_kwargs, aq_manager=aq_manager)
        decoder = Model(decoder)
        super().__init__(decoder, save_ckp, load_ckp)
        self.last_decode_time = None

    def reset(self):
        self.model.reset()

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int=128, eos_token_id=None, **kwargs):
        if eos_token_id is None:
            eos_token_id = [2]
        if input_ids.ndim == 3:
            input_ids = input_ids.flatten(0, 1)
        device = next(iter(self.model.parameters())).device
        input_ids = input_ids.to(device)
        kv_cache = None
        if input_ids.shape[1] > 1:
            prefill_ids = input_ids[:, :-1]
            output = self.model(input_ids=prefill_ids)
            kv_cache = output.past_key_values
        else:
            new_tok = input_ids
        if input_ids.shape[1] > 1:
            new_tok = input_ids[:, -1:]
        new_ids = []
        t_decode_start = time.perf_counter()
        stopping_criteria = kwargs.get('stopping_criteria')
        while len(new_ids) < max_new_tokens:
            output = self.model(input_ids=new_tok, kv_cache=kv_cache)
            (logits, kv_cache) = (output.logits, output.past_key_values)
            next_tok = logits.argmax(dim=-1)
            if next_tok.ravel().item() in eos_token_id:
                break
            new_ids.append(next_tok.ravel().item())
            if stopping_criteria is not None:
                generated = torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device)[None, :]
                if bool(stopping_criteria(torch.cat([input_ids, generated], dim=-1), None)):
                    break
            new_tok = next_tok
        t_decode_end = time.perf_counter()
        self.last_decode_time = t_decode_end - t_decode_start
        self.model.reset()
        if len(new_ids) == 0:
            return input_ids
        new_ids = torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device)[None, :]
        return torch.cat([input_ids, new_ids], dim=-1)
