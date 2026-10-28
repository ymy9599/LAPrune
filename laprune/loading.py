from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'llama31-8b-instruct': 'meta-llama/Llama-3.1-8B-Instruct',
}

def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()

def load_method(*, method, model_name, dtype='fp16', config=None, device_map='auto',
                dense_backend=None):
    if method not in {'dense', 'oracle', 'laprune'}:
        raise ValueError('method must be dense, oracle, or laprune')
    alias = model_name if model_name in MODELS else None
    model_name = MODELS.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    kwargs = {'torch_dtype': {'fp16': torch.float16, 'bf16': torch.bfloat16}[dtype],
              'device_map': device_map}
    if dense_backend is not None:
        kwargs['attn_implementation'] = dense_backend
    base = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
    base.generation_config.do_sample = False
    base.generation_config.num_beams = 1
    base.generation_config.temperature = None
    base.generation_config.top_p = None
    base.generation_config.top_k = None
    if method == 'dense':
        return tokenizer, base
    if config is None:
        if alias is None:
            raise ValueError('Specify --config when using a local model path')
        config = f'configs/quality/{alias}/longbench_{method}.json'
    if method == 'laprune':
        from .llm.modifiers.aq_generation import AQGeneration
        model = AQGeneration(base, None, None, str(resolve_path(config)))
    else:
        from .llm.modifiers.oracle import UpperBound
        model = UpperBound(base, None, None, str(resolve_path(config)))
    return tokenizer, model.eval()
