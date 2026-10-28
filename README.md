# LAPrune

LAPrune accelerates LLM decoding with an attention-logit-aligned KV retrieval proxy built from additive quantization of post-RoPE keys. The base model requires no finetuning.

## Installation

Use Linux with an NVIDIA GPU and a CUDA toolkit. Tested with Python 3.12, CUDA 12.8, PyTorch 2.11.0, and Transformers 4.57.6.

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install setuptools packaging ninja
pip install -r requirements.txt --no-build-isolation
pip install -e . --no-build-isolation
```

Run the commands below from the repository root. Obtain access to the Llama checkpoints on Hugging Face before running.

## Models and codebooks

| Model | Hugging Face checkpoint | Included codebooks |
| --- | --- | --- |
| Llama-2-7B (codebooks only) | `meta-llama/Llama-2-7b-hf` | `codebooks/llama2-7b/` |
| `llama31-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | `codebooks/llama31-8b-instruct/` |

The pretrained codebooks are ready to use; no retraining is needed. Each layer uses 32 additive codebooks with 256 entries each. Experiment settings are in `configs/quality/`; decode benchmark settings are in `configs/decode/`.

## Quality evaluation

Evaluation and benchmarking use Llama-3.1-8B-Instruct. Supported quality methods: `dense`, `oracle`, and `laprune`. Replace `--method` and the output path to evaluate another method.

### LongBench

```bash
python evaluation/longbench/evaluate.py \
  --model llama31-8b-instruct --method laprune \
  --output-dir results/longbench/laprune

python evaluation/longbench/score.py \
  --pred-dir results/longbench/laprune \
  --output results/longbench/laprune_scores.json
```

The default evaluates the seven paper tasks using the included evaluation inputs. Use `--datasets` to select tasks or `--data-dir` to load local LongBench JSONL files.

### RULER

The exact evaluation inputs are included, so results do not depend on regenerating a different random test set.

```bash
python evaluation/ruler/evaluate.py \
  --model llama31-8b-instruct --method laprune \
  --data evaluation/ruler/data/llama31_128k.jsonl.gz \
  --tasks niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue \
  --context-lengths 128000 --samples-per-task 100 \
  --output results/ruler/laprune.json
```

RULER uses 4 sink tokens, a 64-token local window, and a 64-token encoding buffer; it stops once all required answers appear. LongBench uses no sink/local window and encodes each new key immediately. Both use full attention in layers 0 and 1.

## Decode benchmark

The benchmark supports Llama-3.1-8B-Instruct and reports steady-state decode throughput averaged over three fresh-process runs. Available methods are `laprune`, `flashattention2`, and `vanilla` (HF SDPA).

```bash
python benchmarks/decode.py \
  --method laprune --prompt-tokens 128000 \
  --decode-tokens 256 --batch-size 1 \
  --output results/decode/laprune.json
```

Adjust prompt length, decode length, and batch size as needed; use `--repeats 1` for a single run. The paper's LAPrune speed configuration uses a 64-token local encoding buffer. For comparable timings, run on an otherwise idle GPU.

## Train your own codebooks

Supply JSONL corpora with a `text` field, such as `data/arxiv.jsonl` and `data/books.jsonl`.

```bash
# Llama-3.1-8B-Instruct
python training/train_llama31_codebooks.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --corpus_paths data/arxiv.jsonl data/books.jsonl \
  --layers $(seq 0 31) --M 32 --K 256 \
  --save_dir codebooks/custom-llama31-8b-instruct
```

Set `aq_codebook_dir` in a copied method config to your new directory and pass it with `--config`. Each training script exposes additional options through `--help`.

## Citation

If you find LAPrune useful, please cite our paper:

```bibtex
@inproceedings{yu2026laprune,
  title     = {{LAPrune}: Logits-Aligned Scoring Proxy for {KV} Pruning via Vector Quantization},
  author    = {Yu, Mingyang and Tu, Rong-Cheng and Ding, Yifu and Zhao, Hanqing and Jing, Yongcheng and Luo, Xiao and Tao, Dacheng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
