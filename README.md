# README
This is an exploration into signals in a model's activation space that can help us predict the stability of features discovered by Sparse Auto-Encoders

## Running the experiments

### 1. Prerequisites
- Python 3.11 (what the checked-in `.venv` uses).
- A Hugging Face account. The trained SAE checkpoints and the exported per-feature tables
  live on the Hub under `deenais/sae-stability-pythia70m`. The analysis scripts download what
  they need automatically; you only need a token (`export HF_TOKEN=...`) if the repo is
  private to you.
- A GPU is optional for the analysis scripts (they run on CPU, using CUDA automatically if
  present) but required to *train* new SAEs (see step 5).

### 2. Set up the environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Nothing in `requirements.txt` is pinned — this is deliberate so pip leaves an
already-installed, CUDA-matched `torch` build untouched on a GPU machine while still
installing it on a machine that has none.

### 3. Check what checkpoints are available on the Hub
```bash
python budget_transfer.py discover                 # 70m checkpoints for the budget sweep
python cross_scale_transfer.py --list              # everything on the Hub, grouped by model
```

### 4. Run the analysis experiments (no training)
All four are standalone scripts that pull weights/tables from the Hub and compute the same
six single-run statistics from the same shared code (`sae_stats.py`, `eval_activations.py`).

- **Cheap-to-expensive transfer + budget trajectory** (CPU, ~1.5–3 h at full eval size;
  always run `validate` first to confirm the local activation stream reproduces the sweep):
```bash
python budget_transfer.py validate                 # reproduce the sweep's exported numbers
python budget_transfer.py all --k 64               # stats, then labels/transfer/trajectory
SAE_EVAL_BATCHES=12 python budget_transfer.py all  # ~3x faster on ~393K eval tokens
```

- **Rotation null on neighbour crowding** (weight-only, ~40 s, no GPU):
```bash
python rotation_null.py --k 64 --k 128 --k 256 --rotations 20
python rotation_null.py --k 64 --bands             # frequency-band control; needs the cache
                                                   # written by budget_transfer.py stats
```

- **Label-permutation null** (analysis-only, reads exported tables; no checkpoints/GPU):
```bash
python permutation_null.py                         # k=64,128,256 at 1B, B=1000 permutations
```

- **Cross-scale transfer** (CPU to score, but each target model needs its own 3-seed SAE set
  trained first — see step 5):
```bash
python cross_scale_transfer.py --source pythia-70m-deduped \
    --target pythia-160m-deduped --k 64
```

### 5. Train new SAEs (GPU required)
`topk_sweep_experiments.py` trains the multi-seed TopK SAEs that produce the ground-truth
labels. It is a Colab/A100 script (not run locally) and writes checkpoints to the Hub under a
model- and layer-scoped prefix. The model, relative depth and target budget are driven by
environment variables:
```bash
SAE_MODEL=pythia-160m-deduped SAE_REL_DEPTH=0.5 SAE_K_VALUES=64 \
    SAE_MAX_TOKENS=1000000000 SAE_HF_REPO=<your-repo> \
    python topk_sweep_experiments.py
```

### Environment variables
Read across the scripts (all have sensible defaults):

| Variable | Default | Meaning |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token, only needed for a private Hub repo |
| `SAE_HF_REPO` | `deenais/sae-stability-pythia70m` | Hub repo holding checkpoints/tables |
| `SAE_CACHE_DIR` | `./cache` | where downloaded activations/weights are cached |
| `SAE_RESULTS_DIR` | per-script `./outputs/...` | where results are written |
| `SAE_EVAL_BATCHES` | `40` | eval batches (40 × 256 × 128 ≈ 1.31M tokens, the sweep size) |
| `SAE_MIN_FIRINGS` | `100` | measurability floor for the conditional statistics |
| `SAE_MODEL` / `SAE_REL_DEPTH` | `pythia-70m-deduped` / `0.5` | model + relative layer depth for training |
| `SAE_K_VALUES` / `SAE_MAX_TOKENS` | sweep defaults | sparsity arms and token budget for training |

## Relevant papers
- Paulo & Belrose (2025) — arXiv:2501.16615: https://arxiv.org/abs/2501.16615
- Gerasimov et al. (2026) — arXiv:2606.12138: https://arxiv.org/abs/2606.12138


## Notes
- Both Paulo and Gerasimov use cosine similarity threshold of 0.7. However, they use slightly different definition/requirement.
- 

## Log

### prelim-experiments-2
