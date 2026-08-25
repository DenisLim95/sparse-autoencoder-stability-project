# Implementation Plan: Cheap-to-Expensive Transfer + Budget Trajectory

Implements §1 of [next-experiments.md](next-experiments.md).

**Claim under test.** A classifier fitted on a 50M-token SAE still predicts stability for a
1B-token SAE of a different seed. If it holds, the recipe becomes "train one throwaway SAE, fit
once, reuse" instead of "fit on the expensive model you were trying to avoid labelling."

**Secondary result.** AUROC as a function of training budget, which says whether 0.868 is an
artifact of where we stopped.

No GPU, no retraining. Everything runs on a laptop from checkpoints already on the hub.

**Status: implemented.** `sae_stats.py` (§1) and `budget_transfer.py` (§2) exist. Run
`python budget_transfer.py validate` first, then `stats`, then `analyze`.

---

## 0. Scope

All three `k` arms: 6 budgets x 3 seeds x 3 `k` = 54 checkpoints, 100.8 MB each, ~5.4 GB total.

Budgets available on the hub: 50M, 100M, 250M, 500M, 800M, 1B. (The 1M milestone exists for the
old L1 run but not for the TopK sweep.)

Path pattern:

```
checkpoints/topk64-128-256_x16_tied_auxk/seed{42,256,1024}_k{64,128,256}_tokens{50000000,...,1000000000}.pt
```

in the public repo `deenais/sae-stability-pythia70m`. Each file is a dict with `model_state_dict`,
`optimizer_state_dict`, `tokens_seen`, `step`, `seed`, `config`.

---

## 1. Prerequisite: extract `sae_stats.py`

[topk_sweep_experiments.py](topk_sweep_experiments.py) cannot be imported — it executes
top-to-bottom, loading Pythia at line 290 and opening a Pile stream at line 1358, then trains. So
the statistic functions have to be moved into an importable module and imported back, rather than
copied. Copying would let the local pipeline and the run drift apart silently, which would make any
disagreement between local numbers and hub numbers uninterpretable.

**Move into `sae_stats.py`, unchanged:**

| What | Current location |
| --- | --- |
| `TopKSparseAutoencoder` | lines 335-~470 |
| `compute_decoder_similarity`, `compute_reappearance_probability` | lines 1111-1155 |
| `compute_activation_stats` | lines 1168-1197 |
| `compute_geometric_isolation` | lines 1200-1222 |
| `compute_reconstruction_contribution` | lines 1225-~1280 |
| `compute_encoder_stats` | lines 1283-1292 |
| `compute_single_run_statistics` | lines 1295-1323 |
| `build_predictors` | lines 1326-1349 |
| `cv_auroc` | lines 1401-~1420 |
| `THETA=0.7`, `EPSILON=0.05`, `MIN_FIRINGS`, `STAT_BATCH`, `ABLATION_BATCH`, `LOG_EPS` | lines 1107-1165 |

`TopKSparseAutoencoder` has to come along: every statistic function takes a live SAE object and
uses `.encode`, `.decode`, `.W_dec`, `.W_enc`, `.b_enc`, `.n_features`, `.d_model`.

**Constraints on the move.**

- Behaviour-preserving. No renames, no signature changes, no "while I'm here" cleanups.
- `MIN_FIRINGS` and `ABLATION_BATCH` read environment variables at import time; keep that.
- `topk_sweep_experiments.py` then does `from sae_stats import *` (or explicit imports) at the top
  and deletes the moved bodies. Its behaviour must be identical.
- Verify by re-running the analysis-only path on one arm and diffing against
  `results/topk64-128-256_x16_tied_auxk/at1B/sweep_summary.json`, or at minimum by confirming the
  module imports and `python -c "import sae_stats"` has no side effects (no model load, no
  network, no stdout).

---

## 2. `budget_transfer.py`

Modelled on [analyze_from_hub.py](analyze_from_hub.py), which already does hub discovery,
weight-only labelling, and a dependency-light AUROC — but is pointed at the old L1 path
`checkpoints/seed{s}_tokens{t}.pt` and computes only geometric isolation.

### Stage 1 — discovery

Regex over `HfApi().list_repo_files()` for the TopK path. Build `{(k, seed, tokens) -> filename}`.
Skip any `(k, tokens)` cell missing a seed, printing what was skipped, exactly as
`analyze_from_hub.py` does for incomplete token counts. Do not silently label from two seeds where
three were expected.

### Stage 2 — eval activation cache

Generate once, reuse for all 54 checkpoints. Reproduce the run's eval set exactly:

- `monology/pile-uncopyrighted`, streaming, from the beginning
- `seq_len=128`, `batch_size=256`, `N_EVAL_BATCHES=40` -> 1,310,720 tokens
- hook `blocks.3.hook_resid_post` on `pythia-70m-deduped`, `stop_at_layer=4`

Cache to `cache/eval_activations_{n_tokens}.pt` (~2.7 GB fp32). Regenerating costs one CPU pass
over Pythia-70m; every later stage reads the cache.

`SAE_EVAL_BATCHES=12` cuts this to ~393K tokens for a roughly 3x faster grid. Safe: at `k=64` the
average feature fires on `k/n_features` of tokens, so ~3,100 firings at 393K, comfortably above
`MIN_FIRINGS=100`. At `k=256` the live fraction is only 21%, so check the floor diagnostic before
trusting a reduced-token run on that arm.

### Stage 3 — per-checkpoint statistics cache

For each of the 54 checkpoints: download, instantiate `TopKSparseAutoencoder(512, 8192, seed, k)`,
`load_state_dict(ckpt["model_state_dict"])`, call `compute_single_run_statistics`, write to
`cache/stats_k{k}_seed{s}_t{tokens}.npz`, free the SAE.

Cache per checkpoint, not per grid, so an interruption resumes instead of restarting. This is the
expensive stage: the encode pass and the closed-form ablation dominate, ~5-10 min per checkpoint on
CPU.

### Stage 4 — labels, re-derived per scored dictionary

This is the step that is easy to get wrong. For every `(k, budget, anchor_seed)`:

- `compute_reappearance_probability({anchor: sae_anchor, **others})` with the anchor first, since
  the function anchors on `seeds[0]`
- `stable = p >= 1 - EPSILON`, `unstable = p <= EPSILON`, middle discarded
- firing floor re-derived from **that** dictionary's `firing_counts`, not the source's — which
  latents are under-measured is a property of the SAE being scored

With three seeds, p̂ takes only the values 0, 0.5, 1.

### Stage 5 — transfer matrix

For each `k`, for each ordered pair of `(seed, budget)` cells, fit on the source and score the
target. Report both standardization conventions, matching what the sweep already does at
`topk_sweep_experiments.py:1725-1732`:

- **source scaler reused** — tests whether the decision *boundary* transfers, which also requires
  the raw scales to agree
- **scaler refit on the target** — tests whether the learned *ranking* transfers, and is what a
  practitioner with an SAE in hand would actually do

Always alongside two reference points: the target's own cross-validated AUROC (the ceiling) and the
target's frequency-only baseline (the floor).

### Outputs

`budget_transfer_matrix.csv` (one row per source/target/convention), `budget_trajectory.csv`,
`budget_trajectory.png`, and a `config.json` recording eval token count, dataset revision if
available, `MIN_FIRINGS`, `THETA`, `EPSILON`.

---

## 3. Validation gate — run this before anything else

Recompute statistics for seed 42 / `k=64` / 1B locally and diff column-by-column against
`results/topk64-128-256_x16_tied_auxk/at1B/feature_stability_k64.csv`.

The eval stream is deterministic — `activation_stream_generator` re-reads the dataset from the
beginning with fixed `seq_len` and `batch_size`, and the run used the same values — so this should
reproduce to floating-point noise unless the dataset snapshot on the hub has moved.

**If it matches**, local numbers are directly comparable to the hub's, including the 0.868 ceiling
and the 0.872 same-budget transfer.

**If it does not match**, do not debug indefinitely. Record the discrepancy, and restrict every
claim to comparisons *within* the local grid, where all arms are scored on the same locally
generated activations. Cross-referencing a local AUROC against 0.872 would then be invalid. State
this in the write-up rather than quietly mixing the two sources.

---

## 4. Cost

| Stage | Cost |
| --- | --- |
| Download 54 checkpoints | ~5.4 GB, once |
| Eval activation pass | one CPU pass over Pythia-70m, ~30-45 min |
| Statistics, 54 checkpoints | ~5-10 min each, ~4.5-9 hrs total |
| Fitting and scoring | seconds |

Reduce with `SAE_EVAL_BATCHES=12` (~3x faster) or by running `k=64` first and the other arms
overnight. The stats cache makes the grid resumable, so it can be run in pieces.

---

## 5. What to report

**Headline.** Fit seed 42 @ 50M, score seed 256 @ 1B. Compare against the 0.868 within-dictionary
ceiling and the 0.872 same-budget cross-seed transfer. This is the first test that crosses seed and
budget simultaneously.

**Trajectory.** AUROC against budget, labels recomputed per budget. Flat means 0.868 is not an
artifact of the stopping point; rising or collapsing is a finding about when stability crystallizes.

**Decide the reading before running.** 6 budgets x 3 anchors x 3 `k` x 2 conventions is over a
hundred numbers, and some will look good by chance. The headline cell and the trajectory shape are
the result; everything else is context, not a menu to select from.

---

## 6. Caveats to carry into the write-up

**Not independent replications.** Checkpoints along one run are the same SAE at different moments,
sharing initialization and data order. Six budgets x three seeds is not eighteen runs, and an error
bar computed over those points would be pseudo-replication. The seed axis is the only one that
speaks to a fresh draw, and it stays at three until more seeds are trained.

**Same-seed arms are bit-identical at init.** Initialization depends only on the seed and the
tensor shapes, and shapes do not vary with `k`
(`topk_sweep_experiments.py:351-354`), so any comparison holding the seed fixed also holds init and
data order fixed.

**`k=256` is a different regime.** Only ~21% of its dictionary is alive and 83% of labelled latents
are unstable, so "stable" there largely means "fires at all." Expect its trajectory to behave
differently, and do not average it together with the other arms.
