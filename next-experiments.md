# Next Experiments

§1 is *not* cross-scale transfer — it stays inside Pythia-70m and varies training budget and seed.
Cross-scale is §2.

All three are implemented as standalone scripts, each runnable on its own on a remote pod:

| § | Script | Needs a GPU? | Status |
|---|--------|--------------|--------|
| 1 | `budget_transfer.py` | no (40 min on an A100) | **run — see result below** |
| 2 | `cross_scale_transfer.py` | yes, plus training at each target | blocked on target checkpoints |
| 3 | `rotation_null.py` | no, weight-only, ~40s | **done — both nulls and the band control pass** |

Shared machinery lives in `sae_stats.py` (SAE definition, the six statistics, labels, Hub
loader) and `eval_activations.py` (the activation stream, per model). Neither has side effects on
import, so all three scripts compute the same quantities from the same code.

## 1. Cheap-to-expensive transfer

Implementation plan: [plan-cheap-to-expensive.md](plan-cheap-to-expensive.md) ·
Script: `budget_transfer.py`

```bash
python budget_transfer.py validate     # prove the local stream reproduces the sweep first
python budget_transfer.py all --k 64
```

**Question.** Can a classifier fitted on a throwaway SAE be applied to a real one?

Right now the pitch is "measure six numbers on one SAE instead of retraining five." The obvious
objection is that the classifier itself was fitted on the same expensive 1B-token SAE it was
tested on — so where does a new user get one? If a classifier fitted on the 50M-token checkpoint
still works on a 1B-token SAE of a *different seed*, the recipe becomes: train one cheap SAE, fit
once, reuse forever.

**Method.** Fit on seed 42 @ 50M. Score seed 256 @ 1B, with labels re-derived anchored on 256.
Compare against the 0.868 within-dictionary ceiling and the 0.872 same-budget transfer. This is
the first test that crosses seed *and* budget at once.

**Either outcome is reportable.** Holding means the diagnostic is a reusable artifact. Degrading
tells us how much training the source SAE needs, which is itself a number a practitioner wants.

**Result (run on an A100, k = 64, all 18 checkpoints).** The pre-declared "degrading" branch, but
the decomposition is the finding: **the classifier transfers across seeds for free and fails across
budgets.**

| fit → score | AUROC (source scaler) | AUROC (target scaler) | target ceiling |
|---|---|---|---|
| self, 42@1B → 42@1B | 0.869 | 0.869 | 0.868 |
| **seed only**, 42@1B → 256@1B | **0.874** | **0.874** | 0.873 |
| **budget only**, 42@50M → 42@1B | **0.697** | 0.630 | 0.868 |
| both, 42@50M → 256@1B | 0.680 | 0.621 | 0.873 |

Crossing seeds costs nothing (0.874 against a 0.873 ceiling); crossing budgets costs 0.17. Since
the seed axis is the only one that speaks to "would this hold on a fresh run", the core claim is
in better shape than the headline cell suggests — but "train one cheap SAE, fit once, reuse
forever" does not survive.

The cause is visible in the label composition, and it is not a failure of the statistics. Stability
is a *moving target*: at 50M tokens only 15.3% of measurable latents are stable, by 1B it is 76.4%.
A classifier fitted at 50M is calibrated on a different population, not a noisier version of the
same one.

| budget | labelled | stable | live | AUROC | freq-only | delta |
|---|---|---|---|---|---|---|
| 50M | 6,442 | 15.3% | 88.7% | 0.937 | 0.723 | +0.214 |
| 100M | 6,020 | 32.7% | 88.3% | 0.888 | 0.610 | +0.278 |
| 250M | 6,195 | 59.2% | 93.8% | 0.851 | 0.516 | +0.335 |
| 500M | 6,525 | 70.7% | 96.1% | 0.853 | 0.592 | +0.261 |
| 800M | 6,711 | 74.5% | 97.2% | 0.863 | 0.604 | +0.259 |
| 1B | 6,721 | 76.4% | 97.3% | 0.868 | 0.606 | +0.262 |

Read the 0.937 at 50M with care: separating a rare 15% minority is an easier ranking problem, which
is why the frequency baseline peaks there too (0.723). The margin over frequency is widest at 250M
(+0.335), where the classes are closest to balanced. The 1B row reproduces the published 0.868
exactly, which is what validates the locally regenerated token stream.

**Provenance.** `budget_transfer.py validate` reproduces `reappearance_prob` from the exported sweep
CSV to 0.0 and `match_similarity` to ~5e-7, so the local activation stream is the sweep's stream.

**Second part: the budget trajectory.** Run the same analysis at every checkpoint (50M, 100M,
250M, 500M, 800M, 1B) and plot AUROC against budget. Labels are recomputed from the three seeds at
each budget, so every point has its own ground truth. A flat line says 0.868 isn't an artifact of
where we stopped; a rising or collapsing line is a finding about when stability crystallizes.

**Do not oversell it.** Checkpoints along one run are the same SAE at different moments, heavily
autocorrelated. Six budgets x three seeds is not eighteen independent runs, and an error bar over
those points would be pseudo-replication. The seed axis is the only one that speaks to "would this
hold on a fresh run," and it stays at three seeds until we train more.

**Data.** All on the hub, no GPU and no retraining:
`checkpoints/topk64-128-256_x16_tied_auxk/seed{42,256,1024}_k64_tokens{50M,100M,250M,500M,800M,1B}.pt`
— 18 files, 1.8 GB.

**Cost.** Generate the eval activations once and reuse them for every checkpoint. Per-checkpoint
work is then the encode and ablation passes, ~5-10 min each on CPU, so the full `k = 64` grid is
~1.5-3 hours. Cutting the eval set from 1.3M to ~400K tokens is ~3x faster and still well clear of
the firing floor.

**One caution.** Regenerating the stream locally may not reproduce the run's exact tokens, so
absolute AUROCs may drift from the 0.872 in `sweep_summary.json`. Harmless for comparisons within
this grid, since every arm is scored on the same local activations, but the numbers aren't
directly comparable to the hub's.

## 2. Cross-scale transfer

Implementation plan: [plan-cross-scale-transfer.md](plan-cross-scale-transfer.md) ·
Script: `cross_scale_transfer.py`

```bash
python cross_scale_transfer.py --list          # what is on the Hub right now

# each target needs its own 3-seed set first (the remaining GPU cost)
SAE_MODEL=pythia-160m-deduped SAE_K_VALUES=64 SAE_MAX_TOKENS=1000000000 \
    python topk_sweep_experiments.py

python cross_scale_transfer.py --source pythia-70m-deduped \
    --target pythia-160m-deduped --k 64
```

`SAE_MODEL` and `SAE_REL_DEPTH` (default 0.5) now drive the sweep; `d_model`, the hook point and
the dictionary width follow from them and are asserted against the loaded model. Checkpoints for
any model other than the original 70m run are written under a model- and layer-scoped Hub prefix,
so the scales cannot overwrite each other.

**Question.** Does a classifier trained on Pythia-70m work on a *different model* — 160m, 410m,
1B?

This is the rung that decides whether the diagnostic is usable at all. If the classifier has to be
trained on the model you're auditing, you need ground-truth labels for that model, which means
training several SAEs from scratch — exactly the cost the method was meant to avoid. It is only
possible because our predictors are six scalars rather than raw decoder vectors: a classifier over
decoder vectors is tied to the hidden dimension and cannot even be evaluated on a different model.

**What has to be trained.** Labels are needed at every evaluation target, not just at the source,
so each target scale needs its own multi-seed SAE set (3 seeds minimum) at a matched relative
depth so layer position is comparable across models of different depth. This is the main remaining
GPU cost and it bounds how many targets are feasible.

**Report both standardization conventions**, since they answer different questions: reusing the
source scaler tests whether the decision *boundary* transfers; refitting on the target tests
whether the learned *ranking* transfers. Report a classifier retrained on the target as the upper
bound on what transfer could achieve.

**Cross-family** is the same protocol with a target sharing no training data or architecture
lineage. Strongest available claim, and the one that separates "reusable artifact" from "per-model
curiosity."

## 3. Rotation null on geometric isolation

Implementation plan: [plan-rotation-null.md](plan-rotation-null.md) ·
Script: `rotation_null.py`

```bash
python rotation_null.py --k 64 --k 128 --k 256 --rotations 10
python rotation_null.py --k 64 --bands     # frequency-band control, needs budget_transfer stats
```

**Result (run, 20 rotations per arm, seeds 42/256/1024 @ 1B tokens).** The geometry survives both
nulls decisively. Reproduced on two machines with unrelated dependency stacks (laptop CPU at 10
rotations, then a GPU pod at 20), giving identical real AUROCs to three decimals.

| k | real label rate | rotated label rate | real AUROC | rotated AUROC (mean) | rotated range | surplus |
|---|---|---|---|---|---|---|
| 64 | 64.2% | 0.0% | 0.817 | 0.501 | `[0.482, 0.513]` | **+0.315** |
| 128 | 55.7% | 0.0% | 0.809 | 0.502 | `[0.493, 0.515]` | **+0.307** |
| 256 | 3.4% | 0.0% | 0.585 | 0.499 | `[0.472, 0.535]` | +0.086 |

No rotated partner produced a single match at `theta = 0.7`, so the *ground truth* is not
reproducible by dictionary geometry alone. With the labels re-derived from a rotated partner at
matched prevalence, neighbour crowding scores exactly chance — so the predictor is not measuring
sphere density either. Only 0.7% of latents have a near-duplicate inside their own dictionary,
which rules out the redundancy explanation as well. The `k = 256` row is weak because that arm only
labels 3.4% of latents as stable, not because the null behaves differently.

**Frequency-band control (run).** Crowding is not a frequency proxy either. Within activation-
frequency deciles the mean AUROC is **0.814**, against 0.817 unbanded — essentially no loss, and it
holds between 0.79 and 0.87 in nine of ten deciles (only the densest-firing decile drops, to 0.684).
So the geometric signal is not explicable by the frequency baseline it has to beat.

**Naming.** Report this statistic as **neighbour crowding**, not "geometric isolation": it is the
mean cosine to the 10 nearest decoder neighbours, so high means crowded. `rotation_null.py` prints
it that way and never flips a sign, so `>0.5` always means crowded latents are more stable.

**Question.** Is the geometric signal real, or just sphere density?

**Why it's urgent.** Geometric isolation is the strongest single predictor (0.817 raw AUROC at
`k = 64`) *and* it points the opposite way from what the proposal assumed. The statistic is the
mean cosine to the 10 nearest decoder neighbours, so a high value means a **crowded**
neighbourhood, not an isolated one — and crowded features are the ones that survive reseeding. The
name reads backwards relative to the quantity, which needs fixing in the write-up regardless.
Until the null is run we can't rule out that this reflects where the unit sphere happens to be
dense rather than anything about the feature.

**Method.** Rotate the partner dictionary so per-feature correspondence is destroyed while the
directional distribution is preserved, then recompute the labels and re-score. Signal that
survives the rotation is an artifact of density; signal that disappears is genuine correspondence.
Also check whether geometry adds anything *within* narrow frequency bands, which separates it from
being a frequency proxy.

**Ordering.** This belongs *before* the transfer results are interpreted — a predictor that is an
artifact at the source can't be expected to travel.

**Cost.** Weight-only, so no GPU, no activations, minutes. Confirmed: 40s for all three arms.
`_audit_circularity.py` and `matching_audit.py` contained the original rotation nulls but were
pointed at the L1 checkpoint paths; `rotation_null.py` supersedes both, with the Haar sign
correction their `qr` draw was missing.