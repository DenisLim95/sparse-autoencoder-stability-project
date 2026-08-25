# Implementation Plan: Rotation Null on Geometric Isolation

Implements §3 of [next-experiments.md](next-experiments.md).

**Claim under test.** Geometric isolation predicts stability because of something about the
feature, not because of where the unit sphere happens to be dense.

**Why it is urgent.** Geometric isolation is the strongest single predictor at `k=64` (0.817 raw
AUROC) *and* it points the opposite way from what the proposal assumed. There is a structural
reason to suspect circularity:

- the **predictor** is "how close is my nearest neighbour, within my own dictionary"
- the **label** is "how close is my nearest neighbour, in another seed's dictionary"

Both are nearest-neighbour distances, so a feature in a dense region of the sphere scores high on
both for purely geometric reasons, with no stability involved. Until the null is run we cannot
distinguish the two.

Weight-only: no GPU, no activations, minutes to run. The one part that needs activation statistics
(§5) depends on the stats cache from [plan-cheap-to-expensive.md](plan-cheap-to-expensive.md).

---

## 1. Naming problem to fix regardless of the outcome

`compute_geometric_isolation` returns the **mean cosine similarity to the 10 nearest decoder
neighbours** (`topk_sweep_experiments.py:1200-1222`). A high value therefore means a **crowded**
neighbourhood, not an isolated one. The name says the opposite of the quantity.

The empirical direction is that crowded features are the stable ones — the reverse of the
proposal's prediction, and `analyze_from_hub.py:151-152` still negates the statistic to match the
proposal's expected sign:

```149:152:analyze_from_hub.py
        # Hypothesis 1, partial: can geometry alone separate stable from unstable?
        # Sign is negated because the proposal predicts ISOLATED (low neighbour sim) = stable.
        iso = geometric_isolation(W[SEEDS[0]])
        geom_auroc = auroc(-iso, p_primary >= 0.5)
```

Decide one convention and apply it everywhere: either rename the statistic to what it measures
(neighbour density / crowding) and drop the negation, or keep the name and negate consistently.
Mixed conventions across scripts will eventually produce a published sign error.

---

## 2. Consolidate the checkpoint loader

Four audit scripts each define their own loader against the **old L1** path
`checkpoints/seed{seed}_tokens{tokens}.pt`, with `local_files_only=True`:

| Script | What it does | Needs |
| --- | --- | --- |
| [_audit_circularity.py](_audit_circularity.py) | rotation null on predictor AUROC | repoint |
| [matching_audit.py](matching_audit.py) | θ sweep, one-to-one, rotation null on label rate | repoint, budgets |
| [_audit_similarity.py](_audit_similarity.py) | is cross-seed similarity real correspondence | repoint |
| [_audit_predictors.py](_audit_predictors.py) | weight-only predictors vs p̂ | repoint |

Add one shared loader (in `sae_stats.py`, alongside the extraction described in
[plan-cheap-to-expensive.md](plan-cheap-to-expensive.md)) parameterized by `(seed, k, tokens)`:

```
checkpoints/topk64-128-256_x16_tied_auxk/seed{seed}_k{k}_tokens{tokens}.pt
```

Drop `local_files_only=True` so it downloads rather than silently falling through to
`FileNotFoundError` on a fresh machine. Update budgets from the L1 set (100M / 1B / 8B) to the TopK
set (50M / 100M / 250M / 500M / 800M / 1B).

Each audit gets `k` as a parameter, defaulting to 64. Running the null at all three `k` is cheap and
worth it: if the geometry signal is density, its strength should track the live fraction, which
varies from 97% at `k=64` to 21% at `k=256`.

---

## 3. Two distinct nulls — run both

They answer different questions and the repo already contains one of each.

**Null A — does the label rate survive rotation?** (`matching_audit.py:90-94`) Match the anchor
against a randomly rotated partner dictionary and recompute the fraction of features called stable.
This asks whether the reported stability rate is above what dictionary geometry alone produces.

**Null B — does the predictor's AUROC survive rotation?** (`_audit_circularity.py:47-61`) Recompute
labels from the rotated partner and re-score geometric isolation against them. This is the one that
speaks to circularity: predictive power that survives means the statistic was tracking sphere
density all along.

Report both. A high label rate under Null A would undermine the ground truth itself, which is a
much bigger problem than a confounded predictor.

---

## 4. Two implementation details that must not be "cleaned up"

**Prevalence-matched threshold.** `_audit_circularity.py:61` scores the null against
`np.quantile(rot, 1 - (real >= THETA).mean())` rather than a fixed θ=0.7. This is deliberate: the
rotated dictionary's label rate collapses toward zero, so a fixed threshold would leave the null
with almost no positives and make it trivially unbeatable. Matching prevalence makes the comparison
fair. Keep the quantile, and add a comment saying why, because it reads like an inconsistency.

**Use a Haar-uniform rotation.** `matching_audit.py:35-38` does this correctly:

```35:38:matching_audit.py
def random_rotation(d):
    """Haar-random orthogonal matrix: destroys correspondence, preserves dictionary geometry."""
    Q, R = np.linalg.qr(RNG.standard_normal((d, d)))
    return torch.tensor(Q * np.sign(np.diag(R)), dtype=torch.float32)
```

`_audit_circularity.py:48` omits the sign correction and uses `torch.linalg.qr(randn)[0]` directly,
which is not Haar-uniform. Move `random_rotation` into the shared module and have both call it.

**Average over several rotations.** Both scripts use a single fixed seed. One draw gives one number
with no sense of its spread. Use 10-20 rotations and report the mean and range of the null AUROC,
so "survives the null" is a comparison against a distribution rather than a point.

---

## 5. Frequency-band control

Separate from the rotation null, and the second way geometry could be spurious: geometric isolation
might simply be a proxy for activation frequency, which is the baseline we must beat.

Within narrow activation-frequency bands (deciles of log frequency), recompute the AUROC of
geometric isolation alone. If the signal is a frequency proxy it collapses within bands; if it
survives, geometry adds something frequency does not.

This needs `activation_freq`, so it depends on the per-checkpoint stats cache produced by
[plan-cheap-to-expensive.md](plan-cheap-to-expensive.md) §2 stage 3. The rotation nulls themselves
do not — run them first, they are weight-only.

This is the load-bearing decomposition. Measured at `k=64` / 1B: frequency alone is 0.607,
frequency plus geometry is 0.818, and all six predictors reach 0.868. Geometry therefore supplies
+0.211 of the +0.261 total increment over the baseline, with the remaining four predictors adding
+0.050 between them. If the rotation null takes geometry away, most of the headline claim goes with
it — which is why this runs before anything is interpreted.

---

## 6. How to read the outcomes

| Null A (label rate) | Null B (predictor AUROC) | Reading |
| --- | --- | --- |
| collapses | collapses | Clean. Both ground truth and predictor reflect genuine correspondence. |
| collapses | survives | The labels are real but geometric isolation is measuring sphere density. Report the full model without it, or with a density-corrected version. |
| survives | either | Serious. The stability labels themselves are partly explained by dictionary geometry, which affects every result in the project, not just this predictor. |

Any of the three is publishable. The third would be the most valuable finding and the most work.

---

## 7. Cost and ordering

Weight-only. Loading nine 100 MB checkpoints, an 8,192 x 8,192 similarity matrix (268 MB), and 10-20
rotations — minutes on a laptop, no GPU. The only slow piece is the Hungarian check inside
`matching_audit.py`, which is O(n^3) at 8,192 latents; leave it off for the null runs.

**Run this before interpreting any transfer result** — cross-sparsity, cheap-to-expensive, or
cross-scale. Per [paper-outline.md](paper-outline.md), a predictor that is an artifact at the source
cannot be expected to travel, so the null belongs before the transfer ladder rather than after it.
