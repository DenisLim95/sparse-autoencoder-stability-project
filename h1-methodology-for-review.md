# Methodology Review Request: Predicting SAE Feature Stability from Single-Run Statistics

We are testing whether a feature's *stability across random seeds* can be predicted from
statistics computable from a **single** SAE training run. Please review the methodology below
for validity problems, confounds, and circularity. We have listed the weaknesses we already
know about; we are more interested in the ones we have missed.

**Status: the classifier has not been run yet.** Every number below is a measured property of
the trained SAEs or of the labeling procedure; no AUROC exists so far. We are seeking review
before producing the result rather than after, so critiques that would change the design are
actionable.

## 1. The claim under test

**Hypothesis 1:** A classifier trained on four single-run statistics can predict whether an SAE
feature reappears when the SAE is retrained with a different random seed, achieving AUROC ≥ 0.75
and outperforming a baseline using activation frequency alone.

The motivation is cost. Establishing stability normally requires training several SAEs from
scratch. If single-run statistics predict it, practitioners could assess stability of an
existing SAE without retraining.

Both conditions matter. Hitting 0.75 while failing to beat frequency alone would mean the
three additional statistics contribute nothing, which would negate the contribution even
though the headline number looks acceptable.

## 2. Experimental setup

**Base model.** Pythia-70m-deduped, residual stream after layer 3 (`blocks.3.hook_resid_post`);
the model has 6 layers, so this is the middle.

**SAE.** ReLU encoder with L1 sparsity penalty. Input dim 512, dictionary size 2048 (4x
expansion). `l1_coeff = 1.0`, learning rate 1e-3, Adam, batches of 256 sequences × 128 tokens.
Decoder rows are projected to unit norm after every optimizer step, and the gradient component
parallel to each decoder row is projected out before stepping.

**Data.** `monology/pile-uncopyrighted`, streamed.

**Seeds and budgets.** Seeds 42, 256, 1024 trained to 1M, 50M, 100M, 1B, 2B, 3B, 5B, 8B tokens.
(Two further seeds, 137 and 512, exist only up to 100M.) All seeds consume an identical
activation stream, so cross-seed differences are attributable to initialization.

**Observed training state at 8B.** L0 ≈ 346 active features per token out of 2048 (~17%
density); 0% dead features; reconstruction and sparsity losses flat over the final billions of
tokens.

## 3. Step 1 — Constructing the labels

This is the only step that uses more than one SAE.

Represent each feature by its decoder vector, L2-normalized. Fix seed 42 as the anchor. For each
anchor feature, compute cosine similarity against all 2048 features of a comparison SAE and take
the maximum (many-to-one: two anchor features may claim the same partner). Count it as a match
if that maximum is ≥ θ = 0.7.

With two comparison seeds (256, 1024), the reappearance probability p̂ takes values 0, 0.5, or 1.

Endpoint binarization: **stable** if p̂ ≥ 0.95, **unstable** if p̂ ≤ 0.05, otherwise **discarded**.

Result at 8B tokens:

| p̂ | label | count | share |
| --- | --- | --- | --- |
| 1.0 | stable | 1526 | 74.5% |
| 0.5 | discarded | 229 | 11.2% |
| 0.0 | unstable | 293 | 14.3% |

This follows Gerasimov et al. 2026 (arXiv:2606.12138), which uses decoder-only argmax matching
with θ = 0.7 and ε = 0.05.

## 4. Step 2 — The four single-run statistics

Computed from **seed 42's SAE alone**. None of them reference the comparison seeds; this is the
constraint that makes the resulting tool usable without retraining.

1. **Activation frequency** — fraction of evaluation tokens on which the feature is non-zero.
2. **Mean activation** — average activation magnitude conditional on firing.
3. **Geometric isolation** — mean cosine similarity to the 10 nearest other features in the same
   decoder. High means it sits in a crowded region.
4. **Reconstruction contribution** — increase in per-token reconstruction MSE when the feature is
   zeroed after encoding, measured over 10,000 sampled activations.

The original proposal specified *decoder norm* rather than mean activation. Decoder norm is
identically 1.0 for every feature because of the unit-norm constraint, so it carries no
information and was replaced.

## 5. Step 3 — Classifier and evaluation

Features: the 1819 labeled ones (the 229 discarded are excluded entirely).

Standardize columns, then logistic regression with balanced class weights. Evaluation is 5-fold
stratified cross-validation: in each fold the model is fit on ~1455 features and scores the
~364 held out, having seen only their four statistics. AUROC is averaged across folds.

**Baselines, all under the identical protocol:** random; frequency only; frequency + geometric
isolation; all four; oracle.

## 6. Robustness checks already performed

Measured on our own checkpoints, decoder weights only:

- **Matching threshold sweep** at 8B: θ=0.5 → 85.9% stable, θ=0.7 → 74.5%, θ=0.9 → 56.2%.
- **Argmax vs. Hungarian one-to-one matching**: 74.5% vs. 72.8% (1.7 points) at every budget,
  consistent with Gerasimov's reported IoU of 0.978 between the two rules.
- **Rotation null**: matching seed 42 against a randomly rotated copy of a partner dictionary
  gives 0.00% stable, median best-cosine 0.150. The measured correspondence is not a
  dimensionality artifact.
- **Within-dictionary redundancy**: only 0.9% of features have any neighbour above cosine 0.7;
  median nearest-neighbour cosine is 0.273. The dictionary is not geometrically crowded despite
  its high L0.
- **Stricter definition** (Paulo & Belrose: same partner under both encoder and decoder, both
  ≥ 0.7): 67.6% vs. our 74.5%; encoder and decoder agree on the partner for 93.1% of features.

## 7. Weaknesses we are aware of

1. **Only two comparisons per feature.** p̂ has three possible values. ε = 0.05 is therefore
   vacuous — it selects exactly the same sets as ε = 0. Gerasimov used 95 comparisons, where the
   tolerance is meaningful, and report classifier quality improving with the number of seeds.
2. **Class imbalance** of roughly 84/16 among labeled features; only 293 unstable examples.
3. **No demonstration of transfer.** Cross-validation holds out features from the *same*
   dictionary. Geometric isolation is a relational statistic — a feature's value depends on
   neighbours that may sit in the training fold — so folds are not fully independent.
4. **Discarding the ambiguous 11.2%** removes the hardest cases before evaluation, which
   plausibly inflates AUROC relative to labeling every feature.
5. **Density.** L0 ≈ 346 of 2048 is far denser than published SAEs (Gerasimov: TopK = 64 of
   16384, ~0.4%). It is unclear how far conclusions drawn here transfer to sparser dictionaries.
6. **Evaluation set size.** Statistics are currently computed over ~1.3M tokens; the design
   called for 100M. Frequency estimates for rare features are correspondingly noisy, and
   frequency is the baseline we must beat.
7. **Token counts past 1B measure optimization, not unique data.** The activation stream was
   restarted when training resumed from the 1B checkpoints, so 1B→8B replays earlier tokens. All
   seeds replay identically, so cross-seed comparison remains valid.
8. **Seed count is not constant across budgets** — five seeds at ≤100M, three at ≥1B. "Stable"
   is a conjunction over comparisons, so it becomes strictly harder as seeds are added; the two
   halves of any scaling curve are not on the same axis.
9. **Saturation.** Stability is flattening with budget (66.8% at 1B → 74.5% at 8B) and the class
   imbalance worsens as it does, so the largest budget may be the least informative place to
   evaluate a classifier.
10. **Reconstruction contribution via single-feature ablation** may systematically understate
    importance in a dictionary this dense, since redundant features can compensate for each
    other.
11. **Single setting**: one model, one layer, one dictionary width, one SAE architecture.

## 8. What we would most like challenged

- Is there **circularity** between geometric isolation and the label? Both derive from decoder
  cosine geometry. Our redundancy and Hungarian checks suggest crowding is not driving the
  labels, but we may be testing the wrong thing.
- Does **discarding the ambiguous middle** bias the evaluation, and if so, in which direction?
- With only two comparisons, is the **label reliable enough** to support an AUROC claim at all,
  or does the coarseness of p̂ place a ceiling on achievable performance that we would
  misattribute to the predictors?
- Is **cross-validation within one dictionary** sufficient evidence for a tool intended for use
  on SAEs the user did not train? If not, what is the minimum acceptable held-out design?
- Does the **high L0** undermine the interpretation of any of the four statistics, particularly
  geometric isolation and reconstruction contribution?
- Are the four statistics **individually confounded by activation frequency**? If frequency
  drives the others, the multivariate model may add nothing.
