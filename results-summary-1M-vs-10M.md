# SAE Stability: 1M vs 10M Token Run Summary

Consolidated summary of the two preliminary runs of the Hypothesis 1 pipeline
(single-run statistics predicting SAE feature stability), what the comparison
tells us, and what to do next.

> **Status note:** The 1M numbers are reproduced from `preliminary-results-summary.md`.
> The 10M numbers are **recalled from memory and NOT yet reproduced** — the outputs of
> that run were cleared/lost before being saved. Treat every 10M figure below as a
> *hypothesis to re-test*, not an established result. (Persistence to Google Drive has
> since been added to the notebook so this won't recur.)



## Shared Setup

- **Model:** Pythia-70m-deduped, layer 3 (`blocks.3.hook_resid_post`)
- **SAE:** 2048 features (4x expansion), ReLU + L1, decoder normalized
- **Seeds:** 5 (42, 137, 256, 512, 1024)
- **Matching:** Hungarian algorithm on decoder cosine similarity
- **Stability label:** reappearance probability across seeds; grouped at threshold 0.5
- **Predictors:** activation frequency, geometric isolation (avg cosine sim to 10 NN),
reconstruction contribution (MSE increase when ablated), mean activation strength



## Run 1 — 1M Tokens (reproduced)

- **Stable features:** 0 at the 0.9 similarity threshold; ~~10-20 (~~0.5-1%) at 0.5.
- **Stable features are "boring" structural detectors:** document boundaries (newlines
after `<|endoftext|>`), punctuation/formatting; high activation frequency (~70-99%).
- **Classifier (AUROC):**


| Model                | AUROC |
| -------------------- | ----- |
| Random baseline      | 0.500 |
| Frequency only       | 0.773 |
| Frequency + Geometry | 0.976 |
| Full model (all 4)   | 0.992 |


- **Dominant predictor:** geometric isolation (neighbor proximity). Adding geometry to
frequency gave ~+0.20 AUROC.
- **Direction anomaly:** HIGH neighbor similarity (crowded) → stable — the **opposite**
sign to the prior-work hypothesis that isolated/unique directions are the stable ones.



## Run 2 — 10M Tokens (from memory, UNVERIFIED)

- Classifier still predicted stability well (exact AUROCs not recovered).
- **Dominant predictor shifted to activation frequency** (geometry no longer the top
single predictor).
- Presumably more stable features than at 1M (needs confirmation, especially at the
strict 0.9 threshold).



## Update — Evidence from the 1B Checkpoints (`ndasari/SAE_project`)

Two findings recovered from Naveen's uploaded checkpoints at no GPU cost. Both bear
directly on Next Steps items 2 and 4 below.

### 1. The undertraining confound is confirmed, and the transition is between 100M and 1B

Reappearance probability computed by decoder cosine matching (θ=0.7, anchor seed 42) over
the three seeds present at *every* milestone (42, 256, 1024), so the comparisons are
apples-to-apples. Note only 2 comparisons per feature here, so these are not directly
comparable to the 5-seed numbers above.

| tokens | mean p̂ | fully stable | never matched | median best-cosine |
| ------ | ------ | ------------ | ------------- | ------------------ |
| 1M     | 0.000  | 0.0%         | 100.0%        | 0.153              |
| 50M    | 0.093  | 7.4%         | 88.8%         | 0.191              |
| 100M   | 0.180  | 14.8%        | 78.9%         | 0.245              |
| 1B     | 0.731  | 66.8%        | 20.6%         | 0.921              |

At 1M tokens (~30 optimizer steps) the SAEs are barely past initialization, so the 1M row
being 0% is a sanity check, not a result. The curve is **still climbing steeply** between
100M and 1B — median best-match cosine jumps 0.245 → 0.921. Per item 4's own decision rule,
this is the "stability tracks training budget" branch: **the 1M and 10M findings above were
measuring optimization state, not intrinsic structure.** Any predictor ranking derived from
them (including the geometry→frequency flip) should be treated as unreliable until re-derived
at ≥1B.

This also raises a caution for the 5B/8B milestones: if stability keeps rising toward
saturation, the *variance in the target* shrinks, and a classifier's headline AUROC gets
easier while becoming less meaningful. The informative regime for Hypothesis 1 may be the
100M–2B transition rather than the largest budget we can afford.

### 2. Decoder norm is a dead predictor — it carries exactly zero signal

`normalize_decoder()` projects every decoder row to unit norm after every optimizer step, so
decoder norm is the constant 1.0 for every feature. Measured on the 1B checkpoints, the
spread across all 2048 features is float-rounding only (std ≈ 4.6e-08). Naveen's own log
shows the consequence:

```
=== Stable Features (n=253) ===
  Decoder norm:      mean=1.000, std=0.000
=== Unstable Features (n=1578) ===
  Decoder norm:      mean=1.000, std=0.000
```

So the methodology's four single-run statistics are really **three**. This needs a team
decision: either drop decoder norm from the stated method, or replace it with a quantity
that survives normalization (e.g. encoder row norm, or decoder norm captured *before* the
normalization step). Whatever prior work motivated "decoder norm predicts stability" was
presumably not normalizing the decoder, so this is a definitional mismatch of the same kind
as item 8.

Incidentally, this is what broke Naveen's run: a zero-range histogram raises
`ValueError: Too many bins for data range`, which aborted his script *after* training
finished. Training itself was fine. Guarded in `prelim_experiments_update.py`.



## What the Comparison Tells Us

The most informative signal is the **shift in the dominant predictor** (geometry → frequency)
together with the **counterintuitive sign** of the 1M geometry effect.

1. **Geometry was likely a confounded proxy for frequency.** "Crowded features are stable"
  (wrong sign vs theory) is the classic fingerprint of confounding. A plausible mechanism
   is **feature splitting**: important, high-frequency concepts (newlines, punctuation) get
   split into several near-duplicate directions — which are simultaneously high in neighbor
   proximity *and* reliably re-learned by every seed. So at 1M, "neighbor proximity" won
   only because it stood in for "frequent structural feature." As data increased and
   estimates stabilized, the direct driver (frequency) surfaced and the proxy faded.
2. **Undertraining confound.** At 1M the 5 SAEs were very likely not converged. Decoder
  *geometry* is least trustworthy exactly when the SAEs are noisiest — so the 1M geometry
   signal may reflect optimization state, not intrinsic structure.
3. **This threatens the *interesting* version of Hypothesis 1.** The proposal's claim is a
  four-stat tool that **beats frequency-only**. If at 10M frequency alone does most of the
   work, the marginal value of the geometric/reconstruction stats may have collapsed. The
   decisive metric is therefore **Δ = AUROC(full) − AUROC(frequency-only), with a CI** — not
   the headline AUROC. A high absolute AUROC driven by frequency alone would be a much more
   deflationary (though still valid) result: "properly trained, stability ≈ high frequency."
4. **Small positive class caveat.** With only ~10-20 positives, predictor rankings are
  high-variance; the geometry→frequency flip could be partly sampling noise. Rankings need
   bootstrap/CV confidence intervals before being trusted.



## Next Steps (prioritized)

1. **Re-run 10M and record everything.** Persistence to Drive is now in the notebook
  (per-run folder tagged by model/layer/token budget, checkpoints saved immediately,
   `feature_stability.csv`, `matching_info.npz`, `training_losses.json`, `config.json`).
   Recover the actual 10M numbers.
2. **Frequency-stratified geometry test (decisive).** Within narrow frequency bands, does
  geometric isolation still separate stable from unstable? If it vanishes within-band →
   geometry was a frequency proxy. If it survives → genuine independent signal.
3. **Report Δ(full − frequency-only) with bootstrap CIs**, plus corr(frequency, geometry)
  and geometry's partial contribution controlling for frequency. This directly adjudicates
   Hypothesis 1's "beats frequency alone" clause.
4. **Stability-vs-tokens scaling curve.** Run 1M / 3M / 10M / 30M and plot #stable-at-0.9
  (and mean reappearance prob) vs tokens.
  - Saturating/flattening curve → evidence for a genuine, finite "stable core" (paper-worthy).
  - Still-climbing curve → stability is tracking training budget (undertraining artifact).
   Track each predictor's standalone AUROC across budgets; robust predictors should be
   monotonic/stabilizing, artifactual ones erratic.
5. **Separate "more data" from "more optimization."** Log/inspect convergence
  (`training_losses.json`, dead-feature count, L0). A plateau in stable-count is only
   meaningful if the SAEs are demonstrably converged. Ideally train each seed to convergence
   with data fixed to isolate seed-induced instability.
6. **Report the full threshold sweep, not just 0.9.** Show #stable vs similarity threshold
  and the distribution of reappearance probabilities. Bimodal → supports a real
   stable/unstable dichotomy; continuous → reframe as regression on reappearance probability.
7. **Test the feature-splitting mechanism.** For stable features, inspect nearest neighbors —
  near-duplicates (splitting) vs genuinely distinct directions.
8. **Reconcile the sign disagreement with prior work.** Confirm whether it's a confound
  (item 2) or a definitional mismatch (encoder vs decoder isolation, normalization,
   matching method, layer/model/expansion). Align definitions before claiming a contradiction.



## Open Questions

- Is there a finite stable core, or does stability grow without bound as data/compute increase?
- Once frequency is controlled for, do any single-run statistics add real predictive value?
- Are stable features always "boring" structural detectors, or do semantic features stabilize
with more data?
- Does whatever we find at 70m transfer across scale (Hypothesis 2)?

