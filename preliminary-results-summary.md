# SAE Stability Prediction: Preliminary Experiment Summary

## Objective
Test whether single-run statistics can predict SAE feature stability across random seeds, without expensive multi-seed retraining.

## Setup
- **Model:** Pythia-70m-deduped, layer 3 (middle layer)
- **SAE:** 2048 features (4x expansion), standard architecture with ReLU + L1 sparsity
- **Training:** 1M tokens, 10 epochs, 5 different random seeds
- **Matching:** Hungarian algorithm on decoder cosine similarity

## Key Findings

### 1. Most SAE features are unstable
- At 0.9 similarity threshold: **0 stable features**
- At 0.5 similarity threshold: **~10-20 stable features** (~0.5-1%)
- Confirms the core problem: SAE features don't reliably reappear across seeds

### 2. Stable features are "boring" structural detectors
- Document boundaries (newlines after `<|endoftext|>`)
- Punctuation (periods, formatting)
- High activation frequency (fire 70-99% of the time)
- These are fundamental patterns any SAE must learn

### 3. Four statistics CAN predict stability

| Statistic | What it measures | Predictive? |
|-----------|------------------|-------------|
| Activation Frequency | How often feature fires | ✓ Yes |
| Geometric Isolation | Avg similarity to 10 nearest neighbors | ✓ **Strongest** |
| Reconstruction Contribution | MSE increase when ablated | ✓ Yes |
| Mean Activation Strength | How strongly it fires | ✓ Yes |

### 4. Classifier Results (Hypothesis 1)

| Model | AUROC |
|-------|-------|
| Random baseline | 0.500 |
| Frequency only | 0.773 |
| Frequency + Geometry | 0.976 |
| **Full model (all 4)** | **0.992** |

**Hypothesis 1 SUPPORTED:** AUROC = 0.992 >> 0.75 target

### 5. Geometric isolation is the key predictor
- Adding geometry to frequency: +0.20 AUROC improvement
- Alone, it may be the strongest single predictor
- Contradicts initial hypothesis direction (HIGH neighbor similarity → stable, not LOW)

## Limitations
- Small sample size (only ~10-20 stable features)
- Limited training (1M tokens vs 100M in proposal)
- Results may shift with more data

## Next Steps
1. **Scale to 10M tokens** (in progress) - expect more stable features at 0.9 threshold
2. **Scale to 100M tokens** - full experiment per proposal
3. **Cross-scale validation** (Hypothesis 2) - test on Pythia-160m, 410m, 1B

## Key Takeaway
> Single-run geometric statistics, especially nearest-neighbor similarity, strongly predict feature stability. This validates the core research direction and justifies scaling up.
