# Predicting SAE Feature Stability from Single-Run Statistics, with Cross-Scale Validation

## 1 The Core Problem

Sparse Autoencoders (SAEs) are the foundational tool for analyzing AI models, helping researchers discover circuits, steer model behavior, and conduct safety audits. However, most features in a standard SAE do not remain the same if you retrain the model from scratch using a different random starting point (or seed). Instead, they represent just one of many mathematically equivalent ways to map the same data.

Currently, checking if a feature is stable requires training multiple SAEs from scratch, which is too expensive and time-consuming for large, state-of-the-art models. While some new architectural designs can help improve stability, they do not help researchers evaluate the stability of features from existing standard SAEs.

## 2 Our Proposed Solution

Past research shows that unstable features have specific geometric patterns, meaning their instability isn't completely random. We propose building a trained tool (a classifier) that predicts whether a feature is stable using multiple statistics gathered from a **single SAE run**.

To prove this tool actually identifies genuine, universal model features rather than just random noise from the initialization, we will validate it by checking if the features persist as the model gets larger (cross-scale correspondence). If our tool correctly predicts a feature is stable, that feature should still exist in larger versions of the model.

### Key Contributions

- **A practical single-run stability tool:** We will create the first ready-to-use tool that calculates the probability of a feature being stable without needing to retrain the model.  
- **Validation across different model sizes:** We will test if the features we predict to be stable show significantly higher match rates across different model sizes compared to unstable ones.  
- **Testing a major AI theory:** We will test the "Platonic Representation Hypothesis" by seeing if the gap in match rates between stable and unstable features grows consistently as the size difference between models increases.



## 3 Testable Hypotheses

Our experiment is built on two independent hypotheses:

- **Hypothesis 1 (Single-Run Predictability):** A tool trained on four single-run statistics can accurately predict seed-level stability (achieving an accuracy score—or AUROC—of 0.75 or higher), outperforming predictions based on activation frequency alone.  
- **Hypothesis 2 (Cross-Scale Universality):** Features predicted to be stable will have significantly higher match rates across different model sizes (e.g., scaling up from Pythia-70m to 1B) than unstable features. This difference in match rates will grow consistently as the size multiplier between the models increases.



## 4 Methodology Pipeline

We will use the Pythia family of models (70m, 160m, 410m, and 1B), focusing specifically on the middle layers of each model.

- **Phase 1: Generating Ground Truth:** We will train 5 SAEs per model and layer using different random seeds. By comparing them, we will determine each feature's average reappearance probability (a score from 0 to 1). We will group features scoring 0.5 or higher to train our predictive tool.  
- **Phase 2: Extracting Single-Run Statistics:** From just the very first run, we will calculate four statistics over a test set of 100 million tokens: how often the feature activates, its geometric magnitude, how isolated it is from other features, and how much the model's overall performance drops if the feature is removed.

Example


| Feature | Frequency | Decoder Norm | NN Similarity | Recon. Contribution | Stable? (label) |
| ------- | --------- | ------------ | ------------- | ------------------- | --------------- |
| A       | 0.08      | 2.1          | 0.12          | 0.031               | Yes             |
| B       | 0.001     | 0.4          | 0.81          | 0.002               | No              |
| C       | 0.045     | 1.8          | 0.19          | 0.024               | Yes             |
| D       | 0.003     | 0.6          | 0.74          | 0.003               | No              |
| E       | 0.062     | 1.9          | 0.15          | 0.028               | Yes             |
| F       | 0.002     | 0.5          | 0.79          | 0.001               | No              |


- **Activation Frequency:** How often the feature fires across all tokens in the test set. High-frequency features likely represent common, well-defined concepts that the model consistently encodes in a stable direction. Low-frequency features may be marginal or accidental — invented by the SAE to slightly improve reconstruction but not grounded in anything consistent.  
- **Decoder Column Norm:** The length of the feature's decoder vector — essentially how loudly it speaks when it fires. Large norm means the feature strongly influences the model's activations and is doing real work. Small norm suggests the feature is weak and marginal, and the SAE could probably have ignored it or replaced it with something else.  
- **Geometric Isolation:** The average cosine similarity between this feature and its 10 nearest neighbors in the dictionary. A low score means the feature occupies a unique, isolated direction that both seeds are forced to find. A high score means it sits in a crowded region where the SAE had rotational freedom — a strong signal of instability.  
- **Reconstruction Contribution:** How much the SAE's reconstruction error increases when this feature is zeroed out. A high contribution means the SAE genuinely needs this feature to do its job accurately. A low contribution means the feature is redundant — other features can compensate, so different seeds may include it, drop it, or split its role unpredictably.

- **Phase 3: Classifier Training:** We will train our predictive models (starting with standard Logistic Regression) exclusively on the smallest Pythia-70m model. We will then apply this trained tool directly to the larger Pythia models to see how well it transfers without retraining.  
- **Phase 4: Cross-Scale Validation:** We will split features into "predicted-stable" and "predicted-unstable" groups. We will then calculate the match rates across different model sizes using a rotation-corrected matching method to officially test our second hypothesis.



## 5 Baselines & Controls


| Baseline Approach                      | Purpose in Study                                                                                |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Random classifier**                  | Sets the absolute minimum baseline our tool must beat.                                          |
| **Frequency-only logistic regression** | Replicates past research; serves as the main comparison for our first hypothesis.               |
| **Frequency combined with geometry**   | Tests whether geometric statistics actually add predictive value beyond simple frequency.       |
| **Oracle (perfect predictor)**         | Establishes the absolute maximum possible accuracy for the tool.                                |
| **Permutation null (shuffled labels)** | A statistical check to ensure our cross-scale matching results aren't just happening by chance. |
| **Population-level match rate**        | A basic sanity check replicating previous findings on how models share features.                |




## 6 Expected Outcomes & Limitations

**Ideal Results:** The new tool achieves high accuracy (an AUROC of 0.80 or higher), proving it is a highly effective single-run diagnostic. Across different model sizes, predicted-stable features match at 2 to 3 times the rate of unstable features, with the gap growing consistently as the size difference between models increases.

**Acceptable Partial Results:** If the tool works for single runs but fails the cross-model test, it is still a valuable, ready-to-use diagnostic tool. This outcome would also interestingly prove that a feature's stability and its universality across model sizes are completely separate phenomena.

**Key Limitations & Mitigations:**

- **Statistical Confusion:** High-frequency features might naturally be easier to match across sizes, regardless of whether they are truly universal. We will control for this by analyzing match rates within specific frequency brackets.  
- **Matching Failures:** The math used to align features across models can sometimes fail or be unstable. We will test this on a pilot run first and have a fallback mathematical method ready.  
- **Transferring Across Sizes:** Normalizing the statistics helps, but the geometric differences between dictionaries of varying sizes (e.g., 4,096 vs. 16,384 features) might still cause issues. If the tool's accuracy drops too much on the largest model, we will retrain it specifically on that model's seed data.

