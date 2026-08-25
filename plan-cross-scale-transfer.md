# Implementation Plan: Cross-Scale and Cross-Family Transfer

Implements §2 of [next-experiments.md](next-experiments.md).

**Claim under test.** A classifier trained on Pythia-70m predicts stability for a *different model*
— 160m, 410m, 1B — and eventually for a model from another family.

**Why it decides the project.** If the classifier has to be trained on the model you are auditing,
you need ground-truth labels for that model, which means training several SAEs from scratch —
exactly the cost the diagnostic was meant to avoid. Transfer is what makes the method economical
rather than circular.

**Why it is possible at all.** Our predictors are six scalars. A classifier over raw decoder
vectors is tied to the hidden dimension and cannot even be *evaluated* on a model with a different
`d_model`. This is the point made in [paper-outline.md](paper-outline.md): finding interpretable
scalar predictors is a precondition for cross-model transfer, not a separate curiosity.

This is the GPU-bound experiment, so most of this document is about what must be fixed before any
run starts.

---

## 1. Blocking bug: hub checkpoint prefix omits the model

`topk_sweep_experiments.py:240-243`:

```240:243:topk_sweep_experiments.py
RUN_NAME = f"{CONFIG['model_name']}_L{_layer}_{_objective_tag}"
OUTPUT_DIR = Path(RESULTS_BASE) / RUN_NAME

HUB_CHECKPOINT_PREFIX = f"checkpoints/{_objective_tag}"
```

`_objective_tag` is built from `k` values, expansion factor, and the tied/auxk/freedec flags only.
The local directory is namespaced by model and layer; the hub prefix is not. Launching a
Pythia-160m run today would write `checkpoints/topk64-128-256_x16_tied_auxk/seed42_k64_tokens*.pt`
— the same paths the 70m run already occupies.

Consequences, in order of how quietly they fail:

- `restore_checkpoints_from_hub` pulls 70m checkpoints into the 160m run's directory. The
  `d_model` check in `check_config_compatibility` catches the mismatch, but only after the fetch,
  and the folder is already mixed.
- Milestone mirroring uploads 160m weights over 70m filenames. Nothing validates this on the way
  up.
- The 70m results become unreproducible, which matters because they are the source model for
  every transfer claim.

**Fix.** Include the model and layer in the hub prefix, so it mirrors `RUN_NAME`:

```python
HUB_CHECKPOINT_PREFIX = f"checkpoints/{CONFIG['model_name']}_L{_layer}_{_objective_tag}"
HUB_RESULTS_PREFIX = f"results/{CONFIG['model_name']}_L{_layer}_{_objective_tag}"
```

Existing 70m artifacts stay where they are; the new scheme applies to new scales. Document the
split path convention in [topk-sweep-1B-results.md](topk-sweep-1B-results.md) §9 so the old and new
layouts are both findable.

---

## 2. Make the model configurable

`topk_sweep_experiments.py:53-56` hardcodes the three model-dependent settings, while every other
knob reads an environment variable:

```53:56:topk_sweep_experiments.py
CONFIG = {
    "model_name": "pythia-70m-deduped",
    "hook_point": "blocks.3.hook_resid_post",  # Middle layer of Pythia-70m (6 layers, so layer 3)
    "d_model": 512,  # Pythia-70m hidden dimension
```

Change to `SAE_MODEL`, with `d_model` and the layer **derived** rather than separately configured:

- `d_model` read from the loaded `HookedTransformer` (`model.cfg.d_model`) and asserted against
  what the checkpoint config records. Two hand-entered numbers that must agree is a bug waiting to
  happen.
- Layer from a relative depth, `SAE_REL_DEPTH` defaulting to 0.5, so position is comparable across
  models of different depth: `layer = round(rel_depth * n_layers)`.

Matched at 0.5, the grid is:

| Model | layers | layer used | `d_model` | `n_features` @16x | k=64 density |
| --- | --- | --- | --- | --- | --- |
| pythia-70m-deduped | 6 | 3 | 512 | 8,192 | 0.78% |
| pythia-160m-deduped | 12 | 6 | 768 | 12,288 | 0.52% |
| pythia-410m-deduped | 24 | 12 | 1,024 | 16,384 | 0.39% |
| pythia-1b-deduped | 16 | 8 | 2,048 | 32,768 | 0.20% |

Layer 3 of 6 is what the 70m run used, so 0.5 reproduces it exactly.

---

## 3. Design decision to settle before launching: fixed `k` or fixed density?

The table above shows the problem. Holding `k=64` fixed means density falls 4x from 70m to 1B, so
the target SAEs are far sparser than the source. Holding density fixed means `k=256` at 1B, which
at 70m was the degenerate collapse arm.

Neither is obviously right, and the choice changes what a transfer failure would mean — a drop
under fixed `k` could be a scale effect or a sparsity effect, and those are not separable after the
fact. Recommendation: **fixed `k=64`**, because it is the arm the source classifier was fitted on
and because the `k=256` collapse at 70m makes fixed-density the riskier bet; then report the
density shift explicitly as a confound rather than pretending it is absent. If budget allows, one
target scale trained at both `k=64` and matched density separates the two.

---

## 4. Ground truth is required at every target

The binding constraint. A transfer *evaluation* needs labels at the target, so each target scale
needs its own multi-seed SAE set:

- 3 seeds minimum (p̂ over two comparisons, values 0 / 0.5 / 1, matching the source convention)
- same budget as the source evaluation (1B tokens) so a difference is not budget
- same label rule throughout: θ=0.7, ε=0.05, anchor on the first seed, firing floor 100

Per target scale that is 3 SAE trainings, and the SAE work scales with `d_model x n_features`:
2.25x at 160m, 4x at 410m, 16x at 1B relative to 70m, on top of a larger forward pass for the
activations. This is what bounds how many targets are affordable — plan for 160m and 410m, and
treat 1B as a stretch.

**Cost controls.**

- `SAE_HUNGARIAN=0` at the larger scales. The one-to-one check is O(n^3) and 32,768 latents makes
  it the dominant cost by a wide margin. The argmax-vs-Hungarian gap was 0.868 vs 0.851 at 70m, so
  it is a robustness check, not a headline, and can be run once at the smallest target.
- Watch host RAM for the eval activations: 1.31M tokens x `d_model` x 4 bytes is 2.7 GB at 70m but
  10.7 GB at 1B. Reduce `SAE_EVAL_BATCHES` at the large scales, keeping the firing floor
  diagnostic in view.

---

## 5. Predictor comparability

Two of the six predictors are not scale-free in raw units, which is exactly why the standardization
convention has to be reported rather than chosen:

- **Encoder norm** is the norm of a `d_model`-length column, so its scale grows with the model.
- **Mean activation** inherits the residual-stream scale of the host model.
- **Reconstruction contribution** is already divided by `d_model` in the closed form
  (`topk_sweep_experiments.py:1265`), so it is per-dimension and travels better.
- **Activation frequency** and **geometric isolation** are dimension-free by construction, though
  isolation is sensitive to the `n_features / d_model` ratio, which the 16x expansion holds fixed.

**Report both conventions**, as the held-out-seed test already does at
`topk_sweep_experiments.py:1725-1732`:

- **source scaler reused** — does the decision *boundary* transfer? Requires the raw scales to
  agree across models, which the above suggests they will not.
- **scaler refit on the target** — does the learned *ranking* transfer? This is what a practitioner
  with an unlabelled SAE would do, and it is the convention the practical claim rests on.

A large gap between the two is itself the finding: it means the diagnostic transfers as a ranking
but needs per-model recalibration before any absolute threshold is applied.

---

## 6. Baselines that must accompany every transfer number

- **Retrained on target.** A classifier fitted on the target's own labels, cross-validated. This
  upper-bounds what transfer could possibly achieve; transfer within a few points of it is a strong
  result even if the absolute number is lower than at 70m.
- **Frequency-only at target.** The floor. Transfer that fails to beat the target's own
  frequency-only baseline is not useful regardless of its absolute AUROC.
- **Target class balance.** Report it. Absolute AUROC is not comparable across dictionaries of
  differing health, which is why the primary metric is the increment over frequency-only.

---

## 7. Cross-family

Identical protocol with a target sharing no training data or architecture lineage — the strongest
available claim, and per [paper-outline.md](paper-outline.md) the paper's spine. Same requirement
of 3 seeds and ground-truth labels at the target, so it costs the same as one more scale.

Selection criteria: different tokenizer and different pretraining corpus, at a size comparable to
one of the Pythia targets so scale is not confounded with family. Note that a different tokenizer
changes the eval token stream, which means the source and target are no longer measured on the same
text — report this as a difference rather than trying to eliminate it.

---

## 8. Order of work

1. Fix the hub prefix (§1). Blocking — do not launch anything until this is in.
2. Make model, layer, and `d_model` configurable and derived (§2).
3. Settle fixed-`k` versus fixed-density (§3) and write the decision down before training.
4. Train 3 seeds at 160m, 1B tokens, `k=64`. Smallest target, cheapest failure.
5. Score both standardization conventions against both baselines (§5, §6).
6. Only if 160m transfers, proceed to 410m, then cross-family.

The rotation null ([plan-rotation-null.md](plan-rotation-null.md)) should run before any of this is
interpreted: a predictor that is an artifact at the source cannot be expected to travel.
