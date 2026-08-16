# -*- coding: utf-8 -*-
"""Cross-scale (and cross-family) transfer of the stability classifier.

Implements the scoring half of plan-cross-scale-transfer.md.

The claim: a classifier trained on Pythia-70m predicts stability for a DIFFERENT model. This
is what decides whether the diagnostic is usable. If the classifier has to be trained on the
model you are auditing, you need ground-truth labels for that model, which means training
several SAEs from scratch -- exactly the cost the method was meant to avoid.

It is only possible because the predictors are six scalars. A classifier over raw decoder
vectors is tied to the hidden dimension and cannot even be evaluated on a model with a
different d_model.

WHAT THIS SCRIPT DOES NOT DO: train. Ground-truth labels are needed at every target, so each
target model needs its own multi-seed SAE set first:

    SAE_MODEL=pythia-160m-deduped SAE_K_VALUES=64 SAE_MAX_TOKENS=1000000000 \
        SAE_HF_REPO=<repo> python topk_sweep_experiments.py

Once those checkpoints are on the Hub this script finds them, computes the six statistics at
each model (each on its OWN activation stream, since a model can only be measured on its own
residual stream), and scores every source -> target pair.

Reported for every pair, because they answer different questions:
  - source scaler reused : does the decision BOUNDARY transfer? Requires the raw predictor
                           scales to agree across models, which encoder norm and mean
                           activation give no reason to expect.
  - scaler refit on target: does the learned RANKING transfer? This is what a practitioner
                           with an unlabelled SAE would do, and what the practical claim needs.
  - retrained on target  : cross-validated on the target's own labels. The upper bound on what
                           transfer could achieve.
  - frequency-only       : the target's own baseline. Transfer that cannot beat this is
                           useless no matter how high the absolute AUROC looks.

Usage:
    python cross_scale_transfer.py --source pythia-70m-deduped \
                                   --target pythia-160m-deduped --k 64
    python cross_scale_transfer.py --list          # what is on the Hub right now
    python cross_scale_transfer.py --source pythia-70m-deduped \
        --target pythia-160m-deduped --target pythia-410m-deduped --k 64 --stats-only
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from eval_activations import (
    N_EVAL_BATCHES,
    PYTHIA_GEOMETRY,
    build_eval_activations,
    hook_point_for,
    hub_prefix_for,
)
from sae_stats import (
    DEFAULT_PREFIX,
    DEFAULT_REPO,
    EPSILON,
    MIN_FIRINGS,
    THETA,
    build_predictors,
    complete_cells,
    compute_reappearance_probability,
    compute_single_run_statistics,
    cv_auroc,
    discover,
    load_sae,
)

OBJECTIVE_TAG = os.environ.get("SAE_OBJECTIVE_TAG") or DEFAULT_PREFIX
REL_DEPTH = float(os.environ.get("SAE_REL_DEPTH") or 0.5)
CACHE_DIR = Path(os.environ.get("SAE_CACHE_DIR") or "cache")
RESULTS_DIR = Path(os.environ.get("SAE_RESULTS_DIR") or "outputs/cross_scale")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STAT_KEYS = ("activation_freq", "mean_activation", "firing_counts", "geometric_isolation",
             "recon_contribution", "recon_contribution_uncond", "encoder_norm", "encoder_bias",
             "decoder_norm")


def stats_path(model, k, seed, tokens):
    return (CACHE_DIR /
            f"stats_{model}_k{k}_seed{seed}_t{tokens}_e{N_EVAL_BATCHES}.npz")


class ModelGrid:
    """Everything needed to score one model: its checkpoints, its hook, its statistics."""

    def __init__(self, name, repo):
        if name not in PYTHIA_GEOMETRY:
            raise SystemExit(
                f"{name!r} is not in PYTHIA_GEOMETRY (eval_activations.py). Add its "
                f"(d_model, n_layers) before training or scoring it."
            )
        self.name = name
        self.repo = repo
        self.d_model, self.n_layers = PYTHIA_GEOMETRY[name]
        self.hook_point = hook_point_for(name, REL_DEPTH)
        self.prefix = hub_prefix_for(name, OBJECTIVE_TAG, REL_DEPTH)
        self.found = discover(repo, self.prefix)
        self.cells, self.seeds, self.skipped = complete_cells(self.found)
        self._cells = {}

    def describe(self):
        budgets = sorted({t for _, t in self.cells})
        return (f"{self.name:<22} layer {self.hook_point.split('.')[1]:>2}/{self.n_layers:<2} "
                f"d_model {self.d_model:>4}  {len(self.found):>3} ckpts, "
                f"seeds {self.seeds}, budgets {[f'{t / 1e6:.0f}M' for t in budgets]}")

    def activations(self):
        return build_eval_activations(self.name, self.hook_point, DEVICE, cache_dir=CACHE_DIR)

    def ensure_stats(self, k, tokens):
        """Statistics for every seed of one (k, budget) cell, cached per checkpoint."""
        todo = [s for s in self.seeds if not stats_path(self.name, k, s, tokens).exists()]
        if not todo:
            return
        acts = self.activations()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for seed in todo:
            print(f"  {self.name} k={k} seed={seed} @ {tokens:,}")
            sae = load_sae(self.found, k, seed, tokens, device=DEVICE, repo=self.repo)
            stats = compute_single_run_statistics(
                sae, acts, DEVICE, label=f"{self.name} k={k} seed {seed}"
            )
            np.savez(stats_path(self.name, k, seed, tokens),
                     **{key: stats[key] for key in STAT_KEYS})
            del sae
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    def labels(self, k, tokens, anchor):
        """p_hat for one dictionary, weight-only. Anchor first: compute_reappearance_probability
        takes saes[seeds[0]] as the anchor."""
        ordered = [anchor] + [s for s in self.seeds if s != anchor]
        saes = {s: load_sae(self.found, k, s, tokens, device=DEVICE, repo=self.repo)
                for s in ordered}
        probs, _ = compute_reappearance_probability(saes, theta=THETA)
        del saes
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return probs

    def cell(self, k, tokens, anchor):
        """(X, stable, labelled mask) for one dictionary of this model.

        Memoized because every source seed is scored against every target seed, and relabelling
        means reloading all the seeds and taking an n_features-squared cosine each time.

        The firing floor is re-derived from THIS dictionary's counts: which latents are
        under-measured is a property of the SAE being scored, not of the one trained on.
        """
        key = (k, tokens, anchor)
        if key in self._cells:
            return self._cells[key]
        with np.load(stats_path(self.name, k, anchor, tokens)) as z:
            stats = {name: z[name] for name in z.files}
        probs = self.labels(k, tokens, anchor)
        X = np.column_stack([v for _, v in build_predictors(stats)])
        stable = probs >= (1 - EPSILON)
        mask = ((stable | (probs <= EPSILON))
                & (stats["firing_counts"] >= MIN_FIRINGS)
                & np.isfinite(X).all(axis=1))
        self._cells[key] = (X, stable, mask)
        return self._cells[key]


def score_pair(src, tgt, k, tokens):
    """Fit on every seed of the source, score every seed of the target."""
    rows = []
    for src_seed in src.seeds:
        Xs, ys, ms = src.cell(k, tokens, src_seed)
        if int(ms.sum()) < 20 or len(np.unique(ys[ms])) < 2:
            continue
        scaler = StandardScaler().fit(Xs[ms])
        clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        clf.fit(scaler.transform(Xs[ms]), ys[ms].astype(int))

        for tgt_seed in tgt.seeds:
            if src is tgt and tgt_seed == src_seed:
                continue  # same dictionary: that is the within-dictionary number, not transfer
            Xt, yt, mt = tgt.cell(k, tokens, tgt_seed)
            if int(mt.sum()) < 20 or len(np.unique(yt[mt])) < 2:
                continue
            truth = yt[mt].astype(int)
            rows.append(dict(
                source=src.name, target=tgt.name, k=k, tokens=tokens,
                src_seed=src_seed, tgt_seed=tgt_seed,
                src_d_model=src.d_model, tgt_d_model=tgt.d_model,
                auroc_source_scaler=roc_auc_score(
                    truth, clf.predict_proba(scaler.transform(Xt[mt]))[:, 1]),
                auroc_target_scaler=roc_auc_score(
                    truth, clf.predict_proba(
                        StandardScaler().fit(Xt[mt]).transform(Xt[mt]))[:, 1]),
                retrained_on_target=cv_auroc(Xt, yt, mt),
                frequency_only=cv_auroc(Xt[:, 0], yt, mt),
                n_target=int(mt.sum()), target_stable_frac=float(truth.mean()),
            ))
    return rows


def report(df):
    print(f"\n{'=' * 100}\nCROSS-MODEL TRANSFER\n{'=' * 100}")
    group = df.groupby(["source", "target", "k", "tokens"], as_index=False).mean(
        numeric_only=True)
    print(f"  {'source -> target':<46}{'k':>4}{'src':>7}{'tgt':>7}"
          f"{'ceiling':>9}{'freq':>7}{'n':>7}")
    for _, r in group.iterrows():
        print(f"  {r.source + ' -> ' + r.target:<46}{int(r.k):>4}"
              f"{r.auroc_source_scaler:>7.3f}{r.auroc_target_scaler:>7.3f}"
              f"{r.retrained_on_target:>9.3f}{r.frequency_only:>7.3f}{int(r.n_target):>7}")

    print("\n  src = source scaler reused (does the decision BOUNDARY transfer)")
    print("  tgt = scaler refit on target (does the learned RANKING transfer)")
    print("  ceiling = classifier retrained on the target's own labels; the upper bound")
    print("  freq = the target's frequency-only baseline; transfer below this is useless\n")

    for _, r in group.iterrows():
        if r.source == r.target:
            continue
        cost = r.retrained_on_target - r.auroc_target_scaler
        beats = r.auroc_target_scaler - r.frequency_only
        verdict = ("transfers" if cost < 0.05 and beats > 0.05 else
                   "degrades" if beats > 0.05 else "FAILS")
        print(f"  {r.source} -> {r.target} (k={int(r.k)}): {verdict}. "
              f"ranking transfer costs {cost:+.3f} against the ceiling and beats the "
              f"target's frequency baseline by {beats:+.3f}.")
        gap = abs(r.auroc_target_scaler - r.auroc_source_scaler)
        if gap > 0.05:
            print(f"      The two conventions differ by {gap:.3f}: the ranking travels but the "
                  f"decision boundary does not,\n      so the diagnostic needs per-model "
                  f"recalibration before any absolute threshold is applied.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="pythia-70m-deduped")
    ap.add_argument("--target", action="append", default=None,
                    help="target model (repeatable)")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=None,
                    help="budget to compare at; default the largest common to all models")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--list", action="store_true", help="show what is on the Hub and exit")
    ap.add_argument("--stats-only", action="store_true",
                    help="compute and cache statistics, then stop")
    args = ap.parse_args()

    if args.list:
        print(f"Repo {args.repo}, objective tag {OBJECTIVE_TAG}, relative depth {REL_DEPTH:g}\n")
        for name in PYTHIA_GEOMETRY:
            try:
                grid = ModelGrid(name, args.repo)
            except SystemExit:
                continue
            status = grid.describe() if grid.found else (
                f"{name:<22} no checkpoints under {hub_prefix_for(name, OBJECTIVE_TAG, REL_DEPTH)}")
            print(f"  {status}")
        return

    names = [args.source] + (args.target or [])
    grids = {}
    for name in names:
        grid = ModelGrid(name, args.repo)
        if not grid.found:
            raise SystemExit(
                f"No checkpoints for {name} under checkpoints/{grid.prefix} in {args.repo}.\n"
                f"Train them first:\n"
                f"  SAE_MODEL={name} SAE_K_VALUES={args.k} SAE_MAX_TOKENS=1000000000 \\\n"
                f"      SAE_HF_REPO={args.repo} python topk_sweep_experiments.py\n"
                f"Ground-truth labels are required at every target, not just at the source, "
                f"so this needs all {len(grid.seeds) or 3} seeds."
            )
        print(f"  {grid.describe()}")
        grids[name] = grid

    common = set.intersection(*[{t for k, t in g.cells if k == args.k} for g in grids.values()])
    if not common:
        raise SystemExit(
            f"No budget with k={args.k} is complete for all of {names}. Per-model cells: "
            + "; ".join(f"{n}: {sorted({t for k, t in g.cells if k == args.k})}"
                        for n, g in grids.items())
        )
    tokens = args.tokens or max(common)
    if tokens not in common:
        raise SystemExit(f"{tokens:,} is not complete for every model; available: {sorted(common)}")
    print(f"\nComparing at k={args.k}, {tokens:,} tokens on {DEVICE}")

    for grid in grids.values():
        grid.ensure_stats(args.k, tokens)
    if args.stats_only:
        print("Statistics cached; stopping before scoring as requested.")
        return

    rows = []
    src = grids[args.source]
    for tgt in grids.values():
        rows += score_pair(src, tgt, args.k, tokens)
    if not rows:
        raise SystemExit("Nothing scored.")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "cross_scale_transfer.csv", index=False)
    (RESULTS_DIR / "config.json").write_text(json.dumps({
        "repo": args.repo, "objective_tag": OBJECTIVE_TAG, "rel_depth": REL_DEPTH,
        "source": args.source, "targets": args.target or [], "k": args.k, "tokens": tokens,
        "eval_batches": N_EVAL_BATCHES, "theta": THETA, "epsilon": EPSILON,
        "min_firings": MIN_FIRINGS, "device": DEVICE,
        "models": {n: {"d_model": g.d_model, "hook": g.hook_point, "prefix": g.prefix}
                   for n, g in grids.items()},
    }, indent=2))
    report(df)
    print(f"\nWrote {RESULTS_DIR}/cross_scale_transfer.csv and config.json")


if __name__ == "__main__":
    main()
