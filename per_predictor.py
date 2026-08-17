"""Per-predictor AUROCs, leave-one-out ablations and a bootstrap CI on delta.

Fills the three gaps the sweep never reported: how much each of the six statistics is worth
on its own, how much the full model loses when each is removed, and how wide the interval on
delta is at the sample sizes we have.

Weight-only apart from the cached statistics, so this needs no GPU and no activation pass:
labels come from decoder cosines between the three seeds at one budget, and the predictors
come from the npz cache written by budget_transfer.py.

    SAE_CACHE_DIR   where stats_k{k}_seed{s}_t{t}_e{n}.npz live (default ./cache)
    SAE_RESULTS_DIR default ./outputs/per_predictor

    python per_predictor.py --k 64 --tokens 1000000000 --bootstrap 2000
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from sae_stats import (
    CV,
    EPSILON,
    MIN_FIRINGS,
    THETA,
    build_predictors,
    complete_cells,
    compute_reappearance_probability,
    cv_auroc,
    discover,
    load_sae,
)

CACHE_DIR = Path(os.environ.get("SAE_CACHE_DIR") or "cache")
RESULTS_DIR = Path(os.environ.get("SAE_RESULTS_DIR") or "outputs/per_predictor")
N_EVAL_BATCHES = int(os.environ.get("SAE_EVAL_BATCHES") or 40)


def load_stats(k, seed, tokens):
    path = CACHE_DIR / f"stats_k{k}_seed{seed}_t{tokens}_e{N_EVAL_BATCHES}.npz"
    with np.load(path) as z:
        return {key: z[key] for key in z.files}


def cell_arrays(found, k, seed, tokens, seeds):
    """Predictor matrix, binary truth and the labelled mask for one anchor dictionary."""
    ordered = [seed] + [s for s in seeds if s != seed]
    saes = {s: load_sae(found, k, s, tokens) for s in ordered}
    probs, _ = compute_reappearance_probability(saes, theta=THETA)
    del saes

    stats = load_stats(k, seed, tokens)
    names = [n for n, _ in build_predictors(stats)]
    X = np.column_stack([v for _, v in build_predictors(stats)])
    stable = probs >= (1 - EPSILON)
    mask = ((stable | (probs <= EPSILON))
            & (stats["firing_counts"] >= MIN_FIRINGS)
            & np.isfinite(X).all(axis=1))
    return names, X, stable, mask


def oof_scores(X, y):
    """Out-of-fold decision scores under the same protocol cv_auroc uses."""
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    return cross_val_predict(clf, StandardScaler().fit_transform(X), y, cv=CV,
                             method="decision_function")


def bootstrap_delta(full_scores, freq_scores, y, draws, rng):
    """Percentile interval on delta, resampling features from fixed out-of-fold scores.

    Holding the fit and resampling the evaluation set isolates the quantity we actually
    want an interval on: how much delta would move on another dictionary of this size.
    """
    idx = np.arange(len(y))
    deltas = np.empty(draws)
    n_drawn = 0
    while n_drawn < draws:
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[take])) < 2:
            continue
        deltas[n_drawn] = (roc_auc_score(y[take], full_scores[take])
                           - roc_auc_score(y[take], freq_scores[take]))
        n_drawn += 1
    return np.percentile(deltas, [2.5, 97.5]), deltas.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=1_000_000_000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    found = discover()
    _, seeds, _ = complete_cells(found)
    print(f"k={args.k} @ {args.tokens:,} tokens, seeds {seeds}, floor {MIN_FIRINGS}")

    rng = np.random.default_rng(0)
    singles, ablations, intervals = [], [], []

    for anchor in seeds:
        names, X, stable, mask = cell_arrays(found, args.k, anchor, args.tokens, seeds)
        y = stable[mask].astype(int)
        full = cv_auroc(X, stable, mask)
        freq = cv_auroc(X[:, 0], stable, mask)
        print(f"\n  anchor {anchor}: n={int(mask.sum())} stable={y.mean():.1%} "
              f"full={full:.3f} freq={freq:.3f} delta={full - freq:+.3f}")

        for j, name in enumerate(names):
            singles.append(dict(anchor=anchor, predictor=name,
                                auroc=cv_auroc(X[:, j], stable, mask)))
            keep = [c for c in range(X.shape[1]) if c != j]
            without = cv_auroc(X[:, keep], stable, mask)
            ablations.append(dict(anchor=anchor, dropped=name, auroc_without=without,
                                  loss=full - without))

        lo_hi, sd = bootstrap_delta(oof_scores(X[mask], y), oof_scores(X[mask][:, :1], y),
                                    y, args.bootstrap, rng)
        intervals.append(dict(anchor=anchor, n=int(mask.sum()), auroc_full=full,
                              auroc_freq=freq, delta=full - freq,
                              ci_lo=lo_hi[0], ci_hi=lo_hi[1], boot_sd=sd))

    singles = pd.DataFrame(singles)
    ablations = pd.DataFrame(ablations)
    intervals = pd.DataFrame(intervals)

    print("\n  Single-predictor AUROC (mean over anchors, ranked)")
    for name, v in singles.groupby("predictor").auroc.mean().sort_values(ascending=False).items():
        row = singles[singles.predictor == name].auroc
        print(f"    {name:<24} {v:.3f}  [{row.min():.3f}, {row.max():.3f}]")

    print("\n  Leave-one-out: AUROC of the other five, and what dropping it costs")
    for name, v in ablations.groupby("dropped").loss.mean().sort_values(ascending=False).items():
        row = ablations[ablations.dropped == name]
        print(f"    {name:<24} {row.auroc_without.mean():.3f}  {v:+.3f}")

    print("\n  Delta with a 95% feature-level bootstrap interval")
    for r in intervals.itertuples():
        print(f"    anchor {r.anchor:>4}: {r.delta:+.3f}  [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    singles.to_csv(RESULTS_DIR / "single_predictor.csv", index=False)
    ablations.to_csv(RESULTS_DIR / "leave_one_out.csv", index=False)
    intervals.to_csv(RESULTS_DIR / "delta_ci.csv", index=False)
    print(f"\nWritten to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
