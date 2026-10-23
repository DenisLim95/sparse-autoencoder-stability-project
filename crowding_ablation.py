# -*- coding: utf-8 -*-
"""Crowding ablation and decoder-vector baseline: how much rests on one hand-picked statistic?

The paper's central claim is that neighbour crowding -- the mean cosine to the 10 nearest
decoder neighbours -- does most of the predictive work. That definition was chosen once, by
hand. This script asks the two questions a reviewer asked about it:

  1. Is the result fragile to the definition? Crowding is recomputed as the mean of the top
     1, 5, 10, 20 and 50 neighbour cosines (top-1 is the max cosine), as the count of
     neighbours above a cosine of 0.3 and 0.5 over the whole dictionary, and as the sorted
     top-20 cosines fed in as 20 columns, which lets the classifier learn which part of the
     neighbourhood matters instead of fixing it in advance. Each variant is scored alone and
     inside the six-predictor model, replacing the paper's definition.

  2. What does width-independence cost? A classifier on the raw decoder row (d_model columns)
     cannot be applied to a model of another width, which is why the paper uses six scalars.
     Scoring it here, alone and alongside the six, says how much accuracy that portability
     gives up. With hundreds of columns and a few thousand latents, the protocol's fixed
     regularisation overfits (cross-validated AUROC falls below chance where there is little
     to learn), so these rows choose the regularisation strength by an inner cross-validation
     inside each training fold, and the six statistics are rescored the same way so the
     comparison is like for like. The decoder row also gets a shuffled-label null through the
     same tuned estimator, because at k=256 it scores far below chance and the null is what
     separates "nothing to learn" from "systematically wrong".

Single-column variants also get a rank AUROC: the statistic used directly as a score, no
classifier fitted, > 0.5 meaning a higher value goes with stable. Where the signal is weak the
protocol's cross-validated AUROC is unstable, because the fitted coefficient sits near zero
and its sign flips between folds; the rank AUROC is not, and it keeps the direction visible.

Labels, the firing floor and the other five predictors come from the sweep's exported tables
through permutation_null.load_arm, so every AUROC here is on the paper's labelled set. Only
the anchor's decoder is downloaded, to recompute the neighbourhood. Before anything is
reported, the recomputed mean top-10 must match the table's own crowding column, and both of
its AUROCs must match the ones the table gives; the script stops if either does not.

Weights only: no activations, no GPU, a few minutes per arm on a CPU.

Usage:
    python crowding_ablation.py                  # k = 64, 128, 256 at 1B
    python crowding_ablation.py --k 64           # one arm
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from permutation_null import CROWDING_PREDICTOR, FREQ_PREDICTOR, column_of, load_arm
from sae_stats import (
    CV,
    DEFAULT_PREFIX,
    DEFAULT_REPO,
    EPSILON,
    MIN_FIRINGS,
    THETA,
    cv_auroc,
    discover,
    load_decoder,
)

RESULTS_DIR = Path(os.environ.get("SAE_RESULTS_DIR") or "outputs/crowding_ablation")
ANCHOR = 42  # the only anchor the exported tables cover
TOP_M = 50
MEAN_TOP_N = (1, 5, 10, 20, 50)
TAUS = (0.3, 0.5)
SORTED_TOP = 20
PAPER_TOP_N = 10  # the definition in the paper, = compute_geometric_isolation's k_nn
CROWDING_TOLERANCE = 1e-5
AUROC_TOLERANCE = 1e-4
# Inverse regularisation strengths searched inside each training fold for the tuned rows.
TUNED_CS = np.logspace(-4, 1, 6)
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
N_NULL = 10  # shuffled-label replicates for the decoder row


def neighbourhood(A: torch.Tensor, m: int = TOP_M, taus=TAUS, chunk: int = 1024):
    """Sorted top-m neighbour cosines (n x m, descending) and, per tau, the number of
    neighbours above tau, from one chunked pass over A @ A.T with each row's self excluded.

    A must be unit-normalised (load_decoder returns it that way). Every crowding variant is
    derived from these two outputs, so all of them see exactly the same neighbourhood.
    """
    n = A.shape[0]
    top = torch.empty(n, m)
    counts = {tau: torch.empty(n, dtype=torch.long) for tau in taus}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = A[start:stop] @ A.T
        rows = torch.arange(start, stop)
        sims[rows - start, rows] = -float("inf")
        top[start:stop] = sims.topk(m, dim=1).values  # topk sorts, largest first
        for tau in taus:
            counts[tau][start:stop] = (sims > tau).sum(dim=1)
    return top.numpy(), {tau: c.numpy() for tau, c in counts.items()}


def variants(top, counts):
    """(name, family, columns) for every crowding definition under test."""
    out = [(f"mean top-{n}" + (" (= max cosine)" if n == 1 else ""), "mean of top-n",
            top[:, :n].mean(axis=1, keepdims=True)) for n in MEAN_TOP_N]
    # log1p for the same reason build_predictors logs frequency: the counts are heavy-tailed,
    # and in the multivariable model raw units would let other columns bend the response.
    # Monotone, so it does not change the single-column AUROC.
    out += [(f"count above {tau:g}", "count above tau", np.log1p(c)[:, None])
            for tau, c in counts.items()]
    out.append((f"sorted top-{SORTED_TOP}", "sorted cosines", top[:, :SORTED_TOP]))
    return out


def rank_auroc(values, labels, mask):
    """The statistic itself as the score, no fitting: > 0.5 means higher goes with stable."""
    return float(roc_auc_score(labels[mask].astype(int), values[mask]))


def cv_auroc_tuned(values, labels, mask):
    """cv_auroc's outer folds, with the regularisation chosen inside each training fold.

    Standardisation is fitted inside the folds too (cv_auroc fits it on the whole labelled
    set), which only makes this the stricter of the two estimates.
    """
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=TUNED_CS, cv=INNER_CV, scoring="roc_auc", class_weight="balanced",
        max_iter=2000, random_state=42, n_jobs=-1,
    ))
    return float(cross_val_score(model, values[mask], labels[mask].astype(int),
                                 cv=CV, scoring="roc_auc").mean())


def shuffled_null(values, labels, mask, rng, n=N_NULL):
    """cv_auroc_tuned on labels shuffled within the labelled set (n and balance kept)."""
    idx = np.flatnonzero(mask)
    out = []
    for _ in range(n):
        shuffled = labels.copy()
        shuffled[idx] = rng.permutation(labels[idx])
        out.append(cv_auroc_tuned(values, shuffled, mask))
    return np.array(out)


def ablate_arm(k, budget, tokens, repo, prefix, rng):
    names, X, stable, mask = load_arm(k, budget, repo, prefix)
    i_freq = column_of(names, FREQ_PREDICTOR)
    i_crowd = column_of(names, CROWDING_PREDICTOR)
    others = np.delete(X, i_crowd, axis=1)

    found = discover(repo, prefix)
    if (k, ANCHOR, tokens) not in found:
        raise SystemExit(f"No checkpoint k={k} seed={ANCHOR} @ {tokens:,} under {prefix}.")
    A = load_decoder(found, k, ANCHOR, tokens, repo=repo)
    if A.shape[0] != X.shape[0]:
        raise SystemExit(f"k={k}: decoder has {A.shape[0]} rows, the table {X.shape[0]}.")
    top, counts = neighbourhood(A)

    # The recomputed paper definition must be the table's crowding column, and must score
    # what the table scores. Otherwise the variants are being compared on something else.
    paper = top[:, :PAPER_TOP_N].mean(axis=1)
    gap = float(np.nanmax(np.abs(paper - X[:, i_crowd])))
    if gap > CROWDING_TOLERANCE:
        raise SystemExit(f"k={k}: recomputed mean top-{PAPER_TOP_N} differs from the table's "
                         f"crowding by up to {gap:.2e}; check the checkpoint and the table.")
    freq = cv_auroc(X[:, i_freq], stable, mask)
    reference = {
        "alone": cv_auroc(X[:, i_crowd], stable, mask),
        "full": cv_auroc(X, stable, mask),
    }
    rebuilt = {
        "alone": cv_auroc(paper, stable, mask),
        "full": cv_auroc(np.column_stack([others, paper]), stable, mask),
    }
    for key in reference:
        if abs(reference[key] - rebuilt[key]) > AUROC_TOLERANCE:
            raise SystemExit(f"k={k}: the recomputed crowding scores {rebuilt[key]:.4f} "
                             f"({key}) against the table's {reference[key]:.4f}.")
    print(f"  k={k}: recomputed crowding matches the table (max gap {gap:.1e}); "
          f"alone {reference['alone']:.3f}, full {reference['full']:.3f}, "
          f"frequency only {freq:.3f}, n={int(mask.sum())}")

    nan = float("nan")

    def row(name, family, cols, alone, full, rank=nan, null=None):
        return dict(k=k, variant=name, family=family, n_columns=cols.shape[1],
                    n_labelled=int(mask.sum()), auroc_rank=rank, auroc_alone=alone,
                    auroc_full=full, delta_full=full - freq,
                    null_mean=nan if null is None else float(null.mean()),
                    null_min=nan if null is None else float(null.min()),
                    null_max=nan if null is None else float(null.max()))

    def single(cols):
        return rank_auroc(cols[:, 0], stable, mask) if cols.shape[1] == 1 else nan

    freq_col = X[:, [i_freq]]
    rows = [row("frequency only", "baseline", freq_col, freq, nan, single(freq_col)),
            row(f"paper: mean top-{PAPER_TOP_N}", "paper", paper[:, None],
                reference["alone"], reference["full"], single(paper[:, None]))]
    for name, family, cols in variants(top, counts):
        rows.append(row(name, family, cols, cv_auroc(cols, stable, mask),
                        cv_auroc(np.column_stack([others, cols]), stable, mask), single(cols)))

    # Tuned rows: the six statistics as a like-for-like reference, then the decoder vector.
    D = A.numpy()
    print(f"  k={k}: tuning the six-statistic and decoder-vector classifiers "
          f"({len(TUNED_CS)} strengths x {INNER_CV.get_n_splits()} inner folds)...")
    rows.append(row("six statistics, C tuned", "tuned", X, nan, cv_auroc_tuned(X, stable, mask)))
    print(f"  k={k}: decoder-row shuffled-label null ({N_NULL} replicates)...")
    rows.append(row("raw decoder row, C tuned", "tuned", D, cv_auroc_tuned(D, stable, mask),
                    cv_auroc_tuned(np.column_stack([D, X]), stable, mask),
                    null=shuffled_null(D, stable, mask, rng)))
    return pd.DataFrame(rows)


def report(df):
    for k, arm in df.groupby("k", sort=True):
        print(f"\n  k={k}  (n={int(arm.n_labelled.iloc[0])}; rank = the statistic as the "
              f"score, no fitting; full = with the other five predictors, or for the decoder "
              f"row alongside all six)")
        print(f"    {'variant':<30}{'cols':>5}{'rank':>8}{'alone':>8}{'full':>8}{'delta':>8}")

        def fmt(v, spec=".3f"):
            return "" if np.isnan(v) else format(v, spec)

        for r in arm.itertuples():
            print(f"    {r.variant:<30}{r.n_columns:>5}{fmt(r.auroc_rank):>8}"
                  f"{fmt(r.auroc_alone):>8}{fmt(r.auroc_full):>8}{fmt(r.delta_full, '+.3f'):>8}")
            if not np.isnan(r.null_mean):
                verdict = ("below its null" if r.auroc_alone < r.null_min else
                           "above its null" if r.auroc_alone > r.null_max else "within its null")
                print(f"    {'':<30}{'':>5}  decoder alone on shuffled labels: mean "
                      f"{r.null_mean:.3f} [{r.null_min:.3f}, {r.null_max:.3f}] -> {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, action="append", default=None,
                    help="sparsity arm (repeatable); default 64, 128 and 256")
    ap.add_argument("--budget", default="at1B", help="exported table to read (default at1B)")
    ap.add_argument("--anchor-tokens", type=int, default=1_000_000_000,
                    help="checkpoint budget matching --budget (default 1B)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the shuffled-label null")
    args = ap.parse_args()
    ks = args.k or [64, 128, 256]

    print(f"Repo {args.repo}  prefix {args.prefix}  {args.budget}  anchor {ANCHOR}  "
          f"floor {MIN_FIRINGS}")
    rng = np.random.default_rng(args.seed)
    df = pd.concat([ablate_arm(k, args.budget, args.anchor_tokens, args.repo, args.prefix, rng)
                    for k in ks], ignore_index=True)
    report(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "crowding_ablation.csv", index=False)
    (RESULTS_DIR / "config.json").write_text(json.dumps({
        "repo": args.repo, "prefix": args.prefix, "budget": args.budget,
        "anchor": ANCHOR, "anchor_tokens": args.anchor_tokens, "k_values": ks,
        "top_m": TOP_M, "mean_top_n": list(MEAN_TOP_N), "taus": list(TAUS),
        "sorted_top": SORTED_TOP, "theta": THETA, "epsilon": EPSILON,
        "min_firings": MIN_FIRINGS, "tuned_Cs": TUNED_CS.tolist(),
        "null_replicates": N_NULL, "rng_seed": args.seed,
    }, indent=2))
    print(f"\nWrote {RESULTS_DIR}/crowding_ablation.csv and config.json")


if __name__ == "__main__":
    main()
