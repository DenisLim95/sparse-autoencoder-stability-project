# -*- coding: utf-8 -*-
"""Label permutation null: what does the cross-validation protocol score on scrambled labels?

The reported AUROCs come from five-fold stratified cross-validation over features of a single
dictionary. AUROC has a nominal chance level of 0.5, but that is a property of the metric, not
of this pipeline: an indexing slip between predictors and labels, a fold that is not actually
held out, or a preprocessing step that sees the labels would all raise the floor without
raising any obvious alarm. This script measures the floor instead of assuming it.

Each replicate shuffles the stability labels WITHIN the labelled set, so n and the class
balance are preserved exactly, and then re-runs the whole protocol through the same `cv_auroc`
the reported figures use. Permuting once per replicate and re-running the entire procedure --
rather than shuffling inside the fold loop -- is what makes this a test of the protocol rather
than of the estimator.

Three quantities are scored on the SAME permuted vector, which keeps them paired:

    all six predictors      the headline classifier
    activation frequency    the baseline delta is defined against
    neighbour crowding      the strongest single predictor

Pairing is what makes the fourth quantity meaningful. delta = AUROC(all six) - AUROC(frequency)
is the paper's primary metric and is a DIFFERENCE of two AUROCs, so its own null is not
something that can be read off a table. Here it comes out of the same replicates.

What this does NOT establish: neighbour crowding is a relational statistic, so a feature's
value depends on neighbours that sit in the training fold, and folds within one dictionary are
therefore not fully independent. Permuting labels destroys the label-feature relationship
whether or not that dependence exists, so this control cannot detect it. Cross-seed transfer
(budget_transfer.py, cross_scale_transfer.py) is what answers that question.

Analysis-only: reads the per-feature tables the sweep exported, so no checkpoints, no
activations, no GPU. The anchor is seed 42, the only anchor those tables cover; the null is a
property of the protocol, n and the class balance, and those barely move across anchors.

Usage:
    python permutation_null.py                            # k=64, 128, 256 at 1B, B=1000
    python permutation_null.py --k 64                     # one arm
    python permutation_null.py --permutations 200         # quicker, coarser p-value floor
    python permutation_null.py --budget at718M

Environment:
    SAE_RESULTS_DIR   default ./outputs/permutation_null
    SAE_MIN_FIRINGS   default 100, read by sae_stats and applied to the exported counts
    SAE_EVAL_BATCHES  default 40, recorded only (it is what the firing counts were measured over)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sae_stats import (
    DEFAULT_PREFIX,
    DEFAULT_REPO,
    EPSILON,
    MIN_FIRINGS,
    THETA,
    build_predictors,
    cv_auroc,
    load_exported_csv,
)

RESULTS_DIR = Path(os.environ.get("SAE_RESULTS_DIR") or "outputs/permutation_null")
N_EVAL_BATCHES = int(os.environ.get("SAE_EVAL_BATCHES") or 40)

# Looked up by name rather than by position, so a reordering of build_predictors cannot
# silently redefine which column is the baseline and which is the geometric statistic.
FREQ_PREDICTOR = "Activation Freq (log)"
CROWDING_PREDICTOR = "Geometric Isolation"

QUANTITIES = ("auroc_full", "auroc_freq", "auroc_crowding", "delta")


def load_arm(k, budget, repo, prefix):
    """Predictor matrix, binary truth and the labelled mask for one exported arm.

    The mask is the reported one: endpoint-binarized labels, above the firing floor, and
    finite on every predictor. It reproduces `cell_arrays` in budget_transfer.py, which is
    what makes the observed values here comparable with the paper's.
    """
    df = load_exported_csv(k, budget=budget, repo=repo, prefix=prefix)

    # The exported columns are named for the keys build_predictors reads, so the frame is
    # already a stats dict and the log transforms and column order are inherited rather than
    # restated here.
    stats = {column: df[column].to_numpy() for column in df.columns}
    names = [name for name, _ in build_predictors(stats)]
    X = np.column_stack([values for _, values in build_predictors(stats)])

    stable = df["is_stable"].to_numpy().astype(bool)
    mask = (
        (stable | df["is_unstable"].to_numpy().astype(bool))
        & (df["firing_counts"].to_numpy() >= MIN_FIRINGS)
        & np.isfinite(X).all(axis=1)
    )
    return names, X, stable, mask


def column_of(names, predictor):
    if predictor not in names:
        raise SystemExit(
            f"build_predictors no longer emits {predictor!r} (it emits {names}). "
            f"Update FREQ_PREDICTOR / CROWDING_PREDICTOR in this script."
        )
    return names.index(predictor)


def score(X, y, mask, freq_col, crowding_col):
    """The three AUROCs and their derived delta, all from one labelling."""
    full = cv_auroc(X, y, mask)
    freq = cv_auroc(X[:, freq_col], y, mask)
    return {
        "auroc_full": full,
        "auroc_freq": freq,
        "auroc_crowding": cv_auroc(X[:, crowding_col], y, mask),
        "delta": full - freq,
    }


def permute(stable, mask, rng):
    """Shuffle labels within the labelled set only.

    Permuting the masked subset rather than the whole dictionary holds n and the class balance
    fixed at their reported values, so the null describes the ranking problem actually being
    solved rather than a differently-balanced one.
    """
    permuted = stable.copy()
    permuted[mask] = rng.permutation(stable[mask])
    return permuted


def run_arm(k, budget, repo, prefix, n_permutations, seed):
    names, X, stable, mask = load_arm(k, budget, repo, prefix)
    freq_col = column_of(names, FREQ_PREDICTOR)
    crowding_col = column_of(names, CROWDING_PREDICTOR)

    n = int(mask.sum())
    if n < 20 or len(np.unique(stable[mask])) < 2:
        print(f"  k={k}: only {n} labelled latents, skipped")
        return None, None

    observed = score(X, stable, mask, freq_col, crowding_col)
    stable_frac = float(stable[mask].mean())
    print(f"\n=== k={k} @ {budget}: n={n:,} stable={stable_frac:.1%} ===")
    print(f"  observed  full={observed['auroc_full']:.3f} "
          f"frequency={observed['auroc_freq']:.3f} "
          f"crowding={observed['auroc_crowding']:.3f} "
          f"delta={observed['delta']:+.3f}")

    rng = np.random.default_rng(seed)
    draws = []
    # disable=None leaves the bar on for interactive use and silences it when the output is
    # redirected, which a thousand-iteration loop would otherwise fill a log file with.
    for replicate in tqdm(range(n_permutations), desc=f"  permuting k={k}", disable=None):
        row = score(X, permute(stable, mask, rng), mask, freq_col, crowding_col)
        draws.append(dict(k=k, budget=budget, replicate=replicate, **row))
    draws = pd.DataFrame(draws)

    summary = []
    for quantity in QUANTITIES:
        null = draws[quantity].to_numpy()
        null = null[np.isfinite(null)]
        seen = observed[quantity]
        summary.append(dict(
            k=k, budget=budget, quantity=quantity, n_labelled=n, stable_frac=stable_frac,
            n_permutations=len(null), observed=seen,
            null_mean=null.mean(), null_sd=null.std(ddof=1),
            null_p2_5=np.percentile(null, 2.5), null_p97_5=np.percentile(null, 97.5),
            null_max=null.max(),
            # The +1 in both places counts the observed value as one of its own null draws,
            # which is what keeps the p-value from ever being exactly zero and makes it valid
            # at the resolution the number of permutations actually bought.
            p_value=(1 + int((null >= seen).sum())) / (1 + len(null)),
            # How far above the null the observed value sits, in null standard deviations.
            # This is the quantity worth quoting next to a small difference elsewhere.
            z=(seen - null.mean()) / null.std(ddof=1) if null.std(ddof=1) > 0 else float("nan"),
        ))
    return draws, pd.DataFrame(summary)


def report(summary):
    print(f"\n{'=' * 92}\nPERMUTATION NULL\n{'=' * 92}")
    print(f"  {'k':>4} {'quantity':<16}{'observed':>10}{'null mean':>11}{'null sd':>9}"
          f"{'null 95%':>10}{'null max':>10}{'z':>8}{'p':>9}")
    for _, r in summary.iterrows():
        print(f"  {int(r.k):>4} {r.quantity:<16}{r.observed:>10.3f}{r.null_mean:>11.3f}"
              f"{r.null_sd:>9.4f}{r.null_p97_5:>10.3f}{r.null_max:>10.3f}"
              f"{r.z:>8.1f}{r.p_value:>9.4f}")

    print("\n  null mean at 0.500 for the three AUROCs, and at 0.000 for delta, is the result:")
    print("  the protocol returns chance when the labels carry no information.")
    print("  null sd is the yardstick for small differences reported elsewhere -- a gap")
    print("  narrower than this one is not distinguishable from noise at these sample sizes.")

    worst = summary[summary.quantity.isin(("auroc_full", "auroc_freq", "auroc_crowding"))]
    drift = (worst.null_mean - 0.5).abs().max()
    if drift > 0.02:
        print(f"\n  WARNING: a null mean sits {drift:.3f} away from 0.500. The protocol is")
        print("  scoring above chance on scrambled labels, which is a leak, not a null.")
        print("  Do not report the observed AUROCs against a 0.5 floor until this is resolved.")


def plot(draws, summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the null histogram.")
        return

    ks = sorted(draws.k.unique())
    shown = ("auroc_full", "delta")
    titles = {"auroc_full": "AUROC (all six predictors)", "delta": "delta over frequency"}

    fig, axes = plt.subplots(len(ks), len(shown), figsize=(5.5 * len(shown), 3.1 * len(ks)),
                             squeeze=False)
    for row, k in enumerate(ks):
        for col, quantity in enumerate(shown):
            ax = axes[row][col]
            null = draws[draws.k == k][quantity].to_numpy()
            null = null[np.isfinite(null)]
            seen = float(summary[(summary.k == k)
                                 & (summary.quantity == quantity)].observed.iloc[0])
            ax.hist(null, bins=40, color="0.6", edgecolor="none",
                    label=f"null {null.mean():.3f} +/- {null.std(ddof=1):.4f}")
            ax.axvline(seen, color="crimson", lw=2, label=f"observed {seen:.3f}")
            ax.axvline(0.5 if quantity == "auroc_full" else 0.0, color="black", lw=0.8, ls=":",
                       label="chance")
            # Both the null and the observed value have to be on screen, or the figure shows a
            # spike and nothing to compare it against.
            lo = min(null.min(), seen)
            hi = max(null.max(), seen)
            pad = 0.06 * max(hi - lo, 1e-3)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_title(f"k = {k}: {titles[quantity]}", fontsize=10)
            ax.set_ylabel("permutations")
            ax.grid(alpha=0.3)
            # Every panel carries its own legend: the observed value differs per panel, so one
            # shared legend would label five of the six red lines with the wrong number.
            ax.legend(fontsize=7, loc="upper center")
    fig.suptitle("Label permutation null: the CV protocol scored on scrambled labels")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, action="append", default=None,
                    help="sparsity arm (repeatable); default 64, 128 and 256")
    ap.add_argument("--permutations", type=int, default=1000,
                    help="B; the p-value floors at 1/(B+1)")
    ap.add_argument("--budget", default="at1B", help="exported budget directory")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the permutations")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = ap.parse_args()

    ks = args.k or [64, 128, 256]
    print(f"Repo {args.repo}  budget {args.budget}  floor {MIN_FIRINGS}  "
          f"B={args.permutations}  seed {args.seed}")

    all_draws, all_summary = [], []
    for k in ks:
        draws, summary = run_arm(k, args.budget, args.repo, args.prefix,
                                 args.permutations, args.seed)
        if draws is not None:
            all_draws.append(draws)
            all_summary.append(summary)

    if not all_summary:
        raise SystemExit("Nothing scored.")

    draws = pd.concat(all_draws, ignore_index=True)
    summary = pd.concat(all_summary, ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    draws.to_csv(RESULTS_DIR / "permutations.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    (RESULTS_DIR / "config.json").write_text(json.dumps({
        "repo": args.repo, "prefix": args.prefix, "budget": args.budget,
        "source": f"results/{args.prefix}/{args.budget}/feature_stability_k{{k}}.csv",
        "anchor_seed": 42, "k_values": ks,
        "permutations": args.permutations, "rng_seed": args.seed,
        "theta": THETA, "epsilon": EPSILON, "min_firings": MIN_FIRINGS,
        "eval_batches": N_EVAL_BATCHES,
        "freq_predictor": FREQ_PREDICTOR, "crowding_predictor": CROWDING_PREDICTOR,
    }, indent=2))

    report(summary)
    plot(draws, summary, RESULTS_DIR / "permutation_null.png")
    print(f"\nWrote {RESULTS_DIR}/permutations.csv, summary.csv, permutation_null.png "
          f"and config.json")


if __name__ == "__main__":
    main()
