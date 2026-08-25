# -*- coding: utf-8 -*-
"""Rotation null: is the geometric signal real, or just sphere density?

Implements plan-rotation-null.md.

The worry is structural. The PREDICTOR is "how close is my nearest neighbour, inside my own
dictionary". The LABEL is "how close is my nearest neighbour, in another seed's dictionary".
Both are nearest-neighbour distances, so a latent sitting in a dense region of the sphere
scores high on both for reasons that have nothing to do with stability.

The null replaces the partner dictionary with a randomly ROTATED copy of itself. That destroys
every feature correspondence while preserving the dictionary's marginal geometry exactly. Two
questions, and this script answers both because they fail differently:

  NULL A -- does the LABEL RATE survive rotation?
      If a rotated partner still "matches" a large fraction of the anchor's latents, the
      ground truth itself is partly geometry, which would affect every result in the project.

  NULL B -- does the PREDICTOR'S AUROC survive rotation?
      Relabel from the rotated partner and re-score geometric isolation. Predictive power that
      survives means the statistic was tracking sphere density all along.

Weight-only and fast: no activations, no GPU, no training. The one part that needs activation
statistics is the frequency-band control (--bands), which reuses the cache written by
budget_transfer.py.

A naming warning that applies to every number printed here: `geometric_isolation` is the mean
cosine to the 10 nearest decoder neighbours, so HIGH means CROWDED, not isolated. The column is
printed as "neighbour crowding" to stop the sign confusion at the source.

Usage:
    python rotation_null.py                          # k=64 at the final budget
    python rotation_null.py --k 64 --k 128 --k 256   # every arm
    python rotation_null.py --tokens 50000000 --tokens 1000000000
    python rotation_null.py --rotations 20           # wider null distribution
    python rotation_null.py --bands                  # add the frequency-band control
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sae_stats import (
    DEFAULT_PREFIX,
    DEFAULT_REPO,
    EPSILON,
    MIN_FIRINGS,
    THETA,
    complete_cells,
    discover,
    load_decoder,
    random_rotation,
)

CACHE_DIR = Path(os.environ.get("SAE_CACHE_DIR") or "cache")
RESULTS_DIR = Path(os.environ.get("SAE_RESULTS_DIR") or "outputs/rotation_null")
N_EVAL_BATCHES = int(os.environ.get("SAE_EVAL_BATCHES") or 40)
K_NN = 10  # same neighbourhood size as compute_geometric_isolation


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with ties averaged, so a constant predictor scores 0.5 rather than
    something spurious. Kept dependency-free and sign-honest: no negation anywhere in this
    file, so a value above 0.5 always means "higher statistic -> more stable"."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def crowding(A: torch.Tensor, k_nn: int = K_NN, chunk: int = 1024) -> np.ndarray:
    """Mean cosine to the k nearest neighbours within the same dictionary.

    Same quantity as sae_stats.compute_geometric_isolation, computed from a bare decoder
    matrix so this script never has to instantiate an SAE. HIGH = crowded neighbourhood.
    """
    n = A.shape[0]
    out = torch.empty(n)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = A[start:stop] @ A.T
        rows = torch.arange(start, stop)
        sims[rows - start, rows] = -float("inf")  # exclude self
        out[start:stop] = sims.topk(k_nn, dim=1).values.mean(dim=1)
    return out.numpy()


def best_cosine(A: torch.Tensor, P: torch.Tensor, chunk: int = 1024,
                exclude_self: bool = False) -> np.ndarray:
    """For each anchor row, the best cosine against any row of the partner (many-to-one
    argmax, the sweep's matching rule). Chunked so an 8192 x 8192 product stays modest.

    exclude_self must be set when P IS A, or every latent trivially matches itself at 1.0.
    """
    out = torch.empty(A.shape[0])
    for start in range(0, A.shape[0], chunk):
        stop = min(start + chunk, A.shape[0])
        sims = A[start:stop] @ P.T
        if exclude_self:
            rows = torch.arange(start, stop)
            sims[rows - start, rows] = -float("inf")
        out[start:stop] = sims.max(dim=1).values
    return out.numpy()


def p_hat(best_by_partner, theta: float = THETA) -> np.ndarray:
    """Reappearance probability from per-partner best cosines, the sweep's definition."""
    return np.mean([(b >= theta).astype(float) for b in best_by_partner], axis=0)


def run_cell(found, k, tokens, seeds, anchor, n_rotations, rng, repo):
    """Both nulls for one (k, budget, anchor) dictionary."""
    A = load_decoder(found, k, anchor, tokens, repo=repo)
    partners = {s: load_decoder(found, k, s, tokens, repo=repo) for s in seeds if s != anchor}
    d_model = A.shape[1]

    iso = crowding(A)
    # Nearest neighbour inside the anchor itself, for the redundancy read. Self must be
    # excluded or every latent matches itself at 1.0 and the statistic is a constant.
    nn1 = best_cosine(A, A, exclude_self=True)

    real_best = [best_cosine(A, P) for P in partners.values()]
    real_p = p_hat(real_best)
    real_stable = real_p >= (1 - EPSILON)
    real_unstable = real_p <= EPSILON
    real_labelled = real_stable | real_unstable
    real_rate = float(real_stable.mean())

    real_auroc_iso = auroc(iso[real_labelled], real_stable[real_labelled])
    real_auroc_nn1 = auroc(nn1[real_labelled], real_stable[real_labelled])

    # One rotation gives one number with no sense of its spread, so draw several.
    null_rates, null_iso, null_nn1, null_median = [], [], [], []
    for _ in range(n_rotations):
        R = random_rotation(d_model, rng)
        rot_best = [best_cosine(A, P @ R.T) for P in partners.values()]
        rot_p = p_hat(rot_best)
        null_rates.append(float((rot_p >= (1 - EPSILON)).mean()))
        null_median.append(float(np.median(np.mean(rot_best, axis=0))))

        # Prevalence matching, NOT theta. A rotated dictionary matches almost nothing at
        # theta=0.7, so a fixed threshold would leave the null with too few positives to score
        # and make it trivially unbeatable. Taking the top (real stable rate) fraction of the
        # rotated best-cosines gives the null the same class balance as the real labels, which
        # is the only way the two AUROCs are comparable.
        rot_mean = np.mean(rot_best, axis=0)
        cut = np.quantile(rot_mean, 1 - max(real_rate, 1e-6))
        rot_stable = rot_mean >= cut
        null_iso.append(auroc(iso, rot_stable))
        null_nn1.append(auroc(nn1, rot_stable))

    return dict(
        k=k, tokens=tokens, anchor=anchor, n_features=int(A.shape[0]), d_model=int(d_model),
        n_labelled=int(real_labelled.sum()),
        real_label_rate=real_rate,
        real_median_best_cos=float(np.median(np.mean(real_best, axis=0))),
        null_label_rate_mean=float(np.mean(null_rates)),
        null_label_rate_max=float(np.max(null_rates)),
        null_median_best_cos=float(np.mean(null_median)),
        real_auroc_crowding=real_auroc_iso,
        null_auroc_crowding_mean=float(np.mean(null_iso)),
        null_auroc_crowding_min=float(np.min(null_iso)),
        null_auroc_crowding_max=float(np.max(null_iso)),
        real_auroc_nn1=real_auroc_nn1,
        null_auroc_nn1_mean=float(np.mean(null_nn1)),
        # Redundancy inside the anchor: how many latents have a near-duplicate at home. A high
        # value makes "matched in another dictionary" easy for uninteresting reasons.
        self_duplicate_rate=float((nn1 >= THETA).mean()),
        corr_crowding_real=float(np.corrcoef(iso, np.mean(real_best, axis=0))[0, 1]),
    )


def frequency_bands(found, k, tokens, anchor, seeds, n_bands=10, repo=DEFAULT_REPO):
    """Does crowding still predict stability WITHIN narrow activation-frequency bands?

    The second way the geometry result could be spurious: crowding might simply proxy firing
    frequency, which is the baseline the whole claim has to beat. Needs the per-checkpoint
    statistics cache written by budget_transfer.py.
    """
    stats_file = CACHE_DIR / f"stats_k{k}_seed{anchor}_t{tokens}_e{N_EVAL_BATCHES}.npz"
    if not stats_file.exists():
        print(f"\n  Frequency-band control needs {stats_file.name}, which budget_transfer.py "
              f"writes.\n  Run: python budget_transfer.py stats --k {k}")
        return None

    with np.load(stats_file) as z:
        freq, counts = z["activation_freq"], z["firing_counts"]

    A = load_decoder(found, k, anchor, tokens, repo=repo)
    iso = crowding(A)
    best = [best_cosine(A, load_decoder(found, k, s, tokens, repo=repo))
            for s in seeds if s != anchor]
    p = p_hat(best)
    stable, unstable = p >= (1 - EPSILON), p <= EPSILON
    live = counts >= MIN_FIRINGS
    mask = (stable | unstable) & live

    overall = auroc(iso[mask], stable[mask])
    edges = np.quantile(np.log10(freq[mask] + 1e-10), np.linspace(0, 1, n_bands + 1))

    print(f"\n  Crowding AUROC within activation-frequency deciles (k={k}, {tokens:,} tokens)")
    print(f"    overall (no banding): {overall:.3f}")
    print(f"    {'decile':>7}{'n':>7}{'stable':>8}{'AUROC':>8}")
    rows, scored = [], []
    logf = np.log10(freq + 1e-10)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        band = mask & (logf >= lo) & (logf <= hi if i == n_bands - 1 else logf < hi)
        if band.sum() < 20 or len(np.unique(stable[band])) < 2:
            print(f"    {i + 1:>7}{int(band.sum()):>7}{'':>8}{'too few':>8}")
            continue
        a = auroc(iso[band], stable[band])
        scored.append(a)
        rows.append(dict(k=k, tokens=tokens, anchor=anchor, decile=i + 1,
                         n=int(band.sum()), stable_frac=float(stable[band].mean()), auroc=a))
        print(f"    {i + 1:>7}{int(band.sum()):>7}{stable[band].mean():>8.1%}{a:>8.3f}")

    if scored:
        print(f"    mean within-band AUROC: {np.mean(scored):.3f} "
              f"vs {overall:.3f} unbanded")
        if np.mean(scored) < 0.55:
            print("    -> collapses within bands: crowding is largely a frequency proxy.")
        else:
            print("    -> survives banding: crowding carries something frequency does not.")
    return pd.DataFrame(rows)


def report(df):
    print(f"\n{'=' * 92}\nNULL A -- does the LABEL RATE survive rotation?\n{'=' * 92}")
    print(f"  {'k':>5}{'tokens':>14}{'real rate':>11}{'null rate':>11}{'null max':>10}"
          f"{'real cos':>10}{'null cos':>10}")
    for _, r in df.iterrows():
        print(f"  {int(r.k):>5}{int(r.tokens):>14,}{r.real_label_rate:>10.1%} "
              f"{r.null_label_rate_mean:>10.1%}{r.null_label_rate_max:>10.1%}"
              f"{r.real_median_best_cos:>10.3f}{r.null_median_best_cos:>10.3f}")
    worst = df.null_label_rate_max.max()
    print(f"\n  Highest label rate any rotated partner produced: {worst:.1%}.")
    if worst > 0.10:
        print("  WARNING: the ground truth itself is partly explained by dictionary geometry.\n"
              "  That is a bigger problem than a confounded predictor and affects every result.")
    else:
        print("  The labels are not reproducible by geometry alone, so the ground truth stands.")

    print(f"\n{'=' * 92}\nNULL B -- does the PREDICTOR survive rotation?\n{'=' * 92}")
    print("  Statistic: neighbour crowding (mean cosine to 10 nearest decoder neighbours).")
    print("  HIGH = crowded. No sign flipping anywhere: >0.5 means crowded latents are more")
    print("  stable, which is the direction the data actually shows.\n")
    print(f"  {'k':>5}{'tokens':>14}{'real AUROC':>12}{'null mean':>11}"
          f"{'null range':>18}{'surplus':>9}")
    for _, r in df.iterrows():
        rng_s = f"[{r.null_auroc_crowding_min:.3f}, {r.null_auroc_crowding_max:.3f}]"
        print(f"  {int(r.k):>5}{int(r.tokens):>14,}{r.real_auroc_crowding:>12.3f}"
              f"{r.null_auroc_crowding_mean:>11.3f}{rng_s:>18}"
              f"{r.real_auroc_crowding - r.null_auroc_crowding_mean:>+9.3f}")

    print("\n  How to read the surplus (real minus null):")
    print("    ~0     the statistic measures sphere density; the geometry result is an")
    print("           artifact and the full model should be reported without it.")
    print("    large  crowding tracks genuine cross-seed correspondence, and the +0.211 it")
    print("           contributes over the frequency baseline stands.")
    print(f"\n  Self-duplicate rate inside the anchor (nearest neighbour >= {THETA}): "
          f"{df.self_duplicate_rate.mean():.1%} mean. A high value means many latents have a")
    print("  near-copy at home, which makes matching easy for uninteresting reasons.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, action="append", default=None,
                    help="sparsity arm (repeatable); default 64")
    ap.add_argument("--tokens", type=int, action="append", default=None,
                    help="training budget (repeatable); default the largest available")
    ap.add_argument("--anchor", type=int, default=None, help="anchor seed; default the first")
    ap.add_argument("--rotations", type=int, default=10,
                    help="rotations to average the null over (default 10)")
    ap.add_argument("--bands", action="store_true",
                    help="also run the frequency-band control (needs budget_transfer stats)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the rotations")
    args = ap.parse_args()

    found = discover(args.repo, args.prefix)
    if not found:
        raise SystemExit(f"No checkpoints matching {args.prefix} in {args.repo}.")
    cells, seeds, _ = complete_cells(found)
    anchor = args.anchor if args.anchor is not None else seeds[0]
    ks = args.k or [64]
    budgets = args.tokens or [max(t for _, t in cells)]

    print(f"Repo {args.repo}  prefix {args.prefix}")
    print(f"Seeds {seeds}, anchor {anchor}, theta {THETA}, "
          f"{args.rotations} rotations per cell")

    rng = np.random.default_rng(args.seed)
    rows = []
    for k in ks:
        for tokens in budgets:
            if (k, tokens) not in cells:
                print(f"  skipping k={k} @ {tokens:,}: not a complete cell")
                continue
            print(f"  k={k} @ {tokens:,} ...")
            rows.append(run_cell(found, k, tokens, seeds, anchor,
                                 args.rotations, rng, args.repo))

    if not rows:
        raise SystemExit("Nothing to analyse.")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "rotation_null.csv", index=False)
    report(df)

    if args.bands:
        band_frames = [
            frequency_bands(found, k, tokens, anchor, seeds, repo=args.repo)
            for k in ks for tokens in budgets if (k, tokens) in cells
        ]
        band_frames = [b for b in band_frames if b is not None and not b.empty]
        if band_frames:
            pd.concat(band_frames).to_csv(RESULTS_DIR / "frequency_bands.csv", index=False)

    (RESULTS_DIR / "config.json").write_text(json.dumps({
        "repo": args.repo, "prefix": args.prefix, "seeds": seeds, "anchor": anchor,
        "k_values": ks, "budgets": budgets, "rotations": args.rotations,
        "theta": THETA, "epsilon": EPSILON, "k_nn": K_NN, "rng_seed": args.seed,
    }, indent=2))
    print(f"\nWrote {RESULTS_DIR}/rotation_null.csv and config.json")


if __name__ == "__main__":
    main()
