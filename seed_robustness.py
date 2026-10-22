# -*- coding: utf-8 -*-
"""Seed-count robustness: does the headline survive five seeds instead of three?

Every reviewer flagged that stability labels come from three seeds. With an anchor and two
partners, "stable" means matched in both partners, and p_hat can only be 0, 0.5 or 1. This
script re-labels the same anchor against one to four partners and re-scores the classifier at
each, so the row that matters -- 3 seeds against 5 -- comes with the attrition curve between.

It is meant for a run that trained all five seeds fresh, together, on one stream (k=64,
prefix topk64_x16_tied_auxk), so the 3-vs-5 comparison never mixes checkpoints from different
environments. The partner order is fixed (256, 1024, 137, 512), so the two-partner row is the
paper's original seed set and the four-partner row is the full five.

Predictors come from the run's own exported table (feature_stability_k{k}.csv, anchor = the
first seed) or, for a dry run against an older prefix, from a cached statistics npz. Labels are
re-derived here from the decoders on the Hub, by the sweep's rule. When the table is used, the
four-partner labels must reproduce the table's own is_stable / is_unstable, or the script stops:
that is the check that both code paths label identically.

A second table compares the fresh seeds 42/256/1024 against the original sweep's checkpoints at
the same budget (--orig-prefix). Same-index cosine is reported, but GPU nondeterminism can
compound over hundreds of millions of tokens, so the informative numbers are the best-match
cosine and the theta match rate, read against the cross-seed rate between the originals.

Never reads or writes cache/stats_*: cross_scale_transfer.py names those without a prefix, so
fresh seeds scored through it would collide with the original ones.

Decoders only plus one table: no activations, no GPU, minutes.

Usage:
    python seed_robustness.py                                   # the fresh 5-seed run
    python seed_robustness.py --csv outputs/.../feature_stability_k64.csv
    python seed_robustness.py --prefix topk64-128-256_x16_tied_auxk \\
        --stats-npz cache/stats_pythia-70m-deduped_k64_seed42_t500000000_e40.npz
                                                                # dry run, original 3 seeds
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rotation_null import best_cosine, p_hat
from sae_stats import (
    DEFAULT_PREFIX,
    DEFAULT_REPO,
    EPSILON,
    MIN_FIRINGS,
    THETA,
    build_predictors,
    cv_auroc,
    discover,
    load_decoder,
)

RESULTS_DIR = Path(os.environ.get("SAE_RESULTS_DIR") or "outputs/seed_robustness")
FRESH_PREFIX = "topk64_x16_tied_auxk"
PARTNER_ORDER = (256, 1024, 137, 512)
ORIGINAL_SEEDS = (42, 256, 1024)
# The paper's budget-trajectory row at 500M (anchor 42, three seeds), for the reproduction line.
PAPER_500M = {"auroc": 0.853, "auroc_freq": 0.592}
# Looked up by name, so a reordering of build_predictors cannot swap baseline and crowding.
FREQ_PREDICTOR = "Activation Freq (log)"
CROWDING_PREDICTOR = "Geometric Isolation"
# Cosines this close to theta can land either side of it between two fp32 code paths.
THRESHOLD_TOLERANCE = 1e-5


def load_predictor_table(args, k):
    """Per-latent statistics of the anchor: (stats dict, exported frame or None)."""
    if args.stats_npz:
        with np.load(args.stats_npz) as z:
            return {key: z[key] for key in z.files}, None
    csv = args.csv or f"results/{args.prefix}/feature_stability_k{k}.csv"
    if Path(csv).exists():
        df = pd.read_csv(csv)
    else:
        from huggingface_hub import hf_hub_download
        df = pd.read_csv(hf_hub_download(args.repo, csv, repo_type="model"))
    return {col: df[col].to_numpy() for col in df.columns}, df


def labels(best, partners):
    p = p_hat([best[s] for s in partners])
    return p >= (1 - EPSILON), p <= EPSILON


def score_rows(stats, best, partners_present):
    names, columns = zip(*build_predictors(stats))
    X = np.column_stack(columns)
    finite = np.isfinite(X).all(axis=1)
    live = stats["firing_counts"] >= MIN_FIRINGS
    i_freq, i_crowd = names.index(FREQ_PREDICTOR), names.index(CROWDING_PREDICTOR)

    rows = []
    for n in range(1, len(partners_present) + 1):
        partners = partners_present[:n]
        stable, unstable = labels(best, partners)
        mask = (stable | unstable) & live & finite
        full = cv_auroc(X, stable, mask)
        freq = cv_auroc(X[:, i_freq], stable, mask)
        rows.append(dict(
            n_seeds=n + 1, partners=",".join(map(str, partners)),
            n_labelled=int(mask.sum()), stable_frac=float(stable.mean()),
            labelled_stable_frac=float(stable[mask].mean()) if mask.any() else float("nan"),
            auroc_full=full, auroc_freq=freq, delta=full - freq,
            auroc_crowding=cv_auroc(X[:, i_crowd], stable, mask),
        ))
    return pd.DataFrame(rows)


def check_against_table(df, best, partners):
    """The four-partner labels must match the table the sweep exported for the same run."""
    stable, unstable = labels(best, partners)
    near = np.any([np.abs(best[s] - THETA) < THRESHOLD_TOLERANCE for s in partners], axis=0)
    bad = ((stable != df["is_stable"].to_numpy(bool))
           | (unstable != df["is_unstable"].to_numpy(bool))) & ~near
    if bad.any():
        raise SystemExit(
            f"{int(bad.sum())} latents labelled differently here than in the exported table "
            f"(e.g. feature {int(np.flatnonzero(bad)[0])}). The two code paths disagree; do "
            f"not report these numbers until that is explained."
        )
    print(f"  labels reproduce the exported table ({int(near.sum())} latents within "
          f"{THRESHOLD_TOLERANCE:g} of theta, ignored)")


def reproduction(args, found, k, tokens):
    """Fresh seeds against the original sweep's checkpoints of the same seed and budget."""
    orig = discover(args.repo, args.orig_prefix)
    if not all((k, s, tokens) in orig for s in ORIGINAL_SEEDS):
        print(f"\n  reproduction skipped: {args.orig_prefix} lacks k={k} @ {tokens:,} for "
              f"{list(ORIGINAL_SEEDS)}")
        return None

    load_orig = {s: load_decoder(orig, k, s, tokens, repo=args.repo) for s in ORIGINAL_SEEDS}
    cross = best_cosine(load_orig[ORIGINAL_SEEDS[0]], load_orig[ORIGINAL_SEEDS[1]])
    rows = []
    for s in ORIGINAL_SEEDS:
        if (k, s, tokens) not in found:
            continue
        fresh = load_decoder(found, k, s, tokens, repo=args.repo)
        same_index = (fresh * load_orig[s]).sum(dim=1).numpy()
        best = best_cosine(fresh, load_orig[s])
        rows.append(dict(
            seed=s, same_index_median=float(np.median(same_index)),
            same_index_p5=float(np.percentile(same_index, 5)),
            best_match_median=float(np.median(best)),
            matched_at_theta=float((best >= THETA).mean()),
            cross_seed_matched_at_theta=float((cross >= THETA).mean()),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default=FRESH_PREFIX)
    ap.add_argument("--orig-prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=500_000_000)
    ap.add_argument("--anchor", type=int, default=42)
    ap.add_argument("--csv", default=None,
                    help="exported feature table: local path or path in the Hub repo "
                         "(default results/<prefix>/feature_stability_k<k>.csv)")
    ap.add_argument("--stats-npz", default=None,
                    help="cached statistics npz instead of a table (dry run on an old prefix)")
    args = ap.parse_args()
    k, tokens = args.k, args.tokens

    found = discover(args.repo, args.prefix)
    if (k, args.anchor, tokens) not in found:
        raise SystemExit(f"No anchor checkpoint k={k} seed={args.anchor} @ {tokens:,} under "
                         f"{args.prefix} in {args.repo}.")
    partners = [s for s in PARTNER_ORDER if (k, s, tokens) in found]
    print(f"Repo {args.repo}  prefix {args.prefix}  k={k} @ {tokens:,}  anchor {args.anchor}")
    print(f"Partners in order: {partners}  theta {THETA}  epsilon {EPSILON}  "
          f"floor {MIN_FIRINGS}")

    stats, table = load_predictor_table(args, k)
    A = load_decoder(found, k, args.anchor, tokens, repo=args.repo)
    if len(stats["firing_counts"]) != A.shape[0]:
        raise SystemExit("Predictor table and anchor decoder disagree on the number of latents.")
    best = {s: best_cosine(A, load_decoder(found, k, s, tokens, repo=args.repo))
            for s in partners}

    if table is not None and "is_stable" in table and len(partners) == len(PARTNER_ORDER):
        check_against_table(table, best, partners)

    summary = score_rows(stats, best, partners)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.prefix == DEFAULT_PREFIX and tokens == 500_000_000:
        print(f"\n  paper's 500M row: AUROC {PAPER_500M['auroc']:.3f}, "
              f"frequency {PAPER_500M['auroc_freq']:.3f} (three seeds)")

    repro = reproduction(args, found, k, tokens) if args.prefix != args.orig_prefix else None
    if repro is not None and not repro.empty:
        print("\nFresh seeds against the original sweep's checkpoints:")
        print(repro.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"  three-seed AUROC here {summary.loc[summary.n_seeds == 3, 'auroc_full'].item():.3f}"
              f" vs the paper's {PAPER_500M['auroc']:.3f} at 500M")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    if repro is not None:
        repro.to_csv(RESULTS_DIR / "reproduction.csv", index=False)
    (RESULTS_DIR / "config.json").write_text(json.dumps({
        "repo": args.repo, "prefix": args.prefix, "orig_prefix": args.orig_prefix,
        "k": k, "tokens": tokens, "anchor": args.anchor, "partners": partners,
        "theta": THETA, "epsilon": EPSILON, "min_firings": MIN_FIRINGS,
        "predictors": args.stats_npz or args.csv or f"results/{args.prefix}/"
                                                   f"feature_stability_k{k}.csv",
    }, indent=2))
    print(f"\nWrote {RESULTS_DIR}/summary.csv"
          f"{', reproduction.csv' if repro is not None else ''} and config.json")


if __name__ == "__main__":
    main()
