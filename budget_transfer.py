# -*- coding: utf-8 -*-
"""Cheap-to-expensive transfer and the budget trajectory, from checkpoints on the Hub.

Implements plan-cheap-to-expensive.md. Two results:

  1. HEADLINE. Fit the stability classifier on a cheap SAE (50M tokens, seed 42) and score an
     expensive one of a DIFFERENT seed (1B tokens, seed 256), with the labels re-derived
     anchored on the target. This is the first test that crosses seed and budget at once, and
     it is the claim a practitioner cares about: fit once on a throwaway SAE, reuse forever.

  2. TRAJECTORY. AUROC against training budget, labels recomputed at every budget. Flat means
     the headline 0.868 is not an artifact of where training stopped.

No training. Runs on CPU; uses the GPU automatically if one is present.

Usage (each stage caches, so any stage can be re-run or resumed):

    python budget_transfer.py validate     # do this FIRST: does our local pipeline reproduce
                                           # the numbers the sweep exported?
    python budget_transfer.py stats        # the expensive stage: statistics per checkpoint
    python budget_transfer.py analyze      # labels, transfer matrix, trajectory (fast)
    python budget_transfer.py all          # stats then analyze

    python budget_transfer.py stats --k 64            # one arm at a time
    SAE_EVAL_BATCHES=12 python budget_transfer.py all # ~3x faster, ~393K eval tokens

Environment:
    SAE_HF_REPO        default deenais/sae-stability-pythia70m
    SAE_EVAL_BATCHES   default 40 (40 x 256 x 128 = 1,310,720 tokens, matching the sweep)
    SAE_CACHE_DIR      default ./cache
    SAE_RESULTS_DIR    default ./outputs/budget_transfer
    SAE_MIN_FIRINGS    default 100, read by sae_stats
    HF_TOKEN           only needed if the repo is private
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from eval_activations import BATCH_SIZE, N_EVAL_BATCHES, SEQ_LEN, build_eval_activations
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

REPO = DEFAULT_REPO
PREFIX = DEFAULT_PREFIX

# The eval set has to be the sweep's eval set, or a local number cannot be compared with an
# exported one. These are the sweep's values; `validate` is what proves they still reproduce.
MODEL_NAME = "pythia-70m-deduped"
HOOK_POINT = "blocks.3.hook_resid_post"

CACHE_DIR = Path(os.environ.get("SAE_CACHE_DIR") or "cache")
RESULTS_DIR = Path(os.environ.get("SAE_RESULTS_DIR") or "outputs/budget_transfer")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The headline cell: cheap source, expensive target, different seed.
CHEAP_TOKENS = 50_000_000
EXPENSIVE_TOKENS = 1_000_000_000
SOURCE_SEED = 42
TARGET_SEED = 256


def eval_activations():
    """The sweep's eval set: Pythia-70m at blocks.3, 40 x 256 x 128 tokens."""
    return build_eval_activations(MODEL_NAME, HOOK_POINT, DEVICE, cache_dir=CACHE_DIR)


# --------------------------------------------------------------------------------------
# Stage: per-checkpoint statistics
# --------------------------------------------------------------------------------------

STAT_KEYS = ("activation_freq", "mean_activation", "firing_counts", "geometric_isolation",
             "recon_contribution", "recon_contribution_uncond", "encoder_norm", "encoder_bias",
             "decoder_norm")


def stats_path(k, seed, tokens):
    return CACHE_DIR / f"stats_k{k}_seed{seed}_t{tokens}_e{N_EVAL_BATCHES}.npz"


def compute_stats_grid(found, k_filter=None):
    """Statistics for every checkpoint, cached one file per checkpoint so an interruption
    resumes instead of restarting. This is the expensive stage."""
    cells, seeds, skipped = complete_cells(found)
    for k, tokens, missing in skipped:
        print(f"  skipping k={k} @ {tokens:,}: missing seed(s) {missing}")

    todo = [(k, s, t) for (k, t) in cells for s in seeds
            if (k_filter is None or k == k_filter) and not stats_path(k, s, t).exists()]
    if not todo:
        print("All statistics already cached.")
        return
    print(f"{len(todo)} checkpoint(s) to process on {DEVICE}")

    activations = eval_activations()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for i, (k, seed, tokens) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] k={k} seed={seed} tokens={tokens:,}")
        sae = load_sae(found, k, seed, tokens, device=DEVICE)
        stats = compute_single_run_statistics(
            sae, activations, DEVICE, label=f"k={k} seed {seed} @ {tokens:,}"
        )
        np.savez(stats_path(k, seed, tokens), **{key: stats[key] for key in STAT_KEYS})
        del sae
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nStatistics cached in {CACHE_DIR}")


def load_stats(k, seed, tokens):
    with np.load(stats_path(k, seed, tokens)) as z:
        return {key: z[key] for key in z.files}


# --------------------------------------------------------------------------------------
# Stage: labels
# --------------------------------------------------------------------------------------


def labels_for(found, k, tokens, anchor, seeds):
    """Ground truth for ONE dictionary: p_hat for the anchor seed against the other seeds at
    the same budget, using the sweep's rule (many-to-one argmax, theta, endpoint binarization).

    Weight-only, so this needs no activations. The anchor must come first in the dict --
    compute_reappearance_probability takes saes[seeds[0]] as the anchor.

    Returns (p_hat, best cosine in the FIRST comparison). The second value follows the sweep's
    `match_similarity` column, which is similarities[0] rather than a mean over comparisons.
    """
    ordered = [anchor] + [s for s in seeds if s != anchor]
    saes = {s: load_sae(found, k, s, tokens, device=DEVICE) for s in ordered}
    probs, matching = compute_reappearance_probability(saes, theta=THETA)
    del saes
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return probs, matching["similarities"][0]


def cell_arrays(found, k, seed, tokens, seeds, label_cache):
    """Predictor matrix, binary truth and the labelled mask for one (k, seed, budget) cell.

    The firing floor is re-derived from THIS dictionary's counts: which latents are
    under-measured is a property of the SAE being scored, not of the one trained on.
    """
    key = (k, tokens, seed)
    if key not in label_cache:
        label_cache[key] = labels_for(found, k, tokens, seed, seeds)
    probs, _ = label_cache[key]

    stats = load_stats(k, seed, tokens)
    X = np.column_stack([v for _, v in build_predictors(stats)])
    stable = probs >= (1 - EPSILON)
    unstable = probs <= EPSILON
    live = stats["firing_counts"] >= MIN_FIRINGS
    mask = (stable | unstable) & live & np.isfinite(X).all(axis=1)
    return X, stable, mask, live, stats


# --------------------------------------------------------------------------------------
# Stage: transfer matrix and trajectory
# --------------------------------------------------------------------------------------


def fit(X, y):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf.fit(scaler.transform(X), y)
    return clf, scaler


def analyze(found, k_filter=None):
    cells, seeds, _ = complete_cells(found)
    ks = sorted({k for k, _ in cells}) if k_filter is None else [k_filter]
    budgets = sorted({t for _, t in cells})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    label_cache = {}
    rows, traj = [], []

    for k in ks:
        # Analyse whatever is cached rather than demanding the whole grid, so a partial `stats`
        # run still produces the headline cell. Missing cells are named, not silently dropped.
        wanted = [(s, t) for t in budgets for s in seeds if (k, t) in cells]
        arms = [a for a in wanted if stats_path(k, a[0], a[1]).exists()]
        missing = [a for a in wanted if a not in arms]
        if missing:
            print(f"k={k}: no cached statistics for {len(missing)} of {len(wanted)} cells "
                  f"(run `stats` to fill): "
                  + ", ".join(f"seed {s} @ {t:,}" for s, t in missing[:6])
                  + (" ..." if len(missing) > 6 else ""))
        if not arms:
            continue

        print(f"\n=== k={k}: {len(arms)} cells ===")
        cache = {}
        for seed, tokens in arms:
            X, stable, mask, live, stats = cell_arrays(found, k, seed, tokens, seeds, label_cache)
            cache[(seed, tokens)] = (X, stable, mask)
            if int(mask.sum()) < 20 or len(np.unique(stable[mask])) < 2:
                print(f"  k={k} seed={seed} @ {tokens:,}: too few labelled latents, skipped")
                continue

            own = cv_auroc(X, stable, mask)
            freq = cv_auroc(X[:, 0], stable, mask)
            traj.append(dict(
                k=k, tokens=tokens, anchor_seed=seed, n_labelled=int(mask.sum()),
                stable_frac=float(stable[mask].mean()), live_frac=float(live.mean()),
                auroc_full=own, auroc_freq_only=freq, delta_over_freq=own - freq,
            ))
            print(f"  seed={seed} @ {tokens:>13,}: n={int(mask.sum()):>5} "
                  f"stable={stable[mask].mean():>5.1%} AUROC={own:.3f} (freq {freq:.3f})")

        # Fit once per source, score against every target.
        fitted = {}
        for src in arms:
            Xs, ys, ms = cache[src]
            if int(ms.sum()) < 20 or len(np.unique(ys[ms])) < 2:
                continue
            fitted[src] = fit(Xs[ms], ys[ms].astype(int))

        for src, (clf, scaler) in fitted.items():
            for tgt in arms:
                Xt, yt, mt = cache[tgt]
                if int(mt.sum()) < 20 or len(np.unique(yt[mt])) < 2:
                    continue
                truth = yt[mt].astype(int)
                own = cv_auroc(Xt, yt, mt)
                freq = cv_auroc(Xt[:, 0], yt, mt)
                # Two conventions, answering different questions. Reusing the source scaler
                # also requires the raw scales to agree; refitting asks only whether the
                # learned relationship transfers, and is what a practitioner would do.
                scored = {
                    "source_scaler": scaler.transform(Xt[mt]),
                    "target_scaler": StandardScaler().fit(Xt[mt]).transform(Xt[mt]),
                }
                for convention, Xs in scored.items():
                    rows.append(dict(
                        k=k, convention=convention,
                        src_seed=src[0], src_tokens=src[1],
                        tgt_seed=tgt[0], tgt_tokens=tgt[1],
                        same_cell=(src == tgt),
                        auroc=roc_auc_score(truth, clf.predict_proba(Xs)[:, 1]),
                        target_ceiling=own, target_freq_only=freq,
                        n_target=int(mt.sum()), target_stable_frac=float(truth.mean()),
                    ))

    if not rows:
        print("\nNothing analysed.")
        return

    matrix = pd.DataFrame(rows)
    trajectory = pd.DataFrame(traj)
    matrix.to_csv(RESULTS_DIR / "budget_transfer_matrix.csv", index=False)
    trajectory.to_csv(RESULTS_DIR / "budget_trajectory.csv", index=False)
    (RESULTS_DIR / "config.json").write_text(json.dumps({
        "repo": REPO, "prefix": PREFIX, "eval_batches": N_EVAL_BATCHES,
        "eval_tokens": N_EVAL_BATCHES * BATCH_SIZE * SEQ_LEN,
        "theta": THETA, "epsilon": EPSILON, "min_firings": MIN_FIRINGS,
        "device": DEVICE, "seeds": seeds, "budgets": budgets, "k_values": ks,
    }, indent=2))

    report(matrix, trajectory)
    plot_trajectory(trajectory)
    print(f"\nWrote {RESULTS_DIR}/budget_transfer_matrix.csv, budget_trajectory.csv, "
          f"budget_trajectory.png, config.json")


def report(matrix, trajectory):
    """Print the two pre-committed results, and only those, first."""
    print(f"\n{'=' * 78}\nHEADLINE: cheap source -> expensive target, different seed\n{'=' * 78}")
    head = matrix[
        (matrix.src_seed == SOURCE_SEED) & (matrix.src_tokens == CHEAP_TOKENS)
        & (matrix.tgt_seed == TARGET_SEED) & (matrix.tgt_tokens == EXPENSIVE_TOKENS)
    ]
    if head.empty:
        print("  The headline cell is not in this run (filtered k, or missing checkpoints).")
    else:
        for _, r in head.iterrows():
            gap = r.auroc - r.target_ceiling
            print(f"  k={int(r.k):<4} {r.convention:<14}: {r.auroc:.3f}  "
                  f"(target ceiling {r.target_ceiling:.3f}, gap {gap:+.3f}; "
                  f"target frequency-only {r.target_freq_only:.3f})")
        print("\n  Fitted on seed 42 @ 50M tokens, scored on seed 256 @ 1B tokens with labels\n"
              "  re-derived anchored on seed 256. A small gap to the ceiling means a throwaway\n"
              "  SAE is enough to fit the classifier; beating the target's frequency-only\n"
              "  baseline is the minimum for the result to be worth anything.")

    print(f"\n{'=' * 78}\nTRAJECTORY: does predictability depend on the training budget?\n{'=' * 78}")
    for k, g in trajectory.groupby("k"):
        print(f"\n  k={k}")
        print(f"    {'tokens':>13} {'n':>6} {'stable':>7} {'live':>6} {'AUROC':>6} "
              f"{'freq':>6} {'delta':>7}")
        for _, r in g[g.anchor_seed == SOURCE_SEED].sort_values("tokens").iterrows():
            print(f"    {int(r.tokens):>13,} {int(r.n_labelled):>6} {r.stable_frac:>6.1%} "
                  f"{r.live_frac:>5.1%} {r.auroc_full:>6.3f} {r.auroc_freq_only:>6.3f} "
                  f"{r.delta_over_freq:>+7.3f}")

    print("\n  Checkpoints along one run are the same SAE at different moments, so these points\n"
          "  are autocorrelated. Read the SHAPE. Do not compute an error bar over them: six\n"
          "  budgets x three seeds is not eighteen independent runs.")


def plot_trajectory(trajectory):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the trajectory plot.")
        return

    ks = sorted(trajectory.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), squeeze=False)
    for ax, k in zip(axes[0], ks):
        g = trajectory[trajectory.k == k]
        for seed, gs in g.groupby("anchor_seed"):
            gs = gs.sort_values("tokens")
            ax.plot(gs.tokens, gs.auroc_full, "o-", label=f"seed {seed}, full")
            ax.plot(gs.tokens, gs.auroc_freq_only, "o--", alpha=0.4,
                    label=f"seed {seed}, frequency only")
        ax.set_xscale("log")
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("training tokens")
        ax.set_ylabel("AUROC")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Stability predictability vs training budget (labels recomputed per budget)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "budget_trajectory.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Validation gate
# --------------------------------------------------------------------------------------

# Columns the sweep exported, mapped to the statistic that produces them.
VALIDATE_COLUMNS = {
    "activation_freq": "activation_freq",
    "firing_counts": "firing_counts",
    "geometric_isolation": "geometric_isolation",
    "recon_contribution": "recon_contribution",
    "mean_activation": "mean_activation",
    "encoder_norm": "encoder_norm",
    "encoder_bias": "encoder_bias",
}


def validate(found, k=64, seed=SOURCE_SEED, tokens=EXPENSIVE_TOKENS):
    """Recompute one arm locally and diff it against the CSV the sweep exported.

    This decides whether local numbers may be quoted alongside the hub's. It is not a unit
    test of the statistics -- those are imported from the same module the sweep used -- it is
    a test of whether the eval token stream regenerated here is the stream the sweep saw.
    """
    print(f"Validating k={k} seed={seed} @ {tokens:,} against the exported CSV")
    remote = hf_hub_download(
        REPO, f"results/{PREFIX}/at1B/feature_stability_k{k}.csv", repo_type="model"
    )
    ref = pd.read_csv(remote)

    if not stats_path(k, seed, tokens).exists():
        activations = eval_activations()
        sae = load_sae(found, k, seed, tokens, device=DEVICE)
        stats = compute_single_run_statistics(sae, activations, DEVICE, label="validation")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(stats_path(k, seed, tokens), **{key: stats[key] for key in STAT_KEYS})
        del sae
    stats = load_stats(k, seed, tokens)

    if N_EVAL_BATCHES != 40:
        print(f"  NOTE: SAE_EVAL_BATCHES={N_EVAL_BATCHES}, but the sweep used 40. Frequency and\n"
              f"  activation statistics are estimated from a different number of tokens, so a\n"
              f"  mismatch here is expected and uninformative. Re-run with 40 to validate.")

    print(f"\n  {'column':<22}{'max |diff|':>13}{'rel':>10}{'corr':>9}")
    worst_rel = 0.0
    for col, key in VALIDATE_COLUMNS.items():
        a, b = ref[col].to_numpy(dtype=float), np.asarray(stats[key], dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a[both] - b[both])
        scale = max(np.abs(a[both]).max(), 1e-12)
        rel = diff.max() / scale
        corr = np.corrcoef(a[both], b[both])[0, 1] if both.sum() > 2 else float("nan")
        worst_rel = max(worst_rel, rel)
        print(f"  {col:<22}{diff.max():>13.3e}{rel:>10.2e}{corr:>9.5f}")

    # The labels are weight-only, so they must reproduce regardless of the token stream. A
    # mismatch here means something is wrong with the checkpoints or the matching, not the data.
    seeds = sorted({s for (_, s, _) in found})
    probs, best = labels_for(found, k, tokens, seed, seeds)
    label_diff = np.abs(ref["reappearance_prob"].to_numpy(dtype=float) - probs).max()
    match_diff = np.abs(ref["match_similarity"].to_numpy(dtype=float) - best).max()
    print(f"  {'reappearance_prob':<22}{label_diff:>13.3e}{'(weight-only)':>19}")
    print(f"  {'match_similarity':<22}{match_diff:>13.3e}{'(weight-only)':>19}")
    label_diff = max(label_diff, match_diff)

    print()
    if label_diff > 1e-6:
        print("  FAIL: the weight-only labels do not reproduce. This is not a data-stream\n"
              "  problem -- check that the checkpoints and the matching rule are the ones the\n"
              "  sweep used. Stop here.")
    elif worst_rel < 1e-4:
        print("  PASS: the local pipeline reproduces the exported statistics. Local AUROCs are\n"
              "  directly comparable to the sweep's (0.868 ceiling, 0.872 held-out transfer).")
    elif worst_rel < 1e-2:
        print("  MARGINAL: small differences, most likely float accumulation order or a device\n"
              "  difference. Comparable for reporting, but say so in the write-up.")
    else:
        print("  MISMATCH: the local eval stream is not the sweep's, most likely because the\n"
              "  dataset snapshot moved. Do NOT quote local numbers against 0.868 or 0.872.\n"
              "  Restrict every claim to comparisons WITHIN this local grid, where all arms\n"
              "  are scored on the same activations. Record this in the write-up.")


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["validate", "stats", "analyze", "all", "discover"])
    ap.add_argument("--k", type=int, default=None, help="restrict to one sparsity arm")
    args = ap.parse_args()

    print(f"Repo {REPO}  device {DEVICE}  eval batches {N_EVAL_BATCHES} "
          f"({N_EVAL_BATCHES * BATCH_SIZE * SEQ_LEN:,} tokens)  floor {MIN_FIRINGS}")
    found = discover()
    if not found:
        raise SystemExit(
            f"No checkpoints matching {PREFIX} in {REPO}. For a private repo, set HF_TOKEN."
        )
    cells, seeds, skipped = complete_cells(found)
    print(f"Found {len(found)} checkpoints: seeds {seeds}, "
          f"{len(cells)} complete (k, budget) cells")

    if args.stage == "discover":
        for k, tokens in cells:
            print(f"  k={k:<4} {tokens:>13,}")
        for k, tokens, missing in skipped:
            print(f"  INCOMPLETE k={k:<4} {tokens:>13,} missing {missing}")
        return

    if args.stage == "validate":
        validate(found)
        return
    if args.stage in ("stats", "all"):
        compute_stats_grid(found, k_filter=args.k)
    if args.stage in ("analyze", "all"):
        analyze(found, k_filter=args.k)


if __name__ == "__main__":
    main()
