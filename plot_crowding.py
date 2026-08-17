"""Figure: stability against neighbour crowding, banded by activation frequency.

The crowding result is the paper's most counterintuitive one and the ablation table states it
only as an AUROC. This plots the underlying relationship directly, which also makes the
frequency-decile control visible: if crowding separated stable from unstable features only
because it tracks firing rate, the heatmap would vary along the frequency axis and be flat
along the crowding axis. It is the other way round.

Weight-only for the crowding axis and for the labels, so the three anchor dictionaries can be
pooled without an activation pass: crowding is a decoder-geometry statistic and reappearance
probability is a decoder cosine between seeds. The frequency axis is the one quantity that does
need activations, so it is read from the exported per-feature CSV and is available for the
anchor seed alone.

    python plot_crowding.py --k 64 --tokens 1000000000

    SAE_FIGURE_DIR  where the png is written (default ./figures)
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sae_stats import DEFAULT_PREFIX, DEFAULT_REPO, EPSILON, THETA, discover, load_decoder

FIGURE_DIR = Path(os.environ.get("SAE_FIGURE_DIR") or "figures")
N_DECILES = 10
K_NN = 10


def crowding(decoder: torch.Tensor, k_nn: int = K_NN) -> np.ndarray:
    """Mean cosine to the k nearest neighbours in the same dictionary, per feature."""
    W = F.normalize(decoder, dim=1)
    sims = W @ W.T
    sims.fill_diagonal_(-2.0)  # exclude self, whose cosine is 1 by construction
    return sims.topk(k_nn, dim=1).values.mean(dim=1).numpy()


def reappearance(decoders: dict, anchor: int) -> np.ndarray:
    """p_hat for one anchor: fraction of the other seeds holding a match at theta."""
    A = F.normalize(decoders[anchor], dim=1)
    matched = [
        ((A @ F.normalize(decoders[s], dim=1).T).max(dim=1).values.numpy() >= THETA)
        for s in decoders
        if s != anchor
    ]
    return np.mean(matched, axis=0)


def decile_rate(values: np.ndarray, stable: np.ndarray) -> np.ndarray:
    """Share stable within each decile of `values`."""
    bins = pd.qcut(values, N_DECILES, labels=False, duplicates="drop")
    return pd.Series(stable).groupby(bins).mean().values * 100


def anchor_cells(decoders: dict) -> dict:
    """Crowding, the binary label and the labelled mask, for every anchor seed."""
    cells = {}
    for anchor in decoders:
        p = reappearance(decoders, anchor)
        labelled = (p >= 1 - EPSILON) | (p <= EPSILON)
        cells[anchor] = dict(crowding=crowding(decoders[anchor]),
                             stable=(p >= 1 - EPSILON), labelled=labelled)
    return cells


def load_frequency_table(repo: str, prefix: str, k: int) -> pd.DataFrame:
    """The exported per-feature table, for the two activation-derived columns."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, f"results/{prefix}/at1B/feature_stability_k{k}.csv",
                           repo_type="model")
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=1_000_000_000)
    ap.add_argument("--anchor", type=int, default=42,
                    help="anchor whose activation statistics band the heatmap")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = ap.parse_args()

    found = discover(args.repo, args.prefix)
    seeds = sorted({s for (k, s, t) in found if k == args.k and t == args.tokens})
    if args.anchor not in seeds:
        raise SystemExit(f"anchor {args.anchor} not among {seeds}")
    print(f"k={args.k} @ {args.tokens:,}, seeds {seeds}, theta={THETA}, eps={EPSILON}")

    decoders = {s: load_decoder(found, args.k, s, args.tokens, repo=args.repo) for s in seeds}
    cells = anchor_cells(decoders)

    table = load_frequency_table(args.repo, args.prefix, args.k)
    anchor = cells[args.anchor]
    # The heatmap uses the floor because it is scoring the same features the classifier does;
    # the pooled panel cannot, since firing counts exist only for the exported anchor.
    grid_mask = anchor["labelled"] & table.above_firing_floor.values
    frame = pd.DataFrame({
        "stable": anchor["stable"][grid_mask],
        "crowding_decile": pd.qcut(anchor["crowding"][grid_mask], N_DECILES,
                                   labels=False, duplicates="drop"),
        "frequency_decile": pd.qcut(table.activation_freq.values[grid_mask], N_DECILES,
                                    labels=False, duplicates="drop"),
    })
    grid = frame.pivot_table(index="crowding_decile", columns="frequency_decile",
                             values="stable", aggfunc="mean") * 100

    print("\n  stable share by crowding decile")
    for anchor_seed, cell in cells.items():
        m = cell["labelled"]
        rates = decile_rate(cell["crowding"][m], cell["stable"][m])
        print(f"    anchor {anchor_seed:>4} (n={int(m.sum())}, "
              f"{100 * cell['stable'][m].mean():.1f}% stable): "
              f"{rates[0]:.1f}% -> {rates[-1]:.1f}%")
    floored = decile_rate(anchor["crowding"][grid_mask], anchor["stable"][grid_mask])
    unfloored = decile_rate(anchor["crowding"][anchor["labelled"]],
                            anchor["stable"][anchor["labelled"]])
    print(f"    firing floor moves no decile by more than "
          f"{np.abs(floored - unfloored).max():.1f} points")
    freq_rates = decile_rate(table.activation_freq.values[grid_mask],
                             anchor["stable"][grid_mask])
    print(f"    activation frequency, same features: {freq_rates[0]:.1f}% -> "
          f"{freq_rates[-1]:.1f}% (base rate {100 * anchor['stable'][grid_mask].mean():.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    im = axes[0].imshow(grid.values, origin="lower", cmap="viridis", vmin=0, vmax=100,
                        aspect="auto")
    axes[0].set_xlabel("activation-frequency decile")
    axes[0].set_ylabel("neighbour-crowding decile")
    axes[0].set_title(f"(a) share stable, anchor seed {args.anchor}")
    axes[0].set_xticks(range(N_DECILES))
    axes[0].set_yticks(range(N_DECILES))
    axes[0].tick_params(labelsize=8)
    fig.colorbar(im, ax=axes[0], label="% stable")

    for anchor_seed, cell in cells.items():
        m = cell["labelled"]
        axes[1].plot(range(N_DECILES), decile_rate(cell["crowding"][m], cell["stable"][m]),
                     "o-", markersize=4, label=f"crowding, seed {anchor_seed}")
    axes[1].plot(range(N_DECILES), freq_rates, "d--", color="firebrick", markersize=4,
                 label=f"activation frequency, seed {args.anchor}")
    axes[1].axhline(100 * anchor["stable"][grid_mask].mean(), color="gray", ls=":", lw=1,
                    label="base rate")
    axes[1].set_xlabel("predictor decile (low to high)")
    axes[1].set_ylabel("% stable")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(range(N_DECILES))
    axes[1].tick_params(labelsize=8)
    axes[1].set_title("(b) stable share by decile")
    axes[1].legend(fontsize=7.5, loc="lower right")

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"crowding_vs_frequency_k{args.k}.png"
    fig.savefig(out, dpi=200)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
