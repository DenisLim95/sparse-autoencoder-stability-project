# -*- coding: utf-8 -*-
"""
analyze_stability_by_tokens.py

Loads the SAE checkpoints saved by prelim_experiments.py, grouped by (k, token
count) across available seeds, and computes feature stability separately at
each (k, token count) pair, so you can see how stability changes as training
progresses AND how it varies with the TopK sparsity level.

Updated for the TopK sweep: filenames now encode k (e.g.
seed42_k64_tokens100000000.pt), and stability comparisons are only made
within a fixed k — comparing across different k values would conflate
"instability" with "different sparsity level," which is exactly the
confound switching to TopK was meant to remove.
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple

CHECKPOINT_DIR = "."  # .pt files live next to this notebook
SEEDS = [42, 256, 1024]  # matches the actual sweep
KS = [64, 128, 256]
MIN_SEEDS = 2  # analyze a (k, token_count) pair if >= this many seeds exist
THETA = 0.7      # Gerasimov et al. decoder-only matching threshold
EPSILON = 0.05   # endpoint binarization


class SparseAutoencoder(nn.Module):
    """Must match the architecture in prelim_experiments.py exactly, since we're
    loading state dicts saved by that script."""

    def __init__(self, d_model: int, n_features: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.d_model = d_model
        self.n_features = n_features
        self.W_enc = nn.Parameter(torch.randn(d_model, n_features) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.randn(n_features, d_model) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x):
        # NOTE: this is unused by the stability analysis below (which only
        # reads W_dec), but if you reuse this class elsewhere for inference
        # under TopK, this ReLU-only encode() does NOT enforce the hard
        # top-k mask — you'd need to add that here to match training-time
        # behavior.
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec


def compute_decoder_similarity(sae1, sae2) -> torch.Tensor:
    W1 = F.normalize(sae1.W_dec.detach(), dim=1)
    W2 = F.normalize(sae2.W_dec.detach(), dim=1)
    return (W1 @ W2.T).cpu()


def compute_reappearance_probability(saes: Dict[int, SparseAutoencoder], theta: float) -> np.ndarray:
    """Decoder-only, many-to-one argmax matching (Gerasimov et al. Eq. 3-4)."""
    seeds = list(saes.keys())
    anchor_sae = saes[seeds[0]]
    n_features = anchor_sae.n_features
    reappearance_counts = np.zeros(n_features)

    for other_seed in seeds[1:]:
        sim_matrix = compute_decoder_similarity(anchor_sae, saes[other_seed])
        best_sim, _ = sim_matrix.max(dim=1)
        reappearance_counts += (best_sim.numpy() >= theta).astype(float)

    return reappearance_counts / (len(seeds) - 1)


def find_groups_with_seeds(
    checkpoint_dir: str, seeds: list, ks: list, min_seeds: int = 2
) -> Dict[Tuple[int, int], List[int]]:
    """Return {(k, token_count): [seeds present]} for groups with >= min_seeds checkpoints."""
    ckpt_dir = Path(checkpoint_dir)
    pattern = re.compile(r"seed(\d+)_k(\d+)_tokens(\d+)\.pt")
    seed_set = set(seeds)
    k_set = set(ks)
    seeds_by_group: Dict[Tuple[int, int], set] = {}

    for f in ckpt_dir.glob("seed*_k*_tokens*.pt"):
        m = pattern.match(f.name)
        if not m:
            continue
        seed, k, tokens = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if seed not in seed_set or k not in k_set:
            continue
        seeds_by_group.setdefault((k, tokens), set()).add(seed)

    return {
        g: sorted(s)
        for g, s in sorted(seeds_by_group.items())
        if len(s) >= min_seeds
    }


def load_saes_at_group(checkpoint_dir: str, seeds: list, k: int, token_count: int) -> Dict[int, SparseAutoencoder]:
    saes = {}
    for seed in seeds:
        path = Path(checkpoint_dir) / f"seed{seed}_k{k}_tokens{token_count}.pt"
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        sae = SparseAutoencoder(cfg["d_model"], cfg["n_features"], seed=seed)
        sae.load_state_dict(ckpt["model_state_dict"])
        saes[seed] = sae
    return saes


def main():
    available = find_groups_with_seeds(CHECKPOINT_DIR, SEEDS, KS, min_seeds=MIN_SEEDS)

    if not available:
        print(f"No (k, token_count) group has >= {MIN_SEEDS} seeds among {SEEDS} x k={KS}.")
        print("Re-run once more seeds/checkpoints have finished.")
        return

    print("Analyzing (k, token_count) groups (seeds used per group):")
    for (k, t), seeds_here in available.items():
        print(f"  k={k}, tokens={t:,}: {seeds_here}")
    print()

    results = []
    for (k, token_count), seeds_here in available.items():
        saes = load_saes_at_group(CHECKPOINT_DIR, seeds_here, k, token_count)
        p_hat = compute_reappearance_probability(saes, theta=THETA)

        stable = (p_hat >= (1 - EPSILON)).sum()
        unstable = (p_hat <= EPSILON).sum()
        discarded = len(p_hat) - stable - unstable

        results.append({
            "k": k,
            "token_count": token_count,
            "seeds": seeds_here,
            "n_seeds": len(seeds_here),
            "n_features": len(p_hat),
            "mean_p_hat": p_hat.mean(),
            "stable": stable,
            "unstable": unstable,
            "discarded": discarded,
            "pct_stable": 100 * stable / len(p_hat),
        })

        print(f"--- k={k}, {token_count:,} tokens (seeds={seeds_here}) ---")
        print(f"  Mean p_hat:  {p_hat.mean():.4f}")
        print(f"  Stable:      {stable} ({100 * stable / len(p_hat):.1f}%)")
        print(f"  Unstable:    {unstable} ({100 * unstable / len(p_hat):.1f}%)")
        print(f"  Discarded:   {discarded} ({100 * discarded / len(p_hat):.1f}%)")
        print()

    print("=" * 84)
    print("SUMMARY: stability vs. training token count, by k")
    print("=" * 84)
    print(f"{'k':>6} {'Tokens':>15} {'n_seeds':>8} {'Mean p_hat':>12} {'% Stable':>10} {'% Unstable':>12}")
    for r in results:
        print(
            f"{r['k']:>6} {r['token_count']:>15,} {r['n_seeds']:>8} {r['mean_p_hat']:>12.4f} "
            f"{r['pct_stable']:>9.1f}% {100 * r['unstable'] / r['n_features']:>11.1f}%"
        )

    # Save results for later use (e.g. plotting, or feeding into the classifier notebook)
    Path("outputs").mkdir(exist_ok=True)
    np.savez(
        "outputs/stability_by_token_count.npz",
        k=np.array([r["k"] for r in results]),
        token_counts=np.array([r["token_count"] for r in results]),
        n_seeds=np.array([r["n_seeds"] for r in results]),
        mean_p_hat=np.array([r["mean_p_hat"] for r in results]),
        pct_stable=np.array([r["pct_stable"] for r in results]),
    )
    print("\nSaved summary to outputs/stability_by_token_count.npz")

    # Try to plot, if matplotlib is available and there's a display backend
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe backend, saves to file instead of showing
        import matplotlib.pyplot as plt

        ks_present = sorted(set(r["k"] for r in results))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for k in ks_present:
            rows = [r for r in results if r["k"] == k]
            rows.sort(key=lambda r: r["token_count"])
            token_counts = [r["token_count"] for r in rows]
            pct_stable = [r["pct_stable"] for r in rows]
            mean_p_hat = [r["mean_p_hat"] for r in rows]

            axes[0].plot(token_counts, pct_stable, marker="o", label=f"k={k}")
            axes[1].plot(token_counts, mean_p_hat, marker="o", label=f"k={k}")

        axes[0].set_xscale("log")
        axes[0].set_xlabel("Training tokens")
        axes[0].set_ylabel("% features stable")
        axes[0].set_title("Stability vs. training scale, by k")
        axes[0].legend()

        axes[1].set_xscale("log")
        axes[1].set_xlabel("Training tokens")
        axes[1].set_ylabel("Mean reappearance probability")
        axes[1].set_title("Mean p_hat vs. training scale, by k")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("outputs/stability_by_token_count.png", dpi=150)
        print("Saved plot to outputs/stability_by_token_count.png")
    except Exception as e:
        print(f"(Skipped plotting: {e})")


if __name__ == "__main__":
    main()
