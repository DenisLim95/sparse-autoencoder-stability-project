# -*- coding: utf-8 -*-
"""
analyze_stability_by_tokens.py

Loads the SAE checkpoints saved by prelim_experiments.py, grouped by token count
across all seeds, and computes feature stability separately at each token count,
so you can actually see how stability changes as training progresses. This is the
piece the main training script does not do on its own: it only ever analyzes the
last checkpoint (the final trained_saes dict), never compares across checkpoints.
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple

CHECKPOINT_DIR = "outputs/checkpoints"
SEEDS = [42, 137, 256, 512, 1024]
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


def find_available_token_counts(checkpoint_dir: str, seeds: list) -> list:
    """Scan checkpoint filenames to find which token counts have ALL seeds present."""
    ckpt_dir = Path(checkpoint_dir)
    pattern = re.compile(r"seed(\d+)_tokens(\d+)\.pt")
    tokens_by_seed = {seed: set() for seed in seeds}

    for f in ckpt_dir.glob("seed*_tokens*.pt"):
        m = pattern.match(f.name)
        if m:
            seed, tokens = int(m.group(1)), int(m.group(2))
            if seed in tokens_by_seed:
                tokens_by_seed[seed].add(tokens)

    # Only keep token counts present for EVERY seed
    common = set.intersection(*tokens_by_seed.values()) if all(tokens_by_seed.values()) else set()
    return sorted(common)


def load_saes_at_token_count(checkpoint_dir: str, seeds: list, token_count: int) -> Dict[int, SparseAutoencoder]:
    saes = {}
    for seed in seeds:
        path = Path(checkpoint_dir) / f"seed{seed}_tokens{token_count}.pt"
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        sae = SparseAutoencoder(cfg["d_model"], cfg["n_features"], seed=seed)
        sae.load_state_dict(ckpt["model_state_dict"])
        saes[seed] = sae
    return saes


def main():
    available_token_counts = find_available_token_counts(CHECKPOINT_DIR, SEEDS)

    if not available_token_counts:
        print(f"No token count yet has checkpoints for all {len(SEEDS)} seeds ({SEEDS}).")
        print("Re-run this script once more seeds/checkpoints have finished.")
        return

    print(f"Found complete checkpoints (all {len(SEEDS)} seeds) at: {available_token_counts}\n")

    results = []
    for token_count in available_token_counts:
        saes = load_saes_at_token_count(CHECKPOINT_DIR, SEEDS, token_count)
        p_hat = compute_reappearance_probability(saes, theta=THETA)

        stable = (p_hat >= (1 - EPSILON)).sum()
        unstable = (p_hat <= EPSILON).sum()
        discarded = len(p_hat) - stable - unstable

        results.append({
            "token_count": token_count,
            "n_features": len(p_hat),
            "mean_p_hat": p_hat.mean(),
            "stable": stable,
            "unstable": unstable,
            "discarded": discarded,
            "pct_stable": 100 * stable / len(p_hat),
        })

        print(f"--- {token_count:,} tokens ---")
        print(f"  Mean p_hat:  {p_hat.mean():.4f}")
        print(f"  Stable:      {stable} ({100 * stable / len(p_hat):.1f}%)")
        print(f"  Unstable:    {unstable} ({100 * unstable / len(p_hat):.1f}%)")
        print(f"  Discarded:   {discarded} ({100 * discarded / len(p_hat):.1f}%)")
        print()

    print("=" * 60)
    print("SUMMARY: stability vs. training token count")
    print("=" * 60)
    print(f"{'Tokens':>15} {'Mean p_hat':>12} {'% Stable':>10} {'% Unstable':>12}")
    for r in results:
        print(f"{r['token_count']:>15,} {r['mean_p_hat']:>12.4f} {r['pct_stable']:>9.1f}% {100 * r['unstable'] / r['n_features']:>11.1f}%")

    # Save results for later use (e.g. plotting, or feeding into the classifier notebook)
    Path("outputs").mkdir(exist_ok=True)
    np.savez(
        "outputs/stability_by_token_count.npz",
        token_counts=np.array([r["token_count"] for r in results]),
        mean_p_hat=np.array([r["mean_p_hat"] for r in results]),
        pct_stable=np.array([r["pct_stable"] for r in results]),
    )
    print("\nSaved summary to outputs/stability_by_token_count.npz")

    # Try to plot, if matplotlib is available and there's a display backend
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless-safe backend, saves to file instead of showing
        import matplotlib.pyplot as plt

        token_counts = [r["token_count"] for r in results]
        pct_stable = [r["pct_stable"] for r in results]
        mean_p_hat = [r["mean_p_hat"] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(token_counts, pct_stable, marker="o")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Training tokens")
        axes[0].set_ylabel("% features stable")
        axes[0].set_title("Stability vs. training scale")

        axes[1].plot(token_counts, mean_p_hat, marker="o", color="orange")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Training tokens")
        axes[1].set_ylabel("Mean reappearance probability")
        axes[1].set_title("Mean p_hat vs. training scale")

        plt.tight_layout()
        plt.savefig("outputs/stability_by_token_count.png", dpi=150)
        print("Saved plot to outputs/stability_by_token_count.png")
    except Exception as e:
        print(f"(Skipped plotting: {e})")


if __name__ == "__main__":
    main()
