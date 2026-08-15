import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple

CHECKPOINT_DIR = "."
SEEDS = [42, 256, 1024]
KS = [64, 128, 256]
TOKEN_COUNTS = [250_000_000, 500_000_000, 1_000_000_000]  # check early, mid, and late training
N_NULL_SAMPLES = 1000  # number of anchor draws for the corrected max-of-n_candidates null


class SparseAutoencoder(nn.Module):
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


def load_sae(path: Path) -> SparseAutoencoder:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    sae = SparseAutoencoder(cfg["d_model"], cfg["n_features"], seed=0)
    sae.load_state_dict(ckpt["model_state_dict"])
    return sae


def compute_best_sim(sae1: SparseAutoencoder, sae2: SparseAutoencoder) -> np.ndarray:
    """Best cross-seed match similarity per feature (anchor = sae1)."""
    W1 = F.normalize(sae1.W_dec.detach(), dim=1)
    W2 = F.normalize(sae2.W_dec.detach(), dim=1)
    sim = (W1 @ W2.T)
    best_sim, _ = sim.max(dim=1)
    return best_sim.numpy()


def compute_null_distribution(sae: SparseAutoencoder, n_samples: int, n_candidates: int) -> np.ndarray:
    """Null distribution for best_sim under the null hypothesis of no shared
    structure across seeds: for each of n_samples anchor features, sample
    n_candidates random other features from the same SAE and take the max
    similarity among them.
    """
    W = F.normalize(sae.W_dec.detach(), dim=1)
    n_features = W.shape[0]
    n_candidates = min(n_candidates, n_features - 1)
    rng = np.random.default_rng(0)

    null_best_sims = np.empty(n_samples)
    for i in range(n_samples):
        anchor_idx = rng.integers(0, n_features)
        # sample distinct candidate indices, excluding the anchor itself
        candidates = rng.choice(n_features - 1, size=n_candidates, replace=False)
        candidates = np.where(candidates >= anchor_idx, candidates + 1, candidates)
        sims = (W[anchor_idx] * W[candidates]).sum(dim=1)
        null_best_sims[i] = sims.max().item()

    return null_best_sims


def main():
    ckpt_dir = Path(CHECKPOINT_DIR)
    results: Dict[Tuple[int, int], dict] = {}

    for tokens in TOKEN_COUNTS:
        for k in KS:
            paths = {
                seed: ckpt_dir / f"seed{seed}_k{k}_tokens{tokens}.pt"
                for seed in SEEDS
            }
            missing = [s for s, p in paths.items() if not p.exists()]
            if missing:
                print(f"k={k}, tokens={tokens:,}: missing checkpoints for seeds {missing}, skipping.")
                continue

            saes = {seed: load_sae(p) for seed, p in paths.items()}
            anchor_seed = SEEDS[0]
            other_seeds = [s for s in SEEDS if s != anchor_seed]

            best_sims_all = []
            for other in other_seeds:
                best_sims_all.append(compute_best_sim(saes[anchor_seed], saes[other]))
            best_sim = np.concatenate(best_sims_all)

            null_sim = compute_null_distribution(
                saes[anchor_seed], N_NULL_SAMPLES, n_candidates=saes[anchor_seed].n_features
            )

            results[(k, tokens)] = {
                "best_sim": best_sim,
                "null_sim": null_sim,
                "best_sim_mean": best_sim.mean(),
                "best_sim_median": np.median(best_sim),
                "null_sim_mean": null_sim.mean(),
                "null_sim_p95": np.percentile(null_sim, 95),
                "frac_above_theta": (best_sim >= 0.7).mean(),
                "frac_above_null_p95": (best_sim >= np.percentile(null_sim, 95)).mean(),
            }

    print("=" * 100)
    print("Cross-seed match quality vs. null baseline, across k and training scale")
    print("=" * 100)
    header = f"{'k':>5} {'tokens':>13} {'best_sim mean':>15} {'best_sim median':>17} {'null mean':>11} {'null p95':>10} {'%>=0.7':>9} {'%>=null p95':>13}"
    print(header)
    for (k, tokens), r in results.items():
        print(f"{k:>5} {tokens:>13,} {r['best_sim_mean']:>15.4f} {r['best_sim_median']:>17.4f} "
              f"{r['null_sim_mean']:>11.4f} {r['null_sim_p95']:>10.4f} "
              f"{100*r['frac_above_theta']:>8.1f}% {100*r['frac_above_null_p95']:>12.1f}%")

    # Save data for plotting / further analysis
    Path("outputs").mkdir(exist_ok=True)
    np.savez(
        "outputs/similarity_distributions.npz",
        **{f"best_sim_k{k}_tok{tokens}": r["best_sim"] for (k, tokens), r in results.items()},
        **{f"null_sim_k{k}_tok{tokens}": r["null_sim"] for (k, tokens), r in results.items()},
    )
    print("\nSaved raw distributions to outputs/similarity_distributions.npz")

    # Plot overlaid histograms
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_rows = len(TOKEN_COUNTS)
        n_cols = len(KS)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True, squeeze=False)

        for row, tokens in enumerate(TOKEN_COUNTS):
            for col, k in enumerate(KS):
                ax = axes[row][col]
                r = results.get((k, tokens))
                if r is None:
                    ax.set_visible(False)
                    continue
                ax.hist(r["null_sim"], bins=40, alpha=0.5, density=True, label="null (max of n_candidates random)")
                ax.hist(r["best_sim"], bins=40, alpha=0.5, density=True, label="best_sim (cross-seed match)")
                ax.axvline(0.7, color="red", linestyle="--", linewidth=1, label="theta=0.7")
                ax.set_title(f"k={k}, {tokens:,} tokens")
                ax.set_xlabel("Cosine similarity")
                if col == 0:
                    ax.set_ylabel("Density")
                ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig("outputs/similarity_distributions.png", dpi=150)
        print("Saved plot to outputs/similarity_distributions.png")
    except Exception as e:
        print(f"(Skipped plotting: {e})")


if __name__ == "__main__":
    main()
