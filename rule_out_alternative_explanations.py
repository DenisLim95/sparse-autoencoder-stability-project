"""Check on the two alternative explanations for the instability of the SAE"""

import re
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple

CHECKPOINT_DIR = "."
SEEDS = [42, 256, 1024]
KS = [64, 128, 256]
DUPLICATE_THETA = 0.9   # decoder cosine sim above this = "near-duplicate" columns
DEAD_NORM_THRESH = 1e-3  # decoder columns with near-zero norm before normalization


def find_all_checkpoints(checkpoint_dir: str) -> List[Tuple[int, int, int, Path]]:
    """Return list of (seed, k, tokens, path) for every checkpoint on disk."""
    ckpt_dir = Path(checkpoint_dir)
    pattern = re.compile(r"seed(\d+)_k(\d+)_tokens(\d+)\.pt")
    out = []
    for f in ckpt_dir.glob("seed*_k*_tokens*.pt"):
        m = pattern.match(f.name)
        if not m:
            continue
        seed, k, tokens = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if seed in SEEDS and k in KS:
            out.append((seed, k, tokens, f))
    return sorted(out, key=lambda x: (x[1], x[2], x[0]))  # sort by k, tokens, seed


def tier1_structural_diagnostics(path: Path) -> dict:
    """Compute dead-feature and near-duplicate-feature rates from decoder
    weights alone, for a single checkpoint. Requires no activation data."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"]
    W_dec = state["W_dec"]  # [n_features, d_model], as saved (may be pre- or post-normalize)

    # Raw decoder norms, prior to any unit-norm renormalization, to detect
    # collapsed (dead) columns.
    raw_norms = W_dec.norm(dim=1)
    n_features = W_dec.shape[0]
    dead_count = (raw_norms < DEAD_NORM_THRESH).sum().item()

    # Redundancy: normalize decoder columns, then compute the maximum
    # off-diagonal cosine similarity for each feature against all others.
    W_norm = F.normalize(W_dec, dim=1)
    sim = W_norm @ W_norm.T
    sim.fill_diagonal_(-1.0)  # exclude self-similarity
    max_sim_per_feature, _ = sim.max(dim=1)
    near_dup_count = (max_sim_per_feature >= DUPLICATE_THETA).sum().item()

    return {
        "n_features": n_features,
        "dead_count": dead_count,
        "pct_dead": 100 * dead_count / n_features,
        "near_dup_count": near_dup_count,
        "pct_near_dup": 100 * near_dup_count / n_features,
        "mean_max_sim": max_sim_per_feature.mean().item(),
    }


def tier2_activation_diagnostics(path: Path) -> dict:
    """Compute per-feature firing statistics from logged activation data.
    Requires the checkpoint to contain activation statistics under one of
    the recognized key names ('activation_frequency', 'feature_acts_ema',
    'firing_counts'); update `candidate_keys` below if the training script
    logs these under a different name."""
    ckpt = torch.load(path, map_location="cpu")
    candidate_keys = ["activation_frequency", "feature_acts_ema", "firing_counts"]
    freq_key = next((k for k in candidate_keys if k in ckpt), None)

    if freq_key is None:
        return {"available": False}

    freq = ckpt[freq_key]
    if torch.is_tensor(freq):
        freq = freq.numpy()

    return {
        "available": True,
        "source_key": freq_key,
        "mean_firing_rate": float(np.mean(freq)),
        "pct_never_fired": 100 * float((freq == 0).sum()) / len(freq),
        "pct_rarely_fired": 100 * float((freq < 1e-4).sum()) / len(freq),  # tune threshold to your k / batch size
    }

    # If activation statistics were not logged at training time, per-feature
    # firing rates can be recovered by running a forward pass over a
    # held-out batch of activations with the loaded SAE and counting
    # firings directly. This requires access to the original activation
    # dataset or the base model used to generate it.


def main():
    checkpoints = find_all_checkpoints(CHECKPOINT_DIR)
    if not checkpoints:
        print("No checkpoints found — check CHECKPOINT_DIR.")
        return

    print(f"Found {len(checkpoints)} checkpoints.\n")
    print("=" * 100)
    print("Structural diagnostics: dead-feature and near-duplicate-feature rates")
    print("=" * 100)
    header = f"{'k':>5} {'tokens':>13} {'seed':>6} {'%dead':>8} {'%near-dup':>10} {'mean max-sim':>13}"
    print(header)

    tier1_results = []
    for seed, k, tokens, path in checkpoints:
        d = tier1_structural_diagnostics(path)
        tier1_results.append((seed, k, tokens, d))
        print(f"{k:>5} {tokens:>13,} {seed:>6} {d['pct_dead']:>7.1f}% {d['pct_near_dup']:>9.1f}% {d['mean_max_sim']:>13.4f}")

    print()
    print("=" * 100)
    print("Activation-frequency diagnostics (requires logged checkpoint statistics)")
    print("=" * 100)
    sample_check = tier2_activation_diagnostics(checkpoints[0][3])
    if not sample_check.get("available"):
        print("No activation-frequency field was found in the checkpoints (keys checked:")
        print("  activation_frequency, feature_acts_ema, firing_counts).")
        print("If the training script logs firing statistics under a different key name,")
        print("add it to `candidate_keys` in tier2_activation_diagnostics() and re-run.")
        print("Otherwise, per-feature firing rates require a fresh forward pass over")
        print("held-out activations, which requires access to the training data or model.")
    else:
        header2 = f"{'k':>5} {'tokens':>13} {'seed':>6} {'mean rate':>10} {'%never-fired':>13} {'%rarely-fired':>14}"
        print(header2)
        for seed, k, tokens, path in checkpoints:
            d = tier2_activation_diagnostics(path)
            if d.get("available"):
                print(f"{k:>5} {tokens:>13,} {seed:>6} {d['mean_firing_rate']:>10.5f} "
                      f"{d['pct_never_fired']:>12.1f}% {d['pct_rarely_fired']:>13.1f}%")


if __name__ == "__main__":
    main()
