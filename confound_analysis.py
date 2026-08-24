# -*- coding: utf-8 -*-
"""
check_dead_frac_from_checkpoints.py

Extracts the dead-feature fraction already recorded DURING TRAINING, straight
from each checkpoint's saved history -- no model loading, no dataset
streaming, no inference. This is much cheaper than re-running activations
through the model, and uses a real measurement from actual training data
(dead_frac was computed each logging window as the fraction of features that
never fired at all within that window, via `seen_active` in the training
loop), not a proxy.

Caveat vs. a fresh held-out check: this reflects dead status *as measured
during training*, in a recent logging window before each checkpoint was
saved -- not from a separate, dedicated held-out eval pass. For a quick,
laptop-friendly sanity check this should already be informative; if you want
the exact figure the paper's confound-checks paragraph should cite, the
held-out version (check_dead_features.py) remains the more rigorous one, but
this is a fast, cheap way to see the general shape of the result first.
"""

import re
from pathlib import Path

import torch

CHECKPOINT_DIR = "."
SEEDS = [42, 256, 1024]
KS = [64, 128, 256]
TOKEN_COUNTS = [250_000_000, 500_000_000, 1_000_000_000]


def main():
    ckpt_dir = Path(CHECKPOINT_DIR)

    print("=" * 90)
    print("Dead-feature fraction, as recorded during training (from checkpoint history)")
    print("=" * 90)
    header = f"{'k':>5} {'tokens':>13} {'seed':>6} {'dead_frac (last logged window)':>32}"
    print(header)

    found_any = False
    for tokens in TOKEN_COUNTS:
        for k in KS:
            for seed in SEEDS:
                path = ckpt_dir / f"seed{seed}_k{k}_tokens{tokens}.pt"
                if not path.exists():
                    continue
                found_any = True
                # weights_only=False: these checkpoints carry history/config
                # alongside the weights, same as the training script itself uses.
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                history = ckpt.get("history", {})
                dead_frac_series = history.get("dead_frac", [])

                if not dead_frac_series:
                    print(f"{k:>5} {tokens:>13,} {seed:>6} {'no dead_frac in history':>32}")
                    continue

                last_dead_frac = dead_frac_series[-1]
                print(f"{k:>5} {tokens:>13,} {seed:>6} {100 * last_dead_frac:>31.1f}%")

    if not found_any:
        print("\nNo matching checkpoints found -- check CHECKPOINT_DIR, SEEDS, KS, TOKEN_COUNTS.")
        return

    print()
    print("Interpretation:")
    print("  - This is the fraction of features that fired ZERO times in the most recent")
    print("    ~200-step logging window before this checkpoint was saved -- a real,")
    print("    training-time measurement, not a proxy like decoder norm.")
    print("  - If this stays near 0% across all k, it supports the original 'dead features")
    print("    ruled out' claim -- just via a valid method this time, not the invalid")
    print("    decoder-norm check.")
    print("  - If this is notably higher (especially at k=256), the dead-feature explanation")
    print("    for the flat-low stability result needs to be taken seriously again, and the")
    print("    confound-checks paragraph needs revising before submission.")


if __name__ == "__main__":
    main()