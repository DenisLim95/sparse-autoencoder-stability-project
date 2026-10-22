"""Checks for seed_robustness.py's labelling and scoring, on synthetic decoders.

Each anchor latent is planted to reappear in exactly m of the four partner dictionaries (the
first m in partner order), so the expected p_hat and labels are known for every partner-set
size. No Hub, no GPU.

    python test_seed_robustness.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rotation_null import best_cosine
from seed_robustness import check_against_table, labels, score_rows

failures = []


def check(label, ok):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


torch.manual_seed(0)
rng = np.random.default_rng(0)
D, N = 64, 200
PARTNERS = (256, 1024, 137, 512)

A = F.normalize(torch.randn(N, D), dim=1)
planted = rng.integers(0, len(PARTNERS) + 1, size=N)  # partners each latent reappears in
decoders = {}
for j, s in enumerate(PARTNERS):
    P = F.normalize(torch.randn(N, D), dim=1)  # random directions: cosine ~ N(0, 1/8) in 64-d
    keep = torch.as_tensor(planted > j)
    P[keep] = A[keep]
    decoders[s] = P
best = {s: best_cosine(A, decoders[s]) for s in PARTNERS}

print("Labels by partner-set size")
prev_stable = None
for n in range(1, len(PARTNERS) + 1):
    partners = list(PARTNERS[:n])
    stable, unstable = labels(best, partners)
    check(f"{n + 1} seeds: stable iff planted in all {n} partners",
          np.array_equal(stable, planted >= n))
    check(f"{n + 1} seeds: unstable iff planted in none",
          np.array_equal(unstable, planted == 0))
    if prev_stable is not None:
        check(f"{n + 1} seeds: adding a partner never makes a latent stable",
              not (stable & ~prev_stable).any())
    prev_stable = stable

print("\nAgreement with an exported table")
stable, unstable = labels(best, list(PARTNERS))
table = pd.DataFrame({"is_stable": stable, "is_unstable": unstable})
try:
    check_against_table(table, best, list(PARTNERS))
    check("identical labels pass", True)
except SystemExit:
    check("identical labels pass", False)
flipped = table.copy()
flipped.loc[int(np.flatnonzero(stable)[0]), "is_stable"] = False
try:
    check_against_table(flipped, best, list(PARTNERS))
    check("a disagreeing label stops the script", False)
except SystemExit:
    check("a disagreeing label stops the script", True)

print("\nScoring")
# Crowding tracks the planted count, so its AUROC should be high at every set size.
stats = {
    "activation_freq": rng.uniform(1e-4, 1e-2, N),
    "geometric_isolation": planted + rng.normal(0, 0.3, N),
    "recon_contribution": rng.uniform(1e-3, 1e-2, N),
    "mean_activation": rng.uniform(0.5, 1.5, N),
    "encoder_norm": rng.uniform(1, 2, N),
    "encoder_bias": rng.normal(0, 1, N),
    "firing_counts": np.full(N, 1_000),
}
stats["firing_counts"][:5] = 10  # below the floor: must never be scored
rows = score_rows(stats, best, list(PARTNERS))
check("one row per partner-set size, 2..5 seeds", list(rows.n_seeds) == [2, 3, 4, 5])
check("latents under the firing floor are excluded",
      (rows.n_labelled <= N - 5).all())
check("labelled count matches the labels at each size",
      all(r.n_labelled == int(((planted >= n) | (planted == 0))[5:].sum())
          for n, r in zip(range(1, 5), rows.itertuples())))
check("crowding separates the planted classes (AUROC > 0.9 at every size)",
      (rows.auroc_crowding > 0.9).all())

print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
raise SystemExit(1 if failures else 0)
