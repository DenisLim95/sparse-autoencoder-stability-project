"""Checks for crowding_ablation.neighbourhood and the variants built from it, against a brute
force computation on a small synthetic decoder. No Hub, no GPU.

    python test_crowding_ablation.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch.nn.functional as F

from crowding_ablation import MEAN_TOP_N, SORTED_TOP, TOP_M, neighbourhood, variants
from rotation_null import crowding

failures = []


def check(label, ok):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


torch.manual_seed(0)
N, D = 300, 16
A = F.normalize(torch.randn(N, D), dim=1)
A[1] = A[0]  # an exact duplicate: its partner must be found at cosine 1, but never itself
A[3] = F.normalize(A[2] + 0.05 * torch.randn(D), dim=0)  # a near-duplicate pair

sims = (A @ A.T).numpy()
np.fill_diagonal(sims, -np.inf)
brute_top = -np.sort(-sims, axis=1)[:, :TOP_M]

print("neighbourhood")
top, counts = neighbourhood(A, chunk=37)  # a chunk size that does not divide N
top_one, counts_one = neighbourhood(A, chunk=N)
check("chunked equals one pass", np.allclose(top, top_one)
      and all(np.array_equal(counts[t], counts_one[t]) for t in counts))
check("sorted top-m matches brute force", np.allclose(top, brute_top, atol=1e-6))
check("rows are sorted largest first", bool((np.diff(top, axis=1) <= 1e-7).all()))
check("self is excluded: the duplicate finds its partner at 1.0, others stay below 1",
      np.isclose(top[0, 0], 1.0) and np.isclose(top[1, 0], 1.0) and top[4:, 0].max() < 0.999)
check("the near-duplicate pair are each other's nearest neighbour",
      np.isclose(top[2, 0], sims[2, 3]) and np.isclose(top[3, 0], sims[3, 2])
      and sims[2, 3] > 0.9)
for tau in counts:
    check(f"count above {tau:g} matches brute force",
          np.array_equal(counts[tau], (sims > tau).sum(axis=1)))

print("\nvariants")
built = {name: cols for name, _, cols in variants(top, counts)}
check("mean top-1 is the row max", np.allclose(built["mean top-1 (= max cosine)"][:, 0],
                                               sims.max(axis=1)))
check("mean top-10 equals rotation_null.crowding (the paper's definition)",
      np.allclose(built["mean top-10"][:, 0], crowding(A), atol=1e-6))
check("every mean-top-n variant is present",
      all(any(name.startswith(f"mean top-{n}") for name in built) for n in MEAN_TOP_N))
check(f"sorted top-{SORTED_TOP} has {SORTED_TOP} columns",
      built[f"sorted top-{SORTED_TOP}"].shape == (N, SORTED_TOP))
check("count variants are log1p of the counts",
      all(np.allclose(built[f"count above {tau:g}"][:, 0], np.log1p(counts[tau]))
          for tau in counts))

print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
raise SystemExit(1 if failures else 0)
