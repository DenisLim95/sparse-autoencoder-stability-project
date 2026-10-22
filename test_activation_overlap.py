# -*- coding: utf-8 -*-
"""Synthetic checks for activation_overlap.py: no Hub, no GPU, no activations file.

The streaming accumulator is checked against dense numpy on small matrices, the index
constructions against their stated properties, and the strict-label NaN semantics against a
hand table. Run: python test_activation_overlap.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

from activation_overlap import (  # noqa: E402
    PairAccumulator,
    combine_partners,
    frequency_matched_partner,
    nearest_within,
    random_partner,
    top2_cross,
)
from sae_stats import TopKSparseAutoencoder, compute_reappearance_probability  # noqa: E402

failures = []


def check(label, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        failures.append(label)


def dense_reference(Fa, G):
    """Jaccard, Pearson and cosine per column from dense arrays, NaN where undefined."""
    n = Fa.shape[1]
    jac, pear, cos = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for j in range(n):
        a, g = Fa[:, j], G[:, j]
        fa, fg = a > 0, g > 0
        union = (fa | fg).sum()
        if union > 0:
            jac[j] = (fa & fg).sum() / union
        if a.std() > 0 and g.std() > 0:
            pear[j] = np.corrcoef(a, g)[0, 1]
        na, ng = np.linalg.norm(a), np.linalg.norm(g)
        if na > 0 and ng > 0:
            cos[j] = a @ g / (na * ng)
    return jac, pear, cos


def run_accumulator(Fa, G, batch):
    acc = PairAccumulator(Fa.shape[1], "cpu")
    for start in range(0, len(Fa), batch):
        acc.update(torch.as_tensor(Fa[start:start + batch]), torch.as_tensor(G[start:start + batch]))
    return acc.finalize(len(Fa))


print("1. accumulator vs dense reference")
rng = np.random.default_rng(0)
N, n = 500, 16
Fa = rng.random((N, n)) * (rng.random((N, n)) < 0.15)        # sparse, non-negative
# The accumulator receives the partner already gathered into anchor column order, so the
# reference partner is the anchor plus noise where it fires, with a few columns altered.
G = Fa + 0.05 * rng.random((N, n)) * (Fa > 0)
G[:, 3] = 0.0                                                 # dead partner column
Fa[:, 5] = 0.0                                                # dead anchor column
G[:, 7] = (Fa[:, 7] == 0) * rng.random(N)                     # disjoint support with anchor 7
Fa, G = Fa.astype(np.float32), G.astype(np.float32)
jac_ref, pear_ref, cos_ref = dense_reference(Fa.astype(np.float64), G.astype(np.float64))
for batch in (64, 500, 7):
    got = run_accumulator(Fa, G, batch)
    check(f"jaccard matches dense (batch {batch})",
          np.allclose(got["jaccard"], jac_ref, atol=1e-6, equal_nan=True))
    check(f"pearson matches dense (batch {batch})",
          np.allclose(got["pearson"], pear_ref, atol=1e-6, equal_nan=True))
    check(f"cosine matches dense (batch {batch})",
          np.allclose(got["act_cosine"], cos_ref, atol=1e-6, equal_nan=True))
got = run_accumulator(Fa, G, 64)
check("dead partner column: Pearson NaN, Jaccard 0 (anchor fires, partner never)",
      np.isnan(got["pearson"][3]) and got["jaccard"][3] == 0)
check("dead anchor column gives NaN, not 0", np.isnan(got["pearson"][5]) and np.isnan(got["act_cosine"][5]))
check("disjoint support gives Jaccard 0 and non-positive Pearson",
      got["jaccard"][7] == 0 and got["pearson"][7] <= 0)
live = [j for j in range(n) if j not in (3, 5, 7)]
check("permuted copy gives Pearson ~1 on live columns", np.all(got["pearson"][live] > 0.99))

print("2. nearest neighbour and top-2 helpers")
W = torch.tensor([[1, 0, 0, 0], [0.99, 0.1, 0, 0], [0, 1, 0, 0], [0, 0.98, 0.2, 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
idx, cos = nearest_within(W)
check("nearest_within never returns self", np.all(idx != np.arange(6)))
check("nearest_within finds the planted pairs", idx[0] == 1 and idx[1] == 0 and idx[2] == 3 and idx[3] == 2)
b_idx, b_cos, s_idx, s_cos = top2_cross(W, W, exclude_self=True)
check("top2 best equals nearest_within", np.array_equal(b_idx, idx) and np.allclose(b_cos, cos))
check("top2 second is never best or self", np.all(s_idx != b_idx) and np.all(s_idx != np.arange(6)))
check("top2 second cosine <= best cosine", np.all(s_cos <= b_cos + 1e-6))

print("3. random and frequency-matched partners")
best = rng.integers(0, 200, size=200)
rp = random_partner(best, np.random.default_rng(1))
check("random partner is a permutation", np.array_equal(np.sort(rp), np.arange(200)))
check("random partner never equals the true match", np.all(rp != best))
freq = 10.0 ** rng.uniform(-5, -1, size=200)
rf = frequency_matched_partner(best, freq, np.random.default_rng(2), n_strata=10)
logf = np.log10(freq + 1e-10)
edges = np.quantile(logf, np.linspace(0, 1, 11))
stratum = np.clip(np.searchsorted(edges[1:-1], logf, side="right"), 0, 9)
check("freq-matched partner is in the best match's decile", np.all(stratum[rf] == stratum[best]))
check("freq-matched partner never equals the true match", np.all(rf != best))

print("4. combining partners (NaN must propagate)")
a = np.array([[0.9, 0.2, np.nan, 0.9, np.nan], [0.8, 0.7, 0.5, np.nan, np.nan]])
m = combine_partners(a)
check("min over partners where both finite", np.allclose(m[[0, 1]], [0.8, 0.2]))
check("NaN in either partner propagates", np.isnan(m[2]) and np.isnan(m[3]) and np.isnan(m[4]))
with np.errstate(invalid="ignore"):
    agrees = m >= 0.5
check("NaN agreement fails the cutoff", not agrees[2] and not agrees[3] and agrees[0] and not agrees[1])

print("5. end to end on two tiny TopK SAEs")
torch.manual_seed(0)
d, nf, k = 8, 16, 2
A = TopKSparseAutoencoder(d, nf, seed=0, k=k)
B = TopKSparseAutoencoder(d, nf, seed=1, k=k)
perm = torch.randperm(nf)
with torch.no_grad():
    for sae in (A, B):
        sae.b_dec.zero_()
    B.W_enc.copy_(A.W_enc[:, perm])
    B.W_dec.copy_(A.W_dec[perm])
    B.b_enc.copy_(A.b_enc[perm])
x = torch.randn(300, d)
with torch.no_grad():
    Fa, Fb = A.encode(x), B.encode(x)
check("encode respects k", int((Fa > 0).sum(1).max()) <= k)
probs, info = compute_reappearance_probability({0: A, 1: B})
best_idx = info["best_match_idx"][0]
inv = torch.empty(nf, dtype=torch.long)
inv[perm] = torch.arange(nf)
check("label's best match recovers the permutation", np.array_equal(best_idx, inv.numpy()))
check("every latent is stable under a permuted copy", np.all(probs == 1.0))
acc = PairAccumulator(nf, "cpu")
own_idx, _ = nearest_within(A.W_dec.detach())
acc_own = PairAccumulator(nf, "cpu")
idx_t = torch.as_tensor(best_idx)
for start in range(0, 300, 64):
    fa, fb = Fa[start:start + 64], Fb[start:start + 64]
    acc.update(fa, fb.index_select(1, idx_t))
    acc_own.update(fa, fa.index_select(1, torch.as_tensor(own_idx)))
got = acc.finalize(300)
fires = (Fa > 0).sum(0).numpy() > 0
check("matched pairs agree exactly where the anchor fires",
      np.all(got["jaccard"][fires] == 1.0) and np.all(got["pearson"][fires] > 0.999))
check("never-firing anchors are NaN", np.all(np.isnan(got["pearson"][~fires])))
got_own = acc_own.finalize(300)
check("own-neighbour agreement is below the exact match",
      np.nanmedian(got_own["pearson"]) < np.nanmedian(got["pearson"]))

print()
if failures:
    print(f"{len(failures)} check(s) failed:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("all checks passed")
