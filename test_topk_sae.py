"""Numerically exercise the TopK SAE without running the whole experiment.

    python test_topk_sae.py

Extracts the SparseAutoencoder class out of prelim_experiments_update.py and checks the
properties a GPU run would otherwise discover the expensive way: that L0 is exactly k, that
the kept latents really are the largest pre-activations, that gradients reach only the
selected latents, and that the L1 path is byte-identical to what it was before the flag
existed. Selecting on pre-activations rather than after the ReLU is the subtle part, and
getting it wrong produces an SAE that trains fine while silently activating features that
carry no signal.

Runs on CPU in a few seconds and needs no GPU, so it is safe on a contended shared node.
"""

import os

# Before importing torch, and deliberately: this test is pure CPU maths, but Adam.step() runs
# an unconditional accelerator health check that queries the CUDA stream anyway. On a shared
# node whose GPUs are all held in Exclusive_Process mode that check raises
# cudaErrorDevicesUnavailable, failing the test for reasons having nothing to do with the SAE.
# Hiding the devices removes the accelerator entirely, so the result depends only on the code.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT = Path(__file__).parent / "prelim_experiments_update.py"

lines = SCRIPT.read_text().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("class SparseAutoencoder"))
end = next(i for i, l in enumerate(lines) if l.startswith('"""## 3.'))
ns = {"torch": torch, "nn": nn, "F": F, "Tuple": Tuple, "Optional": Optional}
exec(compile("\n".join(lines[start:end]), str(SCRIPT), "exec"), ns)
SAE = ns["SparseAutoencoder"]

D, N, K, B = 512, 2048, 32, 64
torch.manual_seed(0)
x = torch.randn(B, D) * 0.5
failures = []


def check(label, ok):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


print("TopK: sparsity is exact")
sae = SAE(D, N, seed=42, sparsity="topk", k=K)
f = sae.encode(x)
l0 = (f > 0).sum(dim=-1)
check(f"every token has exactly k={K} active latents "
      f"(min {l0.min().item()}, max {l0.max().item()})", bool((l0 == K).all()))
check("activations are non-negative", bool((f >= 0).all()))

pre = (x - sae.b_dec) @ sae.W_enc + sae.b_enc
kept_is_top = all(
    pre[b][f[b] > 0].min() >= pre[b][f[b] == 0].max()
    for b in range(B) if (f[b] > 0).any()
)
check("the kept set is the top-k of the pre-activations", kept_is_top)

print("\nTopK: gradients")
sae.zero_grad()
_, f, ld = sae(x)
ld["recon_loss"].backward()
grad = sae.W_enc.grad
check("reconstruction loss is finite", bool(torch.isfinite(ld["recon_loss"])))
check("encoder gradient has no NaN or Inf", bool(torch.isfinite(grad).all()))
touched = int((grad.abs().sum(dim=0) > 0).sum())
active = int((f > 0).any(dim=0).sum())
check(f"gradient confined to latents that fired ({touched} touched, {active} fired)",
      touched <= active)

print("\nTopK: optimizes")
sae = SAE(D, N, seed=42, sparsity="topk", k=K)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
first = None
for step in range(60):
    _, _, ld = sae(x)
    loss = ld["recon_loss"]
    opt.zero_grad()
    loss.backward()
    opt.step()
    sae.normalize_decoder()
    first = loss.item() if step == 0 else first
check(f"reconstruction loss falls ({first:.5f} -> {loss.item():.5f} over 60 steps)",
      loss.item() < first)
check("L0 is still exactly k after training",
      bool(((sae.encode(x) > 0).sum(-1) == K).all()))

print("\nL1: unchanged by the flag")
sae_l1 = SAE(D, N, seed=42, sparsity="l1")
f1 = sae_l1.encode(x)
ref = F.relu((x - sae_l1.b_dec) @ sae_l1.W_enc + sae_l1.b_enc)
check("encode is exactly plain ReLU", torch.equal(f1, ref))
check(f"code stays dense, L0 {(f1 > 0).sum(-1).float().mean():.0f} >> k",
      (f1 > 0).sum(-1).float().mean() > K)

print("\nEdge case")
small = SAE(D, 16, seed=1, sparsity="topk", k=999)
check(f"k above the dictionary size is clamped to {small.k}", small.k == 16)
check("and forward still runs", small.encode(torch.randn(4, D)).shape == (4, 16))

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed: " + "; ".join(failures))
print("all checks passed")
