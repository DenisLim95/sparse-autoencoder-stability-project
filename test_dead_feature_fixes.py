"""Numerically exercise the dead-latent fixes without running the whole experiment.

    python test_dead_feature_fixes.py

The first TopK run reached 60% dead latents and returned the highest AUROC of any run, 0.981,
because a never-firing latent is trivially separable rather than genuinely predictable. Four
changes address that, and each has a failure mode that a training run would only reveal after
hours of GPU time:

  tied init      -- copying the wrong direction (decoder <- encoder) silently destroys the
                    unit-norm decoder rows that everything downstream assumes.
  AuxK           -- the loss must reach latents the top-k threw away, must NOT add b_dec to
                    the auxiliary reconstruction, and must not leak gradient into the main
                    reconstruction through the residual it is trying to explain.
  firing floor   -- conditional statistics below the floor must be NaN, not an imputed 0.0
                    that the classifier can use as a dead-latent fingerprint.
  live filtering -- a floored feature matrix must contain no non-finite values at all.

Runs on CPU in a few seconds and needs no GPU, so it is safe on a contended shared node.
"""

import os

# See test_topk_sae.py: Adam.step() health-checks the accelerator even for CPU tensors, which
# fails on a node whose GPUs are all held in Exclusive_Process mode. Hide them.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import ast
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
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

D, N, K, B = 64, 256, 8, 128
failures = []


def check(label, ok):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


torch.manual_seed(0)
x = torch.randn(B, D) * 0.5

print("Tied initialization")
tied = SAE(D, N, seed=42, sparsity="topk", k=K, tied_init=True)
untied = SAE(D, N, seed=42, sparsity="topk", k=K, tied_init=False)
check("encoder is the decoder transposed", torch.equal(tied.W_enc.data, tied.W_dec.data.t()))
check("decoder rows are still unit norm",
      torch.allclose(tied.W_dec.data.norm(dim=1), torch.ones(N), atol=1e-6))
check("encoder columns are therefore unit norm too",
      torch.allclose(tied.W_enc.data.norm(dim=0), torch.ones(N), atol=1e-6))
check("the flag actually changes something", not torch.equal(tied.W_enc.data, untied.W_enc.data))
check("the decoder is untouched by the flag (same seed, same draw)",
      torch.equal(tied.W_dec.data, untied.W_dec.data))
# A tied latent reads along the direction it writes, so its own contribution to the
# reconstruction points the right way from step one. This is the whole mechanism.
check("a tied latent's read and write directions agree (cosine 1.0)",
      torch.allclose(F.cosine_similarity(tied.W_enc.data.t(), tied.W_dec.data, dim=1),
                     torch.ones(N), atol=1e-6))
check(f"an untied latent's do not (mean cosine "
      f"{F.cosine_similarity(untied.W_enc.data.t(), untied.W_dec.data, dim=1).mean():+.3f})",
      F.cosine_similarity(untied.W_enc.data.t(), untied.W_dec.data, dim=1).mean().abs() < 0.2)

print("\nAuxK: when it is inactive")
sae = SAE(D, N, seed=42, sparsity="topk", k=K, tied_init=True)
_, _, ld = sae(x)
check("no dead_mask means no auxiliary loss at all", float(ld["aux_loss"]) == 0.0)
no_dead = torch.zeros(N, dtype=torch.bool)
_, _, ld = sae(x, dead_mask=no_dead)
check("an empty dead set means no auxiliary loss", float(ld["aux_loss"]) == 0.0)

print("\nAuxK: it reaches latents the top-k stranded")
# Pick latents that genuinely never win the competition on this batch, which is exactly the
# population that receives no gradient under plain TopK and therefore stays dead forever.
f = sae.encode(x)
never_fires = ~(f > 0).any(dim=0)
check(f"the batch strands some latents ({int(never_fires.sum())} of {N} never fire)",
      bool(never_fires.any()))

sae.zero_grad()
_, _, ld = sae(x)
ld["recon_loss"].backward()
base_grad = sae.W_enc.grad[:, never_fires].abs().sum().item()
check(f"without AuxK they get no gradient (|grad| = {base_grad:.2e})", base_grad == 0.0)

sae.zero_grad()
_, _, ld = sae(x, dead_mask=never_fires, k_aux=N)
check(f"the auxiliary loss is positive ({float(ld['aux_loss']):.5f})", float(ld["aux_loss"]) > 0)
ld["aux_loss"].backward()
aux_grad = sae.W_enc.grad[:, never_fires].abs().sum().item()
check(f"with AuxK they do (|grad| = {aux_grad:.2e})", aux_grad > 0)
check("and the auxiliary gradient is finite", bool(torch.isfinite(sae.W_enc.grad).all()))

print("\nAuxK: it does not corrupt the main objective")
# The residual is a target. If it were not detached, the auxiliary term could reduce its own
# loss by making the main reconstruction worse, which is the opposite of what it is for.
sae.zero_grad()
_, _, ld = sae(x, dead_mask=never_fires, k_aux=N)
ld["aux_loss"].backward()
live = ~never_fires
live_grad = sae.W_enc.grad[:, live].abs().sum().item()
check(f"the auxiliary loss alone leaves live latents' encoders alone (|grad| = "
      f"{live_grad:.2e})", live_grad == 0.0)

# ê = W_dec z, with no b_dec: adding the bias back is a documented way to get this wrong, and
# it is invisible at training time because the loss still decreases.
sae_b = SAE(D, N, seed=7, sparsity="topk", k=K, tied_init=True)
with torch.no_grad():
    sae_b.b_dec.copy_(torch.randn(D) * 3.0)  # large, so a stray b_dec cannot hide in the noise
dead = torch.zeros(N, dtype=torch.bool)
dead[N // 2:] = True
with torch.no_grad():
    f_b, pre = sae_b._encode_with_pre(x)
    x_hat = sae_b.decode(f_b)
    dead_acts = F.relu(pre).masked_fill(~dead, 0.0)
    idx = dead_acts.topk(int(dead.sum()), dim=-1).indices
    keep = torch.zeros_like(dead_acts, dtype=torch.bool).scatter_(-1, idx, True)
    z = torch.where(keep, dead_acts, torch.zeros_like(dead_acts))
    want = F.mse_loss(z @ sae_b.W_dec, x - x_hat)
    wrong = F.mse_loss(z @ sae_b.W_dec + sae_b.b_dec, x - x_hat)
    got = sae_b.auxk_loss(x, x_hat, pre, dead, int(dead.sum()))
check(f"matches the closed form without b_dec ({float(got):.6f} vs {float(want):.6f})",
      torch.allclose(got, want, atol=1e-6))
check(f"and is distinguishable from the b_dec version ({float(wrong):.6f})",
      not torch.allclose(want, wrong, atol=1e-6))

print("\nAuxK: k_aux and edge cases")
one_dead = torch.zeros(N, dtype=torch.bool)
one_dead[0] = True
_, _, ld = sae(x, dead_mask=one_dead, k_aux=512)
check("k_aux above the dead count does not crash", bool(torch.isfinite(ld["aux_loss"])))
_, _, ld_zero = sae(x, dead_mask=one_dead, k_aux=0)
check("k_aux=0 disables the term", float(ld_zero["aux_loss"]) == 0.0)

print("\nAuxK: over training, dead latents stop being frozen")
# This is the death mechanism itself, and the only claim worth asserting. Plain TopK gives a
# latent that never wins exactly zero gradient at every step, so Adam accumulates no state for
# it and its weights never move from initialization -- death is permanent by construction, not
# merely likely. Whether AuxK raises the FINAL live count is an empirical question that depends
# on the data (in this toy, where a 256-latent dictionary is fitting rank-4 data, most latents
# ought to be dead and AuxK does not increase the count), so asserting a direction here would
# be asserting something that is not generally true.
def train(auxk_coeff, rank=4, steps=400, dead_after=20):
    torch.manual_seed(0)
    data = (torch.randn(1024, rank) @ torch.randn(rank, D)) * 0.3
    sae = SAE(D, N, seed=42, sparsity="topk", k=K, tied_init=True)
    init_enc = sae.W_enc.data.clone()
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    idle = torch.zeros(N)
    ever_fired = torch.zeros(N, dtype=torch.bool)
    ever_dead = torch.zeros(N, dtype=torch.bool)
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        batch = data[torch.randint(0, len(data), (B,), generator=gen)]
        mask = idle > dead_after if auxk_coeff else None
        if mask is not None:
            ever_dead |= mask
        _, f, ld = sae(batch, dead_mask=mask, k_aux=N)
        loss = ld["recon_loss"] + auxk_coeff * ld["aux_loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        with torch.no_grad():
            fired = (f > 0).any(dim=0)
            ever_fired |= fired
            idle = torch.where(fired, torch.zeros_like(idle), idle + 1)
    moved = (sae.W_enc.data - init_enc).abs().sum(dim=0) > 0
    return ever_fired, ever_dead, moved


fired_off, _, moved_off = train(0.0)
n_never = int((~fired_off).sum())
check(f"plain TopK strands latents ({n_never} of {N} never fire in 400 steps)", n_never > 0)
check("and their encoders are bit-identical to init, so they can never recover",
      not bool(moved_off[~fired_off].any()))

fired_on, ever_dead, moved_on = train(1 / 32)
check(f"with AuxK some latents get flagged dead ({int(ever_dead.sum())} of {N})",
      bool(ever_dead.any()))
check("and every one of them has moved, so it is back in the competition",
      bool(moved_on[ever_dead].all()))

print("\nFiring floor: conditional statistics below it are undefined, not zero")
MIN_FIRINGS = 100
counts = np.array([0.0, 1.0, 50.0, 100.0, 5000.0])
sums = np.array([0.0, 3.0, 100.0, 400.0, 10000.0])
enough = counts >= MIN_FIRINGS
mean_act = np.divide(sums, counts, out=np.full(len(counts), np.nan), where=enough)
check("a never-firing latent has no conditional mean (NaN, not 0.0)", np.isnan(mean_act[0]))
check("nor does one that fired once", np.isnan(mean_act[1]))
check("nor one just below the floor", np.isnan(mean_act[2]))
check("at the floor it is defined", mean_act[3] == 4.0)
check("well above it, too", mean_act[4] == 2.0)
# The old code imputed 0.0 here, which is what let the classifier fingerprint dead latents:
# frequency 0, mean activation 0, contribution 0 is a perfectly separable signature.
old = np.divide(sums, counts, out=np.zeros(len(counts)), where=counts > 0)
check("the imputed version really was separable (0.0 for every dead latent)",
      old[0] == 0.0 and not np.isnan(old[0]))

print("\nFiring floor: one floor really does govern both conditional statistics")
# The ablation used to run on a 10k-token subsample while the firing rates were measured over
# the whole eval set, so "100 firings" meant two different things to the two estimators and a
# single floor could not police both. Now both walk the same tokens, which is only true if the
# counts they report are identical -- check that rather than trusting it.
_tree = ast.parse(SCRIPT.read_text())


def extract(name):
    """Pull one top-level function out by name. Via the AST rather than by scanning for the
    next unindented line, because a multi-line signature puts its closing paren in column 0."""
    node = next(n for n in _tree.body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


from tqdm import tqdm

stats_ns = {
    "torch": torch, "np": np, "F": F, "tqdm": tqdm, "Tuple": Tuple, "Optional": Optional,
    "SparseAutoencoder": SAE, "STAT_BATCH": 4096, "MIN_FIRINGS": MIN_FIRINGS,
}
for _fn in ("compute_activation_stats", "compute_reconstruction_contribution"):
    exec(compile(extract(_fn), str(SCRIPT), "exec"), stats_ns)

# Straddle the floor by construction rather than by luck: a large bias puts the first 100
# latents permanently in the top-k (~8% of tokens each, so several hundred firings) and keeps
# the remaining 156 permanently out (zero firings), which is the dead-heavy regime under test.
sparse_sae = SAE(D, N, seed=3, sparsity="topk", k=8, tied_init=True)
with torch.no_grad():
    sparse_sae.b_enc[:100] += 50.0
    sparse_sae.b_enc[100:] -= 50.0
acts = torch.randn(6000, D) * 0.5
freq, mean_act, counts = stats_ns["compute_activation_stats"](
    sparse_sae, acts, "cpu", desc="test"
)
cond, uncond, active_counts = stats_ns["compute_reconstruction_contribution"](
    sparse_sae, acts, device="cpu"
)
check("both estimators count the same firings, so one floor covers both",
      bool(np.array_equal(counts, active_counts)))
# Both sides of the floor must be populated or the NaN checks below would pass trivially.
_under, _over = int((counts < MIN_FIRINGS).sum()), int((counts >= MIN_FIRINGS).sum())
check(f"the test SAE straddles the floor ({_under} under, {_over} over)",
      _under > 0 and _over > 0)
check(f"and is dead-heavy as intended ({int((counts == 0).sum())} of {N} never fire)",
      int((counts == 0).sum()) > 0)
check("frequency is a rate, always defined and never NaN", bool(np.isfinite(freq).all()))
check("conditional mean activation is NaN exactly below the floor",
      bool(np.array_equal(np.isnan(mean_act), counts < MIN_FIRINGS)))
check("conditional contribution is NaN exactly below the floor",
      bool(np.array_equal(np.isnan(cond), counts < MIN_FIRINGS)))
check("the unconditional contribution stays defined everywhere (it is a rate too)",
      bool(np.isfinite(uncond).all()))

print("\nFiring floor: nothing non-finite survives into the feature matrix")
rng = np.random.default_rng(0)
firing_counts = np.concatenate([np.zeros(40), rng.integers(1, 100, 60),
                                rng.integers(100, 10000, 100)]).astype(float)
conditional = np.where(firing_counts >= MIN_FIRINGS, rng.normal(size=200), np.nan)
labelled = rng.random(200) > 0.2
live = firing_counts >= MIN_FIRINGS
mask = labelled & live
X = np.column_stack([np.log10(firing_counts / 1e6 + 1e-10), conditional])
check(f"the floor drops the under-measured latents ({int((labelled & ~live).sum())} of "
      f"{int(labelled.sum())} labelled)", bool((labelled & ~live).any()))
check("and the surviving matrix is entirely finite", bool(np.isfinite(X[mask]).all()))
check("whereas without the floor it is not", not bool(np.isfinite(X[labelled]).all()))

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed: " + "; ".join(failures))
print("all checks passed")
