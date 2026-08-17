"""Controlled test: does training every seed on ONE identically-ordered activation
stream (the repo's design) inflate cross-seed feature stability?

Uses the repo's exact SparseAutoencoder, loss, decoder normalisation, gradient
projection and Adam settings. Only the data-feeding policy differs between arms.
Match rate is tracked against optimizer steps, mirroring the repo's scaling curve.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(8)
D, M, BATCH, STEPS, L1, LR = 64, 256, 256, 30_000, 1.0, 3e-3
POOL, EVERY, THETA = 400, 5_000, 0.7

gen0 = torch.Generator().manual_seed(0)
DICT = F.normalize(torch.randn(M, D, generator=gen0), dim=1)
MEAN = torch.randn(D, generator=gen0) * 2.0
K = 6  # ground-truth sparsity


def draw(g):
    s = torch.zeros(BATCH, M)
    idx = torch.stack([torch.randperm(M, generator=g)[:K] for _ in range(BATCH)])
    s.scatter_(1, idx, torch.rand(BATCH, K, generator=g) * 2 + 0.5)
    return s @ DICT + MEAN + 0.05 * torch.randn(BATCH, D, generator=g)


def fresh_stream(data_seed):
    """Regenerable infinite stream: same data_seed -> byte-identical batch sequence."""
    g = torch.Generator().manual_seed(data_seed)
    while True:
        yield draw(g)


def pool_stream(data_seed, order_seed):
    """Fixed finite pool of batches, replayed in an order set by order_seed --
    the repo's post-resume regime, where the Pile stream restarts from the top."""
    g = torch.Generator().manual_seed(data_seed)
    pool = [draw(g) for _ in range(POOL)]
    og = torch.Generator().manual_seed(order_seed)
    while True:
        for i in torch.randperm(POOL, generator=og) if order_seed >= 0 else range(POOL):
            yield pool[i]


class SAE(nn.Module):  # verbatim from prelim_experiments_update.py
    def __init__(self, d, m, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.W_enc = nn.Parameter(torch.randn(d, m) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_dec = nn.Parameter(torch.randn(m, d) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d))
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def forward(self, x):
        f = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        xh = f @ self.W_dec + self.b_dec
        return F.mse_loss(xh, x) + L1 * f.abs().mean(), f


def train(seed, stream):
    sae, snaps = SAE(D, M, seed), {}
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    l0 = recon = 0.0
    for step in range(1, STEPS + 1):
        x = next(stream)
        loss, f = sae(x)
        opt.zero_grad(); loss.backward()
        with torch.no_grad():
            g = sae.W_dec.grad
            sae.W_dec.grad = g - (g * sae.W_dec.data).sum(1, keepdim=True) * sae.W_dec.data
        opt.step()
        with torch.no_grad():
            sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)
            if step % EVERY == 0:
                snaps[step] = sae.W_dec.data.clone()
                l0 = (f > 0).float().sum(1).mean().item()
                recon = F.mse_loss((f @ sae.W_dec + sae.b_dec), x).item()
    return snaps, l0, recon


def match(A, B):
    best = (F.normalize(A, dim=1) @ F.normalize(B, dim=1).T).max(1).values.numpy()
    return 100 * (best >= THETA).mean(), np.median(best)


ARMS = {
    "A. shared stream, identical order  (repo)":
        (lambda: fresh_stream(1), lambda: fresh_stream(1)),
    "B. independent streams":
        (lambda: fresh_stream(11), lambda: fresh_stream(22)),
    "C. same finite pool, identical order":
        (lambda: pool_stream(5, -1), lambda: pool_stream(5, -1)),
    "D. same finite pool, shuffled order":
        (lambda: pool_stream(5, -1), lambda: pool_stream(5, 99)),
}

print(f"d={D} m={M} batch={BATCH} steps={STEPS:,} lr={LR} l1={L1} "
      f"ground-truth k={K}\n")
results = {}
for name, (s1, s2) in ARMS.items():
    r1, l0, rec = train(42, s1())
    r2, _, _ = train(256, s2())
    results[name] = {st: match(r1[st], r2[st]) for st in r1}
    print(f"{name:<44} L0={l0:5.1f} recon={rec:.5f}")

steps = sorted(next(iter(results.values())))
print(f"\nMATCH RATE at theta={THETA} (% of anchor features with a cross-seed match)")
print(f"{'arm':<44}" + "".join(f"{s//1000:>8}k" for s in steps))
for name, r in results.items():
    print(f"{name:<44}" + "".join(f"{r[s][0]:>8.1f}%" for s in steps))
print(f"\nMEDIAN BEST COSINE")
print(f"{'arm':<44}" + "".join(f"{s//1000:>9}k" for s in steps))
for name, r in results.items():
    print(f"{name:<44}" + "".join(f"{r[s][1]:>10.3f}" for s in steps))
