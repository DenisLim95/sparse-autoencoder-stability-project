"""Throwaway audit: is the reported cross-seed decoder similarity real per-feature
correspondence, or a by-product of shared geometry / a crowded dictionary?"""
import numpy as np, torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download

REPOS = {"deenais/sae-stability-pythia70m": [2, 3, 5, 8],
         "ndasari/SAE_project": [0.001, 0.05, 0.1, 1]}
SEEDS = [42, 256, 1024]


def load(seed, tokens):
    for repo in REPOS:
        try:
            p = hf_hub_download(repo, f"checkpoints/seed{seed}_tokens{tokens}.pt",
                                repo_type="model", local_files_only=True)
        except Exception:
            continue
        return torch.load(p, map_location="cpu", weights_only=False)
    return None


def report(tokens):
    ck = {s: load(s, tokens) for s in SEEDS}
    if any(c is None for c in ck.values()):
        print(f"{tokens:,}: missing"); return
    W = {s: F.normalize(c["model_state_dict"]["W_dec"], dim=1) for s, c in ck.items()}
    E = {s: F.normalize(c["model_state_dict"]["W_enc"].T, dim=1) for s, c in ck.items()}
    bd = {s: c["model_state_dict"]["b_dec"] for s, c in ck.items()}
    A, B = W[42], W[256]
    n = A.shape[0]

    print(f"\n{'='*70}\n{tokens:,} tokens\n{'='*70}")

    # 1. global anisotropy: do all decoder rows share one big common direction?
    mu = A.mean(0)
    print(f"  ||mean decoder row||              {mu.norm():.3f}   (0 = isotropic, 1 = all identical)")
    off = (A @ A.T)
    off.fill_diagonal_(0)
    print(f"  mean pairwise cos within seed 42  {off.sum().item()/(n*(n-1)):+.3f}")
    sv = torch.linalg.svdvals(A)
    er = (sv.sum()**2 / (sv**2).sum()).item()
    print(f"  participation ratio of dictionary {er:.1f} of {A.shape[1]} dims")

    # 2. within-dictionary null: best cosine to ANOTHER feature of the SAME SAE
    self_best = off.max(1).values.numpy()
    cross_best = (A @ B.T).max(1).values.numpy()
    print(f"  best cos, cross-seed  (matching)  median {np.median(cross_best):.3f}  >=0.7 {100*(cross_best>=.7).mean():.1f}%")
    print(f"  best cos, WITHIN seed 42 (null)   median {np.median(self_best):.3f}  >=0.7 {100*(self_best>=.7).mean():.1f}%")

    # 3. many-to-one collapse
    idx = (A @ B.T).max(1).indices.numpy()
    hit = idx[cross_best >= .7]
    if hit.size:
        cnt = np.bincount(hit, minlength=n)
        print(f"  distinct partners used            {len(np.unique(hit))} for {hit.size} matched anchors"
              f" (max reuse {cnt.max()})")

    # 4. how much of the match survives removing the shared top-k subspace
    for k in (1, 5, 20):
        U = torch.linalg.svd(torch.cat([A, B]), full_matrices=False).Vh[:k]
        Ak = F.normalize(A - (A @ U.T) @ U, dim=1)
        Bk = F.normalize(B - (B @ U.T) @ U, dim=1)
        cb = (Ak @ Bk.T).max(1).values.numpy()
        print(f"  after removing top-{k:<2} shared PCs   median {np.median(cb):.3f}  >=0.7 {100*(cb>=.7).mean():.1f}%")

    # 5. random-rotation null (destroys correspondence, keeps marginal geometry)
    Q = torch.linalg.qr(torch.randn(A.shape[1], A.shape[1], generator=torch.Generator().manual_seed(0)))[0]
    rb = (A @ (B @ Q).T).max(1).values.numpy()
    print(f"  rotation null                     median {np.median(rb):.3f}  >=0.7 {100*(rb>=.7).mean():.1f}%")

    # 6. does the ENCODER agree with the decoder match?
    enc_at_dec_match = (E[42] * E[256][idx]).sum(1).numpy()
    m = cross_best >= .7
    if m.any():
        print(f"  encoder cos at decoder-matched j  median {np.median(enc_at_dec_match[m]):.3f}")

    # 7. shared bias
    print(f"  cos(b_dec 42, b_dec 256)          {F.cosine_similarity(bd[42], bd[256], dim=0):.4f}")
    print(f"  ||b_dec|| 42 / 256                {bd[42].norm():.2f} / {bd[256].norm():.2f}")


for t in [100_000_000, 1_000_000_000, 8_000_000_000]:
    report(t)
