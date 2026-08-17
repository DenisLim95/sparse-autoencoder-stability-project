"""Is 'geometric isolation predicts stability' circular?

Predictor  = max/mean cosine to nearby vectors WITHIN the anchor dictionary.
Label      = max cosine to any vector in ANOTHER seed's dictionary >= 0.7.
Both are 'how close is the nearest neighbour', so a feature in a dense region of
the sphere scores high on both for purely geometric reasons.

Null: replace the other seed's dictionary with a randomly ROTATED copy of it. That
destroys all feature correspondence while preserving the marginal geometry. Any
predictive power that survives is measuring sphere density, not stability.
"""
import numpy as np, torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download

SEEDS, THETA = [42, 256, 1024], 0.7


def dec(seed, tokens):
    for repo in ("deenais/sae-stability-pythia70m", "ndasari/SAE_project"):
        try:
            p = hf_hub_download(repo, f"checkpoints/seed{seed}_tokens{tokens}.pt",
                                repo_type="model", local_files_only=True)
        except Exception:
            continue
        return F.normalize(torch.load(p, map_location="cpu",
                                      weights_only=False)["model_state_dict"]["W_dec"], dim=1)
    raise FileNotFoundError


def auroc(s, lab):
    lab = np.asarray(lab, bool)
    if lab.all() or not lab.any():
        return float("nan")
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    tot = np.zeros(len(cnt)); np.add.at(tot, inv, r); r = (tot / cnt)[inv]
    return (r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())


for tokens in (1_000_000_000, 8_000_000_000):
    A, B = dec(42, tokens), dec(256, tokens)
    d = A.shape[1]
    sim = A @ A.T; sim.fill_diagonal_(-1)
    iso = sim.topk(10, 1).values.mean(1).numpy()     # the proposal's statistic
    nn1 = sim.max(1).values.numpy()                  # nearest neighbour only

    real = (A @ B.T).max(1).values.numpy()
    Q = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(0)))[0]
    rot = (A @ (B @ Q).T).max(1).values.numpy()

    print(f"\n{'='*74}\n{tokens:,} tokens\n{'='*74}")
    print(f"  {'':<34}{'real seed 256':>16}{'ROTATED (null)':>17}")
    print(f"  {'label rate (best cos >= 0.7)':<34}{100*(real>=THETA).mean():>15.1f}%"
          f"{100*(rot>=THETA).mean():>16.1f}%")
    print(f"  {'median best cos':<34}{np.median(real):>16.3f}{np.median(rot):>17.3f}")
    print()
    print(f"  {'':<34}{'AUROC vs real':>16}{'AUROC vs null':>17}")
    for name, v in [("geometric isolation (10-NN)", iso), ("nearest-neighbour cosine", nn1)]:
        # sign as the DATA says (crowded = stable), not as the proposal predicts
        print(f"  {name:<34}{auroc(v, real>=THETA):>16.3f}"
              f"{auroc(v, rot>=np.quantile(rot,1-(real>=THETA).mean())):>17.3f}")
    print(f"\n  corr(isolation, best cos to real seed 256) = "
          f"{np.corrcoef(iso, real)[0,1]:+.3f}")
    print(f"  corr(isolation, best cos to ROTATED  copy) = "
          f"{np.corrcoef(iso, rot)[0,1]:+.3f}")
