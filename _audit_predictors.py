"""What weight-only single-run statistics survive normalize_decoder(), and do they
predict p-hat at the 8B checkpoint?"""
import numpy as np, torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download

SEEDS, THETA = [42, 256, 1024], 0.7


def sd(seed, tokens):
    for repo in ("deenais/sae-stability-pythia70m", "ndasari/SAE_project"):
        try:
            p = hf_hub_download(repo, f"checkpoints/seed{seed}_tokens{tokens}.pt",
                                repo_type="model", local_files_only=True)
        except Exception:
            continue
        return torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]
    raise FileNotFoundError((seed, tokens))


def auroc(s, lab):
    lab = np.asarray(lab, bool)
    o = np.argsort(s); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    tot = np.zeros(len(cnt)); np.add.at(tot, inv, r); r = (tot / cnt)[inv]
    return (r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())


for tokens in (1_000_000_000, 8_000_000_000):
    S = {s: sd(s, tokens) for s in SEEDS}
    A = F.normalize(S[42]["W_dec"], dim=1)
    p = np.zeros(A.shape[0])
    for s in SEEDS[1:]:
        p += (A @ F.normalize(S[s]["W_dec"], dim=1).T).max(1).values.numpy() >= THETA
    p /= len(SEEDS) - 1
    lab = p >= 0.5

    dec_norm = S[42]["W_dec"].norm(dim=1).numpy()
    enc_norm = S[42]["W_enc"].norm(dim=0).numpy()
    b_enc = S[42]["b_enc"].numpy()
    sim = A @ A.T; sim.fill_diagonal_(-1)
    iso = sim.topk(10, 1).values.mean(1).numpy()

    print(f"\n{'='*72}\n{tokens:,} tokens   |   positive class {100*lab.mean():.1f}%\n{'='*72}")
    print(f"  {'statistic':<26}{'min':>10}{'max':>10}{'rel.spread':>12}{'AUROC':>9}")
    for name, v, sign in [("decoder norm", dec_norm, +1), ("encoder col norm", enc_norm, +1),
                          ("encoder bias b_enc", b_enc, +1),
                          ("geometric isolation", iso, -1)]:
        spread = (v.max() - v.min()) / (abs(v.mean()) + 1e-12)
        print(f"  {name:<26}{v.min():>10.4f}{v.max():>10.4f}{spread:>12.2e}"
              f"{auroc(sign * v, lab):>9.3f}")

    # b_enc sets the activation threshold, so it is the natural weight-only proxy for
    # how often / how strongly a feature fires -- check it is not just frequency again.
    print(f"  corr(enc_norm, b_enc) = {np.corrcoef(enc_norm, b_enc)[0,1]:+.3f}   "
          f"corr(enc_norm, iso) = {np.corrcoef(enc_norm, iso)[0,1]:+.3f}   "
          f"corr(b_enc, iso) = {np.corrcoef(b_enc, iso)[0,1]:+.3f}")
