"""Audit whether the headline stability number is a property of the SAEs or of the matcher.

`reappearance()` in analyze_from_hub.py scores a feature as reappearing if its best decoder
cosine into another seed clears theta. That is many-to-one: nothing stops half the anchor
dictionary from claiming the same partner, and in a dictionary as dense as ours (L0 ~ 347 of
2048) near-duplicate features are exactly what we would expect. This script re-scores the
same checkpoints under stricter rules and against a null, so the reported percentage can be
attributed to a cause.

Five checks, all on saved weights (no GPU, no activations):
  1. theta sweep                -- how much of the result rides on theta=0.7
  2. one-to-one (Hungarian)     -- forbid two anchor features sharing a partner
  3. rotation null              -- match against a randomly rotated partner dictionary
  4. within-dictionary duplicates -- how redundant the anchor dictionary is
  5. encoder+decoder agreement  -- Paulo & Belrose's stricter "shared latent" definition,
     which is what their headline 30-42% numbers measure. Ours is decoder-only, so the
     two are not comparable until this is computed.
"""

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from scipy.optimize import linear_sum_assignment

from seedcount_check import ANCHOR, discover, load_dec

THETAS = (0.5, 0.7, 0.9)
PRIMARY_THETA = 0.7
COMPARISON_SEEDS = (256, 1024)
BUDGETS = (100_000_000, 1_000_000_000, 8_000_000_000)
RNG = np.random.default_rng(0)


def random_rotation(d):
    """Haar-random orthogonal matrix: destroys correspondence, preserves dictionary geometry."""
    Q, R = np.linalg.qr(RNG.standard_normal((d, d)))
    return torch.tensor(Q * np.sign(np.diag(R)), dtype=torch.float32)


def load_enc(found, seed, tokens):
    """Encoder vectors as rows, to match load_dec's orientation. W_enc is (d_model, n_feat)."""
    repo, fname = found[(seed, tokens)]
    path = hf_hub_download(repo, fname, repo_type="model")
    W = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]["W_enc"]
    return F.normalize(W, dim=0).T


def fully_stable(match_masks):
    """Fraction of anchor features matched in EVERY comparison, as the plot defines it."""
    return np.stack(match_masks).all(axis=0).mean()


def shared_latent_mask(A_dec, A_enc, P_dec, P_enc, theta):
    """Paulo & Belrose: same partner under both encoder and decoder matching, both >= theta."""
    S_dec, S_enc = A_dec @ P_dec.T, A_enc @ P_enc.T
    j_dec, j_enc = S_dec.argmax(dim=1), S_enc.argmax(dim=1)
    agree = (j_dec == j_enc).numpy()
    best_dec = S_dec.max(dim=1).values.numpy()
    best_enc = S_enc.max(dim=1).values.numpy()
    return agree & (best_dec >= theta) & (best_enc >= theta), agree, best_enc


def audit(found, tokens):
    A = load_dec(found, ANCHOR, tokens)
    n_feat, d_model = A.shape
    partners = {s: load_dec(found, s, tokens) for s in COMPARISON_SEEDS}

    out = {"tokens": tokens, "n_feat": n_feat}

    # 1. theta sweep under the current many-to-one rule
    sims = {s: A @ P.T for s, P in partners.items()}
    best = {s: S.max(dim=1).values.numpy() for s, S in sims.items()}
    out["theta_sweep"] = {t: fully_stable([best[s] >= t for s in COMPARISON_SEEDS])
                          for t in THETAS}

    # 2. one-to-one matching: each partner feature can be claimed at most once
    one_to_one, collisions = {}, {}
    for s, S in sims.items():
        Sn = S.numpy()
        rows, cols = linear_sum_assignment(-Sn)
        assigned = np.full(n_feat, -1.0)
        assigned[rows] = Sn[rows, cols]
        one_to_one[s] = assigned
        collisions[s] = n_feat - len(np.unique(Sn.argmax(axis=1)))
    out["one_to_one"] = fully_stable([one_to_one[s] >= PRIMARY_THETA
                                      for s in COMPARISON_SEEDS])
    out["collisions"] = collisions

    # 3. null: same dictionaries, correspondence destroyed by a random rotation
    R = random_rotation(d_model)
    null_best = {s: (A @ (P @ R.T).T).max(dim=1).values.numpy() for s, P in partners.items()}
    out["null"] = fully_stable([null_best[s] >= PRIMARY_THETA for s in COMPARISON_SEEDS])
    out["null_median_cos"] = float(np.median(np.mean(list(null_best.values()), axis=0)))

    # 4. redundancy inside the anchor dictionary itself
    self_sim = A @ A.T
    self_sim.fill_diagonal_(-1.0)
    nearest = self_sim.max(dim=1).values.numpy()
    out["dup_rate"] = float((nearest >= PRIMARY_THETA).mean())
    out["median_nearest"] = float(np.median(nearest))

    # 5. the stricter "shared latent" definition the published percentages actually use
    A_enc = load_enc(found, ANCHOR, tokens)
    p_enc = {s: load_enc(found, s, tokens) for s in COMPARISON_SEEDS}
    masks, agrees, encs = {}, {}, {}
    for s in COMPARISON_SEEDS:
        masks[s], agrees[s], encs[s] = shared_latent_mask(
            A, A_enc, partners[s], p_enc[s], PRIMARY_THETA)
    out["shared_latent"] = fully_stable([masks[s] for s in COMPARISON_SEEDS])
    out["partner_agree"] = float(np.mean([agrees[s].mean() for s in COMPARISON_SEEDS]))
    out["enc_only"] = fully_stable([encs[s] >= PRIMARY_THETA for s in COMPARISON_SEEDS])
    out["median_enc_cos"] = float(np.median(np.mean(list(encs.values()), axis=0)))
    return out


def main():
    found = discover()
    results = [audit(found, t) for t in BUDGETS]

    print(f"\nAnchor seed {ANCHOR}, comparisons {list(COMPARISON_SEEDS)}, "
          f"'stable' = matched in every comparison\n")

    print("1. Sensitivity to the matching threshold")
    print(f"{'tokens':>14} | " + " | ".join(f"theta={t:<4}" for t in THETAS))
    print("-" * 52)
    for r in results:
        print(f"{r['tokens']:>14,} | " +
              " | ".join(f"{100*r['theta_sweep'][t]:8.1f}%" for t in THETAS))

    print("\n2. Many-to-one (reported) vs one-to-one (Hungarian), theta=0.7")
    print(f"{'tokens':>14} | {'reported':>9} | {'1-to-1':>8} | {'drop':>6} | "
          f"{'anchor features sharing a partner':>34}")
    print("-" * 92)
    for r in results:
        rep = r["theta_sweep"][PRIMARY_THETA]
        o2o = r["one_to_one"]
        shared = ", ".join(f"seed {s}: {c} ({100*c/r['n_feat']:.0f}%)"
                           for s, c in r["collisions"].items())
        print(f"{r['tokens']:>14,} | {100*rep:8.1f}% | {100*o2o:7.1f}% | "
              f"{100*(rep-o2o):5.1f} | {shared:>34}")

    print("\n3. Rotation null (correspondence destroyed) and dictionary redundancy")
    print(f"{'tokens':>14} | {'null stable%':>12} | {'null med cos':>12} | "
          f"{'dup rate':>9} | {'med nearest nbr':>15}")
    print("-" * 76)
    for r in results:
        print(f"{r['tokens']:>14,} | {100*r['null']:11.2f}% | {r['null_median_cos']:12.3f} | "
              f"{100*r['dup_rate']:8.1f}% | {r['median_nearest']:15.3f}")

    print("\n4. Decoder-only (ours) vs encoder+decoder 'shared latent' (published numbers)")
    print(f"{'tokens':>14} | {'dec only':>9} | {'enc only':>9} | {'enc AND dec':>12} | "
          f"{'same partner':>13} | {'med enc cos':>12}")
    print("-" * 88)
    for r in results:
        print(f"{r['tokens']:>14,} | {100*r['theta_sweep'][PRIMARY_THETA]:8.1f}% | "
              f"{100*r['enc_only']:8.1f}% | {100*r['shared_latent']:11.1f}% | "
              f"{100*r['partner_agree']:12.1f}% | {r['median_enc_cos']:12.3f}")


if __name__ == "__main__":
    main()
