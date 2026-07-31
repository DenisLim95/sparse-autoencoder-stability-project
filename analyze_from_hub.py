"""Rebuild the stability-vs-tokens result from checkpoints on the Hugging Face Hub.

Everything here works from decoder weights alone, so it needs no GPU, no activations and no
access to the training machine. That covers the stability scaling curve, the matching
threshold sweep, and geometric isolation as a standalone predictor.

The three statistics that DO need activations -- activation frequency, reconstruction
contribution and mean activation strength -- are not computed here; they need a forward pass
over real tokens and belong in a short GPU job (see README of results printed at the end).

Usage:
    export HF_TOKEN=hf_...              # needed only for private repos
    python analyze_from_hub.py

Merges checkpoints across repos, preferring the first repo listed when both have the same
(seed, token count).
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

REPOS = [
    "deenais/sae-stability-pythia70m",  # our 2B/3B/5B/8B run
    "ndasari/SAE_project",              # earlier 1M/50M/100M and the 1B resume point
]
SEEDS = [42, 256, 1024]
THETAS = [0.5, 0.7, 0.9]
PRIMARY_THETA = 0.7
OUT_DIR = Path("outputs")

CKPT_RE = re.compile(r"checkpoints/seed(\d+)_tokens(\d+)\.pt")


def discover():
    """Map (seed, tokens) -> (repo, filename), preferring earlier repos in REPOS."""
    found = {}
    for repo in REPOS:
        try:
            files = HfApi().list_repo_files(repo, repo_type="model")
        except Exception as e:
            print(f"  {repo}: unreadable ({type(e).__name__}) -- skipping")
            continue
        n = 0
        for f in files:
            m = CKPT_RE.fullmatch(f)
            if not m:
                continue
            seed, tokens = int(m.group(1)), int(m.group(2))
            if seed in SEEDS and (seed, tokens) not in found:
                found[(seed, tokens)] = (repo, f)
                n += 1
        print(f"  {repo}: {n} usable checkpoint(s)")
    return found


def load_decoders(found, tokens):
    """Return {seed: W_dec} for one token count, or None if any seed is missing."""
    W = {}
    for seed in SEEDS:
        key = (seed, tokens)
        if key not in found:
            return None
        repo, fname = found[key]
        path = hf_hub_download(repo, fname, repo_type="model")
        W[seed] = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]["W_dec"]
    return W


def reappearance(W, theta):
    """Decoder-only many-to-one argmax matching, anchored on SEEDS[0] (Gerasimov et al.)."""
    A = F.normalize(W[SEEDS[0]], dim=1)
    p = np.zeros(A.shape[0])
    best_all = []
    for seed in SEEDS[1:]:
        best = (A @ F.normalize(W[seed], dim=1).T).max(dim=1).values.numpy()
        p += best >= theta
        best_all.append(best)
    return p / (len(SEEDS) - 1), np.mean(best_all, axis=0)


def geometric_isolation(W_dec, k=10):
    """Mean cosine similarity to the k nearest neighbours within the same dictionary."""
    Wn = F.normalize(W_dec, dim=1)
    sim = Wn @ Wn.T
    sim.fill_diagonal_(-1.0)  # exclude self
    return sim.topk(k, dim=1).values.mean(dim=1).numpy()


def auroc(scores, labels):
    """Rank-based AUROC, so this file needs no sklearn."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties so constant predictors score 0.5 rather than something spurious
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    print("Discovering checkpoints")
    found = discover()
    if not found:
        print("\nNo checkpoints readable. For a private repo, set HF_TOKEN to a read token.")
        return 1

    token_counts = sorted({t for (_, t) in found})
    complete = [t for t in token_counts if all((s, t) in found for s in SEEDS)]
    incomplete = [t for t in token_counts if t not in complete]
    print(f"\nComplete for all {len(SEEDS)} seeds {SEEDS}: {[f'{t:,}' for t in complete]}")
    for tokens in incomplete:
        missing = [s for s in SEEDS if (s, tokens) not in found]
        print(f"Incomplete, skipped: {tokens:,} is missing seed(s) {missing}")
    if not complete:
        print("Nothing analysable.")
        return 1

    print(f"\n{'=' * 78}")
    print(f"Feature stability vs training tokens "
          f"({len(SEEDS)} seeds, {len(SEEDS)-1} comparisons, anchor seed {SEEDS[0]})")
    print("=" * 78)
    header = f"{'tokens':>15} | " + " | ".join(f'p̂@θ={t}' for t in THETAS)
    print(header + f" | {'median cos':>10} | {'stable%':>8} | {'geom AUROC':>10}")
    print("-" * len(header + " | median cos |  stable% | geom AUROC"))

    rows = []
    for tokens in complete:
        W = load_decoders(found, tokens)
        ps = {}
        for theta in THETAS:
            p, best = reappearance(W, theta)
            ps[theta] = p
        p_primary = ps[PRIMARY_THETA]
        _, best = reappearance(W, PRIMARY_THETA)

        # Hypothesis 1, partial: can geometry alone separate stable from unstable?
        # Sign is negated because the proposal predicts ISOLATED (low neighbour sim) = stable.
        iso = geometric_isolation(W[SEEDS[0]])
        geom_auroc = auroc(-iso, p_primary >= 0.5)

        print(f"{tokens:>15,} | " + " | ".join(f'{ps[t].mean():7.3f}' for t in THETAS)
              + f" | {np.median(best):10.3f} | {100*(p_primary>=0.5).mean():7.1f}% | {geom_auroc:10.3f}")
        rows.append(dict(tokens=tokens, p_hat=p_primary, isolation=iso,
                         best_cos=best, geom_auroc=geom_auroc))

    OUT_DIR.mkdir(exist_ok=True)
    np.savez(
        OUT_DIR / "stability_from_hub.npz",
        token_counts=np.array([r["tokens"] for r in rows]),
        mean_p_hat=np.array([r["p_hat"].mean() for r in rows]),
        frac_stable=np.array([(r["p_hat"] >= 0.5).mean() for r in rows]),
        geom_auroc=np.array([r["geom_auroc"] for r in rows]),
        p_hat_final=rows[-1]["p_hat"],
        isolation_final=rows[-1]["isolation"],
    )
    print(f"\nSaved {OUT_DIR / 'stability_from_hub.npz'}")

    print(f"\n{'=' * 78}")
    print("Notes")
    print("=" * 78)
    print(f"- 'stable%' uses the proposal's p̂ >= 0.5 grouping. Watch the class balance: a")
    print(f"  heavily skewed positive class makes the AUROC >= 0.75 bar easy but uninformative.")
    print(f"- 'geom AUROC' is geometric isolation as a SINGLE predictor, sign-corrected so that")
    print(f"  >0.5 means isolated features are more stable (the proposal's prediction) and <0.5")
    print(f"  means crowded features are, which is what the 1M run found.")
    print(f"- Decoder norm is omitted: normalize_decoder() pins it to 1.0, so it carries no signal.")
    print(f"- Activation frequency, reconstruction contribution and mean activation strength need")
    print(f"  a forward pass over real tokens. That is a short GPU job (~1-2h on a free Colab T4")
    print(f"  for 100M tokens) and is all that stands between these checkpoints and a full")
    print(f"  Hypothesis 1 result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
