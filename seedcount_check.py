"""Quantify how much the 5-seed -> 3-seed change inflates the stability curve.

The scaling plot mixes seed counts: 1M/50M/100M were run with 5 seeds, 1B-8B with 3.
"Fully stable" is a conjunction over (n_seeds - 1) comparisons, so it gets strictly
harder as seeds are added -- the two halves of the plot are not on the same axis.

All five seeds exist at 1M/50M/100M, so the size of that bias is directly measurable
there, and the per-comparison attrition it implies can be extrapolated to the 1B-8B
points where only three seeds exist. Decoder weights only: no GPU, no activations.
"""

import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

REPOS = ["deenais/sae-stability-pythia70m", "ndasari/SAE_project"]
ANCHOR = 42
THETA = 0.7
EPSILON = 0.05  # stability_check.py's endpoint binarization
CKPT_RE = re.compile(r"checkpoints/seed(\d+)_tokens(\d+)\.pt")


def discover():
    found = {}
    for repo in REPOS:
        for f in HfApi().list_repo_files(repo, repo_type="model"):
            m = CKPT_RE.fullmatch(f)
            if m:
                found.setdefault((int(m.group(1)), int(m.group(2))), (repo, f))
    return found


def load_dec(found, seed, tokens):
    repo, fname = found[(seed, tokens)]
    path = hf_hub_download(repo, fname, repo_type="model")
    W = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]["W_dec"]
    return F.normalize(W, dim=1)


def best_cosines(found, tokens, seeds):
    """Best decoder cosine from each anchor feature into each non-anchor seed."""
    A = load_dec(found, ANCHOR, tokens)
    return {s: (A @ load_dec(found, s, tokens).T).max(dim=1).values.numpy()
            for s in seeds if s != ANCHOR}


def summarize(best, comparison_seeds):
    """fully-stable fraction and mean p-hat over a chosen subset of comparisons."""
    M = np.stack([best[s] >= THETA for s in comparison_seeds])
    p = M.mean(axis=0)
    return (p >= 1 - EPSILON).mean(), p.mean(), (p <= EPSILON).mean()


def main():
    found = discover()
    by_tokens = defaultdict(list)
    for (seed, tokens) in found:
        by_tokens[tokens].append(seed)

    three = [256, 1024]  # comparison seeds for the 3-seed set (anchor 42)

    print(f"Decoder-cosine matching, theta={THETA}, anchor seed {ANCHOR}, "
          f"'stable' = matched in EVERY comparison\n")
    print(f"{'tokens':>14} | {'n':>2} | {'stable%':>8} | {'mean p':>7} | {'never%':>7}")
    print("-" * 56)

    cache = {}
    for tokens in sorted(by_tokens):
        seeds = sorted(by_tokens[tokens])
        best = best_cosines(found, tokens, [s for s in seeds if s != ANCHOR])
        cache[tokens] = best
        for label, comps in [("3", three), ("5", [s for s in seeds if s != ANCHOR])]:
            if not all(s in best for s in comps):
                continue
            if label == "5" and len(comps) == len(three):
                continue  # only three seeds exist here, nothing extra to report
            st, mp, nv = summarize(best, comps)
            print(f"{tokens:>14,} | {label:>2} | {100*st:7.1f}% | {mp:7.3f} | {100*nv:6.1f}%")

    print("\n" + "=" * 56)
    print("Attrition per added comparison (why the two halves differ)")
    print("=" * 56)
    print(f"{'tokens':>14} | " + " | ".join(f"k={k}" for k in (1, 2, 3, 4)) + " |  ratio f2/f1")
    for tokens in sorted(cache):
        best = cache[tokens]
        order = [s for s in (256, 1024, 137, 512) if s in best]
        fs = [summarize(best, order[:k])[0] for k in range(1, len(order) + 1)]
        cells = " | ".join(f"{100*fs[i]:4.1f}" if i < len(fs) else "   -" for i in range(4))
        ratio = fs[1] / fs[0] if len(fs) > 1 and fs[0] > 0 else float("nan")
        print(f"{tokens:>14,} | {cells} |  {ratio:.3f}")

    # Attrition slows as comparisons are added, so assuming a constant per-comparison
    # ratio underestimates the 5-seed value. Calibrate that slowdown on the budgets where
    # all four comparisons exist, then apply it to the 3-seed-only budgets.
    print("\nEstimating the 5-seed value at the 3-seed budgets")
    print("  f(k) = fully-stable fraction using k comparisons")
    print("  naive: f(4) = f(2) * (f(2)/f(1))^2   [assumes constant attrition -> a floor]")
    print("  calibrated: scales (f4/f2)/(f2/f1), measured where k=4 is observed\n")

    calib = []
    for tokens in sorted(cache):
        best = cache[tokens]
        order = [s for s in (256, 1024, 137, 512) if s in best]
        fs = [summarize(best, order[:k])[0] for k in range(1, len(order) + 1)]
        if len(fs) > 3 and fs[0] > 0:
            calib.append((fs[3] / fs[1]) / (fs[1] / fs[0]))
    slowdown = float(np.mean(calib)) if calib else 1.0
    print(f"  attrition slowdown factor measured at 50M/100M: {slowdown:.3f}\n")

    for tokens in sorted(cache):
        best = cache[tokens]
        order = [s for s in (256, 1024, 137, 512) if s in best]
        fs = [summarize(best, order[:k])[0] for k in range(1, len(order) + 1)]
        if fs[0] == 0:
            continue
        r = fs[1] / fs[0]
        naive = fs[1] * r ** 2
        est = fs[1] * min(r * slowdown, 1.0)
        obs = f"{100*fs[3]:5.1f}%" if len(fs) > 3 else "    n/a"
        print(f"  {tokens:>14,}  reported(3-seed) {100*fs[1]:5.1f}%   "
              f"floor {100*naive:5.1f}%   calibrated {100*est:5.1f}%   observed {obs}")

    print("\nMatch margin at the largest budget (how fragile the matches are):")
    for tokens in sorted(cache)[-1:]:
        allbest = np.mean([cache[tokens][s] for s in cache[tokens]], axis=0)
        for lo, hi in [(0.7, 0.75), (0.75, 0.9), (0.9, 1.01)]:
            frac = ((allbest >= lo) & (allbest < hi)).mean()
            print(f"  {tokens:,}: mean best-cos in [{lo}, {hi}): {100*frac:.1f}%")


if __name__ == "__main__":
    main()
