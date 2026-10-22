# -*- coding: utf-8 -*-
"""Activation overlap of cross-seed matched features: is a cosine match the same feature?

The stability label and the strongest predictor are both decoder-cosine quantities. A latent is
"stable" when some latent in each partner seed has decoder cosine >= theta with it; neighbour
crowding is its mean cosine to the ten nearest latents in its own dictionary. In a dense region
of the sphere some partner latent will land within theta of the anchor whether or not it is
functionally the same feature, so crowding could be predicting the matching rule rather than
reproducibility. The rotation null rules out static sphere density (destroy correspondence and
the signal vanishes) but cannot tell "the same feature" from "a neighbour in the same crowd".

This script measures the match at the level of what the latents DO. For each anchor latent and
its best cosine match in each partner seed, it accumulates over the eval set whether the pair
fires on the same tokens (Jaccard of firing sets) and how their activations co-vary (Pearson
over all tokens, and activation cosine). Two baselines are scored on the same tokens:

    second              the SECOND-best cosine match in the partner: in a crowded region it also
                        sits near theta, so if the best match agrees on activations and the
                        second does not, the match singles out one specific latent rather than
                        "anything in the region". This is the direct test of the objection.
    own nearest         the anchor's nearest neighbour INSIDE its own dictionary: the "neighbour
                        in the crowd" itself. One-directional: within one TopK dictionary
                        neighbours compete for the same slots, so low agreement is expected;
                        high agreement would be the bad news.
    random              a uniformly random partner latent (uniform permutation, no collisions
                        with the true match): the floor.
    freq-matched        a random partner latent drawn from the same firing-rate decile as the
                        best match. Agreement between sparse latents scales with firing rate,
                        so a dense-firing anchor gets random overlap for free; this is the
                        null the crowded deciles have to beat, and the one the strict-label
                        cutoff is taken from.

Agreement is then reported by crowding decile. Flat or rising match agreement across deciles,
with own-nearest agreement well below it, means a crowded latent's match is specifically that
latent and not any nearby direction. Falling match agreement in the top deciles, with own-
nearest close behind, is the objection's scenario and is reported as such.

Finally the classifier is re-scored under a STRICT label that requires activation agreement on
top of the cosine match in both partners, through the same `cv_auroc` the paper uses. The
original label is run through the identical code path first, so the published numbers must
reproduce before anything downstream is read.

Framing: this checks that the match is functional, not only geometric, in the regime where the
two could diverge. It is not a test of the inherited label convention in general.

Pearson over all tokens is reported with Jaccard and activation cosine; for sparse latents
Pearson and cosine nearly coincide (the mean correction is tiny at firing rates ~1e-3), which is
expected and not a bug. The optional logit stage compares direct logit effects (W_dec @ W_U);
note that this is a fixed linear reweighting of the same decoder rows, so it tracks decoder
cosine closely and speaks to "same downstream effect" only in the direct-path sense. The
activation overlap is the independent evidence.

No training. The only activation-touching work is one streaming pass that encodes the eval set
with three existing SAEs; nothing of size (tokens x latents) is ever materialised.

Usage (each stage caches; `analyze` and `logits` never touch the activations):

    python activation_overlap.py all --k 64              # pairs, analyze
    python activation_overlap.py all --k 64 --logits     # plus direct-logit-effect cosine
    python activation_overlap.py analyze --k 64 --cutoff 0.3 --cutoff 0.5 --cutoff 0.7

Environment:
    SAE_HF_REPO        default deenais/sae-stability-pythia70m
    SAE_EVAL_BATCHES   default 40 (40 x 256 x 128 = 1,310,720 tokens, matching the sweep)
    SAE_CACHE_DIR      default ./cache (eval activations and the per-latent stats npz)
    SAE_RESULTS_DIR    default ./outputs/activation_overlap
    SAE_MIN_FIRINGS    default 100, read by sae_stats
    HF_TOKEN           only needed if the repo is private
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from eval_activations import BATCH_SIZE, SEQ_LEN, build_eval_activations, load_model
from sae_stats import (
    DEFAULT_PREFIX,
    DEFAULT_REPO,
    EPSILON,
    MIN_FIRINGS,
    STAT_BATCH,
    THETA,
    build_predictors,
    complete_cells,
    compute_reappearance_probability,
    compute_single_run_statistics,
    cv_auroc,
    discover,
    load_sae,
)

MODEL_NAME = "pythia-70m-deduped"
HOOK_POINT = "blocks.3.hook_resid_post"

N_EVAL_BATCHES = int(os.environ.get("SAE_EVAL_BATCHES") or 40)
CACHE_DIR = Path(os.environ.get("SAE_CACHE_DIR") or "cache")
RESULTS_DIR = Path(os.environ.get("SAE_RESULTS_DIR") or "outputs/activation_overlap")
# float64 accumulators: cuda or cpu only, no mps.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same keys and file name as budget_transfer.py, so the stats npz written here is the one
# budget_transfer.py and rotation_null.py --bands read, and vice versa.
STAT_KEYS = [
    "activation_freq", "mean_activation", "firing_counts", "geometric_isolation",
    "recon_contribution", "recon_contribution_uncond", "encoder_norm", "encoder_bias",
    "decoder_norm",
]

# Looked up by name, as permutation_null.py does, so a reordering of build_predictors cannot
# silently redefine which column is the baseline and which is the geometric statistic.
FREQ_PREDICTOR = "Activation Freq (log)"
CROWDING_PREDICTOR = "Geometric Isolation"

# Pairings against a partner seed, and the one against the anchor's own dictionary.
PARTNER_PAIRINGS = ("match", "second", "random", "randf")
PAIRINGS = PARTNER_PAIRINGS + ("own",)
N_DECILES = 10
N_FREQ_STRATA = 10


def stats_path(k, seed, tokens):
    return CACHE_DIR / f"stats_k{k}_seed{seed}_t{tokens}_e{N_EVAL_BATCHES}.npz"


def pairs_path(k, tokens):
    return RESULTS_DIR / f"pairs_k{k}_t{tokens}_e{N_EVAL_BATCHES}.npz"


# --------------------------------------------------------------------------------------
# The accumulator
# --------------------------------------------------------------------------------------


class PairAccumulator:
    """Streaming sums for one pairing of every anchor latent with one partner latent.

    `update` takes the anchor's activations and the partner's activations ALREADY GATHERED into
    anchor column order, so column j of both is the pair (anchor j, partner idx[j]). Eight
    per-column sums in float64 are enough to recover Jaccard, Pearson and cosine exactly at the
    end, which is what keeps this at (batch x latents) memory instead of (tokens x latents).
    """

    def __init__(self, n_features: int, device: str = "cpu"):
        z = lambda: torch.zeros(n_features, dtype=torch.float64, device=device)  # noqa: E731
        self.n_a, self.n_p, self.n_both = z(), z(), z()
        self.s_a, self.s_p, self.s_ap, self.s_a2, self.s_p2 = z(), z(), z(), z(), z()

    @torch.no_grad()
    def update(self, F_a: torch.Tensor, G: torch.Tensor):
        fa, fp = F_a > 0, G > 0
        self.n_a += fa.sum(dim=0).double()
        self.n_p += fp.sum(dim=0).double()
        self.n_both += (fa & fp).sum(dim=0).double()
        Fa, Gd = F_a.double(), G.double()
        self.s_a += Fa.sum(dim=0)
        self.s_p += Gd.sum(dim=0)
        self.s_ap += (Fa * Gd).sum(dim=0)
        self.s_a2 += (Fa * Fa).sum(dim=0)
        self.s_p2 += (Gd * Gd).sum(dim=0)

    def sums(self):
        return {name: getattr(self, name).cpu().numpy() for name in
                ("n_a", "n_p", "n_both", "s_a", "s_p", "s_ap", "s_a2", "s_p2")}

    def finalize(self, n_tokens: int):
        return finalize_sums(self.sums(), n_tokens)


def finalize_sums(s, n_tokens: int):
    """Jaccard, Pearson over all tokens, and activation cosine from the eight sums.

    NaN wherever the quantity is undefined: Jaccard when neither latent ever fires, Pearson
    when either side has zero variance (a dead latent), cosine when either norm is zero. These
    are left as NaN rather than 0.0 so that a dead partner cannot read as "disagrees".
    """
    N = float(n_tokens)
    union = s["n_a"] + s["n_p"] - s["n_both"]
    jaccard = np.divide(s["n_both"], union, out=np.full_like(union, np.nan), where=union > 0)

    mean_a, mean_p = s["s_a"] / N, s["s_p"] / N
    cov = s["s_ap"] / N - mean_a * mean_p
    var_a = s["s_a2"] / N - mean_a ** 2
    var_p = s["s_p2"] / N - mean_p ** 2
    # Rounding can push a true zero variance a hair negative; clip before the sqrt and treat
    # anything at or below zero as undefined.
    denom = np.sqrt(np.clip(var_a, 0, None) * np.clip(var_p, 0, None))
    ok = (var_a > 0) & (var_p > 0)
    pearson = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=ok)

    norm = np.sqrt(s["s_a2"] * s["s_p2"])
    cosine = np.divide(s["s_ap"], norm, out=np.full_like(norm, np.nan), where=norm > 0)

    return {"jaccard": jaccard, "pearson": pearson, "act_cosine": cosine,
            "n_both": s["n_both"], "n_a": s["n_a"], "n_p": s["n_p"]}


def top2_cross(A: torch.Tensor, P: torch.Tensor, exclude_self: bool = False, chunk: int = 1024):
    """Best and second-best cosine match of each row of A among the rows of P.

    Chunked like compute_geometric_isolation. With exclude_self (P is A) the diagonal is masked,
    so "best" is the nearest OTHER row: the anchor's own nearest neighbour.
    Returns (best_idx, best_cos, second_idx, second_cos) as numpy arrays.
    """
    A, P = F.normalize(A, dim=1), F.normalize(P, dim=1)
    n = A.shape[0]
    idx = torch.empty(n, 2, dtype=torch.long, device=A.device)
    cos = torch.empty(n, 2, device=A.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = A[start:stop] @ P.T
        if exclude_self:
            rows = torch.arange(start, stop, device=A.device)
            sims[rows - start, rows] = -float("inf")
        top = sims.topk(2, dim=1)
        idx[start:stop], cos[start:stop] = top.indices, top.values
    idx, cos = idx.cpu().numpy(), cos.cpu().numpy()
    return idx[:, 0], cos[:, 0], idx[:, 1], cos[:, 1]


def nearest_within(W: torch.Tensor, chunk: int = 1024):
    """Index and cosine of each row's nearest OTHER row."""
    best_idx, best_cos, _, _ = top2_cross(W, W, exclude_self=True, chunk=chunk)
    return best_idx, best_cos


def random_partner(best_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A uniform permutation conditioned on never coinciding with the true match."""
    n = len(best_idx)
    perm = rng.permutation(n)
    for _ in range(100):
        clash = np.flatnonzero(perm == best_idx)
        if len(clash) == 0:
            return perm
        # Swap each clashing position with a random other one; still a permutation.
        other = rng.integers(0, n, size=len(clash))
        perm[clash], perm[other] = perm[other], perm[clash]
    raise RuntimeError("could not build a collision-free random pairing")


def frequency_matched_partner(best_idx: np.ndarray, partner_freq: np.ndarray,
                              rng: np.random.Generator, n_strata: int = N_FREQ_STRATA):
    """For each anchor, a random partner latent from the best match's firing-rate decile.

    Strata are deciles of log10 frequency over all partner latents; the draw excludes the
    best match itself. Sampling is with replacement: this is a baseline, not a matching.
    """
    logf = np.log10(partner_freq + 1e-10)
    edges = np.quantile(logf, np.linspace(0, 1, n_strata + 1))
    stratum = np.clip(np.searchsorted(edges[1:-1], logf, side="right"), 0, n_strata - 1)
    members = [np.flatnonzero(stratum == s) for s in range(n_strata)]
    out = np.empty(len(best_idx), dtype=np.int64)
    for i, j in enumerate(best_idx):
        pool = members[stratum[j]]
        if len(pool) > 1:
            pool = pool[pool != j]
        out[i] = pool[rng.integers(len(pool))]
    return out


# --------------------------------------------------------------------------------------
# Stage: pairs
# --------------------------------------------------------------------------------------


def ensure_stats(found, k, seed, tokens, activations, repo):
    """The anchor's six statistics, computed once and cached under budget_transfer's name."""
    path = stats_path(k, seed, tokens)
    if path.exists():
        return
    print(f"Computing statistics for k={k} seed={seed} @ {tokens:,} (not cached)")
    sae = load_sae(found, k, seed, tokens, device=DEVICE, repo=repo)
    stats = compute_single_run_statistics(sae, activations, DEVICE, label=f"seed {seed}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: stats[key] for key in STAT_KEYS})
    del sae
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def run_pairs(found, k, tokens, anchor, partners, rng, repo):
    activations = build_eval_activations(MODEL_NAME, HOOK_POINT, DEVICE, cache_dir=CACHE_DIR)
    n_tokens = len(activations)
    # The anchor's statistics feed the strict-label re-score; the partners' firing rates are
    # what the frequency-matched baseline is drawn from.
    for seed in [anchor] + partners:
        ensure_stats(found, k, seed, tokens, activations, repo)

    # Anchor first: compute_reappearance_probability takes the first key as the anchor and
    # returns one similarity / index array per partner in the remaining key order.
    saes = {s: load_sae(found, k, s, tokens, device=DEVICE, repo=repo) for s in [anchor] + partners}
    probs, info = compute_reappearance_probability(saes, theta=THETA)
    n = saes[anchor].n_features
    W_a = saes[anchor].W_dec.detach()

    idx, cosine = {}, {}
    for i, p in enumerate(partners):
        best_idx, best_cos, second_idx, second_cos = top2_cross(W_a, saes[p].W_dec.detach())
        # The label's own best match is the one from compute_reappearance_probability; the
        # second-best must be on the same footing, so the two best-match computations have to
        # agree (ties aside) or the control is not controlling for anything.
        disagree = int((best_idx != info["best_match_idx"][i]).sum())
        if disagree > n // 1000:
            raise RuntimeError(f"best match disagrees with the label's on {disagree} latents")
        idx[("match", p)] = info["best_match_idx"][i]
        cosine[("match", p)] = info["similarities"][i]
        idx[("second", p)] = second_idx
        cosine[("second", p)] = second_cos
        idx[("random", p)] = random_partner(idx[("match", p)], rng)
        with np.load(stats_path(k, p, tokens)) as z:
            partner_freq = z["activation_freq"]
        idx[("randf", p)] = frequency_matched_partner(idx[("match", p)], partner_freq, rng)
    own_idx, own_cos = nearest_within(W_a)

    to_dev = lambda a: torch.as_tensor(a, dtype=torch.long, device=DEVICE)  # noqa: E731
    idx_t = {key: to_dev(a) for key, a in idx.items()}
    own_t = to_dev(own_idx)

    acc = {key: PairAccumulator(n, DEVICE) for key in idx}
    acc[("own", anchor)] = PairAccumulator(n, DEVICE)

    print(f"Streaming {n_tokens:,} tokens through {len(saes)} SAEs on {DEVICE}")
    with torch.no_grad():
        for start in tqdm(range(0, n_tokens, STAT_BATCH), desc="Pair agreement"):
            x = activations[start : start + STAT_BATCH].to(DEVICE)
            F_a = saes[anchor].encode(x)
            acc[("own", anchor)].update(F_a, F_a.index_select(1, own_t))
            for p in partners:
                F_p = saes[p].encode(x)
                for pairing in PARTNER_PAIRINGS:
                    acc[(pairing, p)].update(F_a, F_p.index_select(1, idx_t[(pairing, p)]))

    out = {
        "anchor": anchor, "partner_seeds": np.array(partners), "n_tokens": n_tokens,
        "probs": probs, "own_idx": own_idx, "own_cos": own_cos,
    }
    for (pairing, p), a in idx.items():
        out[f"{pairing}_{p}_idx"] = a
    for (pairing, p), c in cosine.items():
        out[f"{pairing}_{p}_cos"] = c
    for (pairing, seed), a in acc.items():
        tag = pairing if pairing == "own" else f"{pairing}_{seed}"
        for name, arr in a.sums().items():
            out[f"sum_{tag}_{name}"] = arr
        for name, arr in a.finalize(n_tokens).items():
            out[f"{tag}_{name}"] = arr

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(pairs_path(k, tokens), **out)
    print(f"Wrote {pairs_path(k, tokens)}")

    del saes, activations
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Stage: logits (optional)
# --------------------------------------------------------------------------------------


def run_logits(found, k, tokens, anchor, partners, repo):
    """Cosine between the direct logit effects (W_dec @ W_U) of each matched pair.

    transformer_lens folds ln_final into W_U by default, the standard direct-logit-attribution
    setup. Chunked so the (latents x vocab) matrix is never built in full. Because W_U is a
    fixed linear map applied to both decoder rows, this cosine is a reweighted decoder cosine
    and will track it; it is the direct-path view only (layer 3 of 6).
    """
    path = pairs_path(k, tokens)
    if not path.exists():
        raise SystemExit(f"{path} missing: run the pairs stage first.")
    pairs = dict(np.load(path, allow_pickle=False))

    model = load_model(MODEL_NAME, DEVICE)
    W_U = model.W_U.detach()
    saes = {s: load_sae(found, k, s, tokens, device=DEVICE, repo=repo) for s in [anchor] + partners}
    W_a = saes[anchor].W_dec.detach()

    def logit_cosine(W_p, idx, chunk=1024):
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=DEVICE)
        out = np.empty(len(idx), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(idx), chunk):
                stop = min(start + chunk, len(idx))
                La = W_a[start:stop] @ W_U
                Lp = W_p.index_select(0, idx_t[start:stop]) @ W_U
                out[start:stop] = F.cosine_similarity(La, Lp, dim=1).double().cpu().numpy()
        return out

    for p in partners:
        W_p = saes[p].W_dec.detach()
        for pairing in PARTNER_PAIRINGS:
            pairs[f"{pairing}_{p}_logit_cos"] = logit_cosine(W_p, pairs[f"{pairing}_{p}_idx"])
    pairs["own_logit_cos"] = logit_cosine(W_a, pairs["own_idx"])

    np.savez(path, **pairs)
    print(f"Added logit-effect cosines to {path}")

    del model, saes
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# Stage: analyze
# --------------------------------------------------------------------------------------


def load_cell(k, tokens):
    pairs = dict(np.load(pairs_path(k, tokens), allow_pickle=False))
    with np.load(stats_path(k, int(pairs["anchor"]), tokens)) as z:
        stats = {key: z[key] for key in z.files}
    return pairs, stats


def per_feature(pairs, partners, metric, pairing):
    """(n_partners, n) array of one metric for one pairing, in partner order."""
    if pairing == "own":
        return pairs[f"own_{metric}"][None, :]
    return np.stack([pairs[f"{pairing}_{p}_{metric}"] for p in partners])


def combine_partners(a):
    """Feature-level value from the (partners, n) array: the WORSE partner.

    Stable requires a match in both partners, so a latent whose match is specific in one seed
    and loose in the other is loose. np.minimum, not nanmin: a NaN (dead or under-floor
    partner latent) must propagate, because a decoder match to a latent that never fires is
    not functional reproduction.
    """
    out = a[0].copy()
    for row in a[1:]:
        out = np.minimum(out, row)
    return out


def decile_table(pairs, stats, partners, k, tokens, anchor):
    crowding = stats["geometric_isolation"]
    live = stats["firing_counts"] >= MIN_FIRINGS
    probs = pairs["probs"]
    stable = probs >= (1 - EPSILON)
    matched = np.stack([pairs[f"match_{p}_cos"] >= THETA for p in partners])  # (P, n)

    pear = {pr: per_feature(pairs, partners, "pearson", pr) for pr in PAIRINGS}
    jacc = {pr: per_feature(pairs, partners, "jaccard", pr) for pr in PAIRINGS}
    has_logits = "own_logit_cos" in pairs
    if has_logits:
        logit = {pr: per_feature(pairs, partners, "logit_cos", pr) for pr in PAIRINGS}

    pear_match_min = combine_partners(pear["match"])
    # Matched-only agreement: over (partner, latent) cells where that partner's cosine clears
    # theta, so latents matched in one seed only still contribute that seed's number.
    pear_match_any = np.where(matched, pear["match"], np.nan)

    def med(a):
        a = np.asarray(a, dtype=float)
        return float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")

    edges = np.quantile(crowding[live], np.linspace(0, 1, N_DECILES + 1))
    bands = []
    for i in range(N_DECILES):
        lo, hi = edges[i], edges[i + 1]
        upper = crowding <= hi if i == N_DECILES - 1 else crowding < hi
        bands.append((str(i + 1), live & (crowding >= lo) & upper))
    bands.append(("all", live))

    rows = []
    for label, band in bands:
        if band.sum() < 20:
            continue
        sb = band & stable
        row = dict(
            k=k, tokens=tokens, anchor=anchor, decile=label,
            n=int(band.sum()), n_stable=int(sb.sum()),
            stable_frac=float(stable[band].mean()),
            crowding_mean=float(crowding[band].mean()),
            match_cos_median=med(np.stack([pairs[f"match_{p}_cos"] for p in partners])[:, band]),
            second_cos_median=med(np.stack([pairs[f"second_{p}_cos"] for p in partners])[:, band]),
            own_nn_cos_median=med(pairs["own_cos"][band]),
            pearson_match_median=med(pear_match_min[sb]),
            pearson_match_median_all_matched=med(pear_match_any[:, band]),
            jaccard_match_median=med(combine_partners(jacc["match"])[sb]),
            frac_stable_pearson_ge_0p5=float(np.mean(pear_match_min[sb] >= 0.5))
            if sb.any() else float("nan"),
        )
        # The controls, scored on the same latents as the match (stable ones).
        for pr, name in (("second", "second"), ("own", "own_nn"), ("random", "random"),
                         ("randf", "randf")):
            row[f"pearson_{name}_median"] = med(combine_partners(pear[pr])[sb])
            row[f"jaccard_{name}_median"] = med(combine_partners(jacc[pr])[sb])
        if has_logits:
            row["logit_cos_match_median"] = med(combine_partners(logit["match"])[sb])
            for pr, name in (("second", "second"), ("own", "own_nn"), ("random", "random"),
                             ("randf", "randf")):
                row[f"logit_cos_{name}_median"] = med(combine_partners(logit[pr])[sb])
        rows.append(row)
    return pd.DataFrame(rows), pear_match_min, pear["random"], pear["randf"]


def strict_label_table(pairs, stats, partners, k, tokens, anchor, cutoffs, pear_match_min,
                       pear_random, pear_randf):
    """Re-score the classifier when 'stable' also requires activation agreement.

    The original label (cutoff -inf) goes through this exact code path first; it must reproduce
    the published numbers or the rest of the table means nothing.
    """
    names = [name for name, _ in build_predictors(stats)]
    X = np.column_stack([values for _, values in build_predictors(stats)])
    for predictor in (FREQ_PREDICTOR, CROWDING_PREDICTOR):
        if predictor not in names:
            raise SystemExit(f"build_predictors no longer emits {predictor!r} (it emits {names}).")
    freq_col, crowding_col = names.index(FREQ_PREDICTOR), names.index(CROWDING_PREDICTOR)

    live = stats["firing_counts"] >= MIN_FIRINGS
    finite = np.isfinite(X).all(axis=1)
    probs = pairs["probs"]
    stable, unstable = probs >= (1 - EPSILON), probs <= EPSILON

    # Data-driven cutoffs: what random pairs reach by chance over live latents, for the
    # uniform and the frequency-matched draw. The latter is the honest one.
    c95 = {"c95_random": float(np.nanpercentile(pear_random[:, live], 95)),
           "c95_randf": float(np.nanpercentile(pear_randf[:, live], 95))}
    resolved = ([("original", -np.inf)] + [(f"{c:g}", float(c)) for c in cutoffs]
                + list(c95.items()))

    rows = []
    for name, c in resolved:
        if name == "original":
            # The paper's label exactly, dead partners and all: this row must reproduce the
            # published numbers before any other row is read.
            agrees = np.ones_like(stable)
        else:
            # NaN agreement (a partner match that never fires) compares False: demoted.
            with np.errstate(invalid="ignore"):
                agrees = pear_match_min >= c
        strict = stable & agrees
        loose = stable & ~agrees  # cosine-matched in both seeds, but not on activations
        for variant, unstable_v in (("drop", unstable), ("relabel", unstable | loose)):
            mask = (strict | unstable_v) & live & finite
            full = cv_auroc(X, strict, mask)
            freq = cv_auroc(X[:, freq_col], strict, mask)
            rows.append(dict(
                k=k, tokens=tokens, anchor=anchor, cutoff_name=name, cutoff=c, variant=variant,
                n_labelled=int(mask.sum()), n_reclassified=int(loose.sum()),
                stable_frac=float(strict[mask].mean()) if mask.any() else float("nan"),
                auroc_full=full, auroc_freq=freq,
                auroc_crowding=cv_auroc(X[:, crowding_col], strict, mask),
                delta=full - freq,
            ))
    return pd.DataFrame(rows), c95


def plot_agreement(deciles, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; skipping {out.name} (the CSVs hold everything).")
        return

    d = deciles[deciles.decile != "all"]
    x = d.decile.astype(int)
    panels = [("pearson", "median Pearson r (all tokens)"), ("jaccard", "median Jaccard of firing sets")]
    if "logit_cos_match_median" in d:
        panels.append(("logit_cos", "median cosine of direct logit effects"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.0))
    axes = np.atleast_1d(axes)
    series = [("match", "best cross-seed match"),
              ("second", "second-best cross-seed match"),
              ("own_nn", "own nearest neighbour (same seed)"),
              ("randf", "random, frequency-matched"),
              ("random", "random partner latent")]
    for ax, (metric, ylabel) in zip(axes, panels):
        for key, label in series:
            col = f"{metric}_{key}_median"
            if col in d:
                ax.plot(x, d[col], marker="o", label=label)
        ax.set_xlabel("neighbour-crowding decile (1 = most isolated); stable latents")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def report(deciles, strict, c95):
    d = deciles.set_index("decile")
    allrow = d.loc["all"]
    print("\n--- Activation agreement of matched pairs (stable live latents) ---")
    for col, label in (("pearson_match_median", "best cross-seed match"),
                       ("pearson_second_median", "second-best cross-seed match"),
                       ("pearson_own_nn_median", "own nearest neighbour (same seed)"),
                       ("pearson_randf_median", "random, frequency-matched"),
                       ("pearson_random_median", "random partner latent")):
        print(f"  median Pearson, {label:<36}: {allrow[col]:.3f}")
    for name, c in c95.items():
        print(f"  95th percentile of {name[4:]:<28}: {c:.3f}")

    print("\n  by crowding decile (1 = most isolated, 10 = most crowded)")
    cols = ["n", "stable_frac", "match_cos_median", "second_cos_median", "pearson_match_median",
            "pearson_second_median", "pearson_own_nn_median", "pearson_randf_median",
            "frac_stable_pearson_ge_0p5"]
    print(deciles[["decile"] + cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    body = d.drop(index="all")
    top, bottom = body.iloc[-3:], body.iloc[:3]
    gap_second = (top.pearson_match_median - top.pearson_second_median).mean()
    gap_own = (top.pearson_match_median - top.pearson_own_nn_median).mean()
    trend = top.pearson_match_median.mean() - bottom.pearson_match_median.mean()
    print(f"\n  in the three most crowded deciles, best match minus second-best: {gap_second:+.3f}")
    print(f"  in the three most crowded deciles, best match minus own nearest:  {gap_own:+.3f}")
    print(f"  best-match agreement, top three deciles minus bottom three:        {trend:+.3f}")
    if gap_second > 0.2 and trend > -0.1:
        print("  -> the match is specific: in dense regions the best match agrees on activations "
              "and the second-best, which also sits near theta, does not.")
    elif gap_second < 0.1:
        print("  -> in dense regions the second-best match agrees about as well as the best: "
              "cosine matching does not single out one partner latent there. Report as the "
              "objection's scenario and lean on the strict-label table.")
    else:
        print("  -> mixed; read the table.")

    print("\n--- Classifier under a strict label (cosine match AND activation agreement) ---")
    cols = ["cutoff_name", "cutoff", "variant", "n_labelled", "n_reclassified", "stable_frac",
            "auroc_full", "auroc_freq", "auroc_crowding", "delta"]
    print(strict[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    orig = strict[(strict.cutoff_name == "original") & (strict.variant == "drop")].iloc[0]
    print(f"\n  original label reproduces: AUROC {orig.auroc_full:.3f}, frequency "
          f"{orig.auroc_freq:.3f}, crowding {orig.auroc_crowding:.3f}, n={orig.n_labelled}")
    print("  (k=64 @ 1B, anchor 42 should give ~0.868 / 0.606 / 0.817 on 6,721 latents)")


def run_analyze(k, tokens, cutoffs, args):
    if not pairs_path(k, tokens).exists():
        raise SystemExit(f"{pairs_path(k, tokens)} missing: run the pairs stage first.")
    pairs, stats = load_cell(k, tokens)
    anchor = int(pairs["anchor"])
    partners = [int(p) for p in pairs["partner_seeds"]]

    deciles, pear_match_min, pear_random, pear_randf = decile_table(
        pairs, stats, partners, k, tokens, anchor)
    strict, c95 = strict_label_table(pairs, stats, partners, k, tokens, anchor, cutoffs,
                                     pear_match_min, pear_random, pear_randf)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    deciles.to_csv(RESULTS_DIR / f"by_crowding_decile_k{k}.csv", index=False)
    strict.to_csv(RESULTS_DIR / f"strict_label_auroc_k{k}.csv", index=False)
    plot_agreement(deciles, RESULTS_DIR / f"agreement_by_crowding_k{k}.png")
    (RESULTS_DIR / f"config_k{k}.json").write_text(json.dumps({
        "repo": args.repo, "prefix": args.prefix, "model": MODEL_NAME, "hook": HOOK_POINT,
        "k": k, "tokens": tokens, "anchor": anchor, "partner_seeds": partners,
        "eval_batches": N_EVAL_BATCHES, "n_tokens": int(pairs["n_tokens"]),
        "theta": THETA, "epsilon": EPSILON, "min_firings": MIN_FIRINGS,
        "cutoffs": list(cutoffs), **c95, "rng_seed": args.seed, "device": DEVICE,
        "pairings": list(PAIRINGS), "freq_strata": N_FREQ_STRATA,
        "logits": "own_logit_cos" in pairs,
    }, indent=2))
    report(deciles, strict, c95)
    print(f"\nWrote {RESULTS_DIR}/by_crowding_decile_k{k}.csv, strict_label_auroc_k{k}.csv, "
          f"agreement_by_crowding_k{k}.png, config_k{k}.json")


# --------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["pairs", "analyze", "logits", "all"])
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=None, help="default: largest complete budget")
    ap.add_argument("--anchor", type=int, default=None, help="default: the first seed")
    ap.add_argument("--cutoff", type=float, action="append",
                    help="Pearson cutoffs for the strict label (default 0.3 0.5 0.7)")
    ap.add_argument("--logits", action="store_true",
                    help="with `all`: also compute direct-logit-effect cosines")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--seed", type=int, default=0, help="rng seed for the random pairing")
    args = ap.parse_args()

    found = discover(args.repo, args.prefix)
    if not found:
        raise SystemExit(f"No checkpoints matching {args.prefix} in {args.repo}.")
    cells, seeds, _ = complete_cells(found)
    anchor = args.anchor if args.anchor is not None else seeds[0]
    partners = [s for s in seeds if s != anchor]
    tokens = args.tokens or max(t for kk, t in cells if kk == args.k)
    if (args.k, tokens) not in cells:
        raise SystemExit(f"k={args.k} @ {tokens:,} is not a complete cell.")
    cutoffs = args.cutoff or [0.3, 0.5, 0.7]

    print(f"Repo {args.repo}  device {DEVICE}  eval batches {N_EVAL_BATCHES} "
          f"({N_EVAL_BATCHES * BATCH_SIZE * SEQ_LEN:,} tokens)  floor {MIN_FIRINGS}")
    print(f"k={args.k} @ {tokens:,}  anchor {anchor}  partners {partners}  theta {THETA}")

    rng = np.random.default_rng(args.seed)
    if args.stage in ("pairs", "all"):
        run_pairs(found, args.k, tokens, anchor, partners, rng, args.repo)
    if args.stage == "logits" or (args.stage == "all" and args.logits):
        run_logits(found, args.k, tokens, anchor, partners, args.repo)
    if args.stage in ("analyze", "all"):
        run_analyze(args.k, tokens, cutoffs, args)


if __name__ == "__main__":
    main()
