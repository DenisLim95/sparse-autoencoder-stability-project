# -*- coding: utf-8 -*-
"""Shared SAE definition, label construction and single-run statistics.

Extracted verbatim from `topk_sweep_experiments.py` so that offline analyses can compute the
same quantities the sweep computed, from the same code. The sweep script imports these names
back, so there is exactly one definition of each and no copy to drift.

Nothing here has side effects on import: no model is loaded, no dataset is opened, nothing is
printed. That is the property that makes it importable from a laptop script; keep it.

Environment variables read at import (same names and defaults as the sweep):
    SAE_MIN_FIRINGS     measurability floor, default 100
    SAE_ABLATION_BATCH  batch size for the closed-form ablation, default 4096
"""

import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

THETA = 0.7  # match threshold (Gerasimov et al.)
EPSILON = 0.05  # endpoint binarization: label only the extremes

# A conditional statistic estimated from a handful of firings is mostly sampling noise: at 10
# firings the mean carries roughly +/-30% error, and a latent that never fires has no
# conditional mean to speak of. Features below this floor get NaN for every conditional
# statistic and are dropped from the classifier, rather than being handed a fabricated 0.0
# that the classifier can then use to identify them.
MIN_FIRINGS = int(os.environ.get("SAE_MIN_FIRINGS") or 100)

STAT_BATCH = 4096
# One matmul per batch produces a (batch, n_features) matrix, which is 16x larger here than
# in the 4x run; the smaller batch keeps that intermediate around a few hundred MB.
ABLATION_BATCH = int(os.environ.get("SAE_ABLATION_BATCH") or 4096)
LOG_EPS = 1e-10

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


class TopKSparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with a hard TopK activation.

    Architecture:
        encoder: x -> ReLU(TopK_k(W_enc @ (x - b_dec) + b_enc))
        decoder: f -> W_dec @ f + b_dec

    At most k latents are non-zero per token by construction, so L0 is k and no sparsity term
    enters the loss. (L0 comes in slightly under k on tokens where fewer than k
    pre-activations are positive, since selection happens before the rectifier.)
    """

    def __init__(self, d_model: int, n_features: int, seed: int, k: int,
                 tied_init: bool = True):
        super().__init__()
        # Reseeded here rather than globally, so construction order does not matter and two
        # arms sharing a seed share an initialization exactly -- the shapes do not depend on
        # k, so the k arms are matched on init as well as on data.
        torch.manual_seed(seed)

        self.d_model = d_model
        self.n_features = n_features
        self.k = min(k, n_features)

        self.W_enc = nn.Parameter(torch.randn(d_model, n_features) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        self.W_dec = nn.Parameter(torch.randn(n_features, d_model) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
            if tied_init:
                # Copy decoder -> encoder, not the reverse, so the unit norms just established
                # survive. A latent then reads along the same direction it writes, so the very
                # first time it wins the top-k it contributes something useful instead of
                # noise, which is what keeps it in contention long enough to learn.
                self.W_enc.data = self.W_dec.data.t().contiguous().clone()

    def _encode_with_pre(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (sparse activations, dense pre-activations). The auxiliary loss needs the
        pre-activations of latents the top-k threw away, which `encode` cannot expose."""
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        # Select on the pre-activations, then rectify. Taking the top k of an already
        # rectified vector would pick arbitrary features out of the zeros whenever fewer
        # than k are positive, inventing activations that carry no signal.
        idx = pre_acts.topk(self.k, dim=-1).indices
        keep = torch.zeros_like(pre_acts, dtype=torch.bool).scatter_(-1, idx, True)
        selected = torch.where(keep, pre_acts, torch.zeros_like(pre_acts))
        return F.relu(selected), pre_acts

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations."""
        return self._encode_with_pre(x)[0]

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to input space."""
        return f @ self.W_dec + self.b_dec

    def auxk_loss(self, x, x_hat, pre_acts, dead_mask, k_aux):
        """Gao et al.'s AuxK: let the currently-dead latents try to explain the main model's
        reconstruction error. A latent that keeps losing the top-k competition otherwise
        receives no gradient at all and stays dead permanently; here it gets one."""
        zero = x.new_zeros(())
        if dead_mask is None:
            return zero
        n_dead = int(dead_mask.sum())
        k = min(int(k_aux), n_dead)
        if k == 0:
            return zero

        # Only dead latents compete, so the live ones cannot crowd them out again.
        dead_acts = F.relu(pre_acts).masked_fill(~dead_mask, 0.0)
        idx = dead_acts.topk(k, dim=-1).indices
        keep = torch.zeros_like(dead_acts, dtype=torch.bool).scatter_(-1, idx, True)
        z = torch.where(keep, dead_acts, torch.zeros_like(dead_acts))

        # Deliberately no b_dec: the dead latents have to explain the residual themselves, and
        # adding the bias back in is a well-known way to get this silently wrong.
        e_hat = z @ self.W_dec
        # The residual is a target, not something to optimize. Detaching stops the auxiliary
        # term from making its own job easier by degrading the main reconstruction.
        e = (x - x_hat).detach()
        aux = F.mse_loss(e_hat, e)
        # Reported to go non-finite occasionally; zeroing one step beats losing the run.
        return aux if torch.isfinite(aux) else zero

    def forward(self, x: torch.Tensor, dead_mask: Optional[torch.Tensor] = None,
                k_aux: int = 512) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass returning reconstruction, features, and loss components.

        Args:
            dead_mask: Boolean mask over features marking latents that have not fired
                recently. None (the default, so analysis code is unaffected) skips the
                auxiliary loss entirely.
            k_aux: How many dead latents to enter into the auxiliary reconstruction.
        """
        f, pre_acts = self._encode_with_pre(x)
        x_hat = self.decode(f)

        recon_loss = F.mse_loss(x_hat, x)
        # Reported but never added to the loss: under TopK the constraint is structural, and
        # penalising magnitude on top of it would just shrink the k surviving activations.
        # Kept so the training curves stay comparable with the L1 run's.
        code_magnitude = f.abs().mean()

        return x_hat, f, {
            "recon_loss": recon_loss,
            "code_magnitude": code_magnitude,
            "aux_loss": self.auxk_loss(x, x_hat, pre_acts, dead_mask, k_aux),
        }

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm (call after each optimization step)."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


def compute_decoder_similarity(sae1, sae2) -> torch.Tensor:
    """Cosine similarity between the decoder rows of two SAEs.

    Returns a (n_features, n_features) matrix whose (i, j) entry is the cosine similarity
    between feature i of sae1 and feature j of sae2.
    """
    W1_norm = F.normalize(sae1.W_dec.detach(), dim=1)
    W2_norm = F.normalize(sae2.W_dec.detach(), dim=1)
    return (W1_norm @ W2_norm.T).cpu()


def compute_reappearance_probability(saes: Dict[int, nn.Module], theta: float = THETA):
    """p_hat_i = fraction of the non-anchor SAEs containing ANY feature j with
    cos(d_i, d_j) >= theta, with the first key taken as the anchor (Gerasimov's k=0)."""
    seeds = list(saes.keys())
    anchor_sae = saes[seeds[0]]
    n_features = anchor_sae.n_features

    reappearance_counts = np.zeros(n_features)
    matching_info = {"similarities": [], "best_match_idx": []}

    for other_seed in seeds[1:]:
        sim_matrix = compute_decoder_similarity(anchor_sae, saes[other_seed])

        # KEY: many-to-one argmax -- for each anchor feature (row), take its single best
        # match across ALL features in the other SAE. No 1-to-1 constraint.
        best_sim, best_idx = sim_matrix.max(dim=1)
        best_sim = best_sim.numpy()

        matching_info["similarities"].append(best_sim)
        matching_info["best_match_idx"].append(best_idx.numpy())
        reappearance_counts += (best_sim >= theta).astype(float)

    return reappearance_counts / max(len(seeds) - 1, 1), matching_info


def compute_activation_stats(sae, activations, device, batch_size=STAT_BATCH, desc="stats"):
    """Firing rate and firing strength for every feature of a single SAE.

    Returns mean activation conditioned on the feature firing. The unconditional mean --
    the sum of activations divided by ALL tokens -- is identically
    (firing rate) x (conditional mean), so using it as a predictor alongside activation
    frequency would double-count frequency rather than contribute anything new.

    Returns (activation_freq, mean_activation, firing_counts). The raw counts come back
    because they, not the rate, determine whether the conditional statistics mean anything.
    """
    n_total = len(activations)
    freq_accum = torch.zeros(sae.n_features)
    sum_accum = torch.zeros(sae.n_features)

    with torch.no_grad():
        for start in tqdm(range(0, n_total, batch_size), desc=f"Computing {desc}"):
            batch = activations[start : start + batch_size].to(device)
            feats = sae.encode(batch)               # (B, n_features)
            freq_accum += (feats > 0).float().sum(dim=0).cpu()
            sum_accum += feats.sum(dim=0).cpu()

    firing_counts = freq_accum.numpy()
    activation_freq = firing_counts / max(n_total, 1)
    enough = firing_counts >= MIN_FIRINGS
    mean_activation = np.divide(
        sum_accum.numpy(), firing_counts,
        out=np.full(sae.n_features, np.nan), where=enough,
    )
    return activation_freq, mean_activation, firing_counts


def compute_geometric_isolation(sae, k_nn: int = 10, chunk: int = 1024) -> np.ndarray:
    """Average cosine similarity to the k nearest neighbours, per feature.

    LOW value = isolated, unique direction (more likely stable)
    HIGH value = crowded region, rotational freedom (less stable)

    Chunked over rows and kept on the GPU: the full similarity matrix is n^2, so materializing
    it in numpy and sorting each row -- which was affordable at 2048 latents -- costs 16x more
    at 8192 and grows quadratically with any further widening.
    """
    W = F.normalize(sae.W_dec.detach(), dim=1)
    n = W.shape[0]
    out = torch.empty(n, device=W.device)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = W[start:stop] @ W.T
        # Exclude self, whose similarity is 1.0 and would otherwise be the nearest neighbour.
        rows = torch.arange(start, stop, device=W.device)
        sims[rows - start, rows] = -float("inf")
        out[start:stop] = sims.topk(k_nn, dim=1).values.mean(dim=1)

    return out.cpu().numpy()


def compute_reconstruction_contribution(
    sae,
    activations: torch.Tensor,
    batch_size: int = ABLATION_BATCH,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How much reconstruction error increases when each feature is ablated (zeroed).

    HIGH value = feature is important for reconstruction (more likely stable)
    LOW value = feature is redundant (less stable)

    Ablating a feature perturbs only the tokens where it was active, so averaging the MSE
    increase over ALL sampled tokens gives (firing rate) x (impact while firing). That is
    mechanically proportional to activation frequency, which is separately one of our
    predictors -- so the unconditional form cannot be used to argue that reconstruction
    contribution adds anything beyond frequency. The conditional mean, over firing tokens
    only, is the part that is not already frequency. Both are returned so the size of the
    overlap can be reported rather than assumed.

    Returns (conditional, unconditional, active_counts).
    """
    sae.eval()
    n_features = sae.n_features
    d_model = sae.d_model

    delta_sums = torch.zeros(n_features, dtype=torch.float64, device=device)
    active_counts = torch.zeros(n_features, dtype=torch.float64, device=device)
    n_total = len(activations)

    with torch.no_grad():
        dec_sq_norms = sae.W_dec.pow(2).sum(dim=1)  # 1.0 under normalize_decoder

        # Zeroing feature j shifts the residual by exactly -f_j * W_dec[j], so the change in
        # per-token MSE has a closed form and every feature can be done in one matmul. This
        # is identical to ablating features one at a time, without n_features decode passes.
        for start in tqdm(range(0, n_total, batch_size), desc="Ablation (closed form)"):
            x = activations[start : start + batch_size].to(device)
            f = sae.encode(x)
            residual = sae.decode(f) - x
            cross = residual @ sae.W_dec.T  # (B, n_features)
            delta = (f.pow(2) * dec_sq_norms - 2.0 * f * cross) / d_model

            delta_sums += delta.sum(dim=0).double()
            active_counts += (f > 0).sum(dim=0).double()

    delta_sums = delta_sums.cpu().numpy()
    active_counts = active_counts.cpu().numpy()

    unconditional = delta_sums / max(n_total, 1)
    # delta is exactly zero wherever the feature did not fire, so the running sum over all
    # tokens already equals the sum over firing tokens.
    conditional = np.divide(
        delta_sums, active_counts,
        out=np.full_like(delta_sums, np.nan), where=active_counts >= MIN_FIRINGS,
    )
    return conditional, unconditional, active_counts


def compute_encoder_stats(sae):
    """Encoder-side single-run statistics.

    Everything else here describes the decoder or the activations it produces, but the
    encoder is what actually decides whether a feature fires: b_enc is literally the
    activation threshold, and the encoder column norm sets how sharply the feature responds.
    Both are free to read off the weights and neither is constrained by normalize_decoder.
    """
    enc = sae.W_enc.detach()  # (d_model, n_features)
    return enc.norm(dim=0).cpu().numpy(), sae.b_enc.detach().cpu().numpy()


def compute_single_run_statistics(sae, activations, device, k_nn=10, label=""):
    """Every predictor available from ONE SAE, with no reference to any other seed."""
    suffix = f" ({label})" if label else ""
    print(f"  geometric isolation{suffix}...")
    isolation = compute_geometric_isolation(sae, k_nn=k_nn)

    print(f"  activation statistics{suffix}...")
    freq, mean_act, counts = compute_activation_stats(
        sae, activations, device, desc=f"activation stats{suffix}"
    )

    print(f"  reconstruction contribution{suffix}...")
    recon_cond, recon_uncond, _ = compute_reconstruction_contribution(
        sae, activations, device=device
    )

    enc_norm, enc_bias = compute_encoder_stats(sae)

    return {
        "activation_freq": freq,
        "mean_activation": mean_act,
        "firing_counts": counts,
        "geometric_isolation": isolation,
        "recon_contribution": recon_cond,
        "recon_contribution_uncond": recon_uncond,
        "encoder_norm": enc_norm,
        "encoder_bias": enc_bias,
        "decoder_norm": sae.W_dec.detach().cpu().norm(dim=1).numpy(),
    }


def build_predictors(stats):
    """(name, values) for every predictor, from one SAE's statistics dict.

    Single code path so every arm and every held-out seed is described by identically
    constructed columns in identical order.

    Firing rates and activation magnitudes are heavy-tailed over several orders of magnitude,
    and logistic regression fits a boundary linear in whatever it is handed. Left in raw
    units, the multivariable model can recruit the other predictors purely to bend the
    frequency response, which would read as those predictors "adding signal" when they are
    only supplying curvature. This does NOT change any single-predictor AUROC: a logistic
    coefficient is monotone in its input, AUROC depends only on ranking, and so is a log.

    Decoder norm is deliberately absent: normalize_decoder pins it to 1.000 with zero
    variance, so it carries no information at all.
    """
    return [
        ("Activation Freq (log)", np.log10(stats["activation_freq"] + LOG_EPS)),
        ("Geometric Isolation", stats["geometric_isolation"]),
        ("Recon Contribution", stats["recon_contribution"]),
        ("Mean Activation (log)", np.log10(stats["mean_activation"] + LOG_EPS)),
        ("Encoder Norm", stats["encoder_norm"]),
        ("Encoder Bias", stats["encoder_bias"]),
    ]


def cv_auroc(values, labels, mask):
    """Cross-validated AUROC under the shared protocol, for any predictor set and labelling."""
    cols = values if values.ndim == 2 else values.reshape(-1, 1)
    if mask.sum() < 20:
        return float("nan")
    X_local = StandardScaler().fit_transform(cols[mask])
    y_local = labels[mask].astype(int)
    if len(np.unique(y_local)) < 2:
        return float("nan")
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    return cross_val_score(clf, X_local, y_local, cv=CV, scoring="roc_auc").mean()


# --------------------------------------------------------------------------------------
# Checkpoints on the Hub
#
# One loader for every offline analysis. The audit scripts each grew their own copy pointed
# at the L1 layout (checkpoints/seed{s}_tokens{t}.pt), which is why they all silently stopped
# finding anything once the TopK sweep moved to a per-objective subdirectory.
# --------------------------------------------------------------------------------------

DEFAULT_REPO = os.environ.get("SAE_HF_REPO") or "deenais/sae-stability-pythia70m"
# The 70m sweep predates model-scoped prefixes, so its checkpoints sit directly under the
# objective tag. Runs at other scales must use a model-scoped prefix (see
# plan-cross-scale-transfer.md); pass `prefix=` for those.
DEFAULT_PREFIX = "topk64-128-256_x16_tied_auxk"


def checkpoint_pattern(prefix: str = DEFAULT_PREFIX):
    return re.compile(rf"checkpoints/{re.escape(prefix)}/seed(\d+)_k(\d+)_tokens(\d+)\.pt")


def discover(repo: str = DEFAULT_REPO, prefix: str = DEFAULT_PREFIX):
    """Map (k, seed, tokens) -> filename for every TopK checkpoint under one prefix."""
    from huggingface_hub import HfApi

    pattern = checkpoint_pattern(prefix)
    found = {}
    for f in HfApi().list_repo_files(repo, repo_type="model"):
        m = pattern.fullmatch(f)
        if m:
            seed, k, tokens = (int(g) for g in m.groups())
            found[(k, seed, tokens)] = f
    return found


def complete_cells(found):
    """(k, tokens) cells that have every seed, plus the seed list and what was skipped.

    A cell missing a seed would silently change p_hat's denominator, so those are excluded
    rather than labelled from a partial set.
    """
    seeds = sorted({s for (_, s, _) in found})
    cells, skipped = [], []
    for k, tokens in sorted({(k, t) for (k, _, t) in found}):
        if all((k, s, tokens) in found for s in seeds):
            cells.append((k, tokens))
        else:
            skipped.append((k, tokens, [s for s in seeds if (k, s, tokens) not in found]))
    return cells, seeds, skipped


def load_sae(found, k, seed, tokens, device="cpu", repo: str = DEFAULT_REPO):
    """Rebuild the SAE for one checkpoint.

    Shapes come from the weights, so a checkpoint whose config block is missing or stale still
    loads correctly, and so this works unchanged at any d_model.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, found[(k, seed, tokens)], repo_type="model")
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    d_model, n_features = state["W_enc"].shape
    sae = TopKSparseAutoencoder(d_model, n_features, seed=seed, k=k)
    sae.load_state_dict(state)
    return sae.to(device).eval()


def load_decoder(found, k, seed, tokens, repo: str = DEFAULT_REPO) -> torch.Tensor:
    """Unit-normalized decoder rows only, for weight-only analyses that never need to encode."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, found[(k, seed, tokens)], repo_type="model")
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    return F.normalize(state["W_dec"], dim=1)


def random_rotation(d: int, rng: np.random.Generator) -> torch.Tensor:
    """Haar-uniform orthogonal matrix: destroys feature correspondence, preserves the
    dictionary's marginal geometry.

    The sign correction is not optional. `qr` returns a Q whose column signs depend on the
    implementation, so without multiplying by sign(diag(R)) the draw is not Haar-uniform and
    the null is subtly biased.
    """
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return torch.tensor(Q * np.sign(np.diag(R)), dtype=torch.float32)
