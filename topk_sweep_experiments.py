# -*- coding: utf-8 -*-
"""TopK sparsity sweep — SAE stability project.

A variant of `prelim_experiments_update.py` with four changes, made so our SAEs sit in the
same regime as published ones (Jedryszek & Crook 2026, who train on this exact model and
layer) rather than in the very dense regime the L1 arm settled into:

  1. TopK only. L0 is exactly k by construction, so sparsity is set rather than discovered.
     The L1 arm reached L0 ~= 347 of 2048 (17% density) and could only be steered by
     retraining at a new coefficient.
  2. Tied initialization on by default. Each latent starts reading along the direction it
     writes, which is the SAEBench default and the condition under which published
     cross-seed stability effects appear at all.
  3. 16x expansion (8192 latents of d_model=512) instead of 4x, so the dictionary is
     overcomplete by a normal margin.
  4. A sweep over k = 64, 128, 256, trained as three arms of one run.

Every arm shares a single activation stream, and arms with the same seed share an
initialization (SparseAutoencoder reseeds before drawing, and the shapes do not depend on k),
so a difference between arms is attributable to k alone. Nothing here resumes from the L1
checkpoints: the objective, the dictionary width and the initialization all differ, and the
run directory and Hub prefixes are scoped accordingly.

Not run locally -- this is a Colab/A100 script. Cost scales with the sweep: three arms at 16x
expansion is roughly twelve times the SAE work per seed of the 4x single-arm run, so
SAE_MAX_TOKENS defaults to a budget that answers "which k" rather than reproducing the full
scaling curve.
"""

# Install dependencies (run this cell first in Colab)
# pip install transformer_lens sae_lens datasets torch

import json
import os
import queue
import re
import shutil
import threading
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple, Dict, Optional
from transformer_lens import HookedTransformer
from datasets import load_dataset

# Configuration
CONFIG = {
    "model_name": "pythia-70m-deduped",
    "hook_point": "blocks.3.hook_resid_post",  # Middle layer of Pythia-70m (6 layers, so layer 3)
    "d_model": 512,  # Pythia-70m hidden dimension
    # 16x expansion. The 4x dictionary of the L1 run is barely overcomplete, which made its
    # 17% density worse than the raw L0 suggests: a narrow dictionary firing that densely is
    # necessarily redundant, and redundancy makes features easy to re-find across seeds for
    # reasons that have nothing to do with them being real.
    "expansion_factor": int(os.environ.get("SAE_EXPANSION") or 16),
    # The sweep. Each value is a separate arm trained on the same stream from the same
    # initialization, so k is the only thing that differs between them. 64/8192 = 0.8% density
    # at the low end and 256/8192 = 3.1% at the high end, which brackets the published range
    # instead of sitting an order of magnitude above it.
    "k_values": [
        int(x) for x in (os.environ.get("SAE_K_VALUES") or "64,128,256").split(",") if x.strip()
    ],
    # --- dead-latent mitigations (Gao et al. 2024) --------------------------------------
    # TopK strands latents in a way L1 does not: one that loses the top-k competition receives
    # no gradient and cannot recover on its own. The first TopK run here reached 60% dead and
    # returned the highest AUROC of any run (0.981) by separating the classes on corpses -- a
    # dead latent has frequency exactly zero and a decoder row still at initialization, so
    # nothing matches it and it is labelled unstable automatically. Both mitigations default
    # on because that failure invalidates the result rather than merely wasting capacity.
    #
    # tied_init: start each latent's encoder (read) direction parallel to its decoder (write)
    # direction, so its first contribution to the reconstruction is useful instead of noise
    # and it stays in contention long enough to learn. Initialization only -- the two are free
    # to diverge afterwards, unlike the permanently tied weights of Cunningham et al. 2023.
    # Most of the death happened between step 200 and 400, which is the window this governs.
    "tied_init": (os.environ.get("SAE_TIED_INIT") or "1").lower() not in ("0", "false"),
    # auxk_coeff: weight on the auxiliary loss that lets dead latents reconstruct the residual
    # error, so a latent frozen out of the top-k still receives gradient. Gao et al. use 1/32
    # as a constant; it is not scaled by k. 0 disables it.
    "auxk_coeff": float(os.environ.get("SAE_AUXK_COEFF") or 1.0 / 32),
    "auxk_k": int(os.environ.get("SAE_AUXK_K") or 512),  # dead latents entered per step
    # A latent counts as dead once it has not fired for this many tokens (Gao et al. use 10M).
    "auxk_dead_after_tokens": int(os.environ.get("SAE_AUXK_DEAD_AFTER") or 10_000_000),
    # Kept on to stay comparable with SAEBench-style SAEs, which constrain decoder columns
    # throughout. TopK has no activation penalty to game, so unlike the L1 arm this is a
    # comparability choice rather than a correctness requirement -- but turning it off also
    # changes what "decoder norm" means as a predictor, so it changes two things at once.
    "normalize_decoder": os.environ.get("SAE_NORMALIZE_DECODER", "1") != "0",
    "lr": 1e-3,
    # Overridable because the memory cost per step scales with the dictionary: the dense
    # pre-activation tensor is (batch_size * seq_len, n_features), which at 16x expansion is
    # four times what the 4x run held. Only one arm's graph is alive at a time, so this is the
    # knob to reach for if a smaller GPU runs out rather than dropping an arm.
    "batch_size": int(os.environ.get("SAE_BATCH_SIZE") or 256),
    "seq_len": int(os.environ.get("SAE_SEQ_LEN") or 128),
    # Three seeds, matching the 1B-8B L1 runs and the published comparison. The sweep already
    # triples the SAE work per seed, and reappearance probability over two comparisons gives
    # the same three levels (0, 0.5, 1) the existing ground truth has.
    "seeds": [
        int(s) for s in os.environ.get("SAE_SEEDS", "42,256,1024").split(",") if s.strip()
    ],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
CONFIG["sparsity"] = "topk"
CONFIG["n_features"] = CONFIG["d_model"] * CONFIG["expansion_factor"]

if not CONFIG["k_values"]:
    raise SystemExit("SAE_K_VALUES is empty; there would be no arm to train.")
if any(k > CONFIG["n_features"] for k in CONFIG["k_values"]):
    raise SystemExit(
        f"k values {CONFIG['k_values']} include one above the dictionary size "
        f"{CONFIG['n_features']}; k is clamped to n_features, so two arms would be identical."
    )
if len(set(CONFIG["k_values"])) != len(CONFIG["k_values"]):
    raise SystemExit(f"SAE_K_VALUES has duplicates: {CONFIG['k_values']}.")
if len(CONFIG["seeds"]) < 2:
    raise SystemExit(
        f"Ground truth needs at least two seeds to compare (got {CONFIG['seeds']}); with one "
        f"there is no reappearance probability to compute. Three is what the transfer check "
        f"needs, since the held-out seed must itself have two comparisons."
    )

K_VALUES = CONFIG["k_values"]
SEEDS = CONFIG["seeds"]
ARMS = [(k, s) for k in K_VALUES for s in SEEDS]

print(f"Using device: {CONFIG['device']}")
print(f"Dictionary: {CONFIG['n_features']} latents "
      f"({CONFIG['expansion_factor']}x expansion of d_model={CONFIG['d_model']})")
print(f"Sweep: k={K_VALUES} x seeds={SEEDS} = {len(ARMS)} SAEs on one shared stream")
for _k in K_VALUES:
    print(f"  k={_k:>4}: L0 = {_k} by construction, density {_k / CONFIG['n_features']:.2%}")
print(f"Mitigations: tied_init={CONFIG['tied_init']}, auxk_coeff={CONFIG['auxk_coeff']:g} "
      f"(k_aux={CONFIG['auxk_k']}, dead after {CONFIG['auxk_dead_after_tokens']:,} tokens)")
if not CONFIG["tied_init"] or CONFIG["auxk_coeff"] == 0:
    print("  WARNING: the unmitigated TopK run reached 60% dead latents and produced an "
          "AUROC of 0.981 that was measuring dead features, not stability.")

# Token budgets at which a checkpoint is saved, so stability can be compared across training
# scale. Roughly geometric: on a leased machine that gets deleted on a deadline, each
# milestone is a complete, analyzable result banked early, so a run cut short still yields a
# curve instead of nothing.
CHECKPOINT_TOKENS = [
    1_000_000,
    50_000_000,
    100_000_000,
    1_000_000_000,
    2_000_000_000,
    3_000_000_000,
    5_000_000_000,
    8_000_000_000,
]

# Unlike the L1 script this defaults to a cap rather than the full curve. Three arms at 16x
# expansion is ~12x the SAE work per seed of the 4x single-arm run, and the question this
# sweep exists to answer -- which k gives a defensible sparsity without killing the
# dictionary -- is legible long before 8B tokens. Raise or clear SAE_MAX_TOKENS once k is
# chosen and the scaling curve is what is wanted.
_max_tokens = int(os.environ.get("SAE_MAX_TOKENS") or 100_000_000)
if _max_tokens:
    _capped = [t for t in CHECKPOINT_TOKENS if t <= _max_tokens]
    if not _capped:
        raise SystemExit(
            f"SAE_MAX_TOKENS={_max_tokens:,} is below the first milestone "
            f"({min(CHECKPOINT_TOKENS):,}), so there would be nothing to checkpoint."
        )
    CHECKPOINT_TOKENS = _capped
print(f"Token budget: up to {max(CHECKPOINT_TOKENS):,} "
      f"(milestones {[t for t in CHECKPOINT_TOKENS]})")

# Read-only repo to seed checkpoints from. Nothing on either existing Hub repo matches this
# configuration, so this only does anything once a sweep of this shape has been uploaded.
SEED_REPO_ID = os.environ.get("SAE_SEED_REPO") or None

CHECKPOINT_EVERY_SECONDS = 900
LOG_EVERY_STEPS = 200
PREFETCH_BATCHES = 4

# --- Persistent output directory ---------------------------------------------
if os.environ.get("SAE_RESULTS_BASE"):
    RESULTS_BASE = os.environ["SAE_RESULTS_BASE"]
elif Path("/content/drive/MyDrive").exists():
    RESULTS_BASE = "/content/drive/MyDrive/sae-stability-outputs"
else:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        RESULTS_BASE = "/content/drive/MyDrive/sae-stability-outputs"
    except (ImportError, ModuleNotFoundError):
        RESULTS_BASE = "outputs"

_layer = CONFIG["hook_point"].split(".")[1]


def _token_label(n: int) -> str:
    return f"{n // 1_000_000_000}B" if n >= 1_000_000_000 else f"{n // 1_000_000}M"


# Every axis that changes the objective is in the directory name. The L1 checkpoints live
# under a name without any of these suffixes, so this run can neither resume from them nor
# overwrite them -- two incomparable families of weights sharing a directory is a
# configuration mistake that otherwise presents as a crash mid-run.
_objective_tag = (
    f"_topk{'-'.join(str(k) for k in K_VALUES)}"
    f"_x{CONFIG['expansion_factor']}"
)
if not CONFIG["normalize_decoder"]:
    _objective_tag += "_freedec"
if CONFIG["tied_init"]:
    _objective_tag += "_tied"
if CONFIG["auxk_coeff"]:
    _objective_tag += "_auxk"
RUN_NAME = (
    f"{CONFIG['model_name']}_L{_layer}"
    f"_{_token_label(max(CHECKPOINT_TOKENS))}tok{_objective_tag}"
)
OUTPUT_DIR = Path(RESULTS_BASE) / RUN_NAME

_hub_scope = f"/{_objective_tag.lstrip('_')}"
HUB_CHECKPOINT_PREFIX = f"checkpoints{_hub_scope}"
HUB_RESULTS_PREFIX = f"results{_hub_scope}"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

HF_REPO_ID = os.environ.get("SAE_HF_REPO") or None
if HF_REPO_ID:
    print(f"Mirroring milestone checkpoints to hf.co/{HF_REPO_ID}")
else:
    print("WARNING: SAE_HF_REPO is unset -- checkpoints will only exist on this machine's "
          "disk. If that disk is not durable, an outage or shutdown loses the whole run.")

print(f"Results will be saved to: {OUTPUT_DIR}")

"""## 1. Load Model and Set Up Activation Streaming"""


def load_pythia(model_name: str, device: str) -> HookedTransformer:
    """Load a Pythia (GPT-NeoX) model into transformer_lens.

    transformer_lens's NeoX weight converter reads `hf_model.embed_out`, but recent
    transformers renamed that head to `lm_head`, so letting transformer_lens load the model
    itself dies with "'GPTNeoXForCausalLM' object has no attribute 'embed_out'". Loading the
    HF model here and aliasing the name the converter expects onto the actual output head
    works on either version and leaves the weights untouched.
    """
    from transformers import AutoModelForCausalLM
    from transformer_lens.loading_from_pretrained import get_official_model_name

    official_name = get_official_model_name(model_name)
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(official_name, dtype=torch.float32)
    except TypeError:  # transformers < 4.56 spells this torch_dtype
        hf_model = AutoModelForCausalLM.from_pretrained(official_name, torch_dtype=torch.float32)

    if not hasattr(hf_model, "embed_out"):
        head = hf_model.get_output_embeddings()
        if head is None:
            raise RuntimeError(
                f"{official_name} exposes neither embed_out nor get_output_embeddings(); "
                f"cannot build the state dict transformer_lens expects."
            )
        hf_model.embed_out = head

    return HookedTransformer.from_pretrained(model_name, hf_model=hf_model, device=device)


model = load_pythia(CONFIG["model_name"], CONFIG["device"])
model.eval()
print(f"Loaded {CONFIG['model_name']} with {model.cfg.n_layers} layers")


def activation_stream_generator(model, dataset_name: str, hook_point: str, seq_len: int,
                                batch_size: int, device: str):
    """
    Infinite generator yielding (batch_size * seq_len, d_model) activation tensors.
    Re-tokenizes and streams fresh Pile text; does not pre-collect a fixed n_tokens.
    Calling this function again (e.g. for the eval set) starts a fresh read from the
    beginning of the streaming dataset, so every arm and seed sees the same data in the
    same order -- only the SAE's own k and initialization differ.
    """
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    token_buffer = []

    for example in dataset:
        tokens = model.tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=seq_len * 10
        )["input_ids"][0]
        token_buffer.extend(tokens.tolist())

        while len(token_buffer) >= seq_len * batch_size:
            batch_tokens = torch.tensor(
                token_buffer[: seq_len * batch_size]
            ).reshape(batch_size, seq_len)
            token_buffer = token_buffer[seq_len * batch_size:]

            with torch.no_grad():
                # hook_point looks like "blocks.3.hook_resid_post" -> stop after block 3.
                layer_idx = int(hook_point.split(".")[1])
                _, cache = model.run_with_cache(
                    batch_tokens.to(device),
                    names_filter=[hook_point],
                    stop_at_layer=layer_idx + 1,
                )
                acts = cache[hook_point].reshape(-1, cache[hook_point].shape[-1]).cpu()

            yield acts  # (batch_size * seq_len, d_model)


"""## 2. Define SAE Architecture (TopK)"""


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


"""## 3. Train every (k, seed) arm on one shared stream"""


def remove_parallel_component(W_dec: torch.Tensor, W_dec_grad: torch.Tensor) -> torch.Tensor:
    """Project out the gradient component parallel to each (unit-norm) decoder column,
    so Adam doesn't 'spend' an update on a direction normalize_decoder() will undo anyway."""
    parallel_component = (W_dec_grad * W_dec).sum(dim=1, keepdim=True) * W_dec
    return W_dec_grad - parallel_component


def empty_history() -> Dict[str, list]:
    return {
        "step": [],
        "tokens_seen": [],
        "recon_loss": [],
        "code_magnitude": [],
        "total_loss": [],
        "l0": [],
        "dead_frac": [],
        "aux_loss": [],
    }


def save_checkpoint_atomic(state: dict, path: Path):
    """Write to a temp file and rename, so a disconnect mid-save can't leave a truncated
    checkpoint that breaks the next resume."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def prefetch_batches(iterable, max_queued: int = 4):
    """Produce batches on a background thread so activation generation overlaps SAE
    training instead of alternating with it.

    Building a batch is mostly tokenization (Rust, releases the GIL) plus a Pythia forward
    (CUDA, releases the GIL), so a plain thread genuinely overlaps with the training step
    rather than fighting it for the interpreter. The overlap matters more here than in the
    single-arm run: nine SAE updates per batch leave the producer more time to work ahead.
    """
    q: queue.Queue = queue.Queue(maxsize=max_queued)
    done = object()

    def produce():
        try:
            for item in iterable:
                q.put(item)
        except Exception as e:  # surface it on the consumer side instead of dying silently
            q.put(e)
        finally:
            q.put(done)

    threading.Thread(target=produce, daemon=True).start()

    while True:
        item = q.get()
        if item is done:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def arm_tag(k: int, seed: int) -> str:
    return f"k{k}_seed{seed}"


def milestone_filename(k: int, seed: int, milestone: int) -> str:
    return f"seed{seed}_k{k}_tokens{milestone}.pt"


def rolling_checkpoint_name(k_values, seeds) -> str:
    """Name the shared rolling checkpoint after the arms it holds.

    All arms advance in lockstep on one stream, so they share one rolling file. Naming it
    after the sweep lets two processes covering different k values or seed sets share a
    checkpoint directory without clobbering each other's rolling state.
    """
    return (
        "shared_latest"
        f"_k{'-'.join(str(k) for k in sorted(k_values))}"
        f"_seeds{'-'.join(str(s) for s in sorted(seeds))}.pt"
    )


def restore_checkpoints_from_hub(repo_id: str, checkpoint_dir: Path, k_values=None, seeds=None,
                                 prefix: str = "checkpoints"):
    """Pull any milestone checkpoints already on the Hub into the local checkpoint dir.

    Leased GPU machines get replaced, and the replacement arrives with an empty disk. Since
    milestone checkpoints are enough to resume from (see load_shared_resume_state), fetching
    them here means a run continues on a new machine instead of restarting from zero.

    Restricted to this sweep's arms, so seeding from another repo doesn't drag down weights
    for k values or seeds this run isn't training.
    """
    from huggingface_hub import HfApi, hf_hub_download

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(prefix)}/seed(\d+)_k(\d+)_tokens\d+\.pt")
    want_seeds = None if seeds is None else set(seeds)
    want_ks = None if k_values is None else set(k_values)
    try:
        remote = []
        for f in HfApi().list_repo_files(repo_id, repo_type="model"):
            m = pattern.fullmatch(f)
            if not m:
                continue
            if want_seeds is not None and int(m.group(1)) not in want_seeds:
                continue
            if want_ks is not None and int(m.group(2)) not in want_ks:
                continue
            remote.append(f)
    except Exception as e:
        print(f"Could not list hf.co/{repo_id} ({e}); starting from local state only.")
        return

    missing = [f for f in remote if not (checkpoint_dir / Path(f).name).exists()]
    if not missing:
        print(f"No milestone checkpoints to restore from hf.co/{repo_id}.")
        return

    print(f"Restoring {len(missing)} checkpoint(s) from hf.co/{repo_id}...")
    for f in missing:
        try:
            local = hf_hub_download(repo_id, f, repo_type="model")
            shutil.copyfile(local, checkpoint_dir / Path(f).name)
            print(f"  restored {Path(f).name}")
        except Exception as e:
            print(f"  WARNING: could not restore {f}: {e}")


def load_shared_resume_state(checkpoint_dir: Path, arms: list, k_values: list, seeds: list,
                             checkpoint_tokens: list):
    """Find the furthest point every arm can resume from together.

    Arms advance in lockstep, so they share one rolling checkpoint. If that file is missing
    or unreadable, fall back to the largest milestone every arm reached, which costs one
    milestone interval rather than the whole run.
    """
    shared_path = checkpoint_dir / rolling_checkpoint_name(k_values, seeds)
    if shared_path.exists():
        try:
            # Explicit weights_only=False: these are our own checkpoints and they carry
            # optimizer/config/history alongside the weights. Newer torch flips the default
            # to True, which would break resume part-way through a multi-day run.
            return torch.load(shared_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Ignoring unreadable rolling checkpoint {shared_path.name}: {e}")

    for milestone in sorted(checkpoint_tokens, reverse=True):
        paths = {a: checkpoint_dir / milestone_filename(a[0], a[1], milestone) for a in arms}
        if not all(p.exists() for p in paths.values()):
            continue
        try:
            ckpts = {
                a: torch.load(p, map_location="cpu", weights_only=False)
                for a, p in paths.items()
            }
        except Exception as e:
            print(f"Ignoring unreadable milestone {milestone:,}: {e}")
            continue
        print(f"No rolling checkpoint; falling back to the {milestone:,}-token milestone.")
        return {
            "tokens_seen": min(c["tokens_seen"] for c in ckpts.values()),
            "step": min(c.get("step", 0) for c in ckpts.values()),
            "models": {a: c["model_state_dict"] for a, c in ckpts.items()},
            "optimizers": {a: c["optimizer_state_dict"] for a, c in ckpts.items()},
            "histories": {a: c.get("history", empty_history()) for a, c in ckpts.items()},
            "config": next(iter(ckpts.values())).get("config", {}),
        }
    return None


def check_resume_config(prior: dict, config: dict):
    """Refuse to resume weights trained under a different objective.

    A mid-run change of objective puts a discontinuity in the middle of the curve that
    nothing downstream can detect, so this fails loudly instead of producing unattributable
    numbers. tied_init is deliberately not checked: it only affects step 0, and on resume the
    weights come from the checkpoint regardless.
    """
    mismatches = []
    if prior.get("sparsity", "l1") != config["sparsity"]:
        mismatches.append(("sparsity", prior.get("sparsity", "l1"), config["sparsity"]))
    for key in ("n_features", "d_model", "auxk_coeff", "auxk_k", "normalize_decoder"):
        if key in prior and prior[key] != config[key]:
            mismatches.append((key, prior[key], config[key]))
    prior_ks = prior.get("k_values")
    if prior_ks is not None and sorted(prior_ks) != sorted(config["k_values"]):
        mismatches.append(("k_values", prior_ks, config["k_values"]))
    if mismatches:
        detail = "; ".join(f"{k}: checkpoint={p!r} vs CONFIG={c!r}" for k, p, c in mismatches)
        raise SystemExit(
            f"Refusing to resume, the objective would change mid-run ({detail}). Either "
            f"match CONFIG to the checkpoints, or train from scratch into an empty "
            f"checkpoint directory -- each objective gets its own directory, so this "
            f"usually means two runs were pointed at the same one."
        )


def train_arms_shared_stream(
    k_values,
    seeds,
    activation_stream,
    config,
    checkpoint_tokens,
    checkpoint_dir,
    hf_repo_id=None,
    seed_repo_id=None,
    hub_checkpoint_prefix="checkpoints",
    hub_results_prefix="results",
    checkpoint_every_seconds=CHECKPOINT_EVERY_SECONDS,
    log_every_steps=LOG_EVERY_STEPS,
):
    """Train one SAE per (k, seed) arm on a single shared pass over the activations.

    Generating activations is the expensive shared step -- a Pythia forward plus tokenization
    per batch -- and every arm is meant to see identical data, so producing each batch once
    and updating all arms from it costs one data pipeline instead of len(arms). It also makes
    the k arms data-matched by construction, even across restarts, which is what lets a
    difference between them be attributed to k.

    Returns (saes, histories), both keyed by (k, seed).
    """
    device = config["device"]
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_tokens = sorted(checkpoint_tokens)
    arms = [(k, s) for k in k_values for s in seeds]

    saes = {
        (k, s): TopKSparseAutoencoder(
            config["d_model"], config["n_features"], seed=s, k=k,
            tied_init=config["tied_init"],
        ).to(device)
        for k, s in arms
    }
    optimizers = {a: torch.optim.Adam(saes[a].parameters(), lr=config["lr"]) for a in arms}
    histories = {a: empty_history() for a in arms}
    tokens_seen = 0
    step = 0

    hf_api = None
    if hf_repo_id is not None:
        from huggingface_hub import HfApi
        hf_api = HfApi()
        # upload_file does not create the repo, so without this every mirror would fail
        # with a warning and the run would silently persist nothing off-box. Fail loudly
        # here instead: if the mirror cannot work, better to know before training starts.
        hf_api.create_repo(hf_repo_id, repo_type="model", private=True, exist_ok=True)
        restore_checkpoints_from_hub(
            hf_repo_id, checkpoint_dir, k_values, seeds, prefix=hub_checkpoint_prefix
        )

    # Second, so our own further-along checkpoints win: restore only fetches what is missing.
    if seed_repo_id is not None:
        print(f"Seeding from collaborator repo hf.co/{seed_repo_id} (read-only)")
        restore_checkpoints_from_hub(
            seed_repo_id, checkpoint_dir, k_values, seeds, prefix=hub_checkpoint_prefix
        )

    def mirror_to_hub(path: Path, path_in_repo: str):
        if hf_api is None:
            return
        try:
            hf_api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path_in_repo,
                repo_id=hf_repo_id,
                repo_type="model",
            )
        except Exception as e:
            # Don't let an upload failure kill the training run -- the local copy is
            # already saved, so the push can be retried later.
            print(f"WARNING: HF upload of {path.name} failed ({e}); local copy is safe.")

    resume = load_shared_resume_state(
        checkpoint_dir, arms, k_values, seeds, checkpoint_tokens
    )
    if resume is not None:
        check_resume_config(resume.get("config", {}), config)
        for a in arms:
            if a not in resume["models"]:
                raise SystemExit(
                    f"Rolling checkpoint has no state for arm k={a[0]} seed={a[1]}. It was "
                    f"written by a run covering a different sweep; point this run at its own "
                    f"checkpoint directory."
                )
            saes[a].load_state_dict(resume["models"][a])
            optimizers[a].load_state_dict(resume["optimizers"][a])
            restored = resume["histories"].get(a, empty_history())
            # Checkpoints written before a curve existed lack its key, and the logging code
            # appends to every key unconditionally. Pad with NaN so the new series lines up
            # with the old steps on the x-axis instead of being silently shifted left.
            for key in empty_history():
                restored.setdefault(
                    key, [float("nan")] * len(restored.get("tokens_seen", []))
                )
            histories[a] = restored
        tokens_seen = resume["tokens_seen"]
        step = resume.get("step", 0)
        print(f"Resumed all {len(arms)} arms at {tokens_seen:,} tokens")
        print("CAVEAT: the activation stream restarts from the beginning of the dataset, so "
              "tokens already trained on will be seen again. Past the first restart the "
              "token counts measure optimization, not unique tokens. All arms replay the "
              "same data, so cross-seed and cross-k comparisons stay valid.")

    next_checkpoint_idx = sum(t <= tokens_seen for t in checkpoint_tokens)
    if next_checkpoint_idx >= len(checkpoint_tokens):
        print(f"Already at {tokens_seen:,} tokens; nothing left to train.")
        return saes, histories

    # Accumulate curve metrics as on-device tensors and only pull them to the host at
    # logging time; calling .item() every step would force a GPU sync per arm per step.
    # recon, code magnitude, total, l0, aux
    interval_sums = {a: torch.zeros(5, device=device) for a in arms}
    interval_n = 0
    seen_active = {
        a: torch.zeros(config["n_features"], dtype=torch.bool, device=device) for a in arms
    }
    # Tokens since each latent last fired, which is how AuxK decides what counts as dead.
    # Deliberately not checkpointed: it rebuilds itself within auxk_dead_after_tokens (about
    # 300 steps), which is short next to the gap between milestones, and starting from zero
    # only means the auxiliary loss stays off for that brief window after a restart.
    tokens_idle = {a: torch.zeros(config["n_features"], device=device) for a in arms}
    last_checkpoint_time = time.time()

    def build_arm_state(arm):
        k, seed = arm
        return {
            "model_state_dict": saes[arm].state_dict(),
            "optimizer_state_dict": optimizers[arm].state_dict(),
            "tokens_seen": tokens_seen,
            "step": step,
            "seed": seed,
            "k": k,
            "config": config,
            "history": histories[arm],
        }

    def build_shared_state():
        return {
            "tokens_seen": tokens_seen,
            "step": step,
            "config": config,
            "models": {a: saes[a].state_dict() for a in arms},
            "optimizers": {a: optimizers[a].state_dict() for a in arms},
            "histories": histories,
        }

    def write_histories():
        for a in arms:
            name = f"training_history_{arm_tag(*a)}.json"
            with open(checkpoint_dir.parent / name, "w") as fh:
                json.dump(histories[a], fh)

    unit_norm_decoder = config["normalize_decoder"]
    auxk_coeff = config["auxk_coeff"]
    auxk_k = config["auxk_k"]
    dead_after = config["auxk_dead_after_tokens"]

    for batch in activation_stream:
        batch = batch.to(device, non_blocking=True)

        for a in arms:
            sae, optimizer = saes[a], optimizers[a]
            # Built from the previous steps' firing history, so it reflects what was dead on
            # arrival at this batch rather than what this batch happens to leave out.
            dead_mask = tokens_idle[a] > dead_after if auxk_coeff else None
            x_hat, f, loss_dict = sae(batch, dead_mask=dead_mask, k_aux=auxk_k)
            loss = loss_dict["recon_loss"]
            if auxk_coeff:
                loss = loss + auxk_coeff * loss_dict["aux_loss"]

            optimizer.zero_grad()
            loss.backward()
            if unit_norm_decoder:
                with torch.no_grad():
                    sae.W_dec.grad = remove_parallel_component(sae.W_dec.data, sae.W_dec.grad)
            optimizer.step()
            if unit_norm_decoder:
                sae.normalize_decoder()

            with torch.no_grad():
                active = f > 0
                interval_sums[a][0] += loss_dict["recon_loss"].detach()
                interval_sums[a][1] += loss_dict["code_magnitude"].detach()
                interval_sums[a][2] += loss.detach()
                interval_sums[a][3] += active.float().sum(dim=1).mean()
                interval_sums[a][4] += loss_dict["aux_loss"].detach()
                fired = active.any(dim=0)
                seen_active[a] |= fired
                tokens_idle[a] = torch.where(
                    fired, torch.zeros_like(tokens_idle[a]),
                    tokens_idle[a] + batch.shape[0],
                )

        # batch is already the flattened (batch_size * seq_len, d_model) activations
        # from activation_stream_generator -- batch.shape[0] IS the real token count
        # for this step. Every arm consumed this same batch, so count it once.
        tokens_seen += batch.shape[0]
        step += 1
        interval_n += 1

        if step % log_every_steps == 0:
            for a in arms:
                recon, mag, total, l0, aux = (interval_sums[a] / interval_n).tolist()
                h = histories[a]
                h["step"].append(step)
                h["tokens_seen"].append(tokens_seen)
                h["recon_loss"].append(recon)
                h["code_magnitude"].append(mag)
                h["total_loss"].append(total)
                h["l0"].append(l0)
                h["aux_loss"].append(aux)
                # Dead = never fired anywhere in this logging window.
                h["dead_frac"].append(1.0 - seen_active[a].float().mean().item())
                interval_sums[a].zero_()
                seen_active[a] = torch.zeros(
                    config["n_features"], dtype=torch.bool, device=device
                )
            interval_n = 0
            print(f"step {step:>7} | {tokens_seen:>14,} tok")
            for k in k_values:
                ks = [(k, s) for s in seeds]
                recons = [histories[a]["recon_loss"][-1] for a in ks]
                l0s = [histories[a]["l0"][-1] for a in ks]
                worst_dead = max(100 * histories[a]["dead_frac"][-1] for a in ks)
                # dead% is the headline number to watch: it climbed to 60% by 100M tokens
                # without the mitigations, and aux shows whether the revival term is actually
                # doing anything (exactly 0 while no latent has been idle long enough to
                # qualify, which is expected early on).
                aux_col = ""
                if auxk_coeff:
                    mean_aux = np.mean([histories[a]["aux_loss"][-1] for a in ks])
                    aux_col = f" | aux {mean_aux:.5f}"
                print(f"   k={k:<4} | recon {np.mean(recons):.5f} | "
                      f"L0 {np.mean(l0s):6.1f} (target {k}) | "
                      f"dead {worst_dead:5.1f}%{aux_col}")

        if time.time() - last_checkpoint_time >= checkpoint_every_seconds:
            save_checkpoint_atomic(
                build_shared_state(),
                checkpoint_dir / rolling_checkpoint_name(k_values, seeds),
            )
            write_histories()
            last_checkpoint_time = time.time()
            print(f"rolling checkpoint at {tokens_seen:,} tokens")

        if (next_checkpoint_idx < len(checkpoint_tokens)
                and tokens_seen >= checkpoint_tokens[next_checkpoint_idx]):
            milestone = checkpoint_tokens[next_checkpoint_idx]
            for a in arms:
                ckpt_name = milestone_filename(a[0], a[1], milestone)
                ckpt_path = checkpoint_dir / ckpt_name
                save_checkpoint_atomic(build_arm_state(a), ckpt_path)
                mirror_to_hub(ckpt_path, f"{hub_checkpoint_prefix}/{ckpt_name}")

            save_checkpoint_atomic(
                build_shared_state(),
                checkpoint_dir / rolling_checkpoint_name(k_values, seeds),
            )
            write_histories()
            for a in arms:
                name = f"training_history_{arm_tag(*a)}.json"
                mirror_to_hub(checkpoint_dir.parent / name, f"{hub_results_prefix}/{name}")

            print(f"milestone reached: all {len(arms)} arms checkpointed at "
                  f"{milestone:,} tokens"
                  + (f", mirrored to hf.co/{hf_repo_id}" if hf_api is not None else ""))
            next_checkpoint_idx += 1

        if next_checkpoint_idx >= len(checkpoint_tokens):
            break

    save_checkpoint_atomic(
        build_shared_state(), checkpoint_dir / rolling_checkpoint_name(k_values, seeds)
    )
    write_histories()

    return saes, histories


print(f"=== Training {len(ARMS)} arms ({len(K_VALUES)} k values x {len(SEEDS)} seeds) "
      f"on a shared activation stream ===")
stream = activation_stream_generator(
    model=model,
    dataset_name="monology/pile-uncopyrighted",
    hook_point=CONFIG["hook_point"],
    seq_len=CONFIG["seq_len"],
    batch_size=CONFIG["batch_size"],
    device=CONFIG["device"],
)
trained_saes, training_histories = train_arms_shared_stream(
    k_values=K_VALUES,
    seeds=SEEDS,
    activation_stream=prefetch_batches(stream, max_queued=PREFETCH_BATCHES),
    config=CONFIG,
    checkpoint_tokens=CHECKPOINT_TOKENS,
    checkpoint_dir=CHECKPOINT_DIR,
    hf_repo_id=HF_REPO_ID,
    seed_repo_id=SEED_REPO_ID,
    hub_checkpoint_prefix=HUB_CHECKPOINT_PREFIX,
    hub_results_prefix=HUB_RESULTS_PREFIX,
)

print(f"\nTrained {len(trained_saes)} SAEs: k={K_VALUES} x seeds={SEEDS}")
print(f"Checkpoints saved at token counts: {CHECKPOINT_TOKENS}")

"""### 3b. Training curves — did each arm keep its dictionary alive?

There is no sparsity coefficient to get wrong here: L0 is k by construction, and the curve
should sit flat at k (a little under it early on, when fewer than k pre-activations are
positive). The failure mode moves entirely to dead features, which TopK strands far more
readily than L1 does, and which corrupt the stability analysis rather than merely wasting
capacity — a never-firing latent is labelled unstable automatically and has frequency exactly
zero, so a classifier can separate the classes by detecting corpses.
"""


def plot_training_curves(histories: Dict[tuple, Dict[str, list]], k_values, seeds, config,
                         save_path=None):
    """One column per k: reconstruction, L0 against its target, and dead fraction."""
    histories = {a: h for a, h in histories.items() if h and h["tokens_seen"]}
    if not histories:
        print("No training history recorded yet (fewer than LOG_EVERY_STEPS steps run).")
        return

    live_ks = [k for k in k_values if any(a[0] == k for a in histories)]
    if not live_ks:
        return

    fig, axes = plt.subplots(3, len(live_ks), figsize=(5 * len(live_ks), 10), squeeze=False)

    for col, k in enumerate(live_ks):
        for s in seeds:
            h = histories.get((k, s))
            if not h:
                continue
            axes[0][col].plot(h["tokens_seen"], h["recon_loss"], label=f"seed {s}")
            axes[1][col].plot(h["tokens_seen"], h["l0"], label=f"seed {s}")
            axes[2][col].plot(h["tokens_seen"], [100 * d for d in h["dead_frac"]],
                              label=f"seed {s}")

        axes[0][col].set_title(f"k={k}: reconstruction loss (MSE)")
        axes[1][col].axhline(k, color="black", linestyle="--", linewidth=1, label=f"k={k}")
        axes[1][col].set_title(f"k={k}: L0 (should sit at {k})")
        axes[2][col].set_title(f"k={k}: dead features per logging window")
        axes[2][col].set_ylim(0, 100)
        axes[0][col].set_ylabel("Reconstruction loss (MSE)")
        axes[1][col].set_ylabel("L0 (mean active features/token)")
        axes[2][col].set_ylabel("Dead features (%)")

    for ax in axes.flat:
        ax.set_xlabel("Tokens seen")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    plt.show()

    print("\n=== TopK diagnostic (final logging window) ===")
    for k in live_ks:
        for s in seeds:
            h = histories.get((k, s))
            if not h:
                continue
            print(f"  k={k:<4} seed {s}: L0={h['l0'][-1]:.1f} (target {k}), "
                  f"dead={100 * h['dead_frac'][-1]:.1f}%, recon={h['recon_loss'][-1]:.5f}")
        worst_dead = max(
            100 * histories[(k, s)]["dead_frac"][-1] for s in seeds if (k, s) in histories
        )
        if worst_dead > 20:
            live = config["n_features"] * (1 - worst_dead / 100)
            print(f"    WARNING: up to {worst_dead:.1f}% never fired in the last logging "
                  f"window, leaving about {live:.0f} of {config['n_features']} live. A dead "
                  f"latent's decoder row stays near initialization, so nothing matches it "
                  f"and it is labelled unstable, while its frequency is exactly zero -- the "
                  f"classifier can then separate the classes by detecting corpses.")
            if config["auxk_coeff"] and config["tied_init"]:
                print(f"    Both mitigations are already on, so this level of death at k={k} "
                      f"means they were not enough: a larger k or a higher auxk_coeff is the "
                      f"lever. The MIN_FIRINGS floor keeps the AUROC honest either way, at "
                      f"the cost of scoring a smaller dictionary.")
            else:
                print(f"    Enable the mitigations before trusting this arm: SAE_TIED_INIT=1 "
                      f"SAE_AUXK_COEFF=0.03125 (currently tied_init={config['tied_init']}, "
                      f"auxk_coeff={config['auxk_coeff']:g}).")


plot_training_curves(
    training_histories,
    K_VALUES,
    SEEDS,
    CONFIG,
    save_path=OUTPUT_DIR / "training_curves.png",
)

"""## 4. Feature matching and labels (decoder-only, Gerasimov et al. Section 4)

Represents each feature solely by its decoder vector (ell-2 normalized), and matches features
via many-to-one argmax cosine similarity. For anchor feature i, look across ALL features in
the other SAE and take the single best match; two different anchor features are allowed to
both match the same feature. Cheaper than Hungarian and lets us compute a per-feature
stability score independently (Gerasimov et al. report nearly identical matched sets to
Hungarian: IoU = 0.978 +/- 0.001). The one-to-one alternative is checked per arm in section 8.

Unchanged from the L1 script on purpose: the label definition has to be identical across
objectives, or a difference between this run and that one is not attributable to k.
"""

THETA = 0.7  # match threshold (Gerasimov et al.)
EPSILON = 0.05  # endpoint binarization: label only the extremes


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


"""## 5. Single-run predictors

Same definitions as the L1 script, with the neighbour search done on the GPU in chunks: the
isolation statistic needs an n x n similarity matrix, which at a 16x expansion is 16 times
the work and memory it was at 4x.
"""

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


"""## 6. Build the shared evaluation set

One eval set, used by every arm, so a difference between arms cannot come from measuring them
on different tokens.
"""

eval_stream = activation_stream_generator(
    model=model,
    dataset_name="monology/pile-uncopyrighted",
    hook_point=CONFIG["hook_point"],
    seq_len=CONFIG["seq_len"],
    batch_size=CONFIG["batch_size"],
    device=CONFIG["device"],
)
N_EVAL_BATCHES = int(os.environ.get("SAE_EVAL_BATCHES") or 40)
activations = torch.cat([next(eval_stream) for _ in range(N_EVAL_BATCHES)], dim=0)
print(f"Eval activations shape: {activations.shape} "
      f"({activations.numel() * 4 / 1e9:.1f} GB in host RAM)")

# Under TopK exactly k of n_features latents fire per token, so the AVERAGE feature fires on
# k/n_features of tokens -- 0.8% at k=64 -- and is measured to several significant figures
# here. Whether this eval set is big enough is decided by the low-count tail instead, which
# the per-arm diagnostic below reports.
print(f"Expected mean firing count per feature, by arm (over {len(activations):,} tokens):")
for _k in K_VALUES:
    print(f"  k={_k:<4}: {_k / CONFIG['n_features'] * len(activations):,.0f} "
          f"(floor is MIN_FIRINGS={MIN_FIRINGS})")

"""## 7. Classifier protocol"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hungarian on an n x n matrix is O(n^3) in the worst case, so at a 16x expansion it is the
# slowest check in the script by a wide margin. Off by default under SAE_HUNGARIAN=0 when the
# sweep is being run just to pick k.
RUN_HUNGARIAN = os.environ.get("SAE_HUNGARIAN", "1") != "0"
# Qualitative inspection re-scans the corpus per feature and prints a block each, which at
# thousands of stable latents per arm buries every other result. Off by default.
RUN_TOP_EXAMPLES = os.environ.get("SAE_TOP_EXAMPLES", "0") != "0"
TOP_EXAMPLES_N_FEATURES = int(os.environ.get("SAE_TOP_EXAMPLES_N") or 5)


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


def get_top_activating_examples(model, sae, dataset, feature_indices, n_examples=5,
                               n_tokens_to_scan=50_000, context_window=10):
    """Find text examples that most strongly activate each feature."""
    results = {int(idx): [] for idx in feature_indices}
    tokens_scanned = 0

    for example in tqdm(dataset, total=n_tokens_to_scan // 50, desc="Top examples"):
        tokens = model.tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=512
        )["input_ids"][0]
        if len(tokens) < 10:
            continue

        with torch.no_grad():
            layer_idx = int(CONFIG["hook_point"].split(".")[1])
            _, cache = model.run_with_cache(
                tokens.unsqueeze(0).to(CONFIG["device"]),
                names_filter=[CONFIG["hook_point"]],
                stop_at_layer=layer_idx + 1,
            )
            acts = cache[CONFIG["hook_point"]][0]  # (seq_len, d_model)
            features = sae.encode(acts)            # (seq_len, n_features)

        for feat_idx in results:
            feat_acts = features[:, feat_idx].cpu().numpy()
            max_pos = int(feat_acts.argmax())
            max_val = float(feat_acts[max_pos])
            if max_val <= 0:
                continue
            start = max(0, max_pos - context_window)
            end = min(len(tokens), max_pos + context_window + 1)
            results[feat_idx].append({
                "activation": max_val,
                "context": model.tokenizer.decode(tokens[start:end]),
                "token": model.tokenizer.decode(tokens[max_pos]),
                "position": max_pos,
            })

        tokens_scanned += len(tokens)
        if tokens_scanned >= n_tokens_to_scan:
            break

    for feat_idx in results:
        results[feat_idx] = sorted(
            results[feat_idx], key=lambda x: -x["activation"]
        )[:n_examples]
    return results


"""## 8. Per-arm analysis

The whole pipeline for one k: labels, the firing floor, the classifier and its baselines, and
the four robustness checks. Written as a function returning a summary row so the arms can be
compared at the end instead of being read off three separate printouts.
"""


def analyze_arm(k: int, arm_saes: Dict[int, nn.Module], activations, config) -> dict:
    n_features = config["n_features"]
    print("\n" + "=" * 72)
    print(f"ARM k={k}  ({n_features} latents, density {k / n_features:.2%})")
    print("=" * 72)

    # --- labels -------------------------------------------------------------------------
    reappearance_probs, matching_info = compute_reappearance_probability(arm_saes, theta=THETA)
    anchor_seed = list(arm_saes.keys())[0]
    anchor_sae = arm_saes[anchor_seed]

    stable_mask = reappearance_probs >= (1 - EPSILON)
    unstable_mask = reappearance_probs <= EPSILON
    middle_mask = ~stable_mask & ~unstable_mask

    # Mean of the per-anchor best cosine, and the share clearing theta on the first
    # comparison. These are the two numbers published cross-seed studies report, so they are
    # what makes an arm comparable to anything outside this repo.
    mean_max_cos = float(np.mean(matching_info["similarities"][0]))
    frac_paired = float(np.mean(matching_info["similarities"][0] >= THETA))

    print(f"\n--- Labels (anchor seed {anchor_seed}, theta={THETA}, epsilon={EPSILON}) ---")
    print(f"  mean max cosine (first comparison): {mean_max_cos:.3f}")
    print(f"  fraction paired above theta:        {frac_paired:.1%}")
    print(f"  stable    (p_hat >= {1 - EPSILON}): {int(stable_mask.sum()):5d} "
          f"({stable_mask.mean():.1%})")
    print(f"  unstable  (p_hat <= {EPSILON}): {int(unstable_mask.sum()):5d} "
          f"({unstable_mask.mean():.1%})")
    print(f"  discarded (middle):            {int(middle_mask.sum()):5d} "
          f"({middle_mask.mean():.1%})")

    # --- predictors ---------------------------------------------------------------------
    print(f"\n--- Single-run predictors (seed {anchor_seed}) ---")
    stats = compute_single_run_statistics(
        anchor_sae, activations, config["device"], label=f"k={k} seed {anchor_seed}"
    )
    firing_counts = stats["firing_counts"]
    activation_freq = stats["activation_freq"]

    print(f"\n  Is {len(activations):,} eval tokens enough? (frequency estimate quality)")
    for thresh in (1, 10, 100, 1000):
        n_below = int((firing_counts < thresh).sum())
        print(f"    features firing < {thresh:5,} times: {n_below:5d} "
              f"({n_below / n_features:5.1%}) -- relative error >~ "
              f"{100 / max(thresh, 1) ** 0.5:.0f}%")
    print(f"    median firings per feature: {np.median(firing_counts):,.0f}")
    print(f"    never fire at all: {int((firing_counts == 0).sum())} "
          f"({(firing_counts == 0).mean():.1%})")

    # The floor is not bookkeeping, it decides what the AUROC measures. A latent that never
    # fires is labelled unstable automatically -- its decoder row never moved from
    # initialization, so no feature in another seed matches it -- and it has frequency exactly
    # zero, an encoder bias still at its init value, and no conditional statistics at all.
    # Leave those in and the classifier scores well by detecting corpses, which is a claim
    # about how thoroughly the dictionary died, not about whether stability is predictable.
    live_mask = firing_counts >= MIN_FIRINGS
    labeled_mask = (stable_mask | unstable_mask) & live_mask
    dropped = (stable_mask | unstable_mask) & ~live_mask

    print(f"\n--- Firing floor (MIN_FIRINGS={MIN_FIRINGS}) ---")
    print(f"  above the floor: {int(live_mask.sum())} of {n_features} "
          f"({live_mask.mean():.1%})")
    print(f"  labelled features dropped: {int(dropped.sum())} "
          f"({dropped.sum() / max((stable_mask | unstable_mask).sum(), 1):.1%} of labelled)")
    print(f"    of which labelled stable:   {int((dropped & stable_mask).sum())}")
    print(f"    of which labelled unstable: {int((dropped & unstable_mask).sum())}")
    if dropped.any() and (dropped & unstable_mask).sum() / max(dropped.sum(), 1) > 0.9:
        print("  NOTE: the dropped features are almost entirely 'unstable', which is the "
              "confound this floor exists to remove -- they were separable on frequency "
              "alone.")

    # --- frequency confound check -------------------------------------------------------
    print("\n--- Frequency confound check (Spearman rho vs activation frequency) ---")
    # The unconditional ablation is (firing rate) x (impact while firing) by construction, so
    # a near-1.0 correlation there is expected and is the reason the conditional form is used;
    # anything else near 1.0 would mean that predictor is not independent evidence.
    for name, values in [
        ("recon contribution (conditional, used)", stats["recon_contribution"]),
        ("recon contribution (unconditional)", stats["recon_contribution_uncond"]),
        ("mean activation | firing", stats["mean_activation"]),
        ("geometric isolation", stats["geometric_isolation"]),
        ("encoder column norm", stats["encoder_norm"]),
        ("encoder bias", stats["encoder_bias"]),
    ]:
        rho = spearmanr(activation_freq, values, nan_policy="omit").statistic
        print(f"  {name:42s}: rho={rho:+.3f} (n={int(np.isfinite(values).sum())})")

    # --- effect sizes -------------------------------------------------------------------
    print("\n--- Effect sizes, stable vs unstable (Cohen's d) ---")
    stable_idx = np.where(stable_mask & live_mask)[0]
    unstable_idx = np.where(unstable_mask & live_mask)[0]
    if len(stable_idx) == 0 or len(unstable_idx) == 0:
        print(f"  one class is empty above the firing floor "
              f"({len(stable_idx)} stable, {len(unstable_idx)} unstable); no effect sizes.")
    else:
        for name, values in [
            ("Activation freq", activation_freq),
            ("Geometric isolation", stats["geometric_isolation"]),
            ("Recon contribution", stats["recon_contribution"]),
            ("Mean activation", stats["mean_activation"]),
        ]:
            s_vals, u_vals = values[stable_idx], values[unstable_idx]
            pooled = np.sqrt((np.nanstd(s_vals) ** 2 + np.nanstd(u_vals) ** 2) / 2)
            d = (np.nanmean(s_vals) - np.nanmean(u_vals)) / pooled if pooled > 0 else 0.0
            print(f"  {name:22s}: d = {d:+.3f}")

    # --- classifier ---------------------------------------------------------------------
    predictors = build_predictors(stats)
    feature_names = [name for name, _ in predictors]
    X_all = np.column_stack([v for _, v in predictors])
    X = X_all[labeled_mask]
    y = stable_mask[labeled_mask].astype(int)

    print(f"\n--- Classifier (n={int(labeled_mask.sum())}, "
          f"{int(y.sum())} stable / {len(y) - int(y.sum())} unstable) ---")
    if len(np.unique(y)) < 2 or labeled_mask.sum() < 50:
        print("  Too few usable labelled features to score this arm; skipping the classifier.")
        return {
            "k": k, "density": k / n_features, "mean_max_cos": mean_max_cos,
            "frac_paired": frac_paired, "stable_frac": float(stable_mask.mean()),
            "unstable_frac": float(unstable_mask.mean()),
            "discarded_frac": float(middle_mask.mean()),
            "live_frac": float(live_mask.mean()), "n_labelled": int(labeled_mask.sum()),
            "auroc_full": float("nan"), "auroc_freq_only": float("nan"),
            "delta_over_freq": float("nan"), "auroc_hungarian_full": float("nan"),
            "stable_frac_hungarian": float("nan"), "transfer_auroc": float("nan"),
        }
    if not np.isfinite(X).all():
        raise SystemExit(
            "Non-finite values reached the feature matrix. The firing floor is supposed to "
            "remove every feature whose conditional statistics are NaN, so this means a "
            "predictor is NaN for some reason other than too few firings."
        )

    auroc_full = cv_auroc(X_all, stable_mask, labeled_mask)
    print(f"  full model ({len(feature_names)} predictors): {auroc_full:.3f}")

    single = {}
    for name, values in predictors:
        single[name] = cv_auroc(values, stable_mask, labeled_mask)
    print("  single predictors, best to worst:")
    for rank, (name, auroc) in enumerate(sorted(single.items(), key=lambda x: -x[1]), 1):
        print(f"    {rank}. {name:24s}: {auroc:.3f}")

    # Frequency alone is the baseline the whole hypothesis has to beat: the claim is that
    # interpretable statistics add something over simply looking at how often a latent fires.
    # With a heavily imbalanced label an absolute AUROC threshold is close to automatic, so
    # this increment, not the headline number, is what distinguishes the arms.
    auroc_freq = single["Activation Freq (log)"]
    delta = auroc_full - auroc_freq
    print(f"  frequency-only baseline: {auroc_freq:.3f}")
    print(f"  DELTA (full - frequency-only): {delta:+.3f}")

    X_freq_geom = np.column_stack([
        dict(predictors)["Activation Freq (log)"], stats["geometric_isolation"]
    ])
    print(f"  frequency + geometry: {cv_auroc(X_freq_geom, stable_mask, labeled_mask):.3f}")

    print("  ablation (full model minus one):")
    for i, name in enumerate(feature_names):
        ablated = cv_auroc(np.delete(X_all, i, axis=1), stable_mask, labeled_mask)
        print(f"    without {name:24s}: {ablated:.3f} (drop: {auroc_full - ablated:+.3f})")

    fitted = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    fitted.fit(StandardScaler().fit_transform(X), y)
    print("  coefficients (standardized):")
    for name, coef in sorted(zip(feature_names, fitted.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"    {name:24s}: {coef:+.3f} "
              f"({'up stable' if coef > 0 else 'down stable'})")

    # --- robustness ---------------------------------------------------------------------
    print("\n--- Robustness of the evaluation itself ---")

    # 1. Endpoint binarization discards the ambiguous middle, removing the hardest cases
    # before scoring. Both arms restricted to the firing floor, so this isolates the
    # binarization rule rather than confounding it with the measurability exclusion.
    midpoint_auroc = cv_auroc(X_all, reappearance_probs >= 0.5, live_mask)
    print("  1. discarding the ambiguous middle")
    print(f"     endpoint only (reported, n={int(labeled_mask.sum())}): {auroc_full:.3f}")
    print(f"     all features  (p_hat >= 0.5, n={int(live_mask.sum())}): {midpoint_auroc:.3f}")
    print(f"     inflation from discarding {int((middle_mask & live_mask).sum())} middle "
          f"features: {auroc_full - midpoint_auroc:+.3f}")

    # 2. Many-to-one matching lets several anchor features claim one partner, so a crowded
    # region could read as stable. This is the check that matters most for comparability:
    # published work uses one-to-one Hungarian matching.
    auroc_hungarian = float("nan")
    hungarian_stable_frac = float("nan")
    if RUN_HUNGARIAN:
        print("  2. matching rule (many-to-one vs one-to-one Hungarian)")
        t0 = time.time()
        seeds_here = list(arm_saes.keys())
        hungarian_counts = np.zeros(n_features)
        for other in seeds_here[1:]:
            sim = compute_decoder_similarity(anchor_sae, arm_saes[other]).numpy()
            rows, cols = linear_sum_assignment(-sim)
            assigned = np.full(n_features, -1.0)
            assigned[rows] = sim[rows, cols]
            hungarian_counts += (assigned >= THETA).astype(float)
        hungarian_probs = hungarian_counts / max(len(seeds_here) - 1, 1)
        hungarian_stable = hungarian_probs >= (1 - EPSILON)
        hungarian_labeled = (hungarian_stable | (hungarian_probs <= EPSILON)) & live_mask
        hungarian_stable_frac = float(hungarian_stable.mean())
        auroc_hungarian = cv_auroc(X_all, hungarian_stable, hungarian_labeled)

        print(f"     stable fraction  many-to-one: {stable_mask.mean():6.1%}   "
              f"one-to-one: {hungarian_stable_frac:6.1%}")
        print(f"     labels changed by switching rule: "
              f"{int((stable_mask != hungarian_stable).sum())} of {n_features}")
        print(f"     geometric isolation alone, many-to-one: "
              f"{cv_auroc(stats['geometric_isolation'], stable_mask, labeled_mask):.3f}")
        print(f"     geometric isolation alone, one-to-one : "
              f"{cv_auroc(stats['geometric_isolation'], hungarian_stable, hungarian_labeled):.3f}")
        print(f"     full model,                one-to-one : {auroc_hungarian:.3f}")
        print(f"     (a large gap means isolation tracks the matcher, not stability; "
              f"took {time.time() - t0:.0f}s)")
    else:
        print("  2. matching rule: skipped (SAE_HUNGARIAN=0)")

    # 3. Cross-validation holds out features from the SAME dictionary, and geometric isolation
    # is relational -- a feature's value depends on neighbours that may sit in the training
    # fold. Train on the anchor seed, test on a different seed's dictionary entirely.
    transfer_auroc = float("nan")
    print("  3. transfer to a held-out dictionary (the deployment claim)")
    seeds_here = list(arm_saes.keys())
    if len(seeds_here) < 3:
        print("     needs >=3 seeds: the held-out seed must itself have two comparisons.")
    else:
        held_out_seed = seeds_here[1]
        # Same budget, same eval activations, same code path -- only the dictionary differs,
        # so a drop is transfer failure and not a difference in budget or measurement.
        held_out_saes = {held_out_seed: arm_saes[held_out_seed]}
        held_out_saes.update({s: arm_saes[s] for s in seeds_here if s != held_out_seed})
        held_probs, _ = compute_reappearance_probability(held_out_saes, theta=THETA)
        held_stable = held_probs >= (1 - EPSILON)

        held_stats = compute_single_run_statistics(
            arm_saes[held_out_seed], activations, config["device"],
            label=f"k={k} seed {held_out_seed}",
        )
        X_held = np.column_stack([v for _, v in build_predictors(held_stats)])
        # The floor has to be re-derived from THIS dictionary's firing counts: which latents
        # are under-measured is a property of the SAE being tested, not of the one trained on.
        held_live = held_stats["firing_counts"] >= MIN_FIRINGS
        held_mask = (held_stable | (held_probs <= EPSILON)) & held_live
        held_truth = held_stable[held_mask].astype(int)

        if len(np.unique(held_truth)) < 2:
            print("     held-out dictionary has only one class above the floor; skipped.")
        else:
            clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
            train_scaler = StandardScaler().fit(X_all[labeled_mask])
            clf.fit(train_scaler.transform(X_all[labeled_mask]), stable_mask[labeled_mask])

            # Two ways to normalize the held-out dictionary, answering different questions.
            # Reusing the training scaler also requires the raw scales to agree across
            # dictionaries; refitting asks only whether the learned relationship transfers,
            # and matches what a practitioner would do with an SAE in hand.
            same_scaler = roc_auc_score(held_truth, clf.predict_proba(
                train_scaler.transform(X_held[held_mask]))[:, 1])
            transfer_auroc = roc_auc_score(held_truth, clf.predict_proba(
                StandardScaler().fit(X_held[held_live]).transform(X_held[held_mask]))[:, 1])

            print(f"     trained on seed {anchor_seed}, tested on seed {held_out_seed} "
                  f"(n={int(held_mask.sum())}, {int(held_truth.sum())} stable)")
            print(f"     within-dictionary (cross-validated)    : {auroc_full:.3f}")
            print(f"     held-out, training-set scaler          : {same_scaler:.3f}")
            print(f"     held-out, rescaled on the held-out SAE : {transfer_auroc:.3f}")
            print(f"     transfer cost (rescaled)               : "
                  f"{transfer_auroc - auroc_full:+.3f}")

    # 4. Barely-firing latents are trivially separable, and the imputation reproduces what the
    # code did before the floor existed: a never-firing latent gets frequency ~0, mean
    # activation 0, contribution 0, and is labelled unstable because its untrained decoder row
    # matches nothing, so those values in combination identify it perfectly without saying
    # anything about predictability.
    print("  4. sensitivity to the firing floor")
    imputed = dict(stats)
    for key in ("mean_activation", "recon_contribution"):
        imputed[key] = np.nan_to_num(stats[key], nan=0.0)
    X_imputed = np.column_stack([v for _, v in build_predictors(imputed)])
    definite = stable_mask | unstable_mask
    print(f"     {'floor':>6}  {'n':>6}  {'% of dict':>9}  {'% stable':>8}  {'AUROC':>6}")
    for floor in (0, 1, 10, 100, 1000):
        m = definite & (firing_counts >= floor)
        if m.sum() < 20 or len(np.unique(stable_mask[m])) < 2:
            print(f"     {floor:>6}  {int(m.sum()):>6}  too few features left to score")
            continue
        print(f"     {floor:>6}  {int(m.sum()):>6}  {(firing_counts >= floor).mean():>8.1%}  "
              f"{stable_mask[m].mean():>7.1%}  {cv_auroc(X_imputed, stable_mask, m):>6.3f}")
    print(f"     the reported figure uses floor={MIN_FIRINGS}; a number that falls steeply as "
          f"the floor rises was carried by under-trained latents.")

    # --- artifacts ----------------------------------------------------------------------
    arm_df = pd.DataFrame({
        "feature_idx": np.arange(n_features),
        "reappearance_prob": reappearance_probs,
        "match_similarity": matching_info["similarities"][0],
        "is_stable": stable_mask,
        "is_unstable": unstable_mask,
        "is_discarded": middle_mask,
        "above_firing_floor": live_mask,
        "activation_freq": activation_freq,
        "firing_counts": firing_counts,
        "geometric_isolation": stats["geometric_isolation"],
        "recon_contribution": stats["recon_contribution"],
        "mean_activation": stats["mean_activation"],
        "encoder_norm": stats["encoder_norm"],
        "encoder_bias": stats["encoder_bias"],
    })
    arm_df.to_csv(OUTPUT_DIR / f"feature_stability_k{k}.csv", index=False)
    np.savez(
        OUTPUT_DIR / f"matching_info_k{k}.npz",
        similarities=np.array(matching_info["similarities"]),
        reappearance_probs=reappearance_probs,
        stable_indices=np.where(stable_mask)[0],
        unstable_indices=np.where(unstable_mask)[0],
    )
    for seed, sae in arm_saes.items():
        torch.save(sae.state_dict(), OUTPUT_DIR / f"sae_k{k}_seed{seed}.pt")
    print(f"\n  wrote feature_stability_k{k}.csv, matching_info_k{k}.npz and "
          f"{len(arm_saes)} state dicts")

    if RUN_TOP_EXAMPLES:
        # Ranked by match similarity so the inspected latents are the most confidently stable
        # ones rather than whichever happen to come first by index.
        inspect = [
            int(i) for i in np.argsort(-matching_info["similarities"][0])
            if stable_mask[i] and live_mask[i]
        ][:TOP_EXAMPLES_N_FEATURES]
        print(f"\n--- Top activating examples, {len(inspect)} most-matched stable latents ---")
        examples = get_top_activating_examples(
            model, anchor_sae,
            load_dataset("monology/pile-uncopyrighted", split="train", streaming=True),
            inspect,
        )
        for feat_idx in inspect:
            print(f"\n  FEATURE {feat_idx} (match sim "
                  f"{matching_info['similarities'][0][feat_idx]:.3f}, "
                  f"freq {activation_freq[feat_idx]:.4f})")
            for i, ex in enumerate(examples[feat_idx][:3]):
                print(f"    {i + 1}. act {ex['activation']:.2f} | token '{ex['token']}'")
                print(f"       ...{ex['context']}...")

    return {
        "k": k,
        "density": k / n_features,
        "mean_max_cos": mean_max_cos,
        "frac_paired": frac_paired,
        "stable_frac": float(stable_mask.mean()),
        "unstable_frac": float(unstable_mask.mean()),
        "discarded_frac": float(middle_mask.mean()),
        "live_frac": float(live_mask.mean()),
        "n_labelled": int(labeled_mask.sum()),
        "auroc_full": float(auroc_full),
        "auroc_freq_only": float(auroc_freq),
        "delta_over_freq": float(delta),
        "auroc_hungarian_full": float(auroc_hungarian),
        "stable_frac_hungarian": float(hungarian_stable_frac),
        "transfer_auroc": float(transfer_auroc),
        "best_single": max(single.items(), key=lambda x: x[1])[0],
    }


summaries = []
for _k in K_VALUES:
    _arm_saes = {s: trained_saes[(_k, s)] for s in SEEDS}
    summaries.append(analyze_arm(_k, _arm_saes, activations, CONFIG))

"""## 9. Compare the arms

The point of the sweep. Each row is one k, and the columns are the numbers that decide which
sparsity to carry forward: how much of the dictionary is measurable, how stable the features
look, and how much the predictors add over firing rate alone.
"""

summary_df = pd.DataFrame(summaries)
summary_df.to_csv(OUTPUT_DIR / "sweep_summary.csv", index=False)
with open(OUTPUT_DIR / "sweep_summary.json", "w") as fh:
    json.dump({"config": {k: v for k, v in CONFIG.items()},
               "checkpoint_tokens": CHECKPOINT_TOKENS,
               "theta": THETA, "epsilon": EPSILON, "min_firings": MIN_FIRINGS,
               "eval_tokens": int(len(activations)),
               "arms": summaries}, fh, indent=2, default=str)

print("\n" + "=" * 72)
print("SWEEP SUMMARY")
print("=" * 72)
print(f"{'k':>5} {'density':>8} {'live':>7} {'meanCos':>8} {'paired':>7} {'stable':>7} "
      f"{'AUROC':>7} {'freqOnly':>9} {'delta':>7} {'transfer':>9}")
for row in summaries:
    print(f"{row['k']:>5} {row['density']:>7.2%} {row['live_frac']:>6.1%} "
          f"{row['mean_max_cos']:>8.3f} {row['frac_paired']:>6.1%} "
          f"{row['stable_frac']:>6.1%} {row['auroc_full']:>7.3f} "
          f"{row['auroc_freq_only']:>9.3f} {row['delta_over_freq']:>+7.3f} "
          f"{row['transfer_auroc']:>9.3f}")

print("\nHow to read this:")
print("  live     -- share of the dictionary above the firing floor. A low value means the "
      "arm is scoring a fraction of its latents, and the AUROC is not about the rest.")
print("  meanCos  -- mean best decoder cosine to another seed. The directly comparable "
      "number: published unregularized TopK SAEs on this model/layer report <=0.32 among "
      "alive features, and our L1 arm was far above that.")
print("  stable   -- share with p_hat >= 0.95. A value near 1 makes the classification "
      "target close to degenerate and the absolute AUROC close to meaningless.")
print("  delta    -- AUROC(full) - AUROC(frequency-only). This is the hypothesis's actual "
      "claim, and the number to compare across arms.")

_finite = [r for r in summaries if np.isfinite(r["delta_over_freq"])]
if _finite:
    _best = max(_finite, key=lambda r: r["delta_over_freq"])
    print(f"\nLargest increment over the frequency baseline: k={_best['k']} "
          f"(delta {_best['delta_over_freq']:+.3f}, live {_best['live_frac']:.1%}, "
          f"stable {_best['stable_frac']:.1%})")
    print("Read that alongside `live` and `stable` before choosing k: the least healthy "
          "dictionary produced the highest score in our earlier runs, which is the trap this "
          "table exists to make visible.")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ks = summary_df["k"].tolist()

axes[0, 0].plot(ks, summary_df["stable_frac"], "o-", label="stable (many-to-one)")
if summary_df["stable_frac_hungarian"].notna().any():
    axes[0, 0].plot(ks, summary_df["stable_frac_hungarian"], "s--", label="stable (one-to-one)")
axes[0, 0].set_ylabel("Fraction of latents labelled stable")
axes[0, 0].set_title("Apparent stability vs sparsity")

axes[0, 1].plot(ks, summary_df["mean_max_cos"], "o-")
axes[0, 1].axhline(0.32, color="red", linestyle="--", linewidth=1,
                   label="published unregularized TopK (alive features)")
axes[0, 1].set_ylabel("Mean max decoder cosine")
axes[0, 1].set_title("Cross-seed agreement vs sparsity")

axes[1, 0].plot(ks, summary_df["auroc_full"], "o-", label="full model")
axes[1, 0].plot(ks, summary_df["auroc_freq_only"], "s--", label="frequency only")
axes[1, 0].axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
axes[1, 0].set_ylabel("Cross-validated AUROC")
axes[1, 0].set_title("Predictability vs sparsity")

axes[1, 1].plot(ks, summary_df["delta_over_freq"], "o-", label="AUROC(full) - AUROC(freq)")
axes[1, 1].plot(ks, summary_df["live_frac"], "^--", label="fraction above firing floor")
axes[1, 1].axhline(0.0, color="gray", linestyle=":", linewidth=1)
axes[1, 1].set_ylabel("Increment / fraction")
axes[1, 1].set_title("What the predictors add, and on how much of the dictionary")

for ax in axes.flat:
    ax.set_xlabel("k (active latents per token)")
    ax.set_xticks(ks)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sweep_comparison.png", dpi=150)
plt.show()

with open(OUTPUT_DIR / "config.json", "w") as fh:
    json.dump({**CONFIG, "checkpoint_tokens": CHECKPOINT_TOKENS}, fh, indent=2)

print(f"\nSaved sweep_summary.csv, sweep_summary.json, sweep_comparison.png and config.json "
      f"to {OUTPUT_DIR}")
