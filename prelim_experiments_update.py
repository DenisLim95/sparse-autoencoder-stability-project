# -*- coding: utf-8 -*-
"""prelim-experiments.ipynb

SAE stability project — preliminary experiments.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple, List, Dict
from transformer_lens import HookedTransformer
from datasets import load_dataset

# Configuration
CONFIG = {
    "model_name": "pythia-70m-deduped",
    "hook_point": "blocks.3.hook_resid_post",  # Middle layer of Pythia-70m (6 layers, so layer 3)
    "d_model": 512,  # Pythia-70m hidden dimension
    "n_features": 2048,  # SAE dictionary size (4x expansion)
    # How sparsity is imposed. "l1" is the objective every checkpoint on both Hub repos was
    # trained under; "topk" keeps exactly the k largest latents per token. The L1 arm settled
    # at L0 ~= 345 of 2048 features, i.e. 17% of the dictionary firing on every token, far
    # denser than published SAEs. L1 cannot target an L0 -- you sweep the coefficient and see
    # what you get, and each probe costs a full training run. TopK fixes L0 = k by
    # construction, which is the only affordable way to reach a defensible sparsity on a
    # machine that gets deleted on a deadline.
    # `or` rather than a default, so a variable set to the empty string (easy to do in a
    # notebook when clearing a setting) reads as unset instead of crashing on int("").
    "sparsity": os.environ.get("SAE_SPARSITY") or "l1",  # "l1" | "topk"
    "k": int(os.environ.get("SAE_TOPK_K") or 32),  # active latents per token, TopK only
    # Every checkpoint on both Hub repos (1M through 8B, all seeds) was trained at 1.0 --
    # verified from each checkpoint's saved config and cross-checked against the recorded
    # losses. Resuming those checkpoints under a different coefficient would change the
    # objective mid-curve, so this must stay at 1.0 unless the whole sweep is retrained.
    #
    # On units: the penalty is f.abs().mean(), which averages over the batch AND the feature
    # dimension, while recon_loss averages over the batch and d_model. The usual convention
    # sums over features, so in standard units this coefficient is really
    # l1_coeff * d_model / n_features = 0.25 -- which is why 1.0 looks nothing like the ~1e-3
    # in the literature. Do not switch the penalty to a sum without retraining the whole
    # sweep, as that changes the objective every existing checkpoint was trained under. The
    # side effect worth remembering: effective pressure scales as 1/n_features, so raising
    # the dictionary size silently weakens sparsity too.
    "l1_coeff": 1.0,  # Sparsity coefficient, L1 only
    # L1 needs a unit-norm decoder or the penalty is gamed by shrinking the encoder and
    # inflating the decoder, which is why decoder norm measured identically 1.000 with zero
    # variance and therefore zero predictive value. TopK has no such term to game, so setting
    # this False under TopK makes decoder norm a live predictor again. It changes two things
    # at once though, so leave it True for the first L1-vs-TopK comparison.
    "normalize_decoder": os.environ.get("SAE_NORMALIZE_DECODER", "1") != "0",
    "lr": 1e-3,
    "batch_size": 256,
    "seq_len": 128,
    # Only 42, 256 and 1024 have checkpoints above 100M. The shared stream is paced by its
    # least-trained seed, so including 137 or 512 here drags the whole run back to the 100M
    # milestone they share -- set SAE_SEEDS=42,256,1024 to resume the 1B-8B curve, and train
    # the catch-up seeds as a separate process (SAE_SEEDS=137,512) where they cost it nothing.
    "seeds": [
        int(s) for s in os.environ.get("SAE_SEEDS", "42,137,256,512,1024").split(",") if s.strip()
    ],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"Using device: {CONFIG['device']}")
print(f"Training seeds: {CONFIG['seeds']}")
if CONFIG["sparsity"] == "topk":
    print(f"Objective: TopK, k={CONFIG['k']} (L0 is exactly k by construction)")
elif CONFIG["sparsity"] == "l1":
    _lambda_std = CONFIG["l1_coeff"] * CONFIG["d_model"] / CONFIG["n_features"]
    print(f"Objective: L1, l1_coeff={CONFIG['l1_coeff']:g} "
          f"(= {_lambda_std:g} in sum-over-features units)")
else:
    raise SystemExit(f"Unknown CONFIG['sparsity']={CONFIG['sparsity']!r}; use 'l1' or 'topk'.")
if not CONFIG["normalize_decoder"] and CONFIG["sparsity"] == "l1":
    raise SystemExit(
        "normalize_decoder=False with the L1 objective: the penalty is then trivially gamed "
        "by shrinking the encoder and inflating the decoder, so the run would drift to a "
        "meaningless solution. Unit-norm decoders are only optional under TopK."
    )

# Token budgets at which an SAE checkpoint is saved, so stability can be compared
# across training scale. Roughly geometric: on a leased machine that gets deleted on a
# deadline, each milestone is a complete, analyzable five-seed result banked early, so a
# run cut short still yields a scaling curve instead of nothing.
CHECKPOINT_TOKENS = [
    1_000_000,
    50_000_000,
    100_000_000,
    # Resume point for seeds 42/256/1024 only, supplied by SAE_SEED_REPO. Seeds 137 and 512
    # have nothing above 100M, so a run including them resumes from there instead.
    1_000_000_000,
    2_000_000_000,
    3_000_000_000,
    5_000_000_000,
    8_000_000_000,
]

# Cap the run below the full curve, e.g. SAE_MAX_TOKENS=100000000 to stop at the 100M
# milestone. The TopK arm only needs comparing against the L1 checkpoints at one matched
# budget, so training it out to 8B would spend days of GPU time answering nothing extra.
_max_tokens = int(os.environ.get("SAE_MAX_TOKENS") or 0)
if _max_tokens:
    _capped = [t for t in CHECKPOINT_TOKENS if t <= _max_tokens]
    if not _capped:
        raise SystemExit(
            f"SAE_MAX_TOKENS={_max_tokens:,} is below the first milestone "
            f"({min(CHECKPOINT_TOKENS):,}), so there would be nothing to checkpoint."
        )
    CHECKPOINT_TOKENS = _capped

# Read-only repo to seed checkpoints from, e.g. SAE_SEED_REPO=ndasari/SAE_project. Lets a run
# start from a collaborator's completed milestones; never uploaded to, so their repo is safe.
SEED_REPO_ID = os.environ.get("SAE_SEED_REPO") or None

# Consecutive entries in CHECKPOINT_TOKENS are up to 4B tokens apart -- far more than a
# single Colab session can cover. Without a rolling checkpoint in between, a disconnect
# part-way through a gap rewinds to the previous milestone, so a run can bounce off the
# same gap forever without advancing. This bounds the loss from a disconnect instead.
CHECKPOINT_EVERY_SECONDS = 900

# How often (in optimizer steps) to record a point on the training curves.
LOG_EVERY_STEPS = 200

# Batches buffered ahead by the producer thread. Each holds batch_size * seq_len * d_model
# floats (~64 MB at the current settings), so keep this small.
PREFETCH_BATCHES = 4

# --- Persistent output directory ---------------------------------------------
# An 8B-token run will outlive the session that started it, so results must land
# somewhere that survives a disconnect.
#   - SAE_RESULTS_BASE: set this on any host with a persistent volume (e.g. JupyterHub,
#     where there is no Drive to mount) to keep results off ephemeral scratch space.
#   - Colab: mount Drive. The already-mounted check comes first because drive.mount()
#     needs the notebook's auth flow and fails from a subprocess (`!python ...`).
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


# The default full-curve L1 run keeps its original directory name byte-for-byte, so the 1M-8B
# checkpoints already on both Hub repos still resolve. Anything that changes the objective
# gets a suffix, which also stops two incomparable families of checkpoints being written into
# the same directory and silently resumed from each other.
_objective_tag = "" if CONFIG["sparsity"] == "l1" else f"_topk{CONFIG['k']}"
if not CONFIG["normalize_decoder"]:
    _objective_tag += "_freedec"
RUN_NAME = (
    f"{CONFIG['model_name']}_L{_layer}"
    f"_{_token_label(max(CHECKPOINT_TOKENS))}tok{_objective_tag}"
)
OUTPUT_DIR = Path(RESULTS_BASE) / RUN_NAME

# Hub paths are scoped by objective for the same reason the local directory is, and it matters
# more here: SAE_HF_REPO is both the upload and the download location, so without this a TopK
# run would pull the L1 checkpoints out of the shared repo into its own directory and then
# refuse to resume from them -- a configuration mistake that presents as a crash. The default
# L1 run keeps the original flat prefixes so every file already on the Hub still resolves.
_hub_scope = f"/{_objective_tag.lstrip('_')}" if _objective_tag else ""
HUB_CHECKPOINT_PREFIX = f"checkpoints{_hub_scope}"
HUB_RESULTS_PREFIX = f"results{_hub_scope}"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Hub repo to mirror checkpoints to, e.g. SAE_HF_REPO=yourname/sae-stability. Set this
# whenever the machine's disk is not durable -- leased GPU boxes are commonly on instance
# storage that is destroyed on shutdown, so local checkpoints alone can vanish with the run.
# Point it at a repo you can write to: someone else's fails on auth, and the failure is only
# warned about, so training would continue persisting nothing.
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


def activation_stream_generator(model, dataset_name: str, hook_point: str, seq_len: int, batch_size: int, device: str):
    """
    Infinite generator yielding (batch_size * seq_len, d_model) activation tensors.
    Re-tokenizes and streams fresh Pile text; does not pre-collect a fixed n_tokens.
    Calling this function again (e.g. once per seed) starts a fresh read from the
    beginning of the streaming dataset, so every seed sees the same data in the
    same order -- only the SAE's own initialization differs across seeds.
    """
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    token_buffer = []

    for example in dataset:
        tokens = model.tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=seq_len * 10
        )["input_ids"][0]
        token_buffer.extend(tokens.tolist())

        while len(token_buffer) >= seq_len * batch_size:
            batch_tokens = torch.tensor(token_buffer[:seq_len * batch_size]).reshape(batch_size, seq_len)
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


"""## 2. Define SAE Architecture"""

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with either an L1 penalty or a TopK activation.

    Architecture:
        encoder: x -> ReLU(W_enc @ (x - b_dec) + b_enc)      [sparsity="l1"]
                 x -> ReLU(TopK_k(W_enc @ (x - b_dec) + b_enc))  [sparsity="topk"]
        decoder: f -> W_dec @ f + b_dec

    Under "l1" sparsity is a soft penalty added to the loss and the resulting L0 is whatever
    the coefficient happens to produce. Under "topk" at most k latents are non-zero by
    construction and no sparsity term enters the loss at all.
    """

    def __init__(self, d_model: int, n_features: int, seed: int,
                 sparsity: str = "l1", k: int = 32):
        super().__init__()
        torch.manual_seed(seed)

        self.d_model = d_model
        self.n_features = n_features
        self.sparsity = sparsity
        self.k = min(k, n_features)

        # Encoder weights and bias
        self.W_enc = nn.Parameter(torch.randn(d_model, n_features) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder weights and bias
        self.W_dec = nn.Parameter(torch.randn(n_features, d_model) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations."""
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        if self.sparsity == "topk":
            # Select on the pre-activations, then rectify. Taking the top k of an already
            # rectified vector would pick arbitrary features out of the zeros whenever fewer
            # than k are positive, inventing activations that carry no signal.
            idx = pre_acts.topk(self.k, dim=-1).indices
            keep = torch.zeros_like(pre_acts, dtype=torch.bool).scatter_(-1, idx, True)
            pre_acts = torch.where(keep, pre_acts, torch.zeros_like(pre_acts))
        return F.relu(pre_acts)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to input space."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning reconstruction, features, and loss components.

        Returns:
            x_hat: Reconstructed input
            f: Feature activations
            loss_dict: Dictionary with reconstruction and sparsity losses
        """
        f = self.encode(x)
        x_hat = self.decode(f)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_hat, x)

        # L1 magnitude of the code. Always reported so the training curves stay comparable
        # across objectives, but only added to the loss when sparsity="l1" -- under TopK the
        # constraint is structural and penalising magnitude on top of it would just shrink
        # the k surviving activations.
        sparsity_loss = f.abs().mean()

        return x_hat, f, {"recon_loss": recon_loss, "sparsity_loss": sparsity_loss}

    def normalize_decoder(self):
        """Normalize decoder columns to unit norm (call after each optimization step)."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


"""## 3. Train SAEs with Different Seeds (streaming + checkpointed)"""


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
        "sparsity_loss": [],
        "l1_term": [],
        "total_loss": [],
        "l0": [],
        "dead_frac": [],
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
    rather than fighting it for the interpreter.
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


def rolling_checkpoint_name(seeds) -> str:
    """Name the shared rolling checkpoint after its seed set.

    Two processes training different seed sets -- one resuming from 1B, another catching the
    remaining seeds up on a second GPU -- can then share a checkpoint directory without
    clobbering each other's rolling state. Milestone files are already per-seed.
    """
    return "shared_latest_seeds" + "-".join(str(s) for s in sorted(seeds)) + ".pt"


def restore_checkpoints_from_hub(repo_id: str, checkpoint_dir: Path, seeds=None,
                                 prefix: str = "checkpoints"):
    """Pull any milestone checkpoints already on the Hub into the local checkpoint dir.

    Leased GPU machines get replaced, and the replacement arrives with an empty disk. Since
    milestone checkpoints are enough to resume from (see load_shared_resume_state), fetching
    them here means a run continues on a new machine instead of restarting from zero.

    Restricted to `seeds` when given, so seeding from a collaborator's repo doesn't drag down
    checkpoints for seeds this run isn't training. `prefix` scopes the search to one
    objective's subdirectory, so a run never restores weights trained under a different one.
    """
    from huggingface_hub import HfApi, hf_hub_download

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"{re.escape(prefix)}/seed(\d+)_tokens\d+\.pt")
    try:
        remote = []
        for f in HfApi().list_repo_files(repo_id, repo_type="model"):
            m = pattern.fullmatch(f)
            if m and (seeds is None or int(m.group(1)) in set(seeds)):
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


def load_shared_resume_state(checkpoint_dir: Path, seeds: list, checkpoint_tokens: list):
    """Find the furthest point all seeds can resume from together.

    Seeds advance in lockstep, so they share one rolling checkpoint. If that file is missing
    or unreadable, fall back to the largest milestone every seed reached, which costs one
    milestone interval rather than the whole run.
    """
    shared_path = checkpoint_dir / rolling_checkpoint_name(seeds)
    legacy_path = checkpoint_dir / "shared_latest.pt"
    if not shared_path.exists() and legacy_path.exists():
        shared_path = legacy_path  # rolling checkpoint written before names were seed-scoped
    if shared_path.exists():
        try:
            # Explicit weights_only=False: these are our own checkpoints and they carry
            # optimizer/config/history alongside the weights. Newer torch flips the default
            # to True, which would break resume part-way through a multi-day run.
            return torch.load(shared_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Ignoring unreadable rolling checkpoint {shared_path.name}: {e}")

    for milestone in sorted(checkpoint_tokens, reverse=True):
        paths = {s: checkpoint_dir / f"seed{s}_tokens{milestone}.pt" for s in seeds}
        if not all(p.exists() for p in paths.values()):
            continue
        try:
            ckpts = {
                s: torch.load(p, map_location="cpu", weights_only=False)
                for s, p in paths.items()
            }
        except Exception as e:
            print(f"Ignoring unreadable milestone {milestone:,}: {e}")
            continue
        print(f"No rolling checkpoint; falling back to the {milestone:,}-token milestone.")
        return {
            "tokens_seen": min(c["tokens_seen"] for c in ckpts.values()),
            "step": min(c.get("step", 0) for c in ckpts.values()),
            "models": {s: c["model_state_dict"] for s, c in ckpts.items()},
            "optimizers": {s: c["optimizer_state_dict"] for s, c in ckpts.items()},
            "histories": {s: c.get("history", empty_history()) for s, c in ckpts.items()},
            "config": next(iter(ckpts.values())).get("config", {}),
        }
    return None


def train_saes_shared_stream(
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
    """Train one SAE per seed on a single shared pass over the activations.

    Every seed reads the same data in the same order by design -- only initialization is
    meant to differ -- so training them one after another regenerated identical activations
    once per seed. Producing each batch once and updating all seeds from it does the same
    arithmetic for ~1/5th of the data-pipeline work, and makes the seeds see identical data
    by construction even across restarts.

    Milestone checkpoints are written per seed in the same format as before, so
    stability_check.py needs no changes. The rolling checkpoint is shared, since the seeds
    advance together.

    Returns (saes, histories), both keyed by seed.
    """
    device = config["device"]
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_tokens = sorted(checkpoint_tokens)

    # Each __init__ reseeds the RNG before drawing, so construction order doesn't matter.
    saes = {
        s: SparseAutoencoder(
            config["d_model"], config["n_features"], seed=s,
            sparsity=config["sparsity"], k=config["k"],
        ).to(device)
        for s in seeds
    }
    optimizers = {s: torch.optim.Adam(saes[s].parameters(), lr=config["lr"]) for s in seeds}
    histories = {s: empty_history() for s in seeds}
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
            hf_repo_id, checkpoint_dir, seeds, prefix=hub_checkpoint_prefix
        )

    # Second, so our own further-along checkpoints win: restore only fetches what is missing.
    if seed_repo_id is not None:
        print(f"Seeding from collaborator repo hf.co/{seed_repo_id} (read-only)")
        restore_checkpoints_from_hub(
            seed_repo_id, checkpoint_dir, seeds, prefix=hub_checkpoint_prefix
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

    resume = load_shared_resume_state(checkpoint_dir, seeds, checkpoint_tokens)
    if resume is not None:
        # Resuming under a different objective than the checkpoints were trained with would
        # silently put a discontinuity in the middle of the scaling curve, and nothing
        # downstream could detect it. Refuse rather than produce unattributable numbers.
        prior = resume.get("config", {})
        # Checkpoints written before the sparsity flag existed are all L1, so absence means
        # "l1" rather than "unknown" -- treating it as unknown would let a TopK run silently
        # resume from L1 weights.
        prior_sparsity = prior.get("sparsity", "l1")
        mismatches = []
        if prior_sparsity != config["sparsity"]:
            mismatches.append(("sparsity", prior_sparsity, config["sparsity"]))
        elif prior_sparsity == "topk":
            if prior.get("k") is not None and prior["k"] != config["k"]:
                mismatches.append(("k", prior["k"], config["k"]))
        elif prior.get("l1_coeff") is not None and prior["l1_coeff"] != config["l1_coeff"]:
            mismatches.append(("l1_coeff", prior["l1_coeff"], config["l1_coeff"]))
        prior_norm = prior.get("normalize_decoder", True)
        if prior_norm != config["normalize_decoder"]:
            mismatches.append(("normalize_decoder", prior_norm, config["normalize_decoder"]))
        if mismatches:
            detail = "; ".join(f"{k}: checkpoint={p!r} vs CONFIG={c!r}" for k, p, c in mismatches)
            raise SystemExit(
                f"Refusing to resume, the objective would change mid-run ({detail}). Either "
                f"match CONFIG to the checkpoints, or train from scratch into an empty "
                f"checkpoint directory -- each objective gets its own directory, so this "
                f"usually means two runs were pointed at the same one."
            )
        for s in seeds:
            saes[s].load_state_dict(resume["models"][s])
            optimizers[s].load_state_dict(resume["optimizers"][s])
            histories[s] = resume["histories"].get(s, empty_history())
        tokens_seen = resume["tokens_seen"]
        step = resume.get("step", 0)
        print(f"Resumed all {len(seeds)} seeds at {tokens_seen:,} tokens")
        print("CAVEAT: the activation stream restarts from the beginning of the dataset, so "
              "tokens already trained on will be seen again. Past the first restart the "
              "token counts measure optimization, not unique tokens. All seeds replay the "
              "same data, so cross-seed comparisons stay valid.")

    next_checkpoint_idx = sum(t <= tokens_seen for t in checkpoint_tokens)
    if next_checkpoint_idx >= len(checkpoint_tokens):
        print(f"Already at {tokens_seen:,} tokens; nothing left to train.")
        return saes, histories

    # Accumulate curve metrics as on-device tensors and only pull them to the host at
    # logging time; calling .item() every step would force a GPU sync 244k times.
    interval_sums = {s: torch.zeros(4, device=device) for s in seeds}  # recon, sparsity, total, l0
    interval_n = 0
    seen_active = {
        s: torch.zeros(config["n_features"], dtype=torch.bool, device=device) for s in seeds
    }
    last_checkpoint_time = time.time()

    def build_seed_state(seed):
        return {
            "model_state_dict": saes[seed].state_dict(),
            "optimizer_state_dict": optimizers[seed].state_dict(),
            "tokens_seen": tokens_seen,
            "step": step,
            "seed": seed,
            "config": config,
            "history": histories[seed],
        }

    def build_shared_state():
        return {
            "tokens_seen": tokens_seen,
            "step": step,
            "config": config,
            "models": {s: saes[s].state_dict() for s in seeds},
            "optimizers": {s: optimizers[s].state_dict() for s in seeds},
            "histories": histories,
        }

    def write_histories():
        for s in seeds:
            with open(checkpoint_dir.parent / f"training_history_seed{s}.json", "w") as fh:
                json.dump(histories[s], fh)

    # Zero under TopK, where sparsity is structural rather than penalised. Kept as a weight
    # rather than a branch so the loss and the logged l1_term cannot disagree about it.
    l1_weight = config["l1_coeff"] if config["sparsity"] == "l1" else 0.0
    unit_norm_decoder = config["normalize_decoder"]

    for batch in activation_stream:
        batch = batch.to(device, non_blocking=True)

        for s in seeds:
            sae, optimizer = saes[s], optimizers[s]
            x_hat, f, loss_dict = sae(batch)
            loss = loss_dict["recon_loss"] + l1_weight * loss_dict["sparsity_loss"]

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
                interval_sums[s][0] += loss_dict["recon_loss"].detach()
                interval_sums[s][1] += loss_dict["sparsity_loss"].detach()
                interval_sums[s][2] += loss.detach()
                interval_sums[s][3] += active.float().sum(dim=1).mean()
                seen_active[s] |= active.any(dim=0)

        # batch is already the flattened (batch_size * seq_len, d_model) activations
        # from activation_stream_generator -- batch.shape[0] IS the real token count
        # for this step. Every seed consumed this same batch, so count it once.
        tokens_seen += batch.shape[0]
        step += 1
        interval_n += 1

        if step % log_every_steps == 0:
            l0s, recons = [], []
            for s in seeds:
                recon, sparsity, total, l0 = (interval_sums[s] / interval_n).tolist()
                dead_frac = 1.0 - seen_active[s].float().mean().item()
                h = histories[s]
                h["step"].append(step)
                h["tokens_seen"].append(tokens_seen)
                h["recon_loss"].append(recon)
                h["sparsity_loss"].append(sparsity)
                h["l1_term"].append(l1_weight * sparsity)
                h["total_loss"].append(total)
                h["l0"].append(l0)
                # Dead = never fired anywhere in this logging window.
                h["dead_frac"].append(dead_frac)
                l0s.append(l0)
                recons.append(recon)
                interval_sums[s].zero_()
                seen_active[s] = torch.zeros(
                    config["n_features"], dtype=torch.bool, device=device
                )
            interval_n = 0
            worst_dead = max(100 * histories[s]["dead_frac"][-1] for s in seeds)
            mean_sparsity = np.mean([histories[s]["sparsity_loss"][-1] for s in seeds])
            # Under TopK the penalty is not in the loss, so reporting a weighted term would
            # be a column of zeros; show the raw code magnitude instead.
            sparsity_col = (f"l1_term {l1_weight * mean_sparsity:.5f}" if l1_weight
                            else f"|f|_1 {mean_sparsity:.5f}")
            print(f"step {step:>7} | {tokens_seen:>14,} tok | "
                  f"recon {np.mean(recons):.5f} | "
                  f"{sparsity_col} | "
                  f"L0 {np.mean(l0s):6.1f} ({min(l0s):.0f}-{max(l0s):.0f}) | "
                  f"dead {worst_dead:5.1f}%")

        if time.time() - last_checkpoint_time >= checkpoint_every_seconds:
            save_checkpoint_atomic(build_shared_state(), checkpoint_dir / rolling_checkpoint_name(seeds))
            write_histories()
            last_checkpoint_time = time.time()
            print(f"rolling checkpoint at {tokens_seen:,} tokens")

        if next_checkpoint_idx < len(checkpoint_tokens) and tokens_seen >= checkpoint_tokens[next_checkpoint_idx]:
            milestone = checkpoint_tokens[next_checkpoint_idx]
            for s in seeds:
                ckpt_name = f"seed{s}_tokens{milestone}.pt"
                ckpt_path = checkpoint_dir / ckpt_name
                save_checkpoint_atomic(build_seed_state(s), ckpt_path)
                mirror_to_hub(ckpt_path, f"{hub_checkpoint_prefix}/{ckpt_name}")

            save_checkpoint_atomic(build_shared_state(), checkpoint_dir / rolling_checkpoint_name(seeds))
            write_histories()
            for s in seeds:
                name = f"training_history_seed{s}.json"
                mirror_to_hub(checkpoint_dir.parent / name, f"{hub_results_prefix}/{name}")

            print(f"milestone reached: all {len(seeds)} seeds checkpointed at "
                  f"{milestone:,} tokens -- runnable through stability_check.py now"
                  + (f", mirrored to hf.co/{hf_repo_id}" if hf_api is not None else ""))
            next_checkpoint_idx += 1

        if next_checkpoint_idx >= len(checkpoint_tokens):
            break

    save_checkpoint_atomic(build_shared_state(), checkpoint_dir / rolling_checkpoint_name(seeds))
    write_histories()

    return saes, histories


print(f"=== Training {len(CONFIG['seeds'])} seeds on a shared activation stream ===")
stream = activation_stream_generator(
    model=model,
    dataset_name="monology/pile-uncopyrighted",
    hook_point=CONFIG["hook_point"],
    seq_len=CONFIG["seq_len"],
    batch_size=CONFIG["batch_size"],
    device=CONFIG["device"],
)
trained_saes, training_histories = train_saes_shared_stream(
    seeds=CONFIG["seeds"],
    activation_stream=prefetch_batches(stream, max_queued=PREFETCH_BATCHES),
    config=CONFIG,
    checkpoint_tokens=CHECKPOINT_TOKENS,
    checkpoint_dir=CHECKPOINT_DIR,
    hf_repo_id=HF_REPO_ID,
    seed_repo_id=SEED_REPO_ID,
    hub_checkpoint_prefix=HUB_CHECKPOINT_PREFIX,
    hub_results_prefix=HUB_RESULTS_PREFIX,
)

print(f"\nTrained {len(trained_saes)} SAEs with seeds: {list(trained_saes.keys())}")
print(f"Checkpoints saved at token counts: {CHECKPOINT_TOKENS}")

"""### 3b. Training curves — did the sparsity setting land somewhere usable?

Under the L1 objective the question is whether the penalty is dominating. The warning signs,
in order of how conclusive they are: the dead-feature fraction climbing toward 100%, L0
collapsing toward 0, and the weighted L1 term sitting far above the reconstruction term. Any
of those means most features are being pushed to zero and the SAE is buying sparsity at the
cost of reconstructing anything. Note that `l1_coeff = 1.0` is not comparable to the ~1e-3
quoted in papers: the penalty here is averaged over features rather than summed, so in the
usual units it is `l1_coeff * d_model / n_features = 0.25`.

Under TopK there is no coefficient to get wrong, because L0 is exactly k. The failure mode
moves to dead features, which TopK strands far more readily than L1 does, and which corrupt
the stability analysis rather than merely wasting capacity — see the diagnostic below.
"""


def plot_training_curves(histories: Dict[int, Dict[str, list]], config, save_path=None):
    """Plot the diagnostics that reveal a too-strong sparsity penalty."""
    histories = {s: h for s, h in histories.items() if h and h["tokens_seen"]}
    if not histories:
        print("No training history recorded yet (fewer than LOG_EVERY_STEPS steps run).")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for seed, h in histories.items():
        axes[0, 0].plot(h["tokens_seen"], h["recon_loss"], label=f"seed {seed}")
        axes[1, 0].plot(h["tokens_seen"], h["l0"], label=f"seed {seed}")
        axes[1, 1].plot(h["tokens_seen"], [100 * d for d in h["dead_frac"]], label=f"seed {seed}")

    axes[0, 0].set_ylabel("Reconstruction loss (MSE)")
    axes[0, 0].set_title("Reconstruction loss")

    # Compare the two loss terms directly, on one seed, to see which dominates.
    anchor_seed = list(histories.keys())[0]
    h = histories[anchor_seed]
    axes[0, 1].plot(h["tokens_seen"], h["recon_loss"], label="reconstruction")
    if config["sparsity"] == "topk":
        # l1_term is identically zero under TopK and would plot as nothing on a log axis.
        axes[0, 1].plot(h["tokens_seen"], h["sparsity_loss"], label="mean |f| (unpenalized)")
        axes[0, 1].set_title(f"Reconstruction vs code magnitude (seed {anchor_seed})")
    else:
        axes[0, 1].plot(h["tokens_seen"], h["l1_term"],
                        label=f"l1_coeff x L1 ({config['l1_coeff']:g})")
        axes[0, 1].set_title(f"Loss terms compared (seed {anchor_seed})")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Loss term")

    axes[1, 0].set_ylabel("L0 (mean active features/token)")
    axes[1, 0].set_title(
        f"Sparsity: L0 should sit at k={config['k']}" if config["sparsity"] == "topk"
        else "Sparsity: L0 -> 0 means over-penalized"
    )

    axes[1, 1].set_ylabel("Dead features (%)")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_title("Dead features per logging window")

    for ax in axes.flat:
        ax.set_xlabel("Tokens seen")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    plt.show()

    if config["sparsity"] == "topk":
        print(f"\n=== TopK k={config['k']} diagnostic (final logging window) ===")
        for seed, h in histories.items():
            print(f"  seed {seed}: L0={h['l0'][-1]:.1f} (target {config['k']}), "
                  f"dead={100 * h['dead_frac'][-1]:.1f}%, recon={h['recon_loss'][-1]:.5f}")
        worst_dead = max(100 * h["dead_frac"][-1] for h in histories.values())
        if worst_dead > 20:
            live = config["n_features"] * (1 - worst_dead / 100)
            print(f"\n  WARNING: up to {worst_dead:.1f}% of features never fired in the last "
                  f"logging window, leaving about {live:.0f} of {config['n_features']} live. "
                  f"TopK strands latents this way, which is why the published recipe adds an "
                  f"auxiliary revival loss. This matters for the result and not just for "
                  f"capacity: a dead feature's decoder row stays near initialization, so "
                  f"nothing matches it and it is labelled unstable, while its activation "
                  f"frequency is exactly zero. The classifier can then separate the classes "
                  f"by detecting dead features, inflating AUROC while learning nothing about "
                  f"real ones. Exclude never-firing features before trusting the number.")
        else:
            print(f"\n  Dead fraction at most {worst_dead:.1f}%; the dictionary is in use and "
                  f"L0 is pinned at k by construction, so there is no sparsity to tune.")
        return

    print(f"\n=== l1_coeff = {config['l1_coeff']:g} diagnostic (final logging window) ===")
    for seed, h in histories.items():
        ratio = h["l1_term"][-1] / h["recon_loss"][-1] if h["recon_loss"][-1] > 0 else float("inf")
        print(f"  seed {seed}: L0={h['l0'][-1]:.1f}, dead={100 * h['dead_frac'][-1]:.1f}%, "
              f"l1_term/recon={ratio:.2f}")

    worst_dead = max(100 * h["dead_frac"][-1] for h in histories.values())
    lowest_l0 = min(h["l0"][-1] for h in histories.values())
    if worst_dead > 90 or lowest_l0 < 1:
        print(f"\n  WARNING: up to {worst_dead:.1f}% of features dead and L0 as low as "
              f"{lowest_l0:.2f}. l1_coeff={config['l1_coeff']:g} looks too large -- the SAE is "
              f"collapsing to the trivial all-zero solution. Lowering it is only valid for a "
              f"sweep retrained from scratch; every existing checkpoint was trained at 1.0 and "
              f"resuming one under a different coefficient is refused on purpose.")
    else:
        print(f"\n  L0 and dead-feature fraction look non-degenerate at "
              f"l1_coeff={config['l1_coeff']:g}.")


plot_training_curves(
    training_histories,
    CONFIG,
    save_path=OUTPUT_DIR / "training_curves.png",
)

"""## 4. Feature Matching Between SAEs (decoder-only, Gerasimov et al. Section 4)

Represents each feature solely by its decoder vector (ell-2 normalized), and matches
features via many-to-one argmax cosine similarity as per the Gerasimov paper. For anchor 
feature i, look across ALL features in the other SAE and take the single best match; 
two different anchor features are allowed to both match the same feature in the other SAE. 
This is cheaper than Hungarian and lets us compute a per-feature stability score independently 
(Gerasimov et al. report this gives nearly identical matched sets to Hungarian as a robustness 
check: IoU = 0.978 +/- 0.001).
"""

def compute_decoder_similarity(sae1: SparseAutoencoder, sae2: SparseAutoencoder) -> torch.Tensor:
    """
    Compute cosine similarity between decoder columns of two SAEs.

    Returns:
        similarity_matrix: (n_features, n_features) matrix where entry (i, j) is
                          the cosine similarity between feature i of sae1 and feature j of sae2
    """
    # Get decoder weights (n_features, d_model)
    W1 = sae1.W_dec.detach()
    W2 = sae2.W_dec.detach()

    # ell-2 normalize decoder columns so cosine similarity reduces to a dot product
    # (source: Gerasimov et al., Section 4)
    W1_norm = F.normalize(W1, dim=1)
    W2_norm = F.normalize(W2, dim=1)

    # Compute cosine similarity matrix
    similarity = W1_norm @ W2_norm.T  # (n_features, n_features)

    return similarity.cpu()


def compute_reappearance_probability(
    saes: Dict[int, SparseAutoencoder],
    theta: float = 0.7,
) -> Tuple[np.ndarray, Dict]:
    """
    Trains implicitly assume N+1 SAEs with one anchor (seed index 0). For each
    anchor feature i, p_hat_i = (fraction of the other N SAEs containing ANY
    feature j with cos(e_i, e_j) >= theta).

    Args:
        saes: Dictionary mapping seed to trained SAE (first seed = anchor, k=0)
        theta: Minimum cosine similarity for a match to count as "reappeared"
               (paper default theta=0.7, following Leask et al. 2025)

    Returns:
        reappearance_probs: p_hat_i for each anchor feature i
        matching_info: dict with per-comparison best-match similarities and indices
    """
    seeds = list(saes.keys())
    anchor_seed = seeds[0]  # k=0 anchor, matches Gerasimov's convention
    anchor_sae = saes[anchor_seed]
    n_features = anchor_sae.n_features

    reappearance_counts = np.zeros(n_features)
    matching_info = {"similarities": [], "best_match_idx": []}

    for other_seed in seeds[1:]:
        other_sae = saes[other_seed]

        # (n_features, n_features): rows = anchor features, cols = other SAE's features
        sim_matrix = compute_decoder_similarity(anchor_sae, other_sae)

        # KEY: many-to-one argmax -- for each anchor feature (row), take its single
        # best match across ALL features in the other SAE. No 1-to-1 constraint.
        best_sim_per_anchor_feature, best_idx_per_anchor_feature = sim_matrix.max(dim=1)
        best_sim_per_anchor_feature = best_sim_per_anchor_feature.numpy()
        best_idx_per_anchor_feature = best_idx_per_anchor_feature.numpy()

        matching_info["similarities"].append(best_sim_per_anchor_feature)
        matching_info["best_match_idx"].append(best_idx_per_anchor_feature)

        # Count this comparison as a "reappearance" if the best match clears theta
        reappearance_counts += (best_sim_per_anchor_feature >= theta).astype(float)

    N = len(seeds) - 1  # number of non-anchor SAEs
    reappearance_probs = reappearance_counts / N  # p_hat_i = X_{0,i} / N (Eq. 4)

    return reappearance_probs, matching_info

# Compute reappearance probabilities
THETA = 0.7  # New matching threshold (as per Gerasimov et al)

reappearance_probs, matching_info = compute_reappearance_probability(
    trained_saes,
    theta=THETA,
)

print(f"Computed reappearance probabilities for {len(reappearance_probs)} features")
print(f"Mean reappearance probability: {reappearance_probs.mean():.3f}")
print(f"Median reappearance probability: {np.median(reappearance_probs):.3f}")

"""## 5. Group Stable and Unstable Features (endpoint binarization)"""

# Endpoint binarization (Gerasimov et al.): only label the extremes.
EPSILON = 0.05

stable_mask = reappearance_probs >= (1 - EPSILON)
unstable_mask = reappearance_probs <= EPSILON
middle_mask = ~stable_mask & ~unstable_mask  # discarded -- neither label applies

stable_indices = np.where(stable_mask)[0]
unstable_indices = np.where(unstable_mask)[0]
middle_indices = np.where(middle_mask)[0]

n_total = len(reappearance_probs)
print(f"Epsilon: {EPSILON} (stable if p_hat >= {1 - EPSILON}, unstable if p_hat <= {EPSILON})")
print(f"Stable features:    {len(stable_indices)} ({100 * len(stable_indices) / n_total:.1f}%)")
print(f"Unstable features:  {len(unstable_indices)} ({100 * len(unstable_indices) / n_total:.1f}%)")
print(f"Discarded (middle): {len(middle_indices)} ({100 * len(middle_indices) / n_total:.1f}%)")
print(f"\nUnique p_hat values: {np.unique(reappearance_probs, return_counts=True)}")

"""## 6. Visualize Results"""

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Distribution of match similarities
ax1 = axes[0]
similarities = matching_info["similarities"][0]
ax1.hist(similarities, bins=50, edgecolor="black", alpha=0.7)
ax1.axvline(THETA, color="red", linestyle="--", label=f"Threshold (theta) = {THETA}")
ax1.set_xlabel("Cosine Similarity")
ax1.set_ylabel("Count")
ax1.set_title("Distribution of Feature Match Similarities")
ax1.legend()

# Plot 2: Reappearance probability distribution, with both endpoint-binarization boundaries
ax2 = axes[1]
ax2.hist(reappearance_probs, bins=10, edgecolor="black", alpha=0.7)
ax2.axvline(EPSILON, color="red", linestyle="--", label=f"Unstable <= {EPSILON}")
ax2.axvline(1 - EPSILON, color="green", linestyle="--", label=f"Stable >= {1 - EPSILON}")
ax2.set_xlabel("Reappearance Probability")
ax2.set_ylabel("Count")
ax2.set_title("Distribution of Reappearance Probabilities")
ax2.legend()

# Plot 3: Pie chart of stable vs unstable vs discarded
ax3 = axes[2]
sizes = [len(stable_indices), len(unstable_indices), len(middle_indices)]
labels = [f"Stable\n({sizes[0]})", f"Unstable\n({sizes[1]})", f"Discarded\n({sizes[2]})"]
colors = ["#2ca02c", "#ff7f0e", "#7f7f7f"]
ax3.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
ax3.set_title("Feature Stability Distribution")

plt.tight_layout()
plt.show()

# Create a summary dataframe
import pandas as pd

feature_df = pd.DataFrame({
    "feature_idx": np.arange(len(reappearance_probs)),
    "reappearance_prob": reappearance_probs,
    "match_similarity": matching_info["similarities"][0],
    "is_stable": stable_mask,
    "is_unstable": unstable_mask,
    "is_discarded": middle_mask,
})

print("Feature stability summary:")
print(feature_df.describe())

print("\n\nSample of stable features:")
print(feature_df[feature_df["is_stable"]].head(10))

print("\n\nSample of unstable features:")
print(feature_df[feature_df["is_unstable"]].head(10))

"""## 7. Save Results (Optional)"""

# Save feature stability data
feature_df.to_csv(OUTPUT_DIR / "feature_stability.csv", index=False)
print(f"Saved feature stability data to {OUTPUT_DIR / 'feature_stability.csv'}")

# Save SAE checkpoints
for seed, sae in trained_saes.items():
    torch.save(sae.state_dict(), OUTPUT_DIR / f"sae_seed_{seed}.pt")
    print(f"Saved SAE checkpoint to {OUTPUT_DIR / f'sae_seed_{seed}.pt'}")

# Save matching info
np.savez(
    OUTPUT_DIR / "matching_info.npz",
    similarities=np.array(matching_info["similarities"]),
    reappearance_probs=reappearance_probs,
    stable_indices=stable_indices,
    unstable_indices=unstable_indices,
)
print(f"Saved matching info to {OUTPUT_DIR / 'matching_info.npz'}")

# Save the run config so a resumed/re-analyzed run is self-describing
with open(OUTPUT_DIR / "config.json", "w") as fh:
    json.dump({**CONFIG, "checkpoint_tokens": CHECKPOINT_TOKENS}, fh, indent=2)
print(f"Saved run config to {OUTPUT_DIR / 'config.json'}")

"""## 8. Analyze Stable Features"""

# Compare statistics between stable and unstable features
reference_sae = trained_saes[CONFIG["seeds"][0]]

# Get decoder norms
decoder_norms = reference_sae.W_dec.detach().cpu().norm(dim=1).numpy()

# Build a fixed evaluation set from a fresh stream, since training no longer produces
# one big pre-collected `activations` tensor. This re-reads from the start of the same
# streaming dataset.
eval_stream = activation_stream_generator(
    model=model,
    dataset_name="monology/pile-uncopyrighted",
    hook_point=CONFIG["hook_point"],
    seq_len=CONFIG["seq_len"],
    batch_size=CONFIG["batch_size"],
    device=CONFIG["device"],
)
N_EVAL_BATCHES = 40  # ~40 * batch_size * seq_len tokens worth of eval data; adjust as needed
activations = torch.cat([next(eval_stream) for _ in range(N_EVAL_BATCHES)], dim=0)
print(f"Eval activations shape: {activations.shape}")

# batched accumulation, peak GPU usage ~32 MB
STAT_BATCH = 4096  # reduce to 1024 if still OOM


def compute_activation_stats(sae, activations, device, batch_size=STAT_BATCH, desc="feature stats"):
    """Firing rate and firing strength for every feature of a single SAE.

    Returns mean activation conditioned on the feature firing. The unconditional mean --
    the sum of activations divided by ALL tokens -- is identically
    (firing rate) x (conditional mean), so using it as a predictor alongside activation
    frequency would double-count frequency rather than contribute anything new.
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
    mean_activation = np.divide(
        sum_accum.numpy(), firing_counts,
        out=np.zeros(sae.n_features), where=firing_counts > 0,
    )
    return activation_freq, mean_activation


activation_freq, mean_activation = compute_activation_stats(
    reference_sae, activations, CONFIG["device"], desc="reference-seed feature stats"
)

# Whether this eval set is large enough is decided by the rarest features, not the average
# one. At L0 ~346/2048 the typical feature fires on ~17% of tokens and is pinned down to
# several significant figures here; only a substantial low-count tail would justify the
# streaming rewrite that reaching the intended 100M tokens requires (the eval set is
# materialized as one tensor, so 100M tokens would be ~205 GB).
_firing_counts = activation_freq * len(activations)
print(f"\n=== Is {len(activations):,} eval tokens enough? (frequency estimate quality) ===")
for _thresh in (10, 100, 1000):
    _n = int((_firing_counts < _thresh).sum())
    print(f"  features firing < {_thresh:5,} times: {_n:5d} "
          f"({_n / len(activation_freq):5.1%}) -- relative error >~ {100 / max(_thresh, 1) ** 0.5:.0f}%")
print(f"  median firings per feature: {np.median(_firing_counts):,.0f}")

# Compare stable vs unstable
print("=== Stable Features (n={}) ===".format(len(stable_indices)))
print(f"  Decoder norm:      mean={decoder_norms[stable_indices].mean():.3f}, std={decoder_norms[stable_indices].std():.3f}")
print(f"  Activation freq:   mean={activation_freq[stable_indices].mean():.4f}, std={activation_freq[stable_indices].std():.4f}")
print(f"  Mean act (|firing): mean={mean_activation[stable_indices].mean():.4f}, std={mean_activation[stable_indices].std():.4f}")

print("\n=== Unstable Features (n={}) ===".format(len(unstable_indices)))
print(f"  Decoder norm:      mean={decoder_norms[unstable_indices].mean():.3f}, std={decoder_norms[unstable_indices].std():.3f}")
print(f"  Activation freq:   mean={activation_freq[unstable_indices].mean():.4f}, std={activation_freq[unstable_indices].std():.4f}")
print(f"  Mean act (|firing): mean={mean_activation[unstable_indices].mean():.4f}, std={mean_activation[unstable_indices].std():.4f}")

def safe_hist(ax, values, bins, **kwargs):
    """hist() that tolerates effectively-constant data.

    normalize_decoder() pins every decoder norm to 1.0, so the decoder-norm histogram has no
    usable range and matplotlib raises "Too many bins for data range". That would abort the
    script after training has already finished, so degrade to a single bar instead of losing
    the whole analysis to a plotting detail.

    The test is relative, not a fixed epsilon. Renormalizing every step leaves norms that
    differ in the last few bits rather than being bit-identical: a span around 1e-7 at values
    around 1.0, which clears any small absolute threshold while still being far too narrow to
    cut into 30 distinct float32 bin edges. An absolute cutoff therefore passes runs where the
    norms happen to land exactly equal and crashes ones where they do not, which is a
    coin-flip on the arithmetic rather than a property of the data.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    lo, hi = float(values.min()), float(values.max())
    # Scaled by the data's own magnitude with no floor, so a predictor that genuinely varies
    # over a narrow range near zero still gets binned properly instead of being flattened
    # along with the decoder norms.
    scale = max(abs(lo), abs(hi))
    # Below this the bars are indistinguishable on screen anyway, so one bar is the honest
    # rendering as well as the numerically safe one.
    if hi - lo <= 1e-6 * scale:
        ax.hist(values, bins=1, range=(lo - 0.5, hi + 0.5), **kwargs)
    else:
        ax.hist(values, bins=bins, **kwargs)


# Visualize: stable vs unstable feature properties
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Decoder norms
ax1 = axes[0]
safe_hist(ax1, decoder_norms[unstable_indices], 30, alpha=0.6, label="Unstable", color="#ff7f0e")
safe_hist(ax1, decoder_norms[stable_indices], 10, alpha=0.8, label="Stable", color="#2ca02c")
ax1.set_xlabel("Decoder Norm")
ax1.set_ylabel("Count")
ax1.set_title("Decoder Norms: Stable vs Unstable")
ax1.legend()

# Plot 2: Activation frequency
ax2 = axes[1]
safe_hist(ax2, activation_freq[unstable_indices], 30, alpha=0.6, label="Unstable", color="#ff7f0e")
safe_hist(ax2, activation_freq[stable_indices], 10, alpha=0.8, label="Stable", color="#2ca02c")
ax2.set_xlabel("Activation Frequency")
ax2.set_ylabel("Count")
ax2.set_title("Activation Frequency: Stable vs Unstable")
ax2.legend()

# Plot 3: Scatter - frequency vs decoder norm, colored by stability
ax3 = axes[2]
ax3.scatter(activation_freq[unstable_indices], decoder_norms[unstable_indices],
            alpha=0.3, label="Unstable", color="#ff7f0e", s=10)
ax3.scatter(activation_freq[stable_indices], decoder_norms[stable_indices],
            alpha=0.9, label="Stable", color="#2ca02c", s=50, edgecolor="black")
ax3.set_xlabel("Activation Frequency")
ax3.set_ylabel("Decoder Norm")
ax3.set_title("Feature Properties (Stable = Green)")
ax3.legend()

plt.tight_layout()
plt.show()

# Find top activating tokens for each stable feature
# We need to collect some text data with token information

def get_top_activating_examples(
    model,
    sae,
    dataset,
    feature_indices,
    n_examples=5,
    n_tokens_to_scan=100_000,
    context_window=10
):
    """Find text examples that most strongly activate each feature."""

    results = {idx: [] for idx in feature_indices}
    token_buffer = []
    text_buffer = []

    print(f"Scanning {n_tokens_to_scan:,} tokens for top activating examples...")

    # Collect tokens with their source text
    tokens_scanned = 0
    for example in tqdm(dataset, total=n_tokens_to_scan // 50):
        text = example["text"]
        tokens = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0]

        if len(tokens) < 10:
            continue

        # Get activations for this sequence
        with torch.no_grad():
            layer_idx = int(CONFIG["hook_point"].split(".")[1])
            _, cache = model.run_with_cache(
                tokens.unsqueeze(0).to(CONFIG["device"]),
                names_filter=[CONFIG["hook_point"]],
                stop_at_layer=layer_idx + 1,
            )
            acts = cache[CONFIG["hook_point"]][0]  # (seq_len, d_model)
            features = sae.encode(acts)  # (seq_len, n_features)

        # Check each feature of interest
        for feat_idx in feature_indices:
            feat_acts = features[:, feat_idx].cpu().numpy()
            max_pos = feat_acts.argmax()
            max_val = feat_acts[max_pos]

            if max_val > 0:
                # Get context around the max activation
                start = max(0, max_pos - context_window)
                end = min(len(tokens), max_pos + context_window + 1)
                context_tokens = tokens[start:end]
                context_text = model.tokenizer.decode(context_tokens)
                target_token = model.tokenizer.decode(tokens[max_pos])

                results[feat_idx].append({
                    "activation": max_val,
                    "context": context_text,
                    "token": target_token,
                    "position": max_pos,
                })

        tokens_scanned += len(tokens)
        if tokens_scanned >= n_tokens_to_scan:
            break

    # Sort by activation strength and keep top n
    for feat_idx in feature_indices:
        results[feat_idx] = sorted(results[feat_idx], key=lambda x: -x["activation"])[:n_examples]

    return results

# Get top activating examples for stable features
top_examples_dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

top_examples = get_top_activating_examples(
    model,
    reference_sae,
    top_examples_dataset,
    stable_indices,
    n_examples=5,
    n_tokens_to_scan=50_000,  # Reduced for speed
)

# Display top activating examples for each stable feature
print("=" * 80)
print("TOP ACTIVATING EXAMPLES FOR STABLE FEATURES")
print("=" * 80)

for feat_idx in stable_indices:
    examples = top_examples[feat_idx]
    match_sim = matching_info["similarities"][0][feat_idx]

    print(f"\n{'='*80}")
    print(f"FEATURE {feat_idx} (match similarity: {match_sim:.3f})")
    print(f"  Activation freq: {activation_freq[feat_idx]:.4f}")
    print(f"  Decoder norm: {decoder_norms[feat_idx]:.3f}")
    print("-" * 80)

    if not examples:
        print("  No activating examples found")
        continue

    for i, ex in enumerate(examples[:3]):  # Show top 3
        print(f"\n  Example {i+1} (activation: {ex['activation']:.2f}):")
        print(f"    Token: '{ex['token']}'")
        print(f"    Context: ...{ex['context']}...")

print("\n" + "=" * 80)

# PCA visualization of decoder vectors
from sklearn.decomposition import PCA

# Get decoder vectors
decoder_vectors = reference_sae.W_dec.detach().cpu().numpy()

# Fit PCA
pca = PCA(n_components=2)
decoder_2d = pca.fit_transform(decoder_vectors)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot unstable features
ax.scatter(decoder_2d[unstable_indices, 0], decoder_2d[unstable_indices, 1],
           alpha=0.3, label="Unstable", color="#ff7f0e", s=15)

# Plot stable features (larger, with labels)
ax.scatter(decoder_2d[stable_indices, 0], decoder_2d[stable_indices, 1],
           alpha=1.0, label="Stable", color="#2ca02c", s=100, edgecolor="black", linewidth=1.5)

# Label stable features
for idx in stable_indices:
    ax.annotate(str(idx), (decoder_2d[idx, 0], decoder_2d[idx, 1]),
                fontsize=8, ha="center", va="bottom")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("PCA of Decoder Vectors (Stable Features in Green)")
ax.legend()
plt.tight_layout()
plt.show()

"""## 9. Compute All Four Stability Predictors"""

def compute_geometric_isolation(sae: SparseAutoencoder, k: int = 10) -> np.ndarray:
    """
    Compute geometric isolation for each feature.

    Geometric isolation = average cosine similarity to K nearest neighbors.
    LOW value = isolated, unique direction (more likely stable)
    HIGH value = crowded region, rotational freedom (less stable)

    Returns:
        isolation_scores: (n_features,) array of average NN similarities
    """
    W_dec = sae.W_dec.detach()
    W_norm = F.normalize(W_dec, dim=1)

    # Compute all pairwise similarities
    similarity_matrix = (W_norm @ W_norm.T).cpu().numpy()

    # For each feature, find average similarity to K nearest neighbors
    # (excluding self, which has similarity 1.0)
    n_features = similarity_matrix.shape[0]
    isolation_scores = np.zeros(n_features)

    for i in range(n_features):
        # Get similarities to all other features
        sims = similarity_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self

        # Get top K neighbors
        top_k_indices = np.argsort(sims)[-k:]
        top_k_sims = sims[top_k_indices]

        # Average similarity to nearest neighbors
        isolation_scores[i] = top_k_sims.mean()

    return isolation_scores


def compute_reconstruction_contribution(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    batch_size: int = 8192,
    n_samples: int = 10000,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reconstruction contribution for each feature.

    Measures how much reconstruction error increases when each feature is ablated (zeroed out).
    HIGH value = feature is important for reconstruction (more likely stable)
    LOW value = feature is redundant (less stable)

    Ablating a feature perturbs only the tokens where it was active, so averaging the MSE
    increase over ALL sampled tokens gives (firing rate) x (impact while firing). That is
    mechanically proportional to activation frequency, which is separately one of our
    predictors -- so the unconditional form cannot be used to argue that reconstruction
    contribution adds anything beyond frequency. The conditional mean, taken over firing
    tokens only, is the part that is not already frequency. Both are returned so the size of
    the overlap can be reported rather than assumed.

    Returns:
        conditional:   (n_features,) mean MSE increase over tokens where the feature fired
        unconditional: (n_features,) mean MSE increase over all sampled tokens
    """
    sae.eval()
    n_features = sae.n_features
    d_model = sae.d_model

    # Use a subset of activations for speed
    sample_indices = np.random.choice(len(activations), min(n_samples, len(activations)), replace=False)
    sample_acts = activations[sample_indices]

    print(f"Computing reconstruction contributions for {n_features} features...")

    delta_sums = torch.zeros(n_features, dtype=torch.float64, device=device)
    active_counts = torch.zeros(n_features, dtype=torch.float64, device=device)
    n_total = len(sample_acts)

    with torch.no_grad():
        dec_sq_norms = sae.W_dec.pow(2).sum(dim=1)  # (n_features,), 1.0 under normalize_decoder

        # Zeroing feature j shifts the residual by exactly -f_j * W_dec[j], so the change in
        # per-token MSE has a closed form and every feature can be done in one matmul. This
        # is identical to ablating features one at a time, without the 2048 decode passes.
        for start in tqdm(range(0, n_total, batch_size), desc="Ablation (closed form)"):
            x = sample_acts[start : start + batch_size].to(device)
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
        delta_sums, active_counts, out=np.zeros_like(delta_sums), where=active_counts > 0
    )
    return conditional, unconditional

def compute_encoder_stats(sae):
    """Encoder-side single-run statistics.

    Everything else here describes the decoder or the activations it produces, but the
    encoder is what actually decides whether a feature fires: b_enc is literally the
    activation threshold, and the encoder column norm sets how sharply the feature responds.
    Both are free to read off the weights and neither is constrained by normalize_decoder.
    """
    enc = sae.W_enc.detach()  # (d_model, n_features)
    return enc.norm(dim=0).cpu().numpy(), sae.b_enc.detach().cpu().numpy()


def compute_single_run_statistics(sae, activations, device, k=10, n_samples=10000, label=""):
    """Every predictor available from ONE SAE, with no reference to any other seed.

    Returned as a dict so the same code path can produce the reference seed's statistics and
    a held-out seed's, guaranteeing the two are computed identically.
    """
    suffix = f" ({label})" if label else ""
    print(f"Computing geometric isolation{suffix}...")
    isolation = compute_geometric_isolation(sae, k=k)

    print(f"Computing activation statistics{suffix}...")
    freq, mean_act = compute_activation_stats(
        sae, activations, device, desc=f"activation stats{suffix}"
    )

    print(f"Computing reconstruction contribution{suffix}...")
    recon_cond, recon_uncond = compute_reconstruction_contribution(
        sae, activations, n_samples=n_samples, device=device
    )

    enc_norm, enc_bias = compute_encoder_stats(sae)

    return {
        "activation_freq": freq,
        "mean_activation": mean_act,
        "geometric_isolation": isolation,
        "recon_contribution": recon_cond,
        "recon_contribution_uncond": recon_uncond,
        "encoder_norm": enc_norm,
        "encoder_bias": enc_bias,
        "decoder_norm": sae.W_dec.detach().cpu().norm(dim=1).numpy(),
    }


print("Computing geometric isolation...")
geometric_isolation = compute_geometric_isolation(reference_sae, k=10)

print("\nComputing reconstruction contribution...")
recon_contribution, recon_contribution_uncond = compute_reconstruction_contribution(
    reference_sae,
    activations,
    n_samples=10000,
    device=CONFIG["device"]
)

encoder_norm, encoder_bias = compute_encoder_stats(reference_sae)

# How much of each statistic is really just activation frequency wearing a different hat.
# The unconditional ablation is (firing rate) x (impact while firing) by construction, so a
# near-1.0 correlation there is expected and is the reason the conditional form is used
# instead; anything else near 1.0 would mean that predictor is not independent evidence.
print("\n=== Frequency confound check (Spearman rho vs activation frequency) ===")
from scipy.stats import spearmanr

for _name, _values in [
    ("recon contribution (conditional, used)", recon_contribution),
    ("recon contribution (unconditional)", recon_contribution_uncond),
    ("mean activation | firing", mean_activation),
    ("geometric isolation", geometric_isolation),
    ("encoder column norm", encoder_norm),
    ("encoder bias", encoder_bias),
]:
    _rho = spearmanr(activation_freq, _values).statistic
    print(f"  {_name:42s}: rho={_rho:+.3f}")

print("\nDone! All predictors computed.")

# Compare all four statistics between stable and unstable features
print("=" * 60)
print("COMPARISON OF ALL FOUR STABILITY PREDICTORS")
print("=" * 60)

print("\n=== Stable Features (n={}) ===".format(len(stable_indices)))
print(f"  1. Activation freq:        mean={activation_freq[stable_indices].mean():.4f}, std={activation_freq[stable_indices].std():.4f}")
print(f"  2. Decoder norm:           mean={decoder_norms[stable_indices].mean():.4f}, std={decoder_norms[stable_indices].std():.4f}")
print(f"  3. Geometric isolation:    mean={geometric_isolation[stable_indices].mean():.4f}, std={geometric_isolation[stable_indices].std():.4f}")
print(f"  4. Recon contribution:     mean={recon_contribution[stable_indices].mean():.6f}, std={recon_contribution[stable_indices].std():.6f}")

print("\n=== Unstable Features (n={}) ===".format(len(unstable_indices)))
print(f"  1. Activation freq:        mean={activation_freq[unstable_indices].mean():.4f}, std={activation_freq[unstable_indices].std():.4f}")
print(f"  2. Decoder norm:           mean={decoder_norms[unstable_indices].mean():.4f}, std={decoder_norms[unstable_indices].std():.4f}")
print(f"  3. Geometric isolation:    mean={geometric_isolation[unstable_indices].mean():.4f}, std={geometric_isolation[unstable_indices].std():.4f}")
print(f"  4. Recon contribution:     mean={recon_contribution[unstable_indices].mean():.6f}, std={recon_contribution[unstable_indices].std():.6f}")

# Compute effect sizes (difference in means / pooled std)
print("\n=== Effect Sizes (Cohen's d) ===")
for name, values in [
    ("Activation freq", activation_freq),
    ("Decoder norm", decoder_norms),
    ("Geometric isolation", geometric_isolation),
    ("Recon contribution", recon_contribution),
]:
    stable_vals = values[stable_indices]
    unstable_vals = values[unstable_indices]

    # Cohen's d
    pooled_std = np.sqrt((stable_vals.std()**2 + unstable_vals.std()**2) / 2)
    if pooled_std > 0:
        cohens_d = (stable_vals.mean() - unstable_vals.mean()) / pooled_std
    else:
        cohens_d = 0

    print(f"  {name}: d = {cohens_d:.3f}")

# Visualize all four statistics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

stats = [
    ("Activation Frequency", activation_freq, "Higher = fires more often"),
    ("Geometric Isolation (NN Sim)", geometric_isolation, "Lower = more isolated"),
    ("Reconstruction Contribution", recon_contribution, "Higher = more important"),
    ("Mean Activation Strength", mean_activation, "Higher = fires stronger"),
]

for ax, (name, values, description) in zip(axes.flat, stats):
    safe_hist(ax, values[unstable_indices], 30, alpha=0.6, label="Unstable", color="#ff7f0e", density=True)
    safe_hist(ax, values[stable_indices], 10, alpha=0.8, label="Stable", color="#2ca02c", density=True)
    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    ax.set_title(f"{name}\n({description})")
    ax.legend()

plt.tight_layout()
plt.show()

"""## 10. Train Stability Classifier"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Only train on features with a definite label (exclude the discarded middle)
labeled_mask = stable_mask | unstable_mask

# Firing rates and activation magnitudes are heavy-tailed over several orders of magnitude,
# and logistic regression fits a boundary that is linear in whatever it is handed. Left in
# raw units, the multivariable model can recruit the other predictors purely to bend the
# frequency response, which would show up as those predictors "adding signal" when they are
# only supplying curvature. Note this does NOT change any single-predictor AUROC: one
# logistic regression coefficient is monotone in its input, AUROC depends only on ranking,
# and a log is monotone. It only makes the multivariable comparison honest.
LOG_EPS = 1e-10


def build_predictors(stats):
    """(name, values) for every predictor, from one SAE's statistics dict.

    Single code path so the reference seed and any held-out seed are guaranteed to be
    described by identically constructed columns in identical order.
    """
    return [
        ("Activation Freq (log)", np.log10(stats["activation_freq"] + LOG_EPS)),
        ("Geometric Isolation", stats["geometric_isolation"]),
        ("Recon Contribution", stats["recon_contribution"]),
        ("Mean Activation (log)", np.log10(stats["mean_activation"] + LOG_EPS)),
        ("Encoder Norm", stats["encoder_norm"]),
        ("Encoder Bias", stats["encoder_bias"]),
    ]


reference_stats = {
    "activation_freq": activation_freq,
    "mean_activation": mean_activation,
    "geometric_isolation": geometric_isolation,
    "recon_contribution": recon_contribution,
    "encoder_norm": encoder_norm,
    "encoder_bias": encoder_bias,
}

PREDICTORS = build_predictors(reference_stats)
feature_names = [name for name, _ in PREDICTORS]
log_activation_freq = dict(PREDICTORS)["Activation Freq (log)"]

X = np.column_stack([values for _, values in PREDICTORS])[labeled_mask]

y = stable_mask[labeled_mask].astype(int)

print(f"Features excluded from classifier training (discarded middle): {middle_mask.sum()}")
print(f"Features used for classifier training: {labeled_mask.sum()}")

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {y.sum()} stable, {len(y) - y.sum()} unstable")
print(f"Class imbalance ratio: 1:{(len(y) - y.sum()) / max(y.sum(), 1):.1f}")

# Train and evaluate classifier with cross-validation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic regression with class weighting to handle imbalance
clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compute AUROC scores
auroc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='roc_auc')

print("=" * 60)
print("LOGISTIC REGRESSION CLASSIFIER RESULTS")
print("=" * 60)
print(f"\nCross-validated AUROC: {auroc_scores.mean():.3f} (+/- {auroc_scores.std() * 2:.3f})")
print(f"Individual fold scores: {[f'{s:.3f}' for s in auroc_scores]}")

# Fit on all data to get coefficients
clf.fit(X_scaled, y)

print("\n=== Feature Importance (Logistic Regression Coefficients) ===")
for name, coef in sorted(zip(feature_names, clf.coef_[0]), key=lambda x: -abs(x[1])):
    direction = "up stable" if coef > 0 else "down stable"
    print(f"  {name:25s}: {coef:+.3f} ({direction})")

# Compare with baselines
print("=" * 60)
print("BASELINE COMPARISONS")
print("=" * 60)

# Baseline 1: Random classifier
print("\n1. Random Classifier:")
print(f"   Expected AUROC: 0.500")

# Single-feature classifiers
print("\n" + "-" * 60)
print("SINGLE-FEATURE CLASSIFIERS (Isolated Effects)")
print("-" * 60)

single_feature_results = {}
for name, values in PREDICTORS:
    values_labeled = values[labeled_mask]
    X_single = values_labeled.reshape(-1, 1)
    X_single_scaled = StandardScaler().fit_transform(X_single)
    clf_single = LogisticRegression(class_weight='balanced', random_state=42)
    single_auroc = cross_val_score(clf_single, X_single_scaled, y, cv=cv, scoring='roc_auc')
    single_feature_results[name] = single_auroc.mean()
    print(f"\n   {name}:")
    print(f"      AUROC: {single_auroc.mean():.3f} (+/- {single_auroc.std() * 2:.3f})")

# Rank single features
print("\n   Ranking (best to worst):")
for rank, (name, auroc) in enumerate(sorted(single_feature_results.items(), key=lambda x: -x[1]), 1):
    print(f"      {rank}. {name}: {auroc:.3f}")

# Combined classifiers
print("\n" + "-" * 60)
print("COMBINED CLASSIFIERS")
print("-" * 60)

# Frequency + Geometric isolation
X_freq_geom = np.column_stack([log_activation_freq, geometric_isolation])[labeled_mask]
X_fg_scaled = StandardScaler().fit_transform(X_freq_geom)
clf_fg = LogisticRegression(class_weight='balanced', random_state=42)
fg_auroc = cross_val_score(clf_fg, X_fg_scaled, y, cv=cv, scoring='roc_auc')
print(f"\n   Frequency + Geometry:")
print(f"      AUROC: {fg_auroc.mean():.3f} (+/- {fg_auroc.std() * 2:.3f})")

# All except one (ablation study)
print("\n   Ablation Study (Full Model minus one feature):")
for i, name in enumerate(feature_names):
    X_ablated = np.delete(X, i, axis=1)
    X_ablated_scaled = StandardScaler().fit_transform(X_ablated)
    clf_ablated = LogisticRegression(class_weight='balanced', random_state=42)
    ablated_auroc = cross_val_score(clf_ablated, X_ablated_scaled, y, cv=cv, scoring='roc_auc')
    drop = auroc_scores.mean() - ablated_auroc.mean()
    print(f"      Without {name}: {ablated_auroc.mean():.3f} (drop: {drop:+.3f})")

# Full model
print(f"\n   Full Model ({len(feature_names)} predictors):")
print(f"      AUROC: {auroc_scores.mean():.3f} (+/- {auroc_scores.std() * 2:.3f})")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
best_single = max(single_feature_results.items(), key=lambda x: x[1])
print(f"Best single feature: {best_single[0]} ({best_single[1]:.3f})")
print(f"Full model: {auroc_scores.mean():.3f}")
print(f"Improvement from combining: +{auroc_scores.mean() - best_single[1]:.3f} AUROC")

if auroc_scores.mean() >= 0.75:
    print("\nHypothesis 1 SUPPORTED: AUROC >= 0.75")
else:
    print(f"\nHypothesis 1 NOT YET SUPPORTED: AUROC = {auroc_scores.mean():.3f} < 0.75")

"""## 11. Does the headline number survive its own methodology choices?

Three checks, each aimed at a way the AUROC above could be an artifact of a labelling
decision rather than a property of the SAE:

  1. Endpoint binarization discards the ambiguous middle, which removes the hardest cases
     before scoring. Re-run over ALL features under p_hat >= 0.5 to size that effect.
  2. Many-to-one matching lets several anchor features claim one partner, so a crowded
     region could read as stable. Re-derive the labels with one-to-one Hungarian matching
     and see whether geometric isolation's predictive power moves with them.
  3. Cross-validation holds out features from the SAME dictionary, and geometric isolation
     is relational -- a feature's value depends on neighbours that may sit in the training
     fold. Train on the reference seed and test on a different seed's dictionary entirely.
"""

print("\n" + "=" * 60)
print("ROBUSTNESS OF THE EVALUATION ITSELF")
print("=" * 60)


def cv_auroc(values, labels, mask):
    """Cross-validated AUROC under the shared protocol, for any predictor set and labelling."""
    cols = values if values.ndim == 2 else values.reshape(-1, 1)
    X_local = StandardScaler().fit_transform(cols[mask])
    y_local = labels[mask].astype(int)
    if len(np.unique(y_local)) < 2:
        return float("nan")
    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    return cross_val_score(model, X_local, y_local, cv=cv, scoring="roc_auc").mean()


X_all = np.column_stack([values for _, values in PREDICTORS])

# --- 1. how much does discarding the ambiguous middle flatter the result? ---
print("\n1. Effect of discarding the ambiguous middle")
all_mask = np.ones(len(reappearance_probs), dtype=bool)
midpoint_labels = reappearance_probs >= 0.5
endpoint_auroc = cv_auroc(X_all, stable_mask, labeled_mask)
midpoint_auroc = cv_auroc(X_all, midpoint_labels, all_mask)
print(f"   endpoint only  (n={labeled_mask.sum():4d}, the reported figure): {endpoint_auroc:.3f}")
print(f"   all features   (n={all_mask.sum():4d}, p_hat >= 0.5)           : {midpoint_auroc:.3f}")
print(f"   inflation attributable to discarding {middle_mask.sum()} features: "
      f"{endpoint_auroc - midpoint_auroc:+.3f}")

# --- 2. do the labels, and isolation's power over them, depend on the matching rule? ---
print("\n2. Effect of the matching rule (many-to-one vs one-to-one)")
from scipy.optimize import linear_sum_assignment

_seeds = list(trained_saes.keys())
_anchor_sae = trained_saes[_seeds[0]]
_hungarian_counts = np.zeros(_anchor_sae.n_features)
for _other in _seeds[1:]:
    _sim = compute_decoder_similarity(_anchor_sae, trained_saes[_other]).numpy()
    _rows, _cols = linear_sum_assignment(-_sim)
    _assigned = np.full(_anchor_sae.n_features, -1.0)
    _assigned[_rows] = _sim[_rows, _cols]
    _hungarian_counts += (_assigned >= THETA).astype(float)

hungarian_probs = _hungarian_counts / max(len(_seeds) - 1, 1)
hungarian_stable = hungarian_probs >= (1 - EPSILON)
hungarian_labeled = hungarian_stable | (hungarian_probs <= EPSILON)

print(f"   stable fraction  many-to-one: {stable_mask.mean():6.1%}   "
      f"one-to-one: {hungarian_stable.mean():6.1%}")
print(f"   labels changed by switching rule: "
      f"{int((stable_mask != hungarian_stable).sum())} of {len(stable_mask)} features")
_iso = geometric_isolation
print(f"   geometric isolation alone, many-to-one labels: "
      f"{cv_auroc(_iso, stable_mask, labeled_mask):.3f}")
print(f"   geometric isolation alone, one-to-one labels : "
      f"{cv_auroc(_iso, hungarian_stable, hungarian_labeled):.3f}")
print(f"   full model,                one-to-one labels : "
      f"{cv_auroc(X_all, hungarian_stable, hungarian_labeled):.3f}")
print("   (a large gap here would mean isolation is tracking the matcher, not stability)")

# --- 3. does the classifier transfer to a dictionary it was not trained on? ---
print("\n3. Transfer to a held-out dictionary (the deployment claim)")
if len(_seeds) < 3:
    print("   Needs >=3 seeds: the held-out seed must itself have two comparisons. Skipped.")
else:
    held_out_seed = _seeds[1]
    # Same budget, same eval activations, same code path -- only the dictionary differs, so a
    # drop here is transfer failure and not a difference in training budget or in measurement.
    held_out_saes = {held_out_seed: trained_saes[held_out_seed]}
    held_out_saes.update({s: trained_saes[s] for s in _seeds if s != held_out_seed})
    held_out_probs, _ = compute_reappearance_probability(held_out_saes, theta=THETA)
    held_out_stable = held_out_probs >= (1 - EPSILON)
    held_out_mask = held_out_stable | (held_out_probs <= EPSILON)

    held_out_stats = compute_single_run_statistics(
        trained_saes[held_out_seed], activations, CONFIG["device"],
        label=f"seed {held_out_seed}",
    )
    X_held = np.column_stack([v for _, v in build_predictors(held_out_stats)])

    transfer_clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    transfer_scaler = StandardScaler().fit(X_all[labeled_mask])
    transfer_clf.fit(transfer_scaler.transform(X_all[labeled_mask]), stable_mask[labeled_mask])

    held_truth = held_out_stable[held_out_mask].astype(int)

    # Two ways to normalize the held-out dictionary, which answer different questions.
    # Reusing the training scaler also requires the raw scales to agree across dictionaries;
    # refitting on the held-out SAE asks only whether the learned relationship transfers,
    # and matches what a practitioner would do with an SAE in hand.
    transfer_auroc = roc_auc_score(held_truth, transfer_clf.predict_proba(
        transfer_scaler.transform(X_held[held_out_mask]))[:, 1])
    refit_auroc = roc_auc_score(held_truth, transfer_clf.predict_proba(
        StandardScaler().fit_transform(X_held)[held_out_mask])[:, 1])

    print(f"   trained on seed {_seeds[0]}, tested on seed {held_out_seed} "
          f"(n={held_out_mask.sum()}, {held_truth.sum()} stable)")
    print(f"   within-dictionary (cross-validated)     : {endpoint_auroc:.3f}")
    print(f"   held-out, training-set scaler           : {transfer_auroc:.3f}")
    print(f"   held-out, rescaled on the held-out SAE  : {refit_auroc:.3f}")
    print(f"   transfer cost (rescaled)                : {refit_auroc - endpoint_auroc:+.3f}")
    print("   (the held-out number is the one that supports 'use this on an SAE you did "
          "not train')")
