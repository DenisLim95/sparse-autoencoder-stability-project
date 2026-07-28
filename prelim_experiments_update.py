# -*- coding: utf-8 -*-
"""prelim-experiments.ipynb

SAE stability project — preliminary experiments.
"""

# Install dependencies (run this cell first in Colab)
# pip install transformer_lens sae_lens datasets torch

import json
import os
import re
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
    "l1_coeff": 1.0,  # Sparsity coefficient
    "lr": 1e-3,
    "batch_size": 256,
    "n_tokens": 10_000_000,  # kept for reference but now set later on
    "seq_len": 128,
    "seeds": [42, 137, 256, 512, 1024],  # Five different random seeds
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"Using device: {CONFIG['device']}")

# Token budgets at which an SAE checkpoint is saved, so stability can be compared
# across training scale.
CHECKPOINT_TOKENS = [1_000_000_000, 5_000_000_000, 8_000_000_000]

# Consecutive entries in CHECKPOINT_TOKENS are up to 4B tokens apart -- far more than a
# single Colab session can cover. Without a rolling checkpoint in between, a disconnect
# part-way through a gap rewinds to the previous milestone, so a run can bounce off the
# same gap forever without advancing. This bounds the loss from a disconnect instead.
CHECKPOINT_EVERY_SECONDS = 900

# How often (in optimizer steps) to record a point on the training curves.
LOG_EVERY_STEPS = 200

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
RUN_NAME = f"{CONFIG['model_name']}_L{_layer}_{max(CHECKPOINT_TOKENS) // 1_000_000_000}Btok"
OUTPUT_DIR = Path(RESULTS_BASE) / RUN_NAME
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Set to a Hub repo you have write access to in order to mirror checkpoints remotely.
# Pointing this at someone else's repo is worse than leaving it None: the upload fails on
# auth, the failure is only warned about, and training continues persisting nothing.
HF_REPO_ID = None

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
    Standard Sparse Autoencoder with ReLU activation and L1 sparsity penalty.

    Architecture:
        encoder: x -> ReLU(W_enc @ (x - b_dec) + b_enc)
        decoder: f -> W_dec @ f + b_dec
    """

    def __init__(self, d_model: int, n_features: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)

        self.d_model = d_model
        self.n_features = n_features

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

        # Sparsity loss (L1 on feature activations)
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


def load_resume_state(checkpoint_dir: Path, seed: int):
    """Return whichever checkpoint for this seed is furthest along: the rolling `_latest`
    file or the newest milestone. Milestones are sorted by their ACTUAL token count, not
    alphabetically -- "...tokens5000000000.pt" would otherwise sort after
    "...tokens1000000000.pt" since '5' > '1', even though 5B > 1B."""
    candidates = [checkpoint_dir / f"seed{seed}_latest.pt"]
    candidates += sorted(
        checkpoint_dir.glob(f"seed{seed}_tokens*.pt"),
        key=lambda p: int(re.search(r"tokens(\d+)\.pt", p.name).group(1)),
    )[-1:]

    best = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            # Explicit weights_only=False: these are our own checkpoints and they carry
            # optimizer/config/history alongside the weights. Newer torch flips the default
            # to True, which would break resume part-way through a multi-day run.
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[seed {seed}] ignoring unreadable checkpoint {path.name}: {e}")
            continue
        if best is None or ckpt["tokens_seen"] > best["tokens_seen"]:
            best = ckpt
    return best


def train_sae_single_seed(
    seed,
    activation_stream,
    config,
    checkpoint_tokens,
    checkpoint_dir,
    hf_repo_id=None,
    checkpoint_every_seconds=CHECKPOINT_EVERY_SECONDS,
    log_every_steps=LOG_EVERY_STEPS,
):
    """Train one SAE for a given seed, saving a milestone checkpoint at each token count in
    checkpoint_tokens plus a rolling `_latest` checkpoint every checkpoint_every_seconds.
    Resumes from whichever of the two is furthest along. If hf_repo_id is given, milestone
    checkpoints are also pushed to that Hugging Face model repo.

    Returns (sae, history), where history holds the training curves.
    """
    device = config["device"]
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sae = SparseAutoencoder(config["d_model"], config["n_features"], seed=seed).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config["lr"])

    checkpoint_tokens = sorted(checkpoint_tokens)
    tokens_seen = 0
    step = 0
    history = empty_history()

    if hf_repo_id is not None:
        from huggingface_hub import HfApi
        hf_api = HfApi()

    resume = load_resume_state(checkpoint_dir, seed)
    if resume is not None:
        sae.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        tokens_seen = resume["tokens_seen"]
        step = resume.get("step", 0)
        history = resume.get("history", empty_history())
        print(f"[seed {seed}] resumed at {tokens_seen:,} tokens")
        print(f"[seed {seed}] CAVEAT: the activation stream restarts from the beginning of "
              f"the dataset, so tokens already trained on will be seen again. The token "
              f"counts below therefore measure optimization steps, not unique tokens.")

    next_checkpoint_idx = sum(t <= tokens_seen for t in checkpoint_tokens)
    if next_checkpoint_idx >= len(checkpoint_tokens):
        print(f"[seed {seed}] already at {tokens_seen:,} tokens; nothing left to train.")
        return sae, history

    history_path = checkpoint_dir.parent / f"training_history_seed{seed}.json"
    # Accumulate curve metrics as on-device tensors and only pull them to the host at
    # logging time; calling .item() every step would force a GPU sync 244k times per seed.
    interval_sums = torch.zeros(4, device=device)  # recon, sparsity, total, l0
    interval_n = 0
    seen_active = torch.zeros(config["n_features"], dtype=torch.bool, device=device)
    last_checkpoint_time = time.time()

    def build_state():
        return {
            "model_state_dict": sae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokens_seen": tokens_seen,
            "step": step,
            "seed": seed,
            "config": config,
            "history": history,
        }

    for batch in activation_stream:
        batch = batch.to(device)
        x_hat, f, loss_dict = sae(batch)
        loss = loss_dict["recon_loss"] + config["l1_coeff"] * loss_dict["sparsity_loss"]

        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            sae.W_dec.grad = remove_parallel_component(sae.W_dec.data, sae.W_dec.grad)
        optimizer.step()
        sae.normalize_decoder()

        # batch is already the flattened (batch_size * seq_len, d_model) activations
        # from activation_stream_generator -- batch.shape[0] IS the real token count
        # for this step. Do NOT multiply by seq_len again (that was double-counting).
        tokens_seen += batch.shape[0]
        step += 1

        with torch.no_grad():
            active = f > 0
            interval_sums[0] += loss_dict["recon_loss"].detach()
            interval_sums[1] += loss_dict["sparsity_loss"].detach()
            interval_sums[2] += loss.detach()
            interval_sums[3] += active.float().sum(dim=1).mean()
            interval_n += 1
            seen_active |= active.any(dim=0)

        if step % log_every_steps == 0:
            recon, sparsity, total, l0 = (interval_sums / interval_n).tolist()
            history["step"].append(step)
            history["tokens_seen"].append(tokens_seen)
            history["recon_loss"].append(recon)
            history["sparsity_loss"].append(sparsity)
            history["l1_term"].append(config["l1_coeff"] * sparsity)
            history["total_loss"].append(total)
            history["l0"].append(l0)
            # Dead = never fired anywhere in this logging window.
            history["dead_frac"].append(1.0 - seen_active.float().mean().item())
            print(f"[seed {seed}] step {step:>7} | {tokens_seen:>14,} tok | "
                  f"recon {recon:.5f} | l1_term {config['l1_coeff'] * sparsity:.5f} | "
                  f"L0 {l0:6.1f} | dead {100 * history['dead_frac'][-1]:5.1f}%")
            interval_sums.zero_()
            interval_n = 0
            seen_active = torch.zeros(config["n_features"], dtype=torch.bool, device=device)

        if time.time() - last_checkpoint_time >= checkpoint_every_seconds:
            save_checkpoint_atomic(build_state(), checkpoint_dir / f"seed{seed}_latest.pt")
            with open(history_path, "w") as fh:
                json.dump(history, fh)
            last_checkpoint_time = time.time()
            print(f"[seed {seed}] rolling checkpoint at {tokens_seen:,} tokens")

        if next_checkpoint_idx < len(checkpoint_tokens) and tokens_seen >= checkpoint_tokens[next_checkpoint_idx]:
            ckpt_name = f"seed{seed}_tokens{checkpoint_tokens[next_checkpoint_idx]}.pt"
            ckpt_path = checkpoint_dir / ckpt_name
            save_checkpoint_atomic(build_state(), ckpt_path)
            print(f"[seed {seed}] milestone checkpoint saved at {tokens_seen:,} tokens")

            if hf_repo_id is not None:
                try:
                    hf_api.upload_file(
                        path_or_fileobj=str(ckpt_path),
                        path_in_repo=f"checkpoints/{ckpt_name}",
                        repo_id=hf_repo_id,
                        repo_type="model",
                    )
                    print(f"[seed {seed}] checkpoint pushed to hf.co/{hf_repo_id}/checkpoints/{ckpt_name}")
                except Exception as e:
                    # Don't let an upload failure kill the training run -- local
                    # checkpoint already saved above, so we can retry the push later.
                    print(f"[seed {seed}] WARNING: HF upload failed ({e}); local checkpoint is still safe.")

            next_checkpoint_idx += 1

        if next_checkpoint_idx >= len(checkpoint_tokens):
            break

    save_checkpoint_atomic(build_state(), checkpoint_dir / f"seed{seed}_latest.pt")
    with open(history_path, "w") as fh:
        json.dump(history, fh)

    return sae, history


trained_saes = {}
training_histories = {}
for seed in CONFIG["seeds"]:
    print(f"=== Training seed {seed} ===")
    stream = activation_stream_generator(
        model=model,
        dataset_name="monology/pile-uncopyrighted",
        hook_point=CONFIG["hook_point"],
        seq_len=CONFIG["seq_len"],
        batch_size=CONFIG["batch_size"],
        device=CONFIG["device"],
    )
    trained_saes[seed], training_histories[seed] = train_sae_single_seed(
        seed=seed,
        activation_stream=stream,
        config=CONFIG,
        checkpoint_tokens=CHECKPOINT_TOKENS,
        checkpoint_dir=CHECKPOINT_DIR,
        hf_repo_id=HF_REPO_ID,
    )

print(f"\nTrained {len(trained_saes)} SAEs with seeds: {list(trained_saes.keys())}")
print(f"Checkpoints saved at token counts: {CHECKPOINT_TOKENS}")

"""### 3b. Training curves — is l1_coeff too large?

`l1_coeff` is 1.0 here, three orders of magnitude above the 1e-3 used for the earlier
1M/10M runs, so these curves are the check on whether the sparsity penalty is dominating.
The warning signs, in order of how conclusive they are: the dead-feature fraction climbing
toward 100%, L0 collapsing toward 0, and the weighted L1 term sitting far above the
reconstruction term. Any of those means most features are being pushed to zero and the SAE
is buying sparsity at the cost of reconstructing anything.
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
    axes[0, 1].plot(h["tokens_seen"], h["l1_term"], label=f"l1_coeff x L1 ({config['l1_coeff']:g})")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Loss term")
    axes[0, 1].set_title(f"Loss terms compared (seed {anchor_seed})")

    axes[1, 0].set_ylabel("L0 (mean active features/token)")
    axes[1, 0].set_title("Sparsity: L0 -> 0 means over-penalized")

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
              f"collapsing to the trivial all-zero solution. Consider 1e-3 (the value used "
              f"for the 1M/10M runs).")
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

n_total = len(activations)
freq_accum = torch.zeros(CONFIG["n_features"])
mean_accum = torch.zeros(CONFIG["n_features"])

with torch.no_grad():
    for start in tqdm(range(0, n_total, STAT_BATCH), desc="Computing feature stats"):
        batch = activations[start : start + STAT_BATCH].to(CONFIG["device"])
        feats = reference_sae.encode(batch)               # (B, n_features)
        freq_accum += (feats > 0).float().sum(dim=0).cpu()
        mean_accum += feats.sum(dim=0).cpu()

activation_freq = (freq_accum / n_total).numpy()
mean_activation  = (mean_accum  / n_total).numpy()

# Compare stable vs unstable
print("=== Stable Features (n={}) ===".format(len(stable_indices)))
print(f"  Decoder norm:      mean={decoder_norms[stable_indices].mean():.3f}, std={decoder_norms[stable_indices].std():.3f}")
print(f"  Activation freq:   mean={activation_freq[stable_indices].mean():.4f}, std={activation_freq[stable_indices].std():.4f}")
print(f"  Mean activation:   mean={mean_activation[stable_indices].mean():.4f}, std={mean_activation[stable_indices].std():.4f}")

print("\n=== Unstable Features (n={}) ===".format(len(unstable_indices)))
print(f"  Decoder norm:      mean={decoder_norms[unstable_indices].mean():.3f}, std={decoder_norms[unstable_indices].std():.3f}")
print(f"  Activation freq:   mean={activation_freq[unstable_indices].mean():.4f}, std={activation_freq[unstable_indices].std():.4f}")
print(f"  Mean activation:   mean={mean_activation[unstable_indices].mean():.4f}, std={mean_activation[unstable_indices].std():.4f}")

# Visualize: stable vs unstable feature properties
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Decoder norms
ax1 = axes[0]
ax1.hist(decoder_norms[unstable_indices], bins=30, alpha=0.6, label="Unstable", color="#ff7f0e")
ax1.hist(decoder_norms[stable_indices], bins=10, alpha=0.8, label="Stable", color="#2ca02c")
ax1.set_xlabel("Decoder Norm")
ax1.set_ylabel("Count")
ax1.set_title("Decoder Norms: Stable vs Unstable")
ax1.legend()

# Plot 2: Activation frequency
ax2 = axes[1]
ax2.hist(activation_freq[unstable_indices], bins=30, alpha=0.6, label="Unstable", color="#ff7f0e")
ax2.hist(activation_freq[stable_indices], bins=10, alpha=0.8, label="Stable", color="#2ca02c")
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
    batch_size: int = 1024,
    n_samples: int = 10000,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute reconstruction contribution for each feature.

    Measures how much reconstruction error increases when each feature is ablated (zeroed out).
    HIGH value = feature is important for reconstruction (more likely stable)
    LOW value = feature is redundant (less stable)

    Returns:
        contributions: (n_features,) array of reconstruction contributions
    """
    sae.eval()
    n_features = sae.n_features

    # Use a subset of activations for speed
    sample_indices = np.random.choice(len(activations), min(n_samples, len(activations)), replace=False)
    sample_acts = activations[sample_indices].to(device)

    contributions = np.zeros(n_features)

    print(f"Computing reconstruction contributions for {n_features} features...")

    with torch.no_grad():
        # Get baseline reconstruction error
        features = sae.encode(sample_acts)
        baseline_recon = sae.decode(features)
        baseline_mse = F.mse_loss(baseline_recon, sample_acts, reduction='none').mean(dim=1)

        # For each feature, compute MSE when that feature is ablated
        for feat_idx in tqdm(range(n_features)):
            # Zero out this feature
            ablated_features = features.clone()
            ablated_features[:, feat_idx] = 0

            # Reconstruct
            ablated_recon = sae.decode(ablated_features)
            ablated_mse = F.mse_loss(ablated_recon, sample_acts, reduction='none').mean(dim=1)

            # Contribution = increase in MSE when feature is removed
            contribution = (ablated_mse - baseline_mse).mean().cpu().item()
            contributions[feat_idx] = contribution

    return contributions

# Compute all four statistics
print("Computing geometric isolation...")
geometric_isolation = compute_geometric_isolation(reference_sae, k=10)

print("\nComputing reconstruction contribution...")
recon_contribution = compute_reconstruction_contribution(
    reference_sae,
    activations,
    n_samples=10000,
    device=CONFIG["device"]
)

print("\nDone! All four statistics computed.")

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
    ax.hist(values[unstable_indices], bins=30, alpha=0.6, label="Unstable", color="#ff7f0e", density=True)
    ax.hist(values[stable_indices], bins=10, alpha=0.8, label="Stable", color="#2ca02c", density=True)
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

X = np.column_stack([
    activation_freq,
    geometric_isolation,
    recon_contribution,
    mean_activation,
])[labeled_mask]

y = stable_mask[labeled_mask].astype(int)

print(f"Features excluded from classifier training (discarded middle): {middle_mask.sum()}")
print(f"Features used for classifier training: {labeled_mask.sum()}")

# Feature names for interpretation
feature_names = ["Activation Freq", "Geometric Isolation", "Recon Contribution", "Mean Activation"]

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
for name, values in [
    ("Activation Freq", activation_freq),
    ("Geometric Isolation", geometric_isolation),
    ("Recon Contribution", recon_contribution),
    ("Mean Activation", mean_activation),
]:
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
X_freq_geom = np.column_stack([activation_freq, geometric_isolation])[labeled_mask]
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
print(f"\n   Full Model (All 4 Features):")
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
