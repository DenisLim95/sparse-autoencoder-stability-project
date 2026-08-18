# -*- coding: utf-8 -*-
"""Build (and cache) the evaluation activation set for any Pythia model.

Extracted from the sweep's `load_pythia` and `activation_stream_generator` so that
budget_transfer.py and cross_scale_transfer.py stream tokens the same way the training run
did, rather than each keeping a copy. `topk_sweep_experiments.py` still has its own copy
because it executes on import and cannot be imported here; `budget_transfer.py validate` is
what detects any divergence between the two.

Determinism: the stream re-reads the dataset from the beginning with a fixed sequence length
and batch size, so the same (model, hook, n_batches) gives the same tokens on every machine,
provided the dataset snapshot on the Hub has not moved.
"""

import os
from pathlib import Path

import torch

DATASET = "monology/pile-uncopyrighted"
SEQ_LEN = 128
BATCH_SIZE = 256
N_EVAL_BATCHES = int(os.environ.get("SAE_EVAL_BATCHES") or 40)
CACHE_DIR = Path(os.environ.get("SAE_CACHE_DIR") or "cache")

# (d_model, n_layers) per model. Needed before a model is loaded, because the run directory,
# the Hub prefix and the dictionary width all depend on it. The sweep asserts every entry
# against the loaded model, so a wrong number fails loudly rather than mislabelling a run.
PYTHIA_GEOMETRY = {
    "pythia-70m-deduped": (512, 6),
    "pythia-160m-deduped": (768, 12),
    "pythia-410m-deduped": (1024, 24),
    "pythia-1b-deduped": (2048, 16),
    "pythia-1.4b-deduped": (2048, 24),
}


def relative_depth_layer(n_layers: int, rel_depth: float) -> int:
    """Layer index at a matched *relative* depth.

    Absolute layer numbers are not comparable across models of different depth: layer 3 is the
    middle of Pythia-70m's 6 blocks but a quarter of the way into 160m's 12. rel_depth=0.5
    reproduces the 70m run's blocks.3.
    """
    return max(0, min(n_layers - 1, round(rel_depth * n_layers)))


def hook_point_for(model_name: str, rel_depth: float = 0.5) -> str:
    """Residual-stream hook at a matched relative depth."""
    _, n_layers = PYTHIA_GEOMETRY[model_name]
    return f"blocks.{relative_depth_layer(n_layers, rel_depth)}.hook_resid_post"


def hub_prefix_for(model_name: str, objective_tag: str, rel_depth: float = 0.5) -> str:
    """Where this model's checkpoints live on the Hub, matching the sweep's convention.

    The original 70m sweep predates model-scoped prefixes and its published artifacts sit
    directly under the objective tag, so that one case is preserved; every other model is
    scoped by model and layer to keep incompatible dictionary widths apart.
    """
    _, n_layers = PYTHIA_GEOMETRY[model_name]
    layer = relative_depth_layer(n_layers, rel_depth)
    if model_name == "pythia-70m-deduped" and layer == 3:
        return objective_tag
    return f"{model_name}_L{layer}_{objective_tag}"


def load_model(model_name: str, device: str):
    """Load a Pythia (GPT-NeoX) model into transformer_lens.

    transformer_lens's NeoX weight converter reads `hf_model.embed_out`, but recent
    transformers renamed that head to `lm_head`, so letting transformer_lens load the model
    itself dies with "'GPTNeoXForCausalLM' object has no attribute 'embed_out'". Loading the
    HF model here and aliasing the name the converter expects onto the actual output head
    works on either version and leaves the weights untouched.
    """
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM
    from transformer_lens.loading_from_pretrained import get_official_model_name

    official = get_official_model_name(model_name)
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(official, dtype=torch.float32)
    except TypeError:  # transformers < 4.56 spells this torch_dtype
        hf_model = AutoModelForCausalLM.from_pretrained(official, torch_dtype=torch.float32)
    if not hasattr(hf_model, "embed_out"):
        hf_model.embed_out = hf_model.lm_head

    model = HookedTransformer.from_pretrained(model_name, hf_model=hf_model, device=device)
    model.eval()
    return model


def build_eval_activations(model_name: str, hook_point: str, device: str,
                           n_batches: int = N_EVAL_BATCHES,
                           cache_dir: Path = CACHE_DIR) -> torch.Tensor:
    """Residual-stream activations at `hook_point`, cached to disk.

    Cached per (model, hook, size), so switching scales or shrinking the eval set never reads
    the wrong file. Returns a (n_batches * BATCH_SIZE * SEQ_LEN, d_model) CPU tensor.
    """
    tag = f"{model_name.replace('/', '_')}_{hook_point}_{n_batches}x{BATCH_SIZE}x{SEQ_LEN}"
    cache = Path(cache_dir) / f"eval_activations_{tag}.pt"
    if cache.exists():
        acts = torch.load(cache, map_location="cpu")
        print(f"Eval activations from cache: {tuple(acts.shape)}  ({cache.name})")
        return acts

    from datasets import load_dataset

    n_tokens = n_batches * BATCH_SIZE * SEQ_LEN
    print(f"Building eval activations for {model_name} at {hook_point} "
          f"({n_tokens:,} tokens) on {device}")
    model = load_model(model_name, device)
    layer_idx = int(hook_point.split(".")[1])

    dataset = load_dataset(DATASET, split="train", streaming=True)
    token_buffer, batches = [], []

    for example in dataset:
        tokens = model.tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=SEQ_LEN * 10
        )["input_ids"][0]
        token_buffer.extend(tokens.tolist())

        while len(token_buffer) >= SEQ_LEN * BATCH_SIZE and len(batches) < n_batches:
            batch_tokens = torch.tensor(
                token_buffer[: SEQ_LEN * BATCH_SIZE]
            ).reshape(BATCH_SIZE, SEQ_LEN)
            token_buffer = token_buffer[SEQ_LEN * BATCH_SIZE:]
            with torch.no_grad():
                _, cached = model.run_with_cache(
                    batch_tokens.to(device),
                    names_filter=[hook_point],
                    stop_at_layer=layer_idx + 1,
                )
                a = cached[hook_point]
                batches.append(a.reshape(-1, a.shape[-1]).cpu())
            print(f"  batch {len(batches)}/{n_batches}", end="\r")
        if len(batches) >= n_batches:
            break

    acts = torch.cat(batches, dim=0)
    print(f"\nEval activations: {tuple(acts.shape)} ({acts.numel() * 4 / 1e9:.1f} GB)")

    # Written through a temporary file and only then moved into place: at this size the write
    # is the step most likely to fail, and a half-written .pt at the real path is worse than no
    # cache at all, because the next run finds it, trusts it and dies loading it. A cache is an
    # optimization, so a failure here costs recomputation rather than the run.
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".pt.partial")
    try:
        torch.save(acts, tmp)
        tmp.replace(cache)
        print(f"Cached at {cache}")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        print(f"Could not cache activations ({type(e).__name__}: {e}); "
              f"continuing from memory, and the next run will rebuild them.")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return acts
