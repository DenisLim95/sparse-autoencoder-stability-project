"""Checks for SAE_DROP_FIRST_POSITION_MODELS: off by default, scoped to the listed models, and
removing exactly position 0 when on. No Hub, no GPU, no model download.

    python test_drop_first_position.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("SAE_DROP_FIRST_POSITION_MODELS", None)

import tempfile
from pathlib import Path

import torch

import cross_scale_transfer
import eval_activations
from eval_activations import (
    BATCH_SIZE,
    SEQ_LEN,
    build_eval_activations,
    drops_first_position,
    flatten_positions,
    stream_tag,
)

failures = []


def check(label, ok):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


PYTHIA = "pythia-70m-deduped"
ORIGINAL_STATS = f"stats_{PYTHIA}_k64_seed42_t500000000_e40.npz"


def cache_hit(model_name, hook, expected_name):
    """build_eval_activations returns a cached file without loading any model, so planting
    a file under the expected name and getting it back proves which name it looks for."""
    def refuse(*_, **__):
        raise RuntimeError("cache miss")

    eval_activations.load_model = refuse  # a miss must fail here, not download a model
    with tempfile.TemporaryDirectory() as tmp:
        marker = torch.full((3, 2), 7.0)
        torch.save(marker, Path(tmp) / expected_name)
        try:
            got = build_eval_activations(model_name, hook, "cpu", n_batches=1, cache_dir=tmp)
        except Exception:
            return False  # cache missed and it tried to load a model
        return torch.equal(got, marker)


print("Default (unset)")
check("no model drops position 0", not drops_first_position("gpt2")
      and not drops_first_position(PYTHIA))
check("stream tags are empty", stream_tag("gpt2") == "" and stream_tag(PYTHIA) == "")
check("stats cache names are unchanged",
      cross_scale_transfer.stats_path(PYTHIA, 64, 42, 500_000_000).name == ORIGINAL_STATS)
check("eval cache name is unchanged",
      cache_hit(PYTHIA, "blocks.3.hook_resid_post",
                f"eval_activations_{PYTHIA}_blocks.3.hook_resid_post_1x{BATCH_SIZE}x{SEQ_LEN}.pt"))

print("\nSAE_DROP_FIRST_POSITION_MODELS=gpt2")
os.environ["SAE_DROP_FIRST_POSITION_MODELS"] = "gpt2"
check("gpt2 drops position 0", drops_first_position("gpt2"))
check("pythia-70m is untouched", not drops_first_position(PYTHIA) and stream_tag(PYTHIA) == "")
check("pythia-70m stats name is unchanged",
      cross_scale_transfer.stats_path(PYTHIA, 64, 42, 500_000_000).name == ORIGINAL_STATS)
check("gpt2 stats name carries the tag",
      cross_scale_transfer.stats_path("gpt2", 64, 42, 500_000_000).name
      == "stats_gpt2_k64_seed42_t500000000_e40_nopos0.npz")
check("gpt2 eval cache name carries the tag",
      cache_hit("gpt2", "blocks.6.hook_resid_post",
                f"eval_activations_gpt2_blocks.6.hook_resid_post_1x{BATCH_SIZE}x{SEQ_LEN}"
                f"_nopos0.pt"))
os.environ["SAE_DROP_FIRST_POSITION_MODELS"] = " gpt2 , pythia-160m-deduped "
check("list entries are whitespace-tolerant",
      drops_first_position("gpt2") and drops_first_position("pythia-160m-deduped"))
os.environ.pop("SAE_DROP_FIRST_POSITION_MODELS")

print("\nflatten_positions")
acts = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)  # (batch, seq, d)
check("off: same as a plain reshape", torch.equal(flatten_positions(acts, False),
                                                  acts.reshape(-1, 3)))
dropped = flatten_positions(acts, True)
check("on: seq - 1 positions per sequence", dropped.shape == (2 * 3, 3))
check("on: exactly position 0 removed, order kept",
      torch.equal(dropped, torch.cat([acts[0, 1:], acts[1, 1:]])))

print(f"\n{len(failures)} failure(s)" if failures else "\nAll checks passed.")
raise SystemExit(1 if failures else 0)
