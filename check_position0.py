# -*- coding: utf-8 -*-
"""Pre-flight check: does a model put outsized residual norms at position 0?

GPT-2 is known to give the first position of every context a residual norm far larger than
any other position, whatever token sits there. Our stream chunks concatenated documents into
128-token rows, so 1 in 128 training rows would be such an outlier for GPT-2, and the SAE may
spend capacity reconstructing it. Pythia is the reference: whatever it does, the published
70m numbers were computed with every position kept.

For each model this prints the median residual norm at position 0 against the median over all
other positions, and the same for <|endoftext|> tokens against every other token, then applies
the rule from the camera-ready plan: drop position 0 for gpt2 only if its ratio is above 10x
while Pythia-70m's is below 3x. Dropping it is SAE_DROP_FIRST_POSITION_MODELS=gpt2 on both the
training and the scoring command.

The stream mirrors build_eval_activations (same dataset, tokenizer call, sequence length and
batch size), so the first batches here are the first eval batches there, and nothing is
dropped: this measures the stream as it is.

Usage:
    python check_position0.py --model gpt2 --model pythia-70m-deduped --batches 8
"""

import argparse

import torch

from eval_activations import BATCH_SIZE, DATASET, SEQ_LEN, hook_point_for, load_model

GPT2_RATIO_TO_DROP = 10.0
PYTHIA_RATIO_CEILING = 3.0


def norms_by_position(model_name: str, n_batches: int, device: str):
    """Residual norms (n_batches * BATCH_SIZE, SEQ_LEN) and the token ids that produced them."""
    from datasets import load_dataset

    hook_point = hook_point_for(model_name)
    layer_idx = int(hook_point.split(".")[1])
    model = load_model(model_name, device)

    dataset = load_dataset(DATASET, split="train", streaming=True)
    token_buffer, norms, ids = [], [], []
    for example in dataset:
        tokens = model.tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=SEQ_LEN * 10
        )["input_ids"][0]
        token_buffer.extend(tokens.tolist())
        while len(token_buffer) >= SEQ_LEN * BATCH_SIZE and len(norms) < n_batches:
            batch = torch.tensor(token_buffer[: SEQ_LEN * BATCH_SIZE]).reshape(BATCH_SIZE, SEQ_LEN)
            token_buffer = token_buffer[SEQ_LEN * BATCH_SIZE:]
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    batch.to(device), names_filter=[hook_point], stop_at_layer=layer_idx + 1
                )
            norms.append(cache[hook_point].norm(dim=-1).cpu())
            ids.append(batch)
        if len(norms) >= n_batches:
            break

    eot = model.tokenizer.eos_token_id
    del model
    return torch.cat(norms), torch.cat(ids), hook_point, eot


def ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a.median() / b.median()) if a.numel() and b.numel() else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pos0 = {}
    for name in args.model:
        norms, ids, hook, eot = norms_by_position(name, args.batches, device)
        first, rest = norms[:, 0], norms[:, 1:].flatten()
        is_eot = ids == eot
        pos0[name] = ratio(first, rest)
        print(f"\n{name} at {hook} ({norms.numel():,} tokens)")
        print(f"  median norm, position 0:   {first.median():10.2f}")
        print(f"  median norm, positions 1+: {rest.median():10.2f}   ratio {pos0[name]:6.1f}x")
        print(f"  median norm, <|endoftext|> ({int(is_eot.sum()):,} tokens) vs other tokens: "
              f"ratio {ratio(norms[is_eot], norms[~is_eot]):6.1f}x")

    if "gpt2" in pos0 and "pythia-70m-deduped" in pos0:
        drop = (pos0["gpt2"] > GPT2_RATIO_TO_DROP
                and pos0["pythia-70m-deduped"] < PYTHIA_RATIO_CEILING)
        print(f"\nVerdict: {'DROP' if drop else 'KEEP'} position 0 for gpt2 "
              f"(rule: gpt2 > {GPT2_RATIO_TO_DROP:g}x and pythia-70m < "
              f"{PYTHIA_RATIO_CEILING:g}x).")
        if drop:
            print("  Prefix the gpt2 training and scoring commands with "
                  "SAE_DROP_FIRST_POSITION_MODELS=gpt2.")
        else:
            print("  Leave SAE_DROP_FIRST_POSITION_MODELS unset; the protocol stays identical.")


if __name__ == "__main__":
    main()
