"""
Compare the numpy tiny GPT runner against the Torch model.

This script is a verification helper. Unlike run_tiny_numpy.py, it imports torch
on purpose so it can check checkpoint loading and logits parity.
"""

import argparse
import sys

import numpy as np
import torch

from model import GPT, GPTConfig as TorchGPTConfig
from run_tiny_numpy import (
    NumpyGPT,
    encode,
    infer_config,
    load_meta,
    load_torch_state_dict,
    validate_state_dict,
)


def normalize_torch_state(obj):
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    state = {}
    for key, value in obj.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        if isinstance(value, dict) and set(value) >= {"q", "scale"}:
            value = value["q"].float() * value["scale"].float()
        state[key] = value
    return state


def compare_arrays(torch_state, numpy_state, atol):
    failures = []
    print("--- checkpoint arrays ---")
    for key in sorted(numpy_state):
        if key not in torch_state:
            failures.append(f"{key}: missing from Torch state")
            continue
        torch_arr = torch_state[key].detach().cpu().numpy()
        numpy_arr = numpy_state[key]
        if torch_arr.shape != numpy_arr.shape:
            failures.append(f"{key}: shape mismatch {torch_arr.shape} != {numpy_arr.shape}")
            continue
        diff = np.abs(torch_arr - numpy_arr)
        max_abs = float(diff.max()) if diff.size else 0.0
        print(f"{key:45s} shape={str(numpy_arr.shape):12s} max_abs_err={max_abs:.9g}")
        if max_abs > atol:
            failures.append(f"{key}: max_abs_err {max_abs:.9g} > {atol}")
    return failures


def compare_logits(args, torch_state, numpy_state, config):
    validate_state_dict(numpy_state, config)
    stoi, _ = load_meta(args.meta)
    ids = encode(args.prompt, stoi)
    idx_np = np.array([ids], dtype=np.int64)

    numpy_model = NumpyGPT(config, numpy_state)
    numpy_logits = numpy_model.forward(idx_np)

    torch_config = TorchGPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=0.0,
        bias=config.bias,
    )
    torch_model = GPT(torch_config)
    torch_model.load_state_dict(torch_state)
    torch_model.eval()
    with torch.no_grad():
        torch_logits, _ = torch_model(torch.tensor(idx_np, dtype=torch.long))
    torch_logits = torch_logits.detach().cpu().numpy()

    diff = np.abs(torch_logits - numpy_logits)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    torch_last = torch_logits[0, -1]
    numpy_last = numpy_logits[0, -1]
    torch_argmax = int(torch_last.argmax())
    numpy_argmax = int(numpy_last.argmax())
    torch_top5 = torch_last.argsort()[-5:][::-1]
    numpy_top5 = numpy_last.argsort()[-5:][::-1]

    print("\n--- logits ---")
    print(f"max_abs_err  = {max_abs:.9g}")
    print(f"mean_abs_err = {mean_abs:.9g}")
    print(f"torch_argmax = {torch_argmax}")
    print(f"numpy_argmax = {numpy_argmax}")
    print(f"torch_top5   = {torch_top5.tolist()}")
    print(f"numpy_top5   = {numpy_top5.tolist()}")

    failures = []
    if max_abs > args.logit_atol:
        failures.append(f"logits: max_abs_err {max_abs:.9g} > {args.logit_atol}")
    if torch_argmax != numpy_argmax:
        failures.append(f"logits: argmax mismatch {torch_argmax} != {numpy_argmax}")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Torch and numpy tiny GPT inference.")
    parser.add_argument("--ckpt", default="out-tiny-v2/model_fp32.pt")
    parser.add_argument("--meta", default="data/shakespeare_char/meta.pkl")
    parser.add_argument("--prompt", default="to be or not to be,:")
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--array-atol", type=float, default=0.0)
    parser.add_argument("--logit-atol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    numpy_state = load_torch_state_dict(args.ckpt)
    config = infer_config(numpy_state, args.n_head)
    torch_state = normalize_torch_state(
        torch.load(args.ckpt, map_location="cpu", weights_only=True)
    )

    failures = []
    failures.extend(compare_arrays(torch_state, numpy_state, args.array_atol))
    failures.extend(compare_logits(args, torch_state, numpy_state, config))

    if failures:
        print("\n--- failures ---")
        for failure in failures:
            print(failure)
        return 1
    print("\nAll comparisons passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
