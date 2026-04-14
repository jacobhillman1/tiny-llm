"""
Run the tiny nanoGPT checkpoint with numpy only.

This intentionally does not import torch. It loads the PyTorch zip checkpoint
format directly, rebuilds tensors as numpy arrays, and runs the inference path
from model.py.
"""

import argparse
import io
import math
import pickle
import re
import zipfile
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    bias: bool = False


@dataclass
class StorageRecord:
    raw: bytes
    dtype: np.dtype
    numel: int


class TorchStorageType:
    def __init__(self, name):
        self.__module__ = "torch"
        self.__name__ = name


def _dtype_from_storage(storage_type):
    name = getattr(storage_type, "__name__", str(storage_type))
    dtype_map = {
        "FloatStorage": np.float32,
        "HalfStorage": np.float16,
        "DoubleStorage": np.float64,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool_,
    }
    if name not in dtype_map:
        raise ValueError(f"Unsupported torch storage type: {name}")
    return np.dtype(dtype_map[name])


def _rebuild_tensor(storage, storage_offset, size, stride):
    shape = tuple(int(v) for v in size)
    strides = tuple(int(v) * storage.dtype.itemsize for v in stride)
    arr = np.ndarray(
        shape=shape,
        dtype=storage.dtype,
        buffer=storage.raw,
        offset=int(storage_offset) * storage.dtype.itemsize,
        strides=strides,
    )
    return arr.copy()


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    return _rebuild_tensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return data


class TorchCheckpointUnpickler(pickle.Unpickler):
    def __init__(self, file, archive, prefix):
        super().__init__(file)
        self.archive = archive
        self.prefix = prefix
        self.storages = {}

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or not pid:
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid!r}")
        if pid[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id type: {pid[0]!r}")

        _, storage_type, storage_key, location, numel = pid
        if storage_key not in self.storages:
            dtype = _dtype_from_storage(storage_type)
            path = f"{self.prefix}data/{storage_key}"
            raw = self.archive.read(path)
            expected = int(numel) * dtype.itemsize
            if len(raw) < expected:
                raise ValueError(
                    f"Storage {storage_key} is too small: got {len(raw)} bytes, "
                    f"expected at least {expected}"
                )
            self.storages[storage_key] = StorageRecord(raw=raw, dtype=dtype, numel=int(numel))
        return self.storages[storage_key]

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_tensor":
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return TorchStorageType(name)
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")


def _find_archive_prefix(archive):
    data_pkl = [name for name in archive.namelist() if name.endswith("data.pkl")]
    if not data_pkl:
        raise ValueError("Checkpoint archive does not contain data.pkl")
    if len(data_pkl) > 1:
        exact = [name for name in data_pkl if name == "data.pkl"]
        data_name = exact[0] if exact else sorted(data_pkl)[0]
    else:
        data_name = data_pkl[0]
    return data_name[: -len("data.pkl")]


def load_torch_state_dict(path):
    with zipfile.ZipFile(path) as archive:
        prefix = _find_archive_prefix(archive)
        payload = archive.read(f"{prefix}data.pkl")
        obj = TorchCheckpointUnpickler(io.BytesIO(payload), archive, prefix).load()

    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], (dict, OrderedDict)):
        obj = obj["model"]
    if not isinstance(obj, (dict, OrderedDict)):
        raise TypeError(f"Expected a state dict, got {type(obj).__name__}")

    state = OrderedDict()
    for key, value in obj.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        state[key] = value
    return state


def infer_config(state, n_head):
    wte = state["transformer.wte.weight"]
    wpe = state["transformer.wpe.weight"]
    layer_ids = set()
    for key in state:
        match = re.match(r"transformer\.h\.(\d+)\.", key)
        if match:
            layer_ids.add(int(match.group(1)))
    if not layer_ids:
        raise ValueError("Could not infer transformer layer count from checkpoint")

    vocab_size, n_embd = wte.shape
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")

    has_bias = any(key.endswith(".bias") for key in state)
    return GPTConfig(
        block_size=int(wpe.shape[0]),
        vocab_size=int(vocab_size),
        n_layer=max(layer_ids) + 1,
        n_head=int(n_head),
        n_embd=int(n_embd),
        bias=has_bias,
    )


def validate_state_dict(state, config):
    required = [
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.ln_f.weight",
        "lm_head.weight",
    ]
    for layer in range(config.n_layer):
        prefix = f"transformer.h.{layer}"
        required.extend(
            [
                f"{prefix}.ln_1.weight",
                f"{prefix}.attn.c_attn.weight",
                f"{prefix}.attn.c_proj.weight",
                f"{prefix}.ln_2.weight",
                f"{prefix}.mlp.c_fc.weight",
                f"{prefix}.mlp.c_proj.weight",
            ]
        )
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError("Checkpoint is missing required keys: " + ", ".join(missing))


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def layer_norm(x, weight, bias=None, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    out = out * weight
    if bias is not None:
        out = out + bias
    return out


def gelu(x):
    erf = np.vectorize(math.erf, otypes=[np.float32])
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def linear(x, weight, bias=None):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


class NumpyGPT:
    def __init__(self, config, state):
        self.config = config
        self.state = state

    def _get(self, key):
        return self.state[key]

    def _maybe_bias(self, key):
        return self.state.get(key)

    def attention(self, x, layer):
        cfg = self.config
        prefix = f"transformer.h.{layer}.attn"
        bsz, seq_len, channels = x.shape
        head_size = channels // cfg.n_head

        qkv = linear(
            x,
            self._get(f"{prefix}.c_attn.weight"),
            self._maybe_bias(f"{prefix}.c_attn.bias"),
        )
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(bsz, seq_len, cfg.n_head, head_size).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, cfg.n_head, head_size).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, cfg.n_head, head_size).transpose(0, 2, 1, 3)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_size)
        mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        att = np.where(mask[None, None, :, :], att, -np.inf)
        att = softmax(att, axis=-1)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seq_len, channels)
        return linear(
            y,
            self._get(f"{prefix}.c_proj.weight"),
            self._maybe_bias(f"{prefix}.c_proj.bias"),
        )

    def mlp(self, x, layer):
        prefix = f"transformer.h.{layer}.mlp"
        x = linear(
            x,
            self._get(f"{prefix}.c_fc.weight"),
            self._maybe_bias(f"{prefix}.c_fc.bias"),
        )
        x = gelu(x)
        return linear(
            x,
            self._get(f"{prefix}.c_proj.weight"),
            self._maybe_bias(f"{prefix}.c_proj.bias"),
        )

    def block(self, x, layer):
        prefix = f"transformer.h.{layer}"
        x = x + self.attention(
            layer_norm(
                x,
                self._get(f"{prefix}.ln_1.weight"),
                self._maybe_bias(f"{prefix}.ln_1.bias"),
            ),
            layer,
        )
        x = x + self.mlp(
            layer_norm(
                x,
                self._get(f"{prefix}.ln_2.weight"),
                self._maybe_bias(f"{prefix}.ln_2.bias"),
            ),
            layer,
        )
        return x

    def forward(self, idx):
        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape (B, T), got {idx.shape}")
        bsz, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; "
                f"block size is {self.config.block_size}"
            )

        tok_emb = self._get("transformer.wte.weight")[idx]
        pos_emb = self._get("transformer.wpe.weight")[np.arange(seq_len)]
        x = tok_emb + pos_emb[None, :, :]
        for layer in range(self.config.n_layer):
            x = self.block(x, layer)
        x = layer_norm(
            x,
            self._get("transformer.ln_f.weight"),
            self._maybe_bias("transformer.ln_f.bias"),
        )
        return linear(
            x[:, [-1], :],
            self._get("lm_head.weight"),
            self._maybe_bias("lm_head.bias"),
        )


def load_meta(path):
    with open(path, "rb") as f:
        meta = pickle.load(f)
    return meta["stoi"], meta["itos"]


def encode(text, stoi):
    try:
        return [stoi[ch] for ch in text]
    except KeyError as exc:
        raise ValueError(f"Prompt contains character not in vocabulary: {exc.args[0]!r}") from exc


def decode(ids, itos):
    return "".join(itos[int(idx)] for idx in ids)


def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, rng=None):
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    rng = rng or np.random.default_rng()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits = model.forward(idx_cond)[:, -1, :] / temperature
        if top_k is not None:
            k = min(int(top_k), logits.shape[-1])
            if k <= 0:
                raise ValueError("top_k must be positive when provided")
            kth = np.partition(logits, -k, axis=-1)[:, -k][:, None]
            logits = np.where(logits < kth, -np.inf, logits)
        probs = softmax(logits, axis=-1)
        next_id = rng.choice(logits.shape[-1], p=probs[0])
        idx = np.concatenate([idx, np.array([[next_id]], dtype=np.int64)], axis=1)
    return idx


def parse_args():
    parser = argparse.ArgumentParser(description="Run tiny nanoGPT with numpy only.")
    parser.add_argument("--ckpt", default="out-tiny-v2/model_fp32.pt")
    parser.add_argument("--meta", default="data/shakespeare_char/meta.pkl")
    parser.add_argument("--prompt", default="to be or not to be,:")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-head", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    state = load_torch_state_dict(args.ckpt)
    config = infer_config(state, args.n_head)
    validate_state_dict(state, config)

    stoi, itos = load_meta(args.meta)
    start_ids = encode(args.prompt, stoi)
    idx = np.array([start_ids], dtype=np.int64)

    model = NumpyGPT(config, state)
    rng = np.random.default_rng(args.seed)
    out = generate(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=rng,
    )
    print(decode(out[0], itos))


if __name__ == "__main__":
    main()
