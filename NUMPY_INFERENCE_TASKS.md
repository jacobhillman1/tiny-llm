# Numpy Inference Runner Task Breakdown

Goal: build a standalone numpy inference path for the tiny nanoGPT checkpoint without importing Torch at runtime.

Primary target file:

- `run_tiny_numpy.py`

Optional verification helper:

- `compare_tiny_numpy.py`

Existing inputs:

- `out-tiny-v2/model_fp32.pt`
- `out-tiny-v2/model_fp16.pt`
- `out-tiny-v2/model_int8.pt`
- `out-tiny-v2/model_int8.bin`
- `data/shakespeare_char/meta.pkl`
- `run_tiny.py`
- `tinytrain.py`

## Implementation Progress

Completed:

- [x] Task 1: Confirm Baseline
- [x] Task 2: Create The Numpy Runner Skeleton
- [x] Task 3: Implement The `.pt` Zip Reader
- [x] Task 4: Implement The Custom Pickle Unpickler
- [x] Task 5: Rebuild Tensors As Numpy Arrays
- [x] Task 6: Normalize And Validate The State Dict
- [x] Task 7: Implement Math Primitives
- [x] Task 8: Implement The Numpy GPT Forward Pass
- [x] Task 9: Implement Attention
- [x] Task 10: Implement MLP
- [x] Task 11: Implement Block Forward
- [x] Task 12: Implement Tokenization
- [x] Task 13: Implement Generation
- [x] Task 14: Add CLI Output
- [x] Task 15: Smoke Test The Runner
- [x] Task 16: Compare Checkpoint Loading Against Torch
- [x] Task 17: Compare Logits Against Torch
- [x] Task 18: Clean Up Implementation
- [x] Task 19: Optional Fp16 Support
- [x] Task 20: Optional Int8 `.pt` Support

Commits:

- `c205601` - Add numpy tiny GPT runner
- `af43ab1` - Add numpy runner comparison check
- `2bca7e5` - Track numpy inference implementation progress
- `2670687` - Support fp16 and int8 numpy checkpoints

Deferred optional work:

- [ ] Task 21: Optional Raw Int8 `.bin` Support

Verification completed:

- [x] `python3 run_tiny.py`
- [x] `python3 run_tiny_numpy.py --max-new-tokens 20`
- [x] `python3 run_tiny_numpy.py --ckpt out-tiny-v2/model_fp16.pt --max-new-tokens 20`
- [x] `python3 run_tiny_numpy.py --ckpt out-tiny-v2/model_int8.pt --max-new-tokens 20`
- [x] `python3 run_tiny_numpy.py --help`
- [x] `python3 compare_tiny_numpy.py`
- [x] `python3 compare_tiny_numpy.py --ckpt out-tiny-v2/model_fp16.pt --logit-atol 1e-4`
- [x] `python3 compare_tiny_numpy.py --ckpt out-tiny-v2/model_int8.pt --array-atol 1e-7 --logit-atol 1e-4`
- [x] `python3 -m py_compile run_tiny_numpy.py compare_tiny_numpy.py`

## Acceptance Criteria

- [x] `run_tiny_numpy.py` runs without importing `torch`.
- [x] The fp32 `.pt` checkpoint loads into numpy arrays.
- [x] The numpy model forward pass produces logits close to the Torch model for a fixed prompt.
- [x] Text generation works with the Shakespeare character tokenizer in `meta.pkl`.
- [x] The default CLI behavior matches the existing tiny runner settings.
- [x] Existing untracked files are not overwritten or reverted.

## Task 1: Confirm Baseline

- [ ] Run the existing Torch runner:

  ```bash
  python run_tiny.py
  ```

- [ ] Confirm it loads `out-tiny-v2/model_fp32.pt`.
- [ ] Confirm it loads `data/shakespeare_char/meta.pkl`.
- [ ] Record the default prompt:

  ```text
  to be or not to be,:
  ```

- [ ] Record the model config:

  ```python
  block_size = 64
  vocab_size = 65
  n_layer = 2
  n_head = 2
  n_embd = 16
  dropout = 0.0
  bias = False
  ```

- [ ] Record the generation settings:

  ```python
  max_new_tokens = 200
  temperature = 0.8
  top_k = 20
  ```

## Task 2: Create The Numpy Runner Skeleton

- [ ] Create `run_tiny_numpy.py`.
- [ ] Add imports:

  ```python
  import argparse
  import io
  import math
  import pickle
  import zipfile
  from collections import OrderedDict

  import numpy as np
  ```

- [ ] Add a small `GPTConfig` class or dataclass.
- [ ] Add a `main()` function.
- [ ] Add a CLI parser with these options:

  ```text
  --ckpt
  --meta
  --prompt
  --max-new-tokens
  --temperature
  --top-k
  --seed
  --n-head
  ```

- [ ] Set defaults to match `run_tiny.py`.

## Task 3: Implement The `.pt` Zip Reader

- [ ] Open the checkpoint with `zipfile.ZipFile`.
- [ ] Find the archive root prefix that contains `data.pkl`.
- [ ] Read `data.pkl` from the archive.
- [ ] Identify storage blob paths under the same prefix, usually like:

  ```text
  <root>/data/0
  <root>/data/1
  <root>/data/2
  ```

- [ ] Add a storage dtype map:

  ```python
  FloatStorage -> np.float32
  HalfStorage -> np.float16
  DoubleStorage -> np.float64
  LongStorage -> np.int64
  IntStorage -> np.int32
  ShortStorage -> np.int16
  CharStorage -> np.int8
  ByteStorage -> np.uint8
  BoolStorage -> np.bool_
  ```

- [ ] Support typed and untyped storage formats if the pickle stream uses either.

## Task 4: Implement The Custom Pickle Unpickler

- [ ] Create a `TorchCheckpointUnpickler` subclass of `pickle.Unpickler`.
- [ ] Override `persistent_load`.
- [ ] Intercept storage persistent IDs shaped like:

  ```python
  ("storage", storage_type, storage_key, location, numel)
  ```

- [ ] Load the referenced storage blob from the zip archive.
- [ ] Convert raw storage bytes into a numpy array with the correct dtype.
- [ ] Return a small storage wrapper containing:

  ```python
  array
  dtype
  ```

- [ ] Override `find_class`.
- [ ] Map `torch._utils._rebuild_tensor_v2` to a local tensor rebuild function.
- [ ] Map `torch._utils._rebuild_parameter` to a local parameter rebuild function.
- [ ] Allow safe stdlib classes such as `collections.OrderedDict`.
- [ ] Reject or stub unsupported Torch runtime classes with a clear error.

## Task 5: Rebuild Tensors As Numpy Arrays

- [ ] Implement local `_rebuild_tensor_v2`.
- [ ] Accept arguments equivalent to:

  ```python
  storage, storage_offset, size, stride, requires_grad, backward_hooks
  ```

- [ ] Convert pickle `size` and `stride` values to tuples of ints.
- [ ] Build the numpy view:

  ```python
  arr = np.ndarray(
      shape=size,
      dtype=storage.dtype,
      buffer=storage.array,
      offset=storage_offset * storage.dtype.itemsize,
      strides=tuple(s * storage.dtype.itemsize for s in stride),
  )
  ```

- [ ] Return `arr.copy()` so tensors are detached from the storage buffer.
- [ ] Implement `_rebuild_parameter` as a pass-through to the underlying array.

## Task 6: Normalize And Validate The State Dict

- [ ] Return the top-level loaded object as an `OrderedDict[str, np.ndarray]`.
- [ ] Strip `_orig_mod.` prefixes if present.
- [ ] Validate expected top-level keys:

  ```text
  transformer.wte.weight
  transformer.wpe.weight
  transformer.ln_f.weight
  lm_head.weight
  ```

- [ ] Validate expected per-layer keys for each layer:

  ```text
  transformer.h.<i>.ln_1.weight
  transformer.h.<i>.attn.c_attn.weight
  transformer.h.<i>.attn.c_proj.weight
  transformer.h.<i>.ln_2.weight
  transformer.h.<i>.mlp.c_fc.weight
  transformer.h.<i>.mlp.c_proj.weight
  ```

- [ ] Gracefully support optional bias tensors if present.
- [ ] Print or assert useful shape information while developing.
- [ ] Remove noisy debug output before finishing.

## Task 7: Implement Math Primitives

- [ ] Implement `softmax(x, axis=-1)` with max subtraction.
- [ ] Implement `layer_norm(x, weight, bias=None, eps=1e-5)`.
- [ ] Implement `gelu(x)`.
- [ ] Check GELU parity against Torch if logits drift too much.
- [ ] Implement `linear(x, weight, bias=None)` using `x @ weight.T`.

## Task 8: Implement The Numpy GPT Forward Pass

- [ ] Create a `NumpyGPT` class.
- [ ] Store `config` and `state`.
- [ ] Implement token embedding lookup:

  ```python
  tok_emb = state["transformer.wte.weight"][idx]
  ```

- [ ] Implement position embedding lookup:

  ```python
  pos_emb = state["transformer.wpe.weight"][np.arange(T)]
  ```

- [ ] Add token and position embeddings.
- [ ] Skip dropout at inference.
- [ ] Loop over transformer blocks.
- [ ] Apply final layer norm.
- [ ] Compute logits for only the last position.

## Task 9: Implement Attention

- [ ] Implement `attention(x, layer_index)`.
- [ ] Apply `attn.c_attn` projection.
- [ ] Split projection into `q`, `k`, and `v`.
- [ ] Reshape each into heads:

  ```python
  q = q.reshape(B, T, n_head, head_size).transpose(0, 2, 1, 3)
  k = k.reshape(B, T, n_head, head_size).transpose(0, 2, 1, 3)
  v = v.reshape(B, T, n_head, head_size).transpose(0, 2, 1, 3)
  ```

- [ ] Compute scaled dot-product attention.
- [ ] Apply a causal lower-triangular mask.
- [ ] Apply softmax on the last axis.
- [ ] Multiply attention probabilities by `v`.
- [ ] Merge heads back to `(B, T, C)`.
- [ ] Apply `attn.c_proj`.
- [ ] Skip attention and residual dropout at inference.

## Task 10: Implement MLP

- [ ] Implement `mlp(x, layer_index)`.
- [ ] Apply `mlp.c_fc`.
- [ ] Apply GELU.
- [ ] Apply `mlp.c_proj`.
- [ ] Skip dropout at inference.

## Task 11: Implement Block Forward

- [ ] Implement `block(x, layer_index)`.
- [ ] Apply first residual path:

  ```python
  x = x + attention(layer_norm(x, ln_1_weight, ln_1_bias), layer_index)
  ```

- [ ] Apply second residual path:

  ```python
  x = x + mlp(layer_norm(x, ln_2_weight, ln_2_bias), layer_index)
  ```

- [ ] Confirm tensor shapes remain `(B, T, C)` after each block.

## Task 12: Implement Tokenization

- [ ] Load `data/shakespeare_char/meta.pkl` with stdlib `pickle`.
- [ ] Read `stoi` and `itos`.
- [ ] Implement `encode(text)`.
- [ ] Implement `decode(ids)`.
- [ ] Fail clearly if the prompt contains a character not in `stoi`.

## Task 13: Implement Generation

- [ ] Implement `generate(model, idx, max_new_tokens, temperature, top_k, rng)`.
- [ ] Crop context to `config.block_size`.
- [ ] Run the model forward.
- [ ] Scale logits by temperature.
- [ ] Apply optional top-k filtering with `np.partition`.
- [ ] Convert logits to probabilities with softmax.
- [ ] Sample the next token with `rng.choice`.
- [ ] Append the sampled token to the running context.
- [ ] Return the complete token sequence.

## Task 14: Add CLI Output

- [ ] Print the prompt plus generated continuation.
- [ ] Avoid noisy checkpoint internals in normal output.
- [ ] Add a helpful error if the checkpoint path is missing.
- [ ] Add a helpful error if `meta.pkl` is missing.
- [ ] Add a helpful error if `--n-head` is incompatible with `n_embd`.

## Task 15: Smoke Test The Runner

- [ ] Run:

  ```bash
  python run_tiny_numpy.py --max-new-tokens 20
  ```

- [ ] Confirm Torch is not imported.
- [ ] Confirm generation completes.
- [ ] Confirm output decodes to readable character text.
- [ ] Confirm context cropping works for prompts longer than `block_size`.

## Task 16: Compare Checkpoint Loading Against Torch

- [ ] Create `compare_tiny_numpy.py` or a temporary comparison snippet.
- [ ] Load the same checkpoint with Torch:

  ```python
  torch.load("out-tiny-v2/model_fp32.pt", map_location="cpu", weights_only=True)
  ```

- [ ] Load the checkpoint with `load_torch_state_dict`.
- [ ] Compare every shared key.
- [ ] Report:

  ```text
  key
  shape
  dtype
  max_abs_err
  ```

- [ ] Expect exact equality or near-zero error for fp32.

## Task 17: Compare Logits Against Torch

- [ ] Build the Torch model using the same config.
- [ ] Build the numpy model using the same state dict.
- [ ] Encode a fixed prompt:

  ```text
  to be or not to be,:
  ```

- [ ] Run both models in inference mode.
- [ ] Compare final-position logits.
- [ ] Report:

  ```text
  max_abs_err
  mean_abs_err
  torch_argmax
  numpy_argmax
  torch_top5
  numpy_top5
  ```

- [ ] Acceptance target:

  ```text
  max_abs_err < 1e-4
  argmax token matches
  top-5 tokens mostly match
  ```

- [ ] If mismatch is large, check:

  ```text
  linear weight transpose
  GELU exactness
  causal mask shape
  layer norm epsilon
  tied lm_head/wte handling
  dtype casts
  ```

## Task 18: Clean Up Implementation

- [ ] Remove temporary debug prints.
- [ ] Keep comments only where they clarify tricky checkpoint loading.
- [ ] Keep the model code close to `model.py`.
- [ ] Ensure `run_tiny_numpy.py` remains standalone.
- [ ] Do not modify `model.py` unless a real bug is discovered.
- [ ] Do not modify existing training artifacts.

## Task 19: Optional Fp16 Support

- [ ] Confirm `HalfStorage` maps to `np.float16`.
- [ ] Load `out-tiny-v2/model_fp16.pt`.
- [ ] Cast weights to `np.float32` before numerically sensitive operations.
- [ ] Smoke test generation.
- [ ] Compare logits against a Torch fp16 or fp32 reference.

## Task 20: Optional Int8 `.pt` Support

- [ ] Inspect `out-tiny-v2/model_int8.pt`.
- [ ] Confirm each state entry is shaped like:

  ```python
  {"q": int8_tensor, "scale": fp32_tensor}
  ```

- [ ] Extend the loader if needed to support nested checkpoint values.
- [ ] Reconstruct each weight:

  ```python
  weight = q.astype(np.float32) * scale
  ```

- [ ] Run smoke generation.
- [ ] Compare logits against the fp32 model and record expected quantization drift.

## Task 21: Optional Raw Int8 `.bin` Support

- [ ] Do not attempt to load `model_int8.bin` alone as a first pass.
- [ ] Add a manifest export to `tinytrain.py` if raw int8 loading is required.
- [ ] Manifest should include:

  ```text
  key
  shape
  dtype
  scale
  byte offset
  byte length
  ```

- [ ] Load raw int8 bytes using the manifest.
- [ ] Reconstruct weights with `q * scale`.
- [ ] Smoke test and compare logits.

## Suggested Execution Order

- [ ] Task 1: Confirm Baseline
- [ ] Task 2: Create The Numpy Runner Skeleton
- [ ] Task 3: Implement The `.pt` Zip Reader
- [ ] Task 4: Implement The Custom Pickle Unpickler
- [ ] Task 5: Rebuild Tensors As Numpy Arrays
- [ ] Task 6: Normalize And Validate The State Dict
- [ ] Task 7: Implement Math Primitives
- [ ] Task 8: Implement The Numpy GPT Forward Pass
- [ ] Task 9: Implement Attention
- [ ] Task 10: Implement MLP
- [ ] Task 11: Implement Block Forward
- [ ] Task 12: Implement Tokenization
- [ ] Task 13: Implement Generation
- [ ] Task 14: Add CLI Output
- [ ] Task 15: Smoke Test The Runner
- [ ] Task 16: Compare Checkpoint Loading Against Torch
- [ ] Task 17: Compare Logits Against Torch
- [ ] Task 18: Clean Up Implementation
- [ ] Task 19: Optional Fp16 Support
- [ ] Task 20: Optional Int8 `.pt` Support
- [ ] Task 21: Optional Raw Int8 `.bin` Support

## Known Risks

- [ ] PyTorch checkpoint pickle internals may differ across versions.
- [ ] Typed storage and untyped storage may need separate handling.
- [ ] GELU exact-vs-approx behavior can cause small logit differences.
- [ ] Linear weights must be transposed because PyTorch stores them as `(out_features, in_features)`.
- [ ] The raw int8 `.bin` file is not self-describing without a manifest.

## Done Definition

- [ ] `python run_tiny_numpy.py` generates text using the fp32 checkpoint.
- [ ] The runtime path imports numpy but not Torch.
- [ ] Checkpoint arrays match Torch-loaded arrays.
- [ ] Fixed-prompt logits match Torch within tolerance.
- [ ] The CLI is documented by `--help`.
- [ ] Optional fp16/int8 work is either completed or explicitly left as future work.
