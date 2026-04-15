# Tiny GPT C Runtime

This directory contains a standalone C runtime for the tiny nanoGPT checkpoint.
It embeds the fp32 model weights directly in `tiny_gpt.c`, so it does not load
`data.pkl`, use pickle, import Python, or depend on PyTorch at runtime.

## Build

From `/Users/jacobhillman/projects/tiny-llm`:

```bash
cc -std=c99 -Wall -Wextra -O2 nanoGPT/generated/tiny_gpt.c -lm -o nanoGPT/generated/tiny_gpt
```

## Run

```bash
nanoGPT/generated/tiny_gpt "prompt text" [max_new_tokens] [temperature] [top_k] [seed]
```

Defaults:

```text
max_new_tokens = 200
temperature    = 0.8
top_k          = 20
seed           = 1337
```

Example:

```bash
nanoGPT/generated/tiny_gpt "to be" 40 0.8 20 1337
```

The prompt must use the checkpoint's 65-character Shakespeare vocabulary:
newline, space, punctuation from the training set, uppercase `A-Z`, and
lowercase `a-z`.

## Test

Run the built-in logits parity check:

```bash
nanoGPT/generated/tiny_gpt --logits-test
```

Expected result:

```text
argmax: 33
max_abs_err_first_10: about 1e-6
```

The model is specialized to this tiny checkpoint:

```text
block_size = 32
vocab_size = 65
n_layer = 1
n_head = 1
n_embd = 8
bias = false
dropout = 0.0
```
