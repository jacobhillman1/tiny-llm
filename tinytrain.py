"""
tiny_train.py — train the smallest possible nanoGPT, strip, quantize, measure.

Place this in the root of your nanoGPT clone (same dir as model.py).
Prereq: python data/shakespeare_char/prepare.py

Run:   python tiny_train.py
"""

import os
import pickle
import time
import numpy as np
import torch
from model import GPT, GPTConfig

# ---------------------------------------------------------------------------
# KNOBS — shrink these to shrink the model. These target ~10KB territory.
# ---------------------------------------------------------------------------
N_LAYER    = 2       # transformer blocks
N_HEAD     = 2       # attention heads (n_embd must be divisible by this)
N_EMBD     = 16      # embedding / hidden dim
BLOCK_SIZE = 64      # context length
DROPOUT    = 0.0
BIAS       = False   # turning off bias saves a few params
BATCH_SIZE = 32
MAX_ITERS  = 3000
LR         = 1e-3
EVAL_EVERY = 500
DEVICE     = "cpu"   # "cuda" / "mps" if you have it
OUT_DIR    = "out-tiny-v2"
DATA_DIR   = "data/shakespeare_char"
# ---------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---- load data -------------------------------------------------------------
train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
val_data   = np.memmap(os.path.join(DATA_DIR, "val.bin"),   dtype=np.uint16, mode="r")
with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
VOCAB_SIZE = meta["vocab_size"]
print(f"vocab_size = {VOCAB_SIZE}")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ---- build model -----------------------------------------------------------
cfg = GPTConfig(
    block_size=BLOCK_SIZE,
    vocab_size=VOCAB_SIZE,
    n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
    dropout=DROPOUT, bias=BIAS,
)
model = GPT(cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters = {n_params:,}")

opt = torch.optim.AdamW(model.parameters(), lr=LR)

# ---- train -----------------------------------------------------------------
t0 = time.time()
for it in range(MAX_ITERS):
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if it % EVAL_EVERY == 0 or it == MAX_ITERS - 1:
        model.eval()
        with torch.no_grad():
            vx, vy = get_batch("val")
            _, vloss = model(vx, vy)
        model.train()
        print(f"iter {it:5d} | train {loss.item():.3f} | val {vloss.item():.3f}")
print(f"training took {time.time()-t0:.1f}s")

# ---- save stages & measure -------------------------------------------------
def fsize(path):
    return os.path.getsize(path)

# Stage 1: full checkpoint (model + optimizer + config) — what nanoGPT normally saves
full_ckpt = {
    "model": model.state_dict(),
    "optimizer": opt.state_dict(),
    "model_args": cfg.__dict__,
}
p_full = os.path.join(OUT_DIR, "ckpt_full.pt")
torch.save(full_ckpt, p_full)

# Stage 2: weights only, fp32
p_fp32 = os.path.join(OUT_DIR, "model_fp32.pt")
torch.save(model.state_dict(), p_fp32)

# Stage 3: weights only, fp16
sd_fp16 = {k: v.half() for k, v in model.state_dict().items()}
p_fp16 = os.path.join(OUT_DIR, "model_fp16.pt")
torch.save(sd_fp16, p_fp16)

# Stage 4: int8 quantized — per-tensor symmetric quantization
#   store int8 weights + one fp32 scale per tensor. Reconstruction: w ≈ q * scale.
sd_int8 = {}
for k, v in model.state_dict().items():
    v = v.float()
    if v.numel() == 0:
        sd_int8[k] = {"q": v.to(torch.int8), "scale": torch.tensor(1.0)}
        continue
    scale = v.abs().max() / 127.0
    if scale == 0:
        scale = torch.tensor(1.0)
    q = torch.round(v / scale).clamp(-127, 127).to(torch.int8)
    sd_int8[k] = {"q": q, "scale": scale.float()}
p_int8 = os.path.join(OUT_DIR, "model_int8.pt")
torch.save(sd_int8, p_int8)

# Stage 5: raw int8 bytes — no pickle overhead, just concatenated weight bytes.
#   This is the "true" minimum for shipping; you'd need a sidecar shape/scale manifest.
raw_bytes = b"".join(sd_int8[k]["q"].cpu().numpy().tobytes() for k in sd_int8)
p_raw = os.path.join(OUT_DIR, "model_int8.bin")
with open(p_raw, "wb") as f:
    f.write(raw_bytes)

# ---- report ----------------------------------------------------------------
print("\n--- size report ---")
rows = [
    ("full ckpt (model+optim+meta)", p_full),
    ("weights fp32 (.pt)",           p_fp32),
    ("weights fp16 (.pt)",           p_fp16),
    ("weights int8 (.pt, pickled)",  p_int8),
    ("weights int8 (.bin, raw)",     p_raw),
]
for label, path in rows:
    b = fsize(path)
    if b < 1024:
        s = f"{b} B"
    elif b < 1024*1024:
        s = f"{b/1024:.1f} KB"
    else:
        s = f"{b/(1024*1024):.2f} MB"
    print(f"  {label:35s} {s:>12s}  ({b} bytes)")

print(f"\nparameters: {n_params:,}")
print(f"theoretical minimum (1 byte/param): {n_params} B ≈ {n_params/1024:.2f} KB")

# ---- quick sample ----------------------------------------------------------
print("\n--- sample from trained model ---")
itos = meta["itos"]
model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
with torch.no_grad():
    out = model.generate(ctx, max_new_tokens=200, temperature=0.8, top_k=20)
print("".join(itos[i] for i in out[0].tolist()))