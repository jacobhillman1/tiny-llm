import pickle, torch
from model import GPT, GPTConfig

# Must match what you trained with
cfg = GPTConfig(
    block_size=64, vocab_size=65,
    n_layer=2, n_head=2, n_embd=16,
    dropout=0.0, bias=False,
)
model = GPT(cfg)
model.load_state_dict(torch.load("out-tiny-v2/model_fp32.pt", map_location="cpu", weights_only=True))
model.eval()

# Load the char <-> int mapping
with open("data/shakespeare_char/meta.pkl", "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

# Encode a prompt, generate, decode
prompt = "to be or not to be,:"
ctx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
with torch.no_grad():
    out = model.generate(ctx, max_new_tokens=200, temperature=0.8, top_k=20)
print("".join(itos[i] for i in out[0].tolist()))