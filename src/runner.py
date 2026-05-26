import os
import torch
import random
from contextlib import nullcontext
from transformer import HeteronimosTransformer
from tokenizer import BPETokenizer, CharTokenizer

# Hyperparameters and paths
MODEL_PATH  = "models/charModel_10M.pt"
OUT_PATH    = "out/poems.txt"
NUM_POEMS   = 2
TEMPERATURE = 0.8
TOP_K       = 100
WRITERS     = ['<CAEIRO>', '<CAMPOS>', '<REIS>', '<PESSOA>']

POEM_TITLE  = "A cor violeta"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CTX    = nullcontext() if DEVICE == 'cpu' else torch.amp.autocast(device_type=DEVICE, dtype=torch.float16)

# Load checkpoint
ck = torch.load(MODEL_PATH, map_location=DEVICE)

if ck.get("tokenizer_type") == "bpe":
    tokenizer = BPETokenizer(vocab_size=ck["vocab_size"])
    tokenizer.merges = {tuple(map(int, k.split(","))): v for k, v in ck["merges"].items()}
    tokenizer.vocab  = {int(k): bytes(v) for k, v in ck["vocab"].items()}
    tokenizer._build_special_token_maps()
else:
    tokenizer = CharTokenizer()
    tokenizer.ctoi = ck["stoi"]
    tokenizer.itoc = {int(k): v for k, v in ck["itoc"].items()}

ckg = ck["config"]
model = HeteronimosTransformer(
    length     = ckg["vocab_size"],
    n_embd     = ckg["n_embd"],
    num_heads  = ckg["num_heads"],
    num_blocks = ckg["num_blocks"],
    block_size = ckg["block_size"],
).to(DEVICE)
model.load_state_dict(ck["model"])
model.eval()

# Prepare stop sequence for generation
stop_ids = tokenizer.encode("\n---\n")
writers  = random.choices(WRITERS, k=NUM_POEMS)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Generate poems
with open(OUT_PATH, "w", encoding="utf-8") as f:
    with torch.no_grad():
        with CTX:
            for writer in writers:
                starting_text = writer + "\n" + POEM_TITLE + "\n"
                start_ids     = tokenizer.encode(starting_text)
                x             = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

                poem = model.generate(x, temperature=TEMPERATURE, top_k=TOP_K, stop_sequence=stop_ids)
                text = tokenizer.decode(poem[0].tolist())

                print(text)
                f.write(text + "\n")

print(f"\nPoems saved to {OUT_PATH}")