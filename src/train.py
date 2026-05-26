import os
import time
import torch
import torch.nn as nn
from transformer import HeteronimosTransformer
from tokenizer import BPETokenizer, CharTokenizer

CORPUS_PATH     = "data/corpus_com_token.txt"
CHECKPOINT_DIR  = "models"
CHECKPOINT_FILE = "bpe200_10M.pt"

# Tokenizer — "char" or "bpe"
TOKENIZER_TYPE  = "bpe"
BPE_VOCAB_SIZE  = 350   # only used when TOKENIZER_TYPE == "bpe"

# Architecture
N_EMBD          = 384   # embedding dimension
NUM_HEADS       = 6     # attention heads (384 / 6 = 64 per head)
NUM_BLOCKS      = 6     # layers of the transformer
BLOCK_SIZE      = 256   # maximum length of the context
DROPOUT         = 0.2   # dropout

# Training
BATCH_SIZE      = 64
MAX_ITERS       = 5000
EVAL_INTERVAL   = 500
EVAL_ITERS      = 200
LEARNING_RATE   = 3e-4
TRAIN_SPLIT     = 0.9

CHECKPOINTING   = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus: {len(text):,} caracteres")
    return text

def prepare_data(tokens):
    data = torch.tensor(tokens, dtype=torch.long)
    n    = int(TRAIN_SPLIT * len(data))
    return data[:n], data[n:]

def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix   = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x    = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y    = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# Evaluation
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    results = {}
    for split, _ in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results

# Checkpointing
def save_checkpoint(model, optimizer, step, losses, tokenizer):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)

    if isinstance(tokenizer, CharTokenizer):
        tokenizer_state = {
            "tokenizer_type": "char",
            "stoi": tokenizer.ctoi,
            "itoc": tokenizer.itoc,
        }
    else:
        tokenizer_state = {
            "tokenizer_type": "bpe",
            "merges": {f"{p[0]},{p[1]}": t for p, t in tokenizer.merges.items()},
            "vocab":  {k: list(v) for k, v in tokenizer.vocab.items()},
            "vocab_size": tokenizer.vocab_size,
        }

    torch.save({
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "losses":    losses,
        **tokenizer_state,
        "config": {
            "n_embd":     N_EMBD,
            "num_heads":  NUM_HEADS,
            "num_blocks": NUM_BLOCKS,
            "block_size": BLOCK_SIZE,
            "dropout":    DROPOUT,
            "vocab_size": len(tokenizer.ctoi) if isinstance(tokenizer, CharTokenizer) else tokenizer.vocab_size + len(tokenizer.special_tokens),
        },
    }, path)
    print(f"Checkpoint saved: {path}  (step {step})")

def load_checkpoint(path: str):
    print(f"Loading checkpoint: {path}")
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    return ck

# Training
def train(train_data, val_data, model, optimizer, tokenizer):
    start_step = 0
    all_losses = []

    if CHECKPOINTING:
        ck_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE)
        if os.path.exists(ck_path):
            ck = load_checkpoint(ck_path)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_step = ck["step"] + 1
            all_losses = ck.get("losses", [])
            print(f"Starting from step {start_step}")
        else:
            print(f"  Checkpoint not found at {ck_path} — starting from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters (~{total_params / 1e6:.2f}M)")
    print(f"Device: {DEVICE}")
    print(f"Tokenizer: {TOKENIZER_TYPE.upper()}")
    print(f"Training: {len(train_data):,} tokens  |  Validation: {len(val_data):,} tokens\n")

    model.train()
    t0 = time.time()

    for step in range(start_step, start_step + MAX_ITERS):
        # Periodic evaluation
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            all_losses.append({"step": step, **losses})
            elapsed = time.time() - t0
            print(f"Step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.1f}s")
            save_checkpoint(model, optimizer, step, all_losses, tokenizer)
            t0 = time.time()

        # Forward + backward
        xb, yb       = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping - estabiliza o treino
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Final checkpoint
    losses = estimate_loss(model, train_data, val_data)
    all_losses.append({"step": start_step + MAX_ITERS, **losses})
    print(f"\nFinal | train {losses['train']:.4f} | val {losses['val']:.4f}")
    save_checkpoint(model, optimizer, start_step + MAX_ITERS, all_losses, tokenizer)
    print("\nTraining completed.")

# Init
text = load_data(CORPUS_PATH)

if TOKENIZER_TYPE == "bpe":
    tokenizer = BPETokenizer(vocab_size=BPE_VOCAB_SIZE)
else:
    tokenizer = CharTokenizer()

tokens = tokenizer.train(text)
train_data, val_data = prepare_data(tokens)

vocab_size = tokenizer.vocab_size + len(tokenizer.special_tokens) if isinstance(tokenizer, BPETokenizer) else len(tokenizer.ctoi)
model = HeteronimosTransformer(length=vocab_size, n_embd=N_EMBD, block_size=BLOCK_SIZE, num_heads=NUM_HEADS, num_blocks=NUM_BLOCKS).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

train(train_data, val_data, model, optimizer, tokenizer)
