# Bigram strategy

import regex as re
import torch
import torch.nn as nn
from torch.nn import functional as F

with open("data/corpus_com_token.txt", 'r', encoding='utf-8') as f:
    text = f.read()

def replace_tokens(text):
    replacements = {
        '\xa0': ' ', 
        '\xad': '',  
        '\u2019': "'", 
        '\u201c': '"', 
        '\u201d': '"', 
        '\u2026': '...', 
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

def encode(s):
    ids = []
    for part in special_pattern.split(s):
        if part in dictionary:  # it's a special token
            ids.append(dictionary[part])
        else:
            ids.extend(dictionary[c] for c in part)
    return ids

def decode(l):
    return ''.join([reverse_dictionary[i] for i in l])

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, batch_size, block_size, train_size, text):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)   
        self.batch_size = batch_size
        self.block_size = block_size
        self.data = torch.tensor(encode(text), dtype=torch.long)
        self.train_data = self.data[:int(train_size * len(self.data))]
        self.val_data = self.data[int(train_size * len(self.data)):]
        self.train_losses = []
        self.val_losses = []

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        
        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y
    
    @torch.no_grad()
    def estimate_loss(self, eval_iters=100):
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = self.get_batch(split)
                logits, loss = self(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out
    
    def train_loop(self, max_iters=20000, eval_interval=100, learning_rate=1e-3):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = self.estimate_loss()
                self.train_losses.append(losses['train'])
                self.val_losses.append(losses['val'])
                print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
            
            xb, yb = self.get_batch('train')
            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
        return self.train_losses, self.val_losses 

text = replace_tokens(text)
chars = sorted(list(set(text)))
chars.remove('<') # Remove < and > as theu only appear in the special tokens
chars.remove('>')

print(f"Unique characters: {''.join(chars)}")

# Create dictionaries for character to index and vice-versa
dictionary = {c: i for i, c in enumerate(chars)}
reverse_dictionary = {i: c for i, c in enumerate(chars)}    

special_tokens = ['<CAEIRO>', '<CAMPOS>', '<REIS>']
for token in special_tokens:
    dictionary[token] = len(dictionary)
    reverse_dictionary[len(reverse_dictionary)] = token

print(f"Dictionary: {dictionary}")

special_pattern = re.compile('(' + '|'.join(re.escape(t) for t in special_tokens) + ')')

vocab_size = len(chars) + len(special_tokens)

print(f"Vocabulary size: {vocab_size}")

batch_size = 64
block_size = 256
train_size = 0.9

model = BigramLanguageModel(vocab_size, batch_size, block_size, train_size, text)
print(f"Model initialized with vocab size {vocab_size}, batch size {batch_size}, block size {block_size}, and train size {train_size}")
train_losses, val_losses = model.train_loop(max_iters=10000, eval_interval=1000, learning_rate=1e-3)

print(f"Train Losses: {train_losses}")
print(f"Validation Losses: {val_losses}")

idx = torch.zeros((1, 1), dtype=torch.long)
generated_idx = model.generate(idx, max_new_tokens=250)
print(decode(generated_idx[0].tolist()))