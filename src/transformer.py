import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, head_size, block_size, dropoutHead=0.1, dropoutMultiHead=0.1):
        super().__init__()
        self.headList = nn.ModuleList([Head(n_embd, head_size, block_size, dropoutHead) for _ in range(num_heads)])
        self.w_o = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropoutMultiHead)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.headList], dim=-1)
        out = self.w_o(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropoutHead=0.1, dropoutMultiHead=0.1, dropoutFF=0.1):
        super().__init__()
        self.layerNorm1 = nn.LayerNorm(n_embd)
        self.mma1 = MultiHeadAttention(num_heads=num_heads, n_embd=n_embd, head_size=n_embd//num_heads, block_size=block_size, dropoutHead=dropoutHead, dropoutMultiHead=dropoutMultiHead)
        self.layerNorm2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropoutFF)

    def forward(self, x):
        x = x + self.mma1(self.layerNorm1(x)) # Pre-norm
        x = x + self.ffwd(self.layerNorm2(x))
        return x
    
class HeteronimosTransformer(nn.Module):
    def __init__(self, length, n_embd, block_size, num_heads, num_blocks=6):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(length, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd=n_embd, num_heads=num_heads, block_size=block_size) for _ in range(num_blocks)])
        self.layerNorm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, length)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = token_emb + pos_emb  # (B, T, n_embd)
        
        for block in self.blocks:
            x = block(x)

        x = self.layerNorm(x)  # (B, T, n_embd) 
        logits = self.lm_head(x)  # (B, T, length)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, temperature=1.0, top_k=None, stop_sequence=None, max_tokens=2000):
        generated = 0
        while generated < max_tokens:
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            generated += 1

            if stop_sequence:
                n = len(stop_sequence)
                if idx.shape[1] >= n and idx[0, -n:].tolist() == stop_sequence:
                    break

        return idx