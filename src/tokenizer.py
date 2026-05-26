import json
import regex as re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=250):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.special_tokens = ['<CAEIRO>', '<CAMPOS>', '<REIS>', '<PESSOA>']
        self._pattern = re.compile(
            '(' + '|'.join(re.escape(t) for t in self.special_tokens) + ')'
        )
        self._build_special_token_maps()

    def _build_special_token_maps(self):
        # Special tokens get IDs starting right after the BPE vocab
        self._special_token_to_id = {t: self.vocab_size + i for i, t in enumerate(self.special_tokens)}
        self._id_to_special_token = {self.vocab_size + i: t for i, t in enumerate(self.special_tokens)}

    def _get_most_common_pair(self, tokens):
        token_pair = Counter(zip(tokens, tokens[1:]))
        most_common = token_pair.most_common(1)[0][0]
        return most_common

    def _merge(self, tokens, pair, new_token):
        out = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                out.append(new_token)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    def train(self, text):
        num_merges = self.vocab_size - 256
        # Exclude special tokens from BPE training so they are never split into bytes
        parts = self._pattern.split(text)
        training_text = ''.join(p for p in parts if p not in self.special_tokens)
        tokens = list(training_text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            most_common_pair = self._get_most_common_pair(tokens)
            new_token = 256 + i
            tokens = self._merge(tokens, most_common_pair, new_token)
            self.merges[most_common_pair] = new_token
            self.vocab[new_token] = self.vocab[most_common_pair[0]] + self.vocab[most_common_pair[1]]

            if i % 100 == 0:
                print(f"Merge {i}/{num_merges} | Pair: {most_common_pair} | Unique: {len(set(tokens))}")

        self._build_special_token_maps()
        print(f"Training complete | Tokens: {len(tokens)} | Unique: {len(set(tokens))}")
        return self.encode(text)

    def encode(self, text):
        ids = []
        for part in self._pattern.split(text):
            if part in self._special_token_to_id:
                ids.append(self._special_token_to_id[part])
            else:
                tokens = list(part.encode("utf-8"))
                for pair, new_token in self.merges.items():
                    tokens = self._merge(tokens, pair, new_token)
                ids.extend(tokens)
        return ids

    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self._id_to_special_token:
                parts.append(self._id_to_special_token[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path):
        # merge keys are tuples — JSON doesn't support tuple keys, convert to string
        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{p[0]},{p[1]}": t for p, t in self.merges.items()},
            "vocab": {k: list(v) for k, v in self.vocab.items()}
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        self._build_special_token_maps()
        print(f"Tokenizer loaded from {path}")


class CharTokenizer:
    def __init__(self):
        self.ctoi = {}  # char → id
        self.itoc = {}  # id → char 
        self.special_tokens = ['<CAEIRO>', '<CAMPOS>', '<REIS>', '<PESSOA>']
        self.replacements = {
            '\xa0': ' ', 
            '\xad': '',  
            '\u2019': "'", 
            '\u201c': '"', 
            '\u201d': '"', 
            '\u2026': '...', 
        }
        self._pattern = re.compile(
            '(' + '|'.join(re.escape(t) for t in self.special_tokens) + ')'
        )

    def _build_vocab(self, text):
        chars = sorted(set(text))
        chars = [c for c in chars if c not in ('<', '>')]
        self.ctoi = {ch: i for i, ch in enumerate(chars)}
        self.itoc = {i: ch for i, ch in enumerate(chars)}
        for token in self.special_tokens:
            idx = len(self.ctoi)
            self.ctoi[token] = idx
            self.itoc[idx] = token

    def _replace_tokens(self, text):
        for old, new in self.replacements.items():
            text = text.replace(old, new)
        return text
    
    def encode(self, text):
        ids = []
        for part in self._pattern.split(text):
            if part in self.ctoi:         
                ids.append(self.ctoi[part])
            else:
                ids.extend(self.ctoi[c] for c in part if c in self.ctoi)
        return ids
    
    def decode(self, ids):
        return "".join(self.itoc[i] for i in ids)
    
    def train(self, text):
        text = self._replace_tokens(text)
        self._build_vocab(text)
        return self.encode(text)
    
    def save(self, path):
        data = {
            "ctoi": self.ctoi,
            "itoc": self.itoc
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Tokenizer saved in {path}")

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.ctoi = data["ctoi"]
        self.itoc = data["itoc"]
        print(f"Tokenizer loaded from {path}")