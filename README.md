# HeterónimosGPT

![Fernando Pessoa](assets/pessoa.jpg)

A character-level / BPE GPT trained to generate poetry in the style of Fernando Pessoa's heteronyms, Alberto Caeiro, Álvaro de Campos, Ricardo Reis, and Pessoa himself. The corpus is scraped from [Arquivo Pessoa](http://arquivopessoa.net).

This project is inspired by Andrej Karpathy series: https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=SlLZfng0okpWo4HI

The transformer was developed following the paper "Attention Is All You Need": https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

When training and testing both tokenizers, it can be concluded the CharTokenizer generates better text than the BPETokenizer. This can be due to the ammount of data being too small for the BPE to perform better, as it typically is a better option in more complex cases.

The default transformer architecture was chosen to train a ~10M parameters model.

## Setup

```bash
pip install torch requests beautifulsoup4 regex lingua-language-detector
```

## Pipeline

All commands are run from the project root.

### 1. Scrape

Scrapes poems from arquivopessoa.net by brute-forcing numeric IDs (1–4600).

```bash
# Recommended: 4 authors, verse only, Portuguese only
python src/scraper.py --authors CAEIRO CAMPOS REIS PESSOA --type verso --language pt

# Quick test on a small range
python src/scraper.py --start 1 --end 100 --authors CAEIRO --language pt

# All text types, no filters
python src/scraper.py --type ambos
```

Outputs to `data/`: per-author `.txt` files, `corpus_sem_token.txt`, `corpus_com_token.txt`, `metadata.json`.

### 2. Train

Edit the constants at the top of `src/train.py`, then run:

```bash
python src/train.py
```

Key settings:

| Variable | Default | Description |
|---|---|---|
| `TOKENIZER_TYPE` | `"char"` | `"char"` or `"bpe"` |
| `BPE_VOCAB_SIZE` | `350` | BPE vocab size (ignored for char) |
| `CHECKPOINT_FILE` | `"bpe200_10M.pt"` | Output filename in `models/` |
| `N_EMBD` | `384` | Embedding dimension |
| `NUM_HEADS` | `6` | Attention heads |
| `NUM_BLOCKS` | `6` | Transformer layers |
| `BLOCK_SIZE` | `256` | Context length |
| `MAX_ITERS` | `5000` | Training steps |

Training resumes automatically if the checkpoint file already exists.

### 3. Generate

```bash
python src/runner.py
```

Key settings in `src/runner.py`:

| Variable | Description |
|---|---|
| `MODEL_PATH` | Path to the checkpoint (e.g. `models/charModel_10M.pt`) |
| `NUM_POEMS` | Number of poems to generate |
| `POEM_TITLE` | Title used as prompt for all poems |
| `TEMPERATURE` | Sampling temperature (higher = more random) |
| `TOP_K` | Top-k sampling cutoff |

Generated poems are printed to stdout and saved to `out/poems.txt`. Each poem is randomly attributed to one of the four authors via its special token (`<CAEIRO>`, `<CAMPOS>`, `<REIS>`, `<PESSOA>`).

## Project Structure

```
src/
  scraper.py      - scrapes arquivopessoa.net
  tokenizer.py    - CharTokenizer and BPETokenizer (with special token support)
  transformer.py  - decoder-only transformer (Head, MultiHeadAttention, Block, HeteronimosTransformer)
  train.py        - training loop with checkpointing
  runner.py       - loads a checkpoint and generates poems
  bigram.py       - standalone bigram baseline (reference only)
data/
  corpus_com_token.txt   - full corpus with author tokens (used for training)
  corpus_sem_token.txt   - full corpus without author tokens
  metadata.json          - scraped poem metadata
models/                  - saved checkpoints and tokenizer files
out/                     - generated poem output
```

## Tokenizers

**CharTokenizer**: character-level. Vocabulary built from the corpus characters plus four special tokens. Simpler and faster to train; lower compression.

**BPETokenizer**: byte-pair encoding. Starts from 256 UTF-8 bytes and learns `vocab_size - 256` merges. Special tokens (`<CAEIRO>` etc.) are excluded from BPE training and assigned IDs starting at `vocab_size`, so they are always treated as atomic units.

The checkpoint stores `"tokenizer_type"` so `runner.py` restores the correct tokenizer automatically.
