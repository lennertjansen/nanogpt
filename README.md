# NanoGPT
**Disclaimer: this repo is strictly for my own educational/didactic purposes.**

A miniature version of OpenAI's GPT-2 (11M) for my own educational purposes.

Character-level GPT trained on tinyshakespeare, following Karpathy's "Let's build GPT" lecture and Vaswani et al. (2017).

The final model is a decoder-only transformer with multi-head causal self-attention, pre-norm, learned positional embeddings.

## Model

- 10.8M params: 6 layers, 6 heads, n_embd 384, context length 256, dropout 0.2
- vocab: 65 chars, batch size 64, AdamW lr 3e-4, 5000 iters

## Results

Trained on MacBook Pro 14" (M4 Pro, 48 GB), device `mps`.

| step | train loss | val loss |
|------|-----------|----------|
| 0    | 4.35      | 4.34     |
| 2500 | 1.22      | 1.51     |
| 4500 | 1.02      | 1.52     |

Val loss bottoms out ~1.49 around step 3000, then overfits slightly.

## Run

```
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
uv run train.py
```

Prints losses every 500 steps, then generates a 500-char sample.

## AI statement

No AI was used to write the code. Claude Code (Fable 5) drafted and edited this README. Cursor (GPT-5.6 (sol, medium)) reviewed and refined my handwritten notes and comments in the notebooks.
