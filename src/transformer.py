"""Part 3, Section 7: Transformer encoder for topic classification, from scratch.

No nn.Transformer / nn.MultiheadAttention / nn.TransformerEncoder is used:
every component below is its own self-contained module.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    `mask` (B, 1, T_k) or (B, T_q, T_k): True = may attend, False = padding.
    Returns (output, attention weights)."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # (B, T_q, T_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return self.dropout(weights) @ v, weights


class MultiHeadSelfAttention(nn.Module):
    """h heads, each with its *own* W_Q, W_K, W_V (d_model -> d_k / d_v),
    concatenated and mixed by one shared output projection W_O (h*d_v -> d_model)."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_k: int = 32, d_v: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.W_q = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(n_heads)])
        self.W_v = nn.ModuleList([nn.Linear(d_model, d_v) for _ in range(n_heads)])
        self.W_o = nn.Linear(n_heads * d_v, d_model)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, x, mask=None):
        outs, weights = [], []
        for h in range(self.n_heads):
            o, w = self.attention(self.W_q[h](x), self.W_k[h](x), self.W_v[h](x), mask)
            outs.append(o)
            weights.append(w)
        return self.W_o(torch.cat(outs, dim=-1)), torch.stack(weights, dim=1)  # (B, h, T, T)


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = W_2 Dropout(ReLU(W_1 x + b_1)) + b_2, inner dimension d_ff."""

    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(F.relu(self.lin1(x))))


class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).

    Stored with register_buffer: saved with the model, never trained."""

    def __init__(self, d_model: int = 128, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
        div = torch.pow(10000.0, two_i / d_model)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        self.register_buffer("pe", pe.unsqueeze(0))               # (1, max_len, d)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EncoderBlock(nn.Module):
    """Pre-LayerNorm block:
        x <- x + Dropout(MultiHead(LN(x)))
        x <- x + Dropout(FFN(LN(x)))"""

    def __init__(self, d_model=128, n_heads=4, d_k=32, d_v=32, d_ff=512, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, d_k, d_v, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        a, w = self.attn(self.ln1(x), mask)
        x = x + self.drop1(a)
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x, w


class TransformerClassifier(nn.Module):
    """[CLS] + tokens -> embeddings + sinusoidal PE -> 4 Pre-LN blocks ->
    final LayerNorm -> [CLS] representation -> MLP (128 -> 64 -> n_classes)."""

    def __init__(self, vocab_size: int, n_classes: int = 5, pad_idx: int = 0, d_model: int = 128,
                 n_heads: int = 4, d_k: int = 32, d_v: int = 32, d_ff: int = 512,
                 n_layers: int = 4, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)   # learned [CLS] vector
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_k, d_v, d_ff, dropout)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, n_classes))

    def forward(self, x, return_attention: bool = False):
        B = x.size(0)
        tok = self.embed(x) * math.sqrt(self.d_model)
        h = torch.cat([self.cls.expand(B, 1, -1), tok], dim=1)        # prepend [CLS]
        h = self.drop(self.pos(h))                                      # PE added before block 1
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device),
                          x != self.pad_idx], dim=1)                    # (B, T+1)
        mask = keep.unsqueeze(1)                                        # keys only
        attns = []
        for blk in self.blocks:
            h, w = blk(h, mask)
            attns.append(w)
        logits = self.head(self.ln_f(h)[:, 0])
        return (logits, attns) if return_attention else logits


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """LR multiplier: linear warm-up 0 -> 1 over `warmup_steps`, then cosine decay to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
