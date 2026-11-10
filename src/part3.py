"""Part 3: data preparation, training and evaluation for topic classification.

Section 6  every article -> first 256 token ids of cleaned.txt (padded with
           <PAD>), label from topics.py, 70/15/15 split stratified by category.
Section 7  TransformerClassifier trained with AdamW (5e-4, wd 0.01), cosine
           schedule with 50 warm-up steps, 20 epochs.
Section 8  BiLSTM classifier on the *same* split for the comparison.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from corpus import encode, load_cleaned_articles  # noqa: E402
from metrics import accuracy, confusion_matrix, macro_f1  # noqa: E402
from topics import assign_categories  # noqa: E402
from transformer import TransformerClassifier, cosine_with_warmup  # noqa: E402


# ----------------------------------------------------------------------------
# Section 6: data
# ----------------------------------------------------------------------------
def build_dataset(w2i: dict, max_len: int = C.MAX_LEN, seed: int = C.SEED) -> dict:
    pad = w2i[C.PAD]
    arts = load_cleaned_articles()
    labels = np.array([r["label_id"] for r in assign_categories()])
    X = np.full((len(arts), max_len), pad, dtype=np.int64)
    lengths = np.zeros(len(arts), dtype=np.int64)
    for i, art in enumerate(arts):
        ids = encode([t for s in art for t in s], w2i)[:max_len]     # truncate
        X[i, :len(ids)] = ids                                         # pad
        lengths[i] = len(ids)
    rng = np.random.RandomState(seed)
    split = np.empty(len(arts), dtype=object)
    for c in range(len(C.CATEGORIES)):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_tr = int(round(0.70 * len(idx)))
        n_va = int(round(0.15 * len(idx)))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"
    return {"X": X, "y": labels, "lengths": lengths, "split": split, "pad": pad,
            "full_lengths": np.array([sum(len(s) for s in a) for a in arts])}


def subset(data: dict, name: str):
    m = data["split"] == name
    return (torch.from_numpy(data["X"][m]), torch.from_numpy(data["y"][m]),
            torch.from_numpy(data["lengths"][m]), np.where(m)[0])


# ----------------------------------------------------------------------------
# Section 8: BiLSTM classifier used for the comparison
# ----------------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    """W2V-initialised embeddings -> 2-layer BiLSTM (dropout 0.5 between layers)
    -> masked mean + max pooling over time -> MLP (512 -> 64 -> 5)."""

    def __init__(self, emb_matrix: np.ndarray, pad_idx: int, n_classes: int = 5,
                 hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.as_tensor(emb_matrix, dtype=torch.float32),
                                                freeze=False, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_matrix.shape[1], hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(4 * hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, n_classes))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.emb(x), lengths.clamp(min=1).cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(2)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(~mask, -1e9).max(1).values
        return self.head(torch.cat([mean, mx], dim=1))


# ----------------------------------------------------------------------------
# Training (shared loop)
# ----------------------------------------------------------------------------
def _forward(model, x, lengths):
    return model(x, lengths) if isinstance(model, BiLSTMClassifier) else model(x)


@torch.no_grad()
def predict(model, X, lengths, batch: int = 64):
    model.eval()
    out = []
    for b in range(0, len(X), batch):
        out.append(_forward(model, X[b:b + batch], lengths[b:b + batch]))
    return torch.cat(out)


def evaluate(model, X, y, lengths) -> dict:
    logits = predict(model, X, lengths)
    pred = logits.argmax(1)
    g, p = y.tolist(), pred.tolist()
    return {"loss": F.cross_entropy(logits, y).item(), "accuracy": 100 * accuracy(g, p),
            "macro_f1": 100 * macro_f1(g, p, list(range(len(C.CATEGORIES)))),
            "confusion": confusion_matrix(g, p, list(range(len(C.CATEGORIES)))),
            "pred": p}


def train_classifier(model, data: dict, kind: str, epochs: int, lr: float, weight_decay: float,
                     batch_size: int, seed: int = C.SEED, verbose: bool = True) -> dict:
    """kind = 'transformer' (AdamW + cosine/warm-up) or 'bilstm' (Adam, constant LR)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    Xtr, ytr, Ltr, _ = subset(data, "train")
    Xva, yva, Lva, _ = subset(data, "val")
    steps_per_epoch = int(np.ceil(len(Xtr) / batch_size))
    if kind == "transformer":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = cosine_with_warmup(opt, C.TF_WARMUP, epochs * steps_per_epoch)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = None
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1",
                            "epoch_seconds", "lr")}
    best = {"acc": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(Xtr))
        tot, correct, n = 0.0, 0, 0
        for b in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[b:b + batch_size])
            logits = _forward(model, Xtr[idx], Ltr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                hist["lr"].append(sched.get_last_lr()[0])
                sched.step()
            tot += loss.item() * len(idx)
            correct += (logits.argmax(1) == ytr[idx]).sum().item()
            n += len(idx)
        hist["epoch_seconds"].append(time.time() - t0)
        hist["train_loss"].append(tot / n)
        hist["train_acc"].append(100 * correct / n)
        ev = evaluate(model, Xva, yva, Lva)
        hist["val_loss"].append(ev["loss"])
        hist["val_acc"].append(ev["accuracy"])
        hist["val_f1"].append(ev["macro_f1"])
        if ev["accuracy"] > best["acc"] or (ev["accuracy"] == best["acc"] and ev["loss"] < best["loss"]):
            best = {"acc": ev["accuracy"], "loss": ev["loss"], "epoch": ep,
                    "state": copy.deepcopy(model.state_dict())}
        if verbose:
            print(f"  [{kind}] epoch {ep:2d}  train loss {hist['train_loss'][-1]:.4f} "
                  f"acc {hist['train_acc'][-1]:5.1f}  |  val loss {ev['loss']:.4f} "
                  f"acc {ev['accuracy']:5.1f} macro-F1 {ev['macro_f1']:5.1f}  "
                  f"({hist['epoch_seconds'][-1]:.1f}s)")
    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best["state"])
    return {"model": model, "history": hist, "best_epoch": best["epoch"],
            "best_val_acc": best["acc"], "final_state": final_state}


def make_transformer(w2i: dict) -> TransformerClassifier:
    return TransformerClassifier(vocab_size=len(w2i), n_classes=len(C.CATEGORIES),
                                 pad_idx=w2i[C.PAD], d_model=C.D_MODEL, n_heads=C.N_HEADS,
                                 d_k=C.D_K, d_v=C.D_V, d_ff=C.D_FF, n_layers=C.N_LAYERS,
                                 dropout=C.TF_DROPOUT, max_len=C.MAX_LEN + 1)
