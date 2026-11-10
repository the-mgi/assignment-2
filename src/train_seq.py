"""Training / evaluation loop for the Part 2 sequence labellers."""
from __future__ import annotations

import copy
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import conlleval  # noqa: E402
from metrics import macro_f1  # noqa: E402
from seq_models import BiLSTMTagger  # noqa: E402


@dataclass
class SeqConfig:
    task: str                      # "pos" or "ner"
    name: str = ""
    freeze: bool = False           # frozen vs fine-tuned embeddings
    pretrained: bool = True        # W2V (C3) initialisation vs random (ablation A3)
    bidirectional: bool = True     # ablation A1: False
    dropout: float = C.LSTM_DROPOUT  # ablation A2: 0.0
    use_crf: bool | None = None    # default: CRF for NER, softmax for POS
    hidden: int = C.LSTM_HIDDEN
    layers: int = C.LSTM_LAYERS
    lr: float = C.SEQ_LR
    weight_decay: float = C.SEQ_WEIGHT_DECAY
    batch_size: int = C.SEQ_BATCH
    max_epochs: int = C.SEQ_MAX_EPOCHS
    patience: int = C.SEQ_PATIENCE
    seed: int = C.SEED

    def __post_init__(self):
        if self.use_crf is None:
            self.use_crf = self.task == "ner"


@dataclass
class SeqData:
    tokens: list[list[str]]
    tags: list[list[str]]
    x: list[list[int]] = field(default_factory=list)
    y: list[list[int]] = field(default_factory=list)


def encode_data(tokens, tags, w2i, tag2i) -> SeqData:
    unk = w2i[C.UNK]
    d = SeqData(tokens, tags)
    d.x = [[w2i.get(t, unk) for t in s] for s in tokens]
    d.y = [[tag2i[t] for t in s] for s in tags]
    return d


def batches(d: SeqData, batch_size: int, pad_idx: int, shuffle: bool, rng=None):
    order = np.arange(len(d.x))
    if shuffle:
        rng.shuffle(order)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        lens = [len(d.x[i]) for i in idx]
        T = max(lens)
        x = torch.full((len(idx), T), pad_idx, dtype=torch.long)
        y = torch.full((len(idx), T), -100, dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :lens[r]] = torch.tensor(d.x[i])
            y[r, :lens[r]] = torch.tensor(d.y[i])
        lengths = torch.tensor(lens)
        mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
        yield idx, x, lengths, y, mask


def score(task: str, gold: list[list[str]], pred: list[list[str]]) -> float:
    """POS: token macro-F1 (%); NER: entity-level F1 (%) with conlleval."""
    if task == "pos":
        return 100.0 * macro_f1([t for s in gold for t in s], [t for s in pred for t in s])
    return conlleval.evaluate(gold, pred)["f1"]


@torch.no_grad()
def run_eval(model, d: SeqData, cfg: SeqConfig, pad_idx: int, i2tag: list[str]):
    model.eval()
    total, n = 0.0, 0
    preds: list = [None] * len(d.x)
    for idx, x, lengths, y, mask in batches(d, 64, pad_idx, shuffle=False):
        total += model.loss(x, lengths, y, mask).item() * len(idx)
        n += len(idx)
        for i, p in zip(idx, model.predict(x, lengths, mask)):
            preds[i] = [i2tag[k] for k in p]
    return total / n, preds, score(cfg.task, d.tags, preds)


def train(cfg: SeqConfig, train_d: SeqData, val_d: SeqData, tag_list: list[str],
          emb_matrix: np.ndarray, pad_idx: int, verbose: bool = True) -> dict:
    """Adam + early stopping on validation F1 (patience cfg.patience; ties in F1
    are broken by validation loss); the best validation epoch is restored."""
    torch.manual_seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)
    model = BiLSTMTagger(len(tag_list), vocab_size=emb_matrix.shape[0],
                         emb_dim=emb_matrix.shape[1],
                         pretrained=emb_matrix if cfg.pretrained else None,
                         freeze=cfg.freeze, pad_idx=pad_idx, hidden=cfg.hidden,
                         layers=cfg.layers, dropout=cfg.dropout,
                         bidirectional=cfg.bidirectional, use_crf=cfg.use_crf)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist = {"train_loss": [], "val_loss": [], "val_f1": [], "epoch_seconds": []}
    best_f1, best_loss, best_state, best_epoch, bad = -1.0, float("inf"), None, 0, 0
    for ep in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for idx, x, lengths, y, mask in batches(train_d, cfg.batch_size, pad_idx, True, rng):
            loss = model.loss(x, lengths, y, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        hist["epoch_seconds"].append(time.time() - t0)
        hist["train_loss"].append(tot / n)
        vl, _, vf = run_eval(model, val_d, cfg, pad_idx, tag_list)
        hist["val_loss"].append(vl)
        hist["val_f1"].append(vf)
        # improvement = higher val F1; an exact F1 tie (e.g. 0.0 while a rare-entity
        # NER model still predicts only "O") counts if the val loss went down
        if vf > best_f1 or (vf == best_f1 and vl < best_loss):
            best_f1, best_loss, best_epoch, bad = vf, vl, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose:
            print(f"  [{cfg.name or cfg.task}] epoch {ep:2d}  train loss {hist['train_loss'][-1]:.4f}  "
                  f"val loss {vl:.4f}  val F1 {vf:6.2f}{'  *' if bad == 0 else ''}")
        if bad >= cfg.patience:
            if verbose:
                print(f"  early stop: no val-F1 gain for {cfg.patience} epochs; best epoch {best_epoch}")
            break
    model.load_state_dict(best_state)
    return {"model": model, "history": hist, "best_epoch": best_epoch, "best_val_f1": best_f1,
            "config": asdict(cfg), "epochs_run": len(hist["train_loss"])}
