"""Classification metrics implemented with numpy (checked against sklearn in tests)."""
from __future__ import annotations

import numpy as np


def confusion_matrix(gold: list, pred: list, labels: list) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for g, p in zip(gold, pred):
        M[idx[g], idx[p]] += 1
    return M


def accuracy(gold: list, pred: list) -> float:
    return float(np.mean([g == p for g, p in zip(gold, pred)])) if gold else 0.0


def per_class_prf(gold: list, pred: list, labels: list) -> dict[str, dict[str, float]]:
    M = confusion_matrix(gold, pred, labels)
    out = {}
    for i, l in enumerate(labels):
        tp = M[i, i]
        fp = M[:, i].sum() - tp
        fn = M[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[l] = {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(M[i].sum())}
    return out


def macro_f1(gold: list, pred: list, labels: list | None = None) -> float:
    """Unweighted mean F1 over `labels` (default: every label seen in gold or pred,
    the same convention as sklearn's f1_score(average='macro'))."""
    if labels is None:
        labels = sorted(set(gold) | set(pred))
    prf = per_class_prf(gold, pred, labels)
    return float(np.mean([prf[l]["f1"] for l in labels]))


def most_confused_pairs(M: np.ndarray, labels: list, k: int = 3) -> list[tuple[str, str, int]]:
    """Unordered tag pairs with the largest number of confusions in both directions."""
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((labels[i], labels[j], int(M[i, j] + M[j, i])))
    return sorted(pairs, key=lambda x: -x[2])[:k]
