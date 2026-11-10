"""Evaluation of word embeddings: neighbours, analogies, MRR, t-SNE plots."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from part1_counts import CosineIndex  # noqa: E402


def analogy(index: CosineIndex, a: str, b: str, c: str, k: int = 3) -> list[tuple[str, float]]:
    """a : b :: c : ?   answered with 3CosAdd on unit vectors: v(b) - v(a) + v(c).
    The three query words are excluded from the candidates."""
    q = index.vector(b) - index.vector(a) + index.vector(c)
    q = np.asarray(q.toarray() if hasattr(q, "toarray") else q).ravel()
    s = index.sims(q / max(np.linalg.norm(q), 1e-12))
    for w in (a, b, c):
        s[index.w2i[w]] = -np.inf
    s[index.mask] = -np.inf
    top = np.argpartition(-s, k)[:k]
    top = top[np.argsort(-s[top])]
    return [(index.i2w[j], float(s[j])) for j in top]


def run_analogies(index: CosineIndex, tests: list[tuple[str, str, str, str, str]], k: int = 3):
    """tests: (relation, a, b, c, expected). Returns rows with top-k and flags."""
    rows = []
    for rel, a, b, c, d in tests:
        cands = analogy(index, a, b, c, k)
        words = [w for w, _ in cands]
        rows.append({"relation": rel, "a": a, "b": b, "c": c, "expected": d,
                     "top": cands, "top1": words[0] == d, "top3": d in words})
    return rows


def mean_reciprocal_rank(index: CosineIndex, pairs: list[tuple[str, str]]) -> tuple[float, list[int]]:
    ranks = [index.rank_of(q, t) for q, t in pairs]
    return float(np.mean([1.0 / r for r in ranks])), ranks


def tsne_2d(X, seed: int = C.SEED, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.manifold import TSNE
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return TSNE(n_components=2, metric="cosine", init="pca", perplexity=perplexity,
                random_state=seed, max_iter=2000).fit_transform(X)


def plot_tsne(Y: np.ndarray, words: list[str], cats: list[str], title: str, ax=None):
    import matplotlib.pyplot as plt
    from plotting import PALETTE
    order = list(dict.fromkeys(cats))
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 11))
    for i, cat in enumerate(order):
        m = np.array([c == cat for c in cats])
        ax.scatter(Y[m, 0], Y[m, 1], s=38, color=PALETTE[i % len(PALETTE)],
                   label=f"{cat} ({m.sum()})", alpha=0.85, edgecolors="none")
    for (x, y), w in zip(Y, words):
        ax.annotate(w, (x, y), fontsize=8.5, xytext=(2, 2), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(title="Semantic category", loc="best", fontsize=9, framealpha=0.9)
    return ax
