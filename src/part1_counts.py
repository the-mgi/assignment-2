"""Part 1, Section 1: TF-IDF and PPMI weighted representations.

Vocabulary (shared by every part): the 10,000 most frequent tokens of
cleaned.txt get ids 0..9999, <UNK> = 10000 (all other tokens), <PAD> = 10001.
Count matrices use rows/columns 0..10000, i.e. |V| = 10,001 including <UNK>.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from corpus import encode, is_symbol  # noqa: E402

N_ROWS = C.VOCAB_SIZE + 1          # words + <UNK>


# ----------------------------------------------------------------------------
# 1.1 TF-IDF
# ----------------------------------------------------------------------------
def term_document_counts(articles: list[list[list[str]]], w2i: dict) -> np.ndarray:
    """Term-document count matrix, shape (|V| = 10,001, N documents).

    Entry [w, d] = number of times token w occurs in article d (TF)."""
    M = np.zeros((N_ROWS, len(articles)), dtype=np.float64)
    for d, art in enumerate(articles):
        ids = encode([t for s in art for t in s], w2i)
        M[:, d] = np.bincount(ids, minlength=N_ROWS)[:N_ROWS]
    return M


def tfidf(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TF-IDF(w, d) = TF(w, d) * log[ N / (1 + df(w)) ]   (natural log).

    Returns (tfidf matrix, idf vector, df vector)."""
    n_docs = counts.shape[1]
    df = (counts > 0).sum(axis=1)
    idf = np.log(n_docs / (1.0 + df))
    return counts * idf[:, None], idf, df


def top_words_per_category(tfidf_m: np.ndarray, doc_labels: list[int], i2w: list[str],
                           k: int = 10, min_docs: int = 2) -> dict[int, list[tuple[str, float]]]:
    """Most discriminative words of each category.

    Each document column is L2-normalised (so long articles do not dominate),
    then score(w, c) = mean TF-IDF of w over documents of c
                       - mean TF-IDF of w over all other documents.
    Candidates must appear in >= `min_docs` documents of the category (a word
    found in a single article describes that article, not the category) and
    must not be punctuation / placeholders.
    """
    X = tfidf_m / (np.linalg.norm(tfidf_m, axis=0, keepdims=True) + 1e-12)
    labels = np.asarray(doc_labels)
    out = {}
    for c in sorted(set(doc_labels)):
        inside, outside = labels == c, labels != c
        score = X[:, inside].mean(axis=1) - X[:, outside].mean(axis=1)
        support = (tfidf_m[:, inside] != 0).sum(axis=1)
        order = np.argsort(-score)
        picked = []
        for w in order:
            tok = i2w[w]
            if support[w] >= min_docs and not is_symbol(tok):
                picked.append((tok, float(score[w])))
            if len(picked) == k:
                break
        out[c] = picked
    return out


# ----------------------------------------------------------------------------
# 1.2 PPMI
# ----------------------------------------------------------------------------
def cooccurrence(sentences: list[list[str]], w2i: dict, window: int = C.PPMI_WINDOW) -> sp.csr_matrix:
    """Symmetric word-word co-occurrence counts.

    For every token, each other token at distance 1..window to its left or
    right inside the same sentence is counted once (so every pair at distance
    <= window contributes to both [a, b] and [b, a])."""
    rows, cols = [], []
    for s in sentences:
        ids = np.asarray(encode(s, w2i), dtype=np.int64)
        for d in range(1, window + 1):
            if len(ids) <= d:
                break
            a, b = ids[:-d], ids[d:]
            rows += [a, b]
            cols += [b, a]
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    data = np.ones(len(r), dtype=np.float64)
    return sp.coo_matrix((data, (r, c)), shape=(N_ROWS, N_ROWS)).tocsr()


def ppmi(cooc: sp.csr_matrix) -> sp.csr_matrix:
    """PPMI(w1, w2) = max{0, log2[ P(w1, w2) / (P(w1) P(w2)) ]} with
    P(w1, w2) = c(w1, w2)/T, P(w1) = row-sum/T, P(w2) = column-sum/T,
    T = sum of all counts. Zero counts give PPMI 0."""
    cooc = cooc.tocoo()
    total = cooc.sum()
    row = np.asarray(cooc.sum(axis=1)).ravel()
    col = np.asarray(cooc.sum(axis=0)).ravel()
    pmi = np.log2(cooc.data * total / (row[cooc.row] * col[cooc.col]))
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep].astype(np.float32), (cooc.row[keep], cooc.col[keep])),
                         shape=cooc.shape)


# ----------------------------------------------------------------------------
# Nearest neighbours (works for sparse PPMI rows and dense embeddings)
# ----------------------------------------------------------------------------
def _row_normalise(X):
    if sp.issparse(X):
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        return sp.diags(1.0 / np.maximum(norms, 1e-12)) @ X
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class CosineIndex:
    """Cosine-similarity lookup over the rows of a (sparse or dense) matrix."""

    def __init__(self, X, w2i: dict, exclude_symbols: bool = False,
                 counts=None, min_count: int = 1):
        """`counts[i]` = corpus frequency of row i; rows with fewer than
        `min_count` occurrences are never returned as neighbours/answers."""
        self.Xn = _row_normalise(X)
        self.w2i = {w: i for w, i in w2i.items() if i < self.Xn.shape[0]}
        self.i2w = {i: w for w, i in self.w2i.items()}
        self.n = self.Xn.shape[0]
        self.mask = np.zeros(self.n, dtype=bool)
        for w, i in self.w2i.items():
            if w in (C.UNK, C.PAD) or (exclude_symbols and is_symbol(w)):
                self.mask[i] = True
        if counts is not None and min_count > 1:
            self.mask[: len(counts)] |= np.asarray(counts)[: self.n] < min_count

    def sims(self, vec) -> np.ndarray:
        s = self.Xn @ (vec.T if sp.issparse(vec) else vec)
        if sp.issparse(s):
            s = s.toarray()
        return np.asarray(s).ravel()

    def vector(self, word: str):
        return self.Xn[self.w2i[word]]

    def neighbours(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        i = self.w2i[word]
        s = self.sims(self.vector(word))
        s[i] = -np.inf
        s[self.mask] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        return [(self.i2w[j], float(s[j])) for j in top]

    def rank_of(self, query: str, target: str) -> int:
        """1-based rank of `target` among all words by similarity to `query`."""
        i, j = self.w2i[query], self.w2i[target]
        s = self.sims(self.vector(query))
        s[i] = -np.inf
        s[self.mask] = -np.inf
        return int((s > s[j]).sum()) + 1
