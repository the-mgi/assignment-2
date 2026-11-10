"""Part 1: TF-IDF, PPMI, skip-gram pieces and evaluation utilities."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import config as C
from embed_eval import analogy, mean_reciprocal_rank
from part1_counts import CosineIndex, cooccurrence, ppmi, term_document_counts, tfidf
from word2vec import (SkipGramNS, build_noise_table, keep_probabilities, noise_distribution,
                      skipgram_pairs)


def _toy_vocab(words):
    w2i = {w: i for i, w in enumerate(words)}
    return w2i


def test_tfidf_formula_on_toy_corpus(monkeypatch):
    monkeypatch.setattr(C, "VOCAB_SIZE", 3)
    import part1_counts
    monkeypatch.setattr(part1_counts, "N_ROWS", 4)
    w2i = {"a": 0, "b": 1, "c": 2, C.UNK: 3, C.PAD: 4}
    arts = [[["a", "a", "b"]], [["a", "c", "zzz"]]]          # zzz -> <UNK>
    M = term_document_counts(arts, w2i)
    assert M[:, 0].tolist() == [2, 1, 0, 0] and M[:, 1].tolist() == [1, 0, 1, 1]
    X, idf, df = tfidf(M)
    assert df.tolist() == [2, 1, 1, 1]
    assert idf[0] == pytest.approx(math.log(2 / 3))           # log N / (1 + df)
    assert X[1, 0] == pytest.approx(1 * math.log(2 / 2))
    assert X[0, 0] == pytest.approx(2 * math.log(2 / 3))


def test_cooccurrence_window_and_symmetry(monkeypatch):
    import part1_counts
    monkeypatch.setattr(part1_counts, "N_ROWS", 5)
    w2i = {"a": 0, "b": 1, "c": 2, "d": 3, C.UNK: 4}
    co = cooccurrence([["a", "b", "c", "d"]], w2i, window=2).toarray()
    assert (co == co.T).all()
    assert co[0, 1] == 1 and co[0, 2] == 1 and co[0, 3] == 0      # d is 3 away
    assert co.sum() == 2 * (3 + 2)                                 # pairs at distance 1 and 2


def test_ppmi_formula():
    import scipy.sparse as sp
    C_ = np.array([[0, 4, 1], [4, 0, 2], [1, 2, 0]], dtype=float)
    P = ppmi(sp.csr_matrix(C_)).toarray()
    T = C_.sum()
    r = C_.sum(1)
    for i in range(3):
        for j in range(3):
            expect = 0.0 if C_[i, j] == 0 else max(0.0, math.log2(C_[i, j] * T / (r[i] * r[j])))
            assert P[i, j] == pytest.approx(expect, abs=1e-5)


def test_skipgram_pairs():
    ids = [np.array([10, 11, 12, 13])]
    c, o = skipgram_pairs(ids, window=2)
    pairs = set(zip(c.tolist(), o.tolist()))
    assert len(c) == 2 * (3 + 2)
    assert (11, 10) in pairs and (11, 13) in pairs and (10, 13) not in pairs
    assert all(a != b for a, b in pairs)


def test_noise_table_follows_unigram_power():
    counts = np.array([100, 50, 10, 1, 0])
    table = build_noise_table(counts, 0.75, table_size=200_000)
    emp = np.bincount(table, minlength=5) / len(table)
    expect = counts ** 0.75 / (counts ** 0.75).sum()
    assert np.allclose(emp, expect, atol=1e-4)
    assert np.allclose(noise_distribution(counts), expect)
    assert emp[4] == 0


def test_skipgram_loss_matches_formula():
    torch.manual_seed(0)
    m = SkipGramNS(6, 4)
    torch.nn.init.normal_(m.U.weight)                      # U starts at zero otherwise
    c, o, neg = torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([[0, 5], [5, 1]])
    loss = m(c, o, neg)
    manual = []
    for b in range(2):
        v = m.V.weight[c[b]]
        s = -F.logsigmoid(m.U.weight[o[b]] @ v)
        for k in neg[b]:
            s = s - F.logsigmoid(-(m.U.weight[k] @ v))
        manual.append(s)
    assert loss.item() == pytest.approx(torch.stack(manual).mean().item(), abs=1e-6)
    E = m.embeddings()
    assert np.allclose(E, 0.5 * (m.V.weight + m.U.weight).detach().numpy())


def test_subsampling_keeps_rare_words():
    p = keep_probabilities(np.array([1_000_000, 10, 0]), 1e-4)
    assert p[0] < 0.05 and p[1] == 1.0 and p[2] == 1.0


def test_analogy_and_mrr_on_toy_vectors():
    w2i = _toy_vocab(["king", "queen", "man", "woman", "apple", C.UNK])
    X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0.1], [0, 0, 1.1], [5, -5, 0], [0, 0, 0.0]])
    idx = CosineIndex(X + 1e-9, w2i)
    top = analogy(idx, "man", "woman", "king", k=1)
    assert top[0][0] == "queen"                            # query words are excluded
    mrr, ranks = mean_reciprocal_rank(idx, [("king", "queen"), ("king", "apple")])
    assert ranks[1] == 4 and mrr == pytest.approx(np.mean([1 / ranks[0], 1 / ranks[1]]))
