"""Part 2: CRF exactness, padding/masking, conlleval port, metrics, data files."""
import itertools
import random

import numpy as np
import pytest
import torch

import config as C
import conlleval
from metrics import confusion_matrix, macro_f1, per_class_prf
from seq_models import CRF, BiLSTMTagger


def _brute_force(crf, emis, length):
    """log Z and best path by enumerating every tag sequence."""
    K = emis.shape[-1]
    scores = {}
    for path in itertools.product(range(K), repeat=length):
        s = crf.start[path[0]] + emis[0, path[0]]
        for t in range(1, length):
            s = s + crf.transitions[path[t - 1], path[t]] + emis[t, path[t]]
        s = s + crf.end[path[-1]]
        scores[path] = s
    logZ = torch.logsumexp(torch.stack(list(scores.values())), 0)
    best = max(scores, key=lambda p: scores[p].item())
    return logZ, list(best), scores


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_crf_matches_brute_force_with_padding(seed):
    torch.manual_seed(seed)
    K, T = 3, 5
    crf = CRF(K)
    emis = torch.randn(2, T, K)
    lengths = [5, 3]                                    # second sequence is padded
    mask = torch.arange(T).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    tags = torch.randint(0, K, (2, T))
    tags[1, 3:] = -100                                  # padded labels
    nll = crf.neg_log_likelihood(emis, tags, mask)
    paths = crf.viterbi(emis, mask)
    expect = []
    for b, L in enumerate(lengths):
        logZ, best, scores = _brute_force(crf, emis[b], L)
        gold = tuple(tags[b, :L].tolist())
        expect.append((logZ - scores[gold]).item())
        assert paths[b] == best
        assert len(paths[b]) == L
    assert nll.item() == pytest.approx(np.mean(expect), abs=1e-4)


def test_padding_does_not_change_the_loss():
    torch.manual_seed(0)
    for use_crf in (False, True):
        m = BiLSTMTagger(4, vocab_size=20, emb_dim=8, pad_idx=0, hidden=6, use_crf=use_crf)
        m.eval()
        x = torch.tensor([[3, 4, 5]])
        y = torch.tensor([[1, 2, 3]])
        L = torch.tensor([3])
        mask = torch.ones(1, 3, dtype=torch.bool)
        base = m.loss(x, L, y, mask).item()
        xp = torch.tensor([[3, 4, 5, 0, 0]])
        yp = torch.tensor([[1, 2, 3, -100, -100]])
        maskp = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
        assert m.loss(xp, L, yp, maskp).item() == pytest.approx(base, abs=1e-6)
        assert m.predict(xp, L, maskp) == m.predict(x, L, mask)


def test_frozen_embeddings_do_not_change():
    E = np.random.RandomState(0).randn(10, 5).astype(np.float32)
    m = BiLSTMTagger(3, 10, 5, pretrained=E, freeze=True, pad_idx=9, hidden=4)
    assert not m.emb.weight.requires_grad
    m2 = BiLSTMTagger(3, 10, 5, pretrained=E, freeze=False, pad_idx=9, hidden=4)
    assert m2.emb.weight.requires_grad


def test_fine_tuning_does_not_modify_the_source_embedding_array():
    """Regression test: from_pretrained(torch.as_tensor(array)) shares memory, so
    fine-tuning used to change the caller's array (and every later model)."""
    from part3 import BiLSTMClassifier
    E = np.random.RandomState(0).randn(12, 6).astype(np.float32)
    E_before = E.copy()
    x, L = torch.tensor([[1, 2, 3, 4]]), torch.tensor([4])
    for model, loss_fn in (
            (BiLSTMTagger(3, 12, 6, pretrained=E, freeze=False, pad_idx=11, hidden=4),
             lambda m: m.loss(x, L, torch.tensor([[0, 1, 2, 1]]), torch.ones(1, 4, dtype=torch.bool))),
            (BiLSTMClassifier(E, pad_idx=11, hidden=4),
             lambda m: m(x, L).sum())):
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.1)
        loss_fn(model).backward()
        opt.step()
        assert not np.allclose(model.emb.weight.detach().numpy(), E_before)   # it did train
        assert np.array_equal(E, E_before)                                    # source untouched


def _random_bio(rng, n):
    types = ["PER", "LOC", "ORG", "MISC"]
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.6:
            out.append("O")
        elif r < 0.8:
            out.append("B-" + rng.choice(types))
        else:
            out.append("I-" + rng.choice(types))   # includes ill-formed I- starts
    return out


def test_conlleval_port_matches_perl_original():
    rng = random.Random(0)
    toks, gold, pred = [], [], []
    for _ in range(60):
        n = rng.randint(1, 12)
        toks.append([f"w{i}" for i in range(n)])
        gold.append(_random_bio(rng, n))
        pred.append([g if rng.random() < 0.7 else t for g, t in zip(gold[-1], _random_bio(rng, n))])
    ours = conlleval.evaluate(gold, pred)
    perl = conlleval.run_perl(toks, gold, pred)
    head = perl.splitlines()[1]
    assert f"{ours['precision']:6.2f}%" in head and f"{ours['recall']:6.2f}%" in head
    assert f"FB1: {ours['f1']:6.2f}" in head
    first = perl.splitlines()[0]
    assert f"with {ours['gold_phrases']} phrases" in first
    assert f"found: {ours['found_phrases']} phrases; correct: {ours['correct_phrases']}." in first
    for t, v in ours["per_type"].items():
        line = next(l for l in perl.splitlines() if l.strip().startswith(t + ":"))
        assert f"FB1: {v['f1']:6.2f}" in line


def test_metrics_match_sklearn():
    from sklearn.metrics import confusion_matrix as skcm, f1_score
    rng = np.random.RandomState(0)
    labels = list("ABCD")
    g = list(rng.choice(labels, 300))
    p = [x if rng.rand() < 0.6 else rng.choice(labels) for x in g]
    assert macro_f1(g, p) == pytest.approx(f1_score(g, p, average="macro"))
    assert (confusion_matrix(g, p, labels) == skcm(g, p, labels=labels)).all()
    assert per_class_prf(g, p, labels)["A"]["support"] == g.count("A")


def test_conll_files_are_consistent():
    from part2_data import read_conll
    sizes = {}
    for split in ("train", "val", "test"):
        tp, pos = read_conll(C.DATA / f"pos_{split}.conll")
        tn, ner = read_conll(C.DATA / f"ner_{split}.conll")
        assert tp == tn                                   # same sentences, same tokens
        assert all(t in C.POS_TAGS for s in pos for t in s)
        assert all(t in C.NER_TAGS for s in ner for t in s)
        sizes[split] = len(tp)
    assert sizes == {"train": 350, "val": 75, "test": 75}
