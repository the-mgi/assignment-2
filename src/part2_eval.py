"""Part 2, Section 5 helpers: error analysis for POS and NER."""
from __future__ import annotations


def chunks(tags: list[str]) -> set[tuple[int, int, str]]:
    """Entity spans (start, end_exclusive, type) with conlleval's chunk rules
    (an I- tag that does not continue a chunk of the same type starts one)."""
    spans, start, typ = set(), None, None
    for i, t in enumerate(tags + ["O"]):
        b, ty = (t.split("-", 1) + [""])[:2] if t != "O" else ("O", "")
        if start is not None and (b in ("B", "O") or ty != typ):
            spans.add((start, i, typ))
            start, typ = None, None
        if b in ("B", "I") and start is None:
            start, typ = i, ty
    return spans


def entity_errors(tokens, gold, pred):
    """False positives (predicted spans not in gold) and false negatives
    (gold spans not predicted exactly), with the overlapping counterpart."""
    fps, fns = [], []
    for s, (toks, g, p) in enumerate(zip(tokens, gold, pred)):
        gs, ps = chunks(g), chunks(p)
        for (a, b, t) in sorted(ps - gs):
            over = [x for x in gs if x[0] < b and a < x[1]]
            fps.append({"sent": s, "span": " ".join(toks[a:b]), "pred_type": t,
                        "gold": [(" ".join(toks[x[0]:x[1]]), x[2]) for x in over],
                        "context": " ".join(toks[max(0, a - 4):b + 4])})
        for (a, b, t) in sorted(gs - ps):
            over = [x for x in ps if x[0] < b and a < x[1]]
            fns.append({"sent": s, "span": " ".join(toks[a:b]), "gold_type": t,
                        "pred": [(" ".join(toks[x[0]:x[1]]), x[2]) for x in over],
                        "context": " ".join(toks[max(0, a - 4):b + 4])})
    return fps, fns


def confused_examples(tokens, gold, pred, pair: tuple[str, str], k: int = 2):
    """Up to k sentences in which tag pair (A, B) is confused (either direction)."""
    a, b = pair
    out = []
    for toks, g, p in zip(tokens, gold, pred):
        hits = [i for i, (x, y) in enumerate(zip(g, p)) if {x, y} == {a, b}]
        if hits:
            marked = " ".join(f"[{t} gold={x} pred={y}]" if i in hits else t
                              for i, (t, x, y) in enumerate(zip(toks, g, p)))
            out.append(marked)
        if len(out) == k:
            break
    return out
