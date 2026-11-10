"""Line-by-line Python port of conlleval.pl (CoNLL-2000/2003 evaluation
script by Erik Tjong Kim Sang, version 2004-01-26; the Perl original is in
tools/conlleval.pl and tests/test_part2.py checks that both agree).

Entity-level precision / recall / F1: an entity counts as correct only if
its type and both boundaries match the gold annotation exactly.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

PERL_SCRIPT = Path(__file__).resolve().parent.parent / "tools" / "conlleval.pl"


def _split(tag: str) -> tuple[str, str]:
    if "-" in tag:
        a, b = tag.split("-", 1)
        return a, b
    return tag, ""


def _end_of_chunk(prev_tag, tag, prev_type, typ) -> bool:
    end = False
    if prev_tag == "B" and tag in ("B", "O"):
        end = True
    if prev_tag == "I" and tag in ("B", "O"):
        end = True
    if prev_tag == "E" and tag in ("E", "I", "O"):
        end = True
    if prev_tag not in ("O", ".") and prev_type != typ:
        end = True
    if prev_tag in ("]", "["):
        end = True
    return end


def _start_of_chunk(prev_tag, tag, prev_type, typ) -> bool:
    start = False
    if prev_tag in ("B", "I", "O") and tag == "B":
        start = True
    if prev_tag == "O" and tag in ("I", "E"):
        start = True
    if prev_tag == "E" and tag in ("E", "I"):
        start = True
    if tag not in ("O", ".") and prev_type != typ:
        start = True
    if tag in ("[", "]"):
        start = True
    return start


def evaluate(gold_seqs: list[list[str]], pred_seqs: list[list[str]]) -> dict:
    """Sentence lists of BIO tags -> conlleval counts and P/R/F1 (in %)."""
    correct_chunk = found_correct = found_guessed = correct_tags = tokens = 0
    cc: dict[str, int] = {}
    fc: dict[str, int] = {}
    fg: dict[str, int] = {}
    in_correct = False
    last_c, last_ct, last_g, last_gt = "O", "", "O", ""

    def step(c, ct, g, gt, boundary=False):
        nonlocal in_correct, correct_chunk, found_correct, found_guessed
        nonlocal correct_tags, tokens, last_c, last_ct, last_g, last_gt
        if boundary:
            g = "O"
        if in_correct:
            ec = _end_of_chunk(last_c, c, last_ct, ct)
            eg = _end_of_chunk(last_g, g, last_gt, gt)
            if ec and eg and last_gt == last_ct:
                in_correct = False
                correct_chunk += 1
                cc[last_ct] = cc.get(last_ct, 0) + 1
            elif ec != eg or gt != ct:
                in_correct = False
        sc = _start_of_chunk(last_c, c, last_ct, ct)
        sg = _start_of_chunk(last_g, g, last_gt, gt)
        if sc and sg and gt == ct:
            in_correct = True
        if sc:
            found_correct += 1
            fc[ct] = fc.get(ct, 0) + 1
        if sg:
            found_guessed += 1
            fg[gt] = fg.get(gt, 0) + 1
        if not boundary:
            if c == g and gt == ct:
                correct_tags += 1
            tokens += 1
        last_c, last_ct, last_g, last_gt = c, ct, g, gt

    for gold, pred in zip(gold_seqs, pred_seqs):
        assert len(gold) == len(pred)
        for gt_tag, pr_tag in zip(gold, pred):
            c, ct = _split(gt_tag)
            g, gt = _split(pr_tag)
            step(c, ct, g, gt)
        step("O", "", "O", "", boundary=True)          # empty line between sentences
    if in_correct:
        correct_chunk += 1
        cc[last_ct] = cc.get(last_ct, 0) + 1

    def prf(corr, guessed, gold):
        p = 100.0 * corr / guessed if guessed else 0.0
        r = 100.0 * corr / gold if gold else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p, r, f = prf(correct_chunk, found_guessed, found_correct)
    per_type = {}
    for t in sorted(set(fc) | set(fg)):
        tp, tr, tf = prf(cc.get(t, 0), fg.get(t, 0), fc.get(t, 0))
        per_type[t] = {"precision": tp, "recall": tr, "f1": tf, "found": fg.get(t, 0),
                       "gold": fc.get(t, 0), "correct": cc.get(t, 0)}
    return {"tokens": tokens, "gold_phrases": found_correct, "found_phrases": found_guessed,
            "correct_phrases": correct_chunk,
            "accuracy": 100.0 * correct_tags / tokens if tokens else 0.0,
            "precision": p, "recall": r, "f1": f, "per_type": per_type}


def conll_lines(tokens: list[list[str]], gold: list[list[str]], pred: list[list[str]]) -> str:
    out = []
    for ts, gs, ps in zip(tokens, gold, pred):
        for t, g, p in zip(ts, gs, ps):
            out.append(f"{t} {g} {p}")
        out.append("")
    return "\n".join(out) + "\n"


def run_perl(tokens, gold, pred) -> str:
    """Run the original conlleval.pl and return its text report."""
    res = subprocess.run(["perl", str(PERL_SCRIPT)], input=conll_lines(tokens, gold, pred),
                         capture_output=True, text=True, check=True)
    return res.stdout
