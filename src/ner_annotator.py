"""Rule-based BIO NER annotation (PER / LOC / ORG / MISC).

1. Gazetteer matching over the whitespace words of the sentence, longest
   match first (so پاکستان تحریک انصاف -> ORG beats پاکستان -> LOC).
   On equal length the priority is ORG > LOC > PER > MISC.
2. Person pattern: a known given name followed by name parts
   (e.g. محمد + رضوان, سلمان + علی + آغا) -> PER, if not already covered.
3. Word-level spans are projected onto the cleaned tokens: the first token of
   a span gets B-X, the others I-X; postpositions that A1 split off a word
   (امریکی -> امری + کی) and punctuation stay O.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ner_gazetteer as G  # noqa: E402
from pos_tagger import CASE_MARKERS, norm  # noqa: E402
import pos_lexicon as L  # noqa: E402

TYPE_PRIORITY = {"ORG": 0, "LOC": 1, "PER": 2, "MISC": 3}


@lru_cache(maxsize=1)
def gazetteer_index() -> dict[str, list[tuple[tuple[str, ...], str]]]:
    """first word -> [(entry words, type)], longest entries first."""
    idx: dict[str, list] = {}
    seen = set()
    for typ, entries in G.GAZETTEER.items():
        for e in entries:
            words = tuple(norm(w) for w in e.split())
            if words and (words, typ) not in seen:
                seen.add((words, typ))
                idx.setdefault(words[0], []).append((words, typ))
    for k in idx:
        idx[k].sort(key=lambda x: (-len(x[0]), TYPE_PRIORITY[x[1]]))
    return idx


FIRST = {norm(w) for w in G.FIRST_NAMES}
PARTS = {norm(w) for w in G.NAME_PARTS} | FIRST


def word_spans(words: list[str]) -> list[tuple[int, int, str]]:
    """(start, end_exclusive, type) spans over a list of normalised words."""
    idx = gazetteer_index()
    spans, i = [], 0
    taken = [False] * len(words)
    while i < len(words):
        match = None
        for entry, typ in idx.get(words[i], []):
            n = len(entry)
            if tuple(words[i:i + n]) == entry:
                match = (i, i + n, typ)
                break
        if match:
            spans.append(match)
            for k in range(match[0], match[1]):
                taken[k] = True
            i = match[1]
        else:
            i += 1
    # person pattern on the remaining words
    i = 0
    while i < len(words):
        if not taken[i] and words[i] in FIRST:
            j = i + 1
            while j < len(words) and not taken[j] and words[j] in PARTS:
                j += 1
            spans.append((i, j, "PER"))
            i = j
        else:
            i += 1
    return sorted(spans)


def annotate_records(recs: list[dict]) -> list[str]:
    """BIO tags for one sentence of A1 token records."""
    # distinct words in order (punctuation words are not part of names)
    order, wtext = [], {}
    for r in recs:
        if r["word_id"] not in wtext and r["word"] not in L.PUNCT:
            order.append(r["word_id"])
            wtext[r["word_id"]] = norm(r["word"])
    words = [wtext[w] for w in order]
    word_label = {}
    for s, e, typ in word_spans(words):
        for k in range(s, e):
            word_label[order[k]] = (typ, s)        # span id = start index
    tags, prev_span = [], None
    for r in recs:
        lab = word_label.get(r["word_id"])
        surf = norm(r["surface"])
        split_post = (surf in CASE_MARKERS and norm(r["word"]) != surf)
        if lab is None or split_post or r["clean"] in L.PUNCT or r["clean"].startswith("<"):
            tags.append("O")
            continue
        typ, span_id = lab
        tags.append(("I-" if prev_span == span_id else "B-") + typ)
        prev_span = span_id
        continue
    # a span id only continues while tokens are adjacent
    fixed, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            t = "B-" + t[2:]
        fixed.append(t)
        prev = t
    return fixed
