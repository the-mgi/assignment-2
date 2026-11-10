"""Part 2, Section 3: build the POS / NER dataset.

1. 500 sentences are drawn at random (seed 42) from cleaned.txt: 100 from
   each of the 5 topic categories (so >= 100 from each of >= 3 categories),
   only sentences with 5..50 tokens and no duplicates.
2. POS tags: rule-based tagger (pos_tagger.py) on the A1 surface forms.
3. NER tags: gazetteer + person pattern (ner_annotator.py), BIO.
   Unknown POS tokens inside a PER/LOC/ORG/MISC span are proper nouns -> NOUN.
4. 70/15/15 split *within each topic* (stratified): 70/15/15 per category.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from a1_bridge import aligned_corpus  # noqa: E402
from ner_annotator import annotate_records  # noqa: E402
from pos_tagger import tag_records  # noqa: E402
from topics import assign_categories  # noqa: E402

SPLITS = ("train", "val", "test")


def annotate_sentence(recs: list[dict]) -> dict:
    pos = tag_records(recs)
    ner = annotate_records(recs)
    pos = ["NOUN" if p == "UNK" and n != "O" else p for p, n in zip(pos, ner)]
    return {"tokens": [r["clean"] for r in recs], "surface": [r["surface"] for r in recs],
            "pos": pos, "ner": ner}


def select_sentences(seed: int = C.SEED) -> list[dict]:
    """100 random sentences per category (5 x 100 = 500)."""
    rng = random.Random(seed)
    labels = assign_categories()
    arts = aligned_corpus()
    pools: dict[str, list] = {c: [] for c in C.CATEGORIES}
    seen = set()
    for a_idx, art in enumerate(arts):
        cat = labels[a_idx]["label"]
        for s_idx, recs in enumerate(art):
            toks = tuple(r["clean"] for r in recs)
            if C.MIN_SENT_LEN <= len(toks) <= C.MAX_SENT_LEN and toks not in seen:
                seen.add(toks)
                pools[cat].append((a_idx, s_idx))
    chosen = []
    for cat in C.CATEGORIES:
        for a_idx, s_idx in rng.sample(pools[cat], C.SENT_PER_CATEGORY):
            rec = annotate_sentence(arts[a_idx][s_idx])
            rec.update({"article_number": a_idx + 1, "sentence_index": s_idx, "topic": cat})
            chosen.append(rec)
    assert len(chosen) == C.N_SENTENCES
    return chosen


def stratified_split(items: list[dict], seed: int = C.SEED,
                     fractions=(0.70, 0.15, 0.15)) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    out = {s: [] for s in SPLITS}
    for cat in C.CATEGORIES:
        group = [x for x in items if x["topic"] == cat]
        rng.shuffle(group)
        n_tr = round(fractions[0] * len(group))
        n_va = round(fractions[1] * len(group))
        out["train"] += group[:n_tr]
        out["val"] += group[n_tr:n_tr + n_va]
        out["test"] += group[n_tr + n_va:]
    for s in SPLITS:
        for x in out[s]:
            x["split"] = s
    return out


def write_conll(sents: list[dict], key: str, path: Path) -> None:
    """One `token<TAB>tag` per line, blank line between sentences."""
    lines = []
    for s in sents:
        lines += [f"{t}\t{g}" for t, g in zip(s["tokens"], s[key])]
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_conll(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    toks, tags, cur_t, cur_g = [], [], [], []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            if cur_t:
                toks.append(cur_t)
                tags.append(cur_g)
                cur_t, cur_g = [], []
            continue
        t, g = line.split("\t")
        cur_t.append(t)
        cur_g.append(g)
    if cur_t:
        toks.append(cur_t)
        tags.append(cur_g)
    return toks, tags


def build_and_save() -> dict[str, list[dict]]:
    splits = stratified_split(select_sentences())
    for s in SPLITS:
        write_conll(splits[s], "pos", C.DATA / f"pos_{s}.conll")
        write_conll(splits[s], "ner", C.DATA / f"ner_{s}.conll")
    meta = [{k: x[k] for k in ("split", "topic", "article_number", "sentence_index", "tokens")}
            for s in SPLITS for x in splits[s]]
    (C.DATA / "part2_sentences.json").write_text(json.dumps(meta, ensure_ascii=False),
                                                 encoding="utf-8")
    return splits


def label_distribution(splits: dict[str, list[dict]], key: str) -> dict[str, Counter]:
    return {s: Counter(t for x in splits[s] for t in x[key]) for s in SPLITS}
