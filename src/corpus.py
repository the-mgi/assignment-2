"""Corpus loading and vocabulary construction shared by all three parts."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import a1_preprocess as P  # noqa: E402


# ----------------------------------------------------------------------------
# Reading the A1 deliverables
# ----------------------------------------------------------------------------
def _read_blocks(path: Path) -> list[list[str]]:
    """Split a '### ARTICLE n ###' file into per-article lists of lines."""
    blocks: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("### ARTICLE "):
            n = int(line.split()[2])
            assert n == len(blocks) + 1, f"article numbering broken at {n}"
            blocks.append([])
        elif blocks and line.strip():
            blocks[-1].append(line.strip())
    return blocks


@lru_cache(maxsize=1)
def load_cleaned_articles() -> list[list[list[str]]]:
    """cleaned.txt -> articles -> sentences (one per line) -> tokens."""
    return [[ln.split() for ln in blk] for blk in _read_blocks(C.CLEANED_TXT)]


@lru_cache(maxsize=1)
def load_raw_bodies() -> list[str]:
    """raw.txt -> one body string per article (paragraphs joined by newline)."""
    return ["\n".join(blk) for blk in _read_blocks(C.RAW_TXT)]


def load_raw_sentences_unprocessed() -> list[list[list[str]]]:
    """The *unprocessed* corpus used for condition C2.

    Exactly the A1 'raw pipeline' baseline: raw.txt split into sentences on the
    Urdu terminators and then on whitespace, with no normalisation, no
    punctuation splitting, no lemmatization and no stemming.
    """
    arts = []
    for body in load_raw_bodies():
        sents = []
        for line in body.split("\n"):
            for chunk in P.SENT_SPLIT_RE.split(line.strip()):
                toks = chunk.split()
                if len(toks) >= 2:
                    sents.append(toks)
        arts.append(sents)
    return arts


@lru_cache(maxsize=1)
def load_metadata() -> list[dict]:
    meta = json.loads(C.METADATA_JSON.read_text(encoding="utf-8"))
    assert [m["article_number"] for m in meta] == list(range(1, len(meta) + 1))
    return meta


def flatten(articles: list[list[list[str]]]) -> list[list[str]]:
    return [s for art in articles for s in art]


# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
def build_vocab(sentences: list[list[str]], size: int = C.VOCAB_SIZE) -> dict[str, int]:
    """The `size` most frequent tokens get ids 0..size-1 (most frequent first,
    ties broken alphabetically so the result is deterministic); <UNK> = size,
    <PAD> = size + 1. Count-based matrices use rows 0..size (words + <UNK>);
    <PAD> only exists for the neural models.
    """
    counts = Counter(t for s in sentences for t in s)
    words = sorted(counts, key=lambda w: (-counts[w], w))[:size]
    w2i = {w: i for i, w in enumerate(words)}
    w2i[C.UNK] = len(w2i)
    w2i[C.PAD] = len(w2i)
    return w2i


def encode(tokens: list[str], w2i: dict[str, int]) -> list[int]:
    unk = w2i[C.UNK]
    return [w2i.get(t, unk) for t in tokens]


def save_word2idx(w2i: dict[str, int], path: Path = C.WORD2IDX_JSON) -> None:
    path.write_text(json.dumps(w2i, ensure_ascii=False, indent=0), encoding="utf-8")


def load_word2idx(path: Path = C.WORD2IDX_JSON) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))


def token_counts(sentences: list[list[str]]) -> Counter:
    return Counter(t for s in sentences for t in s)


def is_symbol(tok: str) -> bool:
    """Punctuation, <NUM>/<UNK>/<PAD> placeholders and single letters."""
    return tok in C.PUNCT or tok.startswith("<") or len(tok) == 1


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int = C.SEED) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
