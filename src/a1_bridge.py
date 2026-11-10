"""Bridge to the Assignment 1 preprocessing pipeline (src/a1_preprocess.py).

`a1_preprocess.py` is an unmodified copy of the A1 module (normaliser,
tokenizer, lemmatizer, stemmer). Running `preprocess()` on every article body
of raw.txt reproduces cleaned.txt exactly (checked in tests/test_a1_bridge.py),
so this module can recover, for every token of cleaned.txt:

    clean    the token as it appears in cleaned.txt (lemmatized + stemmed)
    surface  the same token before lemmatization/stemming
    word     the whitespace word it came from (before punctuation and fused
             postpositions were split off), used for gazetteer matching
    word_id  index of that word inside the sentence

Part 2 builds its rule-based POS tagger and NER annotator on top of this.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1_preprocess as P  # noqa: E402  (A1 code, used verbatim)

NUM_TOKEN = P.NUM_TOKEN


def a1_token(token: str) -> str:
    """Apply exactly the A1 lemma-then-stem rule to one tokenized piece."""
    lemma = P.lemmatize(token)
    return lemma if lemma != token else P.stem(token)


def a1_normalize(text: str) -> list[str]:
    """Run arbitrary text (e.g. a lexicon entry) through the A1 pipeline.

    Unlike `P.preprocess`, single-token inputs are kept (preprocess drops
    sentences shorter than two tokens).
    """
    text = P.clean_text(text)
    return [t for t in (a1_token(x) for x in P.tokenize(text)) if t.strip()]


def a1_surface(text: str) -> list[str]:
    """A1 normalisation + tokenization, but *without* lemmatization/stemming."""
    return [t for t in P.tokenize(P.clean_text(text)) if t.strip()]


def _tokenize_with_words(sentence: str) -> list[tuple[str, int, str]]:
    """Replicates P.tokenize() and records the source word of each piece."""
    out: list[tuple[str, int, str]] = []
    wid = 0
    for chunk in sentence.split():
        buf = ""
        for ch in chunk:
            if ch in P.PUNCT_CHARS:
                if buf:
                    out.extend((piece, wid, buf) for piece in P._emit(buf))
                    wid += 1
                    buf = ""
                out.append((ch, wid, ch))
                wid += 1
            else:
                buf += ch
        if buf:
            out.extend((piece, wid, buf) for piece in P._emit(buf))
            wid += 1
    return out


def analyse_article(body: str) -> list[list[dict]]:
    """Return the article as sentences of token records (see module doc)."""
    sentences = []
    for sent in P.segment_sentences(P.clean_text(body)):
        recs = []
        for surface, wid, word in _tokenize_with_words(sent):
            clean = a1_token(surface)
            if clean.strip():
                recs.append({"clean": clean, "surface": surface,
                             "word_id": wid, "word": word})
        if len(recs) >= 2:              # same filter as P.preprocess
            sentences.append(recs)
    return sentences


@lru_cache(maxsize=1)
def aligned_corpus() -> list[list[list[dict]]]:
    """All 250 articles analysed from raw.txt, verified against cleaned.txt."""
    from corpus import load_cleaned_articles, load_raw_bodies
    raw = load_raw_bodies()
    cleaned = load_cleaned_articles()
    out = []
    for i, (body, ref) in enumerate(zip(raw, cleaned)):
        art = analyse_article(body)
        got = [[r["clean"] for r in s] for s in art]
        if got != ref:
            raise AssertionError(f"A1 pipeline does not reproduce article {i + 1}")
        out.append(art)
    return out
