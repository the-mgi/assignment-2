"""Rule-based Urdu POS tagger built on the Assignment 1 lemmatizer/stemmer.

Input: the token records of `a1_bridge.analyse_article` (cleaned token +
surface token + source word). Output: one tag per cleaned token.

Pipeline for every token
  1. punctuation / <NUM>                            -> PUNC / NUM
  2. letters of a spelled-out acronym (پی ٹی آئی)     -> NOUN
  3. lexicon lookup of the surface form             (pos_lexicon.py)
  4. lexicon lookup of the A1 *lemma* (lemmatize)    e.g. حکومتوں -> حکومت
  5. lexicon lookup of the A1 *stem*  (cleaned form) e.g. دھمکیوں -> دھمک
     (only if all lexicon words with that stem share one tag)
  6. morphology on the surface form, using the A1 suffix inventory:
     verbal endings on a known verb root -> VERB, plural / abstract-noun
     endings -> NOUN, nisba -ی on a known noun -> ADJ
  7. context rules for ambiguous function words (میں، کی، کیا، لیے، تو،
     یہ/وہ/اس/ان..., ہو/جا/رہ forms, قریب/بعد/پہلے...)
  8. context rules for unknown words (before a case marker or light verb,
     after a determiner/adjective/numeral -> NOUN), otherwise UNK
"""
from __future__ import annotations

import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a1_preprocess as P  # noqa: E402
from a1_bridge import a1_token  # noqa: E402
import pos_lexicon as L  # noqa: E402

PRIORITY = ["NUM", "POST", "CONJ", "PRON", "DET", "AUX", "VAUX", "ADV", "ADJ", "NOUN", "VERB"]
CASE_MARKERS = {"کا", "کی", "کے", "کو", "نے", "سے", "میں", "پر", "تک", "والا", "والی",
                "والے", "والوں"}
LIGHT_VERBS = {"کر", "کرنے", "کرنا", "کرتا", "کرتے", "کرتی", "کیا", "کیے", "کریں", "کرے",
               "کرکے", "ہو", "ہونے", "ہوا", "ہوئی", "ہوئے", "ہوتا", "ہوتی", "ہوتے",
               "دیا", "دی", "دیے", "دینے", "دیتا", "دیتے", "دیتی", "رکھا", "رکھنے"}
VERB_SUFFIXES = ("تیں", "نا", "نے", "نی", "تا", "تے", "تی")


def norm(w: str) -> str:
    """Same character normalisation as A1 (Unicode folding + no diacritics)."""
    return P.remove_diacritics(P.normalize_unicode(w)).strip()


@lru_cache(maxsize=1)
def lexicon() -> dict[str, str]:
    """surface form -> tag, resolving overlaps by PRIORITY (closed classes first)."""
    groups = {
        "NUM": L.NUM, "POST": L.POST, "CONJ": L.CONJ, "PRON": L.PRON, "DET": L.DET,
        "AUX": L.AUX, "VAUX": L.VERB_OR_AUX, "ADV": L.ADV, "ADJ": L.ADJ, "NOUN": L.NOUN,
        "VERB": L.IRREGULAR_VERB_FORMS + [f for lem in L.VERB_LEMMAS for f in L.verb_forms(lem)],
    }
    lex: dict[str, str] = {}
    for tag in PRIORITY:
        for w in groups[tag]:
            w = norm(w)
            if w and w not in lex:
                lex[w] = tag
    return lex


@lru_cache(maxsize=1)
def lexicon_entry_counts() -> dict[str, int]:
    """Number of distinct hand-written entries per category (verbs: lemmas)."""
    return {
        "NOUN": len({norm(w) for w in L.NOUN}), "VERB (lemmas)": len(set(L.VERB_LEMMAS)),
        "VERB (all forms)": sum(1 for t in lexicon().values() if t == "VERB"),
        "ADJ": len({norm(w) for w in L.ADJ}), "ADV": len({norm(w) for w in L.ADV}),
        "PRON": len(set(L.PRON)), "DET": len(set(L.DET)), "CONJ": len(set(L.CONJ)),
        "POST": len(set(L.POST)), "AUX": len(set(L.AUX)) + len(set(L.VERB_OR_AUX)),
        "NUM": len(set(L.NUM)),
    }


@lru_cache(maxsize=1)
def stem_lexicon() -> dict[str, str]:
    """A1 cleaned form of every lexicon entry -> tag (only unambiguous stems)."""
    seen: dict[str, set] = {}
    for w, t in lexicon().items():
        seen.setdefault(a1_token(w), set()).add("VERB" if t == "VAUX" else t)
    return {s: next(iter(ts)) for s, ts in seen.items() if len(ts) == 1}


@lru_cache(maxsize=1)
def verb_roots() -> set[str]:
    roots = {norm(lem[:-2]) for lem in L.VERB_LEMMAS if lem.endswith("نا")}
    return roots | {"کر", "دے", "لے", "آ", "پی", "چاہ", "پڑ", "مل", "ہو", "جا", "رہ"}


NOUNS = None


def _nouns() -> set[str]:
    global NOUNS
    if NOUNS is None:
        NOUNS = {w for w, t in lexicon().items() if t == "NOUN"}
    return NOUNS


def lexical_tag(surface: str) -> str | None:
    """Steps 3-6: lexicon, A1 lemma, A1 stem, morphology. None = unknown."""
    w = norm(surface)
    lex = lexicon()
    if w in lex:
        return lex[w]
    lemma = P.lemmatize(w)
    if lemma != w and lemma in lex:
        t = lex[lemma]
        return "VERB" if t == "VAUX" else t
    stem = a1_token(w)
    if stem in stem_lexicon() and len(stem) >= 3:
        return stem_lexicon()[stem]
    # --- morphology (A1 suffix inventory) ---
    for suf in VERB_SUFFIXES:
        if w.endswith(suf) and w[: -len(suf)] in verb_roots():
            return "VERB"
    if len(w) >= 4 and (w.endswith("وں") or w.endswith("ات") or w.endswith("یاں")):
        return "NOUN"
    if len(w) >= 4 and w.endswith(("یت", "گی", "ہٹ", "ائی", "وائی")):
        return "NOUN"
    if w.endswith("ی") and len(w) >= 4 and (w[:-1] in _nouns() or w[:-1] + "ہ" in _nouns()
                                            or w[:-1] + "ا" in _nouns()):
        return "ADJ"
    if len(w) >= 5 and w.endswith("انہ"):
        return "ADJ"
    return None


def _acronym_mask(surfaces: list[str]) -> list[bool]:
    mask = [False] * len(surfaces)
    i = 0
    while i < len(surfaces):
        j = i
        while j < len(surfaces) and norm(surfaces[j]) in L.ACRONYM_LETTERS:
            j += 1
        if j - i >= 2:
            for k in range(i, j):
                mask[k] = True
        i = max(j, i + 1)
    return mask


def tag_records(recs: list[dict]) -> list[str]:
    """Tag one sentence given its A1 token records."""
    n = len(recs)
    surf = [norm(r["surface"]) for r in recs]
    acr = _acronym_mask(surf)
    base: list[str | None] = []
    for i, r in enumerate(recs):
        s = surf[i]
        if r["clean"] in L.PUNCT or s in L.PUNCT:
            base.append("PUNC")
        elif r["clean"] == P.NUM_TOKEN:
            base.append("NUM")
        elif acr[i]:
            base.append("NOUN")
        else:
            word = norm(r["word"])
            # piece of a word that A1 split before a fused postposition
            # (امریکی -> امری + کی): tag the piece with the tag of the whole word
            if (word != s and word not in P.FUSED and word.startswith(s)
                    and word[len(s):] in CASE_MARKERS):
                t = lexical_tag(word)
                base.append(t if t not in (None, "POST") else lexical_tag(s))
            else:
                base.append(lexical_tag(s))

    tags = list(base)
    # ---- step 7: ambiguous function words (left-to-right) ----
    seen_if = False
    for i in range(n):
        s = surf[i]
        prev_s = surf[i - 1] if i > 0 else None
        next_s = surf[i + 1] if i + 1 < n else None
        prev_t = tags[i - 1] if i > 0 else None
        next_b = base[i + 1] if i + 1 < n else None
        if s in ("اگر", "جب", "جیسے"):
            seen_if = True
        if acr[i] or base[i] in ("PUNC", "NUM"):
            continue
        if s == "میں":
            pron = (i == 0 or prev_t in ("PUNC", "CONJ") or next_s in ("نے", "تو", "بھی", "خود"))
            tags[i] = "PRON" if pron else "POST"
        elif s == "کی":
            tags[i] = "VERB" if (next_s is None or next_b in ("AUX", "VAUX", "PUNC")) else "POST"
        elif s == "کیا":
            # interrogative 'what / whether' at the start of a clause, else 'did'
            question = i == 0 or base[i - 1] in ("PUNC", "CONJ")
            tags[i] = "PRON" if question else "VERB"
        elif s in ("لیے", "لئے"):
            # کے لیے / میرے لیے / اپنے لیے / اس لیے = 'for'; otherwise 'took'
            post = prev_s is not None and (prev_s.endswith("ے") or
                                           prev_s in ("اس", "ان", "جس", "کس", "اسی"))
            tags[i] = "POST" if post else "VERB"
        elif s == "تو":
            tags[i] = "CONJ" if seen_if else "ADV"
        elif s in L.PRON_DET:
            if next_s is None:
                tags[i] = "PRON"
            elif next_b in ("NOUN", "ADJ", "NUM"):
                tags[i] = "DET"          # determiner of the following noun phrase
            elif next_b is None and next_s not in CASE_MARKERS:
                tags[i] = "DET"          # followed by an unknown content word
            else:
                tags[i] = "PRON"         # اس کے / اس نے / یہ ہے ...
        elif s in L.POST_ADV:
            tags[i] = "POST" if prev_s in ("کے", "کی", "سے", "کا") else "ADV"
        elif base[i] == "VAUX":
            tags[i] = "AUX" if prev_t in ("VERB", "AUX") else "VERB"

    # ---- step 8: unknown words from context ----
    for i in range(n):
        if tags[i] is not None:
            continue
        next_s = surf[i + 1] if i + 1 < n else None
        next_t = tags[i + 1] if i + 1 < n else None
        prev_t = tags[i - 1] if i > 0 else None
        if (next_s in CASE_MARKERS and next_t == "POST") or next_s in LIGHT_VERBS \
                or prev_t in ("DET", "ADJ", "NUM", "NOUN") and next_t in ("POST", "NOUN", "ADJ"):
            tags[i] = "NOUN"
        elif prev_t in ("DET", "NUM"):
            tags[i] = "NOUN"
        else:
            tags[i] = "UNK"
    return tags


def tag_distribution(tag_lists: list[list[str]]) -> Counter:
    return Counter(t for ts in tag_lists for t in ts)
