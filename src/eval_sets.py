"""Fixed evaluation sets for Part 1 (written down before the final runs).

All words are given in their surface (raw.txt) spelling; `cleaned()` maps
them through the A1 pipeline to the form used in cleaned.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_bridge import a1_normalize  # noqa: E402

# Section 2.2: query words of the brief (Roman Urdu -> Urdu script)
QUERY_WORDS = {"Pakistan": "پاکستان", "Hukumat": "حکومت", "Adalat": "عدالت",
               "Maeeshat": "معیشت", "Fauj": "فوج", "Sehat": "صحت", "Taleem": "تعلیم",
               "Aabadi": "آبادی"}
# two more query words so that 10 words are reported (grading rubric, Section 2.2)
EXTRA_QUERY_WORDS = {"Cricket": "کرکٹ", "Tail (oil)": "تیل"}

# Section 1.2: >= 10 query words for PPMI nearest neighbours
PPMI_QUERIES = ["پاکستان", "حکومت", "عدالت", "فوج", "صحت", "تعلیم", "کرکٹ", "تیل",
                "پولیس", "ایران"]

# Four-condition comparison: 5 query words + 20 manually labelled related pairs
COND_QUERIES = ["پاکستان", "حکومت", "عدالت", "فوج", "صحت"]
MRR_PAIRS = [("حکومت", "وزیر"), ("عدالت", "جج"), ("فوج", "جنرل"), ("کرکٹ", "ٹیم"),
             ("تیل", "گیس"), ("ہسپتال", "ڈاکٹر"), ("پولیس", "ملزم"), ("ایران", "تہران"),
             ("اسرائیل", "غزہ"), ("میزائل", "ڈرون"), ("ٹرمپ", "امریکہ"), ("عمران", "خان"),
             ("صحت", "علاج"), ("تعلیم", "سکول"), ("معیشت", "مہنگائی"), ("فلم", "اداکار"),
             ("پاکستان", "انڈیا"), ("روس", "یوکرین"), ("جنگ", "حملے"), ("ڈالر", "روپے")]

# Section 2.2: 10 analogy tests a : b :: c : d (each pair co-occurs >= 3 times
# within a 5-token window in cleaned.txt, i.e. the relation is attested)
ANALOGIES = [
    ("country : capital", "روس", "ماسکو", "انڈیا", "دہلی"),
    ("country : leader", "امریکہ", "ٹرمپ", "روس", "پوتن"),
    ("country : leader", "امریکہ", "ٹرمپ", "اسرائیل", "نیتن"),
    ("country : demonym", "پاکستان", "پاکستانی", "انڈیا", "انڈین"),
    ("first name : surname", "عمران", "خان", "شہباز", "شریف"),
    ("first name : surname", "ڈونلڈ", "ٹرمپ", "نریندر", "مودی"),
    ("institution : professional", "عدالت", "جج", "ہسپتال", "ڈاکٹر"),
    ("male : female", "مرد", "عورت", "بھائی", "بہن"),
    ("armed group : territory", "حماس", "غزہ", "طالبان", "افغانستان"),
    ("commodity : unit", "تیل", "بیرل", "پیٹرول", "لیٹر"),
]

# Development pool (used only to choose the training configuration of C3;
# disjoint from ANALOGIES): every ordered combination of two pairs of a relation.
DEV_RELATIONS = {
    "capital": [("ایران", "تہران"), ("روس", "ماسکو"), ("افغانستان", "کابل"), ("امریکہ", "واشنگٹن"),
                ("برطانیہ", "لندن"), ("انڈیا", "دہلی"), ("چین", "بیجنگ"), ("فرانس", "پیرس")],
    "leader": [("امریکہ", "ٹرمپ"), ("انڈیا", "مودی"), ("روس", "پوتن"), ("اسرائیل", "نیتن"),
               ("ایران", "خامنہ")],
    "demonym": [("پاکستان", "پاکستانی"), ("ایران", "ایرانی"), ("انڈیا", "انڈین"),
                ("برطانیہ", "برطانوی"), ("افغانستان", "افغان")],
    "first-last": [("عمران", "خان"), ("نواز", "شریف"), ("شہباز", "شریف"), ("ڈونلڈ", "ٹرمپ"),
                   ("نریندر", "مودی"), ("اسامہ", "لادن"), ("بابر", "اعظم"), ("محسن", "نقوی")],
    "inst-prof": [("عدالت", "جج"), ("ہسپتال", "ڈاکٹر"), ("سکول", "استاد"), ("فوج", "جنرل"),
                  ("ٹیم", "کپتان")],
    "gender": [("شوہر", "بیوی"), ("والد", "والدہ"), ("مرد", "عورت"), ("بھائی", "بہن")],
    "group-territory": [("طالبان", "افغانستان"), ("حماس", "غزہ"), ("حوثی", "یمن")],
    "commodity-unit": [("تیل", "بیرل"), ("پیٹرول", "لیٹر")],
}


def cleaned(word: str) -> str:
    toks = a1_normalize(word)
    assert len(toks) == 1, (word, toks)
    return toks[0]


def cleaned_analogies(tests=None):
    tests = ANALOGIES if tests is None else tests
    return [(rel,) + tuple(cleaned(w) for w in ws) for rel, *ws in tests]


def dev_pool():
    test_keys = {tuple(cleaned(w) for w in t[1:]) for t in ANALOGIES}
    pool = []
    for rel, pairs in DEV_RELATIONS.items():
        cp = [(cleaned(a), cleaned(b)) for a, b in pairs]
        for (a, b) in cp:
            for (c, d) in cp:
                if (a, b) != (c, d) and a != c and b != d and (a, b, c, d) not in test_keys:
                    pool.append((rel, a, b, c, d))
    return pool
