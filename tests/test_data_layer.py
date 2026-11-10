"""Data layer: A1 bridge, corpus loading, vocabulary and topic labels."""
import json

import config as C
from a1_bridge import a1_normalize, a1_token, aligned_corpus
from corpus import (build_vocab, encode, flatten, load_cleaned_articles,
                    load_metadata, load_raw_bodies, load_raw_sentences_unprocessed)
from topics import KEYWORDS, OVERRIDES, assign_categories, normalized_keywords


def test_three_files_have_250_aligned_articles():
    assert len(load_cleaned_articles()) == len(load_raw_bodies()) == len(load_metadata()) == 250


def test_a1_pipeline_reproduces_cleaned_txt_exactly():
    # aligned_corpus() raises if any article differs from cleaned.txt
    arts = aligned_corpus()
    ref = load_cleaned_articles()
    assert sum(len(a) for a in arts) == sum(len(a) for a in ref) == 15018
    rec = arts[0][0][0]
    assert set(rec) == {"clean", "surface", "word_id", "word"}


def test_a1_normalisation_examples():
    # behaviour of the A1 lemmatizer/stemmer that later parts rely on
    assert a1_normalize("پاکستان") == ["پاکستان"]
    assert a1_normalize("آبادی") == ["آباد"]          # nisba -ی stripped
    assert a1_normalize("امریکی") == ["امر", "کی"]     # A1 splits fused 'کی'
    assert a1_token("ہوتے") == "ہوتا"                  # dictionary lemma
    assert a1_normalize("وزیر اعظم") == ["وزیر", "اعظم"]


def test_vocab_layout():
    sents = flatten(load_cleaned_articles())
    w2i = build_vocab(sents)
    assert len(w2i) == C.VOCAB_SIZE + 2
    assert w2i[C.UNK] == C.VOCAB_SIZE and w2i[C.PAD] == C.VOCAB_SIZE + 1
    assert w2i["کے"] == 0                               # most frequent token
    assert sorted(w2i.values()) == list(range(len(w2i)))
    assert encode(["کے", "zzz-not-a-word"], w2i) == [0, C.VOCAB_SIZE]


def test_raw_corpus_is_unprocessed():
    raw = flatten(load_raw_sentences_unprocessed())
    assert any(t.endswith("۔") for s in raw for t in s)   # punctuation still attached


def test_topic_labels_cover_all_articles():
    recs = assign_categories()
    assert len(recs) == 250
    assert {r["label"] for r in recs} == set(C.CATEGORIES)
    assert all(r["label_id"] == C.CATEGORIES.index(r["label"]) for r in recs)
    for n, (lab, why) in OVERRIDES.items():
        assert recs[n - 1]["label"] == lab and why


def test_every_category_has_its_seed_keywords():
    kw = normalized_keywords()
    for cat in C.CATEGORIES:
        assert len(kw[cat]) >= 5 and len(KEYWORDS[cat]) >= 5
