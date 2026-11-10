"""Generate the assignment notebook (all logic lives in src/; the notebook runs it).

    python build_notebook.py            # writes i25-8007_Assignment2_DS-A.ipynb
    python build_notebook.py --smoke    # tiny settings, for a quick end-to-end check

Then execute it, e.g.
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 i25-8007_Assignment2_DS-A.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import config as C  # noqa: E402

SMOKE = "--smoke" in sys.argv
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else \
    ROOT / f"{C.SUBMISSION_STEM}.ipynb"

RUN = dict(seeds=[42, 7, 123], w2v_epochs=C.W2V_EPOCHS, seq_max_epochs=C.SEQ_MAX_EPOCHS,
           tf_epochs=C.TF_EPOCHS, w2v_limit=None)
if SMOKE:
    RUN = dict(seeds=[42, 7], w2v_epochs=5, seq_max_epochs=3, tf_epochs=2, w2v_limit=1500)

cells = []


def md(s: str):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s: str):
    cells.append(nbf.v4.new_code_cell(s.strip()))


# =============================================================================
md(f"""
# {C.COURSE} — Assignment 2
## Neural NLP Pipeline — a continuation of the BBC Urdu NLP pipeline

**{C.STUDENT_NAME}** · **{C.STUDENT_ID}** · Section **{C.SECTION}**

GitHub repository: {C.REPO_URL}

Everything is implemented from scratch in PyTorch / NumPy: no pretrained models, no Gensim, no
HuggingFace, and none of `nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoder`.
scikit-learn is used only for the t-SNE projection. All logic lives in `src/` (unit-tested in
`tests/`); this notebook runs the complete pipeline end to end and shows every result.

**Inputs** — the unmodified Assignment 1 deliverables in `data/`: `cleaned.txt` (primary corpus),
`raw.txt` (ablation baseline) and `Metadata.json` (topic information); `src/a1_preprocess.py` is the
unmodified Assignment 1 normaliser / tokenizer / lemmatizer / stemmer.
""")

code(f"""
%matplotlib inline
import os, sys, json, time, platform, copy
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path.cwd() / "src"))

import numpy as np, pandas as pd, torch, matplotlib, sklearn
import config as C
from plotting import plt, savefig, PALETTE
from glosses import gloss, with_gloss

torch.set_num_threads(4)
pd.set_option("display.max_colwidth", 220); pd.set_option("display.width", 250)
RUN = {RUN!r}            # seeds and epoch budgets used in this run
RESULTS = {{}}                      # collected numbers -> outputs/results.json
print("Python", platform.python_version(), "| torch", torch.__version__, "| numpy", np.__version__,
      "| pandas", pd.__version__, "| matplotlib", matplotlib.__version__, "| scikit-learn", sklearn.__version__)
print("CPU threads:", torch.get_num_threads(), "| run settings:", RUN)
print("Student:", C.STUDENT_NAME, "|", C.STUDENT_ID, "|", C.SECTION, "| repository:", C.REPO_URL)
""")

md("""
## 0. Data and the Assignment 1 pipeline

`cleaned.txt` is the output of the A1 pipeline (normalise → segment → tokenize → lemmatize →
stem), one sentence per line, articles separated by `### ARTICLE n ###`. Re-running the A1 module on
`raw.txt` reproduces `cleaned.txt` **exactly**, so every cleaned token can be traced back to its
surface form — Part 2 builds the POS tagger on this alignment.
""")

code("""
from corpus import (load_cleaned_articles, load_metadata, flatten, build_vocab, token_counts,
                    save_word2idx, load_raw_sentences_unprocessed, is_symbol)
from a1_bridge import aligned_corpus, a1_normalize

arts = load_cleaned_articles(); meta = load_metadata(); sents = flatten(arts)
counts = token_counts(sents)
print(f"articles: {len(arts)} | sentences: {len(sents):,} | tokens: {sum(map(len, sents)):,} | "
      f"types: {len(counts):,}")
t0 = time.time(); aligned = aligned_corpus()
print(f"A1 pipeline re-run on raw.txt == cleaned.txt for all {len(aligned)} articles "
      f"({sum(len(a) for a in aligned):,} sentences, {time.time()-t0:.1f}s)")
r = aligned[0][0]
pd.DataFrame([{"cleaned token": x["clean"], "surface token": x["surface"], "source word": x["word"]}
              for x in r[:12]])
""")

md("""
### 0.1 Topic categories (used in Parts 1–3)

Each article is assigned to one of the 5 categories of the brief (Section 6.1). The English
indicative keywords were translated and expanded into Urdu keyword lists, every keyword was passed
through the A1 pipeline so it matches the stemmed text, and each article is scored with

$S_c = \\text{(keyword hits per 100 body tokens)} + \\text{(keyword hits in the headline)} + 0.5\\,\\times\\text{(BBC topic tags of } c)$

and assigned to $\\arg\\max_c S_c$. After reading all 250 headlines, 9 keyword collisions were
corrected by hand (listed below with the reason, e.g. *جنگ* = the newspaper *Jang*, not "war").
""")

code("""
from topics import assign_categories, KEYWORDS, OVERRIDES, normalized_keywords
recs = assign_categories()
labels_all = np.array([r["label_id"] for r in recs])
dist = pd.Series([r["label"] for r in recs]).value_counts().reindex(C.CATEGORIES)
brief_kw = {"Politics": "election, government, minister, parliament",
            "Sports": "cricket, match, team, player, score",
            "Economy": "inflation, trade, bank, GDP, budget",
            "International": "UN, treaty, foreign, bilateral, conflict",
            "Health & Society": "hospital, disease, vaccine, flood, education"}
display(pd.DataFrame({"category": C.CATEGORIES, "brief keywords": [brief_kw[c] for c in C.CATEGORIES],
                      "Urdu keywords": [len(normalized_keywords()[c]) for c in C.CATEGORIES],
                      "first Urdu keywords": ["، ".join(KEYWORDS[c][:6]) for c in C.CATEGORIES],
                      "articles": dist.values}))
pd.DataFrame([{"article": n, "automatic": recs[n-1]["auto_label"], "final": lab, "reason": why}
              for n, (lab, why) in OVERRIDES.items()])
""")

code("""
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(dist.index, dist.values, color=PALETTE[:5])
for i, v in enumerate(dist.values): ax.text(i, v + 1, str(v), ha="center")
ax.set_title("Articles per topic category (all 250 articles)"); ax.set_xlabel("category"); ax.set_ylabel("articles")
savefig(fig, "topic_distribution.png"); plt.show()
RESULTS["topic_distribution"] = dist.to_dict()
print(dist.to_dict())
""")

# =============================================================================
md("""
# Part 1 — Word Embeddings

## 1.1 TF-IDF weighting

**Vocabulary (shared by all parts):** the 10,000 most frequent tokens of `cleaned.txt` get ids
0–9,999 (ties broken alphabetically), every other token maps to `<UNK>` (id 10,000). `<PAD>` (id
10,001) exists only for the neural models. The term–document matrix therefore has |V| = 10,001 rows
(10,000 words + `<UNK>`) and N = 250 columns (articles); entry = raw count TF(w, d).

$\\text{TF-IDF}(w,d) = \\text{TF}(w,d)\\times \\log\\big[N / (1+\\text{df}(w))\\big]$ (natural log).
""")

code("""
from part1_counts import term_document_counts, tfidf, top_words_per_category
w2i = build_vocab(sents); save_word2idx(w2i)
i2w = [None] * len(w2i)
for w, i in w2i.items(): i2w[i] = w
cov = sum(counts[w] for w in w2i if w in counts) / sum(counts.values())
print(f"word2idx.json: {len(w2i):,} entries = 10,000 words + <UNK> ({w2i[C.UNK]}) + <PAD> ({w2i[C.PAD]})")
print(f"the 10,000 words cover {100*cov:.2f}% of all tokens; {len(counts) - C.VOCAB_SIZE:,} rare types -> <UNK>")

td = term_document_counts(arts, w2i)
X_tfidf, idf, df = tfidf(td)
np.save(C.TFIDF_NPY, X_tfidf.astype(np.float32))
print(f"term-document matrix: {td.shape} (terms x documents), {int(td.sum()):,} tokens, "
      f"<UNK> occurrences: {int(td[w2i[C.UNK]].sum())}")
print(f"saved {C.TFIDF_NPY.relative_to(Path.cwd())}: shape {X_tfidf.shape}, float32")
print("note: a word present in all 250 articles (df = 250) gets idf = log(250/251) < 0 under this formula")
pd.DataFrame([{"token": w, "gloss": gloss(w), "TF in article 1": int(td[w2i[w], 0]),
               "df": int(df[w2i[w]]), "idf": round(float(idf[w2i[w]]), 4),
               "TF-IDF in article 1": round(float(X_tfidf[w2i[w], 0]), 3)}
              for w in ["کے", "پاکستان", "صحت", "ڈاکٹر", "سہیل"]])
""")

md("""
**Top-10 most discriminative words per topic category.** Each document column is L2-normalised (so
long articles do not dominate) and words are ranked by
mean TF-IDF inside the category − mean TF-IDF in the other categories; a word must occur in at
least 2 articles of the category, and punctuation / placeholders are skipped.
""")

code("""
tops = top_words_per_category(X_tfidf, labels_all.tolist(), i2w, k=10)
top_df = pd.DataFrame({C.CATEGORIES[c]: [with_gloss(w) for w, _ in lst] for c, lst in tops.items()},
                      index=range(1, 11))
RESULTS["tfidf_top10"] = {C.CATEGORIES[c]: [w for w, _ in lst] for c, lst in tops.items()}
top_df
""")

md("""
## 1.2 Pointwise Mutual Information (PPMI)

Word–word co-occurrence counts with a symmetric window **k = 5** inside each sentence (every pair of
tokens at distance 1…5 is counted for both words), then

$\\text{PPMI}(w_1,w_2)=\\max\\{0,\\log_2[P(w_1,w_2)/(P(w_1)P(w_2))]\\}$ with probabilities from the
co-occurrence counts (joint = cell / total, marginals = row / column sums / total).
""")

code("""
from part1_counts import cooccurrence, ppmi, CosineIndex
t0 = time.time()
cooc = cooccurrence(sents, w2i, window=C.PPMI_WINDOW)
P = ppmi(cooc)
print(f"co-occurrence matrix {cooc.shape}, {cooc.nnz:,} non-zero cells, total count {int(cooc.sum()):,}, "
      f"symmetric: {(cooc != cooc.T).nnz == 0}")
print(f"PPMI matrix: {P.nnz:,} positive cells ({100*P.nnz/P.shape[0]**2:.2f}% dense), max {P.max():.2f} "
      f"({time.time()-t0:.1f}s)")
np.save(C.PPMI_NPY, P.toarray().astype(np.float32))
print(f"saved {C.PPMI_NPY.relative_to(Path.cwd())}: shape {P.shape}, float32, "
      f"{C.PPMI_NPY.stat().st_size/1e6:.0f} MB (dense, as .npy)")
""")

md("""
**t-SNE of the 200 most frequent tokens** (PPMI row vectors, cosine distance, perplexity 30). Colours
are hand-assigned semantic categories (`src/semantic_labels.py`). Plot (a) uses the 200 most frequent
tokens literally (punctuation and `<NUM>` excluded), which in Urdu news are dominated by function
words; plot (b) shows the 200 most frequent *content* words, where the topical structure is visible.
""")

code("""
from embed_eval import tsne_2d, plot_tsne
from semantic_labels import category, TOKEN2CAT
freq_words = [w for w, _ in counts.most_common() if not is_symbol(w) and w in w2i]
top200 = freq_words[:200]
content200 = [w for w in freq_words if category(w) not in ("Function word", "Verb")][:200]
fig, axes = plt.subplots(2, 1, figsize=(16, 24))
for ax, words, title in [(axes[0], top200, "(a) t-SNE of PPMI vectors: 200 most frequent tokens"),
                         (axes[1], content200, "(b) t-SNE of PPMI vectors: 200 most frequent content words")]:
    Y = tsne_2d(P[[w2i[w] for w in words]])
    plot_tsne(Y, words, [category(w) for w in words], title, ax=ax)
savefig(fig, "tsne_ppmi.png"); plt.show()
print("category counts (a):", Counter(category(w) for w in top200).most_common())
print("category counts (b):", Counter(category(w) for w in content200).most_common())
""")

md("""
**Top-5 nearest neighbours (cosine similarity of PPMI vectors) for 10 query words.** Candidate
neighbours are restricted to words that occur at least 5 times (the vectors of rarer words rest on
1–4 contexts and are unreliable); the same rule is used for every embedding in this notebook.
""")

code("""
from eval_sets import PPMI_QUERIES, QUERY_WORDS, COND_QUERIES, MRR_PAIRS, ANALOGIES, cleaned, cleaned_analogies
from word2vec import encode_corpus, unigram_counts
clean_counts = unigram_counts(encode_corpus(sents, w2i), len(w2i) - 1)
ppmi_idx = CosineIndex(P, w2i, counts=clean_counts, min_count=5)
rows = []
for q in PPMI_QUERIES:
    nn = ppmi_idx.neighbours(q, 5)
    rows.append({"query": with_gloss(q), **{f"#{i+1}": f"{w} ({s:.2f})" for i, (w, s) in enumerate(nn)}})
RESULTS["ppmi_neighbours"] = {r["query"]: [r[f"#{i}"] for i in range(1, 6)] for r in rows}
pd.DataFrame(rows)
""")

# -----------------------------------------------------------------------------
md(f"""
## 2.1 Skip-gram Word2Vec — implementation

* separate centre and context matrices **V**, **U** ∈ ℝ^(|V|×d), d = 100 (V ~ U(±0.5/d), U = 0 as in word2vec.c)
* noise distribution $P_n(w) \\propto f(w)^{{3/4}}$, realised as a 10⁷-slot unigram table; **K = 10** noise words per positive pair
* loss $L = -\\log\\sigma(u_o^\\top v_c) - \\sum_{{k=1}}^{{K}} \\log\\sigma(-u_{{w_k}}^\\top v_c)$ (binary cross-entropy)
* every (centre, context) pair with 1 ≤ |offset| ≤ **k = 5** inside a sentence; Adam (**η = 0.001**), batch **512**
* **{RUN['w2v_epochs']} epochs** (brief: ≥ 5). The number of epochs and whether to use word2vec's frequent-word
  sub-sampling were chosen *before* the final run on a separate development set of 180 analogies, disjoint
  from the 10 analogy tests reported below (`scripts/w2v_config_search.py`, results in
  `outputs/w2v_config_search.txt`): 15 epochs without sub-sampling had the best development top-3 accuracy.
""")

code("""
from word2vec import build_noise_table, noise_distribution, skipgram_pairs, train_skipgram
sent_ids = encode_corpus(sents, w2i)
table = build_noise_table(clean_counts)
pn = noise_distribution(clean_counts); uni = clean_counts / clean_counts.sum()
emp = np.bincount(table, minlength=len(clean_counts)) / len(table)
print(f"noise table: {len(table):,} slots; max |empirical - f^0.75 distribution| = {np.abs(emp - pn).max():.2e}")
c_ids, o_ids = skipgram_pairs(sent_ids, C.W2V_WINDOW)
print(f"training pairs per epoch: {len(c_ids):,}  (from {sum(map(len, sent_ids)):,} tokens, window k={C.W2V_WINDOW})")
s0 = sents[0]
c0, o0 = skipgram_pairs([np.array([w2i.get(t, w2i[C.UNK]) for t in s0])], C.W2V_WINDOW)
centre_pos = 6
ctx = [i2w[o] for c, o in zip(c0, o0) if c == w2i.get(s0[centre_pos], w2i[C.UNK])]
print("example sentence:", " ".join(s0))
print(f"its {len(c0)} training pairs include, for centre '{s0[centre_pos]}' (position {centre_pos}):", ctx)
pd.DataFrame([{"word": w, "gloss": gloss(w), "count": int(clean_counts[w2i[w]]),
               "unigram P(w)": f"{uni[w2i[w]]:.5f}", "noise P_n(w)": f"{pn[w2i[w]]:.5f}",
               "P_n / P": round(pn[w2i[w]] / uni[w2i[w]], 2)}
              for w in ["کے", "ہے", "پاکستان", "حکومت", "کرکٹ", "معیشت", "ماسکو"]])
""")

code(f"""
corpus_c3 = sents if RUN["w2v_limit"] is None else sents[:RUN["w2v_limit"]]
torch.set_num_threads(4)
res_c3 = train_skipgram(corpus_c3, w2i, dim=C.W2V_DIM, epochs=RUN["w2v_epochs"], subsample_t=C.W2V_SUBSAMPLE,
                        log_every=200, print_every=5000)
E_c3 = res_c3["embeddings"]
np.save(C.W2V_NPY, E_c3.astype(np.float32))
print(f"saved {{C.W2V_NPY.relative_to(Path.cwd())}}: 1/2(V+U), shape {{E_c3.shape}}; training took {{res_c3['seconds']/60:.1f}} min")
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(res_c3["log_steps"], res_c3["step_losses"], lw=0.8)
ax[0].set_title("Skip-gram (C3) training loss, mean over every 200 batches"); ax[0].set_xlabel("training step (batch of 512 pairs)"); ax[0].set_ylabel("BCE loss")
ax[1].plot(range(1, len(res_c3["epoch_losses"]) + 1), res_c3["epoch_losses"], marker="o")
ax[1].set_title("Skip-gram (C3) mean training loss per epoch"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("BCE loss")
savefig(fig, "w2v_loss_curve.png"); plt.show()
RESULTS["w2v_epoch_losses"] = res_c3["epoch_losses"]
print("epoch losses:", [round(x, 4) for x in res_c3["epoch_losses"]])
""")

md("""
## 2.2 Evaluation

### Nearest neighbours and analogy

The Roman-Urdu query words are mapped to Urdu script and then through the A1 pipeline. Note
*Aabadi* (آبادی) becomes `آباد` after A1 stemming — the same token as the *-abad* of city names
(اسلام آباد، ایبٹ آباد), so its neighbours are dominated by places.
""")

code("""
idx_c3 = CosineIndex(E_c3, w2i, counts=clean_counts, min_count=5)
rows = []
for roman, urdu in QUERY_WORDS.items():
    q = cleaned(urdu)
    nn = idx_c3.neighbours(q, 10)
    rows.append({"query": roman, "Urdu": urdu, "A1 form": q, "freq": int(clean_counts[w2i[q]]),
                 "top-10 neighbours (cosine)": "، ".join(f"{w} {s:.2f}" for w, s in nn)})
RESULTS["w2v_neighbours"] = {r["query"]: r["top-10 neighbours (cosine)"] for r in rows}
pd.DataFrame(rows)
""")

md("""
**Analogy tests** `a : b :: c : ?` answered with $v(b)-v(a)+v(c)$ (unit vectors, cosine; a, b, c
excluded). The 10 tests were fixed in advance (`src/eval_sets.py`): each covers a sensible relation
and each of its two word pairs co-occurs at least 3 times in the corpus. A test counts as **correct
when the expected word is among the reported top-3 candidates** (top-1 is shown as well).
""")

code("""
from embed_eval import run_analogies, mean_reciprocal_rank
an_rows = run_analogies(idx_c3, cleaned_analogies(), k=3)
tab = pd.DataFrame([{"relation": r["relation"], "analogy": f"{r['a']} : {r['b']} :: {r['c']} : ?",
                     "expected": with_gloss(r["expected"]),
                     "top-3 candidates": "، ".join(f"{w} {s:.2f}" for w, s in r["top"]),
                     "top-1": "✓" if r["top1"] else "✗", "in top-3": "✓" if r["top3"] else "✗"}
                    for r in an_rows])
n1, n3 = sum(r["top1"] for r in an_rows), sum(r["top3"] for r in an_rows)
RESULTS["analogies"] = {"top1": n1, "top3": n3, "rows": tab.to_dict("records")}
print(f"correct: {n3}/10 (expected word in top-3), {n1}/10 at top-1")
tab
""")

md("ANALYSIS_EMBEDDINGS")

md("""
### Four-condition comparison

| ID | Condition | Description |
|---|---|---|
| C1 | PPMI baseline | PPMI-weighted co-occurrence vectors (Section 1.2) |
| C2 | Skip-gram on `raw.txt` | the unprocessed corpus: sentence split + whitespace tokens, no normalisation / lemmatization / stemming |
| C3 | Skip-gram on `cleaned.txt` | the model above |
| C4 | Skip-gram, d = 200 | C3 with doubled embedding dimension |

C2 and C4 use exactly the C3 training settings. MRR is computed on 20 manually labelled
related word pairs (query → related word): the rank of the related word among all candidates by
cosine similarity to the query; for C2 the pairs are looked up in their raw surface spelling.
""")

code(f"""
raw_sents = flatten(load_raw_sentences_unprocessed())
w2i_raw = build_vocab(raw_sents)
raw_counts = unigram_counts(encode_corpus(raw_sents, w2i_raw), len(w2i_raw) - 1)
print(f"raw corpus: {{len(raw_sents):,}} sentences, {{sum(map(len, raw_sents)):,}} tokens, "
      f"{{len(token_counts(raw_sents)):,}} types (cleaned: {{len(counts):,}})")
corpus_c2 = raw_sents if RUN["w2v_limit"] is None else raw_sents[:RUN["w2v_limit"]]
res_c2 = train_skipgram(corpus_c2, w2i_raw, dim=C.W2V_DIM, epochs=RUN["w2v_epochs"], subsample_t=C.W2V_SUBSAMPLE, verbose=True)
res_c4 = train_skipgram(corpus_c3, w2i, dim=200, epochs=RUN["w2v_epochs"], subsample_t=C.W2V_SUBSAMPLE, verbose=True)
print(f"C2 trained in {{res_c2['seconds']/60:.1f}} min, C4 in {{res_c4['seconds']/60:.1f}} min")
""")

code("""
conds = {"C1 PPMI": (CosineIndex(P, w2i, counts=clean_counts, min_count=5), "clean"),
         "C2 skip-gram raw": (CosineIndex(res_c2["embeddings"], w2i_raw, counts=raw_counts, min_count=5), "raw"),
         "C3 skip-gram cleaned": (idx_c3, "clean"),
         "C4 skip-gram d=200": (CosineIndex(res_c4["embeddings"], w2i, counts=clean_counts, min_count=5), "clean")}
def form(word, kind): return cleaned(word) if kind == "clean" else word
nn_rows, mrr_rows = [], []
for name, (idx, kind) in conds.items():
    for q in COND_QUERIES:
        qq = form(q, kind)
        nn = idx.neighbours(qq, 5) if qq in idx.w2i else []
        nn_rows.append({"condition": name, "query": qq, "top-5 neighbours": "، ".join(w for w, _ in nn)})
    ranks = []
    for a, b in MRR_PAIRS:
        qa, qb = form(a, kind), form(b, kind)
        ranks.append(idx.rank_of(qa, qb) if qa in idx.w2i and qb in idx.w2i else None)
    rr = [1.0 / r if r else 0.0 for r in ranks]
    mrr_rows.append({"condition": name, "MRR": round(float(np.mean(rr)), 4),
                     "pairs found": sum(r is not None for r in ranks),
                     "median rank": int(np.median([r for r in ranks if r])),
                     "related word ranked 1st": sum(r == 1 for r in ranks),
                     "in top-10": sum(1 for r in ranks if r and r <= 10)})
mrr_df = pd.DataFrame(mrr_rows)
RESULTS["mrr"] = mrr_df.to_dict("records")
display(pd.DataFrame(nn_rows).pivot(index="query", columns="condition", values="top-5 neighbours"))
mrr_df
""")

code("""
pair_tab = pd.DataFrame([{"query": with_gloss(cleaned(a)), "related": with_gloss(cleaned(b)),
                          **{name: (idx.rank_of(form(a, k), form(b, k)) if form(a, k) in idx.w2i and form(b, k) in idx.w2i else "n/a")
                             for name, (idx, k) in conds.items()}} for a, b in MRR_PAIRS])
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(mrr_df["condition"], mrr_df["MRR"], color=PALETTE[:4])
for i, v in enumerate(mrr_df["MRR"]): ax.text(i, v + 0.005, f"{v:.3f}", ha="center")
ax.set_title("Mean reciprocal rank on 20 related word pairs"); ax.set_xlabel("condition"); ax.set_ylabel("MRR")
savefig(fig, "mrr_conditions.png"); plt.show()
pair_tab
""")

md("ANALYSIS_CONDITIONS")

# =============================================================================
md("""
# Part 2 — Sequence Labelling: POS Tagging & NER

## 3. Dataset preparation

1. **500 sentences** are drawn at random (seed 42) from `cleaned.txt`: **100 from each of the 5 topic
   categories** (so ≥ 100 from each of ≥ 3 categories); sentences with 5–50 tokens, no duplicates.
2. **POS** — rule-based tagger (`src/pos_tagger.py`) built on the A1 lemmatizer/stemmer and a hand-crafted
   lexicon (`src/pos_lexicon.py`): lookup of the surface form, then of its **A1 lemma**, then of its **A1 stem**
   (only stems that are unambiguous in the lexicon), then morphology with the A1 suffix inventory (verbal
   endings on a known verb root → VERB, plural/abstract endings → NOUN, nisba *-ی* on a known noun → ADJ),
   then context rules for ambiguous function words (میں PRON/POST, کی POST/VERB, کیا PRON/VERB, لیے POST/VERB,
   یہ/وہ/اس/ان DET/PRON, ہو/جا/رہ VERB/AUX, …) and for unknown words (before a case marker or light verb, after a
   determiner → NOUN, else UNK). The brief says "12 tags" but lists 11; **AUX** (auxiliaries ہے، تھا، گا، رہا،
   سکتا، چکا, passive جا) was added as the 12th tag.
3. **NER** — seed gazetteer (`src/ner_gazetteer.py`) matched on the original words, longest match first, plus a
   given-name + name-part pattern for persons; BIO tags projected onto the cleaned tokens.
4. **Split** 70/15/15 inside every topic (stratified): 350 / 75 / 75 sentences.
""")

code("""
from pos_tagger import lexicon_entry_counts
from ner_gazetteer import coverage
from part2_data import build_and_save, label_distribution, read_conll, SPLITS
lex = lexicon_entry_counts()
display(pd.DataFrame([lex], index=["hand-crafted lexicon entries"]))
display(pd.DataFrame([coverage()], index=["gazetteer entries"]))
splits = build_and_save()
print({s: len(v) for s, v in splits.items()}, "| sentences per topic and split:")
pd.DataFrame({s: Counter(x["topic"] for x in splits[s]) for s in SPLITS}).loc[C.CATEGORIES]
""")

code("""
for x in splits["train"][:3]:
    print(f"[{x['topic']}] article {x['article_number']}")
    print("  POS:", " ".join(f"{t}/{p}" for t, p in zip(x["tokens"], x["pos"])))
    print("  NER:", " ".join(f"{t}/{n}" for t, n in zip(x["tokens"], x["ner"]) if n != "O") or "(no entity)")
""")

code("""
pos_dist = pd.DataFrame(label_distribution(splits, "pos")).reindex(C.POS_TAGS).fillna(0).astype(int)
ner_dist = pd.DataFrame(label_distribution(splits, "ner")).reindex(C.NER_TAGS).fillna(0).astype(int)
pos_dist["total"] = pos_dist.sum(axis=1); ner_dist["total"] = ner_dist.sum(axis=1)
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
pos_dist[list(SPLITS)].plot.bar(ax=ax[0]); ax[0].set_title("POS tag distribution per split"); ax[0].set_xlabel("tag"); ax[0].set_ylabel("tokens")
ner_dist.drop(index="O")[list(SPLITS)].plot.bar(ax=ax[1]); ax[1].set_title("NER tag distribution per split (O omitted)"); ax[1].set_xlabel("tag"); ax[1].set_ylabel("tokens")
savefig(fig, "part2_label_distribution.png"); plt.show()
RESULTS["pos_distribution"] = pos_dist.to_dict(); RESULTS["ner_distribution"] = ner_dist.to_dict()
display(pos_dist); ner_dist
""")

md("""
## 4. BiLSTM sequence labeller

`src/seq_models.py`: embedding (initialised with the C3 Word2Vec vectors, `<PAD>` row = 0) → **2-layer
bidirectional LSTM** (hidden 128 per direction, **dropout 0.5 between the layers**) → per-token
representation $h_t=[\\overrightarrow{h_t};\\overleftarrow{h_t}]$ → linear layer.
**POS:** softmax + cross-entropy. **NER:** **CRF** with a learnable tag-transition matrix (plus start/end
scores), trained with the forward algorithm and decoded with **Viterbi**. Variable-length batches are
packed; padded positions carry label −100 (ignored by cross-entropy) and mask = 0 (skipped by the CRF),
so **padding never contributes to the loss** (unit-tested). Adam (η = 10⁻³, weight decay 10⁻⁴), batch 16,
**early stopping on validation F1 with patience 5** (POS: token macro-F1; NER: entity F1 from
conlleval; an exact F1 tie counts as progress only if the validation loss fell), best epoch restored.

Every configuration is trained with 3 seeds (42, 7, 123). For each task the embedding mode (frozen or
fine-tuned) with the higher **mean validation F1 over the seeds** is selected; its seed-42 model is saved and
analysed in Section 5, and mean ± std over the seeds is given wherever configurations are compared.
""")

code("""
from train_seq import SeqConfig, encode_data, train, run_eval
from seq_models import BiLSTMTagger
emb = np.vstack([E_c3, np.zeros((1, E_c3.shape[1]), dtype=np.float32)])     # + <PAD> row
pad_idx = w2i[C.PAD]
assert emb.shape[0] == len(w2i) and pad_idx == emb.shape[0] - 1
DATA2 = {}
for task, tags in (("pos", C.POS_TAGS), ("ner", C.NER_TAGS)):
    t2i = {t: i for i, t in enumerate(tags)}
    DATA2[task] = {s: encode_data(*read_conll(C.DATA / f"{task}_{s}.conll"), w2i, t2i) for s in SPLITS}
unk_rate = np.mean([i == w2i[C.UNK] for s in DATA2["pos"]["train"].x for i in s])
print(f"embedding matrix {emb.shape} (C3 + <PAD>); <UNK> rate in Part-2 train tokens: {100*unk_rate:.2f}%")
print(BiLSTMTagger(len(C.NER_TAGS), emb.shape[0], 100, pretrained=emb, pad_idx=pad_idx, use_crf=True))
""")

code("""
TAGSETS = {"pos": C.POS_TAGS, "ner": C.NER_TAGS}
EXPERIMENTS = {   # name -> (task, config overrides)
    "POS frozen":            ("pos", dict(freeze=True)),
    "POS fine-tuned":        ("pos", dict(freeze=False)),
    "POS A1 unidirectional": ("pos", dict(bidirectional=False)),
    "POS A2 no dropout":     ("pos", dict(dropout=0.0)),
    "POS A3 random init":    ("pos", dict(pretrained=False)),
    "NER frozen":            ("ner", dict(freeze=True)),
    "NER fine-tuned (CRF)":  ("ner", dict(freeze=False)),
    "NER A1 unidirectional": ("ner", dict(bidirectional=False)),
    "NER A2 no dropout":     ("ner", dict(dropout=0.0)),
    "NER A3 random init":    ("ner", dict(pretrained=False)),
    "NER A4 softmax (no CRF)": ("ner", dict(use_crf=False)),
}
MAIN = ["POS frozen", "POS fine-tuned", "NER frozen", "NER fine-tuned (CRF)"]
RUNS = {}
t_all = time.time()
for seed in RUN["seeds"]:
    for name, (task, over) in EXPERIMENTS.items():
        cfg = SeqConfig(task=task, name=name, seed=seed, max_epochs=RUN["seq_max_epochs"], **over)
        verbose = seed == 42 and name in MAIN
        if verbose: print(f"=== {name} (seed {seed}) ===")
        r = train(cfg, DATA2[task]["train"], DATA2[task]["val"], TAGSETS[task], emb, pad_idx, verbose=verbose)
        _, test_pred, test_f1 = run_eval(r["model"], DATA2[task]["test"], cfg, pad_idx, TAGSETS[task])
        r.update(test_pred=test_pred, test_f1=test_f1)
        RUNS[(name, seed)] = r
        print(f"{name:26s} seed {seed:3d}: best epoch {r['best_epoch']:2d}/{r['epochs_run']:2d}  "
              f"val F1 {r['best_val_f1']:6.2f}  test F1 {test_f1:6.2f}  "
              f"({np.mean(r['history']['epoch_seconds']):.1f}s/epoch)")
print(f"all {len(RUNS)} runs finished in {(time.time()-t_all)/60:.1f} min")
""")

code("""
def ms(vals): return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
summary = []
for name, (task, _) in EXPERIMENTS.items():
    rs = [RUNS[(name, s)] for s in RUN["seeds"]]
    summary.append({"model": name, "metric": "macro-F1" if task == "pos" else "entity F1",
                    "val F1 (seed 42)": round(RUNS[(name, 42)]["best_val_f1"], 2),
                    "test F1 (seed 42)": round(RUNS[(name, 42)]["test_f1"], 2),
                    "val F1 mean ± std": ms([r["best_val_f1"] for r in rs]),
                    "test F1 mean ± std": ms([r["test_f1"] for r in rs]),
                    "best epoch (seeds)": [r["best_epoch"] for r in rs]})
summary_df = pd.DataFrame(summary)
RESULTS["seq_summary"] = summary_df.to_dict("records")
print("Validation F1 of the frozen and fine-tuned embedding modes (Section 4):")
summary_df[summary_df.model.isin(MAIN)]
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, name in zip(axes, MAIN):
    h = RUNS[(name, 42)]["history"]; ep = range(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_loss"], marker=".", label="train loss"); ax.plot(ep, h["val_loss"], marker=".", label="validation loss")
    ax.axvline(RUNS[(name, 42)]["best_epoch"], color="grey", ls="--", lw=1, label="best val-F1 epoch")
    ax.set_title(name + " (seed 42)"); ax.set_xlabel("epoch"); ax.set_ylabel("CE loss" if name.startswith("POS") else "CRF NLL / sentence"); ax.legend()
savefig(fig, "part2_training_curves.png"); plt.show()
mean_val = {n: float(np.mean([RUNS[(n, s)]["best_val_f1"] for s in RUN["seeds"]])) for n in MAIN}
best_mode = {t: max([n for n in MAIN if n.lower().startswith(t)], key=lambda n: mean_val[n]) for t in ("pos", "ner")}
print("mean validation F1 over seeds:", {n: round(v, 2) for n, v in mean_val.items()})
print("selected embedding mode per task (highest mean validation F1):", best_mode)
""")

code("""
for task, fname in (("pos", "bilstm_pos.pt"), ("ner", "bilstm_ner.pt")):
    name = best_mode[task]; r = RUNS[(name, 42)]
    torch.save({"state_dict": r["model"].state_dict(), "config": r["config"], "tags": TAGSETS[task],
                "pad_idx": pad_idx, "vocab": "embeddings/word2idx.json", "val_f1": r["best_val_f1"],
                "test_f1": r["test_f1"], "best_epoch": r["best_epoch"]}, C.MODELS / fname)
    print(f"saved models/{fname}  <- {name} (seed 42)")
""")

md("""
## 5. Evaluation

### 5.1 POS tagging (test set, fine-tuned model, seed 42)
""")

code("""
from metrics import accuracy, macro_f1, confusion_matrix, per_class_prf, most_confused_pairs
from part2_eval import confused_examples, entity_errors
pos_name = best_mode["pos"]
pos_gold = DATA2["pos"]["test"].tags; pos_pred = RUNS[(pos_name, 42)]["test_pred"]
g = [t for s in pos_gold for t in s]; p = [t for s in pos_pred for t in s]
acc, mf1 = 100 * accuracy(g, p), 100 * macro_f1(g, p)
RESULTS["pos_test"] = {"model": pos_name, "accuracy": acc, "macro_f1": mf1, "tokens": len(g)}
print(f"{pos_name}: token accuracy {acc:.2f}%, macro-F1 {mf1:.2f} over {len(set(g) | set(p))} tags ({len(g):,} test tokens)")
prf = per_class_prf(g, p, C.POS_TAGS)
pd.DataFrame(prf).T.round(3)
""")

code("""
cm = confusion_matrix(g, p, C.POS_TAGS)
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(np.log1p(cm), cmap="Blues")
for i in range(len(C.POS_TAGS)):
    for j in range(len(C.POS_TAGS)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 3 else "black")
ax.set_xticks(range(12)); ax.set_xticklabels(C.POS_TAGS, rotation=45); ax.set_yticks(range(12)); ax.set_yticklabels(C.POS_TAGS)
ax.set_xlabel("predicted tag"); ax.set_ylabel("gold tag"); ax.set_title(f"POS confusion matrix, test set ({pos_name}); colour = log(1+count)")
savefig(fig, "pos_confusion_matrix.png"); plt.show()
pairs3 = most_confused_pairs(cm, C.POS_TAGS, 3)
RESULTS["pos_confused_pairs"] = pairs3
print("3 most confused tag pairs (both directions):", pairs3)
""")

code("""
test_tokens = DATA2["pos"]["test"].tokens
for a, b, n in pairs3:
    print(f"--- {a} <-> {b}: {n} confusions ---")
    for ex in confused_examples(test_tokens, pos_gold, pos_pred, (a, b), k=2):
        print("  •", ex)
""")

md("ANALYSIS_POS_CONFUSIONS")

code("""
rows = []
for name in ("POS frozen", "POS fine-tuned"):
    accs, f1s = [], []
    for s in RUN["seeds"]:
        pp = [t for x in RUNS[(name, s)]["test_pred"] for t in x]
        accs.append(100 * accuracy(g, pp)); f1s.append(100 * macro_f1(g, pp))
    rows.append({"embedding mode": name, "val macro-F1 (seed 42)": round(RUNS[(name, 42)]["best_val_f1"], 2),
                 "test accuracy (seed 42)": round(accs[0], 2), "test macro-F1 (seed 42)": round(f1s[0], 2),
                 "test accuracy mean ± std": ms(accs), "test macro-F1 mean ± std": ms(f1s),
                 "epochs to best (seed 42)": RUNS[(name, 42)]["best_epoch"]})
pos_modes = pd.DataFrame(rows); RESULTS["pos_modes"] = pos_modes.to_dict("records")
pos_modes
""")

md("""
### 5.2 NER (test set, entity level, conlleval)

The test predictions are written in CoNLL format and scored with the **original `conlleval` Perl script**
(`tools/conlleval.pl`, Tjong Kim Sang, version 2004-01-26); `src/conlleval.py` is a port used for the
per-type table and during training (the two agree exactly — unit-tested).
""")

code("""
import conlleval
ner_name = best_mode["ner"]
ner_tokens, ner_gold = DATA2["ner"]["test"].tokens, DATA2["ner"]["test"].tags
ner_pred = RUNS[(ner_name, 42)]["test_pred"]
print(f"conlleval.pl on {ner_name} (seed 42):")
print(conlleval.run_perl(ner_tokens, ner_gold, ner_pred))
ev = conlleval.evaluate(ner_gold, ner_pred)
tab = pd.DataFrame(ev["per_type"]).T[["precision", "recall", "f1", "gold", "found", "correct"]]
tab.loc["overall"] = [ev["precision"], ev["recall"], ev["f1"], ev["gold_phrases"], ev["found_phrases"], ev["correct_phrases"]]
RESULTS["ner_test"] = {"model": ner_name, **{k: ev[k] for k in ("precision", "recall", "f1")}, "per_type": ev["per_type"]}
tab.round(2)
""")

code("""
rows = []
for name in ("NER fine-tuned (CRF)", "NER A4 softmax (no CRF)"):
    evs = [conlleval.evaluate(ner_gold, RUNS[(name, s)]["test_pred"]) for s in RUN["seeds"]]
    rows.append({"output layer": "CRF + Viterbi" if "CRF" in name and "no CRF" not in name else "softmax (per-token argmax)",
                 "P (seed 42)": round(evs[0]["precision"], 2), "R (seed 42)": round(evs[0]["recall"], 2),
                 "F1 (seed 42)": round(evs[0]["f1"], 2), "F1 mean ± std": ms([e["f1"] for e in evs]),
                 "P mean": round(np.mean([e["precision"] for e in evs]), 2), "R mean": round(np.mean([e["recall"] for e in evs]), 2),
                 **{f"{t} F1 (seed 42)": round(evs[0]["per_type"].get(t, {}).get("f1", 0.0), 2) for t in ("PER", "LOC", "ORG", "MISC")}})
crf_df = pd.DataFrame(rows); RESULTS["ner_crf_vs_softmax"] = crf_df.to_dict("records")
print("Softmax model (seed 42), conlleval.pl:")
print(conlleval.run_perl(ner_tokens, ner_gold, RUNS[("NER A4 softmax (no CRF)", 42)]["test_pred"]))
crf_df
""")

code("""
fps, fns = entity_errors(ner_tokens, ner_gold, ner_pred)
print(f"{len(fps)} false-positive and {len(fns)} false-negative entities on the test set ({ner_name}, seed 42)")
fp_df = pd.DataFrame(fps[:5]); fn_df = pd.DataFrame(fns[:5])
RESULTS["ner_fp"] = fps[:5]; RESULTS["ner_fn"] = fns[:5]
print("False positives (predicted entity not in the gold annotation):"); display(fp_df)
print("False negatives (gold entity not predicted exactly):"); fn_df
""")

md("ANALYSIS_NER_ERRORS")

md("""
### 5.3 Ablation study

Each ablation changes one thing relative to the fine-tuned model of Section 4 and is trained on the same
split with the same 3 seeds: **A1** unidirectional (forward) 2-layer LSTM, **A2** no dropout,
**A3** random embedding initialisation (instead of C3 Word2Vec), **A4** softmax output instead of the CRF (NER).
""")

code("""
abl = []
for task, base in (("pos", "POS fine-tuned"), ("ner", "NER fine-tuned (CRF)")):
    base_f1 = [RUNS[(base, s)]["test_f1"] for s in RUN["seeds"]]
    for name, (t, _) in EXPERIMENTS.items():
        if t != task or not (name == base or " A" in name): continue
        f1s = [RUNS[(name, s)]["test_f1"] for s in RUN["seeds"]]
        vf = [RUNS[(name, s)]["best_val_f1"] for s in RUN["seeds"]]
        abl.append({"task": task.upper(), "model": name, "val F1 mean ± std": ms(vf), "test F1 mean ± std": ms(f1s),
                    "Δ test F1 vs full model": "—" if name == base else f"{np.mean(f1s) - np.mean(base_f1):+.2f}",
                    "test F1 per seed": [round(x, 1) for x in f1s]})
abl_df = pd.DataFrame(abl); RESULTS["ablations"] = abl_df.to_dict("records")
abl_df
""")

md("ANALYSIS_ABLATIONS")

# =============================================================================
md("""
# Part 3 — Transformer Encoder for Topic Classification

## 6. Dataset preparation

Every article (labels from Section 0.1) is represented by the token ids of its first **256** tokens in
`cleaned.txt` (shorter articles are padded with `<PAD>`), using the shared 10k vocabulary. Split 70/15/15
**stratified by category** (seed 42).
""")

code("""
from part3 import build_dataset, subset, make_transformer, train_classifier, evaluate, BiLSTMClassifier
data3 = build_dataset(w2i)
d3 = pd.DataFrame({s: Counter(C.CATEGORIES[i] for i in data3["y"][data3["split"] == s]) for s in SPLITS}).loc[C.CATEGORIES]
d3["total"] = d3.sum(axis=1)
L = data3["full_lengths"]
print(f"article length: min {L.min()}, median {int(np.median(L))}, max {L.max()} tokens; "
      f"{(L > C.MAX_LEN).sum()} of 250 articles truncated to {C.MAX_LEN}, {(L < C.MAX_LEN).sum()} padded")
RESULTS["part3_distribution"] = d3.to_dict()
fig, ax = plt.subplots(figsize=(8, 3.5))
d3[list(SPLITS)].plot.bar(ax=ax); ax.set_title("Topic classification: articles per category and split")
ax.set_xlabel("category"); ax.set_ylabel("articles"); plt.xticks(rotation=0)
savefig(fig, "part3_class_distribution.png"); plt.show()
d3
""")

md("""
## 7. Transformer encoder (from scratch, `src/transformer.py`)

* `ScaledDotProductAttention` — softmax(QKᵀ/√d_k)V with an optional padding mask; returns output **and** weights
* `MultiHeadSelfAttention` — **h = 4** heads, d_model = 128, **separate** W_Q, W_K, W_V (128→32) per head, shared W_O (128→128)
* `PositionwiseFeedForward` — Linear(128→512) → ReLU → Dropout → Linear(512→128)
* `SinusoidalPositionalEncoding` — fixed `register_buffer`, added to the input embeddings
* `EncoderBlock` ×4 with **Pre-LayerNorm**: x ← x + Dropout(MHA(LN(x))); x ← x + Dropout(FFN(LN(x)))
* learned **[CLS]** vector prepended to every sequence → final LayerNorm → MLP **128 → 64 → 5**

Training: **AdamW** (η = 5×10⁻⁴, weight decay 0.01), **cosine schedule with 50 linear warm-up steps**,
**20 epochs**, batch 16, dropout 0.1, cross-entropy. The checkpoint of the epoch with the best validation
accuracy is used for testing.
""")

code("""
torch.manual_seed(C.SEED)
tf_model = make_transformer(w2i)
print(tf_model)
n_params = sum(p.numel() for p in tf_model.parameters())
print(f"trainable parameters: {n_params:,} (of which embeddings {tf_model.embed.weight.numel():,}); "
      f"positional encoding is a buffer: {'pos.pe' in dict(tf_model.named_buffers())}, "
      f"in parameters: {any('pe' in n for n, _ in tf_model.named_parameters())}")
""")

code("""
res_tf = train_classifier(tf_model, data3, "transformer", RUN["tf_epochs"], C.TF_LR, C.TF_WEIGHT_DECAY, C.TF_BATCH, seed=C.SEED)
torch.save({"state_dict": res_tf["model"].state_dict(), "best_epoch": res_tf["best_epoch"],
            "categories": C.CATEGORIES, "pad_idx": w2i[C.PAD], "vocab": "embeddings/word2idx.json"},
           C.MODELS / "transformer_cls.pt")
print(f"best validation accuracy {res_tf['best_val_acc']:.1f}% at epoch {res_tf['best_epoch']}; saved models/transformer_cls.pt")
""")

code("""
h = res_tf["history"]; ep = range(1, len(h["train_loss"]) + 1)
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ax[0].plot(ep, h["train_loss"], marker="o", label="train"); ax[0].plot(ep, h["val_loss"], marker="o", label="validation")
ax[0].set_title("Transformer: cross-entropy loss per epoch"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
ax[1].plot(ep, h["train_acc"], marker="o", label="train"); ax[1].plot(ep, h["val_acc"], marker="o", label="validation")
ax[1].set_title("Transformer: accuracy per epoch"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy (%)"); ax[1].legend()
ax[2].plot(range(1, len(h["lr"]) + 1), h["lr"]); ax[2].axvline(C.TF_WARMUP, color="grey", ls="--", label="end of warm-up")
ax[2].set_title("Learning rate: 50 warm-up steps + cosine decay"); ax[2].set_xlabel("optimizer step"); ax[2].set_ylabel("learning rate"); ax[2].legend()
savefig(fig, "transformer_training_curves.png"); plt.show()
print("val accuracy per epoch:", [round(x, 1) for x in h["val_acc"]])
""")

md("""
## 8. Evaluation

### 8.1 Results (test set, best-validation checkpoint, seed 42)
""")

code("""
Xte, yte, Lte, test_idx = subset(data3, "test")
ev_tf = evaluate(res_tf["model"], Xte, yte, Lte)
final_model = make_transformer(w2i); final_model.load_state_dict(res_tf["final_state"])
ev_tf_final = evaluate(final_model, Xte, yte, Lte)
RESULTS["transformer_test"] = {"accuracy": ev_tf["accuracy"], "macro_f1": ev_tf["macro_f1"], "best_epoch": res_tf["best_epoch"],
                               "final_epoch_accuracy": ev_tf_final["accuracy"], "final_epoch_macro_f1": ev_tf_final["macro_f1"]}
print(f"Transformer test accuracy {ev_tf['accuracy']:.2f}%, macro-F1 {ev_tf['macro_f1']:.2f}  ({len(yte)} test articles)")
print(f"(model after the last epoch: accuracy {ev_tf_final['accuracy']:.2f}%, macro-F1 {ev_tf_final['macro_f1']:.2f})")
cm5 = ev_tf["confusion"]
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(cm5, cmap="Blues")
for i in range(5):
    for j in range(5): ax.text(j, i, cm5[i, j], ha="center", va="center", color="white" if cm5[i, j] > cm5.max() / 2 else "black")
ax.set_xticks(range(5)); ax.set_xticklabels(C.CATEGORIES, rotation=30); ax.set_yticks(range(5)); ax.set_yticklabels(C.CATEGORIES)
ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title("Transformer: 5 x 5 confusion matrix (test)")
savefig(fig, "transformer_confusion_matrix.png"); plt.show()
pd.DataFrame(per_class_prf(yte.tolist(), ev_tf["pred"], list(range(5))), ).T.set_axis(C.CATEGORIES).round(3)
""")

md("""
**Attention heatmaps** — final encoder layer, all 4 heads, for 3 correctly classified test articles.
Each map shows the first 24 positions ([CLS] + 23 tokens; the full 257 × 257 map is unreadable); the bar
chart lists the 12 tokens the [CLS] position attends to most over the whole sequence (head average).
""")

code("""
model_tf = res_tf["model"].eval()
correct = [k for k in range(len(yte)) if ev_tf["pred"][k] == int(yte[k])]
chosen = []
for c in range(5):                    # prefer 3 different categories
    for k in correct:
        if int(yte[k]) == c and len(chosen) < 3: chosen.append(k); break
for k in correct:
    if len(chosen) < 3 and k not in chosen: chosen.append(k)
SHOW = 24
for k in chosen:
    art_no = int(test_idx[k]) + 1
    with torch.no_grad(): _, att = model_tf(Xte[k:k+1], return_attention=True)
    A = att[-1][0].numpy()                                   # (heads, T+1, T+1)
    toks = ["[CLS]"] + [i2w[int(t)] for t in Xte[k][:SHOW - 1]]
    fig, axes = plt.subplots(1, 5, figsize=(30, 6.2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.8]})
    for hd in range(4):
        ax = axes[hd]; ax.imshow(A[hd, :SHOW, :SHOW], cmap="viridis")
        ax.set_xticks(range(SHOW)); ax.set_xticklabels(toks, rotation=90, fontsize=9)
        ax.set_yticks(range(SHOW)); ax.set_yticklabels(toks, fontsize=9)
        ax.set_title(f"layer 4, head {hd + 1}"); ax.set_xlabel("attended token (key)"); ax.set_ylabel("query token")
    n_real = int(Lte[k]); cls_att = A[:, 0, 1:n_real + 1].mean(0)
    top = np.argsort(-cls_att)[:12]
    axes[4].barh(range(len(top)), cls_att[top][::-1], color=PALETTE[0])
    axes[4].set_yticks(range(len(top))); axes[4].set_yticklabels([f"[{j+1}]  {i2w[int(Xte[k][j])]}" for j in top][::-1], fontsize=10)
    axes[4].set_title("[CLS] attention (mean of 4 heads), top 12"); axes[4].set_xlabel("attention weight"); axes[4].set_ylabel("[position]  token")
    fig.subplots_adjust(wspace=0.35)
    fig.suptitle(f"Article {art_no} — gold = predicted = {C.CATEGORIES[int(yte[k])]} — {meta[art_no-1]['title'][:70]}", fontsize=13)
    savefig(fig, f"attention_article_{art_no}.png"); plt.show()
    print(f"article {art_no} ({C.CATEGORIES[int(yte[k])]}): top [CLS]-attended tokens:",
          "، ".join(f"{i2w[int(Xte[k][j])]}" for j in top))
""")

md("""
### 8.2 BiLSTM vs Transformer

For a like-for-like comparison a BiLSTM classifier is trained on the **same** split and inputs:
C3-initialised embeddings (fine-tuned) → 2-layer BiLSTM (128 per direction, dropout 0.5 between layers)
→ masked mean + max pooling → MLP (512 → 64 → 5); Adam (10⁻³, wd 10⁻⁴), 20 epochs, batch 16,
best-validation checkpoint. Both models are run with the 3 seeds; the seed-42 runs are shown in detail.
""")

code("""
torch.manual_seed(C.SEED)
bl = BiLSTMClassifier(emb, pad_idx)
res_bl = train_classifier(bl, data3, "bilstm", RUN["tf_epochs"], C.SEQ_LR, C.SEQ_WEIGHT_DECAY, C.TF_BATCH, seed=C.SEED)
ev_bl = evaluate(res_bl["model"], Xte, yte, Lte)
print(f"BiLSTM test accuracy {ev_bl['accuracy']:.2f}%, macro-F1 {ev_bl['macro_f1']:.2f}; best val epoch {res_bl['best_epoch']}")
""")

code("""
cmp_runs = {"Transformer": {42: (res_tf, ev_tf)}, "BiLSTM": {42: (res_bl, ev_bl)}}
for seed in RUN["seeds"][1:]:
    torch.manual_seed(seed); m = make_transformer(w2i)
    r = train_classifier(m, data3, "transformer", RUN["tf_epochs"], C.TF_LR, C.TF_WEIGHT_DECAY, C.TF_BATCH, seed=seed, verbose=False)
    cmp_runs["Transformer"][seed] = (r, evaluate(r["model"], Xte, yte, Lte))
    torch.manual_seed(seed); m = BiLSTMClassifier(emb, pad_idx)
    r = train_classifier(m, data3, "bilstm", RUN["tf_epochs"], C.SEQ_LR, C.SEQ_WEIGHT_DECAY, C.TF_BATCH, seed=seed, verbose=False)
    cmp_runs["BiLSTM"][seed] = (r, evaluate(r["model"], Xte, yte, Lte))
rows = []
for name, d in cmp_runs.items():
    accs = [ev["accuracy"] for _, ev in d.values()]; f1s = [ev["macro_f1"] for _, ev in d.values()]
    best_eps = [r["best_epoch"] for r, _ in d.values()]
    secs = [np.mean(r["history"]["epoch_seconds"]) for r, _ in d.values()]
    mdl = d[42][0]["model"]
    rows.append({"model": name, "parameters": sum(p.numel() for p in mdl.parameters()),
                 "test acc (seed 42)": round(d[42][1]["accuracy"], 2), "test macro-F1 (seed 42)": round(d[42][1]["macro_f1"], 2),
                 "test acc mean ± std": ms(accs), "test macro-F1 mean ± std": ms(f1s),
                 "best val epoch per seed": best_eps, "seconds / epoch (mean)": round(float(np.mean(secs)), 2)})
cmp_df = pd.DataFrame(rows); RESULTS["bilstm_vs_transformer"] = cmp_df.to_dict("records")
cmp_df
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for name, d in cmp_runs.items():
    hh = d[42][0]["history"]; ep = range(1, len(hh["val_acc"]) + 1)
    ax[0].plot(ep, hh["val_acc"], marker="o", label=name); ax[1].plot(ep, hh["train_loss"], marker="o", label=name)
ax[0].set_title("Validation accuracy per epoch (seed 42)"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy (%)"); ax[0].legend()
ax[1].set_title("Training loss per epoch (seed 42)"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("cross-entropy"); ax[1].legend()
savefig(fig, "bilstm_vs_transformer.png"); plt.show()
print("BiLSTM confusion matrix (seed 42):"); print(pd.DataFrame(ev_bl["confusion"], index=C.CATEGORIES, columns=C.CATEGORIES))
""")

md("ANALYSIS_COMPARISON")

md("## Saved artefacts")

code("""
RESULTS["run"] = RUN
(C.OUT / "results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
for p in sorted(list(C.EMB.glob("*")) + list(C.MODELS.glob("*")) + list(C.DATA.glob("*.conll"))):
    print(f"{str(p.relative_to(Path.cwd())):35s} {p.stat().st_size/1e6:9.2f} MB")
print("results -> outputs/results.json; figures ->", len(list(C.FIG.glob('*.png'))), "PNG files in figures/")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
ANALYSIS = ROOT / "notebook_analysis.json"
if ANALYSIS.exists():
    texts = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and c["source"].strip() in texts:
            c["source"] = texts[c["source"].strip()]
nbf.write(nb, OUT)
print("wrote", OUT, "with", len(cells), "cells", "(SMOKE)" if SMOKE else "")
