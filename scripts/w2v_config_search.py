"""Choose the C3 training configuration (run once; output in outputs/w2v_config_search.txt).

Three sub-sampling settings x snapshots after 5/10/15/20 epochs, all other
hyper-parameters fixed by the brief (d=100, k=5, K=10, Adam 1e-3, batch 512).
Selection criterion, fixed before looking at the 10 reported analogies:
top-3 accuracy on the *development* analogy pool (eval_sets.dev_pool(), 180
analogies disjoint from eval_sets.ANALOGIES), ties broken by top-1 accuracy.

Usage:  python scripts/w2v_config_search.py      (~30 min on 4 CPU cores)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import torch

import config as C
from corpus import build_vocab, flatten, load_cleaned_articles
from embed_eval import mean_reciprocal_rank, run_analogies
from eval_sets import MRR_PAIRS, cleaned, cleaned_analogies, dev_pool
from part1_counts import CosineIndex
from word2vec import encode_corpus, train_skipgram, unigram_counts

torch.set_num_threads(4)
sents = flatten(load_cleaned_articles())
w2i = build_vocab(sents)
counts = unigram_counts(encode_corpus(sents, w2i), len(w2i) - 1)
dev, test = dev_pool(), cleaned_analogies()
pairs = [(cleaned(a), cleaned(b)) for a, b in MRR_PAIRS]
rows = []
for t in (None, 1e-4, 1e-3):
    r = train_skipgram(sents, w2i, epochs=20, subsample_t=t, verbose=False,
                       snapshot_epochs=(5, 10, 15, 20))
    for ep, E in sorted(r["snapshots"].items()):
        idx = CosineIndex(E, w2i, counts=counts, min_count=5)
        d = run_analogies(idx, dev)
        mrr, _ = mean_reciprocal_rank(idx, pairs)
        rows.append((str(t), ep, np.mean([x["top1"] for x in d]), np.mean([x["top3"] for x in d]), mrr))
        print(f"subsample={t!s:6} epochs={ep:2d}  DEV top1={rows[-1][2]:.3f} top3={rows[-1][3]:.3f}  MRR={mrr:.3f}",
              flush=True)
best = max(rows, key=lambda x: (x[3], x[2], -x[1]))
print(f"selected: subsample={best[0]} epochs={best[1]}")
