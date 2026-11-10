"""Part 1, Section 2: Skip-gram Word2Vec with negative sampling, from scratch.

Model
    V (centre) and U (context) are two separate |V| x d embedding matrices.
    For a positive pair (centre c, context o) and K noise words w_1..w_K drawn
    from P_n(w) ~ f(w)^(3/4):

        L = - log sigma(u_o . v_c) - sum_{k=1..K} log sigma(-u_{w_k} . v_c)

    which is the binary cross-entropy of "o is a real context word" (label 1)
    plus "w_k is noise" (label 0). The final embedding of a word is
    1/2 (V + U).

Training pairs
    Every (centre, context) pair with 1 <= |offset| <= k inside a sentence,
    optionally after word2vec's frequent-word sub-sampling (Mikolov et al.,
    2013, "Distributed Representations of Words and Phrases", Sec. 2.3).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from corpus import encode  # noqa: E402


# ----------------------------------------------------------------------------
# Data: vocabulary counts, noise table, training pairs
# ----------------------------------------------------------------------------
def encode_corpus(sentences: list[list[str]], w2i: dict) -> list[np.ndarray]:
    return [np.asarray(encode(s, w2i), dtype=np.int64) for s in sentences]


def unigram_counts(sent_ids: list[np.ndarray], n_words: int) -> np.ndarray:
    return np.bincount(np.concatenate(sent_ids), minlength=n_words)[:n_words]


def build_noise_table(counts: np.ndarray, power: float = C.W2V_NOISE_POWER,
                      table_size: int = C.W2V_TABLE_SIZE) -> np.ndarray:
    """Unigram table as in word2vec.c: word w fills a share of the table
    proportional to f(w)^power, so uniform indexing samples from P_n(w)."""
    p = counts.astype(np.float64) ** power
    p /= p.sum()
    slots = np.floor(p * table_size).astype(np.int64)
    # hand the slots lost to flooring to the words with the largest remainders
    missing = table_size - slots.sum()
    if missing > 0:
        rema = p * table_size - slots
        slots[np.argsort(-rema)[:missing]] += 1
    return np.repeat(np.arange(len(counts), dtype=np.int64), slots)


def noise_distribution(counts: np.ndarray, power: float = C.W2V_NOISE_POWER) -> np.ndarray:
    p = counts.astype(np.float64) ** power
    return p / p.sum()


def skipgram_pairs(sent_ids: list[np.ndarray], window: int = C.W2V_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """All (centre, context) pairs with 1 <= |offset| <= window, per sentence."""
    centres, contexts = [], []
    for ids in sent_ids:
        n = len(ids)
        for d in range(1, window + 1):
            if n <= d:
                break
            a, b = ids[:-d], ids[d:]
            centres += [a, b]      # a sees b at +d, b sees a at -d
            contexts += [b, a]
    return np.concatenate(centres), np.concatenate(contexts)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class SkipGramNS(nn.Module):
    def __init__(self, n_words: int, dim: int):
        super().__init__()
        self.V = nn.Embedding(n_words, dim)   # centre-word vectors v
        self.U = nn.Embedding(n_words, dim)   # context-word vectors u
        nn.init.uniform_(self.V.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.U.weight)         # word2vec.c initialisation

    def forward(self, centre: torch.Tensor, context: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        v_c = self.V(centre)                                   # (B, d)
        u_o = self.U(context)                                  # (B, d)
        u_k = self.U(negatives)                                # (B, K, d)
        pos = F.logsigmoid((u_o * v_c).sum(-1))                # log s(u_o.v_c)
        neg = F.logsigmoid(-torch.bmm(u_k, v_c.unsqueeze(2)).squeeze(2)).sum(-1)
        return -(pos + neg).mean()

    def embeddings(self) -> np.ndarray:
        """Averaged final embeddings 1/2 (V + U)."""
        return (0.5 * (self.V.weight + self.U.weight)).detach().cpu().numpy()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def keep_probabilities(counts: np.ndarray, t: float) -> np.ndarray:
    """word2vec.c sub-sampling: a token of word w with relative frequency
    f(w) is kept with probability (sqrt(f/t) + 1) * t / f, capped at 1."""
    f = counts / counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (np.sqrt(f / t) + 1.0) * t / f
    return np.where(counts > 0, np.minimum(p, 1.0), 1.0)


def subsample(sent_ids: list[np.ndarray], keep_p: np.ndarray,
              rng: np.random.Generator) -> list[np.ndarray]:
    return [ids[rng.random(len(ids)) < keep_p[ids]] for ids in sent_ids]


def train_skipgram(sentences: list[list[str]], w2i: dict, dim: int = C.W2V_DIM,
                   window: int = C.W2V_WINDOW, n_neg: int = C.W2V_NEG,
                   epochs: int = C.W2V_EPOCHS, batch_size: int = C.W2V_BATCH,
                   lr: float = C.W2V_LR, subsample_t: float | None = C.W2V_SUBSAMPLE,
                   seed: int = C.SEED, log_every: int = 200, verbose: bool = True,
                   snapshot_epochs: tuple[int, ...] = ()) -> dict:
    """Train on `sentences`; vocabulary rows 0..|V|-1 exclude <PAD>.

    With `subsample_t`, frequent tokens are randomly discarded *before* the
    context windows are formed (fresh draw every epoch), exactly as in
    word2vec.c; the noise distribution still uses the full counts f(w)."""
    assert batch_size >= 512 and epochs >= 5
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_words = len([w for w in w2i if w != C.PAD])         # words + <UNK>
    sent_ids = encode_corpus(sentences, w2i)
    counts = unigram_counts(sent_ids, n_words)
    table = torch.from_numpy(build_noise_table(counts))
    keep_p = keep_probabilities(counts, subsample_t) if subsample_t else None
    full_pairs = len(skipgram_pairs(sent_ids, window)[0])

    model = SkipGramNS(n_words, dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    step_losses, epoch_losses, log_steps, pairs_per_epoch = [], [], [], []
    snapshots: dict[int, np.ndarray] = {}
    running, n_run, step = 0.0, 0, 0
    t0 = time.time()
    if verbose:
        print(f"vocab={n_words:,}  tokens={int(counts.sum()):,}  all pairs={full_pairs:,}  "
              f"d={dim} k={window} K={n_neg} batch={batch_size} lr={lr} "
              f"subsample_t={subsample_t}")
    for ep in range(1, epochs + 1):
        ids_ep = subsample(sent_ids, keep_p, rng) if keep_p is not None else sent_ids
        centre, context = (torch.from_numpy(x) for x in skipgram_pairs(ids_ep, window))
        n_pairs = len(centre)
        pairs_per_epoch.append(n_pairs)
        perm = torch.randperm(n_pairs, generator=gen)
        ep_sum, ep_n = 0.0, 0
        for b in range(0, n_pairs, batch_size):
            idx = perm[b:b + batch_size]
            neg = table[torch.randint(len(table), (len(idx), n_neg), generator=gen)]
            loss = model(centre[idx], context[idx], neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            lv = loss.item()
            running += lv
            n_run += 1
            ep_sum += lv * len(idx)
            ep_n += len(idx)
            if step % log_every == 0:
                step_losses.append(running / n_run)
                log_steps.append(step)
                running, n_run = 0.0, 0
        epoch_losses.append(ep_sum / ep_n)
        if ep in snapshot_epochs:
            snapshots[ep] = model.embeddings()
        if verbose:
            print(f"  epoch {ep}/{epochs}  pairs {n_pairs:,}  mean loss {epoch_losses[-1]:.4f}  "
                  f"({time.time() - t0:.0f}s elapsed)")
    return {"model": model, "embeddings": model.embeddings(), "step_losses": step_losses,
            "log_steps": log_steps, "epoch_losses": epoch_losses,
            "pairs_per_epoch": pairs_per_epoch, "all_pairs": full_pairs,
            "counts": counts, "seconds": time.time() - t0, "steps": step,
            "snapshots": snapshots}
