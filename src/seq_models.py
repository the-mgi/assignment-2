"""Part 2, Section 4: BiLSTM sequence labeller with an optional CRF layer.

Everything except the LSTM cell itself (torch.nn.LSTM) is written here:
the CRF negative log-likelihood (forward algorithm), Viterbi decoding,
masking of padded positions and the frozen / fine-tuned embedding modes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CRF(nn.Module):
    """Linear-chain CRF with a learnable tag-transition matrix.

    score(x, y) = start[y_1] + sum_t emit[t, y_t] + sum_t trans[y_{t-1}, y_t] + end[y_T]
    NLL = log Z(x) - score(x, y), with log Z from the forward algorithm.
    Padded positions (mask = False) are skipped in both recursions.
    """

    def __init__(self, n_tags: int):
        super().__init__()
        self.n_tags = n_tags
        self.transitions = nn.Parameter(torch.empty(n_tags, n_tags).uniform_(-0.1, 0.1))
        self.start = nn.Parameter(torch.empty(n_tags).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(n_tags).uniform_(-0.1, 0.1))

    def _gold_score(self, emis, tags, mask):
        B, T, _ = emis.shape
        maskf = mask.float()
        score = self.start[tags[:, 0]] + emis[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            step = self.transitions[tags[:, t - 1], tags[:, t]] + \
                emis[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + step * maskf[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return score + self.end[last]

    def _log_partition(self, emis, mask):
        B, T, K = emis.shape
        alpha = self.start.unsqueeze(0) + emis[:, 0]                    # (B, K)
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
                                  + emis[:, t].unsqueeze(1), dim=1)       # (B, K)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def neg_log_likelihood(self, emis, tags, mask) -> torch.Tensor:
        """Mean NLL over the batch. `tags` may hold any value at padded steps."""
        tags = tags.clamp(min=0)
        return (self._log_partition(emis, mask) - self._gold_score(emis, tags, mask)).mean()

    @torch.no_grad()
    def viterbi(self, emis, mask) -> list[list[int]]:
        B, T, K = emis.shape
        score = self.start.unsqueeze(0) + emis[:, 0]
        backptr = []
        for t in range(1, T):
            cand = score.unsqueeze(2) + self.transitions.unsqueeze(0)    # (B, K_prev, K)
            best, idx = cand.max(dim=1)
            nxt = best + emis[:, t]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, nxt, score)
            # at padded steps keep the identity pointer so back-tracking is a no-op
            ident = torch.arange(K, device=emis.device).unsqueeze(0).expand(B, K)
            backptr.append(torch.where(m, idx, ident))
        score = score + self.end.unsqueeze(0)
        best_last = score.argmax(dim=1)
        lengths = mask.sum(1).tolist()
        paths = []
        for b in range(B):
            tag = best_last[b].item()
            path = [tag]
            for t in range(T - 2, -1, -1):
                tag = backptr[t][b, tag].item()
                path.append(tag)
            path.reverse()
            paths.append(path[: lengths[b]])
        return paths


class BiLSTMTagger(nn.Module):
    """Embedding -> 2-layer (Bi)LSTM -> linear emission layer (-> CRF).

    The contextual representation of token t is h_t = [h_fwd_t ; h_bwd_t],
    the concatenation of the forward and backward hidden states.
    Dropout p between the two LSTM layers is applied by nn.LSTM(dropout=p).
    """

    def __init__(self, n_tags: int, vocab_size: int, emb_dim: int = 100,
                 pretrained: np.ndarray | None = None, freeze: bool = False,
                 pad_idx: int = 0, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.5, bidirectional: bool = True, use_crf: bool = False):
        super().__init__()
        if pretrained is not None:
            weights = torch.as_tensor(pretrained, dtype=torch.float32)
            self.emb = nn.Embedding.from_pretrained(weights, freeze=freeze, padding_idx=pad_idx)
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
            self.emb.weight.requires_grad_(not freeze)
        self.hidden = hidden
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(self.emb.embedding_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), n_tags)
        self.crf = CRF(n_tags) if use_crf else None

    def emissions(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.emb(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        if self.bidirectional:
            h_fwd, h_bwd = out[..., : self.hidden], out[..., self.hidden:]
            out = torch.cat([h_fwd, h_bwd], dim=-1)          # h_t = [->h_t ; <-h_t]
        return self.out(out)

    def loss(self, x, lengths, y, mask) -> torch.Tensor:
        emis = self.emissions(x, lengths)
        if self.crf is not None:
            return self.crf.neg_log_likelihood(emis, y, mask)
        # padded positions carry label -100 and are ignored by the loss
        return F.cross_entropy(emis.reshape(-1, emis.size(-1)), y.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def predict(self, x, lengths, mask) -> list[list[int]]:
        emis = self.emissions(x, lengths)
        if self.crf is not None:
            return self.crf.viterbi(emis, mask)
        pred = emis.argmax(-1)
        return [pred[b, : int(lengths[b])].tolist() for b in range(x.size(0))]
