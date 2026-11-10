"""Part 3: from-scratch transformer components."""
import inspect
import math

import pytest
import torch
import torch.nn as nn

import transformer as T


def test_scaled_dot_product_attention_matches_manual():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 6)
    out, w = T.ScaledDotProductAttention()(q, k, v)
    ref_w = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(4), dim=-1)
    assert torch.allclose(w, ref_w, atol=1e-6)
    assert torch.allclose(out, ref_w @ v, atol=1e-6)


def test_padding_mask_zeroes_attention_to_pad_keys():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[[True, True, False, False]]])
    _, w = T.ScaledDotProductAttention()(x, x, x, mask)
    assert torch.all(w[..., 2:] == 0)
    assert torch.allclose(w.sum(-1), torch.ones(1, 4))


def test_multi_head_has_separate_projections_per_head():
    m = T.MultiHeadSelfAttention(d_model=128, n_heads=4, d_k=32, d_v=32)
    assert len(m.W_q) == len(m.W_k) == len(m.W_v) == 4
    assert all(l.in_features == 128 and l.out_features == 32 for l in m.W_q)
    assert m.W_o.in_features == 4 * 32 and m.W_o.out_features == 128
    out, w = m(torch.randn(2, 7, 128))
    assert out.shape == (2, 7, 128) and w.shape == (2, 4, 7, 7)


def test_positional_encoding_is_fixed_buffer_and_matches_formula():
    pe = T.SinusoidalPositionalEncoding(d_model=8, max_len=20)
    assert "pe" in dict(pe.named_buffers()) and len(list(pe.parameters())) == 0
    pos, i = 7, 1
    assert pe.pe[0, pos, 2 * i].item() == pytest.approx(math.sin(pos / 10000 ** (2 * i / 8)), abs=1e-6)
    assert pe.pe[0, pos, 2 * i + 1].item() == pytest.approx(math.cos(pos / 10000 ** (2 * i / 8)), abs=1e-6)
    x = torch.zeros(1, 5, 8)
    assert torch.allclose(pe(x), pe.pe[:, :5])


def test_pre_ln_block_is_identity_when_sublayers_output_zero():
    blk = T.EncoderBlock(d_model=16, n_heads=2, d_k=8, d_v=8, d_ff=32, dropout=0.0)
    for lin in (blk.attn.W_o, blk.ffn.lin2):
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)
    x = torch.randn(2, 5, 16)
    y, _ = blk(x)
    assert torch.allclose(x, y, atol=1e-6)        # x + 0 + 0: residual path untouched


def test_classifier_shapes_cls_and_padding_invariance():
    torch.manual_seed(0)
    m = T.TransformerClassifier(vocab_size=50, n_classes=5, pad_idx=0, d_model=32, n_heads=4,
                                d_k=8, d_v=8, d_ff=64, n_layers=4, dropout=0.1, max_len=64)
    m.eval()
    assert len(m.blocks) == 4 and isinstance(m.cls, nn.Parameter)
    assert [l.out_features for l in m.head if isinstance(l, nn.Linear)] == [64, 5]
    x = torch.tensor([[5, 6, 7, 8]])
    xp = torch.tensor([[5, 6, 7, 8, 0, 0, 0]])
    logits, attn = m(x, return_attention=True)
    assert logits.shape == (1, 5) and attn[-1].shape == (1, 4, 5, 5)   # +1 for [CLS]
    assert torch.allclose(m(xp), logits, atol=1e-5)


def test_cosine_schedule_with_warmup():
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=5e-4)
    sch = T.cosine_with_warmup(opt, warmup_steps=50, total_steps=200)
    lrs = []
    for _ in range(200):
        lrs.append(sch.get_last_lr()[0])
        opt.step()
        sch.step()
    assert lrs[0] == pytest.approx(5e-4 / 50)
    assert lrs[49] == pytest.approx(5e-4)
    assert max(lrs) == pytest.approx(5e-4) and lrs[-1] < 1e-6
    assert all(a >= b for a, b in zip(lrs[49:], lrs[50:]))           # monotone decay


def test_no_builtin_transformer_classes_are_used():
    import ast
    tree = ast.parse(inspect.getsource(T))           # code only, not docstrings
    used = {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "nn"}
    assert not used & {"Transformer", "MultiheadAttention", "TransformerEncoder",
                       "TransformerEncoderLayer"}
    assert {"Linear", "LayerNorm", "Embedding"} <= used
    m = T.TransformerClassifier(vocab_size=10)
    banned_types = (nn.Transformer, nn.MultiheadAttention, nn.TransformerEncoder,
                    nn.TransformerEncoderLayer)
    assert not any(isinstance(mod, banned_types) for mod in m.modules())
