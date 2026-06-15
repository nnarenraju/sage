"""
Unit tests for the multi-detector consistency heads
(:mod:`sage.architecture.network.consistency`).

CPU-only, no config/data needed — the heads take ``in_ch`` directly.
"""

import math

import torch

from sage.architecture.network.consistency import (
    AttentionPool1d,
    GlobalAvgMaxPool1d,
    PerDetHead,
)

B, C, T = 4, 128, 81


def test_attention_pool_shapes_and_normalisation():
    ap = AttentionPool1d(C)
    pooled, attn, scores, entropy = ap(torch.randn(B, C, T))
    assert pooled.shape == (B, C)
    assert attn.shape == (B, T)
    assert scores.shape == (B, T)
    assert entropy.shape == (B,)
    # attention is a distribution over time
    assert torch.allclose(attn.sum(-1), torch.ones(B), atol=1e-5)
    # 0 <= entropy <= log(T)
    assert (entropy >= -1e-5).all()
    assert (entropy <= math.log(T) + 1e-4).all()


def test_attention_entropy_peaked_vs_uniform():
    ap = AttentionPool1d(C)
    # Make the score read channel 0 only, so we can drive the scores directly.
    with torch.no_grad():
        ap.score.weight.zero_()
        ap.score.weight[0, 0, 0] = 1.0
        ap.score.bias.zero_()
    # uniform: channel 0 flat across time -> equal scores -> entropy ~ log(T)
    x_uniform = torch.zeros(1, C, T)
    _, _, _, ent_uniform = ap(x_uniform)
    # peaked: a large spike in channel 0 at one step -> peaked attention -> ~0
    x_peak = torch.zeros(1, C, T)
    x_peak[0, 0, 10] = 50.0
    _, _, _, ent_peak = ap(x_peak)
    assert ent_peak < ent_uniform
    assert ent_peak < 0.1
    assert ent_uniform > math.log(T) - 0.1


def test_global_avg_max_pool_shape():
    out = GlobalAvgMaxPool1d()(torch.randn(B, C, T))
    assert out.shape == (B, 2 * C)


def test_perdethead_shapes_and_sigma_clamp():
    head = PerDetHead(C, log_sigma_clamp=(-7.0, 3.0))
    t_pos = torch.linspace(0.0, 12.0, T)
    out = head(torch.randn(B, C, T), t_pos)
    for field in out._fields:
        assert getattr(out, field).shape == (B,), field
    for ls in (out.log_sigma_tc, out.log_sigma_mc):
        assert (ls >= -7.0 - 1e-4).all() and (ls <= 3.0 + 1e-4).all()
    # soft-argmax mu_tc must lie inside the physical-time window
    assert (out.mu_tc >= t_pos.min() - 1e-3).all()
    assert (out.mu_tc <= t_pos.max() + 1e-3).all()


def test_perdethead_soft_argmax_localises():
    head = PerDetHead(C)
    t_pos = torch.linspace(0.0, 12.0, T)
    # Drive the saliency from channel 0; a spike at step k must give mu_tc = t_pos[k].
    with torch.no_grad():
        head.tc_saliency.weight.zero_()
        head.tc_saliency.weight[0, 0, 0] = 50.0
        head.tc_saliency.bias.zero_()
    for k in (5, 40, 75):
        x = torch.zeros(1, C, T)
        x[0, 0, k] = 1.0
        mu = head(x, t_pos).mu_tc
        assert abs(float(mu) - float(t_pos[k])) < 1e-2, (k, float(mu))


def test_perdethead_compile_fullgraph():
    head = torch.compile(PerDetHead(C), fullgraph=True)
    out = head(torch.randn(B, C, T), torch.linspace(0.0, 12.0, T))
    assert out.mu_tc.shape == (B,)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(">>> ALL CONSISTENCY-HEAD TESTS PASSED <<<")
