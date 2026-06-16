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
    PerDetOutput,
    consistency_statistic,
    corroboration_features,
)
from sage.architecture.custom_losses import ConsistencyNLLLoss

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


def _mk(mu_tc, log_sigma_tc, mu_mc=0.0, log_sigma_mc=0.0):
    n = mu_tc.shape[0]
    return PerDetOutput(
        mu_tc, torch.full((n,), log_sigma_tc), torch.full((n,), mu_mc),
        torch.full((n,), log_sigma_mc), torch.zeros(n), torch.zeros(n),
    )


def test_consistency_statistic_is_uncertainty_weighted():
    ltt = torch.tensor(0.0100128)  # H1-L1
    # agreement within the light-travel time -> exactly zero
    s_tc, _ = consistency_statistic(_mk(torch.tensor([11.0]), -3.0),
                                    _mk(torch.tensor([11.005]), -3.0), ltt)
    assert float(s_tc) < 1e-6
    # large disagreement, both confident -> large statistic
    s_conf, _ = consistency_statistic(_mk(torch.tensor([11.0]), -3.0),
                                      _mk(torch.tensor([11.1]), -3.0), ltt)
    # same disagreement but one detector uncertain (large sigma) -> tolerated
    s_faint, _ = consistency_statistic(_mk(torch.tensor([11.0]), -3.0),
                                       _mk(torch.tensor([11.1]), 0.0), ltt)
    assert float(s_conf) > 1.0
    assert float(s_faint) < float(s_conf) / 10.0  # uncertainty makes it tolerant


def test_corroboration_features_shape():
    a = _mk(torch.zeros(B), -1.0)
    b = _mk(torch.zeros(B), -1.0)
    s_tc, s_mc = consistency_statistic(a, b, torch.tensor(0.01))
    assert corroboration_features(a, b, s_tc, s_mc).shape == (B, 8)


def test_consistency_loss_masking_and_grad():
    loss = ConsistencyNLLLoss()
    assert loss.num_components == 3
    D = 2
    mu = torch.zeros(B, D, requires_grad=True)
    ls = torch.zeros(B, D, requires_grad=True)
    y = torch.zeros(B, D)
    # perfect mean, log_sigma=0 -> nll = 0
    out = loss(mu, ls, mu, ls, y, y.mean(1), torch.ones(B, D))
    assert torch.allclose(out, torch.zeros(3), atol=1e-6)
    # fully-masked-out batch -> zero loss regardless of error
    out0 = loss(mu + 5, ls, mu + 5, ls, y, y.mean(1), torch.zeros(B, D))
    assert float(out0[0]) == 0.0
    # gradient flows from the total
    out[0].backward()
    assert torch.isfinite(mu.grad).all()


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(">>> ALL CONSISTENCY-HEAD TESTS PASSED <<<")
