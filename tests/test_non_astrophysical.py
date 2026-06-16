"""Tests for the non-astrophysical (decoherent) sample masker (CPU, no data)."""

import torch

from sage.data.non_astrophysical import NonAstrophysicalMasker

F, N, D = 64, 64, 2
DELTA_F = 1.0 / 16.0
TC_BOUNDS = (11.0, 11.2)
LEN_S = 12.0
MARGIN = 0.1


def _pool(seed=0):
    """A pool of coherent injections: complex FD strain, per-det tc, per-det mc.

    ``mc`` is broadcast across detectors (identical per row) — exactly what the
    signal sampler hands over for a coherent injection.
    """
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(N, D, F, generator=g)
    im = torch.randn(N, D, F, generator=g)
    data = torch.complex(re, im)
    tc = torch.rand(N, D, generator=g) * (TC_BOUNDS[1] - TC_BOUNDS[0]) + TC_BOUNDS[0]
    mc = (torch.rand(N, 1, generator=g) * 2.0 - 1.0).expand(N, D).contiguous()
    return data, tc, mc


def _masker(**kw):
    return NonAstrophysicalMasker(
        DELTA_F, TC_BOUNDS, LEN_S, seed=kw.pop("seed", 0), **kw
    )


def test_shapes_and_dtype():
    data, tc, mc = _pool()
    d, t, m, mask = _masker()(data, tc, mc)
    assert d.shape == (N, D, F) and d.is_complex()
    assert t.shape == (N, D) and m.shape == (N, D) and mask.shape == (N, D)


def test_tc_within_window_and_favours_band():
    data, tc, mc = _pool(1)
    _, na_tc, _, _ = _masker(seed=1)(data, tc, mc)
    assert (na_tc >= MARGIN - 1e-6).all()
    assert (na_tc <= LEN_S - MARGIN + 1e-6).all()
    in_band = ((na_tc >= TC_BOUNDS[0]) & (na_tc <= TC_BOUNDS[1])).float().mean()
    # ~0.2s band out of ~11.8s window — only the in-band weighting lifts this.
    assert float(in_band) > 0.3


def test_signal_noise_drops_one_detector():
    data, tc, mc = _pool(2)
    d, _, _, mask = _masker(p_signal_noise=1.0, seed=2)(data, tc, mc)
    assert (mask.sum(1) == 1).all()                      # exactly one supervised
    for i in range(N):
        off = int((mask[i] == 0).nonzero())
        assert float(d[i, off].abs().sum()) == 0.0       # dropped det is silent


def test_signal_signal_keeps_both_and_mc_differs():
    data, tc, mc = _pool(3)
    d, _, na_mc, mask = _masker(p_signal_noise=0.0, seed=3)(data, tc, mc)
    assert (mask == 1).all()                             # both detectors supervised
    assert (d.abs().sum(-1) > 0).all()                   # neither detector zeroed
    # each detector's chirp mass comes from a *different* event -> they disagree.
    differ = (na_mc[:, 0] != na_mc[:, 1]).float().mean()
    assert float(differ) > 0.9


def test_retiming_preserves_magnitude():
    # The FD phase shift has unit modulus, so |strain| is unchanged; detector 0
    # keeps its own event (src index identity), only its phase moves.
    data, tc, mc = _pool(4)
    d, _, _, _ = _masker(p_signal_noise=0.0, seed=4)(data, tc, mc)
    assert torch.allclose(d[:, 0].abs(), data[:, 0].abs(), atol=1e-4)


def test_empty_pool_is_safe():
    d, t, m, mask = _masker()(
        torch.empty(0, D, F, dtype=torch.complex64),
        torch.empty(0, D),
        torch.empty(0, D),
    )
    assert d.shape == (0, D, F) and mask.shape == (0, D)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print(">>> ALL NON-ASTROPHYSICAL MASKER TESTS PASSED <<<")
