"""Tests for the non-astrophysical (decoherent) sample masker and
injector, on CPU."""

import torch

from sage.data.non_astrophysical import NON_ASTROPHYSICAL_CLASS, NonAstrophysicalInjector, NonAstrophysicalMasker

F, N, D = 64, 64, 2
DELTA_F = 1.0 / 16.0
TC_BOUNDS = (11.0, 11.2)
LEN_S = 12.0
MARGIN = 0.1


def _pool(seed=0, detectors=D):
    """Coherent injections: FD strain, per-detector tc and shared mc."""
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(N, detectors, F, generator=g)
    im = torch.randn(N, detectors, F, generator=g)
    data = torch.complex(re, im)
    span = TC_BOUNDS[1] - TC_BOUNDS[0]
    tc = torch.rand(N, detectors, generator=g) * span + TC_BOUNDS[0]
    mc = (torch.rand(N, 1, generator=g) * 2.0 - 1.0).expand(N, detectors).contiguous()
    return data, tc, mc


def _masker(signal_noise_share=0.5, in_band_share=0.5, seed=0):
    """A seeded NonAstrophysicalMasker with the given shares."""
    return NonAstrophysicalMasker(
        DELTA_F, TC_BOUNDS, LEN_S, signal_noise_share, in_band_share, torch.Generator().manual_seed(seed)
    )


def test_tc_within_window_and_favours_band():
    """New tc stays inside the window and favours the tc band."""
    data, tc, mc = _pool(1)
    _, na_tc, _, _ = _masker(seed=1)(data, tc, mc)
    assert (na_tc >= MARGIN - 1e-6).all()
    assert (na_tc <= LEN_S - MARGIN + 1e-6).all()
    in_band = ((na_tc >= TC_BOUNDS[0]) & (na_tc <= TC_BOUNDS[1])).float().mean()
    # A 0.2 s band in an 11.8 s window; in-band weighting lifts this.
    assert float(in_band) > 0.3


def test_signal_noise_drops_one_detector():
    """Signal+noise keeps one detector; the other is zeroed."""
    data, tc, mc = _pool(2)
    d, _, _, mask = _masker(signal_noise_share=1.0, seed=2)(data, tc, mc)
    assert (mask.sum(1) == 1).all()                      # exactly one supervised
    for i in range(N):
        off = int((mask[i] == 0).nonzero())
        assert float(d[i, off].abs().sum()) == 0.0       # dropped det is silent


def test_signal_signal_keeps_both_and_mc_differs():
    """Signal+signal keeps both detectors with different events."""
    data, tc, mc = _pool(3)
    d, _, na_mc, mask = _masker(signal_noise_share=0.0, seed=3)(data, tc, mc)
    assert (mask == 1).all()                             # both detectors supervised
    assert (d.abs().sum(-1) > 0).all()                   # neither detector zeroed
    # Each detector's chirp mass comes from a different event.
    differ = (na_mc[:, 0] != na_mc[:, 1]).float().mean()
    assert float(differ) > 0.9


def test_retiming_zeros_wraparound():
    """Re-timing is a linear shift: wrapped samples are zero."""
    # Detector 0 keeps its own event, so its shift is na_tc - tc.
    data, tc, mc = _pool(4)
    d, na_tc, _, _ = _masker(signal_noise_share=0.0, seed=4)(data, tc, mc)
    nsamples = 2 * (F - 1)
    sample_rate = nsamples * DELTA_F
    td = torch.fft.irfft(d, n=nsamples, dim=-1)              # (N, D, nsamples)
    ds = torch.round((na_tc[:, 0] - tc[:, 0]) * sample_rate).long()
    for i in range(N):
        k = int(ds[i])
        if k > 0:        # right shift -> leading k samples wrapped -> zeroed
            assert torch.allclose(td[i, 0, :k], torch.zeros(k), atol=1e-3)
        elif k < 0:      # left shift -> trailing |k| wrapped -> zeroed
            assert torch.allclose(td[i, 0, k:], torch.zeros(-k), atol=1e-3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print(">>> ALL NON-ASTROPHYSICAL MASKER TESTS PASSED <<<")


def test_injector_labels_noise_windows():
    """Share 1 turns every noise window class 2, and nothing else."""
    data, tc, mc = _pool(5)
    # Columns: 2 estimates, class, then tc and mc per detector.
    pool = lambda: (data[:4], torch.cat([torch.zeros(4, 3), tc[:4], mc[:4]], dim=1))
    injector = NonAstrophysicalInjector(pool, _masker(seed=5), 1.0, 2, torch.Generator().manual_seed(5))
    x, targets = torch.zeros(8, D, F, dtype=torch.complex64), torch.zeros(8, 3)
    signal_idx = torch.tensor([0, 2, 4, 6])
    targets[signal_idx, -1] = 1.0
    x, targets = injector(x, targets, signal_idx)
    assert targets[:, -1].tolist() == [1, NON_ASTROPHYSICAL_CLASS] * 4
    assert (x[signal_idx] == 0).all() and (x[1::2].abs().sum(-1) > 0).any(dim=1).all()
    assert (targets[:, :2] == 0).all()


def test_injector_draws_only_what_it_uses():
    """The pool is called only when pairs run out; each used once."""
    calls = []

    def pool():
        """The next 4 pairs, valued by their running index."""
        start = 4 * len(calls)
        calls.append(start)
        data = torch.arange(start, start + 4, dtype=torch.float32).view(4, 1, 1).expand(4, D, F).to(torch.complex64)
        return data, torch.zeros(4, 3 + 2 * D)

    injector = NonAstrophysicalInjector(pool, lambda data, *_: (data,), 0.3, 2, torch.Generator().manual_seed(6))
    used = []
    for _ in range(10):
        x, targets = torch.zeros(8, D, F, dtype=torch.complex64), torch.zeros(8, 3)
        x, targets = injector(x, targets, torch.tensor([0, 2, 4, 6]))
        used += x[targets[:, -1] == NON_ASTROPHYSICAL_CLASS, 0, 0].real.tolist()
    assert used and used == list(range(len(used))) and len(calls) == -(-len(used) // 4)


def test_three_detectors():
    """Three detectors: signal+noise keeps one; signal+signal differ."""
    data, tc, mc = _pool(7, detectors=3)
    d, _, _, mask = _masker(signal_noise_share=1.0, seed=7)(data, tc, mc)
    assert (mask.sum(1) == 1).all() and set(mask.argmax(1).tolist()) == {0, 1, 2}
    assert ((d.abs().sum(-1) > 0) == mask.bool()).all()
    d, _, na_mc, mask = _masker(signal_noise_share=0.0, seed=7)(data, tc, mc)
    assert (mask == 1).all() and (d.abs().sum(-1) > 0).all()
    assert all((na_mc[:, i] != na_mc[:, j]).all() for i, j in ((0, 1), (0, 2), (1, 2)))
