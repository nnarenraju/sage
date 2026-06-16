"""Tests for the non-astrophysical (decoherent) sample masker."""

import torch

from sage.data.non_astrophysical import NonAstrophysicalMasker

S, F = 64, 100


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    sig = torch.randn(S, 2, F, generator=g)
    tc = torch.rand(S, 2, generator=g) * 0.02 + 11.0
    return sig, tc


def test_p0_is_noop():
    sig, tc = _batch()
    s2, t2, mask, coh = NonAstrophysicalMasker(p_non_astro=0.0)(sig, tc)
    assert torch.equal(sig, s2) and torch.equal(tc, t2)
    assert (mask == 1).all() and (coh == 1).all()


def test_signal_noise_drops_one_detector():
    sig, tc = _batch()
    s2, t2, mask, coh = NonAstrophysicalMasker(
        p_non_astro=1.0, p_signal_noise=1.0, seed=1
    )(sig, tc)
    assert (coh == 0).all()                 # all decohered -> not a detection
    assert (mask.sum(1) == 1).all()         # exactly one detector supervised
    for i in range(S):
        d_off = int((mask[i] == 0).nonzero())
        assert torch.allclose(s2[i, d_off], torch.zeros(F))  # dropped det is silent


def test_signal_signal_keeps_both_masked():
    sig, tc = _batch()
    s2, t2, mask, coh = NonAstrophysicalMasker(
        p_non_astro=1.0, p_signal_noise=0.0, seed=2
    )(sig, tc)
    assert (coh == 0).all()
    assert (mask == 1).all()                # both detectors supervised (own truth)
    assert not torch.equal(sig, s2)         # one detector replaced, none zeroed
    # no detector is fully zeroed in signal+signal'
    assert (s2.abs().sum(-1) > 0).all()


def test_fraction_decohered_matches_probability():
    sig, tc = _batch()
    _, _, _, coh = NonAstrophysicalMasker(p_non_astro=0.3, seed=3)(sig, tc)
    frac = float((coh == 0).float().mean())
    assert 0.1 < frac < 0.55                # ~0.3 given S=64


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print(">>> ALL NON-ASTROPHYSICAL MASKER TESTS PASSED <<<")
