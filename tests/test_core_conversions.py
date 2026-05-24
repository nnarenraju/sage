"""Unit tests for sage.core.conversions."""

import pytest
import torch

from sage.core.conversions import (
    seconds_to_samples,
    samples_to_seconds,
    mchirp_eta_to_mass1_mass2,
    eta_from_mass1_mass2,
    mchirp_from_mass1_mass2,
    mass1_mass2_to_mchirp_eta,
)


# ---------------------------------------------------------------------------
# seconds_to_samples / samples_to_seconds
# ---------------------------------------------------------------------------

def test_seconds_to_samples_exact():
    assert seconds_to_samples(1.0, 4096) == 4096


def test_seconds_to_samples_half_second():
    assert seconds_to_samples(0.5, 4096) == 2048


def test_seconds_to_samples_rounding():
    # 0.1 * 4096 = 409.6 → rounds to 410
    result = seconds_to_samples(0.1, 4096)
    assert result == 410


def test_seconds_to_samples_no_rounding():
    # rounding=False uses int truncation: int(0.1 * 4096) = int(409.6) = 409
    # Compare: rounding=True gives round(409.6) = 410
    result_truncated = seconds_to_samples(0.1, 4096, rounding=False)
    result_rounded = seconds_to_samples(0.1, 4096, rounding=True)
    assert result_truncated == 409
    assert result_rounded == 410


def test_samples_to_seconds_exact():
    assert samples_to_seconds(4096, 4096) == pytest.approx(1.0)


def test_samples_to_seconds_roundtrip():
    for fs in [2048, 4096, 8192]:
        for t in [0.5, 1.0, 4.0, 16.0]:
            n = seconds_to_samples(t, fs)
            assert samples_to_seconds(n, fs) == pytest.approx(t, rel=1e-6)


# ---------------------------------------------------------------------------
# eta_from_mass1_mass2
# ---------------------------------------------------------------------------

def test_eta_from_equal_mass():
    eta = eta_from_mass1_mass2(1.4, 1.4)
    assert eta == pytest.approx(0.25)


def test_eta_from_mass1_mass2_range():
    # η ∈ (0, 0.25] for any positive masses
    for m1, m2 in [(10, 1), (1.4, 1.4), (30, 5)]:
        eta = eta_from_mass1_mass2(m1, m2)
        assert 0 < eta <= 0.25 + 1e-10


def test_eta_symmetric():
    # η(m1, m2) == η(m2, m1)
    assert eta_from_mass1_mass2(10.0, 3.0) == pytest.approx(
        eta_from_mass1_mass2(3.0, 10.0)
    )


# ---------------------------------------------------------------------------
# mchirp_from_mass1_mass2
# ---------------------------------------------------------------------------

def test_mchirp_from_mass1_mass2_known():
    # For equal-mass binary with m1=m2=m: Mc = m * (0.25)^(3/5) * 2 = m * 2^(4/5)
    m = 10.0
    expected = m * (0.25 ** (3.0 / 5)) * 2
    result = mchirp_from_mass1_mass2(m, m)
    assert result == pytest.approx(expected, rel=1e-6)


def test_mchirp_positive():
    for m1, m2 in [(5, 5), (30, 1), (1.4, 1.2)]:
        assert mchirp_from_mass1_mass2(m1, m2) > 0


# ---------------------------------------------------------------------------
# mchirp_eta_to_mass1_mass2 (torch tensors)
# ---------------------------------------------------------------------------

def test_mchirp_eta_roundtrip():
    m1_orig = torch.tensor([10.0, 30.0, 1.4])
    m2_orig = torch.tensor([5.0, 5.0, 1.4])
    mchirp, eta = mass1_mass2_to_mchirp_eta(m1_orig, m2_orig)
    m1_rec, m2_rec = mchirp_eta_to_mass1_mass2(mchirp, eta)
    assert torch.allclose(m1_rec, m1_orig, rtol=1e-5)
    assert torch.allclose(m2_rec, m2_orig, rtol=1e-5)


def test_mass_ordering_preserved():
    m1 = torch.tensor([30.0, 10.0, 1.4])
    m2 = torch.tensor([5.0, 3.0, 1.2])
    mchirp, eta = mass1_mass2_to_mchirp_eta(m1, m2)
    m1_out, m2_out = mchirp_eta_to_mass1_mass2(mchirp, eta)
    assert (m1_out >= m2_out).all()


def test_mchirp_eta_batch_shapes():
    m1 = torch.rand(16) * 25 + 5
    m2 = torch.rand(16) * 5 + 1
    mchirp, eta = mass1_mass2_to_mchirp_eta(m1, m2)
    assert mchirp.shape == (16,)
    assert eta.shape == (16,)
    m1_out, m2_out = mchirp_eta_to_mass1_mass2(mchirp, eta)
    assert m1_out.shape == (16,)
    assert m2_out.shape == (16,)


def test_mass1_mass2_to_mchirp_eta_symmetry():
    # Swapping masses → same mchirp, same eta
    mchirp_a, eta_a = mass1_mass2_to_mchirp_eta(10.0, 3.0)
    mchirp_b, eta_b = mass1_mass2_to_mchirp_eta(3.0, 10.0)
    assert abs(mchirp_a - mchirp_b) < 1e-10
    assert abs(eta_a - eta_b) < 1e-10
