"""Unit tests for sage.data.waveform.conversions."""

import pytest

# sage.data.waveform.__init__ imports IMRPhenomPv2 -> sage.core.config -> matplotlib
pytest.importorskip("matplotlib", reason="sage.data.waveform requires matplotlib")

import torch
from sage.data.waveform.conversions import (
    mass1_mass2_to_mchirp_q,
    chirp_distance_to_distance,
)


class TestMass1Mass2ToMchirpQ:
    def test_equal_mass_q_is_one(self):
        mchirp, q = mass1_mass2_to_mchirp_q(10.0, 10.0)
        assert q == pytest.approx(1.0)

    def test_equal_mass_mchirp_formula(self):
        # For m1=m2=m: Mc = m * 2^(4/5) * 0.25^(3/5) = m * 2^(4/5-3/5) = m * 2^(1/5)
        # Actually: Mc = (m*m)^(3/5) / (m+m)^(1/5) = m^(6/5) / (2m)^(1/5)
        #          = m^(6/5) / (2^(1/5) * m^(1/5)) = m / 2^(1/5)
        m = 10.0
        mchirp, _ = mass1_mass2_to_mchirp_q(m, m)
        expected = m / (2 ** (1 / 5))
        assert mchirp == pytest.approx(expected, rel=1e-6)

    def test_asymmetric_masses_q_greater_than_one(self):
        mchirp, q = mass1_mass2_to_mchirp_q(30.0, 5.0)
        assert q == pytest.approx(6.0, rel=1e-6)

    def test_mchirp_always_positive(self):
        for m1, m2 in [(5.0, 5.0), (30.0, 1.0), (1.4, 1.2)]:
            mchirp, _ = mass1_mass2_to_mchirp_q(m1, m2)
            assert mchirp > 0

    def test_tensor_input_shapes(self):
        N = 16
        m1 = torch.rand(N) * 25 + 5
        m2 = torch.rand(N) * 5 + 1
        mchirp, q = mass1_mass2_to_mchirp_q(m1, m2)
        assert mchirp.shape == (N,)
        assert q.shape == (N,)

    def test_tensor_q_equals_ratio(self):
        m1 = torch.tensor([10.0, 20.0, 5.0])
        m2 = torch.tensor([5.0, 4.0, 5.0])
        _, q = mass1_mass2_to_mchirp_q(m1, m2)
        expected_q = m1 / m2
        assert torch.allclose(q, expected_q)

    def test_mchirp_less_than_total_mass(self):
        # Chirp mass is always < total mass
        m1, m2 = 30.0, 10.0
        mchirp, _ = mass1_mass2_to_mchirp_q(m1, m2)
        assert mchirp < m1 + m2


class TestChirpDistanceToDistance:
    def test_reference_mass_identity(self):
        # mchirp = 1.2 Msun → d_L = d_chirp * 1.0
        d_chirp = 100.0
        d_L = chirp_distance_to_distance(d_chirp, mchirp=1.2)
        assert d_L == pytest.approx(d_chirp, rel=1e-6)

    def test_heavier_system_larger_d_L(self):
        # Heavier mchirp → larger (m/1.2)^(5/6) factor → larger d_L
        d_chirp = 200.0
        d_light = chirp_distance_to_distance(d_chirp, mchirp=1.2)
        d_heavy = chirp_distance_to_distance(d_chirp, mchirp=10.0)
        assert d_heavy > d_light

    def test_lighter_system_smaller_d_L(self):
        d_chirp = 150.0
        d_ref = chirp_distance_to_distance(d_chirp, mchirp=1.2)
        d_light = chirp_distance_to_distance(d_chirp, mchirp=0.5)
        assert d_light < d_ref

    def test_scaling_with_chirp_distance(self):
        # d_L should scale linearly with d_chirp
        mchirp = 5.0
        d1 = chirp_distance_to_distance(100.0, mchirp)
        d2 = chirp_distance_to_distance(200.0, mchirp)
        assert d2 == pytest.approx(2 * d1, rel=1e-6)

    def test_tensor_input(self):
        d_chirp = torch.tensor([100.0, 200.0, 300.0])
        mchirp = torch.tensor([1.2, 5.0, 10.0])
        d_L = chirp_distance_to_distance(d_chirp, mchirp)
        assert d_L.shape == (3,)
        assert (d_L > 0).all()
