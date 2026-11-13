"""Unit tests for sage.data.waveform.conversions."""

import pytest

from sage.data.waveform.conversions import mass1_mass2_to_mchirp_q


class TestMass1Mass2ToMchirpQ:
    """Tests of mass1_mass2_to_mchirp_q."""

    def test_equal_mass_q_is_one(self):
        """Equal masses give q = 1."""
        mchirp, q = mass1_mass2_to_mchirp_q(10.0, 10.0)
        assert q == pytest.approx(1.0)

    def test_equal_mass_mchirp_formula(self):
        """Equal masses m give mchirp = m / 2^(1/5)."""
        # mchirp = (m m)^(3/5) / (2m)^(1/5) = m / 2^(1/5)
        m = 10.0
        mchirp, _ = mass1_mass2_to_mchirp_q(m, m)
        expected = m / (2 ** (1 / 5))
        assert mchirp == pytest.approx(expected, rel=1e-6)

    def test_asymmetric_masses_q_greater_than_one(self):
        """mass1 = 6 * mass2 gives q = 6."""
        mchirp, q = mass1_mass2_to_mchirp_q(30.0, 5.0)
        assert q == pytest.approx(6.0, rel=1e-6)


