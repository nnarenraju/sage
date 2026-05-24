"""Unit tests for sage.core.utils and sage.core.constants."""

import math
import pytest
import numpy as np

from sage.core.utils import to_sequence, ensure_1d
import sage.core.constants as C


# ---------------------------------------------------------------------------
# to_sequence
# ---------------------------------------------------------------------------

class TestToSequence:
    def test_scalar_int(self):
        assert to_sequence(5) == (5,)

    def test_scalar_float(self):
        assert to_sequence(3.14) == (3.14,)

    def test_scalar_bool(self):
        assert to_sequence(True) == (True,)

    def test_string_not_split(self):
        # A string must NOT be split into characters
        result = to_sequence("abc")
        assert result == ("abc",)

    def test_bytes_not_split(self):
        result = to_sequence(b"abc")
        assert result == (b"abc",)

    def test_none_returns_none(self):
        assert to_sequence(None) is None

    def test_list_becomes_tuple(self):
        assert to_sequence([1, 2, 3]) == (1, 2, 3)

    def test_tuple_passthrough(self):
        assert to_sequence((4, 5)) == (4, 5)

    def test_numpy_array(self):
        arr = np.array([7, 8, 9])
        result = to_sequence(arr)
        assert result == (7, 8, 9)


# ---------------------------------------------------------------------------
# ensure_1d
# ---------------------------------------------------------------------------

class TestEnsure1d:
    def test_list_input(self):
        result = ensure_1d([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_numpy_1d_passthrough(self):
        arr = np.array([0.1, 0.2, 0.3])
        result = ensure_1d(arr)
        assert result.ndim == 1

    def test_2d_array_raises(self):
        with pytest.raises(ValueError, match="1D"):
            ensure_1d(np.zeros((3, 4)))

    def test_scalar_raises(self):
        # np.asarray(5).ndim == 0, not 1
        with pytest.raises(ValueError):
            ensure_1d(5)

    def test_empty_array(self):
        result = ensure_1d([])
        assert result.ndim == 1
        assert len(result) == 0


# ---------------------------------------------------------------------------
# sage.core.constants
# ---------------------------------------------------------------------------

class TestPhysicalConstants:
    def test_speed_of_light(self):
        # NIST: 299 792 458 m/s exactly
        assert C.C == pytest.approx(299_792_458.0, rel=1e-9)

    def test_gravitational_constant(self):
        # NIST: 6.674 30 × 10^-11 m^3 kg^-1 s^-2
        assert C.G == pytest.approx(6.67430e-11, rel=1e-4)

    def test_solar_mass(self):
        # IAU 2015: 1.988 409 902... × 10^30 kg
        assert C.MSUN == pytest.approx(1.989e30, rel=1e-3)

    def test_pi(self):
        assert C.PI == pytest.approx(math.pi, rel=1e-15)

    def test_two_pi(self):
        assert C.TWOPI == pytest.approx(2.0 * math.pi, rel=1e-15)

    def test_euler_gamma(self):
        # Euler–Mascheroni constant ≈ 0.5772...
        assert C.EulerGamma == pytest.approx(0.5772156649, rel=1e-8)

    def test_mpc_in_metres(self):
        # 1 Mpc ≈ 3.086 × 10^22 m
        assert C.Mpc == pytest.approx(3.086e22, rel=1e-2)

    def test_gm_has_correct_units_of_time(self):
        # GM = G * MSUN / C^3 should be ~4.93e-6 s (half-period of a 1-Msun binary)
        expected = C.G * C.MSUN / (C.C ** 3)
        assert C.GM == pytest.approx(expected, rel=1e-9)
