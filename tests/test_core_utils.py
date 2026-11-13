"""Unit tests for sage.core.utils and sage.core.constants."""

import math
import pytest
import numpy as np

from sage.core.utils import to_sequence, ensure_1d
import sage.core.constants as C


class TestToSequence:
    """Tests of to_sequence."""

    def test_scalar_int(self):
        """An int becomes a one-element tuple."""
        assert to_sequence(5) == (5,)

    def test_scalar_float(self):
        """A float becomes a one-element tuple."""
        assert to_sequence(3.14) == (3.14,)

    def test_scalar_bool(self):
        """A bool becomes a one-element tuple."""
        assert to_sequence(True) == (True,)

    def test_string_not_split(self):
        """A string is kept whole, not split into characters."""
        result = to_sequence("abc")
        assert result == ("abc",)

    def test_bytes_not_split(self):
        """Bytes are kept whole."""
        result = to_sequence(b"abc")
        assert result == (b"abc",)

    def test_none_returns_none(self):
        """None is returned as None."""
        assert to_sequence(None) is None

    def test_list_becomes_tuple(self):
        """A list becomes a tuple."""
        assert to_sequence([1, 2, 3]) == (1, 2, 3)

    def test_tuple_passthrough(self):
        """A tuple is returned unchanged."""
        assert to_sequence((4, 5)) == (4, 5)

    def test_numpy_array(self):
        """A numpy array becomes a tuple of its elements."""
        arr = np.array([7, 8, 9])
        result = to_sequence(arr)
        assert result == (7, 8, 9)


class TestEnsure1d:
    """Tests of ensure_1d."""

    def test_list_input(self):
        """A list becomes a 1-D numpy array."""
        result = ensure_1d([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        np.testing.assert_array_equal(result, [1, 2, 3])


    def test_2d_array_raises(self):
        """A 2-D array is refused."""
        with pytest.raises(ValueError, match="1D"):
            ensure_1d(np.zeros((3, 4)))

    def test_scalar_raises(self):
        """A scalar is refused."""
        # np.asarray(5).ndim == 0, not 1
        with pytest.raises(ValueError):
            ensure_1d(5)


class TestPhysicalConstants:
    """Tests of the values in sage.core.constants."""

    def test_speed_of_light(self):
        """C is the exact SI speed of light."""
        # NIST: 299 792 458 m/s exactly
        assert C.C == pytest.approx(299_792_458.0, rel=1e-9)

    def test_gravitational_constant(self):
        """G matches the NIST value to 1e-4."""
        # NIST: 6.674 30 x 10^-11 m^3 kg^-1 s^-2
        assert C.G == pytest.approx(6.67430e-11, rel=1e-4)

    def test_solar_mass(self):
        """MSUN matches 1.989e30 kg to 1e-3."""
        assert C.MSUN == pytest.approx(1.989e30, rel=1e-3)

    def test_pi(self):
        """PI equals math.pi."""
        assert C.PI == pytest.approx(math.pi, rel=1e-15)


    def test_euler_gamma(self):
        """EulerGamma is the Euler-Mascheroni constant."""
        assert C.EulerGamma == pytest.approx(0.5772156649, rel=1e-8)

    def test_mpc_in_metres(self):
        """Mpc is 3.086e22 m to 1e-2."""
        assert C.Mpc == pytest.approx(3.086e22, rel=1e-2)

