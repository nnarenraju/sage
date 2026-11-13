"""Unit tests for sage.core.math."""

import math
import pytest
import numpy as np

from sage.core.math import Normalise, Standardise, rotation_matrix


class TestNormalise:
    """Tests of Normalise on [2, 10]."""

    def setup_method(self):
        """A Normalise over [2, 10]."""
        self.n = Normalise(min_val=2.0, max_val=10.0)

    def test_min_maps_to_zero(self):
        """The minimum maps to 0."""
        assert self.n.norm(2.0) == pytest.approx(0.0)

    def test_max_maps_to_one(self):
        """The maximum maps to 1."""
        assert self.n.norm(10.0) == pytest.approx(1.0)

    def test_midpoint_maps_to_half(self):
        """The midpoint maps to 0.5."""
        assert self.n.norm(6.0) == pytest.approx(0.5)

    def test_roundtrip_scalar(self):
        """unnorm inverts norm for scalars."""
        for x in [2.0, 3.5, 7.0, 10.0]:
            assert self.n.unnorm(self.n.norm(x)) == pytest.approx(x, rel=1e-10)

    def test_roundtrip_array(self):
        """unnorm inverts norm for arrays."""
        xs = np.linspace(2.0, 10.0, 50)
        recovered = self.n.unnorm(self.n.norm(xs))
        np.testing.assert_allclose(recovered, xs, rtol=1e-12)

    def test_unnorm_at_zero_gives_min(self):
        """unnorm(0) is the minimum."""
        assert self.n.unnorm(0.0) == pytest.approx(2.0)

    def test_unnorm_at_one_gives_max(self):
        """unnorm(1) is the maximum."""
        assert self.n.unnorm(1.0) == pytest.approx(10.0)


class TestStandardise:
    """Tests of Standardise with mean 5 and std 2."""

    def setup_method(self):
        """A Standardise with mean 5 and std 2."""
        self.s = Standardise(mean=5.0, std=2.0)

    def test_mean_maps_to_zero(self):
        """The mean maps to 0."""
        assert self.s.norm(5.0) == pytest.approx(0.0, abs=1e-7)

    def test_norm_unit_variance(self):
        """One std above the mean maps to 1."""
        result = self.s.norm(5.0 + 2.0)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_roundtrip_scalar(self):
        """unnorm inverts norm for scalars."""
        for x in [1.0, 3.0, 5.0, 8.0, 12.0]:
            assert self.s.unnorm(self.s.norm(x)) == pytest.approx(x, rel=1e-10)

    def test_roundtrip_array(self):
        """unnorm inverts norm for arrays."""
        xs = np.linspace(-10.0, 20.0, 100)
        recovered = self.s.unnorm(self.s.norm(xs))
        np.testing.assert_allclose(recovered, xs, rtol=1e-12)

    def test_eps_guard_zero_std(self):
        """A zero std gives a finite value through eps."""
        s_zero = Standardise(mean=3.0, std=0.0, eps=1e-8)
        result = s_zero.norm(5.0)
        assert math.isfinite(result)


class TestRotationMatrix:
    """Tests of rotation_matrix."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_orthogonality(self, axis):
        """R R^T is the identity about each axis."""
        R = rotation_matrix(math.pi / 4, axis=axis)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_determinant_one(self, axis):
        """The determinant is 1 about each axis."""
        R = rotation_matrix(math.pi / 3, axis=axis)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)

    def test_zero_angle_is_identity(self):
        """A zero angle gives the identity."""
        for axis in (0, 1, 2):
            R = rotation_matrix(0.0, axis=axis)
            np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_pi_rotation_axis_z(self):
        """A rotation by pi about z maps x to -x."""
        R = rotation_matrix(math.pi, axis=2)
        # 180 deg about z: (x, y) -> (-x, -y), z unchanged
        v = np.array([1.0, 0.0, 0.0])
        result = R @ v
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-14)

    def test_invalid_axis_raises(self):
        """An axis above 2 is refused."""
        with pytest.raises(ValueError, match="Axis must be"):
            rotation_matrix(1.0, axis=3)

    def test_invalid_axis_negative_raises(self):
        """A negative axis is refused."""
        with pytest.raises(ValueError):
            rotation_matrix(1.0, axis=-1)

