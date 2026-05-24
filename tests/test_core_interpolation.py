"""Unit tests for sage.core.interpolation — pure-torch interpolation functions.

These interpolants underpin the IMRPhenomD/Pv2 waveform approximants: correctness
here means the phase and amplitude splines match their C/LAL counterparts.
"""

import math
import pytest
import torch
import numpy as np
from sage.core.interpolation import (
    torch_linear_interp,
    torch_scipylike_cubic_interp,
    torch_catmull_rom_cubic_interp,
    torch_natural_cubic_coeffs,
    torch_natural_cubic_interp,
)


# ---------------------------------------------------------------------------
# torch_linear_interp
# ---------------------------------------------------------------------------

class TestTorchLinearInterp:
    def test_exact_at_nodes(self):
        xp = torch.linspace(0.0, 1.0, 20)
        fp = torch.sin(xp)
        out = torch_linear_interp(xp, xp, fp)
        assert torch.allclose(out, fp, atol=1e-6)

    def test_midpoint_is_average(self):
        xp = torch.tensor([0.0, 2.0])
        fp = torch.tensor([0.0, 4.0])
        out = torch_linear_interp(torch.tensor([1.0]), xp, fp)
        assert out.item() == pytest.approx(2.0)

    def test_linear_function_exact(self):
        # f(x) = 3x + 2 must be reproduced exactly everywhere interior
        xp = torch.linspace(0.0, 5.0, 12)
        fp = 3.0 * xp + 2.0
        x = torch.linspace(0.1, 4.9, 60)
        out = torch_linear_interp(x, xp, fp)
        assert torch.allclose(out, 3.0 * x + 2.0, atol=1e-5)

    def test_extrapolates_linearly_below(self):
        # Below xp[0]: function uses the first interval and continues the slope
        xp = torch.tensor([1.0, 2.0, 3.0])
        fp = torch.tensor([10.0, 20.0, 30.0])  # slope = 10
        out = torch_linear_interp(torch.tensor([0.0]), xp, fp)
        assert out.item() == pytest.approx(0.0)  # 10 + 10*(0-1) = 0

    def test_extrapolates_linearly_above(self):
        xp = torch.tensor([0.0, 1.0, 2.0])
        fp = torch.tensor([0.0, 3.0, 6.0])  # slope = 3
        out = torch_linear_interp(torch.tensor([4.0]), xp, fp)
        assert out.item() == pytest.approx(12.0)  # 3 + 3*(4-1) = 12

    def test_output_shape_preserved(self):
        xp = torch.linspace(0.0, 1.0, 20)
        fp = torch.rand(20)
        x = torch.rand(7)
        assert torch_linear_interp(x, xp, fp).shape == (7,)

    def test_matches_numpy_interp_interior(self):
        # numpy.interp clamps at boundaries; only compare on interior points
        xp = torch.linspace(0.0, 2 * math.pi, 50)
        fp = torch.sin(xp)
        x = torch.linspace(0.5, 5.8, 80)  # strictly inside [0, 2π]
        out = torch_linear_interp(x, xp, fp)
        ref = torch.tensor(
            np.interp(x.numpy(), xp.numpy(), fp.numpy()), dtype=torch.float32
        )
        assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# torch_scipylike_cubic_interp
# ---------------------------------------------------------------------------

class TestTorchScipylikeCubicInterp:
    def test_linear_function_exact(self):
        xp = torch.linspace(0.0, 4.0, 12)
        fp = 2.0 * xp - 1.0
        x = torch.linspace(0.5, 3.5, 40)
        out = torch_scipylike_cubic_interp(x, xp, fp)
        assert torch.allclose(out, 2.0 * x - 1.0, atol=1e-4)

    def test_linear_at_interval_midpoints(self):
        # Midpoints between consecutive nodes — these land cleanly inside one interval
        xp = torch.linspace(0.0, 4.0, 12)
        fp = 5.0 * xp + 3.0
        x_mid = (xp[1:-2] + xp[2:-1]) / 2  # midpoints of interior intervals
        out = torch_scipylike_cubic_interp(x_mid, xp, fp)
        assert torch.allclose(out, 5.0 * x_mid + 3.0, atol=1e-5)

    def test_output_shape_matches_input(self):
        xp = torch.linspace(0.0, 1.0, 20)
        fp = torch.rand(20)
        x = torch.rand(15)
        assert torch_scipylike_cubic_interp(x, xp, fp).shape == (15,)

    def test_values_bounded_near_data(self):
        # For monotone data, cubic values should stay in the data range
        xp = torch.linspace(0.0, 1.0, 15)
        fp = xp  # identity, strictly increasing
        x = torch.linspace(0.05, 0.95, 60)
        out = torch_scipylike_cubic_interp(x, xp, fp)
        assert out.min() >= -0.01
        assert out.max() <= 1.01


# ---------------------------------------------------------------------------
# torch_catmull_rom_cubic_interp
# ---------------------------------------------------------------------------

class TestTorchCatmullRomCubicInterp:
    def test_linear_function_exact(self):
        # Catmull-Rom is exact for linear functions on a uniform grid
        N, x0, dx = 30, 0.0, 0.1
        xgrid = torch.arange(N, dtype=torch.float32) * dx + x0
        y = 2.0 * xgrid + 1.0  # f(x) = 2x + 1
        # Query at non-node interior points
        xs = torch.tensor([x0 + dx * (i + 0.5) for i in range(2, N - 3)])
        out = torch_catmull_rom_cubic_interp(xs, y, x0, dx)
        assert torch.allclose(out, 2.0 * xs + 1.0, atol=1e-4)

    def test_constant_function(self):
        N, x0, dx = 20, 0.0, 0.5
        y = torch.full((N,), 3.14)
        xs = torch.linspace(x0 + dx * 2, x0 + dx * (N - 4), 12)
        out = torch_catmull_rom_cubic_interp(xs, y, x0, dx)
        assert torch.allclose(out, torch.full_like(out, 3.14), atol=1e-5)

    def test_output_shape_preserved(self):
        N, x0, dx = 50, 0.0, 0.01
        y = torch.rand(N)
        xs = torch.rand(16) * (dx * (N - 6)) + x0 + dx * 2
        out = torch_catmull_rom_cubic_interp(xs, y, x0, dx)
        assert out.shape == (16,)

    def test_monotone_ramp_stays_in_range(self):
        # For a linear ramp, output should stay in the data range
        N, x0, dx = 30, 0.0, 0.1
        y = torch.linspace(0.0, 1.0, N)
        xs = torch.linspace(x0 + dx * 2, x0 + dx * (N - 4), 50)
        out = torch_catmull_rom_cubic_interp(xs, y, x0, dx)
        assert out.min() >= -0.01
        assert out.max() <= 1.01


# ---------------------------------------------------------------------------
# torch_natural_cubic_coeffs + torch_natural_cubic_interp
# ---------------------------------------------------------------------------

class TestTorchNaturalCubicSpline:
    """Natural cubic spline interpolation used to match LAL's gsl_interp_cspline."""

    @staticmethod
    def _spline(f, n=15, x0=0.0, x1=1.0, dtype=torch.float64):
        xp = torch.linspace(x0, x1, n, dtype=dtype)
        fp = f(xp)
        M = torch_natural_cubic_coeffs(xp, fp)
        return xp, fp, M

    def test_node_recovery(self):
        xp, fp, M = self._spline(torch.sin)
        out = torch_natural_cubic_interp(xp, xp, fp, M)
        assert torch.allclose(out, fp, atol=1e-8)

    def test_linear_function_exact(self):
        xp, fp, M = self._spline(lambda x: 3.0 * x + 1.0)
        x = torch.linspace(0.05, 0.95, 30, dtype=torch.float64)
        out = torch_natural_cubic_interp(x, xp, fp, M)
        assert torch.allclose(out, 3.0 * x + 1.0, atol=1e-8)

    def test_close_to_scipy_natural_cubicspline(self):
        from scipy.interpolate import CubicSpline
        xp, fp, M = self._spline(torch.sin, n=20, x0=0.0, x1=math.pi)
        x = torch.linspace(0.2, 3.0, 60, dtype=torch.float64)
        out = torch_natural_cubic_interp(x, xp, fp, M)
        cs = CubicSpline(xp.numpy(), fp.numpy(), bc_type="natural")
        ref = torch.tensor(cs(x.numpy()), dtype=torch.float64)
        # Implementation closely approximates scipy's natural cubic spline
        assert torch.allclose(out, ref, atol=1e-4)

    def test_derivative_of_linear_is_slope(self):
        # d/dx (3x + 1) = 3 everywhere
        xp, fp, M = self._spline(lambda x: 3.0 * x + 1.0)
        x = torch.linspace(0.1, 0.9, 20, dtype=torch.float64)
        deriv = torch_natural_cubic_interp(x, xp, fp, M, derivative=True)
        assert torch.allclose(deriv, torch.full_like(deriv, 3.0), atol=1e-6)

    def test_derivative_close_to_scipy(self):
        from scipy.interpolate import CubicSpline
        xp, fp, M = self._spline(torch.cos, n=20, x0=0.0, x1=math.pi)
        x = torch.linspace(0.1, 3.0, 40, dtype=torch.float64)
        deriv = torch_natural_cubic_interp(x, xp, fp, M, derivative=True)
        cs = CubicSpline(xp.numpy(), fp.numpy(), bc_type="natural")
        ref = torch.tensor(cs(x.numpy(), 1), dtype=torch.float64)
        assert torch.allclose(deriv, ref, atol=0.01)

    def test_batch_query_shape(self):
        xp, fp, M = self._spline(torch.exp)
        x = torch.rand(24, dtype=torch.float64) * 0.8 + 0.1
        out = torch_natural_cubic_interp(x, xp, fp, M)
        assert out.shape == (24,)
