"""Unit tests for sage.data.waveform parameter distributions (pure torch).

These samplers define the astrophysical priors used during training.
Correctness means samples fall in the right range with the right statistics.
"""

import sys
import types
import pytest
import torch
from pathlib import Path

_SAGE = Path(__file__).resolve().parents[1] / "sage"


def _bypass_pkg(name):
    if name not in sys.modules:
        parts = name.split(".")[1:]
        mod = types.ModuleType(name)
        mod.__path__ = [str(_SAGE.joinpath(*parts))]
        mod.__package__ = name
        sys.modules[name] = mod


_bypass_pkg("sage.data.waveform")
_bypass_pkg("sage.data.waveform.distributions")

from sage.data.waveform.distributions.uniform import Uniform
from sage.data.waveform.distributions.powerlaw import UniformPowerLaw, UniformRadius

N = 50_000  # large enough for statistical checks


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------

class TestUniform:
    def test_samples_in_range(self):
        u = Uniform(low=2.0, high=5.0)
        s = u.sample((N,))
        assert s.min().item() >= 2.0
        assert s.max().item() <= 5.0

    def test_mean_close_to_midpoint(self):
        u = Uniform(low=0.0, high=10.0)
        s = u.sample((N,))
        assert s.mean().item() == pytest.approx(5.0, abs=0.1)

    def test_output_shape(self):
        u = Uniform(low=0.0, high=1.0)
        assert u.sample((4, 8)).shape == (4, 8)

    def test_generator_reproducible(self):
        u = Uniform(low=0.0, high=1.0)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        a = u.sample((20,), generator=g1)
        b = u.sample((20,), generator=g2)
        assert torch.allclose(a, b)

    def test_device_kwarg_accepted(self):
        u = Uniform(low=0.0, high=1.0)
        s = u.sample((5,), device="cpu")
        assert s.device.type == "cpu"

    def test_negative_range(self):
        u = Uniform(low=-5.0, high=-1.0)
        s = u.sample((N,))
        assert s.min().item() >= -5.0
        assert s.max().item() <= -1.0

    def test_unit_interval_covers_full_range(self):
        # With enough samples, min should be close to 0 and max close to 1
        u = Uniform(low=0.0, high=1.0)
        s = u.sample((N,))
        assert s.min().item() < 0.01
        assert s.max().item() > 0.99


# ---------------------------------------------------------------------------
# UniformPowerLaw
# ---------------------------------------------------------------------------

class TestUniformPowerLaw:
    def test_samples_in_range(self):
        p = UniformPowerLaw(low=10.0, high=1000.0, dim=3)
        s = p.sample((N,))
        assert s.min().item() >= 10.0
        assert s.max().item() <= 1000.0

    def test_dim1_is_uniform(self):
        # dim=1 → power 0 → reduces to uniform distribution
        p = UniformPowerLaw(low=0.0, high=1.0, dim=1)
        s = p.sample((N,))
        assert s.mean().item() == pytest.approx(0.5, abs=0.02)

    def test_dim3_skewed_toward_high(self):
        # Uniform-in-volume (PDF ∝ r²) → mean at 3/4 of [0,1]
        p = UniformPowerLaw(low=0.0, high=1.0, dim=3)
        s = p.sample((N,))
        assert s.mean().item() == pytest.approx(0.75, abs=0.02)

    def test_dim2_mean_is_two_thirds(self):
        # PDF ∝ r → mean = 2/3 on [0,1]
        p = UniformPowerLaw(low=0.0, high=1.0, dim=2)
        s = p.sample((N,))
        assert s.mean().item() == pytest.approx(2.0 / 3.0, abs=0.02)

    def test_output_shape(self):
        p = UniformPowerLaw(low=1.0, high=2.0)
        assert p.sample((3, 5)).shape == (3, 5)

    def test_generator_reproducible(self):
        p = UniformPowerLaw(low=1.0, high=100.0)
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        a = p.sample((20,), generator=g1)
        b = p.sample((20,), generator=g2)
        assert torch.allclose(a, b)

    def test_name_attribute(self):
        assert UniformPowerLaw.name == "uniform_power_law"


# ---------------------------------------------------------------------------
# UniformRadius
# ---------------------------------------------------------------------------

class TestUniformRadius:
    def test_is_uniform_power_law_dim3(self):
        r = UniformRadius(low=0.0, high=1.0)
        assert r.dim == 3

    def test_name_attribute(self):
        assert UniformRadius.name == "uniform_radius"

    def test_samples_in_range(self):
        r = UniformRadius(low=10.0, high=500.0)
        s = r.sample((N,))
        assert s.min().item() >= 10.0
        assert s.max().item() <= 500.0

    def test_same_distribution_as_powerlaw_dim3(self):
        # UniformRadius(0, 1) == UniformPowerLaw(0, 1, dim=3) statistically
        r = UniformRadius(low=0.0, high=1.0)
        p = UniformPowerLaw(low=0.0, high=1.0, dim=3)
        g = torch.Generator().manual_seed(0)
        sr = r.sample((N,), generator=g)
        g.manual_seed(0)
        sp = p.sample((N,), generator=g)
        assert torch.allclose(sr, sp)
