"""Tests for the sage.data.waveform prior distributions: ranges and
sample statistics."""

import sys
import types
import pytest
import torch
from pathlib import Path

_SAGE = Path(__file__).resolve().parents[1] / "sage"


def _bypass_pkg(name):
    """Register a bare package so its __init__ is not run."""
    if name not in sys.modules:
        parts = name.split(".")[1:]
        mod = types.ModuleType(name)
        mod.__path__ = [str(_SAGE.joinpath(*parts))]
        mod.__package__ = name
        sys.modules[name] = mod


_bypass_pkg("sage.data.waveform")
_bypass_pkg("sage.data.waveform.distributions")

from sage.data.waveform.distributions.uniform import Uniform
from sage.data.waveform.distributions.cosmology import UniformComovingVolume
from sage.data.waveform.distributions.powerlaw import UniformPowerLaw, UniformRadius

N = 50_000  # large enough for statistical checks


class TestUniform:
    """Tests of Uniform."""

    def test_samples_in_range(self):
        """Samples lie within [low, high]."""
        u = Uniform(low=2.0, high=5.0)
        s = u.sample((N,))
        assert s.min().item() >= 2.0
        assert s.max().item() <= 5.0

    def test_mean_close_to_midpoint(self):
        """The sample mean is the midpoint."""
        u = Uniform(low=0.0, high=10.0)
        s = u.sample((N,))
        assert s.mean().item() == pytest.approx(5.0, abs=0.1)

    def test_float64_uses_exact_bounds(self):
        """float64 samples are low + (high - low) * u exactly."""
        s = Uniform(low=0.1, high=0.99).sample((1000,), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        u = torch.rand((1000,), dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        assert torch.equal(s, 0.1 + (0.99 - 0.1) * u)


    def test_generator_reproducible(self):
        """The same seed gives the same samples."""
        u = Uniform(low=0.0, high=1.0)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        a = u.sample((20,), generator=g1)
        b = u.sample((20,), generator=g2)
        assert torch.allclose(a, b)


    def test_negative_range(self):
        """A negative range is respected."""
        u = Uniform(low=-5.0, high=-1.0)
        s = u.sample((N,))
        assert s.min().item() >= -5.0
        assert s.max().item() <= -1.0

    def test_unit_interval_covers_full_range(self):
        """Samples reach both ends of [0, 1]."""
        u = Uniform(low=0.0, high=1.0)
        s = u.sample((N,))
        assert s.min().item() < 0.01
        assert s.max().item() > 0.99


class TestUniformPowerLaw:
    """Tests of UniformPowerLaw."""

    def test_samples_in_range(self):
        """Samples lie within [low, high]."""
        p = UniformPowerLaw(low=10.0, high=1000.0, dim=3)
        # With a float32 inverse CDF, seed 56 draws above high.
        s = p.sample((100_000,), generator=torch.Generator().manual_seed(56))
        assert s.min().item() >= 10.0
        assert s.max().item() <= 1000.0

    def test_dim1_is_uniform(self):
        """dim=1 is uniform."""
        p = UniformPowerLaw(low=0.0, high=1.0, dim=1)
        s = p.sample((N,))
        assert s.mean().item() == pytest.approx(0.5, abs=0.02)

    def test_dim3_skewed_toward_high(self):
        """dim=3 has mean 3/4 on [0, 1]."""
        # Uniform in volume: pdf proportional to r^2.
        p = UniformPowerLaw(low=0.0, high=1.0, dim=3)
        s = p.sample((N,))
        assert s.mean().item() == pytest.approx(0.75, abs=0.02)

    def test_dim2_mean_is_two_thirds(self):
        """dim=2 has mean 2/3 on [0, 1]."""
        p = UniformPowerLaw(low=0.0, high=1.0, dim=2)
        s = p.sample((N,))
        assert s.mean().item() == pytest.approx(2.0 / 3.0, abs=0.02)


    def test_generator_reproducible(self):
        """The same seed gives the same samples."""
        p = UniformPowerLaw(low=1.0, high=100.0, dim=3)
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        a = p.sample((20,), generator=g1)
        b = p.sample((20,), generator=g2)
        assert torch.allclose(a, b)


class TestUniformRadius:
    """Tests of UniformRadius."""


    def test_samples_in_range(self):
        """Samples lie within [low, high]."""
        r = UniformRadius(low=10.0, high=500.0)
        s = r.sample((N,))
        assert s.min().item() >= 10.0
        assert s.max().item() <= 500.0


def test_comoving_volume_warns():
    """UniformComovingVolume warns that it is not from PyCBC."""
    with pytest.warns(UserWarning, match="not a PyCBC distribution"):
        UniformComovingVolume(100.0, 200.0)
