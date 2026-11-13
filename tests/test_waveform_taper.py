"""Unit tests for sage.data.waveform.taper."""

import sys
import types
import pytest
import torch
from pathlib import Path

# Bypass sage.data.waveform.__init__, which imports IMRPhenomPv2 and so
# astropy, pycbc and lal; taper.py needs only torch.
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

from sage.data.waveform.taper import (
    fd_low_freq_taper,
    fd_high_freq_taper,
    fd_taper,
)


_F_MIN = 20.0
_F_CUT = 512.0
_DF = 0.125       # 1/8 Hz, as for 8 s segments
_WIDTH = 16       # taper width in bins


def _freq_array(f_start, f_end, df):
    """Frequencies from f_start to f_end in steps of df."""
    return torch.arange(f_start, f_end + df, df)


class TestFdLowFreqTaper:
    """Tests of fd_low_freq_taper."""

    def test_zero_below_f_min(self):
        """Weights at or below f_min are zero."""
        f = _freq_array(0.0, _F_MIN, _DF)
        w = fd_low_freq_taper(f, _F_MIN, _DF, _WIDTH)
        assert (w == 0.0).all()

    def test_one_far_above_f_min(self):
        """Weights past the transition band are one."""
        f_start = _F_MIN + (_WIDTH + 5) * _DF
        f = _freq_array(f_start, f_start + 10 * _DF, _DF)
        w = fd_low_freq_taper(f, _F_MIN, _DF, _WIDTH)
        assert torch.allclose(w, torch.ones_like(w), atol=1e-6)

    def test_values_in_zero_one(self):
        """Weights lie in [0, 1]."""
        f = _freq_array(0.0, _F_MIN + (_WIDTH + 10) * _DF, _DF)
        w = fd_low_freq_taper(f, _F_MIN, _DF, _WIDTH)
        assert (w >= 0.0).all()
        assert (w <= 1.0 + 1e-6).all()

    def test_monotone_increasing_in_transition(self):
        """Weights do not decrease across the transition."""
        f = _freq_array(_F_MIN, _F_MIN + _WIDTH * _DF, _DF)
        w = fd_low_freq_taper(f, _F_MIN, _DF, _WIDTH)
        assert (w[1:] >= w[:-1] - 1e-8).all()

    def test_transition_has_intermediate_values(self):
        """The transition midpoint lies strictly between 0 and 1."""
        f_mid = torch.tensor([_F_MIN + (_WIDTH / 2) * _DF])
        w = fd_low_freq_taper(f_mid, _F_MIN, _DF, _WIDTH)
        assert 0.0 < w.item() < 1.0


class TestFdHighFreqTaper:
    """Tests of fd_high_freq_taper."""

    def test_zero_above_f_cut(self):
        """Weights above f_cut are zero."""
        f = _freq_array(_F_CUT + _DF, _F_CUT + 20 * _DF, _DF)
        w = fd_high_freq_taper(f, _F_CUT, _DF, _WIDTH)
        assert (w == 0.0).all()

    def test_one_far_below_f_cut(self):
        """Weights well below the transition band are one."""
        f_end = _F_CUT - (_WIDTH + 5) * _DF
        f = _freq_array(f_end - 10 * _DF, f_end, _DF)
        w = fd_high_freq_taper(f, _F_CUT, _DF, _WIDTH)
        assert torch.allclose(w, torch.ones_like(w), atol=1e-6)

    def test_values_in_zero_one(self):
        """Weights lie in [0, 1]."""
        f = _freq_array(_F_CUT - (_WIDTH + 10) * _DF, _F_CUT + 10 * _DF, _DF)
        w = fd_high_freq_taper(f, _F_CUT, _DF, _WIDTH)
        assert (w >= 0.0).all()
        assert (w <= 1.0 + 1e-6).all()

    def test_monotone_decreasing_in_transition(self):
        """Weights do not increase across the transition."""
        f = _freq_array(_F_CUT - _WIDTH * _DF, _F_CUT, _DF)
        w = fd_high_freq_taper(f, _F_CUT, _DF, _WIDTH)
        assert (w[1:] <= w[:-1] + 1e-8).all()

    def test_transition_has_intermediate_values(self):
        """The transition midpoint lies strictly between 0 and 1."""
        f_mid = torch.tensor([_F_CUT - (_WIDTH / 2) * _DF])
        w = fd_high_freq_taper(f_mid, _F_CUT, _DF, _WIDTH)
        assert 0.0 < w.item() < 1.0


class TestFdTaper:
    """Tests of fd_taper, the combined band-pass."""

    def test_values_in_zero_one(self):
        """Weights lie in [0, 1]."""
        f = _freq_array(0.0, _F_CUT + 20 * _DF, _DF)
        w = fd_taper(f, _F_MIN, _F_CUT, _DF, low_width=_WIDTH, high_width=_WIDTH)
        assert (w >= 0.0).all()
        assert (w <= 1.0 + 1e-6).all()

    def test_zero_below_f_min(self):
        """Weights below f_min are zero."""
        f = _freq_array(0.0, _F_MIN - _DF, _DF)
        w = fd_taper(f, _F_MIN, _F_CUT, _DF, low_width=_WIDTH, high_width=_WIDTH)
        assert (w == 0.0).all()

    def test_zero_above_f_cut(self):
        """Weights above f_cut are zero."""
        f = _freq_array(_F_CUT + _DF, _F_CUT + 20 * _DF, _DF)
        w = fd_taper(f, _F_MIN, _F_CUT, _DF, low_width=_WIDTH, high_width=_WIDTH)
        assert (w == 0.0).all()

    def test_interior_is_one(self):
        """Weights well inside [f_min, f_cut] are one."""
        f_lo = _F_MIN + (_WIDTH + 5) * _DF
        f_hi = _F_CUT - (_WIDTH + 5) * _DF
        if f_lo >= f_hi:
            pytest.skip("No interior region with these test parameters")
        f = _freq_array(f_lo, f_hi, _DF)
        w = fd_taper(f, _F_MIN, _F_CUT, _DF, low_width=_WIDTH, high_width=_WIDTH)
        assert torch.allclose(w, torch.ones_like(w), atol=1e-5)

