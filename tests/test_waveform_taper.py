"""Unit tests for sage.data.waveform.taper."""

import pytest
import torch

from sage.data.waveform.taper import (
    fd_low_freq_taper,
    fd_high_freq_taper,
    fd_taper,
)


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

_F_MIN = 20.0
_F_CUT = 512.0
_DF = 0.125       # 1/8 Hz — typical for 8-second segments
_WIDTH = 16       # taper width in bins


def _freq_array(f_start, f_end, df):
    return torch.arange(f_start, f_end + df, df)


# ---------------------------------------------------------------------------
# fd_low_freq_taper
# ---------------------------------------------------------------------------

class TestFdLowFreqTaper:
    def test_zero_below_f_min(self):
        f = _freq_array(0.0, _F_MIN, _DF)
        w = fd_low_freq_taper(f, _F_MIN, _DF, _WIDTH)
        # Frequencies at or below f_min → weight = 0
        assert (w == 0.0).all()

    def test_one_far_above_f_min(self):
        # Well past the transition band, weights should be 1
        f_start = _F_MIN + (_WIDTH + 5) * _DF
        f = _freq_array(f_start, f_start + 10 * _DF, _DF)
        w = fd_low_freq_taper(f, _F_MIN, _DF, _WIDTH)
        assert torch.allclose(w, torch.ones_like(w), atol=1e-6)

    def test_values_in_zero_one(self):
        f = _freq_array(0.0, _F_MIN + (_WIDTH + 10) * _DF, _DF)
        w = fd_low_freq_taper(f, _F_MIN, _DF, _WIDTH)
        assert (w >= 0.0).all()
        assert (w <= 1.0 + 1e-6).all()

    def test_monotone_increasing_in_transition(self):
        # Taper must be non-decreasing across the transition band
        f = _freq_array(_F_MIN, _F_MIN + _WIDTH * _DF, _DF)
        w = fd_low_freq_taper(f, _F_MIN, _DF, _WIDTH)
        assert (w[1:] >= w[:-1] - 1e-8).all()

    def test_transition_has_intermediate_values(self):
        # At the midpoint of the transition, values should be between 0 and 1
        f_mid = torch.tensor([_F_MIN + (_WIDTH / 2) * _DF])
        w = fd_low_freq_taper(f_mid, _F_MIN, _DF, _WIDTH)
        assert 0.0 < w.item() < 1.0


# ---------------------------------------------------------------------------
# fd_high_freq_taper
# ---------------------------------------------------------------------------

class TestFdHighFreqTaper:
    def test_zero_above_f_cut(self):
        f = _freq_array(_F_CUT + _DF, _F_CUT + 20 * _DF, _DF)
        w = fd_high_freq_taper(f, _F_CUT, _DF, _WIDTH)
        assert (w == 0.0).all()

    def test_one_far_below_f_cut(self):
        f_end = _F_CUT - (_WIDTH + 5) * _DF
        f = _freq_array(f_end - 10 * _DF, f_end, _DF)
        w = fd_high_freq_taper(f, _F_CUT, _DF, _WIDTH)
        assert torch.allclose(w, torch.ones_like(w), atol=1e-6)

    def test_values_in_zero_one(self):
        f = _freq_array(_F_CUT - (_WIDTH + 10) * _DF, _F_CUT + 10 * _DF, _DF)
        w = fd_high_freq_taper(f, _F_CUT, _DF, _WIDTH)
        assert (w >= 0.0).all()
        assert (w <= 1.0 + 1e-6).all()

    def test_monotone_decreasing_in_transition(self):
        f = _freq_array(_F_CUT - _WIDTH * _DF, _F_CUT, _DF)
        w = fd_high_freq_taper(f, _F_CUT, _DF, _WIDTH)
        assert (w[1:] <= w[:-1] + 1e-8).all()

    def test_transition_has_intermediate_values(self):
        f_mid = torch.tensor([_F_CUT - (_WIDTH / 2) * _DF])
        w = fd_high_freq_taper(f_mid, _F_CUT, _DF, _WIDTH)
        assert 0.0 < w.item() < 1.0


# ---------------------------------------------------------------------------
# fd_taper (combined band-pass)
# ---------------------------------------------------------------------------

class TestFdTaper:
    def test_values_in_zero_one(self):
        f = _freq_array(0.0, _F_CUT + 20 * _DF, _DF)
        w = fd_taper(f, _F_MIN, _F_CUT, _DF, low_width=_WIDTH, high_width=_WIDTH)
        assert (w >= 0.0).all()
        assert (w <= 1.0 + 1e-6).all()

    def test_zero_below_f_min(self):
        f = _freq_array(0.0, _F_MIN - _DF, _DF)
        w = fd_taper(f, _F_MIN, _F_CUT, _DF, low_width=_WIDTH, high_width=_WIDTH)
        assert (w == 0.0).all()

    def test_zero_above_f_cut(self):
        f = _freq_array(_F_CUT + _DF, _F_CUT + 20 * _DF, _DF)
        w = fd_taper(f, _F_MIN, _F_CUT, _DF, low_width=_WIDTH, high_width=_WIDTH)
        assert (w == 0.0).all()

    def test_interior_is_one(self):
        # Well inside [f_min, f_cut], the taper should be 1
        f_lo = _F_MIN + (_WIDTH + 5) * _DF
        f_hi = _F_CUT - (_WIDTH + 5) * _DF
        if f_lo >= f_hi:
            pytest.skip("No interior region with these test parameters")
        f = _freq_array(f_lo, f_hi, _DF)
        w = fd_taper(f, _F_MIN, _F_CUT, _DF, low_width=_WIDTH, high_width=_WIDTH)
        assert torch.allclose(w, torch.ones_like(w), atol=1e-5)

    def test_output_same_shape_as_input(self):
        f = _freq_array(0.0, 600.0, _DF)
        w = fd_taper(f, _F_MIN, _F_CUT, _DF)
        assert w.shape == f.shape
