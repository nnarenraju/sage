"""Tests for sage.core.geometry: the window computed from data settings."""

import pytest

from sage.core.geometry import window_from_data


def data(sample_length: float, padding: float, rate: float) -> dict:
    """A checked ``data`` section with the given window."""
    return {
        "detectors": ["H1", "L1"],
        "sample_rate_in_hz": rate,
        "sample_length_in_s": sample_length,
        "padding_length_in_s": padding,
        "dtype": "float32",
    }


def test_production_window():
    window = window_from_data(data(12.0, 2.0, 2048.0))
    assert window.sample_length_in_samples == 24576
    assert window.padding_in_samples == 4096
    assert window.padded_length_in_s == 16.0
    assert window.padded_length_in_samples == 32768
    assert window.delta_f_in_hz == 0.0625
    assert window.n_frequency_bins == 16385


def test_exact_padded_length():
    assert 0.1 + 2 * 0.1 != 0.3
    assert window_from_data(data(0.1, 0.1, 10.0)).padded_length_in_s == 0.3


def test_partial_sample():
    with pytest.raises(ValueError, match="padding_length_in_s 0.05 s is 102.4 samples"):
        window_from_data(data(12.0, 0.05, 2048.0))
