"""Tests for sage.core.checks: start-up checks across settings."""

import pytest

from sage.core import checks


def test_whole_samples_nearest():
    with pytest.raises(ValueError, match=r"102.4 samples .* nearest whole length is 0.0498046875 s"):
        checks.check_whole_samples("padding_length_in_s", 0.05, 2048.0)


def test_whole_samples_exact_decimal():
    assert 4.35 * 100.0 != 435.0
    checks.check_whole_samples("sample_length_in_s", 4.35, 100.0)


def test_on_grid():
    checks.check_on_grid("low_frequency_cutoff_in_hz", 20.0625, 16.0)
    with pytest.raises(ValueError, match=r"20.03 Hz is not on the 1/16 Hz grid; nearest is 20.0"):
        checks.check_on_grid("low_frequency_cutoff_in_hz", 20.03, 16.0)


def test_on_grid_inexact_spacing():
    assert (checks.exact(20.0) / checks.exact(1.0 / 15.0)).denominator != 1
    checks.check_on_grid("low_frequency_cutoff_in_hz", 20.0, 15.0)


def test_power_of_two():
    checks.check_power_of_two("lowest_sample_rate_in_hz", 64.0)
    with pytest.raises(ValueError, match="100.0 is not a power of two; nearest is 128"):
        checks.check_power_of_two("lowest_sample_rate_in_hz", 100.0)
    with pytest.raises(ValueError, match="64.5 is not a power of two; nearest is 64"):
        checks.check_power_of_two("lowest_sample_rate_in_hz", 64.5)


def test_unique_detectors():
    with pytest.raises(ValueError, match="more than once: H1"):
        checks.check_unique_detectors(["H1", "L1", "H1"])


def test_tc_window():
    checks.check_tc_window(0.0, 12.0, 12.0)
    with pytest.raises(ValueError, match=r"tc prior \[11.0, 12.5\] s is outside"):
        checks.check_tc_window(11.0, 12.5, 12.0)
