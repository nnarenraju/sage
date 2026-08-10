#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_geometry.py
Description   : Time and index conventions: exactness, and the light-travel maximum.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Every time-to-index conversion in the search goes through this object, and a search of
O3a steps through 92.1 million windows. An error of one part in a million per window is
not a rounding detail at that length; it is a systematic drift in every reported time.
So the arithmetic is asserted to be exact in integers, not merely close in floats.

The light-travel maximum is the other thing worth pinning. It sets the smallest admissible
time-slide lag, and it must be a maximum over every detector pair rather than over pairs
involving whichever detector happens to be the reference.

Runs anywhere; needs no data and no GPU. The light-travel tests need pycbc, which supplies
the detector geometry.
"""

import pytest

from sage.search.geometry import SearchGeometry

# The production configuration, read from the checkpoint: 12 s of content with 2 s of
# whitening padding each side at 2048 Hz, and a 205-sample stride.
PRODUCTION = dict(
    sample_rate=2048.0,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=205,
    tc_lower_s=5.0,
    tc_upper_s=7.0,
)

# Exact light-travel times between the LIGO and Virgo sites, in seconds.
H1L1 = 0.010012846152267725
H1V1 = 0.027287979933397113
L1V1 = 0.02644834101635671


@pytest.fixture
def geometry():
    return SearchGeometry(**PRODUCTION)


class TestWindowArithmetic:
    """The window and stride resolve to exact integers."""

    def test_window_samples(self, geometry):
        """12 s of content plus 2 x 2 s of padding is 32768 samples at 2048 Hz."""
        assert geometry.window_samples == 32768
        assert isinstance(geometry.window_samples, int)

    def test_window_seconds(self, geometry):
        """The padded window is exactly 16 s."""
        assert geometry.window_s == 16.0

    def test_whiten_padding_is_two_sided(self, geometry):
        """The padding attribute is one-sided; the property is the total."""
        assert geometry.whiten_padding_s == 4.0

    def test_stride_is_exactly_representable(self, geometry):
        """
        205 / 2048 is exact in binary floating point, and must round-trip.

        The sample rate is a power of two, so the stride has an exact representation.
        Asserting the round trip catches a change to a rate that does not.
        """
        assert geometry.stride_s == 0.10009765625
        assert geometry.stride_s * geometry.sample_rate == geometry.stride_samples

    def test_stride_does_not_accumulate_error(self, geometry):
        """
        Advancing 92.1 million windows lands on an exact integer sample.

        This is the O3a window count. Accumulating a float stride instead of multiplying
        an integer one would drift, and the drift would appear as a systematic offset in
        every reported coalescence time.
        """
        n = 92_100_000
        assert n * geometry.stride_samples == 18_880_500_000
        naive = 0.0
        for _ in range(1000):
            naive += geometry.stride_s
        assert naive == pytest.approx(1000 * geometry.stride_s, abs=0.0)


class TestCoalescenceTime:
    """Where in a window the merger is expected."""

    def test_tc_midpoint(self, geometry):
        """The prior midpoint is the mean of its bounds."""
        assert geometry.tc_mid_s == 6.0

    def test_peak_offset_from_raw_window_start(self, geometry):
        """
        The merger sits at the padding plus the prior midpoint from the raw start.

        Both ways of writing it agree, which is what the constructor asserts rather than
        assumes.
        """
        assert geometry.peak_offset_s == pytest.approx(8.0, abs=1e-12)
        assert geometry.peak_offset_s == pytest.approx(
            geometry.tc_mid_s + geometry.whiten_padding_s / 2, abs=1e-12
        )

    def test_window_gps(self, geometry):
        """A window's nominal trigger time is its start plus the peak offset."""
        start = 1238166018.0
        assert geometry.window_gps(start) == pytest.approx(
            start + geometry.peak_offset_s, abs=1e-9
        )


class TestValidation:
    """Configurations that cannot be right are refused at construction."""

    def test_non_integer_stride_is_refused(self):
        """A fractional stride would put windows off the sample lattice."""
        with pytest.raises((TypeError, ValueError)):
            SearchGeometry(**{**PRODUCTION, "stride_samples": 204.8})

    def test_non_positive_stride_is_refused(self):
        """A zero or negative stride does not advance."""
        for bad in (0, -1):
            with pytest.raises(ValueError):
                SearchGeometry(**{**PRODUCTION, "stride_samples": bad})

    def test_stride_longer_than_window_is_refused(self):
        """A stride past the window length would leave unanalysed gaps."""
        with pytest.raises(ValueError):
            SearchGeometry(**{**PRODUCTION, "stride_samples": 40000})

    def test_window_must_be_a_whole_number_of_samples(self):
        """A window length that is not an integer sample count is refused."""
        with pytest.raises(ValueError):
            SearchGeometry(**{**PRODUCTION, "signal_length_s": 12.0001})

    def test_inverted_tc_bounds_are_refused(self):
        """The prior's lower bound must be below its upper."""
        with pytest.raises(ValueError):
            SearchGeometry(**{**PRODUCTION, "tc_lower_s": 7.0, "tc_upper_s": 5.0})

    def test_tc_bounds_must_lie_inside_the_content(self):
        """A coalescence time outside the analysis content cannot be recovered."""
        with pytest.raises(ValueError):
            SearchGeometry(**{**PRODUCTION, "tc_upper_s": 13.0})

    def test_non_positive_sample_rate_is_refused(self):
        with pytest.raises(ValueError):
            SearchGeometry(**{**PRODUCTION, "sample_rate": 0.0})


class TestLightTravel:
    """The maximum is over every pair, which is what a third detector changes."""

    def test_two_detector_baseline(self, geometry):
        """H1 to L1."""
        assert geometry.max_light_travel_s(("H1", "L1")) == pytest.approx(H1L1, abs=1e-12)

    def test_three_detector_maximum_is_the_longest_baseline(self, geometry):
        """
        Adding Virgo raises the maximum to the H1-V1 baseline, not the H1-L1 one.

        This is the value that sets the minimum slide lag. Taking the reference
        detector's baseline instead would understate it by a factor of 2.7 and let a
        slide sit inside the physical coincidence window.
        """
        got = geometry.max_light_travel_s(("H1", "L1", "V1"))
        assert got == pytest.approx(H1V1, abs=1e-12)
        assert got > geometry.max_light_travel_s(("H1", "L1"))

    def test_maximum_ignores_which_detector_is_listed_first(self, geometry):
        """The answer is a property of the network, not of the ordering."""
        for order in (("V1", "L1", "H1"), ("L1", "V1", "H1"), ("H1", "V1", "L1")):
            assert geometry.max_light_travel_s(order) == pytest.approx(H1V1, abs=1e-12)

    def test_pair_not_involving_the_first_detector(self, geometry):
        """L1 to V1 is found even though neither is listed first elsewhere."""
        assert geometry.max_light_travel_s(("L1", "V1")) == pytest.approx(L1V1, abs=1e-12)

    def test_single_detector_has_no_baseline(self, geometry):
        """One detector has no pair, so the maximum is zero."""
        assert geometry.max_light_travel_s(("H1",)) == 0.0

    def test_empty_network_is_refused(self, geometry):
        """A network with no detectors is a configuration error."""
        with pytest.raises(ValueError):
            geometry.max_light_travel_s(())

    def test_repeated_detector_is_refused(self, geometry):
        """A detector named twice indicates a configuration mistake."""
        with pytest.raises(ValueError):
            geometry.max_light_travel_s(("H1", "H1"))
