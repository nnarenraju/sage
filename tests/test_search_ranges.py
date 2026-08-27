#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_ranges.py
Description   : Detector range and sensitive distance.

Created on 2026-08-22

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Range is an instrument quantity and sensitive distance is a search result. They are
quoted in the same units and are routinely confused, so what is checked here is that each
is what it claims: the range against PyCBC's own recipe and against the published design
figure, the sensitive distance against the definition it is derived from.
"""

import numpy as np
import pytest

pycbc = pytest.importorskip("pycbc")

from sage.search.sensitivity.ranges import (  # noqa: E402
    HORIZON_TO_RANGE,
    horizon_distance_mpc,
    inspiral_range_mpc,
    sensitive_distance_mpc,
)

SECONDS_PER_YEAR = 31557600.0


@pytest.fixture(scope="module")
def design_asd():
    """The aLIGO zero-detuning high-power design curve, as an amplitude spectrum."""
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.types import FrequencySeries

    flow, delta_f, length = 15.0, 0.25, 8193
    psd = aLIGOZeroDetHighPower(length, delta_f, flow)
    asd = FrequencySeries(np.sqrt(np.asarray(psd)), delta_f=delta_f)
    asd.low_frequency_cutoff = flow
    return asd


class TestRange:
    """``pycbc_plot_range``: sigma at one megaparsec, over eight, over 2.26."""

    def test_design_curve_gives_the_published_figure(self, design_asd):
        """
        The check that says the whole chain is right rather than merely self-consistent.
        Advanced LIGO's design binary-neutron-star range is quoted at about 180 Mpc, and
        a recipe that had the distance scaling, the antenna average or the noise curve
        wrong would not land there.
        """
        assert inspiral_range_mpc(design_asd) == pytest.approx(190.0, rel=0.10)

    def test_range_is_horizon_over_the_antenna_factor(self, design_asd):
        """
        2.26 is the sky- and orientation-average, from ``pycbc_plot_range``'s own caption.
        A constant, not something recomputed per waveform.
        """
        horizon = horizon_distance_mpc(design_asd, 1.4, 1.4)

        assert inspiral_range_mpc(design_asd) == pytest.approx(
            horizon / HORIZON_TO_RANGE
        )

    def test_threshold_scales_the_distance(self, design_asd):
        """
        The horizon is where a source rings up the threshold, so halving the threshold
        doubles it. A recipe that dropped the threshold would pass every other check here.
        """
        at_eight = horizon_distance_mpc(design_asd, 1.4, 1.4, snr_threshold=8.0)
        at_four = horizon_distance_mpc(design_asd, 1.4, 1.4, snr_threshold=4.0)

        assert at_four == pytest.approx(2.0 * at_eight, rel=1e-9)

    def test_heavier_binary_reaches_further(self, design_asd):
        """A louder source is seen further; the ordering is the sanity of the waveform."""
        assert horizon_distance_mpc(design_asd, 30.0, 30.0) > horizon_distance_mpc(
            design_asd, 1.4, 1.4
        )

    def test_bare_array_refused(self, design_asd):
        """
        The range depends on the frequency spacing and an array does not carry it.
        Accepting one would compute a range against whatever spacing was assumed.
        """
        with pytest.raises(TypeError, match="delta_f"):
            inspiral_range_mpc(np.asarray(design_asd))


class TestSensitiveDistance:
    """The radius of the sphere whose volume-time equals a measured one."""

    def test_round_trips_through_the_definition(self):
        """``VT = (4/3) pi D^3 T`` in one direction has to come back in the other."""
        distance, years = 250.0, 1.0
        volume_time = (4.0 / 3.0) * np.pi * distance**3 * years

        assert sensitive_distance_mpc(
            volume_time, years * SECONDS_PER_YEAR
        ) == pytest.approx(distance)

    def test_longer_analysis_is_not_greater_sensitivity(self):
        """
        The distinction the quantity exists to draw. Twice the volume-time from twice the
        observing is the same reach, and a definition that divided by the wrong thing
        would report a search as improving simply by running longer.
        """
        distance = 100.0
        one_year = (4.0 / 3.0) * np.pi * distance**3

        assert sensitive_distance_mpc(one_year, SECONDS_PER_YEAR) == pytest.approx(
            sensitive_distance_mpc(2.0 * one_year, 2.0 * SECONDS_PER_YEAR)
        )

    def test_refusals(self):
        """A volume-time divided by no time is not a distance."""
        with pytest.raises(ValueError, match="analysis_time_s"):
            sensitive_distance_mpc(1.0, 0.0)
        with pytest.raises(ValueError, match="vt"):
            sensitive_distance_mpc(-1.0, SECONDS_PER_YEAR)
