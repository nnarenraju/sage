#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : ranges.py
Description   : Detector range, horizon distance and surveyed time-volume.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Range summarises instrument sensitivity over time and provides the horizontal axis for
presenting detections against surveyed volume rather than calendar time.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

#: Julian year, as every rate in the search is quoted per.
_SECONDS_PER_JULIAN_YEAR: float = 31557600.0


@dataclass
class RangeSeries:
    """Range as a function of time for one detector."""

    detector: str
    gps: np.ndarray
    range_mpc: np.ndarray

    def median(self) -> float:
        """Median range over the run."""
        raise NotImplementedError

    def duty_cycle(self) -> float:
        """Fraction of the run with usable data."""
        raise NotImplementedError


#: The factor between horizon distance and the sky- and orientation-averaged range, from
#: ``pycbc_plot_range``: "a factor of 2.26 smaller than the horizon distance". It is the
#: average of the antenna response over sky position and inclination, and a constant
#: rather than something to recompute per waveform.
HORIZON_TO_RANGE: float = 2.26


def inspiral_range_mpc(asd, m1: float = 1.4, m2: float = 1.4, snr_threshold: float = 8.0) -> float:
    """
    Sky- and orientation-averaged range for a fiducial binary.

    ``pycbc_plot_range``: the optimal signal-to-noise of a source at unit distance gives
    the horizon distance at the threshold, and the range is that divided by 2.26. Quoted
    at 1.4+1.4 by default because that is the canonical figure instruments are compared
    on, not because this search looks for such a binary.

    An instrument quantity, computed from a noise curve alone. It says nothing about
    whether a search recovers anything and is not a substitute for a measured sensitive
    distance -- which is why the two are separate figures.
    """
    return horizon_distance_mpc(asd, m1, m2, snr_threshold) / HORIZON_TO_RANGE


def horizon_distance_mpc(asd, m1: float, m2: float, snr_threshold: float = 8.0) -> float:
    """
    Optimally oriented and located distance at the threshold signal-to-noise ratio.

    ``sigma`` is the optimal signal-to-noise of a source placed at one megaparsec, so the
    distance at which that source rings up ``snr_threshold`` is ``sigma`` divided by it.
    Computed through :func:`pycbc.filter.sigma` on a waveform PyCBC generates, so the
    number is the one PyCBC would quote for the same noise curve.

    Parameters
    ----------
    asd : pycbc FrequencySeries
        Amplitude spectral density. Squared here, because ``sigma`` takes a *power*
        spectral density -- handing it an amplitude spectrum returns a distance too large
        by the square root of the spectrum, quietly.
    """
    import numpy as np
    from pycbc.filter import sigma
    from pycbc.types import FrequencySeries, zeros
    from pycbc.waveform import get_waveform_filter

    if not isinstance(asd, FrequencySeries):
        raise TypeError(
            f"expected a pycbc FrequencySeries carrying delta_f, got "
            f"{type(asd).__name__}; the range depends on the frequency spacing and a "
            "bare array does not carry it"
        )
    # Double precision throughout, where ``pycbc_plot_range`` works in single. Its PSDs
    # are stored in dynamic-range-scaled units and its waveform is generated at
    # ``distance = 1/DYN_RANGE_FAC`` to match; an unscaled strain PSD is of order 1e-46
    # and underflows float32 to zero, which turns the whole integrand into a division by
    # nothing. Same arithmetic, one more octave of exponent.
    power = np.asarray(asd, dtype=np.float64) ** 2
    # Bins the PSD marks unusable -- below the cutoff, and the Nyquist bin -- are zero.
    # Sent to infinity rather than left at zero so they weight nothing, which is what a
    # zero is meant to express here and is not what dividing by it does.
    power = np.where(power > 0.0, power, np.inf)
    psd = FrequencySeries(power, delta_f=asd.delta_f)
    flow = float(getattr(asd, "low_frequency_cutoff", None) or 15.0)

    out = zeros(len(psd), dtype=np.complex128)
    delta_t = 1.0 / ((len(psd) - 1) * 2 * psd.delta_f)
    htilde = get_waveform_filter(
        out,
        mass1=float(m1),
        mass2=float(m2),
        approximant="IMRPhenomD",
        f_lower=flow,
        delta_f=psd.delta_f,
        delta_t=delta_t,
        distance=1.0,
    ).astype(np.complex128)
    return float(sigma(htilde, psd=psd, low_frequency_cutoff=flow)) / float(
        snr_threshold
    )


def range_time_series(release_dir, detector: str, run: str, cadence_s: float = 600.0) -> RangeSeries:
    """Estimate range at a regular cadence across an observing run."""
    raise NotImplementedError


def surveyed_time_volume(
    ranges: Sequence[RangeSeries], coincident_intervals: Sequence[Tuple[float, float]]
) -> np.ndarray:
    """
    Cumulative surveyed time-volume.

    Uses the second most sensitive detector's range, so the measure reflects the
    coincident network rather than the best single instrument.
    """
    raise NotImplementedError


def sensitive_distance_mpc(vt: float, analysis_time_s: float) -> float:
    """
    Radius of the sphere whose volume-time equals a measured sensitive volume-time.

    A distance is easier to compare against a detector range than a volume-time is, and
    it is the quantity conventionally quoted for a search. Defined by
    ``VT = (4/3) pi D^3 T``, so ``D = (3 VT / (4 pi T))^(1/3)``.

    Parameters
    ----------
    vt : float
        Sensitive volume-time, in Mpc^3 yr.
    analysis_time_s : float
        Analysed time the volume-time was measured over.

    Returns
    -------
    float
        Sensitive distance in Mpc.

    Notes
    -----
    Lives here rather than in the figure layer because it is a physical quantity that the
    tables, the candidate store and the figures all quote. A copy inside a figure builder
    would put an analysis result somewhere that is meant to compute nothing.
    """
    import numpy as np

    years = float(analysis_time_s) / _SECONDS_PER_JULIAN_YEAR
    if years <= 0:
        raise ValueError(
            f"analysis_time_s must be positive, got {analysis_time_s}; a volume-time "
            "divided by no time is not a distance"
        )
    if vt < 0:
        raise ValueError(f"vt must not be negative, got {vt}")
    return float(np.cbrt(3.0 * float(vt) / (4.0 * np.pi * years)))
