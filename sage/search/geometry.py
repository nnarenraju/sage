#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : geometry.py
Description   : Sole owner of the search time/index conventions.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Every conversion between time and sample index in the search passes through here, and a
search of one observing run steps through of order a hundred million windows. At that
length an error of one part in a million per window is not a rounding detail, so window
positions are integers and the stride is an integer count of samples, never a float that
is accumulated.
"""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class SearchGeometry:
    """
    Window, stride and coalescence-time conventions for a search.

    All time-to-index conversion goes through this object. ``stride_samples`` is an
    integer so that analysed time is exact; 0.1 s at 2048 Hz is 204.8 samples, and
    the production value 205 gives a stride of 0.100097656250 s.

    Attributes
    ----------
    sample_rate : float
        Samples per second.
    signal_length_s : float
        Un-padded analysis content, ``data_cfg.sample_length_in_s``.
    padding_length_s : float
        One-sided whitening padding, ``data_cfg.padding_length_in_s``.
    stride_samples : int
        Window-start advance, in samples.
    tc_lower_s, tc_upper_s : float
        Coalescence-time prior bounds measured from the start of the signal content.
    """

    sample_rate: float
    signal_length_s: float
    padding_length_s: float
    stride_samples: int
    tc_lower_s: float
    tc_upper_s: float

    def __post_init__(self) -> None:
        """
        Reject a configuration that cannot describe a window lattice.

        Checked here rather than at first use, because every one of these produces
        plausible-looking numbers downstream instead of an error: a fractional stride
        puts windows off the sample lattice, a stride longer than the window leaves
        unanalysed gaps between consecutive windows, and a coalescence-time prior
        reaching outside the analysis content asks for a merger the network never sees.

        Also asserts the padding identity :attr:`peak_offset_s` relies on, so that
        relation is verified rather than assumed.
        """
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.signal_length_s <= 0:
            raise ValueError(
                f"signal_length_s must be positive, got {self.signal_length_s}"
            )
        if self.padding_length_s < 0:
            raise ValueError(
                f"padding_length_s must not be negative, got {self.padding_length_s}"
            )
        if isinstance(self.stride_samples, bool) or not isinstance(
            self.stride_samples, int
        ):
            raise TypeError(
                "stride_samples must be an integer number of samples, got "
                f"{self.stride_samples!r}; 0.1 s at 2048 Hz is 204.8 samples, which is "
                "why the stride is specified in samples and not in seconds"
            )
        if self.stride_samples <= 0:
            raise ValueError(
                f"stride_samples must be positive, got {self.stride_samples}"
            )

        total_s = self.signal_length_s + 2.0 * self.padding_length_s
        exact = total_s * self.sample_rate
        if abs(exact - round(exact)) > 1e-6:
            raise ValueError(
                f"window of {total_s} s at {self.sample_rate} Hz is {exact} samples, "
                "which is not a whole number"
            )
        if self.stride_samples > round(exact):
            raise ValueError(
                f"stride of {self.stride_samples} samples exceeds the window length of "
                f"{round(exact)} samples, which would leave unanalysed gaps"
            )

        if self.tc_lower_s >= self.tc_upper_s:
            raise ValueError(
                f"tc bounds must be increasing, got [{self.tc_lower_s}, "
                f"{self.tc_upper_s}]"
            )
        if self.tc_lower_s < 0.0 or self.tc_upper_s > self.signal_length_s:
            raise ValueError(
                f"tc bounds [{self.tc_lower_s}, {self.tc_upper_s}] lie outside the "
                f"{self.signal_length_s} s of analysis content"
            )

        # Two ways of placing the merger in the raw window must agree; they differ if
        # whiten_padding_s ever stops being the two-sided total.
        if not math.isclose(
            self.tc_mid_s + self.whiten_padding_s / 2.0,
            self.padding_length_s + self.tc_mid_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("whiten_padding_s is not twice padding_length_s")

    @property
    def whiten_padding_s(self) -> float:
        """Total (two-sided) whitening padding."""
        return 2.0 * self.padding_length_s

    @property
    def window_samples(self) -> int:
        """Padded window length in samples (16 s -> 32768 at 2048 Hz)."""
        return int(
            round((self.signal_length_s + self.whiten_padding_s) * self.sample_rate)
        )

    @property
    def window_s(self) -> float:
        """
        Padded window length in seconds.

        Derived from the integer sample count rather than from the sum of the two
        durations, so it agrees exactly with what is read from disk.
        """
        return self.window_samples / self.sample_rate

    @property
    def stride_s(self) -> float:
        """Exact stride in seconds, ``stride_samples / sample_rate``."""
        return self.stride_samples / self.sample_rate

    @property
    def tc_mid_s(self) -> float:
        """Midpoint of the coalescence-time prior."""
        return 0.5 * (self.tc_lower_s + self.tc_upper_s)

    @property
    def peak_offset_s(self) -> float:
        """
        Offset from a window's raw start to the expected merger time.

        Equals ``tc_mid_s + whiten_padding_s / 2``; ``__post_init__`` asserts this
        agrees with ``padding_length_s + tc_mid_s`` rather than relying on it.
        """
        return self.tc_mid_s + self.whiten_padding_s / 2.0

    def max_light_travel_s(self, detectors: Sequence[str]) -> float:
        """
        Longest light-travel time between any two of ``detectors``.

        A maximum over **every pair**, not over pairs involving a reference detector. For
        H1 and L1 the two are the same, 10.013 ms; add Virgo and the answer becomes the
        H1-V1 baseline at 27.288 ms, nearly three times larger. It sets the minimum slide
        lag, so taking the reference baseline instead would let a time slide sit inside
        the physical coincidence window and count genuine coincidences as background.

        Returns
        -------
        float
            Seconds. Zero for a single detector, which has no baseline.
        """
        names = tuple(detectors)
        if not names:
            raise ValueError("a detector network must name at least one detector")
        if len(set(names)) != len(names):
            raise ValueError(f"detectors repeated in network {names}")
        if len(names) == 1:
            return 0.0
        # Deferred: pycbc supplies the detector geometry and is an optional dependency,
        # so importing this module must not require it.
        from sage.core.detectors import pairwise_light_travel_times

        return float(pairwise_light_travel_times(names).max())

    def window_gps(self, window_start_gps: float) -> float:
        """Nominal trigger GPS time for a window beginning at ``window_start_gps``."""
        return window_start_gps + self.peak_offset_s
