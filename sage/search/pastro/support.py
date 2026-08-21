#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : support.py
Description   : The shared threshold, support and quadrature grid.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Every density in the mixture is truncated at the same threshold, renormalised over the
same region and evaluated on the same grid. Truncating one component but not another
makes the ratio above the untruncated region a property of the truncation rather than of
the data, and the resulting probability saturates for reasons unrelated to evidence.

The threshold is expressed in false-alarm-rate units so it means the same thing across
observing runs and across changes to the ranking statistic's scale.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CommonSupport:
    """The region and grid over which every component density is defined."""

    stat_lo: float
    stat_hi: float
    n_stat: int
    mchirp_lo: Optional[float] = None
    mchirp_hi: Optional[float] = None
    n_mchirp: Optional[int] = None
    threshold_far_per_day: float = 2.0
    threshold_stat: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a support no density can be defined on."""
        if not np.isfinite([self.stat_lo, self.stat_hi]).all():
            raise ValueError("the statistic bounds must be finite")
        if self.stat_hi <= self.stat_lo:
            raise ValueError(
                f"the support runs from {self.stat_lo} to {self.stat_hi}, which encloses "
                "nothing"
            )
        if self.n_stat < 2:
            raise ValueError(f"n_stat must be at least 2, got {self.n_stat}")
        mchirp_given = [
            value is not None
            for value in (self.mchirp_lo, self.mchirp_hi, self.n_mchirp)
        ]
        if any(mchirp_given) and not all(mchirp_given):
            raise ValueError(
                "a chirp-mass axis needs all of mchirp_lo, mchirp_hi and n_mchirp; a "
                "partial axis would silently drop the dimension for one component and "
                "keep it for another"
            )
        if all(mchirp_given):
            if self.mchirp_hi <= self.mchirp_lo:
                raise ValueError("the chirp-mass bounds enclose nothing")
            if self.n_mchirp < 2:
                raise ValueError(f"n_mchirp must be at least 2, got {self.n_mchirp}")

    @property
    def is_2d(self) -> bool:
        """Whether the densities resolve chirp mass as well as ranking statistic."""
        return self.mchirp_lo is not None

    def grid(self) -> Tuple[np.ndarray, ...]:
        """
        Quadrature nodes for each axis.

        Uniformly spaced and inclusive of both endpoints, so the weights below are the
        trapezoid rule and the same nodes serve as plotting abscissae. The grid is a
        property of the support rather than of either component, which is what makes the
        two densities comparable point for point.
        """
        nodes = (np.linspace(self.stat_lo, self.stat_hi, int(self.n_stat)),)
        if self.is_2d:
            nodes = nodes + (
                np.linspace(self.mchirp_lo, self.mchirp_hi, int(self.n_mchirp)),
            )
        return nodes

    def cell_volume(self) -> np.ndarray:
        """
        Quadrature weights matching :meth:`grid`.

        Trapezoid weights, so the end nodes carry half a cell. Using a full cell at the
        ends over-counts the edges of the support, and the noise density's edge is the
        analysis threshold -- exactly where the mass is, and exactly where an
        over-counted edge would inflate the noise model.
        """
        def weights(nodes: np.ndarray) -> np.ndarray:
            step = float(nodes[1] - nodes[0])
            out = np.full(nodes.size, step, dtype=np.float64)
            out[0] = out[-1] = 0.5 * step
            return out

        axes = [weights(nodes) for nodes in self.grid()]
        if len(axes) == 1:
            return axes[0]
        return np.outer(axes[0], axes[1])

    def contains(
        self, stat: np.ndarray, mchirp: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Mask of points inside the support.

        Inclusive at both edges. A trigger exactly at the analysis threshold was analysed,
        so excluding it would drop evidence the search collected.
        """
        stat = np.asarray(stat, dtype=np.float64)
        inside = (stat >= self.stat_lo) & (stat <= self.stat_hi)
        if self.is_2d:
            if mchirp is None:
                raise ValueError(
                    "this support resolves chirp mass, so a point needs one; a query "
                    "without it would be tested against half the support it lives in"
                )
            mchirp = np.asarray(mchirp, dtype=np.float64)
            inside &= (mchirp >= self.mchirp_lo) & (mchirp <= self.mchirp_hi)
        return inside


def build_support(
    far_curve,
    threshold_far_per_day: float = 2.0,
    stat_pad: float = 1.0,
    n_stat: int = 512,
    mchirp_bounds: Optional[Tuple[float, float]] = None,
    n_mchirp: Optional[int] = None,
    must_include: Optional[Sequence[float]] = None,
) -> CommonSupport:
    """
    Derive the shared support from the analysis threshold and observed range.

    The lower edge is the statistic at ``threshold_far_per_day``, which is what makes the
    threshold mean the same thing across observing runs and across changes to the ranking
    statistic's scale. The upper edge is padded past the loudest measured background so the
    signal density, which reaches further, is not truncated where it still has mass.

    ``stat_pad`` is in units of the statistic. Both densities are truncated and
    renormalised here and only here, so neither can be defined on a region the other is
    not -- the ratio of two densities with different supports is a property of the
    truncation rather than of the data, and it is the ratio that p_astro is built from.

    Parameters
    ----------
    must_include : sequence of float, optional
        Statistics the support has to contain: the recovered injections the signal density
        is built from, and the zero-lag candidates that will be scored. **Supplying these
        is not optional in practice.** The upper edge taken from ``far_curve`` alone is the
        loudest *background* event, and a candidate is confident precisely because it is
        louder than all background -- so a support bounded that way excludes every genuine
        detection, and the signal density is truncated where it still carries mass
        (measured at 4 to 19 per cent depending on the population). FGMC Eq. (9) asks for
        ``Fhat(inf) = 1``; this is how much of that is kept.

        The noise side costs nothing to extend, since the fitted tail is defined
        arbitrarily far out.
    """
    if stat_pad < 0:
        raise ValueError(f"stat_pad must not be negative, got {stat_pad}")
    if n_stat < 2:
        raise ValueError(f"n_stat must be at least 2, got {n_stat}")
    threshold = stat_at_far(far_curve, threshold_far_per_day)
    reach = [float(np.asarray(far_curve.stat, dtype=np.float64).max())]
    if must_include is not None:
        values = np.asarray(must_include, dtype=np.float64).ravel()
        if values.size:
            if not np.isfinite(values).all():
                raise ValueError(
                    "must_include holds a non-finite statistic; the support cannot be "
                    "stretched to contain a value that is not a number"
                )
            reach.append(float(values.max()))
    top = max(reach) + float(stat_pad)
    if top <= threshold:
        raise ValueError(
            f"the threshold at {threshold} lies above the loudest background event plus "
            f"padding at {top}; no support is left for either density"
        )
    lo = hi = n = None
    if mchirp_bounds is not None:
        lo, hi = (float(value) for value in mchirp_bounds)
        n = int(n_mchirp if n_mchirp is not None else n_stat)
    return CommonSupport(
        stat_lo=float(threshold),
        stat_hi=float(top),
        n_stat=int(n_stat),
        mchirp_lo=lo,
        mchirp_hi=hi,
        n_mchirp=n,
        threshold_far_per_day=float(threshold_far_per_day),
        threshold_stat=float(threshold),
    )


# Days in a Julian year, the unit FarCurve reports rates in.
DAYS_PER_JULIAN_YEAR: float = 365.25


def stat_at_far(far_curve, far_per_day: float) -> float:
    """
    Ranking statistic corresponding to a false-alarm rate.

    Inverts the measured curve, in ``log(FAR)`` against statistic, matching the
    interpolation :meth:`~sage.search.far.FarCurve.far_of` uses in the forward direction
    so that the two are consistent inversions of one another.

    The counted curve is used, never the fitted tail. A threshold is a statement about
    how much data was analysed, and taking it from an extrapolation would put the edge of
    both densities in a region where no background was ever counted.
    """
    if not np.isfinite(far_per_day) or far_per_day <= 0:
        raise ValueError(f"far_per_day must be finite and positive, got {far_per_day}")
    stat = np.asarray(far_curve.stat, dtype=np.float64)
    far_per_yr = np.asarray(far_curve.far_per_yr, dtype=np.float64)
    if stat.size == 0:
        raise ValueError("an empty FAR curve cannot be inverted")
    target = float(far_per_day) * DAYS_PER_JULIAN_YEAR
    # Refused rather than clamped. np.interp returns the endpoint for a query outside its
    # range, so an unreachable threshold silently comes back as the loudest background
    # statistic -- indistinguishable from a threshold that genuinely lands there, and the
    # support is then pinned above every event. The failure surfaces much later and
    # somewhere else: the tail blender reports its threshold "at or above the loudest
    # sample", naming the tail for a problem that is the background's length.
    lo, hi = float(np.min(far_per_yr)), float(np.max(far_per_yr))
    if not lo <= target <= hi:
        reachable_lo, reachable_hi = lo / DAYS_PER_JULIAN_YEAR, hi / DAYS_PER_JULIAN_YEAR
        raise ValueError(
            f"a false-alarm rate of {far_per_day:g}/day is outside what this background "
            f"measures: the counted curve spans {reachable_lo:.4g} to {reachable_hi:.4g} "
            f"per day over {far_curve.background_livetime_s:.6g} s of background. "
            + (
                "The threshold is quieter than the quietest rate the background can "
                "resolve, so it would have to be extrapolated; lengthen the background "
                "ladder or raise pastro.threshold_far_per_day"
                if target < lo
                else "The threshold is louder than the loudest rate measured, which "
                "would put the support below every counted event"
            )
        )
    # FAR falls as the statistic rises, so ascending in log(FAR) means descending in stat.
    order = np.argsort(np.log(far_per_yr))
    return float(
        np.interp(np.log(target), np.log(far_per_yr)[order], stat[order])
    )
