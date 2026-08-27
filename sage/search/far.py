#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : far.py
Description   : FAR and IFAR with conservative counting, plus the cumulative-vs-IFAR curve.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

FAR uses the conservative ``(1 + n_b) / T_b`` counting. ``T_b`` is always the summed
per-slide livetime from the slide plan; there is no closed form for it. Beyond the
measured background the curve holds flat at the counting floor, and the
region is reported with its uncertainty band rather than silently.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from sage.search.background import (
    SECONDS_PER_JULIAN_YEAR,
    BackgroundSet,
    far_of_stat,
    n_louder,
)
from sage.search.fingerprint import combine, digest_h5

__all__ = [
    "SECONDS_PER_JULIAN_YEAR",
    "FarCurve",
    "far_of_stat",
    "n_louder",
    "build_far_curve",
    "expected_count",
    "poisson_band",
    "cumulative_vs_ifar",
    "p_value_from_ifar",
]



@dataclass
class FarCurve:
    """The statistic-to-FAR mapping for one observing run and removal mode."""

    stat: np.ndarray
    far_per_yr: np.ndarray
    n_louder: np.ndarray
    background_livetime_s: float
    foreground_livetime_s: float
    removal: str
    ifar_cap_yr: float = 1000.0

    def __post_init__(self) -> None:
        """Refuse a curve that is not a usable statistic-to-rate mapping."""
        self.stat = np.asarray(self.stat, dtype=np.float64).ravel()
        self.far_per_yr = np.asarray(self.far_per_yr, dtype=np.float64).ravel()
        self.n_louder = np.asarray(self.n_louder).ravel()
        if not (self.stat.size == self.far_per_yr.size == self.n_louder.size):
            raise ValueError(
                f"curve arrays disagree: {self.stat.size} statistics, "
                f"{self.far_per_yr.size} rates, {self.n_louder.size} counts"
            )
        if self.stat.size and np.any(np.diff(self.stat) <= 0):
            raise ValueError("curve statistics must be strictly ascending")
        if self.removal not in ("inclusive", "exclusive", "hierarchical"):
            raise ValueError(f"unknown removal mode {self.removal!r}")

    def far_of(self, stat: np.ndarray) -> np.ndarray:
        """
        The counted FAR: interpolate the measured rate at arbitrary statistic values.

        This is the false-alarm rate of record. It is counting and nothing else --
        ``(1 + n_b) / T_b`` read off the background that was actually accumulated -- so
        every value it returns is supported by background events somebody can point at.

        Interpolated in ``log(FAR)``, because the rate spans many decades over the curve
        and a linear interpolant between two decades sits an order of magnitude high in
        between -- always in the direction that makes a candidate look less significant
        than the measured background says.

        Below the curve the first rate is held; there is no more background to count.
        Above the loudest background event the last measured rate is held, flat. That
        floor is ``(1 + 1) / T_b`` -- the counting is inclusive, so the loudest background
        event counts itself. The background ran out, and every candidate above it is
        reported at the same rate because the counting cannot tell them apart. Separating
        There is no extrapolated counterpart: continuing the curve past the count with a
        fitted tail is what this package used to do, and see SB-64 for what it cost.
        """
        stat = np.asarray(stat, dtype=np.float64)
        if self.stat.size == 0:
            raise ValueError("an empty curve cannot be interpolated")
        return np.exp(
            np.interp(
                stat,
                self.stat,
                np.log(self.far_per_yr),
                left=float(np.log(self.far_per_yr[0])),
                right=float(np.log(self.far_per_yr[-1])),
            )
        )


    def ifar_of(self, stat: np.ndarray) -> np.ndarray:
        """
        Inverse FAR in years, capped and flagged where extrapolated.

        The cap exists because an IFAR far beyond the background that measured it is a
        statement about how long the background ran, not about the candidate. Reporting
        ``1e6`` years from a 23-year background would be quoting the extrapolation as a
        measurement.
        """
        far = self.far_of(stat)
        with np.errstate(divide="ignore"):
            ifar = np.where(far > 0, 1.0 / far, np.inf)
        return np.minimum(ifar, self.ifar_cap_yr)


    def is_extrapolated(self, stat: np.ndarray) -> np.ndarray:
        """
        Whether a statistic lies beyond the measured background.

        Every candidate above the loudest background event carries this, so a table can
        never present an extrapolated rate as a measured one.
        """
        stat = np.asarray(stat, dtype=np.float64)
        if self.stat.size == 0:
            return np.ones(stat.shape, dtype=bool)
        return stat > self.stat[-1]

    def save(self, path: str | Path) -> None:
        """
        Write ``far/far_curve_<run>_<removal>.h5``.

        Both livetimes travel with the curve. A rate is a count over a time, and the two
        halves must describe the same time: a curve read back beside a livetime taken from
        somewhere else is the exact error the exclusive and hierarchical modes exist to
        avoid, and it is invisible afterwards because both numbers look ordinary.

        A curve is the count and the exposure it was counted over, so a stored curve
        without it would silently lose the ability to separate "1 in 2 yr" from "1 in
        100 yr" -- and would raise at the point of use, one stage later.

        Written under ``atomic_h5``, so a kill mid-write leaves the previous curve rather
        than a truncated file that reads as a shorter one.
        """
        from sage.utils.atomic_io import atomic_h5

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_h5(target, mode="w") as handle:
            handle.attrs["removal"] = str(self.removal)
            handle.attrs["background_livetime_s"] = float(self.background_livetime_s)
            handle.attrs["foreground_livetime_s"] = float(self.foreground_livetime_s)
            handle.attrs["ifar_cap_yr"] = float(self.ifar_cap_yr)
            handle.create_dataset("stat", data=np.asarray(self.stat, dtype=np.float64))
            handle.create_dataset(
                "far_per_yr", data=np.asarray(self.far_per_yr, dtype=np.float64)
            )
            handle.create_dataset(
                "n_louder", data=np.asarray(self.n_louder, dtype=np.int64)
            )

    @classmethod
    def load(cls, path: str | Path) -> "FarCurve":
        """
        Read a persisted FAR curve.

        The curve is the counted rate and nothing else, so what comes back is what was
        counted. Older files may carry a ``tail`` group from the generalised-Pareto fit
        this package no longer makes; it is ignored rather than read, since nothing can
        consume it any more.
        """
        import h5py

        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"no FAR curve at {target}")
        with h5py.File(target, "r") as handle:
            return cls(
                stat=np.asarray(handle["stat"], dtype=np.float64),
                far_per_yr=np.asarray(handle["far_per_yr"], dtype=np.float64),
                n_louder=np.asarray(handle["n_louder"], dtype=np.int64),
                background_livetime_s=float(handle.attrs["background_livetime_s"]),
                foreground_livetime_s=float(handle.attrs["foreground_livetime_s"]),
                removal=str(handle.attrs["removal"]),
                ifar_cap_yr=float(handle.attrs["ifar_cap_yr"]),
            )


# Re-exported, not defined here. The counting convention is a property of a background --
# how many of its events lie at or above a statistic, over how much time -- and
# ``hierarchical_removal`` in the background stage needs it while running, one stage
# before this one. Defining it here and importing it back would make the background stage
# depend on a module scheduled after it, which is the contradiction
# ``test_no_stage_module_imports_a_later_stage`` exists to catch. Both names stay in this
# module's namespace so ``far.far_of_stat`` keeps working for every existing caller.


def build_far_curve(
    background: BackgroundSet,
    foreground_livetime_s: float,
    ifar_cap_yr: float = 1000.0,
) -> FarCurve:
    """
    Assemble the FAR curve, capping IFAR relative to the measured background.

    One point per distinct background statistic: the rate only changes where a background
    event sits, so any denser grid would be interpolation presented as measurement.

    The background must be clustered. An unclustered trigger train counts one event per
    window of a glitch instead of one per glitch -- several times too many -- and since
    the count is the FAR numerator, every rate taken from it is wrong by that factor
    while looking entirely ordinary. This is the failure that invalidated the reference
    analysis, so it is refused here rather than checked by convention.
    """
    if background.histogram is not None and not background.histogram.clustered:
        raise ValueError(
            "refusing to build a FAR curve from an unclustered background: the count "
            "would be one event per window of each glitch rather than one per glitch, "
            "and it is the numerator of every rate"
        )
    if not bool(getattr(background, "clustered", True)):
        raise ValueError("refusing to build a FAR curve from an unclustered background")

    stats = np.unique(np.asarray(background.stats, dtype=np.float64).ravel())
    if stats.size == 0:
        raise ValueError(
            "the background holds no events, so no rate can be measured from it"
        )
    counts = n_louder(stats, background.stats)
    far_per_s = (1.0 + counts) / float(background.livetime_s)
    return FarCurve(
        stat=stats,
        far_per_yr=far_per_s * SECONDS_PER_JULIAN_YEAR,
        n_louder=counts,
        background_livetime_s=float(background.livetime_s),
        foreground_livetime_s=float(foreground_livetime_s),
        removal=background.removal,
        ifar_cap_yr=float(ifar_cap_yr),
    )


def expected_count(ifar_yr: np.ndarray, observation_time_s: float) -> np.ndarray:
    """
    Background events expected at or above each IFAR: ``T / IFAR``.

    Follows from the analysed time alone -- that is what an inverse false-alarm *rate*
    means -- so it is a prediction the measured foreground is compared against, never
    something fitted to it.
    """
    ifar_yr = np.asarray(ifar_yr, dtype=np.float64)
    if np.any(ifar_yr <= 0):
        raise ValueError("IFAR must be positive")
    return (float(observation_time_s) / SECONDS_PER_JULIAN_YEAR) / ifar_yr


def poisson_band(expected: np.ndarray, sigma: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Central Poisson interval about ``expected`` at a Gaussian-equivalent ``sigma``.

    Quantiles of the Poisson distribution, not ``expected +/- sigma * sqrt(expected)``.
    In the tail the expectation is far below one, where the Gaussian approximation gives
    a negative lower edge and a band that does not contain the integers the count can
    actually take -- and the tail is the whole point of the plot.
    """
    from scipy import stats as _stats

    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    expected = np.asarray(expected, dtype=np.float64)
    # The two-sided Gaussian-equivalent coverage, e.g. 0.6827 at one sigma.
    coverage = float(_stats.norm.cdf(sigma) - _stats.norm.cdf(-sigma))
    tail = 0.5 * (1.0 - coverage)
    lower = _stats.poisson.ppf(tail, expected)
    upper = _stats.poisson.ppf(1.0 - tail, expected)
    return np.maximum(lower, 0.0), upper


def cumulative_vs_ifar(
    foreground_stats: np.ndarray,
    curve: FarCurve,
    sigma_levels: Sequence[int] = (1, 2, 3),
) -> dict:
    """
    Cumulative count of candidates at or above each IFAR, with Poisson bands.

    The expected background curve is ``T_analysis / IFAR``; the bands are Poisson
    quantiles about it.

    Returns
    -------
    dict
        ``ifar_yr`` (descending), ``observed`` cumulative counts, ``expected``, and
        ``band_<n>`` giving ``(lower, upper)`` for each requested sigma.
    """
    foreground_stats = np.asarray(foreground_stats, dtype=np.float64).ravel()
    ifar = curve.ifar_of(foreground_stats)
    order = np.argsort(ifar)[::-1]
    ifar_sorted = ifar[order]
    observed = np.arange(1, ifar_sorted.size + 1, dtype=np.int64)

    out = {
        "ifar_yr": ifar_sorted,
        "observed": observed,
        "expected": expected_count(ifar_sorted, curve.foreground_livetime_s),
        "foreground_livetime_s": float(curve.foreground_livetime_s),
    }
    for sigma in sigma_levels:
        out[f"band_{sigma}"] = poisson_band(out["expected"], sigma)
    return out


def p_value_from_ifar(ifar_yr: np.ndarray, observation_time_s: float) -> np.ndarray:
    """
    ``1 - exp(-T / IFAR)`` for a single trial.

    The probability that a Poisson background of rate ``1 / IFAR`` produces at least one
    event this loud somewhere in the observation. It is a single-trial p-value: the trials
    factor over the arms of the campaign is applied separately, in ``trials.py``, and is
    deliberately not folded in here so both views stay available.

    ``expm1`` rather than ``1 - exp``: for a confident candidate the exponent is tiny and
    ``1 - exp(-x)`` cancels to zero in float64 around ``x ~ 1e-17``, which would report a
    p-value of exactly zero for candidates whose significance is finite and worth stating.
    """
    ifar_yr = np.asarray(ifar_yr, dtype=np.float64)
    if observation_time_s < 0:
        raise ValueError(
            f"observation time must not be negative, got {observation_time_s}"
        )
    if np.any(ifar_yr <= 0):
        raise ValueError(
            "IFAR must be positive; a non-positive inverse rate has no p-value"
        )
    years = float(observation_time_s) / SECONDS_PER_JULIAN_YEAR
    return -np.expm1(-years / ifar_yr)


def _time_binned_counts(background, bin_width_s: float):
    """
    Background events per equal-width time bin, for the dispersion test.

    Returns ``(counts, detail)`` or ``None`` when the set carries no event times.

    Bins are equal *width*; whether they are equal *exposure* is the caller's problem and
    is reported rather than assumed. Empty bins at the ends of a run of analysed time are
    real -- a quiet stretch is data -- but a bin lying entirely inside a gap between
    segments is not, and counting it as an observed zero makes any process look
    over-dispersed. Leading and trailing empties are therefore trimmed, and the fraction
    of the background livetime the bins actually span is reported beside the result:
    within a slid background the per-slide coverage still varies across a bin, so this is
    a diagnostic to read rather than a corrected number.

    ``bin_width_s`` follows sgwc-1's 10 s. At the 0.1 s stride that is 100 windows per bin
    before clustering, which is wide enough that a bin's count is not dominated by a
    single cluster and narrow enough to resolve a glitch-active hour.
    """
    times = getattr(background, "gps", None)
    if times is None:
        return None
    times = np.asarray(times, dtype=np.float64).ravel()
    if times.size < 2 or not np.isfinite(times).all():
        return None
    if bin_width_s <= 0:
        raise ValueError(f"dispersion_bin_s must be positive, got {bin_width_s}")

    edges = np.arange(times.min(), times.max() + bin_width_s, bin_width_s)
    if edges.size < 3:
        return None
    counts, _ = np.histogram(times, bins=edges)
    occupied = np.flatnonzero(counts)
    counts = counts[occupied[0] : occupied[-1] + 1]
    if counts.size < 2:
        return None
    spanned = counts.size * bin_width_s
    livetime = float(getattr(background, "livetime_s", 0.0) or 0.0)
    return counts.astype(np.int64), {
        "bin_width_s": float(bin_width_s),
        "n_bins": int(counts.size),
        "binned_by": "event time",
        "spanned_s": float(spanned),
        # Below one where the ladder's analysed time is not contiguous in the reference
        # frame; the bins cover only what lies between the first and last background
        # event. Reported so the dispersion is read against how much of the run it saw.
        "livetime_fraction_spanned": float(spanned / livetime) if livetime else float("nan"),
        "exposure_corrected": False,
    }


def run(spec, **kwargs) -> dict:
    """
    Stage driver: build a counted FAR curve per removal mode.

    The count is the rate, and the only rate. Nothing is fitted and nothing is continued
    past the loudest background event, which is sgwc-1's construction -- it reads its
    reporting thresholds straight off the background (``search.ipynb`` cell 297: 1/month
    at 11.640625 for HL) and fits no tail anywhere.

    Over-dispersion relative to Poisson is measured and recorded beside the curve rather
    than acted on: simple order-statistic counting assumes independence, and a reader needs
    to know whether the background supports that. A driver that silently switched behaviour
    on a test outcome would make the reported curve depend on a result nobody saw.
    """
    from sage.search.background import BackgroundSet, overdispersion_lrt

    foreground_s = kwargs.pop("foreground_livetime_s", None)
    if foreground_s is None:
        from sage.search.slides import SlidePlan

        foreground_s = SlidePlan.load(
            spec.path("slides", "slide_plan.h5")
        ).foreground_livetime_s

    curves = {}
    checks = {}
    for mode in spec.significance.removal_modes:
        source = spec.path("background", f"bg_{mode}.h5")
        if not Path(source).is_file():
            raise FileNotFoundError(
                f"no {mode} background at {source}; the background stage builds every "
                "mode this campaign asked for, so a missing one means it did not finish"
            )
        background = BackgroundSet.load(source)
        stats = np.asarray(background.stats, dtype=np.float64)

        # Each removal mode takes zero-lag exposure away as well as background events:
        # a veto that removes a loud event removes the time around it from both sides of
        # the analysis. The mode's own reduced exposure is what its curve must be read
        # against -- using the inclusive one would quote expected counts and p-values
        # over time this mode had already vetoed, and always in the optimistic direction.
        mode_foreground_s = (
            float(background.foreground_livetime_s)
            if background.foreground_livetime_s is not None
            else float(foreground_s)
        )

        detail = {
            "n_background": int(stats.size),
            "foreground_livetime_s": mode_foreground_s,
            "foreground_reduced": background.foreground_livetime_s is not None,
        }

        curve = build_far_curve(
            background,
            mode_foreground_s,
            ifar_cap_yr=float(spec.significance.ifar_cap_yr),
        )
        target = spec.path(
            "far", f"far_curve_{spec.data.observing_run}_{mode}.h5"
        )
        curve.save(target)
        curves[mode] = str(target)
        checks[mode] = detail

    # Poisson vs negative binomial on the inclusive set, which is what sgwc-1 tests
    # (search.ipynb cell 331: bin_triggers(trigger_times, bin_width=10.0) into
    # likelihood_ratio_test). An over-dispersed background invalidates simple
    # order-statistic counting, so it is reported beside the curve rather than assumed
    # away.
    #
    # Counts per unit *time*, not per ranking-statistic bin. Feeding the statistic
    # histogram here was measuring the shape of the statistic distribution and nothing
    # else: it reported a background that is Poisson by construction as over-dispersed at
    # p = 0, and gave bit-identical output for arrival times uniform in time and for the
    # same events packed into 200 one-second bursts -- the two cases the test exists to
    # separate.
    inclusive = BackgroundSet.load(spec.path("background", "bg_inclusive.h5"))
    binned = _time_binned_counts(inclusive, float(spec.significance.dispersion_bin_s))
    if binned is not None:
        counts, detail = binned
        checks["overdispersion"] = {**overdispersion_lrt(counts), **detail}
    else:
        checks["overdispersion"] = {
            "measured": False,
            "reason": (
                "the background carries no event times, so counts per unit time cannot "
                "be formed; the statistic histogram is not a substitute"
            ),
        }

    return {
        "curves": curves,
        "checks": checks,
        "foreground_livetime_s": float(foreground_s),
        # Digest the curves that were written, not the tail shapes that went into them.
        # A FAR curve is counts divided by a livetime and capped at an IFAR ceiling; none
        # of those three reach a fitted shape parameter, so a curve rebuilt against a
        # different background livetime -- every rate in it wrong by that ratio -- carried
        # a byte-identical fingerprint and cascaded nothing.
        "fingerprint": combine(
            *(
                f"{mode}={checks[mode].get('tail_shape', 'none')}"
                for mode in sorted(curves)
            ),
            digest_h5([curves[mode] for mode in sorted(curves)]),
        ),
    }
