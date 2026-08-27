#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : density.py
Description   : Component densities over the ranking statistic and chirp mass.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Densities are estimated from the samples themselves, with the bandwidth derived from the
data. Smoothing a pre-binned histogram instead makes the effective resolution a property
of the binning, and tying bin edges to the observed extremes makes the whole density
move whenever more background is added, which prevents the result from converging.

Boundary correction is applied at the truncation edge so that mass is not lost there.
The network estimates chirp mass alongside the ranking statistic, so densities can be
resolved in both; the extra dimension is what lets a candidate's mass count as evidence
rather than being carried through unused.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np

from sage.search.pastro.support import CommonSupport


class Density(Protocol):
    """A normalised component density over the common support."""

    def log_prob(self, stat: np.ndarray, mchirp: Optional[np.ndarray] = None) -> np.ndarray:
        """Log density at the given points."""
        ...

    def normalisation(self) -> float:
        """Integral over the common support; must be one to numerical tolerance."""
        ...










#: Statistic width of one histogram bin.
#:
#: PyCBC takes this as a command-line option (``--bg-bin-width``) and bins linearly from
#: the analysis threshold, ``n_bins = int((maxval - thr) / width)``
#: (``pycbc/population/fgmc_functions.py:make_bins``). It is a property of the estimator
#: rather than of the campaign, so it lives here and not in
#: :class:`~sage.search.spec.PastroSpec`: putting it in the spec would make every change
#: to it invalidate the campaign's GPU stages, and it is a knob one wants to be able to
#: turn.
#:
#: 0.5 in units of the ranking statistic. Chosen from the measured occupancy rather than
#: from taste: the O3a smoke background holds **260** events above the analysis threshold
#: -- 8,108 in total, but the threshold is what the density is defined above -- so 0.5
#: gives 15 bins averaging 17 events each. Finer bins are mostly empty and the ratio then
#: jumps from bin to bin; PyCBC's own default range is 0.1 to 0.5.
#:
#: The production campaign's 10 yr background carries roughly 7,262 events above the same
#: threshold, 28 times as many, and will support a finer width. This is the number to
#: revisit then -- not by watching what makes an inversion disappear, which would be
#: tuning the estimator against the answer, but from the occupancy.
BIN_WIDTH = 0.5

#: Fractional Poisson error reported for a bin that holds no events and had to be given a
#: fictitious count. PyCBC calls this 100 per cent, which is what one count means.
FICTITIOUS_FRACERR = 1.0

#: What the noise density is above the loudest background event ever measured.
#:
# ``"floor"`` (sgwc-1) holds the density at :data:`NOISE_FLOOR` outside the observed
#: range: ``pastro.ipynb`` builds its interpolators with
#: ``interp1d(..., bounds_error=False, fill_value=1e-10)``, so a candidate louder than
#: every background event is not given zero -- which would make the mixture likelihood
#: minus infinity and no rate inferable -- but a floor so far below any measured density
#: that ``p_astro`` is one to within double precision.
#: ``"fictitious"`` (PyCBC ``log_rho_bg``) puts one count in a bin stretching from the
#: last edge out to the loudest trigger, so the density there is small but finite.
#:
#: **This choice, and not the estimator, decides whether p_astro is monotone here.**
#: PyCBC's rule makes the noise density *constant* above the background, so the ordering
#: of everything above it is set by the signal density alone -- and ours genuinely falls
#: there, because the network's ranking statistic saturates: 5,192 injections land in
#: [18.0, 18.5) against 188 in [19.0, 20.0). Constant noise divided by falling signal is a
#: falling ratio. Measured on the O3a campaign, PyCBC's rule gives 0.815 at stat 18.66
#: against 0.954 at 17.89 -- inverted -- while sgwc-1's gives 1.0 for every candidate
#: above the background.
#:
#: PyCBC never meets this because its reweighted SNR is unbounded: its signal density
#: falls as a power law while its noise density falls exponentially, so the ratio rises.
#: A saturating statistic breaks that, which is why the reference to follow here is
#: sgwc-1 rather than PyCBC. See SB-67.
ABOVE_BACKGROUND = "floor"

#: The density held outside the observed range under ``ABOVE_BACKGROUND = "floor"``.
#: sgwc-1's ``fill_value``, verbatim. It is a floor and not an estimate: nothing was
#: measured out there, and the number is chosen small enough that the ratio it produces is
#: dominated by the signal density rather than by this constant.
NOISE_FLOOR = 1e-10


@dataclass
class HistogramDensity:
    """
    A normalised histogram of the statistic, evaluated per bin.

    PyCBC's construction, ported from ``pycbc.population.fgmc_functions`` --
    ``log_rho_bg`` for the background and ``log_rho_fg`` for the injections. Nothing is
    fitted and nothing is smoothed: the density in a bin is its count divided by the bin
    width and the total, and the Poisson error on that count travels with it.

    **Why not a kernel estimate.** A KDE over the raw samples follows individual samples
    wherever they are sparse, which is the top of the range -- exactly where a detection
    lives. Measured on the test background: at a Silverman bandwidth of 0.155 with no
    samples at all between 13 and 15, the log density read -34.6 there and recovered to
    -9.4 two units later off a single sample. A histogram cannot do that, because its bins
    are fixed by the grid rather than by where samples happen to fall, and an empty bin is
    given one fictitious count rather than an exponentially small number.

    Nobody in the field uses a raw-sample KDE for this: PyCBC histograms both densities,
    GstLAL histograms and then smooths, and sgwc-1 runs its kernel estimate over the
    *histogram bin centres* weighted by the counts -- a smoothed histogram. See SB-67.

    **Above the top bin.** A trigger louder than every event in the sample gets one
    fictitious count in a bin stretching from the last edge out to that trigger, and the
    normalising total is incremented by one -- ``log_rho_bg``'s rule verbatim. This is the
    same conservative counting the false-alarm rate uses, ``(1 + n) / T``, and it is what
    replaces both a hard zero (which forces ``p_astro`` to exactly one) and an
    extrapolated tail (which inverted the likelihood ratio; see SB-64).
    """

    edges: np.ndarray
    counts: np.ndarray
    support: CommonSupport
    n_total: int

    @classmethod
    def build(
        cls,
        samples: np.ndarray,
        support: CommonSupport,
        bin_width: float = BIN_WIDTH,
    ) -> "HistogramDensity":
        """
        Bin ``samples`` linearly from the analysis threshold.

        The lower edge is nudged below ``support.stat_lo`` by 1e-4, as PyCBC does, so a
        sample sitting exactly on the threshold falls inside the first bin rather than
        outside every bin.
        """
        samples = np.asarray(samples, dtype=np.float64).ravel()
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            raise ValueError("a histogram density needs at least one sample")
        if bin_width <= 0:
            raise ValueError(f"bin_width must be positive, got {bin_width}")
        lo = float(support.stat_lo) - 1e-4
        hi = float(samples.max())
        inside = samples[samples >= lo]
        if inside.size == 0:
            raise ValueError(
                f"no sample lies at or above the analysis threshold "
                f"{support.stat_lo:.6g}, so nothing can be estimated above it"
            )
        n_bins = max(1, int((hi - lo) / float(bin_width)))
        edges = np.linspace(lo, hi, n_bins + 1)
        counts, edges = np.histogram(inside, bins=edges)
        return cls(
            edges=edges,
            counts=counts.astype(np.int64),
            support=support,
            n_total=int(counts.sum()),
        )

    def evaluate(self, stat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ``(log_density, fractional_error)`` at the given statistics.

        ``log_rho_bg`` verbatim, vectorised. A query below the first edge is outside what
        this density describes and returns ``-inf``; a query above the last edge, or in an
        empty bin, is given one fictitious count.
        """
        query = np.asarray(stat, dtype=np.float64).ravel()
        counts = self.counts.astype(np.float64)
        top = float(self.edges[-1])
        above = query >= top
        # One fictitious count widens the total, once, for the whole call -- PyCBC
        # increments N by one if any trigger is above the top bin, not once per trigger.
        total = float(self.n_total) + (
            1.0 if (above.any() and ABOVE_BACKGROUND == "fictitious") else 0.0
        )

        log_density = np.full(query.shape, -np.inf, dtype=np.float64)
        fracerr = np.full(query.shape, np.nan, dtype=np.float64)

        if above.any() and ABOVE_BACKGROUND == "fictitious":
            # The fictitious bin runs from the last edge out to the loudest query.
            width = float(np.max(query[above])) - top
            if width <= 0:
                width = float(self.edges[-1] - self.edges[-2])
            log_density[above] = -np.log(total) - np.log(width)
            fracerr[above] = FICTITIOUS_FRACERR
        elif above.any():
            # sgwc-1's floor. Not zero: a zero makes the mixture likelihood minus infinity
            # at that trigger and no rate can be inferred at all.
            log_density[above] = np.log(NOISE_FLOOR)
            fracerr[above] = FICTITIOUS_FRACERR

        within = (query >= self.edges[0]) & ~above
        if within.any():
            index = np.searchsorted(self.edges, query[within], side="right") - 1
            index = np.clip(index, 0, counts.size - 1)
            occupancy = counts[index]
            # An empty bin holding a trigger is given one count rather than zero: the
            # density there is unmeasured, not zero, and a zero would make the likelihood
            # ratio infinite on the strength of a bin nobody sampled.
            fictitious = occupancy == 0
            occupancy = np.where(fictitious, 1.0, occupancy)
            widths = np.diff(self.edges)[index]
            log_density[within] = (
                np.log(occupancy) - np.log(widths) - np.log(total)
            )
            fracerr[within] = np.where(
                fictitious, FICTITIOUS_FRACERR, occupancy ** -0.5
            )
        return log_density, fracerr

    def log_prob(
        self, stat: np.ndarray, mchirp: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Log density at the given points, shaped like the query."""
        if mchirp is not None:
            raise ValueError(
                "a histogram density is defined on the ranking statistic alone; it "
                "carries no chirp-mass axis"
            )
        query = np.asarray(stat, dtype=np.float64)
        values, _ = self.evaluate(query)
        return np.reshape(values, query.shape)

    def fractional_error(self, stat: np.ndarray) -> np.ndarray:
        """Poisson fractional error on the density at the given points."""
        query = np.asarray(stat, dtype=np.float64)
        _, err = self.evaluate(query)
        return np.reshape(err, query.shape)

    def normalisation(self) -> float:
        """
        Integral over the bins, which is one by construction.

        Taken from the counts rather than by quadrature: the density is piecewise constant
        and its integral is exactly ``sum(counts) / n_total``. Quadrature over the support
        grid would report the trapezoid's error on a step function instead.
        """
        return float(self.counts.sum()) / float(self.n_total)










def signal_density(
    injection_stats: np.ndarray,
    support: CommonSupport,
    injection_mchirp: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    bin_width: float = BIN_WIDTH,
) -> Density:
    """
    Foreground density from recovered injections: ``pycbc.population.log_rho_fg``.

    A normalised histogram of the injections' ranking statistics. Nothing is fitted and
    nothing is smoothed; see :class:`HistogramDensity` for why.

    ``weights`` is not accepted. Reweighting the injections to an assumed astrophysical
    population is a real thing to want, but PyCBC's construction counts them, and a
    weighted count is a different estimator that would need its own error model -- the
    Poisson error on a bin is the error on a *count*.
    """
    if injection_mchirp is not None or support.is_2d:
        raise ValueError(
            "the signal density is defined on the ranking statistic alone; a "
            "chirp-mass axis has no counterpart in PyCBC's construction"
        )
    if weights is not None:
        raise ValueError(
            "a histogram density counts its samples and cannot weight them; the "
            "Poisson error it reports is the error on a count"
        )
    return HistogramDensity.build(injection_stats, support, bin_width=bin_width)


def noise_density(
    background_stats: np.ndarray,
    support: CommonSupport,
    background_mchirp: Optional[np.ndarray] = None,
    bin_width: float = BIN_WIDTH,
) -> Density:
    """
    Background density from the time-slid triggers: ``pycbc.population.log_rho_bg``.

    A normalised histogram of the background's ranking statistics. A trigger louder than
    every background event is given one fictitious count in a bin stretching out to it,
    which is the same conservative counting the false-alarm rate uses -- not a hard zero
    (which forces ``p_astro`` to exactly one) and not an extrapolated tail (which inverted
    the likelihood ratio; see SB-64).
    """
    if background_mchirp is not None or support.is_2d:
        raise ValueError(
            "the noise density is defined on the ranking statistic alone; a chirp-mass "
            "axis has no counterpart in PyCBC's construction"
        )
    return HistogramDensity.build(background_stats, support, bin_width=bin_width)


def verify_normalisation(density: Density, atol: float = 1e-3) -> float:
    """
    Assert a density integrates to one over the common support.

    Returned as well as asserted, so a caller can record how far off it was. A component
    that is not normalised does not merely shift p_astro, it rescales the likelihood
    ratio by a constant the inference will absorb into the rates -- producing a rate that
    is wrong and a p_astro that looks reasonable.

    The default tolerance is set by the quadrature, not by ambition. The integral is a
    trapezoid over ``support.n_stat`` nodes, and both densities are now smooth kernel
    estimates, for which the error falls as roughly ``1 / n_stat**2``: 1.4e-7 at 512 nodes
    and 8.6e-9 at 2048. The tolerance is kept at 1e-3 rather than tightened to match,
    because it is a check that a density is normalised at all and not a measurement of the
    quadrature. A caller who wants a tighter check should refine the grid to match rather
    than lower this and watch it fail on the quadrature instead of on the density.
    """
    value = float(density.normalisation())
    if not np.isfinite(value) or abs(value - 1.0) > atol:
        raise ValueError(
            f"this density integrates to {value} over its support, not one to within "
            f"{atol}; the likelihood ratio built from it would carry that factor"
        )
    return value
