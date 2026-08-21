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


def _robust_scale(values: np.ndarray) -> float:
    """
    Scale estimate that a handful of extreme values cannot inflate.

    ``min(std, IQR / 1.349)``, Silverman's robust choice. The plain standard deviation is
    dominated by the loudest few triggers -- which is precisely the tail this density is
    read in -- and a bandwidth set from it over-smooths the whole distribution to
    accommodate a handful of points. The 1.349 converts an interquartile range to the
    equivalent Gaussian sigma.
    """
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = float(q3 - q1) / 1.349
    candidates = [value for value in (std, iqr) if value > 0.0]
    if not candidates:
        raise ValueError(
            "every sample is identical, so the distribution has no width and no "
            "bandwidth can be derived from it"
        )
    return min(candidates)


def bandwidth_from_data(samples: np.ndarray, rule: str = "silverman") -> np.ndarray:
    """
    Bandwidth per axis, derived from the samples alone.

    Never from histogram bin centres and never tied to the largest observed statistic:
    both make the density move whenever more background is accumulated, which is what
    stops the inference converging as a campaign deepens.

    Parameters
    ----------
    rule : {"silverman", "scott"}
        Both are ``factor * A * n ** (-1 / (d + 4))`` with ``A`` the robust scale above;
        they differ only in the constant.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    if samples.shape[0] < samples.shape[1] and samples.shape[0] <= 2:
        samples = samples.T
    n, d = samples.shape
    if n < 2:
        raise ValueError("a bandwidth needs at least two samples")
    if rule == "silverman":
        factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    elif rule == "scott":
        factor = 1.0
    else:
        raise ValueError(f"unknown bandwidth rule {rule!r}, expected silverman or scott")
    exponent = -1.0 / (d + 4.0)
    return np.array(
        [factor * _robust_scale(samples[:, k]) * n**exponent for k in range(d)],
        dtype=np.float64,
    )


def _as_columns(stat, mchirp) -> np.ndarray:
    """Stack a query into an ``(n, d)`` array without copying the caller's shape."""
    stat = np.asarray(stat, dtype=np.float64).ravel()
    if mchirp is None:
        return stat[:, None]
    mchirp = np.asarray(mchirp, dtype=np.float64).ravel()
    if mchirp.size != stat.size:
        raise ValueError(
            f"{stat.size} statistics against {mchirp.size} chirp masses; read side by "
            "side they would describe different triggers"
        )
    return np.column_stack([stat, mchirp])


@dataclass
class TruncatedKDE:
    """
    Kernel density estimate, truncated and renormalised on the common support.

    Boundary corrected per kernel: a sample near an edge has part of its kernel outside
    the support, and dividing that kernel by the fraction of its own mass that lies inside
    puts the missing weight back where it came from. Renormalising the finished density
    globally instead would spread the correction over the whole support, thinning the
    interior to repair the edge. The edge here is the analysis threshold, where the noise
    density has most of its mass, so the difference is not a detail.
    """

    support: CommonSupport
    bandwidth: np.ndarray
    samples: np.ndarray
    weights: Optional[np.ndarray] = None
    boundary_corrected: bool = True

    def __post_init__(self) -> None:
        """Coerce, drop samples outside the support, and normalise the weights."""
        self.samples = np.atleast_2d(np.asarray(self.samples, dtype=np.float64))
        if self.samples.shape[0] == 1 and self.samples.shape[1] > 2:
            self.samples = self.samples.T
        self.bandwidth = np.asarray(self.bandwidth, dtype=np.float64).ravel()
        if self.bandwidth.size != self.samples.shape[1]:
            raise ValueError(
                f"{self.bandwidth.size} bandwidths for {self.samples.shape[1]} axes"
            )
        if np.any(self.bandwidth <= 0) or not np.isfinite(self.bandwidth).all():
            raise ValueError("every bandwidth must be finite and positive")
        weights = (
            np.ones(self.samples.shape[0], dtype=np.float64)
            if self.weights is None
            else np.asarray(self.weights, dtype=np.float64).ravel()
        )
        if weights.size != self.samples.shape[0]:
            raise ValueError(
                f"{weights.size} weights for {self.samples.shape[0]} samples"
            )
        if np.any(weights < 0) or not np.isfinite(weights).all():
            raise ValueError("weights must be finite and non-negative")
        columns = [self.samples[:, 0]]
        if self.samples.shape[1] > 1:
            columns.append(self.samples[:, 1])
        inside = self.support.contains(*columns) if len(columns) > 1 else (
            self.support.contains(columns[0])
        )
        if not inside.any():
            raise ValueError(
                "no sample lies inside the common support, so nothing can be estimated "
                "on it"
            )
        self.samples = self.samples[inside]
        weights = weights[inside]
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("the surviving samples carry no weight between them")
        self.weights = weights / total

    @property
    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-axis support edges, in the sample column order."""
        lo = [self.support.stat_lo]
        hi = [self.support.stat_hi]
        if self.samples.shape[1] > 1:
            lo.append(self.support.mchirp_lo)
            hi.append(self.support.mchirp_hi)
        return np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64)

    def _kernel_mass(self) -> np.ndarray:
        """Fraction of each kernel's mass lying inside the support."""
        from scipy.stats import norm

        if not self.boundary_corrected:
            return np.ones(self.samples.shape[0], dtype=np.float64)
        lo, hi = self._bounds
        inside = np.ones(self.samples.shape[0], dtype=np.float64)
        for axis in range(self.samples.shape[1]):
            scaled = self.bandwidth[axis]
            inside *= norm.cdf(
                (hi[axis] - self.samples[:, axis]) / scaled
            ) - norm.cdf((lo[axis] - self.samples[:, axis]) / scaled)
        # A kernel whose mass is numerically zero inside the support sits many bandwidths
        # outside it and cannot be rescued by division; it is dropped by the floor rather
        # than allowed to become an infinity.
        return np.maximum(inside, 1e-300)

    def log_prob(
        self, stat: np.ndarray, mchirp: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Log density at the given points.

        ``-inf`` outside the support: the density is defined on the support and nowhere
        else, and returning a small positive number instead would let a candidate beyond
        the truncation carry a finite likelihood ratio built from two extrapolations.
        """
        from scipy.special import logsumexp

        query = _as_columns(stat, mchirp)
        if query.shape[1] != self.samples.shape[1]:
            raise ValueError(
                f"this density has {self.samples.shape[1]} axes and was queried with "
                f"{query.shape[1]}"
            )
        log_weights = np.log(np.maximum(self.weights, 1e-300)) - np.log(
            self._kernel_mass()
        )
        # (n_query, n_samples), summed over samples.
        exponent = np.zeros((query.shape[0], self.samples.shape[0]), dtype=np.float64)
        for axis in range(query.shape[1]):
            scaled = (
                query[:, axis][:, None] - self.samples[:, axis][None, :]
            ) / self.bandwidth[axis]
            exponent += -0.5 * scaled**2
        norm_const = float(
            np.sum(np.log(self.bandwidth))
            + 0.5 * query.shape[1] * np.log(2.0 * np.pi)
        )
        out = logsumexp(exponent + log_weights[None, :], axis=1) - norm_const
        columns = [query[:, 0]] + ([query[:, 1]] if query.shape[1] > 1 else [])
        outside = ~self.support.contains(*columns)
        out = np.where(outside, -np.inf, out)
        return np.reshape(out, np.shape(np.asarray(stat, dtype=np.float64)))

    def normalisation(self) -> float:
        """
        Integral over the common support.

        One by construction when boundary correction is on -- each kernel contributes
        exactly its own weight -- so a value away from one is a report that something is
        wrong rather than a factor to divide by.
        """
        return _integrate(self, self.support)

    def resample_bandwidth(self, rule: str = "scott") -> np.ndarray:
        """Recompute the bandwidth from the samples and adopt it."""
        self.bandwidth = bandwidth_from_data(self.samples, rule=rule)
        return self.bandwidth


@dataclass
class TailBlendedDensity:
    """
    A kernel density whose tail is the shared generalised Pareto fit, anchored by mass.

    Above the peaks-over-threshold value the kernel estimate is not estimating anything:
    it is built from a handful of kernels there, so what it reports is the shape of the
    Gaussian kernel rather than the shape of the data. The fitted tail is used instead.

    **The join is anchored by mass, not by height.** In FGMC Eq. (10) the noise density is
    the *shape of the background rate*, so its survival function has to agree with the
    false-alarm rate measured from the same background, normalised at the join::

        P(X > x) / P(X > join)  =  FAR(x) / FAR(join)      for x >= join

    exactly at ``x = join`` by construction, and above it as far as the fitted tail
    tracks the counted one. That is a statement about mass. Matching the fitted tail's *height* to the kernel's at
    the join and renormalising afterwards satisfies no such constraint: it makes the noise
    density and the FAR curve two different descriptions of one background, and the error
    is a single scale factor on the whole tail -- the only region either is read in.

    Both PyCBC paths anchor by mass, by different routes that agree. Low latency derives
    the noise density from the false-alarm rate directly, ``b(rho) = alpha * FAR(rho)``
    (``pycbc.population.live_pastro.noise_density_from_far``), which is ``-dFAR/drho`` for
    an exponential noise model -- so the two cannot disagree by construction. Offline
    histograms the very slide triggers the rate counts
    (``pycbc.population.fgmc_functions.log_rho_bg``), so each bin's mass is its share of
    the count and the survival is the counted fraction by definition.

    ``tail_mass`` is the share of the background above the join, taken from the counted
    false-alarm curve when one is supplied and from the exceedance count otherwise. The
    body carries the rest.

    Attributes
    ----------
    step : float
        Ratio of the tail's height to the kernel's at the join. Mass anchoring does not
        force continuity, so this is generally not one, and it is reported rather than
        removed: a large step means the peaks-over-threshold value is misplaced -- the
        kernel and the fit disagree about the density where they meet -- which is a fact
        about the fit worth surfacing, not an artefact to be smoothed away by rescaling.
    """

    kde: TruncatedKDE
    tail: object
    join: float
    support: CommonSupport
    tail_mass: float = 1.0
    log_body_norm: float = 0.0
    log_tail_norm: float = 0.0
    step: float = 1.0
    # True when the fit threshold is at or below the analysis threshold, so the whole
    # support lies in the region the fit describes and the kernel estimate has nothing to
    # contribute. Ordinary rather than exceptional: choose_threshold picks the threshold
    # from the background alone and has no reason to land above the analysis cut.
    body_empty: bool = False

    @classmethod
    def build(cls, kde: "TruncatedKDE", tail, support: CommonSupport, far_curve=None):
        """
        Split the support at the fit threshold and give each side its measured mass.

        ``far_curve`` supplies the mass split as ``FAR(join) / FAR(stat_lo)`` off the
        counted curve, which makes the noise density and the false-alarm rate exactly
        consistent. Without one the split is the exceedance fraction of the samples
        themselves, which is the same quantity up to the conservative ``1 +`` in the FAR
        counting -- and is what PyCBC's offline histogram uses.
        """
        from scipy.stats import genpareto

        join = max(float(tail.threshold), float(support.stat_lo))
        top = float(np.max(kde.samples[:, 0]))
        if join >= top:
            raise ValueError(
                f"the tail threshold at {join} is at or above the loudest sample at "
                f"{top}, so the kernel estimate covers nothing the tail does not"
            )
        if join >= support.stat_hi:
            return kde

        body_empty = join <= float(support.stat_lo)
        if far_curve is not None:
            rate = np.asarray(
                far_curve.far_of(np.array([join, support.stat_lo])), dtype=np.float64
            )
            tail_mass = float(rate[0] / rate[1])
        else:
            values = kde.samples[:, 0]
            inside = values >= support.stat_lo
            tail_mass = float(np.count_nonzero(values > join)) / float(
                max(np.count_nonzero(inside), 1)
            )
        tail_mass = 1.0 if body_empty else float(np.clip(tail_mass, 1e-12, 1.0 - 1e-12))

        # Mass of the fitted tail that falls inside the support, so the piece above the
        # join integrates to exactly `tail_mass` rather than to whatever the fit puts
        # between the join and the top of the support.
        inside_tail = float(
            genpareto.cdf(
                support.stat_hi - tail.threshold, tail.shape, loc=0.0, scale=tail.scale
            )
            - genpareto.cdf(join - tail.threshold, tail.shape, loc=0.0, scale=tail.scale)
        )
        if inside_tail <= 0.0:
            raise ValueError(
                f"the fitted tail places no mass between the join at {join} and the top "
                f"of the support at {support.stat_hi}; with shape {tail.shape} its finite "
                f"endpoint is {tail.finite_endpoint}, which lies below the support"
            )
        if body_empty:
            log_body_norm = 0.0
        else:
            body = _integrate_between(kde, support.stat_lo, join)
            if body <= 0.0:
                raise ValueError("the kernel estimate carries no mass below the join")
            log_body_norm = float(np.log(body))

        blended = cls(
            kde=kde,
            tail=tail,
            join=join,
            support=support,
            tail_mass=tail_mass,
            log_body_norm=log_body_norm,
            log_tail_norm=float(np.log(inside_tail)),
            body_empty=body_empty,
        )
        if body_empty:
            # Nothing meets the tail here, so there is no step to measure.
            blended.step = 1.0
            return blended
        just_below = float(np.exp(blended.log_prob(np.array([join]))[0]))
        just_above = float(
            np.exp(
                np.log(tail_mass)
                - blended.log_tail_norm
                + genpareto.logpdf(
                    max(join - tail.threshold, 0.0),
                    tail.shape,
                    loc=0.0,
                    scale=tail.scale,
                )
            )
        )
        blended.step = just_above / just_below if just_below > 0 else float("inf")
        return blended

    def log_prob(
        self, stat: np.ndarray, mchirp: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Log density: rescaled kernel below the join, fitted tail above it."""
        from scipy.stats import genpareto

        stat_array = np.asarray(stat, dtype=np.float64)
        flat = stat_array.ravel()
        out = np.full(flat.shape, -np.inf, dtype=np.float64)
        beyond = flat >= self.join if self.body_empty else flat > self.join
        below = ~beyond & self.support.contains(flat)
        # Evaluated only where it is used. The kernel estimate costs an
        # (n_query, n_samples) block, so running it over the whole query and overwriting
        # the tail afterwards allocates hundreds of gigabytes on a campaign-sized
        # background for values that are then discarded.
        if np.any(below):
            out[below] = (
                np.log1p(-self.tail_mass)
                + np.asarray(self.kde.log_prob(flat[below]), dtype=np.float64)
                - self.log_body_norm
            )
        if np.any(beyond):
            out[beyond] = (
                np.log(self.tail_mass)
                - self.log_tail_norm
                + genpareto.logpdf(
                    flat[beyond] - self.tail.threshold,
                    self.tail.shape,
                    loc=0.0,
                    scale=self.tail.scale,
                )
            )
        out = np.where(self.support.contains(flat), out, -np.inf)
        return np.reshape(out, stat_array.shape)

    def survival(self, stat: np.ndarray) -> np.ndarray:
        """
        Fraction of the noise density above ``stat``, for comparison with the FAR curve.

        Exposed because the agreement between this and ``FarCurve.far_of`` normalised at
        the threshold is the property mass anchoring exists to guarantee, and a property
        nothing asserts is a property nobody checks.

        **The guarantee is anchored at the join, and holds above it.** What mass anchoring
        fixes is the single number ``tail_mass``, the share of the background above the
        join, so ``P(X > x) / P(X > join) == FAR(x) / FAR(join)`` is exact at ``x = join``
        by construction and holds above it to the extent the fitted tail describes the
        counted one. Below the join the two are different estimates of the same body --
        a kernel density and a counted survival -- and they agree only to within the
        smoothing, which is not a defect but is not the identity either.
        """
        stat = np.asarray(stat, dtype=np.float64)
        nodes = self.support.grid()[0]
        weights = self.support.cell_volume()
        values = np.exp(self.log_prob(nodes)) * weights
        above = np.concatenate([np.cumsum(values[::-1])[::-1], [0.0]])
        index = np.searchsorted(nodes, stat, side="left")
        return above[np.clip(index, 0, above.size - 1)]

    def normalisation(self) -> float:
        """Integral over the common support."""
        return _integrate(self, self.support)


def _integrate_between(density, lo: float, hi: float, n: int = 2048) -> float:
    """Trapezoid integral of a density over part of its support."""
    nodes = np.linspace(float(lo), float(hi), int(n))
    values = np.exp(np.asarray(density.log_prob(nodes), dtype=np.float64))
    return float(np.trapezoid(values, nodes))


def _integrate(density, support: CommonSupport) -> float:
    """Trapezoid integral of a density over the whole common support."""
    nodes = support.grid()
    weights = support.cell_volume()
    if len(nodes) == 1:
        values = np.exp(density.log_prob(nodes[0]))
        return float(np.sum(values * weights))
    stat, mchirp = np.meshgrid(nodes[0], nodes[1], indexing="ij")
    values = np.exp(density.log_prob(stat.ravel(), mchirp.ravel())).reshape(stat.shape)
    return float(np.sum(values * weights))


def signal_density(
    injection_stats: np.ndarray,
    support: CommonSupport,
    injection_mchirp: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    rule: str = "silverman",
) -> Density:
    """
    Foreground density from recovered injections.

    ``weights`` reweights the injections to the assumed astrophysical population, so the
    density describes signals as they would arrive rather than as they were drawn. The
    draw distribution is a property of the injection campaign and has no business in a
    statement about what the universe produces.

    Truncated on the same support as the noise density and renormalised there. Both
    components have to live on one support or their ratio is a property of the truncation.
    """
    samples = _as_columns(injection_stats, injection_mchirp)
    if support.is_2d and injection_mchirp is None:
        raise ValueError(
            "the support resolves chirp mass but the injections carry none, so the "
            "signal density would be defined on fewer axes than the noise density"
        )
    return TruncatedKDE(
        support=support,
        bandwidth=bandwidth_from_data(samples, rule=rule),
        samples=samples,
        weights=weights,
    )


def noise_density(
    background_stats: np.ndarray,
    support: CommonSupport,
    background_mchirp: Optional[np.ndarray] = None,
    tail: Optional[object] = None,
    background_livetime_s: float = 0.0,
    foreground_livetime_s: float = 0.0,
    rule: str = "silverman",
    far_curve=None,
) -> Density:
    """
    Background density from time-slid triggers, continued by the fitted tail.

    The tail model is the one the false-alarm-rate layer uses, so the two cannot describe
    the same background differently. It is used only above the loudest background trigger,
    where the kernel estimate would otherwise be reporting the shape of its own outermost
    kernels rather than the shape of the data -- the same discipline the FAR curve keeps,
    counting where it can and extrapolating only past the end of the count.

    ``background_livetime_s`` and ``foreground_livetime_s`` are accepted and recorded but
    do not scale the density: this is a probability density over the statistic, and the
    rate that converts it into an expected count is inferred separately by
    :func:`~sage.search.pastro.rates.fit_rates`. Folding livetime in here would apply the
    same information twice.

    Passing no ``tail`` gives the kernel estimate alone, truncated at the top of the
    support. Passing ``far_curve`` as well makes the mass split at the join exactly the
    counted false-alarm ratio, so this density and that curve describe one background
    rather than two; see :class:`TailBlendedDensity`.
    """
    samples = _as_columns(background_stats, background_mchirp)
    if support.is_2d and background_mchirp is None:
        raise ValueError(
            "the support resolves chirp mass but the background carries none, so the "
            "noise density would be defined on fewer axes than the signal density"
        )
    kde = TruncatedKDE(
        support=support,
        bandwidth=bandwidth_from_data(samples, rule=rule),
        samples=samples,
    )
    if tail is None:
        return kde
    if support.is_2d:
        raise ValueError(
            "a tail continuation is defined on the ranking statistic alone; a "
            "two-dimensional noise density has no single edge to continue past"
        )
    return TailBlendedDensity.build(kde, tail, support, far_curve=far_curve)


def verify_normalisation(density: Density, atol: float = 1e-3) -> float:
    """
    Assert a density integrates to one over the common support.

    Returned as well as asserted, so a caller can record how far off it was. A component
    that is not normalised does not merely shift p_astro, it rescales the likelihood
    ratio by a constant the inference will absorb into the rates -- producing a rate that
    is wrong and a p_astro that looks reasonable.

    The default tolerance is set by the quadrature, not by ambition. The integral is a
    trapezoid over ``support.n_stat`` nodes, and a :class:`TailBlendedDensity` is
    discontinuous at the join by construction -- mass anchoring does not force continuity,
    and ``step`` reports the size of the jump. Measured on an exponential background with
    the tail fitted at a 1000-count threshold, the error falls as roughly ``1 / n_stat**2``
    but from a much worse starting point than a smooth density::

        n_stat        256      512     1024     2048     4096
        blended    1.5e-3   3.7e-4   1.0e-4   1.9e-5   5.3e-6
        smooth KDE      -    1.4e-7        -   8.6e-9        -

    So ``1e-6`` -- an earlier default -- was unreachable for the density this is most
    often called on, at any grid a campaign would actually use. A caller who wants a
    tighter check should refine the grid to match rather than lower this and watch it
    fail on the quadrature instead of on the density.
    """
    value = float(density.normalisation())
    if not np.isfinite(value) or abs(value - 1.0) > atol:
        raise ValueError(
            f"this density integrates to {value} over its support, not one to within "
            f"{atol}; the likelihood ratio built from it would carry that factor"
        )
    return value
