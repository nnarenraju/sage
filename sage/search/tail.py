#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : tail.py
Description   : Peaks-over-threshold tail fitting, shared by far.py and pastro.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One tail model serves both the FAR extrapolation and the p_astro noise density, so the
two consumers cannot disagree about the same data.

The fitted shape is always reported as fitted. There is deliberately no branch that
replaces it with zero when a test fails to reject the exponential: "the data cannot
distinguish xi from zero" is not "xi is zero", and substituting the second for the first
is a pre-test estimator -- a model chosen from the data and then reported as if it had
been known all along. It would also bias in one direction only. Far above the threshold
an exponential falls off enormously faster than any xi > 0 power law, so zeroing a
genuine but unresolved shape always makes the extrapolated tail lighter, the extrapolated
rate smaller and the candidate look better. ``lrt_p_value`` still reports what the test
found, as a diagnostic to read beside the fit rather than a switch acting on it.

Sign convention. ``TailFit.shape`` is ``xi`` in the extreme-value convention,

    P(X > u + y | X > u) = (1 + xi * y / scale) ** (-1 / xi),    y >= 0,

so ``xi > 0`` is an unbounded power-law tail, ``xi = 0`` the exponential branch, and
``xi < 0`` a tail bounded above at ``u - scale / xi``. That is exactly the shape
parameter ``c`` of ``scipy.stats.genpareto``, which is why the fitted value is passed
through unaltered. The other convention in circulation (Hosking and Wallis' ``k = -xi``)
differs by sign alone, and a number carried in under the wrong one turns a background
that cannot exceed a finite value into one with a power-law tail, or the reverse. There
is no symptom: both fits look entirely ordinary, and the only quantity that moves is the
extrapolated FAR, which is the one number nobody can check against the data.

Monte Carlo. Both diagnostics use simulated null distributions rather than tabulated
asymptotics, because the generalised Pareto likelihood is irregular where they are read
(see :func:`exponential_lrt`). A default fit is therefore about three thousand
maximum-likelihood fits -- one for the tail, ``n_bootstrap`` for its covariance and
``n_null`` for each of the two nulls -- which is a few seconds for a background of a few
thousand exceedances, paid once per campaign per removal mode.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

# Row and column order of TailFit.covariance. Named rather than implied, because a
# transposed 2x2 is symmetric-looking and a swapped one is not detectable by eye.
PARAMETER_ORDER: Tuple[str, str] = ("scale", "shape")

# Below SHAPE_FLOOR the generalised Pareto likelihood has no maximum: it diverges as the
# fitted upper endpoint is squeezed onto the largest excess, so anything reported there
# describes one order statistic rather than a tail. SHAPE_CEIL is far above anything a
# ranking statistic can produce -- the mean of the exceedances already fails to exist
# above one -- and exists only to close the search bracket.
SHAPE_FLOOR: float = -1.0
SHAPE_CEIL: float = 10.0

# Fewest exceedances a two-parameter tail may be fitted from. Well below the 500 that
# choose_threshold defaults to; this is the point past which the fit is arithmetic
# rather than inference, not a recommendation.
MIN_FIT_EXCEEDANCES: int = 10


@dataclass
class TailFit:
    """
    A generalised-Pareto fit above a threshold, with uncertainty.

    Attributes
    ----------
    threshold : float
        The peaks-over-threshold value ``u``. Exceedances are the statistics strictly
        above it; a value exactly at the threshold is not an exceedance, so a heavily
        tied statistic cannot contribute a point mass at zero excess that no continuous
        density can represent.
    scale : float
        Scale of the fitted tail, in units of the ranking statistic.
    shape : float
        ``xi`` in the convention stated in the module docstring: positive is unbounded,
        zero is exponential, negative is bounded above at :attr:`finite_endpoint`.
    covariance : np.ndarray
        Bootstrap covariance of the two-parameter fit, 2x2 over ``PARAMETER_ORDER``. A
        tail that continues a FAR curve past the measured background is only quotable
        with a band on it, so the covariance travels with the point estimate rather than
        being recomputed by whoever plots it.
    n_exceedances : int
        Number of statistics strictly above the threshold, the sample the fit was made
        from.
    lrt_p_value : float
        p-value of :func:`exponential_lrt`. Large means the exceedances are consistent
        with an exponential tail; it is reported, and acted on by nobody. See the module
        docstring for why the fit is never replaced by that branch.
    ad_p_value : float
        p-value of :func:`anderson_darling` for the two-parameter fit. Small means no
        generalised Pareto tail describes these exceedances, whichever branch was taken,
        and the extrapolation should not be quoted.
    """

    threshold: float
    scale: float
    shape: float
    covariance: np.ndarray
    n_exceedances: int
    lrt_p_value: float
    ad_p_value: float

    def __post_init__(self) -> None:
        """
        Refuse a fit that cannot be used as a tail model.

        Checked on construction rather than at the point of use, because this object is
        persisted and then read by two independent consumers: a scale of zero or a shape
        below the floor would surface as an extrapolated rate in a candidate table,
        several stages away from whatever produced it.
        """
        self.threshold = float(self.threshold)
        self.scale = float(self.scale)
        self.shape = float(self.shape)
        self.n_exceedances = int(self.n_exceedances)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        if not np.isfinite(self.threshold):
            raise ValueError(f"threshold must be finite, got {self.threshold}")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(
                f"scale must be finite and positive, got {self.scale}; every excess is "
                "divided by it"
            )
        if not np.isfinite(self.shape):
            raise ValueError(f"shape must be finite, got {self.shape}")
        if self.shape < SHAPE_FLOOR:
            raise ValueError(
                f"shape {self.shape} lies below {SHAPE_FLOOR}, where the generalised "
                "Pareto likelihood is unbounded rather than maximised; such a fit "
                "describes the largest excess and not the tail"
            )
        if self.covariance.shape != (2, 2):
            raise ValueError(
                f"covariance must be 2x2 over {PARAMETER_ORDER}, got shape "
                f"{self.covariance.shape}"
            )
        if not np.isfinite(self.covariance).all():
            raise ValueError("covariance holds non-finite entries")
        if self.n_exceedances < 0:
            raise ValueError(
                f"n_exceedances must not be negative, got {self.n_exceedances}"
            )
        for name in ("lrt_p_value", "ad_p_value"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
            setattr(self, name, value)

    @property
    def finite_endpoint(self) -> Optional[float]:
        """
        Upper endpoint when the shape is negative, else ``None``.

        ``threshold - scale / shape``: with ``shape < 0`` the fitted model says the
        background cannot produce a statistic above this value at all. ``None`` for
        ``shape >= 0`` rather than ``inf``, so a caller that forgets to handle the
        unbounded case fails on the ``None`` instead of silently comparing against an
        infinity that is always true.
        """
        if self.shape >= 0.0:
            return None
        return self.threshold - self.scale / self.shape

    def survival(self, stat: np.ndarray) -> np.ndarray:
        """
        Exceedance probability above the fit threshold.

        ``P(X > stat | X > threshold)`` under the fitted model, which is what
        :meth:`~sage.search.far.FarCurve.far_of` multiplies by the measured rate at the
        threshold to continue the FAR curve past the loudest background event. Being
        conditional, it is one at the threshold by construction, so the continuation
        joins the measured curve there without a step.

        Below the threshold it returns one rather than raising or extrapolating
        backwards. The conditional model makes no statement below its own threshold;
        one is the value that leaves the caller with the measured rate it already has.
        Raising instead would force every caller to mask before interpolating, and
        running the fitted tail backwards would lay a model underneath a region where
        the background was actually counted.

        Above the finite endpoint of a bounded fit it returns zero: the fitted model
        says the noise does not reach there. The caller then sees an IFAR limited by
        ``FarCurve.ifar_cap_yr``, which is the honest reading -- what produced the
        number is the extrapolation, not the background.
        """
        from scipy.stats import genpareto

        stat = _finite_stats(stat, name="stat", ravel=False)
        excess = stat - self.threshold
        out = np.where(
            excess > 0.0,
            genpareto.sf(np.maximum(excess, 0.0), self.shape, loc=0.0,
                         scale=self.scale),
            1.0,
        )
        return np.asarray(np.clip(out, 0.0, 1.0))

    def survival_band(
        self, stat: np.ndarray, level: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Credible band on the survival function from the fit covariance.

        Delta method applied to the log survival, not to the survival itself. The band
        is read several decades below one, where a symmetric linear band puts its lower
        edge below zero; in the log both edges stay positive and the band is
        multiplicative, which is how a rate uncertainty is quoted anyway.

        Parameters
        ----------
        level : float
            Central coverage, so ``0.9`` leaves five per cent outside each edge.

        Returns
        -------
        tuple
            ``(lower, upper)``, both clipped into ``[0, 1]`` and bracketing
            :meth:`survival`.

        Notes
        -----
        Below the threshold the band is ``(1, 1)``: the survival is one there by
        definition, not by estimation, and carrying the parameter uncertainty into a
        region the model does not describe would be inventing it.

        Beyond a bounded fit's :attr:`finite_endpoint` the band collapses to ``(0, 0)``.
        The fitted survival is identically zero there and the delta method has nothing
        to expand about; the uncertainty that matters in that region is the endpoint's
        own, which is a function of both parameters and is available from
        :attr:`covariance`.
        """
        from scipy.stats import norm

        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level}")
        stat = _finite_stats(stat, name="stat", ravel=False)
        survival = self.survival(stat)
        excess = np.asarray(stat, dtype=np.float64) - self.threshold
        inside = (excess > 0.0) & (survival > 0.0)

        d_scale, d_shape = _log_survival_gradient(
            np.where(inside, excess, 0.0), self.scale, self.shape
        )
        variance = (
            self.covariance[0, 0] * d_scale**2
            + 2.0 * self.covariance[0, 1] * d_scale * d_shape
            + self.covariance[1, 1] * d_shape**2
        )
        # A covariance that is only positive semi-definite to rounding can put a tiny
        # negative number here, which would propagate as a NaN through the square root.
        deviation = float(norm.ppf(0.5 + 0.5 * level)) * np.sqrt(
            np.maximum(variance, 0.0)
        )
        lower = np.where(inside, survival * np.exp(-deviation), survival)
        upper = np.where(inside, survival * np.exp(deviation), survival)
        return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)


def threshold_at_count(stats: np.ndarray, n_exceedances: int = 500) -> float:
    """
    PyCBC's threshold rule: the value leaving exactly ``n_exceedances`` above it.

    ``pycbc.events.trigger_fits.tail_threshold(vals, N)``, which returns
    ``min(sorted(vals)[-N:])`` -- the smallest of the loudest N. One difference in
    convention: PyCBC counts inclusively at the threshold, so N values are ``>=`` it,
    while everything here treats exceedances as strictly above. The order statistic one
    place lower is returned, so with no ties exactly ``n_exceedances`` values lie strictly
    above.

    **Under ties the count is a lower bound, not an equality.** If the returned order
    statistic is repeated, every copy of it also falls at or below the threshold and fewer
    than ``n_exceedances`` values exceed it. A ranking statistic read from a float32 head
    ties rarely, but it ties, and :func:`fit_tail` sees the true exceedance count rather
    than the requested one.

    Simpler and more predictable than :func:`choose_threshold`, and it makes the sample
    size an input rather than an outcome. What it does not do is check that a generalised
    Pareto describes the exceedances at that depth -- it fixes how many points the fit
    gets, not whether they are in the tail. Available as an option so the two rules can be
    compared on the same background.
    """
    stats = _finite_stats(stats)
    n = stats.size
    if n_exceedances < MIN_FIT_EXCEEDANCES:
        raise ValueError(
            f"n_exceedances must be at least {MIN_FIT_EXCEEDANCES}, got {n_exceedances}"
        )
    if n <= n_exceedances:
        raise ValueError(
            f"{n} statistics cannot leave {n_exceedances} strictly above any threshold"
        )
    return float(np.sort(stats)[n - n_exceedances - 1])


def threshold_ladder(
    stats: np.ndarray, min_exceedances: int = 500, n_candidates: int = 15
) -> dict:
    """
    Fit the tail at a ladder of thresholds and report the shape at each.

    This is the threshold-stability diagnostic, and it is PyCBC's:
    ``pycbc_fit_sngl_trigs`` takes ``--stat-threshold`` as a *list*, fits at every value,
    writes ``alpha`` with its standard error and a KS p-value per threshold, and plots
    both against threshold. sgwc-1's ``pastro_alternate.ipynb`` sweeps 25 thresholds and
    plots the answer against them for the same purpose.

    Returned rather than acted on. Every reference computes this ladder and has a human
    read it; none lets code pick the winner. :func:`choose_threshold` can select from it
    when asked, but the default threshold rule is the count.

    Returns
    -------
    dict
        ``threshold``, ``shape``, ``scale``, ``std_error`` and ``n_exceedances``, one
        entry per rung, ordered from the lowest threshold up. ``shape`` is ``xi``; see the
        sign convention in the module docstring.
    """
    stats = _finite_stats(stats)
    n = stats.size
    if min_exceedances < MIN_FIT_EXCEEDANCES:
        raise ValueError(
            f"min_exceedances must be at least {MIN_FIT_EXCEEDANCES}, got "
            f"{min_exceedances}; below that a two-parameter tail is arithmetic on a "
            "handful of points rather than an estimate"
        )
    if n <= min_exceedances:
        raise ValueError(
            f"{n} statistics cannot leave {min_exceedances} exceedances above any "
            "threshold; lower min_exceedances or accumulate more background"
        )
    if n_candidates < 2:
        raise ValueError(f"n_candidates must be at least 2, got {n_candidates}")

    ordered = np.sort(stats)
    wanted = np.unique(
        np.round(
            np.geomspace(max(min_exceedances, n // 2), min_exceedances, n_candidates)
        ).astype(np.int64)
    )[::-1]

    thresholds, shapes, scales, errors, counts = [], [], [], [], []
    for m in wanted:
        value = float(ordered[n - m - 1])
        if thresholds and value == thresholds[-1]:
            continue
        exceedance = ordered[ordered > value] - value
        if exceedance.size < min_exceedances:
            # Ties at this order statistic leave fewer exceedances than the count asked
            # for; accepting it would fit a smaller sample than the caller allowed.
            continue
        scale, shape = _gpd_mle(exceedance)
        thresholds.append(value)
        shapes.append(shape)
        scales.append(scale)
        # Smith (1985) asymptotic standard error of the shape.
        errors.append((1.0 + shape) / np.sqrt(exceedance.size))
        counts.append(int(exceedance.size))
    if not thresholds:
        raise ValueError(
            f"no candidate threshold leaves {min_exceedances} exceedances; the "
            "statistic is too heavily tied for a peaks-over-threshold fit"
        )
    return {
        "threshold": np.asarray(thresholds, dtype=np.float64),
        "shape": np.asarray(shapes, dtype=np.float64),
        "scale": np.asarray(scales, dtype=np.float64),
        "std_error": np.asarray(errors, dtype=np.float64),
        "n_exceedances": np.asarray(counts, dtype=np.int64),
    }


def choose_threshold(
    stats: np.ndarray,
    min_exceedances: int = 500,
    n_candidates: int = 15,
    n_sigma: float = 2.0,
    method: str = "count",
) -> float:
    """
    Select a POT threshold, by a fixed exceedance count or by shape stability.

    ``method="count"`` is the default and delegates to :func:`threshold_at_count`, which
    is PyCBC's rule; ``n_candidates`` and ``n_sigma`` are then unused. It is the default
    because it is the only one with a counterpart: every PyCBC fit threshold is supplied
    by the caller or set by the count rule, and neither GWTC-4.0 nor GWTC-5.0 Methods says
    anything about selecting one from the data.

    ``method="stability"`` reads :func:`threshold_ladder` by machine: accept the lowest
    candidate whose shape agrees with the shape at every candidate above it,
    ``abs(xi_j - xi_k) <= n_sigma * sqrt(s_j**2 + s_k**2)`` for all ``j`` above ``k``.
    Below the right threshold the fit is pulled by the bulk of the distribution and
    ``xi(u)`` drifts with ``u``; above it ``xi(u)`` is flat and merely noisier, which is
    the signature being tested for. No reference selects a threshold this way -- the
    ladder is published as a plot and read -- so it is available rather than default.

    The lowest accepted candidate is returned rather than the highest: it keeps the most
    exceedances, and every fit above it already agrees with it, so moving up pays
    variance for a shape nobody disputes. The topmost rung is never accepted on its own
    merits: it has nothing above it to agree with, so calling it stable would be vacuous.

    The standard error is the asymptotic ``(1 + xi) / sqrt(n_u)`` of Smith (1985) rather
    than a bootstrap. It is deterministic, which matters more here than accuracy: a
    bootstrap would make the threshold -- and hence every extrapolated rate above it --
    depend on a seed. Where ``xi < -1/2`` the asymptotics do not hold and this
    understates the true error, which tightens the criterion and selects a higher
    threshold; that is the safe direction.

    Parameters
    ----------
    min_exceedances : int
        Fewest exceedances any candidate may leave, and the count itself under
        ``method="count"``.
    n_candidates : int
        Size of the stability ladder, geometrically spaced in exceedance count between
        ``min_exceedances`` and half the sample.
    n_sigma : float
        Width of the agreement test, in combined standard errors.

    Notes
    -----
    When no candidate is accepted, the exceedances are not generalised Pareto anywhere on
    the ladder. The least unstable candidate is returned rather than raising -- the caller
    still needs a threshold, and :func:`fit_tail` reports ``ad_p_value`` next to the fit,
    which is where an inadequate tail model is supposed to be read.
    """
    if method == "count":
        return threshold_at_count(stats, n_exceedances=min_exceedances)
    if method != "stability":
        raise ValueError(
            f"unknown threshold method {method!r}, expected count or stability"
        )
    ladder = threshold_ladder(
        stats, min_exceedances=min_exceedances, n_candidates=n_candidates
    )
    thresholds = ladder["threshold"]
    shapes = ladder["shape"]
    errors = ladder["std_error"]
    if shapes.size == 1:
        return float(thresholds[0])

    # Only rungs with at least one rung above them can be tested; the topmost is
    # vacuously stable and is excluded rather than accepted by default.
    testable = shapes.size - 1
    worst = np.zeros(testable)
    for k in range(testable):
        above = slice(k + 1, None)
        combined = np.sqrt(errors[k] ** 2 + errors[above] ** 2)
        # A shape at the floor has a vanishing asymptotic error; the guard keeps the
        # comparison finite instead of turning it into a division by zero.
        worst[k] = np.max(
            np.abs(shapes[above] - shapes[k]) / np.maximum(combined, 1e-12)
        )
    accepted = np.flatnonzero(worst <= n_sigma)
    index = int(accepted[0]) if accepted.size else int(np.argmin(worst))
    return float(thresholds[index])


def fit_tail(
    stats: np.ndarray,
    threshold: Optional[float] = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    min_exceedances: int = 500,
    n_null: int = 1000,
) -> TailFit:
    """
    Fit a generalised Pareto tail by maximum likelihood with a bootstrap covariance.

    The reported ``scale`` and ``shape`` are the two-parameter maximum likelihood
    estimate, always. Nothing here selects a model: the fit is done once, so the FAR
    layer and the p_astro noise density read the same two numbers and cannot describe
    the same background differently.

    ``exponential_lrt`` and ``anderson_darling`` are run and reported, and neither
    changes the fit. The first says whether an exponential would have described the
    exceedances as well; the second whether any generalised Pareto describes them at all.
    A small ``ad_p_value`` is the one that should stop a tail being quoted, and it is a
    statement about the fit rather than a reason to substitute a different one.

    ``covariance`` is the bootstrap covariance of the two-parameter fit, so the shape's
    own uncertainty travels with the point estimate into the extrapolated region, which
    is the only place the band is read.

    Parameters
    ----------
    threshold : float, optional
        POT threshold. Chosen by :func:`choose_threshold` when omitted.
    n_bootstrap : int
        Resamples behind ``covariance``. The excesses are resampled with replacement at
        a fixed exceedance count; resampling the whole statistic array and
        re-thresholding instead would fold in the Poisson variation of how many
        exceedances the campaign happened to collect, which is a property of the
        livetime rather than of the tail shape, and which the FAR layer already carries
        in the measured rate it anchors the tail to.
    seed : int
        Seeds everything random here. The bootstrap and the two null distributions take
        independent substreams derived from it, so no two of them share draws and the
        whole fit reproduces from this one number.
    min_exceedances : int
        Passed to :func:`choose_threshold` when ``threshold`` is omitted.
    n_null : int
        Monte Carlo replicates for each of the two null distributions.
    Notes
    -----
    Cost is ``1 + n_bootstrap + 2 * n_null`` maximum-likelihood fits, each of which is a
    one-dimensional search over the profile likelihood costing a few passes over the
    exceedances. At the defaults and a few thousand exceedances that is seconds, once
    per campaign per removal mode. Reducing ``n_null`` is what makes it cheaper, at the
    cost of resolution in the two p-values, which cannot fall below ``1 / (1 + n_null)``.
    """
    stats = _finite_stats(stats)
    if n_bootstrap < 2:
        raise ValueError(
            f"n_bootstrap must be at least 2 to form a covariance, got {n_bootstrap}"
        )
    if threshold is None:
        threshold = choose_threshold(stats, min_exceedances=min_exceedances)
    threshold = float(threshold)
    exceedance = _excesses(stats, threshold)

    scale, shape = _gpd_mle(exceedance)
    boot_seed, lrt_seed, ad_seed = _substream_seeds(seed, 3)
    _, lrt_p = exponential_lrt(stats, threshold, n_null=n_null, seed=lrt_seed)
    _, ad_p = anderson_darling(stats, threshold, n_null=n_null, seed=ad_seed)

    rng = np.random.default_rng(boot_seed)
    n = exceedance.size
    replicates = np.empty((n_bootstrap, 2), dtype=np.float64)
    for k in range(n_bootstrap):
        replicates[k] = _gpd_mle(exceedance[rng.integers(0, n, size=n)])
    covariance = np.cov(replicates, rowvar=False, ddof=1)

    return TailFit(
        threshold=threshold,
        scale=scale,
        shape=shape,
        covariance=covariance,
        n_exceedances=n,
        lrt_p_value=lrt_p,
        ad_p_value=ad_p,
    )


def exponential_lrt(
    stats: np.ndarray, threshold: float, n_null: int = 1000, seed: int = 0
) -> Tuple[float, float]:
    """
    Likelihood-ratio test of shape == 0; returns ``(statistic, p_value)``.

    The statistic is ``2 * (loglik_gpd - loglik_exponential)`` at the two maxima, with
    the exponential nested in the generalised Pareto at ``shape == 0``. A large value is
    evidence of a genuine shape; a large p-value means the exponential describes the
    exceedances as well as the two-parameter fit does.

    **Reported, never acted on.** :func:`fit_tail` has no exponential branch and does not
    read this: a large p-value is a failure to demonstrate a shape, which is not a
    demonstration that the shape is zero, and substituting one for the other sets ``xi``
    to a value the data did not supply. An earlier revision switched branches here and
    the switch was removed.

    The null distribution is simulated rather than taken from chi-squared with one
    degree of freedom. The generalised Pareto likelihood is irregular exactly where this
    test is read: the density is smooth in the shape only after a removable
    zero-over-zero at ``shape == 0``, the Fisher information exists only above
    ``-1/2``, and the maximum itself exists only above ``-1``. Convergence to the
    chi-squared limit is slow, and it errs towards **over-rejection** -- at five hundred
    exceedances the chi-squared form rejects a true exponential near eight per cent of
    the time at a nominal five. Each of those rejections is a heavy tail claimed where
    there is none, which is what inflates an extrapolated FAR.

    Simulation is exact here rather than merely better. The statistic is invariant under
    rescaling of the excesses, because both maxima are scale equivariant, so its null
    distribution depends on nothing but the number of exceedances; unit-scale
    exponential draws are draws from the true null, with no nuisance parameter left to
    plug in. The p-value is ``(1 + #{D_sim >= D_obs}) / (1 + n_null)``, the Monte Carlo
    form that is valid at any finite ``n_null`` instead of only in the limit, and which
    can therefore never report zero.
    """
    stats = _finite_stats(stats)
    if n_null < 1:
        raise ValueError(f"n_null must be at least 1, got {n_null}")
    exceedance = _excesses(stats, float(threshold))
    statistic = _lrt_statistic(exceedance)
    null = _lrt_null(exceedance.size, int(n_null), int(seed))
    p_value = (1.0 + float(np.count_nonzero(null >= statistic))) / (1.0 + n_null)
    return float(statistic), float(p_value)


def anderson_darling(
    stats: np.ndarray, threshold: float, n_null: int = 1000, seed: int = 0
) -> Tuple[float, float]:
    """
    Anderson-Darling goodness of fit of the tail; returns ``(statistic, p_value)``.

    Tests whether any generalised Pareto describes the exceedances, which is a separate
    question from which branch of it :func:`fit_tail` selected. Anderson-Darling rather
    than Kolmogorov-Smirnov because its weight ``1 / (u * (1 - u))`` puts the sensitivity
    at the ends of the distribution: the far tail is where a POT fit is used and where
    the Kolmogorov-Smirnov statistic is least sensitive.

    The null distribution is a parametric bootstrap at the fitted parameters, not the
    published tables. The tables are for known parameters; with both estimated from the
    same exceedances the statistic is stochastically smaller, and reading a table would
    accept tails that do not fit. Scale invariance again makes the simulation exact up
    to Monte Carlo error given the fitted shape, and the p-value uses the same
    ``(1 + count) / (1 + n_null)`` form as :func:`exponential_lrt`.
    """
    stats = _finite_stats(stats)
    if n_null < 1:
        raise ValueError(f"n_null must be at least 1, got {n_null}")
    exceedance = _excesses(stats, float(threshold))
    scale, shape = _gpd_mle(exceedance)
    statistic = _ad_statistic(exceedance, scale, shape)
    null = _ad_null(exceedance.size, int(n_null), float(shape), int(seed))
    p_value = (1.0 + float(np.count_nonzero(null >= statistic))) / (1.0 + n_null)
    return float(statistic), float(p_value)


def ks_test(
    stats: np.ndarray,
    threshold: float,
    n_null: int = 1000,
    seed: int = 0,
    bootstrap: bool = True,
) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov goodness of fit of the tail; returns ``(statistic, p_value)``.

    Provided for comparison with PyCBC, whose
    ``pycbc.events.trigger_fits.KS_test(distr, vals, alpha, thresh)`` compares the
    exceedances against the fitted cumulative through ``scipy.stats.kstest``.
    :func:`anderson_darling` is the production test: its weight ``1 / (u * (1 - u))``
    puts the sensitivity at the ends of the distribution, and the far tail is both where
    a peaks-over-threshold fit is used and where the Kolmogorov-Smirnov statistic is
    least sensitive. The two are reported side by side so that claim is measured on the
    real background rather than assumed.

    Parameters
    ----------
    bootstrap : bool
        ``True`` simulates the null by refitting each replicate, as
        :func:`anderson_darling` does, which is the correct treatment when both
        parameters were estimated from the same exceedances. ``False`` returns
        ``scipy.stats.kstest``'s tabulated p-value, which is what PyCBC reports; that
        form assumes known parameters and is therefore anti-conservative here -- it
        accepts tails that do not fit. Both are available so the size of that effect can
        be seen rather than argued about.
    """
    from scipy.stats import genpareto, kstest

    stats = _finite_stats(stats)
    if n_null < 1:
        raise ValueError(f"n_null must be at least 1, got {n_null}")
    exceedance = _excesses(stats, float(threshold))
    scale, shape = _gpd_mle(exceedance)
    statistic = float(
        kstest(
            exceedance, lambda x: genpareto.cdf(x, shape, loc=0.0, scale=scale)
        ).statistic
    )
    if not bootstrap:
        return statistic, float(
            kstest(
                exceedance, lambda x: genpareto.cdf(x, shape, loc=0.0, scale=scale)
            ).pvalue
        )
    null = _ks_null(exceedance.size, int(n_null), float(shape), int(seed))
    p_value = (1.0 + float(np.count_nonzero(null >= statistic))) / (1.0 + n_null)
    return statistic, float(p_value)


@lru_cache(maxsize=16)
def _ks_null(n: int, n_null: int, shape: float, seed: int) -> np.ndarray:
    """
    Null distribution of the KS statistic with both parameters estimated.

    Refit per replicate, for the same reason :func:`_ad_null` is: it is the refitting that
    makes the statistic stochastically smaller than the known-parameter tables PyCBC reads
    it against.
    """
    from scipy.stats import genpareto, kstest

    rng = np.random.default_rng(seed)
    null = np.empty(n_null, dtype=np.float64)
    for k in range(n_null):
        sample = genpareto.rvs(shape, loc=0.0, scale=1.0, size=n, random_state=rng)
        fitted_scale, fitted_shape = _gpd_mle(sample)
        null[k] = kstest(
            sample,
            lambda x: genpareto.cdf(x, fitted_shape, loc=0.0, scale=fitted_scale),
        ).statistic
    null.setflags(write=False)
    return null


def _finite_stats(
    stats: np.ndarray, name: str = "stats", ravel: bool = True
) -> np.ndarray:
    """
    Coerce to float64 and refuse anything that is not a finite number.

    A NaN ranking statistic means the network produced nothing usable for that window,
    which is a fault to report rather than a value to fit; it also compares false against
    everything, so it would vanish from the exceedance count without being counted
    anywhere. Infinities are refused for the same reason a finite endpoint matters: an
    infinite excess drives the profile likelihood to its boundary and the fit reports
    that one value rather than the tail.
    """
    values = np.asarray(stats, dtype=np.float64)
    if ravel:
        values = values.ravel()
    if not np.isfinite(values).all():
        bad = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(
            f"{bad} of {values.size} values in {name} are not finite; a tail cannot be "
            "fitted to, or evaluated at, a value that is not a number"
        )
    return values


def _excesses(stats: np.ndarray, threshold: float) -> np.ndarray:
    """
    Exceedances strictly above a threshold, shifted to start at zero.

    Strictly above, so the threshold itself is not an exceedance. With a heavily tied
    statistic the alternative would place a run of zero excesses at the origin, which no
    continuous density can carry, and the maximum likelihood fit answers by driving the
    scale toward zero.
    """
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")
    exceedance = stats[stats > threshold] - threshold
    if exceedance.size < MIN_FIT_EXCEEDANCES:
        raise ValueError(
            f"{exceedance.size} statistics lie above a threshold of {threshold}, fewer "
            f"than the {MIN_FIT_EXCEEDANCES} a two-parameter tail needs; lower the "
            "threshold or accumulate more background"
        )
    if exceedance.max() <= exceedance.min():
        raise ValueError(
            "every exceedance is identical, so the likelihood is unbounded as the scale "
            "goes to zero and there is no fit to report"
        )
    return exceedance


def _substream_seeds(seed: int, count: int) -> Tuple[int, ...]:
    """
    Independent integer seeds derived from one, for stages that must not share draws.

    Spawned through ``SeedSequence`` rather than taken as ``seed``, ``seed + 1``, ...:
    consecutive seeds are a habit that happens to be safe for this generator and is not
    safe in general, and the two p-values reported side by side are supposed to be
    independent given the data.
    """
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in np.random.SeedSequence(int(seed)).spawn(int(count))
    )


def _gpd_loglik(excess: np.ndarray, scale: float, shape: float) -> float:
    """
    Generalised Pareto log-likelihood of excesses over zero.

    Written out rather than summed from ``genpareto.logpdf`` because it is the inner
    loop of every fit here, and because the ``shape == 0`` limit has to be taken exactly
    rather than by a division that is only nearly zero. Returns ``-inf`` outside the
    support, which is what makes the bracketed search treat the boundary as unreachable.
    """
    n = excess.size
    if scale <= 0.0:
        return -np.inf
    if shape == 0.0:
        return -n * np.log(scale) - float(excess.sum()) / scale
    scaled = shape * excess / scale
    if np.min(scaled) <= -1.0:
        return -np.inf
    return -n * np.log(scale) - (1.0 + 1.0 / shape) * float(np.log1p(scaled).sum())


def _shape_at(theta: float, excess: np.ndarray) -> float:
    """
    Shape maximising the likelihood at fixed ``theta = shape / scale``.

    The closed form ``mean(log(1 + theta * y))`` is what reduces the two-dimensional fit
    to one dimension. Monotone increasing in ``theta``, which is what lets the search
    bracket be found by bisection.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.mean(np.log1p(theta * excess)))


def _profile_loglik(theta: float, excess: np.ndarray) -> float:
    """
    Profile log-likelihood at ``theta = shape / scale``, the one-dimensional objective.

    Substituting the closed-form shape leaves ``-n * (log(shape / theta) + shape + 1)``.
    The removable singularity at ``theta == 0`` is the exponential fit, taken exactly
    rather than approached.
    """
    n = excess.size
    if theta == 0.0:
        return -n * (1.0 + np.log(float(excess.mean())))
    shape = _shape_at(theta, excess)
    if not np.isfinite(shape) or shape / theta <= 0.0:
        return -np.inf
    return -n * (np.log(shape / theta) + shape + 1.0)


def _profile_loglik_grid(thetas: np.ndarray, excess: np.ndarray) -> np.ndarray:
    """Profile log-likelihood over a ladder of ``theta``, in one pass over the data."""
    n = excess.size
    with np.errstate(invalid="ignore", divide="ignore"):
        shapes = np.log1p(np.outer(thetas, excess)).mean(axis=1)
        values = -n * (np.log(shapes / thetas) + shapes + 1.0)
    values[thetas == 0.0] = -n * (1.0 + np.log(float(excess.mean())))
    return np.where(np.isfinite(values), values, -np.inf)


def _theta_at_shape(target: float, excess: np.ndarray) -> float:
    """
    Invert the monotone ``theta -> shape`` map, for the ends of the search bracket.

    Bisection on a monotone function, returning the side that satisfies the bound, so a
    bracket built from it can never contain a shape outside ``[SHAPE_FLOOR,
    SHAPE_CEIL]``. When the target is unreachable -- with many exceedances the single
    largest one cannot pull the mean of the logs below the floor -- the edge of the
    support is returned instead, and the bracket is then the whole domain.
    """
    largest = float(excess.max())
    if target < 0.0:
        # The support ends at theta = -1 / max(y); a step back from it keeps the log
        # finite while getting as close to the boundary as float64 allows.
        low = -(1.0 - 2.0**-40) / largest
        if _shape_at(low, excess) > target:
            return low
        high = 0.0
    else:
        low = 0.0
        # log1p(x) > log(x), so this theta gives a shape above the target for certain,
        # which makes the doubling search that would otherwise be needed unnecessary.
        high = float(np.exp(target - np.mean(np.log(excess))))
    for _ in range(60):
        middle = 0.5 * (low + high)
        if _shape_at(middle, excess) < target:
            low = middle
        else:
            high = middle
    return high


def _gpd_mle(excess: np.ndarray) -> Tuple[float, float]:
    """
    Maximum-likelihood ``(scale, shape)`` for excesses over zero.

    The two-dimensional likelihood is reduced to one dimension by ``theta = shape /
    scale``: at fixed ``theta`` the maximising shape is a closed-form mean of logs, and
    what is left is a smooth profile in a single parameter. Used in place of
    ``scipy.stats.genpareto.fit``, which is a generic two-parameter optimisation from a
    moment start: several times slower here, and it returns shapes below ``SHAPE_FLOOR``
    where the likelihood is unbounded rather than maximised, which is not a fit but the
    largest excess wearing one.

    The search is a coarse ladder over ``theta`` on both sides of zero followed by a
    bounded refinement in the neighbouring interval. The profile can carry two
    stationary points, so a local search from a single start can converge to the wrong
    one; the ladder costs a single vectorised pass over the exceedances and removes that
    failure. Both the ladder and the refinement tolerance are built from the data, so
    the whole estimator is scale equivariant, which is what makes the null distributions
    in this module depend on the exceedance count alone.
    """
    from scipy.optimize import minimize_scalar

    excess = np.asarray(excess, dtype=np.float64)
    if excess.size < 2:
        raise ValueError("a generalised Pareto fit needs at least two exceedances")
    if excess.min() <= 0.0:
        raise ValueError("exceedances must be strictly positive")
    if excess.max() <= excess.min():
        raise ValueError(
            "every exceedance is identical, so the likelihood is unbounded as the "
            "scale goes to zero"
        )
    low = _theta_at_shape(SHAPE_FLOOR, excess)
    high = _theta_at_shape(SHAPE_CEIL, excess)
    ladder = np.concatenate(
        [
            -np.geomspace(-low, -low * 1e-6, 33),
            [0.0],
            np.geomspace(high * 1e-6, high, 33),
        ]
    )
    values = _profile_loglik_grid(ladder, excess)
    best = int(np.argmax(values))
    left = ladder[max(best - 1, 0)]
    right = ladder[min(best + 1, ladder.size - 1)]
    theta = ladder[best]
    if right > left:
        result = minimize_scalar(
            lambda t: -_profile_loglik(t, excess),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-12 * max(abs(left), abs(right))},
        )
        if -float(result.fun) > values[best]:
            theta = float(result.x)
    if theta == 0.0:
        return float(excess.mean()), 0.0
    shape = _shape_at(theta, excess)
    return float(shape / theta), float(shape)


def _lrt_statistic(excess: np.ndarray) -> float:
    """
    Twice the log-likelihood gain of the generalised Pareto over the exponential.

    Floored at zero. The exponential is nested at ``shape == 0``, so the wider maximum
    cannot be the smaller one and a negative value is the one-dimensional search
    stopping short, not evidence for the null.
    """
    scale, shape = _gpd_mle(excess)
    n = excess.size
    exponential = -n * (1.0 + np.log(float(excess.mean())))
    return max(2.0 * (_gpd_loglik(excess, scale, shape) - exponential), 0.0)


@lru_cache(maxsize=16)
def _lrt_null(n: int, n_null: int, seed: int) -> np.ndarray:
    """
    Null distribution of the exponential likelihood-ratio statistic at ``n`` exceedances.

    Cached and returned read-only: the statistic is scale invariant, so this depends on
    nothing but the exceedance count, and a campaign that fits several removal modes or
    several thresholds of the same depth would otherwise simulate the same distribution
    again each time.
    """
    rng = np.random.default_rng(seed)
    null = np.empty(n_null, dtype=np.float64)
    for k in range(n_null):
        null[k] = _lrt_statistic(rng.exponential(1.0, size=n))
    null.setflags(write=False)
    return null


def _ad_statistic(excess: np.ndarray, scale: float, shape: float) -> float:
    """
    Anderson-Darling statistic of exceedances against a fitted generalised Pareto.

    ``-n - sum((2i - 1) * (log(u_i) + log(1 - u_(n+1-i)))) / n`` on the sorted
    probability integral transform. The transform is clipped away from zero and one only
    where float64 has already rounded it there; the statistic is otherwise infinite for
    a single point at the fitted endpoint, which is a rounding artefact rather than
    infinitely strong evidence.
    """
    from scipy.stats import genpareto

    n = excess.size
    u = np.sort(genpareto.cdf(excess, shape, loc=0.0, scale=scale))
    u = np.clip(u, 1e-300, 1.0 - np.finfo(np.float64).epsneg)
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(
        -n - np.sum((2.0 * index - 1.0) * (np.log(u) + np.log1p(-u[::-1]))) / n
    )


@lru_cache(maxsize=16)
def _ad_null(n: int, n_null: int, shape: float, seed: int) -> np.ndarray:
    """
    Null distribution of the Anderson-Darling statistic with both parameters estimated.

    Simulated at the fitted shape and unit scale; the statistic is scale invariant, so
    the scale the tail was fitted at does not enter. Each replicate is refitted, which
    is the whole point: it is the refitting that makes the statistic smaller than the
    known-parameter tables it would otherwise be read against.
    """
    from scipy.stats import genpareto

    rng = np.random.default_rng(seed)
    null = np.empty(n_null, dtype=np.float64)
    for k in range(n_null):
        sample = genpareto.rvs(shape, loc=0.0, scale=1.0, size=n, random_state=rng)
        fitted_scale, fitted_shape = _gpd_mle(sample)
        null[k] = _ad_statistic(sample, fitted_scale, fitted_shape)
    null.setflags(write=False)
    return null


def _log_survival_gradient(
    excess: np.ndarray, scale: float, shape: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient of the log survival with respect to ``(scale, shape)``.

    ``d/dscale = y / (scale**2 * (1 + w))`` and ``d/dshape = (y / scale)**2 * g(w)``
    with ``w = shape * y / scale`` and ``g(w) = (log1p(w) - w / (1 + w)) / w**2``.

    ``g`` is finite at ``w == 0``, where it is one half, but its two terms agree to
    their own leading order there and cancel; the series is used below ``abs(w) = 1e-3``
    so that a nearly exponential fit -- the branch this module deliberately prefers --
    does not get a band built from a cancelled difference.
    """
    excess = np.asarray(excess, dtype=np.float64)
    ratio = excess / scale
    w = shape * ratio
    d_scale = ratio / (scale * (1.0 + w))

    g = np.empty(np.shape(w), dtype=np.float64)
    small = np.abs(w) < 1e-3
    # g(w) = 1/2 - 2w/3 + 3w**2/4 - 4w**3/5 + 5w**4/6 - ...
    near = w[small]
    g[small] = 0.5 + near * (
        -2.0 / 3.0 + near * (0.75 + near * (-0.8 + near * (5.0 / 6.0)))
    )
    far = w[~small]
    g[~small] = (np.log1p(far) - far / (1.0 + far)) / far**2
    return d_scale, ratio**2 * g
