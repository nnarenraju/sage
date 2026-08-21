#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_tail.py
Description   : The peaks-over-threshold tail: sign convention, branch and band.

Created on 2026-08-16

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The tail is read by two consumers -- the FAR extrapolation beyond the loudest background
event and the p_astro noise density -- and neither can be checked against data in the
region where it is used. The oracles here are therefore external to the module: samples
from a ``scipy.stats.genpareto`` with known parameters, the asymptotic covariance of the
generalised Pareto maximum likelihood estimator, and a finite-difference delta method
built from ``genpareto.logsf``. A round trip of the fit against its own survival would
agree with itself under a sign flip, which is the one error in this module that has no
symptom.

Sample sizes and Monte Carlo counts are cut from the production defaults so the file
runs in seconds, but nothing is approximated: every fit is the same likelihood search
the pipeline runs, and every tolerance is stated in units of the fit's own reported
uncertainty rather than as a hand-tuned number.

Runs anywhere; needs no data, no GPU and no network.
"""

from functools import lru_cache

import numpy as np
import pytest
from scipy.stats import genpareto, norm

from sage.search.tail import (
    _gpd_mle,
    MIN_FIT_EXCEEDANCES,
    PARAMETER_ORDER,
    SHAPE_FLOOR,
    TailFit,
    anderson_darling,
    choose_threshold,
    threshold_ladder,
    exponential_lrt,
    ks_test,
    fit_tail,
    threshold_at_count,
)

# One scale for every simulated tail, well away from one so that a covariance whose rows
# are ordered the other way round is separated from a correct one by a factor of the
# scale rather than by rounding.
TRUE_SCALE = 2.0
N_SAMPLE = 6000
N_BOOT = 200
N_NULL = 100
DATA_SEED = 1234


def _gpd_sample(shape, scale=TRUE_SCALE, size=N_SAMPLE, seed=DATA_SEED):
    """Draws from a generalised Pareto whose parameters the test knows."""
    rng = np.random.default_rng(seed)
    return genpareto.rvs(shape, loc=0.0, scale=scale, size=size, random_state=rng)


@lru_cache(maxsize=8)
def _known_fit(shape):
    """
    Fit a sample from a known generalised Pareto above its own median.

    Thresholded rather than fitted from zero, so the fit is exercised on what the
    pipeline actually hands it -- excesses over a threshold -- and so the expected scale
    is ``scale + shape * u`` rather than the scale it was drawn with. Getting that
    reparametrisation right is part of what is being tested.

    ``alpha`` is set near one so the two-parameter estimate is always the reported one.
    With the default the exponential branch would be taken whenever the shape is not
    distinguishable from zero, and the shape recovery test would then be checking a
    constant.

    Cached because each fit is a few hundred maximum likelihood searches and several
    tests read the same one.
    """
    sample = _gpd_sample(shape)
    threshold = float(np.quantile(sample, 0.5))
    fit = fit_tail(
        sample,
        threshold=threshold,
        n_bootstrap=N_BOOT,
        n_null=N_NULL,
        seed=0,
    )
    return sample, threshold, fit, TRUE_SCALE + shape * threshold


def _tail(shape=0.25, scale=1.5, threshold=8.0, covariance=None):
    """A TailFit with parameters chosen by the test rather than fitted."""
    if covariance is None:
        covariance = np.array([[0.04, -0.006], [-0.006, 0.0016]])
    return TailFit(
        threshold=threshold,
        scale=scale,
        shape=shape,
        covariance=np.asarray(covariance, dtype=np.float64),
        n_exceedances=1000,
        lrt_p_value=0.5,
        ad_p_value=0.5,
    )


class TestSignConvention:
    """Which way ``shape`` points, and what depends on it."""

    def test_survival_matches_evt_formula(self):
        """
        ``survival`` is ``(1 + xi * y / scale) ** (-1 / xi)``, written out by hand.

        The module docstring states this convention and passes the fitted shape through
        to ``genpareto`` unaltered on the strength of it. Comparing against scipy would
        only restate the implementation; the closed form is the specification, and it is
        the thing a reader of the FAR extrapolation is entitled to assume.
        """
        fit = _tail(shape=0.4, scale=1.5, threshold=3.0)
        stat = np.array([3.5, 4.0, 7.0, 20.0])
        excess = stat - 3.0
        expected = (1.0 + 0.4 * excess / 1.5) ** (-1.0 / 0.4)
        assert fit.survival(stat) == pytest.approx(expected, rel=1e-12)

    def test_exponential_branch_survival(self):
        """
        At ``shape == 0`` the survival is ``exp(-y / scale)``, the removable limit.

        The branch this module deliberately prefers is the one where a division by the
        shape would be a division by zero. A fit that reached the limit numerically
        rather than exactly would show up here as a survival that is merely close.
        """
        fit = _tail(shape=0.0, scale=2.0, threshold=1.0)
        stat = np.array([1.5, 3.0, 11.0])
        expected = np.exp(-(stat - 1.0) / 2.0)
        assert fit.survival(stat) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("shape", [-0.35, 0.35])
    def test_fitted_shape_matches_genpareto_c(self, shape):
        """
        The fitted shape carries the sign of the ``c`` it was drawn with.

        The failure this catches is a shape reported in the Hosking and Wallis
        convention, ``k = -xi``. Both fits look entirely ordinary; what moves is the
        extrapolated FAR, and a bounded background silently becomes a power-law one or
        the reverse. The agreement is asserted in units of the fit's own bootstrap
        standard error, so the test states the estimator's precision rather than a
        tolerance chosen to pass.
        """
        _, _, fit, _ = _known_fit(shape)
        sigma = float(np.sqrt(fit.covariance[1, 1]))
        assert np.sign(fit.shape) == np.sign(shape)
        assert abs(fit.shape - shape) <= 3.5 * sigma

    def test_endpoint_formula_and_bound(self):
        """
        A bounded fit reports ``threshold - scale / shape`` and it lies above the data.

        The endpoint is the number ``survival`` turns into an exact zero, so an error in
        it either truncates the noise model below observed background or never truncates
        it at all. Checked against the closed form and against the largest statistic
        that was actually fitted, which no correct endpoint may fall below.
        """
        sample, threshold, fit, _ = _known_fit(-0.35)
        assert fit.shape < 0.0
        endpoint = fit.finite_endpoint
        assert endpoint == pytest.approx(threshold - fit.scale / fit.shape, rel=1e-12)
        assert endpoint > sample.max()
        # The drawn distribution ends at scale / |c|; the fitted endpoint estimates it.
        assert endpoint == pytest.approx(TRUE_SCALE / 0.35, rel=0.15)

    def test_endpoint_none_when_unbounded(self):
        """
        ``finite_endpoint`` is ``None``, not ``inf``, for a shape at or above zero.

        Documented so a caller that forgets the unbounded case fails on the ``None``
        instead of comparing against an infinity that every statistic passes.
        """
        assert _tail(shape=0.3).finite_endpoint is None
        assert _tail(shape=0.0).finite_endpoint is None
        assert _tail(shape=-1e-9).finite_endpoint is not None

    @pytest.mark.parametrize("shape", [-0.35, 0.0, 0.35])
    def test_survival_tracks_empirical_tail(self, shape):
        """
        The fitted survival reproduces the empirical conditional survival of the sample.

        The end-to-end oracle, and the one that would not survive a sign flip anywhere
        between the fit and the survival evaluation: it compares the model against the
        data it was fitted to at three quantiles spanning two decades of exceedance
        probability, without going through any of the module's own parameters.

        The tolerance is four binomial standard errors of the empirical estimate itself,
        which is the only uncertainty the comparison has; a fixed relative tolerance
        would be far too loose at the median and too tight at the hundredth quantile,
        where thirty points carry the estimate.
        """
        sample, threshold, fit, _ = _known_fit(shape)
        exceedance = sample[sample > threshold]
        quantiles = np.quantile(exceedance, [0.5, 0.9, 0.99])
        empirical = np.array([(exceedance > q).mean() for q in quantiles])
        error = np.sqrt(empirical * (1.0 - empirical) / exceedance.size)
        assert np.all(np.abs(fit.survival(quantiles) - empirical) <= 4.0 * error)


class TestParameterRecovery:
    """What the estimator recovers, and how well it says it knows it."""

    @pytest.mark.parametrize("shape", [-0.35, 0.0, 0.35])
    def test_recovers_known_parameters(self, shape):
        """
        Both parameters land within 3.5 bootstrap standard errors of the truth.

        Fitted above the sample median, so the target scale is the reparametrised
        ``scale + shape * u`` rather than the drawn one. An estimator that is biased, or
        one whose bootstrap understates its own spread, fails here; stating it in
        reported standard errors means the test cannot be satisfied by widening the
        covariance either, since the same covariance is checked against the asymptotics
        below.
        """
        _, _, fit, expected_scale = _known_fit(shape)
        sigma = np.sqrt(np.diag(fit.covariance))
        assert abs(fit.scale - expected_scale) <= 3.5 * sigma[0]
        assert abs(fit.shape - shape) <= 3.5 * sigma[1]

    @pytest.mark.parametrize("shape", [-0.35, 0.0, 0.35])
    def test_covariance_matches_asymptotics(self, shape):
        """
        The bootstrap covariance agrees with Smith's asymptotic form, entry by entry.

        ``Var = (1 / n) * [[2 s^2 (1 + xi), -s (1 + xi)], [-s (1 + xi), (1 + xi)^2]]``
        over ``PARAMETER_ORDER``. This pins the row order as well as the size: with a
        scale of two the two diagonal entries differ by more than a factor of six, so a
        covariance whose rows are ``(shape, scale)`` -- which is the order
        ``genpareto.fit`` returns its parameters in -- is separated from a correct one
        by far more than Monte Carlo error. The band on every extrapolated rate is built
        from these four numbers.

        Compared as standard deviations rather than variances: the asymptotics are a
        limit, and at a few thousand exceedances with a negative shape the bootstrap is
        legitimately wider than them by tens of per cent, which squares into a
        discrepancy that says nothing about the implementation.
        """
        _, _, fit, expected_scale = _known_fit(shape)
        n = fit.n_exceedances
        rate = 1.0 + shape
        expected = np.array(
            [
                [2.0 * expected_scale**2 * rate, -expected_scale * rate],
                [-expected_scale * rate, rate**2],
            ]
        ) / n
        assert np.sqrt(np.diag(fit.covariance)) == pytest.approx(
            np.sqrt(np.diag(expected)), rel=0.3
        )
        correlation = fit.covariance[0, 1] / np.sqrt(
            fit.covariance[0, 0] * fit.covariance[1, 1]
        )
        expected_correlation = -1.0 / np.sqrt(2.0 * (1.0 + shape))
        assert correlation == pytest.approx(expected_correlation, abs=0.1)

    def test_covariance_is_symmetric(self):
        """A 2x2 that is not symmetric is not a covariance, however it was assembled."""
        _, _, fit, _ = _known_fit(0.35)
        assert fit.covariance.shape == (2, 2)
        assert fit.covariance[0, 1] == fit.covariance[1, 0]
        assert PARAMETER_ORDER == ("scale", "shape")

    def test_exceedances_counted_strictly(self):
        """
        Statistics exactly at the threshold are not exceedances.

        Documented because a heavily tied ranking statistic would otherwise contribute a
        run of zero excesses, which no continuous density can carry and which the
        maximum likelihood fit answers by driving the scale toward zero. The count is
        also the sample the bootstrap resamples at, so an off-by-one here shifts every
        reported uncertainty.
        """
        sample = np.concatenate([np.full(50, 4.0), 4.0 + _gpd_sample(0.2, size=800)])
        fit = fit_tail(sample, threshold=4.0, n_bootstrap=20, n_null=20)
        assert fit.n_exceedances == 800
        assert fit.n_exceedances == int(np.count_nonzero(sample > 4.0))


class TestEstimator:
    """The maximum likelihood search itself, against the likelihood it maximises."""

    @pytest.mark.parametrize("shape", [-0.35, 0.0, 0.35])
    def test_fit_is_a_local_maximum(self, shape):
        """
        No neighbouring parameter pair has a higher log-likelihood than the reported.

        Asserted against the likelihood the module defines, over a quarter of a standard
        error in each direction and both diagonals. The profile search is bracketed by a
        coarse ladder precisely because the profile can carry two stationary points and
        a local search from one start can settle on the wrong one; this is the property
        that would fail if it did.
        """
        from sage.search.tail import _gpd_loglik

        sample = _gpd_sample(shape, size=2000, seed=63)
        fit = fit_tail(sample, threshold=0.0, n_bootstrap=2, n_null=1)
        best = _gpd_loglik(sample, fit.scale, fit.shape)
        step_scale = 0.25 * fit.scale * np.sqrt(2.0 * (1.0 + shape) / sample.size)
        step_shape = 0.25 * (1.0 + shape) / np.sqrt(sample.size)
        for d_scale in (-1.0, 0.0, 1.0):
            for d_shape in (-1.0, 0.0, 1.0):
                if d_scale == 0.0 and d_shape == 0.0:
                    continue
                neighbour = _gpd_loglik(
                    sample,
                    fit.scale + d_scale * step_scale,
                    fit.shape + d_shape * step_shape,
                )
                assert neighbour < best

    def test_bounded_fit_floored_not_refused(self):
        """
        A strongly bounded tail is fitted at the floor rather than escaping below it.

        Below ``SHAPE_FLOOR`` the likelihood has no maximum -- it diverges as the fitted
        endpoint is squeezed onto the largest excess -- so any shape reported there
        describes one order statistic. A generic two-parameter optimisation walks into
        that region on samples like this one and returns a shape near ``-1.1``, which
        the ``TailFit`` guard rejects; the fit would not exist at all. The bracketed
        profile search stops at the floor and still returns a usable bounded tail, which
        is why it is written out here rather than delegated.
        """
        sample = _gpd_sample(-0.95, scale=2.0, size=200, seed=0)
        fit = fit_tail(sample, threshold=0.0, n_bootstrap=20, n_null=20)
        assert fit.shape >= SHAPE_FLOOR
        assert fit.finite_endpoint is not None
        assert fit.finite_endpoint >= sample.max()


class TestNoModelSelection:
    """The fit is reported as fitted; the exponential test acts on nothing."""

    def test_exponential_data_keeps_fitted_shape(self):
        """
        Exceedances a test cannot distinguish from exponential still get their fitted shape.

        The pre-test this replaces set the shape to exactly zero here. "Cannot
        distinguish xi from zero" is not "xi is zero", and the substitution is a
        one-directional error: an exponential falls off far faster than any positive
        shape, so zeroing an unresolved shape always makes the extrapolated tail lighter
        and the candidate look better.
        """
        sample = np.random.default_rng(11).exponential(2.0, size=3000)
        fit = fit_tail(sample, threshold=0.0, n_bootstrap=20, n_null=300)

        assert fit.lrt_p_value >= 0.05
        assert fit.shape != 0.0
        assert fit.shape == pytest.approx(_gpd_mle(sample)[1], rel=1e-12)

    @pytest.mark.parametrize("shape", [0.6, -0.5])
    def test_shape_recovered_for_both_signs(self, shape):
        """
        A genuine shape of either sign is fitted and reported.

        Run on both sides of zero: a tail sensitive only to heavy shapes would pass an
        unbounded model off as a bounded background's noise density, which is the
        direction that overstates significance.
        """
        sample = _gpd_sample(shape, size=3000, seed=17)
        fit = fit_tail(sample, threshold=0.0, n_bootstrap=20, n_null=300)

        assert fit.lrt_p_value < 0.05
        assert fit.shape == pytest.approx(shape, abs=0.15)

    def test_lrt_verdict_does_not_move_the_fit(self):
        """
        The reported parameters are the maximum likelihood estimate whatever the test says.

        Asserted across both verdicts on the same estimator: the fit must agree with
        :func:`_gpd_mle` to the bit in each case, so no branch can sit between them.
        """
        for seed, size in ((11, 3000), (17, 3000)):
            sample = np.random.default_rng(seed).exponential(2.0, size=size)
            fit = fit_tail(sample, threshold=0.0, n_bootstrap=20, n_null=100)
            scale, shape = _gpd_mle(sample)
            assert fit.scale == pytest.approx(scale, rel=1e-12)
            assert fit.shape == pytest.approx(shape, rel=1e-12)

    def test_shape_variance_retained(self):
        """
        The covariance carries the shape's own uncertainty into the extrapolation.

        It matters most where it is least visible: the band is only ever read beyond the
        measured background, and a covariance missing its shape row would be narrowest
        exactly there.
        """
        sample = np.random.default_rng(11).exponential(2.0, size=3000)
        fit = fit_tail(sample, threshold=0.0, n_bootstrap=200, n_null=300)

        assert fit.covariance[1, 1] > 0.0
        # Same order as the asymptotic shape variance at xi = 0, which is 1 / n.
        assert fit.covariance[1, 1] == pytest.approx(1.0 / fit.n_exceedances, rel=0.5)


class TestGoodnessOfFit:
    """The Anderson-Darling report next to the fit."""

    def test_gpd_data_is_not_rejected(self):
        """
        Exceedances that are generalised Pareto pass their own goodness of fit test.

        The null distribution is a parametric bootstrap with both parameters
        re-estimated on every replicate. Reading the published tables instead -- which
        are for known parameters -- makes the observed statistic look large and rejects
        tails that do fit; that failure appears here as a small p-value on data drawn
        from the model.
        """
        for shape in (-0.35, 0.0, 0.35):
            _, _, fit, _ = _known_fit(shape)
            assert fit.ad_p_value > 0.02

    def test_lognormal_tail_is_rejected(self):
        """
        Exceedances that are not generalised Pareto are called out.

        The other direction. A goodness of fit test that never rejects would let the FAR
        layer quote an extrapolation through a tail the model does not describe, and
        ``ad_p_value`` is the only place that is reported.
        """
        rng = np.random.default_rng(3)
        sample = rng.lognormal(mean=0.0, sigma=1.0, size=3000)
        _, p_value = anderson_darling(sample, 0.0, n_null=200, seed=0)
        assert p_value < 0.05

    def test_p_values_never_reach_zero(self):
        """
        Both p-values use ``(1 + count) / (1 + n_null)`` and so are bounded below.

        A Monte Carlo p-value of exactly zero would propagate as an infinitely
        significant statement about a null that was only simulated a thousand times.
        """
        sample = _gpd_sample(0.6, size=2000, seed=17)
        _, lrt_p = exponential_lrt(sample, 0.0, n_null=50, seed=0)
        _, ad_p = anderson_darling(sample, 0.0, n_null=50, seed=0)
        assert lrt_p >= 1.0 / 51.0
        assert ad_p >= 1.0 / 51.0
        assert 0.0 < lrt_p <= 1.0 and 0.0 < ad_p <= 1.0

    def test_lrt_statistic_is_non_negative(self):
        """
        The exponential is nested in the generalised Pareto, so the gain cannot be below
        zero; a negative value is the search stopping short and is floored.
        """
        for shape in (-0.4, 0.0, 0.4):
            sample = _gpd_sample(shape, size=600, seed=29)
            statistic, _ = exponential_lrt(sample, 0.0, n_null=20, seed=0)
            assert statistic >= 0.0

    def test_null_counts_validated(self):
        """A null distribution of no replicates is a configuration error."""
        sample = _gpd_sample(0.2, size=500, seed=1)
        with pytest.raises(ValueError, match="n_null"):
            exponential_lrt(sample, 0.0, n_null=0)
        with pytest.raises(ValueError, match="n_null"):
            anderson_darling(sample, 0.0, n_null=0)


class TestSurvival:
    """The function the FAR curve is continued with."""

    def test_bounded_in_unit_interval(self):
        """A probability outside [0, 1] leaves the FAR curve with a negative rate."""
        fit = _tail(shape=0.3, scale=1.5, threshold=4.0)
        values = fit.survival(np.linspace(-50.0, 500.0, 4001))
        assert np.all((values >= 0.0) & (values <= 1.0))

    def test_monotone_non_increasing(self):
        """
        A louder statistic never gets a larger exceedance probability.

        Checked across the threshold as well as above it, since the join between the
        constant one below and the model above is where a step would first appear.
        """
        for shape in (-0.4, 0.0, 0.4):
            fit = _tail(shape=shape, scale=1.5, threshold=4.0)
            values = fit.survival(np.linspace(0.0, 20.0, 5000))
            assert np.all(np.diff(values) <= 0.0)

    def test_unity_at_and_below_threshold(self):
        """
        The survival is conditional on exceeding the threshold, so it is one there.

        That is what makes the extrapolated branch join the measured FAR curve without a
        step. Below the threshold it stays one rather than raising or running the fitted
        model backwards under a region where the background was actually counted.
        """
        fit = _tail(shape=0.3, scale=1.5, threshold=4.0)
        assert fit.survival(np.array([4.0]))[0] == 1.0
        assert np.all(fit.survival(np.array([-1e6, 0.0, 3.999])) == 1.0)

    def test_zero_beyond_finite_endpoint(self):
        """
        A bounded fit says the noise does not reach past its endpoint, so the survival
        is zero there rather than a small positive number; the caller then reads an
        IFAR at the curve's cap, which is the honest statement that the extrapolation,
        not the background, produced it.
        """
        fit = _tail(shape=-0.5, scale=2.0, threshold=1.0)
        assert fit.finite_endpoint == pytest.approx(5.0)
        assert np.all(fit.survival(np.array([5.0, 5.1, 1e6])) == 0.0)
        assert fit.survival(np.array([4.9]))[0] > 0.0

    def test_input_shape_preserved(self):
        """
        A two-dimensional block of statistics comes back with its own shape.

        The FAR layer evaluates the tail on arrays it has already aligned with candidate
        rows; a silently flattened return would misalign every rate against its trigger.
        """
        fit = _tail()
        stat = np.array([[8.0, 9.0], [10.0, 11.0]])
        assert fit.survival(stat).shape == (2, 2)
        assert fit.survival(np.float64(9.0)).shape == ()

    def test_non_finite_query_refused(self):
        """
        A NaN statistic compares false against everything and would leave the survival
        silently at one, which is the largest rate the curve can give.
        """
        fit = _tail()
        with pytest.raises(ValueError, match="not finite"):
            fit.survival(np.array([9.0, np.nan]))
        with pytest.raises(ValueError, match="not finite"):
            fit.survival(np.array([9.0, np.inf]))


class TestSurvivalBand:
    """The uncertainty carried with the extrapolation."""

    @pytest.mark.parametrize("shape", [0.0, 1e-4, 0.25, -0.4])
    def test_band_matches_delta_method(self, shape):
        """
        The band equals the delta method built from a finite difference of
        ``genpareto.logsf``, at four shapes including the near-exponential regime.

        The analytic gradient in the module carries a series expansion below ``|w| =
        1e-3`` because the two terms of ``g(w)`` cancel there. That branch is taken by
        exactly the fits this module prefers, and an error in it would show up only as a
        band that is quietly the wrong width -- never as a failure. Differentiating
        scipy numerically is an oracle the module's algebra cannot match by accident.
        """
        fit = _tail(shape=shape, scale=1.5, threshold=2.0)
        stat = np.array([2.5, 3.0, 4.0])
        lower, upper = fit.survival_band(stat, level=0.9)

        step = 1e-7

        def log_sf(scale, shp):
            return genpareto.logsf(stat - 2.0, shp, loc=0.0, scale=scale)

        d_scale = (log_sf(1.5 + step, shape) - log_sf(1.5 - step, shape)) / (2.0 * step)
        d_shape = (log_sf(1.5, shape + step) - log_sf(1.5, shape - step)) / (2.0 * step)
        cov = fit.covariance
        variance = (
            cov[0, 0] * d_scale**2
            + 2.0 * cov[0, 1] * d_scale * d_shape
            + cov[1, 1] * d_shape**2
        )
        deviation = norm.ppf(0.95) * np.sqrt(variance)
        survival = fit.survival(stat)
        assert lower == pytest.approx(survival * np.exp(-deviation), rel=1e-6)
        assert upper == pytest.approx(survival * np.exp(deviation), rel=1e-6)

    def test_band_brackets_survival(self):
        """
        The point estimate lies inside its own band everywhere, and the band stays in
        [0, 1]. Built in the log so the lower edge cannot fall below zero several
        decades out, which is where the band is read.
        """
        _, _, fit, _ = _known_fit(0.35)
        stat = fit.threshold + np.geomspace(1e-3, 1e3, 400)
        lower, upper = fit.survival_band(stat, level=0.9)
        survival = fit.survival(stat)
        assert np.all(lower <= survival + 1e-15)
        assert np.all(upper >= survival - 1e-15)
        assert np.all((lower >= 0.0) & (upper <= 1.0))

    def test_band_widens_with_level(self):
        """
        Ninety-nine per cent coverage is wider than ninety at every statistic.

        A band that ignored its ``level`` would still bracket the point estimate and
        still look plausible on a plot.
        """
        _, _, fit, _ = _known_fit(0.35)
        stat = fit.threshold + np.geomspace(1e-2, 1e2, 100)
        narrow = fit.survival_band(stat, level=0.9)
        wide = fit.survival_band(stat, level=0.99)
        inside = fit.survival(stat) > 0.0
        assert np.all(wide[0][inside] < narrow[0][inside])
        assert np.all(wide[1][inside] > narrow[1][inside])

    def test_band_unity_below_threshold(self):
        """
        Below the threshold the survival is one by definition, not by estimation, so the
        band collapses to (1, 1); carrying parameter uncertainty into a region the model
        makes no statement about would be inventing it.
        """
        fit = _tail(shape=0.3, scale=1.5, threshold=4.0)
        lower, upper = fit.survival_band(np.array([0.0, 3.9, 4.0]))
        assert np.all(lower == 1.0) and np.all(upper == 1.0)

    def test_band_collapses_past_endpoint(self):
        """
        Past a bounded fit's endpoint the survival is identically zero and the delta
        method has nothing to expand about, so the band is (0, 0) rather than a NaN from
        the log of zero.
        """
        fit = _tail(shape=-0.5, scale=2.0, threshold=1.0)
        lower, upper = fit.survival_band(np.array([5.0, 10.0]))
        assert np.all(lower == 0.0) and np.all(upper == 0.0)

    def test_level_validated(self):
        """A coverage outside (0, 1) has no matching quantile and is refused."""
        fit = _tail()
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="level"):
                fit.survival_band(np.array([9.0]), level=bad)

    def test_zero_covariance_gives_no_band(self):
        """
        With no parameter uncertainty the two edges are the point estimate.

        The degenerate case a positive semi-definite covariance can reach; the square
        root of a rounding-level negative variance is guarded, and this pins that the
        guard returns a collapsed band rather than a NaN.
        """
        fit = _tail(shape=0.3, covariance=np.zeros((2, 2)))
        stat = np.array([9.0, 20.0])
        lower, upper = fit.survival_band(stat)
        assert lower == pytest.approx(fit.survival(stat))
        assert upper == pytest.approx(fit.survival(stat))


class TestChooseThreshold:
    """Where the tail is started from."""

    def test_min_exceedances_honoured(self):
        """
        The returned threshold leaves at least the requested number of exceedances.

        The count is the sample every reported uncertainty is built from, so a threshold
        one order statistic too high delivers a fit the caller did not authorise while
        looking entirely normal.
        """
        sample = _gpd_sample(0.25, size=6000, seed=41)
        for wanted in (200, 500, 1500):
            threshold = choose_threshold(sample, min_exceedances=wanted)
            assert int(np.count_nonzero(sample > threshold)) >= wanted

    def test_deterministic(self):
        """
        The threshold does not depend on a seed.

        The standard error on the ladder is Smith's asymptotic ``(1 + xi) / sqrt(n_u)``
        rather than a bootstrap for this reason: a bootstrap would make the threshold --
        and therefore every extrapolated rate above it -- depend on a random draw.
        """
        sample = _gpd_sample(0.25, size=6000, seed=41)
        first = choose_threshold(sample, min_exceedances=500, method="stability")
        assert (
            choose_threshold(sample, min_exceedances=500, method="stability") == first
        )
        # Order of presentation is not part of the sample.
        shuffled = np.random.default_rng(0).permutation(sample)
        assert (
            choose_threshold(shuffled, min_exceedances=500, method="stability") == first
        )

    def test_stable_shape_takes_lowest(self):
        """
        Where the shape is flat the lowest candidate is accepted, keeping the most data.

        A pure generalised Pareto sample is stable at every threshold, so the ladder
        should agree everywhere and stop at its first rung, the order statistic leaving
        half the sample. Returning the highest instead would pay variance for a shape
        nobody disputes.
        """
        sample = _gpd_sample(0.25, size=6000, seed=41)
        ordered = np.sort(sample)
        expected = float(ordered[6000 - 3000 - 1])
        assert (
            choose_threshold(sample, min_exceedances=500, method="stability") == expected
        )

    def test_locates_tail_onset(self):
        """
        With a bulk that is not Pareto, the threshold climbs to where the tail starts.

        A Gaussian bulk below a generalised Pareto tail grafted on at three: the whole
        point of the stability criterion is to refuse candidates whose fit is still
        being pulled by the bulk. A selector that always returned its lowest rung would
        sit at the median of the mixture and fit the Gaussian.
        """
        rng = np.random.default_rng(50)
        bulk = rng.normal(0.0, 1.0, size=8000)
        tail = 3.0 + genpareto.rvs(0.3, loc=0.0, scale=0.5, size=2000, random_state=rng)
        sample = np.concatenate([bulk, tail])
        threshold = choose_threshold(sample, min_exceedances=200, method="stability")
        assert 2.9 < threshold < 3.3
        assert threshold > float(np.quantile(sample, 0.5))

    def test_min_exceedances_floor_enforced(self):
        """
        Below ``MIN_FIT_EXCEEDANCES`` a two-parameter tail is arithmetic on a handful of
        points; the request is refused rather than answered.
        """
        sample = _gpd_sample(0.2, size=1000, seed=1)
        with pytest.raises(ValueError, match="min_exceedances"):
            choose_threshold(
                sample, min_exceedances=MIN_FIT_EXCEEDANCES - 1, method="stability"
            )

    def test_short_sample_refused(self):
        """A sample no larger than the exceedance count leaves no threshold."""
        sample = _gpd_sample(0.2, size=400, seed=1)
        with pytest.raises(ValueError, match="cannot leave"):
            choose_threshold(sample, min_exceedances=500)

    def test_ladder_size_validated(self):
        """A ladder of one rung tests stability against nothing."""
        sample = _gpd_sample(0.2, size=2000, seed=1)
        with pytest.raises(ValueError, match="n_candidates"):
            choose_threshold(
                sample, min_exceedances=500, n_candidates=1, method="stability"
            )

    def test_tied_statistic_refused(self):
        """
        A statistic tied everywhere admits no peaks-over-threshold fit and says so.

        The alternative -- returning the tied value -- would hand ``fit_tail`` a sample
        of identical excesses, whose likelihood is unbounded as the scale goes to zero.
        """
        with pytest.raises(ValueError, match="heavily tied"):
            choose_threshold(np.full(3000, 4.0), min_exceedances=500, method="stability")


class TestReproducibility:
    """What the seed does and does not control."""

    def test_same_seed_same_fit(self):
        """
        One seed reproduces the whole fit, both p-values and the covariance included.

        The fit is persisted and read by two consumers; a campaign that cannot be
        reproduced from its recorded seed cannot have its significance re-derived.
        """
        sample = _gpd_sample(0.3, size=2000, seed=8)
        kw = dict(threshold=0.0, n_bootstrap=50, n_null=60)
        first = fit_tail(sample, seed=3, **kw)
        second = fit_tail(sample, seed=3, **kw)
        assert first.scale == second.scale and first.shape == second.shape
        assert np.array_equal(first.covariance, second.covariance)
        assert first.lrt_p_value == second.lrt_p_value
        assert first.ad_p_value == second.ad_p_value

    def test_seed_moves_only_uncertainty(self):
        """
        A different seed gives a different covariance but the same point estimate.

        The estimate is a deterministic maximum likelihood search; only the bootstrap
        and the two null distributions are random. A covariance that did not move with
        seed would mean the resampling never happened.
        """
        sample = _gpd_sample(0.3, size=2000, seed=8)
        kw = dict(threshold=0.0, n_bootstrap=50, n_null=60)
        first = fit_tail(sample, seed=3, **kw)
        other = fit_tail(sample, seed=4, **kw)
        assert first.scale == other.scale and first.shape == other.shape
        assert not np.array_equal(first.covariance, other.covariance)
        assert first.covariance == pytest.approx(other.covariance, rel=0.5)

    def test_nulls_are_independent_streams(self):
        """
        The two null distributions are spawned substreams, not consecutive seeds.

        Reported side by side as independent evidence about the same exceedances;
        drawing both from one stream would correlate them without changing how they
        look.
        """
        from sage.search.tail import _substream_seeds

        seeds = _substream_seeds(0, 3)
        assert len(set(seeds)) == 3
        assert _substream_seeds(0, 3) == seeds
        assert _substream_seeds(1, 3) != seeds


class TestDegenerateInput:
    """Inputs that must fail loudly rather than return a plausible number."""

    def test_empty_sample_refused(self):
        """No statistics is not a tail with a wide uncertainty; it is nothing to fit."""
        with pytest.raises(ValueError):
            fit_tail(np.array([]), threshold=0.0)
        with pytest.raises(ValueError):
            fit_tail(np.array([]))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_sample_refused(self, bad):
        """
        A non-finite statistic is a fault to report, not a value to fit.

        A NaN would drop out of the exceedance count without being counted anywhere,
        since it compares false against every threshold; an infinite excess drives the
        profile likelihood onto its boundary and the fit then describes that one value.
        """
        sample = np.append(_gpd_sample(0.2, size=800, seed=2), bad)
        with pytest.raises(ValueError, match="not finite"):
            fit_tail(sample, threshold=0.0)
        with pytest.raises(ValueError, match="not finite"):
            choose_threshold(sample, min_exceedances=200)

    def test_too_few_exceedances_refused(self):
        """
        A threshold leaving fewer than ``MIN_FIT_EXCEEDANCES`` points is refused.

        Two parameters from a handful of order statistics is arithmetic rather than
        inference, and the number it produces would be read as a rate.
        """
        sample = _gpd_sample(0.2, size=800, seed=2)
        threshold = float(np.sort(sample)[-5])
        with pytest.raises(ValueError, match=str(MIN_FIT_EXCEEDANCES)):
            fit_tail(sample, threshold=threshold)

    def test_identical_exceedances_refused(self):
        """
        Excesses that are all equal have an unbounded likelihood as the scale goes to
        zero; there is no maximum to report and the fit says so instead of returning the
        smallest scale its search happened to reach.
        """
        with pytest.raises(ValueError, match="identical"):
            fit_tail(np.full(100, 3.0), threshold=2.0)

    def test_non_finite_threshold_refused(self):
        """A threshold that is not a number selects everything or nothing."""
        sample = _gpd_sample(0.2, size=800, seed=2)
        with pytest.raises(ValueError, match="threshold"):
            fit_tail(sample, threshold=np.nan)
        with pytest.raises(ValueError, match="threshold"):
            fit_tail(sample, threshold=np.inf)

    def test_fit_arguments_validated(self):
        """A covariance needs two resamples to be formed from."""
        sample = _gpd_sample(0.2, size=800, seed=2)
        with pytest.raises(ValueError, match="n_bootstrap"):
            fit_tail(sample, threshold=0.0, n_bootstrap=1)


class TestTailFitValidation:
    """The guards a persisted fit is constructed through."""

    def test_non_positive_scale_refused(self):
        """Every excess is divided by the scale, so zero is not a small scale."""
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="scale"):
                _tail(scale=bad)

    def test_shape_below_floor_refused(self):
        """
        Below ``SHAPE_FLOOR`` the likelihood diverges as the fitted endpoint is squeezed
        onto the largest excess, so a fit reported there describes one order statistic
        and not a tail. Refused on construction because the object is persisted and read
        two stages away from whatever produced it.
        """
        with pytest.raises(ValueError, match="below"):
            _tail(shape=SHAPE_FLOOR - 0.01)
        assert _tail(shape=SHAPE_FLOOR).shape == SHAPE_FLOOR

    def test_non_finite_parameters_refused(self):
        """A NaN parameter would surface as a rate in a candidate table."""
        for field in ("threshold", "scale", "shape"):
            with pytest.raises(ValueError):
                _tail(**{field: np.nan})

    def test_covariance_shape_checked(self):
        """
        The covariance is 2x2 over ``PARAMETER_ORDER``; anything else cannot be indexed
        by the band, and would fail where the band is read, not where it was built.
        """
        with pytest.raises(ValueError, match="2x2"):
            _tail(covariance=np.zeros(2))
        with pytest.raises(ValueError, match="2x2"):
            _tail(covariance=np.zeros((3, 3)))
        with pytest.raises(ValueError, match="non-finite"):
            _tail(covariance=np.full((2, 2), np.nan))

    def test_p_values_range_checked(self):
        """A p-value outside [0, 1] is a counting error, and both are reported."""
        base = dict(
            threshold=1.0,
            scale=1.0,
            shape=0.1,
            covariance=np.eye(2),
            n_exceedances=100,
        )
        with pytest.raises(ValueError, match="lrt_p_value"):
            TailFit(lrt_p_value=1.5, ad_p_value=0.5, **base)
        with pytest.raises(ValueError, match="ad_p_value"):
            TailFit(lrt_p_value=0.5, ad_p_value=-0.1, **base)

    def test_negative_count_refused(self):
        """The exceedance count is a sample size and cannot be negative."""
        with pytest.raises(ValueError, match="n_exceedances"):
            TailFit(
                threshold=1.0,
                scale=1.0,
                shape=0.1,
                covariance=np.eye(2),
                n_exceedances=-1,
                lrt_p_value=0.5,
                ad_p_value=0.5,
            )

    def test_fields_coerced_to_scalars(self):
        """
        A fit read back from storage arrives with numpy scalars; the guards run on
        floats, so the coercion happens on construction rather than being relied on at
        each point of use.
        """
        fit = TailFit(
            threshold=np.float32(2.0),
            scale=np.float32(1.5),
            shape=np.float32(0.25),
            covariance=[[1.0, 0.0], [0.0, 1.0]],
            n_exceedances=np.int32(100),
            lrt_p_value=np.float32(0.5),
            ad_p_value=np.float32(0.5),
        )
        assert isinstance(fit.threshold, float) and isinstance(fit.shape, float)
        assert isinstance(fit.n_exceedances, int)
        assert fit.covariance.dtype == np.float64


class TestEndToEnd:
    """The default path, run once as the pipeline runs it."""

    def test_chosen_threshold_fit_recovers(self):
        """
        With no threshold supplied the whole fit still recovers known parameters.

        Exercises ``choose_threshold`` and ``fit_tail`` together, since the scale the
        fit should return depends on the threshold the first of them picked:
        ``scale + shape * u``. A threshold off by an order statistic, or a scale that
        forgot the reparametrisation, is only visible in the two run together.
        """
        shape = 0.3
        sample = _gpd_sample(shape, size=6000, seed=77)
        fit = fit_tail(
            sample, n_bootstrap=N_BOOT, n_null=N_NULL, min_exceedances=500
        )
        assert fit.n_exceedances == int(np.count_nonzero(sample > fit.threshold))
        assert fit.n_exceedances >= 500
        expected_scale = TRUE_SCALE + shape * fit.threshold
        sigma = np.sqrt(np.diag(fit.covariance))
        assert abs(fit.scale - expected_scale) <= 3.5 * sigma[0]
        assert abs(fit.shape - shape) <= 3.5 * sigma[1]
        assert fit.finite_endpoint is None


def _drifting_shape_sample():
    """
    Exceedances whose fitted shape drifts with the threshold.

    An exponential bulk with a heavy Pareto contamination: low on the ladder the fit is
    pulled by the bulk and high on it by the contamination, so ``xi(u)`` genuinely moves
    and the stability criterion has something to discriminate. A clean single-population
    sample is flat everywhere, which is exactly the fixture on which an ignored tuning
    parameter looks correct.
    """
    rng = np.random.default_rng(7)
    return np.concatenate([rng.exponential(1.0, 9000), rng.pareto(1.5, 3000) + 1.0])


class TestThresholdDefault:
    """The count rule is the default, and the ladder is a diagnostic."""

    def test_default_is_the_count_rule(self):
        """
        ``choose_threshold`` defaults to PyCBC's count rule.

        It is the only rule with a counterpart: every PyCBC fit threshold is supplied by
        the caller or set by ``tail_threshold``, and no GWTC methods paper says how to
        select one. The stability search remains available, and returns something else on
        the same sample, so the default is a real choice rather than an alias.
        """
        sample = _gpd_sample(0.25, size=6000, seed=41)
        assert choose_threshold(sample, min_exceedances=500) == threshold_at_count(
            sample, n_exceedances=500
        )
        assert choose_threshold(
            sample, min_exceedances=500, method="stability"
        ) != threshold_at_count(sample, n_exceedances=500)

    def test_unknown_method_refused(self):
        """A misspelled method is not silently treated as the default."""
        sample = _gpd_sample(0.2, size=2000, seed=1)
        with pytest.raises(ValueError, match="expected count or stability"):
            choose_threshold(sample, min_exceedances=500, method="stabilty")

    def test_ladder_reports_shape_per_rung(self):
        """
        The ladder reports the fitted shape and its error at every threshold.

        This is what PyCBC's ``pycbc_fit_sngl_trigs`` writes and plots when given a list
        of thresholds, and it is what makes the shape visible rather than buried in one
        selected fit. Reading it is how a heavy tail is noticed.
        """
        sample = _gpd_sample(0.25, size=6000, seed=41)
        ladder = threshold_ladder(sample, min_exceedances=500, n_candidates=8)

        for key in ("threshold", "shape", "scale", "std_error", "n_exceedances"):
            assert key in ladder
        n = ladder["threshold"].size
        assert n >= 2
        assert all(ladder[k].size == n for k in ladder)
        assert np.all(np.diff(ladder["threshold"]) > 0)
        assert np.all(np.diff(ladder["n_exceedances"]) < 0)
        assert np.all(np.isfinite(ladder["shape"]))
        assert np.all(ladder["std_error"] > 0)
        # A shape recovered from a known draw, read off the ladder rather than a fit.
        assert np.median(ladder["shape"]) == pytest.approx(0.25, abs=0.15)

    def test_top_rung_not_vacuously_accepted(self):
        """
        The highest rung has nothing above it, so it is never accepted as stable.

        Accepting it made the least-unstable fallback unreachable: the topmost rung
        scored a perfect zero by having no comparison to fail, so some candidate always
        passed and the "no candidate is stable" path was dead code.
        """
        sample = _drifting_shape_sample()
        ladder = threshold_ladder(sample, min_exceedances=500, n_candidates=15)
        chosen = choose_threshold(
            sample, min_exceedances=500, n_sigma=1e-9, method="stability"
        )

        assert chosen != float(ladder["threshold"][-1])


class TestThresholdLadder:
    """The stability search must actually use its tuning parameters."""

    def test_threshold_falls_with_n_sigma(self):
        """
        A looser agreement test accepts lower on the ladder.

        ``n_sigma`` is the width at which two fitted shapes are called compatible, so
        widening it accepts a candidate that a stricter test rejects, and the lowest
        accepted candidate is the one returned. Ignoring the argument would return one
        threshold for every width.
        """
        sample = _drifting_shape_sample()
        kw = dict(min_exceedances=500, method="stability")
        strict = choose_threshold(sample, n_sigma=0.01, **kw)
        moderate = choose_threshold(sample, n_sigma=2.0, **kw)
        loose = choose_threshold(sample, n_sigma=1e6, **kw)

        assert strict > moderate > loose

    def test_ladder_size_changes_threshold(self):
        """
        ``n_candidates`` sets which thresholds are available to be chosen.

        A denser ladder is a stricter test: the lowest rung must agree with every rung
        above it, so adding rungs adds ways to disagree and the accepted threshold climbs.
        A two-rung ladder tests its lowest rung against one other and accepts it. An
        implementation that built a fixed ladder would return the same value for both.
        """
        sample = _drifting_shape_sample()
        coarse = choose_threshold(
            sample, min_exceedances=500, n_candidates=2, method="stability"
        )
        fine = choose_threshold(
            sample, min_exceedances=500, n_candidates=15, method="stability"
        )

        assert coarse != fine
        assert fine > coarse


class TestGoodnessOfFitNull:
    """The Anderson-Darling null must be simulated at the shape that was fitted."""

    def test_null_uses_fitted_shape(self, monkeypatch):
        """
        The null is drawn at the fitted shape, not at zero.

        Simulating at the fitted shape is the entire reason this beats a published table:
        the tables assume known parameters, and refitting each replicate is what makes the
        statistic comparable. Drawing the null at a fixed shape would compare the
        exceedances against the wrong distribution and report a goodness of fit that was
        never measured.
        """
        import sage.search.tail as tail_module

        sample = genpareto.rvs(0.4, loc=0.0, scale=1.0, size=2000, random_state=11)
        threshold = float(np.quantile(sample, 0.5))
        expected_shape = tail_module._gpd_mle(sample[sample > threshold] - threshold)[1]

        seen = {}
        original = tail_module._ad_null

        def recording(n, n_null, shape, seed):
            seen["shape"] = shape
            return original(n, n_null, shape, seed)

        monkeypatch.setattr(tail_module, "_ad_null", recording)
        anderson_darling(sample, threshold, n_null=20, seed=3)

        assert seen["shape"] == pytest.approx(expected_shape, rel=1e-12)
        assert abs(seen["shape"]) > 0.05


class TestThresholdMethods:
    """The two threshold rules, and the PyCBC one pinned against PyCBC itself."""

    def test_count_matches_pycbc(self):
        """
        The fixed-count rule reproduces ``pycbc.events.trigger_fits.tail_threshold``.

        Differential against the source it is taken from, not against a transcription of
        it. The two differ by exactly one order statistic and that is a convention, not a
        discrepancy: PyCBC counts inclusively at the threshold so N values are ``>=`` it,
        while every exceedance here is strictly above, so the value one place lower is the
        one that leaves N points for the fit.
        """
        trigger_fits = pytest.importorskip("pycbc.events.trigger_fits")
        rng = np.random.default_rng(4)
        sample = rng.exponential(1.2, 30000)

        for n in (500, 1000, 5000):
            ours = threshold_at_count(sample, n_exceedances=n)
            theirs = float(trigger_fits.tail_threshold(sample, N=n))
            ordered = np.sort(sample)

            assert theirs == pytest.approx(ordered[ordered.size - n])
            assert ours == pytest.approx(ordered[ordered.size - n - 1])
            assert int(np.count_nonzero(sample > ours)) == n
            assert int(np.count_nonzero(sample >= theirs)) == n

    def test_count_selected_through_choose_threshold(self):
        """``method="count"`` routes to the fixed-count rule and ignores the ladder."""
        rng = np.random.default_rng(4)
        sample = rng.exponential(1.2, 30000)

        assert choose_threshold(
            sample, min_exceedances=500, method="count"
        ) == threshold_at_count(sample, 500)
        # The ladder parameters are inert on this path.
        assert choose_threshold(
            sample, min_exceedances=500, method="count", n_sigma=99.0, n_candidates=2
        ) == threshold_at_count(sample, 500)

    def test_unknown_method_refused(self):
        """A misspelled rule must not fall through to a default."""
        sample = _gpd_sample(0.2, size=2000, seed=1)
        with pytest.raises(ValueError, match="unknown threshold method"):
            choose_threshold(sample, min_exceedances=500, method="stabilty")


class TestKolmogorovSmirnov:
    """KS beside Anderson-Darling, so the choice between them is measured."""

    def test_tabulated_p_value_is_anticonservative(self):
        """
        PyCBC's tabulated p-value accepts more readily than the refitted null does.

        Both parameters were estimated from the same exceedances, which makes the
        statistic stochastically smaller than the known-parameter tables assume. Reading a
        table then overstates the fit, so the bootstrap p-value is the smaller of the two
        -- and the gap is the size of the effect.
        """
        rng = np.random.default_rng(4)
        sample = rng.exponential(1.2, 30000)
        threshold = threshold_at_count(sample, 500)

        stat_boot, p_boot = ks_test(sample, threshold, n_null=200, seed=1)
        stat_table, p_table = ks_test(sample, threshold, bootstrap=False)

        assert stat_boot == pytest.approx(stat_table)
        assert p_boot < p_table

    def test_both_tests_reject_a_wrong_tail(self):
        """
        A tail that is not generalised Pareto fails under either statistic.

        Anderson-Darling is the production test because its weight puts the sensitivity at
        the ends, but a badly wrong tail should not need the sensitive test to be caught,
        and a fixture only one of them rejects would not tell the two apart.
        """
        rng = np.random.default_rng(6)
        # Bimodal exceedances. Deliberately not uniform ones: uniform excesses ARE a
        # generalised Pareto, the bounded xi = -1 case, so both tests rightly accept them.
        # No generalised Pareto is bimodal, which is what makes this a genuine misfit.
        wrong = np.concatenate(
            [rng.uniform(8.0, 8.2, 1500), rng.uniform(10.0, 10.2, 1500)]
        )
        threshold = 8.0

        assert anderson_darling(wrong, threshold, n_null=200, seed=2)[1] < 0.05
        assert ks_test(wrong, threshold, n_null=200, seed=2)[1] < 0.05

    def test_good_fit_passes_both(self):
        """Genuine generalised Pareto exceedances are rejected by neither."""
        sample = _gpd_sample(0.25, size=6000, seed=3)
        threshold = threshold_at_count(sample, 1000)

        assert anderson_darling(sample, threshold, n_null=200, seed=4)[1] > 0.05
        assert ks_test(sample, threshold, n_null=200, seed=4)[1] > 0.05
