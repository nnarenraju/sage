#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_pastro.py
Description   : Rate inference and per-candidate probability.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Checked against closed forms taken from the local references:
docs/references/arxiv_1302.5341.pdf Eq. (21) and Eq. (35), and
docs/references/arxiv_2305.00071.pdf Eq. (10) and Eq. (11).

These are gates on publishability, not diagnostics, so each has a definite pass
condition.
"""

import numpy as np
import pytest

from sage.search.pastro.assign import assign_pastro, sum_consistency
from sage.search.pastro.density import (
    noise_density,
    signal_density,
    verify_normalisation,
)
from sage.search.pastro.io import ContractViolation, require_clustered
from sage.search.pastro.monotonic import (
    apply_policy,
    check_monotonicity,
)
from sage.search.pastro.rates import fit_rates, log_prior
from sage.search.pastro.support import CommonSupport
from sage.search.pastro.validate import analytic_oracle, quadrature_oracle

# A realistic pair. The signal statistic inherits the heavy tail of a uniform-in-volume
# population -- p(rho) ~ rho^-4 -- rather than a Gaussian, which is what makes the density
# ratio monotone at the top of the range; a narrow Gaussian signal against an
# exponential-tailed background is non-monotone there for reasons of the fixture rather
# than of the search.
SUPPORT = CommonSupport(stat_lo=1.0, stat_hi=18.0, n_stat=512)
N_SIGNAL, N_NOISE = 45, 400


def _noise_samples(n=30000, seed=4):
    return np.random.default_rng(seed).exponential(1.2, n)


def _signal_samples(n=6000, seed=5):
    rng = np.random.default_rng(seed)
    values = 8.0 * rng.pareto(3.0, n) + 8.0
    return values[values < SUPPORT.stat_hi]


def _densities(noise=None, seed=4):
    background = _noise_samples() if noise is None else np.asarray(noise)
    return {
        "BBH": signal_density(_signal_samples(), SUPPORT),
        "Terrestrial": noise_density(background, SUPPORT),
    }


def _observed(seed=9):
    """A mixture drawn from the same populations the densities were built from."""
    rng = np.random.default_rng(seed)
    signal = 8.0 * rng.pareto(3.0, N_SIGNAL) + 8.0
    noise = rng.exponential(1.2, 4 * N_NOISE)
    values = np.concatenate([signal, noise])
    return values[SUPPORT.contains(values)]


class TestClosedForms:
    """Cases where the answer is known analytically."""

    def test_foreground_dominated_limit(self):
        """
        With negligible background the rate posterior peaks at N - 1/2.

        Eq. (35) of arxiv_1302.5341 reduces the posterior to Rf^(N-1/2) exp(-Rf); the
        half comes from the Jeffreys prior, so this also confirms the prior is applied.
        Asserted on the marginal in the signal rate, which is the quantity that closed
        form is stated for -- not on the mode of the joint, which is a different point.
        """
        densities = _densities()
        rng = np.random.default_rng(11)
        for n in (10, 30, 60):
            # Every trigger far above the background, so the noise component is negligible.
            observed = 8.0 * rng.pareto(3.0, n) + 12.0
            observed = observed[observed < SUPPORT.stat_hi]
            posterior = fit_rates(
                observed, densities, SUPPORT, clustered=True, n_grid=400
            )
            nodes, density = posterior.marginal("BBH")
            mode = float(nodes[np.argmax(density)])
            assert mode == pytest.approx(observed.size - 0.5, abs=0.35)

    def test_thresholded_half_normal_rates(self):
        """
        A threshold that removes half the noise halves the observable noise rate.

        Standard-normal noise cut at its own median leaves exactly half, while a signal
        centred three sigma above is barely touched, so both observable rates follow in
        closed form from the survival functions and can be compared directly.
        """
        from scipy.stats import norm

        oracle = analytic_oracle(
            signal_loc=3.0, noise_rate=1000.0, signal_rate=100.0, threshold=0.0
        )
        assert oracle["noise_survival"] == pytest.approx(0.5)
        assert oracle["observable_noise_rate"] == pytest.approx(500.0)
        assert oracle["observable_signal_rate"] == pytest.approx(
            100.0 * norm.sf(-3.0)
        )

        rng = np.random.default_rng(17)
        support = CommonSupport(stat_lo=0.0, stat_hi=9.0, n_stat=512)
        noise = rng.normal(0.0, 1.0, 200000)
        signal = rng.normal(3.0, 1.0, 40000)
        densities = {
            "BBH": signal_density(signal[support.contains(signal)], support),
            "Terrestrial": noise_density(noise[support.contains(noise)], support),
        }
        observed = np.concatenate(
            [
                rng.normal(0.0, 1.0, int(oracle["observable_noise_rate"] * 2)),
                rng.normal(3.0, 1.0, int(oracle["observable_signal_rate"] * 1.001)),
            ]
        )
        observed = observed[support.contains(observed)]
        posterior = fit_rates(observed, densities, support, clustered=True, n_grid=400)
        rates = posterior.mean_rates

        assert rates["Terrestrial"] == pytest.approx(500.0, rel=0.2)
        assert rates["BBH"] == pytest.approx(100.0, rel=0.35)

    def test_grid_matches_adaptive_quadrature(self):
        """
        The gridded posterior agrees with an independent integration of Eq. (10).

        The cross-check integrates in the original rate variables with ``quad``, sharing
        no code with the grid: a mistake in the node layout, the quadrature weights or the
        Jacobian moves one and not the other.
        """
        rng = np.random.default_rng(21)
        observed = np.concatenate(
            [8.0 * rng.pareto(3.0, 12) + 8.0, rng.exponential(1.2, 60)]
        )
        observed = observed[SUPPORT.contains(observed)]
        result = quadrature_oracle(observed, _densities(), SUPPORT)

        assert abs(result["fractional_difference"]) < 0.02

    def test_jeffreys_jacobian_applied_once(self):
        """
        The prior in the reparameterised variables includes its Jacobian, once.

        ``1/sqrt(Ls Ln)`` with the ``lam`` from the change of variables cancels the total
        rate exactly, leaving a prior flat in ``lam``. Both the correct form and the form
        without the Jacobian are constructed here, and the second is asserted to differ --
        without that half the test would pass on the version that omits it, which is what
        the sgwc-1 notebook does.
        """
        total = np.array([5.0, 50.0, 500.0])[:, None]
        fraction = np.array([0.05, 0.5, 0.95])[None, :]
        rate_s, rate_n = fraction * total, (1.0 - fraction) * total

        with_jacobian = -0.5 * np.log(rate_s) - 0.5 * np.log(rate_n) + np.log(total)
        without = -0.5 * np.log(rate_s) - 0.5 * np.log(rate_n)
        computed = log_prior(total, fraction)

        assert np.allclose(computed, with_jacobian)
        assert not np.allclose(computed, without)
        # Flat in the total rate: every row identical.
        assert np.allclose(computed - computed[0][None, :], 0.0)


class TestAssignment:
    """Per-candidate probability, Eq. (11) of arxiv_2305.00071."""

    @staticmethod
    def _fitted():
        densities = _densities()
        observed = _observed()
        posterior = fit_rates(observed, densities, SUPPORT, clustered=True, n_grid=256)
        return observed, densities, posterior

    def test_probability_is_bounded(self):
        """Values lie in the unit interval for every input."""
        observed, densities, posterior = self._fitted()
        probe = np.linspace(SUPPORT.stat_lo, SUPPORT.stat_hi, 400)
        values = assign_pastro(probe, densities, posterior).probabilities["BBH"]

        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)
        assert np.all(np.isfinite(values))

    def test_monotonic_in_statistic(self):
        """
        Where the density ratio increases, so does the probability.

        Asserted as the correspondence rather than as unconditional monotonicity. The
        probability is a strictly increasing function of the ratio and of nothing else, so
        it must rise exactly where the ratio rises and fall exactly where it falls. The
        densities are histograms, so the ratio is piecewise constant and most probe points
        sit inside a bin where nothing moves -- which is why the comparison is made only
        where the ratio actually changes, and why an unconditionally rising probability
        would be testing the binning rather than the assignment.
        """
        observed, densities, posterior = self._fitted()
        probe = np.linspace(SUPPORT.stat_lo, SUPPORT.stat_hi, 400)
        values = assign_pastro(probe, densities, posterior).probabilities["BBH"]
        log_ratio = densities["BBH"].log_prob(probe) - densities[
            "Terrestrial"
        ].log_prob(probe)

        step_p = np.diff(values)
        step_r = np.diff(log_ratio)
        moving = np.isfinite(step_r) & (np.abs(step_r) > 1e-9)
        assert moving.any(), "the ratio never moves; the probe resolves no bin edge"
        assert np.all(np.sign(step_p[moving]) == np.sign(step_r[moving]))

        # And the ratio rises across the range, so the probability does. Tested as a
        # trend rather than step by step: the densities are counts, so neighbouring bins
        # differ by Poisson noise and a fraction-of-steps-rising threshold measures the
        # occupancy of the fixture rather than the ordering of the statistic.
        # And the ratio rises across the range as a whole. Compared between the ends
        # rather than step by step or by rank: the densities are counts, so neighbouring
        # bins differ by Poisson noise, and how often consecutive bins happen to rise is a
        # property of how well populated this fixture is -- not of the assignment.
        finite = np.isfinite(log_ratio)
        low, high = log_ratio[finite][:100], log_ratio[finite][-100:]
        assert np.median(high) > np.median(low)

    def test_marginalises_over_the_full_rate_grid(self):
        """
        The average runs over the whole grid, not its diagonal.

        Checked against the double sum written out in the original rate variables, which
        is Eq. (11) as printed. An implementation pairing the two rate axes elementwise --
        or evaluating at a point estimate -- differs from it detectably.
        """
        observed, densities, posterior = self._fitted()
        probe = np.array([4.0, 8.0, 13.0])
        values = assign_pastro(probe, densities, posterior).probabilities["BBH"]

        log_ps = densities["BBH"].log_prob(probe)
        log_pn = densities["Terrestrial"].log_prob(probe)
        weights = posterior.weights
        rate_s = posterior.total_grid[:, None] * posterior.fraction_grid[None, :]
        rate_n = posterior.total_grid[:, None] * (1.0 - posterior.fraction_grid)[None, :]
        for k, expected_stat in enumerate(probe):
            numerator = rate_s * np.exp(log_ps[k])
            denominator = numerator + rate_n * np.exp(log_pn[k])
            brute = float(np.sum((numerator / denominator) * weights))
            assert values[k] == pytest.approx(brute, rel=1e-9)

        diagonal = float(
            np.sum(np.diag(rate_s) / (np.diag(rate_s) + np.diag(rate_n)))
        ) / posterior.total_grid.size
        assert not np.isclose(values[0], diagonal)

    def test_credible_interval_brackets_the_value(self):
        """The reported interval contains the point estimate and narrows with data."""
        densities = _densities()
        observed = _observed()
        # Shuffled, so every prefix is a fair sample of the same mixture. Taking a prefix
        # of the concatenated array instead would vary the signal fraction between the
        # subsets, and the interval would then be responding to a different population
        # rather than to more data about one population.
        rng = np.random.default_rng(0)
        rng.shuffle(observed)
        probe = np.array([11.0])

        widths = []
        for size in (len(observed) // 4, len(observed) // 2, len(observed)):
            posterior = fit_rates(
                observed[:size], densities, SUPPORT, clustered=True, n_grid=256
            )
            table = assign_pastro(probe, densities, posterior)
            value = table.probabilities["BBH"][0]
            assert table.lower["BBH"][0] <= value <= table.upper["BBH"][0]
            widths.append(table.upper["BBH"][0] - table.lower["BBH"][0])

        assert widths[-1] < widths[0]
        assert widths == sorted(widths, reverse=True)

    def test_sum_recovers_the_inferred_rate(self):
        """
        Summing over the analysed set returns the inferred signal rate.

        Each term is the probability that one trigger is signal, so the sum is the
        expected number of signals -- which is what the rate parameter means. This is the
        consistency sgwc-1 reported at 58.78 per cent and did not act on.
        """
        observed, densities, posterior = self._fitted()
        table = assign_pastro(observed, densities, posterior)
        result = sum_consistency(table, posterior)

        assert abs(result["fractional"]) < 0.05








class TestNormalisationTolerance:
    """The tolerance has to be reachable by the quadrature it is checked with."""

    def test_blended_density_meets_the_default(self):
        """
        A tail-blended density passes ``verify_normalisation`` at its default tolerance.

        The blend is discontinuous at the join -- mass anchoring does not force
        continuity -- so a trapezoid over the support grid carries far more error than it
        does on a smooth density. A default the production density cannot meet would make
        the check fire on the quadrature rather than on the density.
        """
        densities = _densities()
        for name, density in densities.items():
            assert verify_normalisation(density) == pytest.approx(1.0, abs=1e-3)

    def test_normalisation_is_exact_at_any_grid(self):
        """
        A histogram's integral is its counts, so refining the support changes nothing.

        Under the kernel estimate this was a quadrature error that fell as 1/n_stat**2 and
        set the tolerance. A piecewise-constant density integrates to
        ``sum(counts) / n_total`` exactly, so the check is now on the estimator rather than
        on the grid it is sampled with.
        """
        samples = _noise_samples()
        for n_stat in (256, 2048):
            support = CommonSupport(
                stat_lo=SUPPORT.stat_lo, stat_hi=SUPPORT.stat_hi, n_stat=n_stat
            )
            inside = samples[(samples > support.stat_lo) & (samples < support.stat_hi)]
            density = noise_density(inside, support)
            assert verify_normalisation(density, atol=1e-12) == pytest.approx(1.0)


class TestInvariance:
    """Properties the answer must have if it is measuring anything."""

    def test_threshold_invariant(self):
        """
        A candidate's probability does not depend on where the threshold was placed.

        Compared against the credible intervals rather than a fixed allowance, so the test
        tightens as the estimate becomes more precise. Drift here is the visible symptom of
        a failure elsewhere -- unclustered triggers, mismatched truncation, or a bandwidth
        following the observed extremes.
        """
        from sage.search.pastro.validate import threshold_invariance

        observed = _observed()
        background = _noise_samples()
        signal = _signal_samples()
        thresholds = [1.0, 1.5, 2.0]
        built = {}
        for threshold in thresholds:
            support = CommonSupport(
                stat_lo=threshold, stat_hi=SUPPORT.stat_hi, n_stat=512
            )
            built[threshold] = {
                "BBH": signal_density(signal[signal > threshold], support),
                "Terrestrial": noise_density(
                    background[background > threshold], support
                ),
            }
        result = threshold_invariance(observed, built, thresholds, k_sigma=3.0)

        assert result["passed"], result

    def test_converges_as_background_accumulates(self):
        """
        The value settles and its interval narrows as background is added.

        Continued drift is what a density following the sample extremes produces: every
        new batch of background moves the model, so the answer never converges however
        much is accumulated.
        """
        from sage.search.pastro.validate import convergence_with_background

        background = _noise_samples(n=60000)
        signal = _signal_samples()

        def builder(subset):

            return {
                "BBH": signal_density(signal, SUPPORT),
                "Terrestrial": noise_density(subset, SUPPORT),
            }

        result = convergence_with_background(
            _observed(),
            [background[:15000], background[:30000], background],
            builder,
        )

        assert result["narrowing"], result
        assert max(result["steps"]) < 0.05

    def test_grid_range_does_not_affect_the_result(self):
        """
        Widening the rate grid does not move the answer.

        The bracket is derived from the observed count rather than configured, so a result
        that moved with it would be reporting the edge of the grid instead of the
        posterior -- which is what a grid too narrow to contain the mass does.
        """
        densities = _densities()
        observed = _observed()
        probe = np.array([11.0])

        values = []
        for n_grid in (192, 384, 768):
            posterior = fit_rates(
                observed, densities, SUPPORT, clustered=True, n_grid=n_grid
            )
            values.append(
                float(assign_pastro(probe, densities, posterior).probabilities["BBH"][0])
            )

        assert max(values) - min(values) < 1e-3


class TestSupportReachesCandidates:
    """The support must contain what it is asked to score."""

    def test_must_include_extends_the_upper_edge(self):
        """
        Injections and candidates stretch the support past the loudest background.

        Taken from the FAR curve alone the upper edge is the loudest *background* event,
        and a candidate is confident precisely because it is louder than that. A support
        bounded by the background therefore excludes every genuine detection and truncates
        the signal density where it still carries mass.
        """
        from sage.search.background import BackgroundSet
        from sage.search.far import build_far_curve
        from sage.search.pastro.support import build_support

        rng = np.random.default_rng(2)
        background = BackgroundSet(
            stats=rng.exponential(1.0, 4000),
            livetime_s=3.0e7,
            n_slides=8,
            removal="inclusive",
        )
        curve = build_far_curve(background, 8.64e4)
        loudest_background = float(curve.stat.max())

        bare = build_support(curve, threshold_far_per_day=2.0, stat_pad=1.0)
        reaching = build_support(
            curve,
            threshold_far_per_day=2.0,
            stat_pad=1.0,
            must_include=[loudest_background + 12.0],
        )

        assert bare.stat_hi == pytest.approx(loudest_background + 1.0)
        assert reaching.stat_hi == pytest.approx(loudest_background + 13.0)
        assert reaching.contains(np.array([loudest_background + 12.0]))[0]
        assert not bare.contains(np.array([loudest_background + 12.0]))[0]

    def test_loud_candidate_refused_not_nan(self):
        """
        A candidate above the support raises, rather than returning NaN.

        Outside the support both densities are zero, so the log odds is ``inf - inf``.
        Without the guard the loudest candidate of a campaign -- the one the stage exists
        to assess -- comes back NaN behind nothing but a RuntimeWarning, and NaN reads as
        a value all the way into a candidate table.
        """
        densities = _densities()
        observed = _observed()
        posterior = fit_rates(observed, densities, SUPPORT, clustered=True, n_grid=256)
        beyond = np.array([SUPPORT.stat_hi + 5.0])

        # Beyond both histograms every density sits at the same floor, so the ratio is
        # exactly one and the log odds exactly zero -- carrying no information about the
        # candidate at all, while looking like a perfectly ordinary number. That is what
        # the guard prevents: not a NaN, which announces itself, but a fabricated tie.
        log_ratio = densities["Terrestrial"].log_prob(beyond) - densities[
            "BBH"
        ].log_prob(beyond)
        assert np.allclose(log_ratio, 0.0)

        with pytest.raises(ValueError, match="outside the common support"):
            assign_pastro(beyond, densities, posterior)


class TestSumConsistencyOffset:
    """The summed probability differs from the inferred rate by exactly one half."""

    def test_offset_is_exactly_a_half(self):
        """
        ``sum_i p_astro_i = E[Ls] - 1/2`` under the Jeffreys prior, as an identity.

        The same half that puts the FGMC rate posterior's mode at ``N - 1/2``: it comes
        from the prior, not from any disagreement between the densities and the rates.
        PyCBC's ``count_posterior`` obeys the identity too, with the constant
        ``-(alpha + 1)`` at power-law prior ``alpha``.
        """
        densities = _densities()
        observed = _observed()
        posterior = fit_rates(observed, densities, SUPPORT, clustered=True, n_grid=256)
        result = sum_consistency(assign_pastro(observed, densities, posterior), posterior)

        assert result["raw_difference"] == pytest.approx(-0.5, abs=1e-6)
        assert result["difference"] == pytest.approx(0.0, abs=1e-6)
        assert result["expected"] == pytest.approx(result["inferred"] - 0.5)

    def test_low_rate_run_is_not_failed(self):
        """
        A correct run with few signals passes the gate; the naive residual would fail it.

        The constant is half an event, so as a fraction of the inferred rate it grows
        without bound as the rate falls: below five signals it exceeds ten per cent on its
        own. That is the regime a real search sits in, so gating on the naive residual
        would reject exactly the campaigns that matter.
        """
        densities = _densities()
        rng = np.random.default_rng(31)
        quiet = np.concatenate(
            [8.0 * rng.pareto(3.0, 3) + 8.0, rng.exponential(1.2, 600)]
        )
        quiet = quiet[SUPPORT.contains(quiet)]
        posterior = fit_rates(quiet, densities, SUPPORT, clustered=True, n_grid=256)
        result = sum_consistency(assign_pastro(quiet, densities, posterior), posterior)

        assert result["inferred"] < 5.0, "the fixture must be in the low-count regime"
        assert abs(result["fractional"]) < 0.1
        # The residual the old gate used, on the same correct run.
        assert abs(result["raw_difference"] / result["inferred"]) > 0.1








class TestHistogramDensities:
    """
    PyCBC's construction: both densities are normalised histograms.

    ``pycbc.population.fgmc_functions.log_rho_bg`` and ``log_rho_fg``. Nothing is fitted
    and nothing is smoothed. Replaced a kernel estimate over the raw samples, which nobody
    in the field uses and which followed individual samples wherever they were sparse --
    the top of the range, where a detection lives.
    """

    def test_normalised_by_construction(self):
        """
        The integral is exact, not a quadrature result.

        A piecewise-constant density integrates to ``sum(counts) / n_total`` exactly, so
        this is one by construction rather than to a tolerance.
        """
        for density in _densities().values():
            assert density.normalisation() == pytest.approx(1.0, abs=1e-12)

    def test_empty_bin_gets_one_fictitious_count(self):
        """
        An unsampled bin is unmeasured, not empty.

        Zero density there would make the likelihood ratio infinite on the strength of a
        bin nobody sampled. PyCBC puts one count in; the 100 per cent error says so.
        """
        from sage.search.pastro.density import noise_density

        gapped = np.concatenate([np.full(500, 2.0), np.full(500, 12.0)])
        density = noise_density(gapped, SUPPORT, bin_width=0.5)
        probe = np.array([7.0])
        assert np.isfinite(density.log_prob(probe)[0])
        assert density.fractional_error(probe)[0] == 1.0

    def test_occupied_bin_carries_its_poisson_error(self):
        from sage.search.pastro.density import noise_density

        density = noise_density(_noise_samples(), SUPPORT, bin_width=0.5)
        err = density.fractional_error(np.array([1.5]))[0]
        assert 0.0 < err < 1.0

    def test_below_the_first_edge_is_undefined(self):
        """Outside what the density describes, not small: the threshold is a boundary."""
        from sage.search.pastro.density import noise_density

        density = noise_density(_noise_samples(), SUPPORT, bin_width=0.5)
        assert density.log_prob(np.array([SUPPORT.stat_lo - 5.0]))[0] == -np.inf

    def test_above_the_background_is_floored_not_zero(self):
        """
        sgwc-1's ``fill_value=1e-10``.

        Zero would make the mixture likelihood minus infinity at that trigger and no rate
        could be inferred at all; the floor is far below any measured density, so p_astro
        is one to within double precision.
        """
        from sage.search.pastro.density import NOISE_FLOOR, noise_density

        density = noise_density(_noise_samples(), SUPPORT, bin_width=0.5)
        loud = np.array([float(density.edges[-1]) + 3.0])
        assert density.log_prob(loud)[0] == pytest.approx(np.log(NOISE_FLOOR))

    def test_no_chirp_mass_axis(self):
        """PyCBC's construction is one-dimensional; a second axis has no counterpart."""
        from sage.search.pastro.density import noise_density

        with pytest.raises(ValueError, match="ranking statistic alone"):
            noise_density(_noise_samples(), SUPPORT, background_mchirp=np.zeros(3))

    def test_weights_are_refused(self):
        """A histogram counts; the Poisson error it reports is the error on a count."""
        from sage.search.pastro.density import signal_density

        with pytest.raises(ValueError, match="cannot weight"):
            signal_density(_signal_samples(), SUPPORT, weights=np.ones(3))


class TestClusteringGate:
    """Conditions that block the stage rather than annotate it."""

    def test_unclustered_input_is_refused(self):
        """
        An unclustered set inflates every rate by the windows per event.

        Refused in two places: the contract check on the trigger table, and the estimator
        itself, so a caller that bypasses the first still cannot fit through the second.
        """

        class _Table:
            attrs = {"clustered": False}

        with pytest.raises(ContractViolation, match="unclustered"):
            require_clustered(_Table())

        class _Undeclared:
            attrs = {}

        with pytest.raises(ContractViolation, match="does not declare"):
            require_clustered(_Undeclared())

        with pytest.raises(ValueError, match="clustered"):
            fit_rates(_observed(), _densities(), SUPPORT, clustered=False)

    def test_no_reparameterising_policy(self):
        """
        There is deliberately no policy that re-expresses the statistic.

        A monotone regression fitted to the same densities it then reparameterises is a
        place where the answer can come from the model instead of the data.
        """
        report = check_monotonicity(
            _densities()["BBH"], _densities()["Terrestrial"], SUPPORT, tolerance=0.1
        )
        with pytest.raises(ValueError, match="expected report, stop or restrict"):
            apply_policy(report, policy="transform")
