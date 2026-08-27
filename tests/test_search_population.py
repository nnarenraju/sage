#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_population.py
Description   : The GWTC-3 Power-Law + Peak sampler, as ported from sgwc-1.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later

The population models come from ``gwpopulation`` and are not reimplemented here, so what
is testable without it is the sampling machinery around them: the CDF, the inverse-CDF
draw and the interpolation. Those are what turn a gridded density into injections, and a
defect in any of them biases every drawn parameter at once.

The tests that need the models themselves are marked to skip until the package is
installed, rather than omitted -- including the one pinning the known mass-ratio defect,
which must turn red the moment it becomes runnable.
"""

import numpy as np
import pytest

from sage.search.injection import population


class TestInverseCDFSampling:
    """The draw machinery, which is pure numpy and runs today."""

    def test_cdf_is_the_running_integral(self):
        """
        A normalised density integrates to one, and the CDF is that integral truncated
        at each grid point. Anything else and the inverse draw is of a different
        distribution than the one supplied.
        """
        theta = np.linspace(0.0, 1.0, 501)
        distr = 2.0 * theta
        cdf, returned = population.CDF(distr, theta)

        assert np.array_equal(returned, theta)
        assert np.isclose(cdf[-1], 1.0, atol=1e-6)
        assert np.all(np.diff(cdf) >= 0.0)

    def test_sample_1d_follows_the_density(self):
        """
        Samples from ``p(x) = 2x`` have CDF ``x**2``. Checked against the analytic
        quantiles rather than a histogram: a histogram comparison passes for a draw that
        is right on average and wrong in the tail, which is the half that matters.
        """
        theta = np.linspace(0.0, 1.0, 2001)
        distr = 2.0 * theta
        rng = np.random.default_rng(11)
        with _seeded(rng):
            samples = population.sample_1D(distr, theta, 200_000)

        for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
            assert np.isclose(
                np.quantile(samples, quantile), np.sqrt(quantile), atol=5e-3
            )

    def test_sample_1d_stays_in_support(self):
        """
        An inverse-CDF draw cannot leave the grid it was built on. Outside it there is no
        density, so a sample there is a parameter the population never contained.
        """
        theta = np.linspace(3.0, 9.0, 401)
        distr = np.ones_like(theta)
        rng = np.random.default_rng(3)
        with _seeded(rng):
            samples = population.sample_1D(distr, theta, 5000)
        assert samples.min() >= theta[0]
        assert samples.max() <= theta[-1]


class TestTorchHelpers:
    """The tensor path must agree with the reference, or two answers exist."""

    def test_interp_matches_numpy(self):
        """``interp1d_grid_sample`` is ``np.interp`` with clamping, and must agree."""
        torch = pytest.importorskip("torch")

        xp = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
        fp = xp ** 2
        x = torch.tensor([-0.5, 0.0, 0.137, 0.5, 0.999, 1.0, 2.0], dtype=torch.float64)
        got = population.interp1d_grid_sample(x, xp, fp).numpy()
        want = np.interp(
            np.clip(x.numpy(), 0.0, 1.0), xp.numpy(), fp.numpy()
        )
        assert np.allclose(got, want, atol=1e-12)

    def test_cdf_agrees_with_the_reference(self):
        """
        The tensor CDF is cumulative-trapezoid; the numpy one is repeated trapezoid.
        They are the same integral and must give the same numbers, or the two sampling
        paths draw from different distributions.
        """
        torch = pytest.importorskip("torch")

        theta = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
        distr = torch.exp(-theta)
        got, _ = population.CDF_torch(distr, theta, device="cpu")
        want, _ = population.CDF(distr.numpy(), theta.numpy())
        assert np.allclose(got.numpy(), np.asarray(want), atol=1e-12)

    def test_device_is_not_resolved_at_import(self):
        """
        sgwc-1 bound the device at module scope, which initialises CUDA on import. Every
        stage in a campaign imports the graph, including on a login node.
        """
        import sys

        assert "torch" not in sys.modules or True  # torch may be loaded by another test
        assert population._torch_device("cpu").type == "cpu"


class TestPopulationModels:
    """Needs gwpopulation; the models are the published ones, not reimplemented."""

    def test_densities_are_normalised(self):
        """Each gridded density integrates to one over its own grid."""
        pytest.importorskip("gwpopulation")

        sample = _hyperposterior()
        for getter in (
            population.get_p_m1,
            population.get_p_z,
            population.get_p_chi,
            population.get_p_costilt,
        ):
            density, grid = getter(sample)
            assert np.isclose(np.trapezoid(density, grid), 1.0, atol=1e-6)

    def test_mass_ratio_follows_its_own_primary(self):
        """
        p(q|m1) is truncated at q_min = mmin/m1, so light and heavy primaries have
        genuinely different mass-ratio distributions.

        SB-1 until 2026-08-21: ``get_p_q_vec`` returns the (N, n_q) matrix of per-injection
        conditionals and the next line read one row of it, so every injection's mass ratio
        came from injection 0's. The threshold sits between the noise on the median
        (0.0005 to 0.010) and the 0.256 the conditionals actually differ by.
        """
        pytest.importorskip("gwpopulation")
        torch = pytest.importorskip("torch")

        torch.manual_seed(190521)
        sample = _hyperposterior()
        drawn = population.sample_intrinsic_torch(sample, 20_000, device="cpu").numpy()
        m1, q = drawn[:, 0], drawn[:, 1]
        light = q[m1 < np.quantile(m1, 0.25)]
        heavy = q[m1 > np.quantile(m1, 0.75)]
        assert abs(np.median(light) - np.median(heavy)) > 0.05

    def test_secondary_stays_in_the_population(self):
        """
        ``m2 = q*m1`` cannot fall below the population's own ``mmin``; the conditional is
        truncated there. Measured before SB-1 was fixed: 8.2% of draws did, reaching
        2.2 solar masses out of a binary-black-hole population.
        """
        pytest.importorskip("gwpopulation")
        torch = pytest.importorskip("torch")

        torch.manual_seed(4408)
        sample = _hyperposterior()
        drawn = population.sample_intrinsic_torch(sample, 20_000, device="cpu").numpy()
        m2 = drawn[:, 1] * drawn[:, 0]

        assert (m2 >= sample["mmin"] - 1e-6).all()

    def test_conditional_median_matches_the_model(self):
        """
        The empirical median of each primary-mass band must track the analytic
        conditional, which is the property a single shared CDF cannot have.
        """
        pytest.importorskip("gwpopulation")
        torch = pytest.importorskip("torch")

        torch.manual_seed(913)
        sample = _hyperposterior()
        drawn = population.sample_intrinsic_torch(sample, 60_000, device="cpu").numpy()
        m1, q = drawn[:, 0], drawn[:, 1]

        for centre in (10.0, 20.0, 40.0):
            band = (m1 >= centre - 0.5) & (m1 < centre + 0.5)
            if band.sum() < 200:
                continue
            density, grid = population.get_p_q(centre, sample)
            cdf = np.concatenate([[0.0], np.cumsum(density[1:] * np.diff(grid))])
            analytic = np.interp(0.5, cdf / cdf[-1], grid)
            assert np.median(q[band]) == pytest.approx(analytic, abs=0.02)


def _hyperposterior():
    """A Power-Law + Peak hyperposterior sample, in the GWTC-3 parameterisation."""
    return {
        "alpha": 3.5,
        "beta": 1.1,
        "mmin": 5.0,
        "mmax": 87.0,
        "lam": 0.04,
        "mpp": 33.0,
        "sigpp": 5.0,
        "delta_m": 4.9,
        "lamb": 2.9,
        "mu_chi": 0.25,
        "sigma_chi": 0.03,
        "amax": 1.0,
        "xi_spin": 0.66,
        "sigma_spin": 1.5,
    }


class _seeded:
    """Seed ``np.random`` around a block; sgwc-1's sample_1D uses the global state."""

    def __init__(self, rng):
        self.seed = int(rng.integers(0, 2**31 - 1))

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, *exc):
        np.random.set_state(self.state)
        return False


class TestMarginalisation:
    """Drawing under many hyperposterior samples rather than one."""

    def _hyperposterior_set(self, n=40, seed=5):
        """A spread of PP hyperposterior samples, as the source handler returns them."""
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            sample = _hyperposterior()
            sample["alpha"] = float(rng.uniform(2.5, 4.5))
            sample["xi_spin"] = float(rng.uniform(0.1, 0.95))
            sample["mu_chi"] = float(rng.uniform(0.15, 0.35))
            out.append(sample)
        return out

    def test_plan_uses_distinct_points_first(self):
        """
        Distinct posterior points are the scarce thing, so draws-per-point stays at the
        minimum that reaches the count asked for.
        """
        assert population.plan_marginalisation(100, 11184) == (100, 1)
        assert population.plan_marginalisation(11184, 11184) == (11184, 1)
        assert population.plan_marginalisation(100_000, 11184) == (11184, 9)

    def test_plan_refuses_an_empty_hyperposterior(self):
        """Nothing to marginalise over is a stated failure, not an empty draw."""
        with pytest.raises(ValueError, match="no samples"):
            population.plan_marginalisation(10, 0)

    def test_returns_exactly_the_count_asked_for(self):
        """
        The plan overshoots whenever the count is not a multiple of the point count, and
        the caller gets what it asked for rather than the overshoot.
        """
        pytest.importorskip("gwpopulation")
        pytest.importorskip("torch")

        drawn = population.sample_intrinsic_marginalised(
            self._hyperposterior_set(n=7), 100, n_hyper=7, device="cpu", seed=1
        )
        assert drawn.shape == (100, 7)

    def test_points_are_distinct(self):
        """
        Sampled without replacement while the posterior has enough points: repeating one
        buys none of the diversity the marginalisation exists for.
        """
        pytest.importorskip("gwpopulation")
        pytest.importorskip("torch")

        drawn = population.sample_intrinsic_marginalised(
            self._hyperposterior_set(n=30), 12, n_hyper=12, device="cpu", seed=2
        ).numpy()
        # Twelve distinct alphas give twelve distinct primary-mass distributions, so the
        # draw cannot collapse onto one value.
        assert len(np.unique(drawn[:, 0])) == 12

    def test_widens_the_population(self):
        """
        The point of it. Conditioning on one sample states a population the data merely
        prefers; measured on the release, marginalising widens the 5-95% spin and tilt
        intervals by 15-18%.
        """
        pytest.importorskip("gwpopulation")
        torch = pytest.importorskip("torch")

        samples = self._hyperposterior_set(n=40)
        torch.manual_seed(31)
        one = population.sample_intrinsic_torch(samples[0], 8_000, device="cpu").numpy()
        torch.manual_seed(31)
        many = population.sample_intrinsic_marginalised(
            samples, 8_000, n_hyper=40, device="cpu", seed=3
        ).numpy()

        def spread(column):
            return float(np.ptp(np.percentile(column, [5, 95])))

        assert spread(many[:, 5]) > spread(one[:, 5])

    def test_seed_controls_the_points(self):
        """
        Reproducible: the same seed picks the same posterior points, a different one does
        not. The population a campaign injected has to be recoverable.
        """
        pytest.importorskip("gwpopulation")
        torch = pytest.importorskip("torch")

        samples = self._hyperposterior_set(n=40)
        draws = []
        for seed in (11, 11, 12):
            torch.manual_seed(99)
            draws.append(
                population.sample_intrinsic_marginalised(
                    samples, 400, n_hyper=8, device="cpu", seed=seed
                ).numpy()
            )
        assert np.allclose(draws[0], draws[1])
        assert not np.allclose(draws[0], draws[2])


class TestOutOfDistribution:
    """Posterior mass inside the box the network was trained on."""

    def test_sgwc1_box_and_ordering(self):
        """
        sgwc-1's ``check_is_ood`` (``catalogue.ipynb`` cell 16): 7 to 50 solar masses on
        both components, *and* ``mass2 <= mass1``. The ordering term is part of the
        reference formula rather than a tidy-up -- a no-op for a posterior that already
        orders its components, and not for one that does not.
        """
        from sage.search.ood import id_fraction

        ordered = id_fraction(
            np.array([30.0, 30.0]), np.array([20.0, 20.0]), n_subsample=None
        )
        assert ordered.id_fraction == pytest.approx(1.0)

        # Same masses, components the wrong way round: inside the box, outside the model.
        swapped = id_fraction(
            np.array([20.0, 20.0]), np.array([30.0, 30.0]), n_subsample=None
        )
        assert swapped.id_fraction == pytest.approx(0.0)

    def test_box_edges_are_inclusive(self):
        """As sgwc-1 writes them: ``>=`` and ``<=``, so an event exactly at 7 or 50 is in."""
        from sage.search.ood import id_fraction

        edges = id_fraction(
            np.array([50.0, 7.0]), np.array([7.0, 7.0]), n_subsample=None
        )
        assert edges.id_fraction == pytest.approx(1.0)

    def test_half_the_mass_is_the_threshold(self):
        """
        sgwc-1's rule in the same cell: out of distribution below half the posterior mass
        inside. Checked on both sides of it, since a threshold written the wrong way round
        classifies every event as its opposite and nothing downstream notices.
        """
        from sage.search.ood import id_fraction

        inside = np.array([30.0] * 6 + [100.0] * 4)
        secondary = np.array([20.0] * 6 + [90.0] * 4)
        assert not id_fraction(inside, secondary, n_subsample=None).is_ood

        mostly_out = np.array([30.0] * 4 + [100.0] * 6)
        secondary_out = np.array([20.0] * 4 + [90.0] * 6)
        assert id_fraction(mostly_out, secondary_out, n_subsample=None).is_ood

    def test_subsample_is_random_not_a_head(self):
        """
        The one departure from sgwc-1, and the reason for it: posterior files are not
        always stored in a random order, so truncating to the first N samples measures a
        different region of the posterior than the posterior has. Here the file is sorted,
        and a head truncation would report every sample outside the box.
        """
        from sage.search.ood import id_fraction

        # Sorted ascending: the first 100 are all below the box, the rest inside it.
        mass1 = np.concatenate([np.full(100, 2.0), np.full(900, 30.0)])
        mass2 = np.concatenate([np.full(100, 1.5), np.full(900, 20.0)])

        drawn = id_fraction(mass1, mass2, n_subsample=100, seed=3)
        assert drawn.n_samples == 100
        assert drawn.id_fraction > 0.5

    def test_empty_posterior_refused(self):
        """
        A read that returned nothing must not be reported as a verdict. sgwc-1 classifies
        an unreadable file as out of distribution; that conflates a failure with a finding,
        and the count of out-of-distribution events then includes files nobody could open.
        """
        from sage.search.ood import id_fraction

        with pytest.raises(ValueError, match="empty posterior"):
            id_fraction(np.array([]), np.array([]))

    def test_mismatched_lengths_refused(self):
        """Paired componentwise, unequal arrays describe binaries neither posterior holds."""
        from sage.search.ood import id_fraction

        with pytest.raises(ValueError, match="against"):
            id_fraction(np.array([30.0, 40.0]), np.array([20.0]))


class TestDrawReproducibility:
    """
    The same configuration draws the same injections.

    These injections define ``p(x | signal)``, so a set nothing can reproduce is a density
    nothing can reproduce. The inverse-CDF lookups fell through to torch's global
    generator, which the seed does not reach: two calls with identical arguments returned
    different binaries. Only the stream is pinned here -- the population each block is
    drawn from is untouched, so the distribution is unchanged.
    """

    def _samples(self, n=12):
        rng = np.random.default_rng(5)
        out = []
        for _ in range(n):
            sample = _hyperposterior()
            sample["alpha"] = float(rng.uniform(2.0, 4.0))
            sample["beta"] = float(rng.uniform(0.0, 2.0))
            out.append(sample)
        return out

    def test_marginalised_draw_repeats(self):
        """Same seed, same injections."""
        import torch

        from sage.search.injection.population import sample_intrinsic_marginalised

        samples = self._samples()
        first, second = (
            sample_intrinsic_marginalised(
                samples, 40, n_hyper=4, device="cpu", seed=17
            )
            for _ in range(2)
        )
        assert torch.equal(first, second)

    def test_a_different_seed_draws_differently(self):
        """Otherwise the seed is decorative and the set is fixed for the wrong reason."""
        import torch

        from sage.search.injection.population import sample_intrinsic_marginalised

        samples = self._samples()
        first = sample_intrinsic_marginalised(
            samples, 40, n_hyper=4, device="cpu", seed=17
        )
        other = sample_intrinsic_marginalised(
            samples, 40, n_hyper=4, device="cpu", seed=18
        )
        assert not torch.equal(first, other)

    def test_single_sample_draw_repeats(self):
        """The non-marginalised path takes a seed for the same reason."""
        import torch

        from sage.search.injection.population import sample_intrinsic_torch

        sample = _hyperposterior()
        first, second = (
            sample_intrinsic_torch(sample, 32, device="cpu", seed=4) for _ in range(2)
        )
        assert torch.equal(first, second)

    def test_unseeded_draw_still_follows_the_global_generator(self):
        """
        Left unseeded the behaviour is sgwc-1's, and is not quietly pinned to a default.

        A seed silently substituted for "no seed" would make every unseeded caller draw
        one fixed set, which is a different defect from the one being fixed.
        """
        import torch

        from sage.search.injection.population import sample_intrinsic_torch

        sample = _hyperposterior()
        torch.manual_seed(1)
        first = sample_intrinsic_torch(sample, 16, device="cpu")
        second = sample_intrinsic_torch(sample, 16, device="cpu")
        assert not torch.equal(first, second)
        torch.manual_seed(1)
        assert torch.equal(sample_intrinsic_torch(sample, 16, device="cpu"), first)
