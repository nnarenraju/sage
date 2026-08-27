#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_far.py
Description   : False-alarm rate counting, slides and background validity.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Counting rules are checked against inputs whose answer is known by construction.
See docs/references/arxiv_1508.02357.pdf for the conservative counting convention.

Runs anywhere; needs no data, no GPU and no network.
"""

import numpy as np
import pytest

from sage.search.background import (
    BackgroundSet,
    exclusive_background,
    hierarchical_removal,
    overdispersion_lrt,
)
from sage.search.far import (
    SECONDS_PER_JULIAN_YEAR,
    build_far_curve,
    cumulative_vs_ifar,
    expected_count,
    far_of_stat,
    n_louder,
    p_value_from_ifar,
    poisson_band,
)
from sage.search.geometry import SearchGeometry
from sage.search.segments import Segment
from sage.search.slides import SlidePlan
from sage.search.triggers import TriggerTable, histogram_stats

GEOMETRY = SearchGeometry(
    sample_rate=2048.0,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=205,
    tc_lower_s=5.5,
    tc_upper_s=6.5,
)
BACKGROUND_S = 100.0


def _segments(detector, n_chunks=6, gps0=1238166018.0, chunk_s=4096.0):
    """A short segment list for one detector, in the release's own layout."""
    rate = 2048.0
    nsamples = int(chunk_s * rate)
    offset = 0.0 if detector == "H1" else 137.0
    return [
        Segment(
            segment_index=k,
            detector=detector,
            observing_run="O3a",
            gps_start=gps0 + offset + k * (chunk_s + 600.0),
            gps_end=gps0 + offset + k * (chunk_s + 600.0) + chunk_s,
            sample_rate=rate,
            nsamples=nsamples,
            sample_start_idx=k * nsamples,
            dyn_range_fac=5.902958103587057e20,
            noise_low_freq_cutoff=15.0,
        )
        for k in range(n_chunks)
    ]


def _background(stats=(1.0, 2.0, 3.0, 4.0, 5.0), livetime_s=BACKGROUND_S, **kw):
    return BackgroundSet(
        stats=np.asarray(stats, dtype=float),
        livetime_s=livetime_s,
        n_slides=kw.pop("n_slides", 4),
        removal=kw.pop("removal", "inclusive"),
        **kw,
    )


def _plan(n_slides, seed=11):
    return SlidePlan.build(
        GEOMETRY,
        {"H1": _segments("H1"), "L1": _segments("L1")},
        n_slides=n_slides,
        reference_detector="H1",
        min_separation_s=0.0,
        tau_max_s=4096.0,
        seed=seed,
    )


# Long enough that a candidate louder than the whole background reaches 0.32 per year,
# so the removals below turn on the background count rather than on the ladder's length.
REMOVAL_LIVETIME_S = 1.0e8
REMOVAL_THRESHOLD_PER_YR = 1.0
T0 = 1238166018.0


def _removal_background():
    """
    A background whose three loudest events sit inside one candidate's veto window.

    The hot spot is what gives the procedure something to do: removing the loud
    candidate at ``T0 + 500`` takes those three out, and only then does the quieter one
    at ``T0 + 9000`` clear the threshold. The ordinary events are offset off the lattice
    so none of them falls in that window, keeping the count the veto removes a property
    of the fixture rather than of the spacing.
    """
    rng = np.random.default_rng(7)
    times = T0 + 5.0 + np.arange(400) * 20.0
    stats = 2.0 + 6.0 * rng.random(400)
    times = np.concatenate([times, [T0 + 499.5, T0 + 500.5, T0 + 501.0]])
    stats = np.concatenate([stats, [12.0, 11.0, 10.5]])
    order = np.argsort(times)
    times, stats = times[order], stats[order]
    return BackgroundSet(
        stats=stats,
        livetime_s=REMOVAL_LIVETIME_S,
        n_slides=8,
        removal="inclusive",
        histogram=histogram_stats(stats, clustered=True),
        gps=times,
    )


def _zerolag(stat, gps):
    """A clustered zero-lag candidate list in the schema the removal stages read."""
    return TriggerTable(
        columns={
            "stat": np.asarray(stat, dtype=np.float64),
            "gps": np.asarray(gps, dtype=np.float64),
        },
        attrs={"clustered": True},
    )


class TestCounting:
    """The rate assigned to a statistic."""

    def test_conservative_counting(self):
        """
        With n louder background events in time T the rate is (1 + n) / T.

        The ``1 +`` says a candidate louder than every background event has not been
        shown to have a rate of zero -- the background merely ran out. Without it the
        loudest candidate of any campaign gets an infinite IFAR, which measures how long
        the background ran rather than anything about the candidate.
        """
        background = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert far_of_stat(np.array([3.0]), background, BACKGROUND_S) == pytest.approx(
            4.0 / BACKGROUND_S
        )
        assert far_of_stat(np.array([1.0]), background, BACKGROUND_S) == pytest.approx(
            6.0 / BACKGROUND_S
        )

    def test_counting_identity_holds(self):
        """
        ``far(s) * T - n_above(s) == 1`` at every statistic, exactly.

        The identity rather than a few hand-checked points: an implementation that
        special-cases the empty count, or adds the one only when nothing is louder,
        passes a spot check at the ends and fails here in the middle.
        """
        background = np.array([1.0, 2.0, 2.0, 3.5, 4.0, 5.0])
        queries = np.linspace(-1.0, 7.0, 200)
        counts = np.array([np.count_nonzero(background >= q) for q in queries])
        # The count itself is exact; the identity is asserted on it rather than on the
        # rate, since dividing by T and multiplying back is not a float64 round trip.
        assert n_louder(queries, background).tolist() == counts.tolist()
        far = far_of_stat(queries, background, BACKGROUND_S)
        assert np.allclose(far * BACKGROUND_S - counts, 1.0, rtol=0, atol=1e-12)

    def test_above_all_background(self):
        """A statistic beyond every background event still gets a finite rate, 1 / T."""
        far = far_of_stat(np.array([1e6]), np.array([1.0, 2.0]), BACKGROUND_S)
        assert far == pytest.approx(1.0 / BACKGROUND_S)
        assert np.isfinite(far).all()

    def test_ties_counted_at_or_above(self):
        """
        Background events equal to the candidate count toward it.

        An event exactly as loud is evidence the noise reaches that value, so excluding
        it makes the rate too small at precisely the statistic being asked about.
        """
        background = np.array([1.0, 3.0, 3.0, 3.0, 5.0])
        assert n_louder(np.array([3.0]), background)[0] == 4
        assert far_of_stat(np.array([3.0]), background, BACKGROUND_S) == pytest.approx(
            5.0 / BACKGROUND_S
        )

    def test_monotonic_in_statistic(self):
        """A louder candidate never receives a higher rate."""
        rng = np.random.default_rng(0)
        background = rng.normal(size=2000)
        queries = np.sort(rng.normal(size=500))
        assert np.all(np.diff(far_of_stat(queries, background, BACKGROUND_S)) <= 0)

    def test_uses_clustered_background_only(self):
        """
        An unclustered background is refused rather than silently counted.

        The count is the FAR numerator, and an unclustered trigger train contributes one
        event per window of a glitch instead of one per glitch. Every rate is then wrong
        by that multiplicity while the numbers stay entirely ordinary -- the failure that
        invalidated the reference analysis.
        """
        raw = histogram_stats(np.array([1.0, 2.0]), clustered=False)
        with pytest.raises(ValueError, match="clustered"):
            BackgroundSet(
                stats=np.array([1.0, 2.0]),
                livetime_s=BACKGROUND_S,
                n_slides=2,
                removal="inclusive",
                histogram=raw,
            )

    def test_multiplicity_scales_the_rate(self):
        """
        The guard protects a number, not a label.

        A background whose events are each repeated k times gives a rate k times higher.
        Testing the flag alone would pass for an implementation where ``clustered`` is a
        boolean anyone can set; this pins the count it stands for.
        """
        background = np.array([1.0, 2.0, 3.0, 4.0])
        smeared = np.repeat(background, 4)
        clean = far_of_stat(np.array([2.0]), background, BACKGROUND_S)[0]
        dirty = far_of_stat(np.array([2.0]), smeared, BACKGROUND_S)[0]
        assert (dirty * BACKGROUND_S - 1) / (clean * BACKGROUND_S - 1) == pytest.approx(
            4.0
        )

    def test_nan_candidate_refused(self):
        """
        A NaN ranking statistic is refused rather than assigned the smallest rate.

        NaN compares false against everything, so searchsorted places it past the end of
        the background and it counts as louder than every event there -- collecting
        ``1 / T_b``, the smallest FAR the search can assign. A window the network could
        not rank would arrive at the top of the candidate list as the most significant
        thing in the campaign, and every number about it would look ordinary.
        """
        background = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="NaN"):
            far_of_stat(np.array([2.0, np.nan]), background, BACKGROUND_S)

        # The direction of the bug, shown rather than asserted about: without the guard
        # the NaN outranks the whole background.
        assert n_louder(np.array([np.nan]), background)[0] == 0
        assert n_louder(np.array([1e9]), background)[0] == 0

    def test_zero_livetime_refused(self):
        """Dividing by a livetime of zero is a configuration error, not an infinity."""
        with pytest.raises(ValueError, match="livetime"):
            far_of_stat(np.array([1.0]), np.array([1.0]), 0.0)


class TestFarCurve:
    """The persisted statistic-to-rate mapping."""

    def test_curve_matches_direct_counting(self):
        """The curve agrees with counting the background directly at every knot."""
        background = _background()
        curve = build_far_curve(background, foreground_livetime_s=50.0)
        direct = far_of_stat(curve.stat, background.stats, background.livetime_s)
        assert np.allclose(curve.far_per_yr, direct * SECONDS_PER_JULIAN_YEAR)

    def test_ifar_is_capped(self):
        """
        An IFAR beyond the background that measured it is not reported as measured.

        Past the loudest background event the rate is bounded by ``1 / T_b``, so a larger
        inverse rate describes how long the background ran, not the candidate. The
        background here is deep enough for that bound to exceed the cap, which is the
        only regime where the cap does anything.
        """
        deep = _background(livetime_s=1e12)
        uncapped = 1.0 / (
            far_of_stat(np.array([1e6]), deep.stats, deep.livetime_s)[0]
            * SECONDS_PER_JULIAN_YEAR
        )
        assert uncapped > 10.0, "the fixture must reach past the cap to test it"

        curve = build_far_curve(deep, 50.0, ifar_cap_yr=10.0)
        assert curve.ifar_of(np.array([1e6]))[0] == pytest.approx(10.0)
        # Below the cap the value is passed through untouched.
        shallow = build_far_curve(_background(), 50.0, ifar_cap_yr=1e6)
        assert shallow.ifar_of(np.array([5.0]))[0] < 1e6

    def test_extrapolation_flagged(self):
        """Candidates past the measured background carry a flag, so tables can say so."""
        curve = build_far_curve(_background(), 50.0)
        assert curve.is_extrapolated(np.array([2.0, 5.0, 5.5])).tolist() == [
            False,
            False,
            True,
        ]

    def test_curve_is_monotone(self):
        """Interpolation preserves the ordering the counting guarantees."""
        rng = np.random.default_rng(8)
        curve = build_far_curve(_background(stats=rng.normal(size=500)), 50.0)
        queries = np.linspace(curve.stat[0] - 1.0, curve.stat[-1] + 1.0, 500)
        assert np.all(np.diff(curve.far_of(queries)) <= 1e-12)

    def test_empty_background_refused(self):
        """No background events means no measured rate, and no curve to pretend one."""
        with pytest.raises(ValueError, match="no events"):
            build_far_curve(_background(stats=[]), 50.0)


class TestLivetime:
    """Background time is measured, not inferred."""

    def test_background_time_is_the_slide_sum(self):
        """
        Total background time equals the sum over slides, not slides times zero-lag.

        Per-slide retention falls with lag, so the closed form always overstates -- and
        an overstated denominator divides the false-alarm count by too much, reporting
        every rate too low.
        """
        plan = _plan(20)
        slid = [s for s in plan.slides if s.slide_id != 0]
        assert plan.background_livetime_s == pytest.approx(
            sum(s.livetime_s for s in slid), abs=1e-9
        )
        assert plan.background_livetime_s < 20 * plan.foreground_livetime_s

    def test_retention_falls_with_lag(self):
        """
        Larger lags retain less coincident time; the plan records each.

        Asserted as a trend across the ladder, not slide by slide. A segment list with
        gaps in it -- which every real one has -- makes retention locally non-monotone:
        a slightly longer lag can move a stretch of one detector back into a gap of the
        other. The trend is the physics; strict monotonicity is a property of a fixture
        with no gaps.
        """
        plan = _plan(40)
        slid = sorted(plan.slides[1:], key=lambda s: abs(s.offsets_s["L1"]))
        livetimes = np.array([s.livetime_s for s in slid])
        quarter = len(livetimes) // 4
        assert livetimes[:quarter].mean() > livetimes[-quarter:].mean()
        assert livetimes[-1] < livetimes[0]

    def test_zero_lag_excluded_from_ladder(self):
        """The ladder never contains a zero offset."""
        for slide in _plan(30).slides[1:]:
            assert slide.offsets_s["L1"] != 0.0

    def test_minimum_separation_respected(self):
        """Every lag exceeds the window content plus light travel plus the guard."""
        plan = _plan(30)
        assert plan.min_separation_s >= GEOMETRY.signal_length_s
        for slide in plan.slides[1:]:
            assert abs(slide.offsets_s["L1"]) >= plan.min_separation_s

    def test_lags_are_stride_multiples(self):
        """A slid window lands on the same lattice as the unslid one."""
        stride_s = GEOMETRY.stride_s
        for slide in _plan(30).slides[1:]:
            multiple = slide.offsets_s["L1"] / stride_s
            assert multiple == pytest.approx(round(multiple), abs=1e-9)


class TestExpectedBackground:
    """The expected curve and its bands."""

    def test_expected_count_is_time_over_ifar(self):
        """
        The expected cumulative count follows from the analysed time alone.

        That is what an inverse false-alarm *rate* means, so the curve is a prediction
        the foreground is compared against and never something fitted to it.
        """
        one_year = SECONDS_PER_JULIAN_YEAR
        assert expected_count(np.array([1.0, 10.0, 100.0]), 5 * one_year).tolist() == [
            5.0,
            0.5,
            0.05,
        ]

    def test_poisson_bands_match_quantiles(self):
        """
        Shaded bands are the Poisson quantiles about the expectation.

        Not ``mu +/- n*sqrt(mu)``: in the tail the expectation is far below one, where
        the Gaussian approximation puts the lower edge below zero and gives a band that
        excludes the integers the count can actually take. The tail is the part of the
        plot the figure exists for.
        """
        scipy_stats = pytest.importorskip("scipy.stats")
        expected = np.array([0.01, 0.5, 5.0, 100.0])
        lower, upper = poisson_band(expected, sigma=1)
        tail = 0.5 * (1.0 - (scipy_stats.norm.cdf(1) - scipy_stats.norm.cdf(-1)))
        assert lower.tolist() == scipy_stats.poisson.ppf(tail, expected).tolist()
        assert upper.tolist() == scipy_stats.poisson.ppf(1 - tail, expected).tolist()
        assert np.all(lower >= 0.0)

    def test_bands_widen_with_sigma(self):
        """Three sigma contains one sigma at every expectation."""
        pytest.importorskip("scipy.stats")
        expected = np.array([0.1, 1.0, 20.0])
        narrow, wide = poisson_band(expected, 1), poisson_band(expected, 3)
        assert np.all(wide[0] <= narrow[0])
        assert np.all(wide[1] >= narrow[1])

    def test_cumulative_descends_in_ifar(self):
        """The observed curve is the candidate list, ordered loudest first."""
        pytest.importorskip("scipy.stats")
        curve = build_far_curve(_background(), SECONDS_PER_JULIAN_YEAR)
        out = cumulative_vs_ifar(np.array([1.5, 4.5, 3.0]), curve)
        assert np.all(np.diff(out["ifar_yr"]) <= 0)
        assert out["observed"].tolist() == [1, 2, 3]

    def test_p_value_survives_a_confident_candidate(self):
        """
        A tiny expected count gives a tiny p-value, not exactly zero.

        ``1 - exp(-x)`` cancels to zero in float64 near ``x ~ 1e-17``, which would report
        a p-value of zero for a candidate whose significance is finite and worth quoting.
        """
        p = p_value_from_ifar(np.array([1e18]), SECONDS_PER_JULIAN_YEAR)
        assert 0.0 < p[0] < 1e-17

    def test_p_value_matches_definition(self):
        """One year against a one-year IFAR is ``1 - 1/e``."""
        p = p_value_from_ifar(np.array([1.0]), SECONDS_PER_JULIAN_YEAR)
        assert p[0] == pytest.approx(1.0 - np.exp(-1.0))

    def test_overdispersion_detected(self):
        """
        Clustered counts are separated from Poisson ones, in both directions.

        Order-statistic counting of a FAR assumes the background events arrive as a
        Poisson process; if they arrive in bursts the quoted rate understates how often
        the noise reaches a given statistic. Both halves are asserted here because a
        test made only on clustered counts passes for a check that always reports
        over-dispersion, which would condemn every valid background the search makes.
        """
        pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(4)
        alpha, mean = 0.5, 20.0
        rates = rng.gamma(shape=1.0 / alpha, scale=alpha * mean, size=400)
        clustered = overdispersion_lrt(rng.poisson(rates))
        plain = np.random.default_rng(3).poisson(mean, size=400)
        poissonian = overdispersion_lrt(plain)

        assert clustered["overdispersed"] is True
        assert clustered["p_value"] < 1e-6
        assert clustered["alpha"] == pytest.approx(alpha, rel=0.3)
        assert poissonian["overdispersed"] is False
        assert poissonian["p_value"] > 0.05
        assert poissonian["index_of_dispersion"] == pytest.approx(1.0, rel=0.2)


class TestHierarchicalRemoval:
    """What removing foreground contamination does to the published rate."""

    LOUD_GPS = T0 + 500.0
    QUIET_GPS = T0 + 6000.0

    def _background(self, plan):
        """
        A background on the real ladder carrying both frames of contamination.

        One real signal contributes a copy per slide. Seen in the reference detector those
        copies all land on the same reference time; seen in the follower they land on a
        different reference time in every slide, since the follower's data is read at
        ``gps + offset``. Both families are here so a reference-frame-only veto cannot
        pass.
        """
        slid = [s.slide_id for s in plan if s.slide_id != 0]
        offsets = {int(s.slide_id): dict(s.offsets_s) for s in plan}
        rng = np.random.default_rng(7)
        times, stats, ids = [], [], []
        for slide_id in slid:
            n = 100
            times.append(T0 + 10.0 + np.arange(n) * 37.0)
            stats.append(2.0 + 6.0 * rng.random(n))
            ids.append(np.full(n, slide_id))
        for rank, slide_id in enumerate(slid[:3]):
            times.append(np.array([self.LOUD_GPS]))
            stats.append(np.array([12.0 - 0.5 * rank]))
            ids.append(np.array([slide_id]))
        for rank, slide_id in enumerate(slid[:3]):
            shift = float(offsets[int(slide_id)].get("L1", 0.0))
            times.append(np.array([self.LOUD_GPS - shift]))
            stats.append(np.array([11.9 - 0.5 * rank]))
            ids.append(np.array([slide_id]))
        times, stats = np.concatenate(times), np.concatenate(stats)
        ids = np.concatenate(ids)
        order = np.argsort(times)
        return BackgroundSet(
            stats=stats[order],
            livetime_s=plan.background_livetime_s,
            n_slides=len(slid),
            removal="inclusive",
            histogram=histogram_stats(stats[order], clustered=True),
            gps=times[order],
            tc_gps=times[order],
            slide_id=ids[order],
        )

    def _zerolag_pair(self):
        gps = np.array([self.LOUD_GPS, self.QUIET_GPS])
        return TriggerTable(
            columns={
                "stat": np.array([20.0, 10.0]),
                "gps": gps,
                "tc_gps": gps,
            },
            attrs={"clustered": True},
        )

    def test_removal_raises_the_quieter_candidate(self):
        """
        Taking a detection out of the background raises everything behind it.

        Stated in the quantity the search publishes. A real signal left in the background
        raises the apparent noise rate and suppresses the significance of every quieter
        candidate; removing it is what lets the second candidate be assessed against noise
        rather than against the first candidate's slid copies.
        """
        plan = _plan(4)
        inclusive = self._background(plan)
        removed = hierarchical_removal(
            inclusive,
            self._zerolag_pair(),
            plan,
            GEOMETRY,
            {"H1": _segments("H1"), "L1": _segments("L1")},
            min_background_livetime_s=0.0,
        )

        before = build_far_curve(inclusive, SECONDS_PER_JULIAN_YEAR)
        after = build_far_curve(removed, SECONDS_PER_JULIAN_YEAR)
        quiet = np.array([10.0])

        assert removed.removal == "hierarchical"
        assert removed.removed_gps.size >= 1
        assert after.ifar_of(quiet)[0] > before.ifar_of(quiet)[0]

    def test_removal_is_order_independent(self):
        """
        The result does not depend on the order the candidates were supplied in.

        Each removal changes the background the next candidate is assessed against, so an
        implementation that walked the table in storage order would give a different
        background -- and a different significance for every candidate -- according to
        which job happened to write the rows first.
        """
        plan = _plan(4)
        inclusive = self._background(plan)
        network = {"H1": _segments("H1"), "L1": _segments("L1")}
        forward = self._zerolag_pair()
        reversed_table = TriggerTable(
            columns={k: v[::-1].copy() for k, v in forward.columns.items()},
            attrs=dict(forward.attrs),
        )
        kw = dict(min_background_livetime_s=0.0)
        first = hierarchical_removal(
            inclusive, forward, plan, GEOMETRY, network, **kw
        )
        second = hierarchical_removal(
            inclusive, reversed_table, plan, GEOMETRY, network, **kw
        )

        assert first.removed_gps.tolist() == second.removed_gps.tolist()
        assert first.stats.tolist() == second.stats.tolist()
        assert first.livetime_s == second.livetime_s

    def test_hierarchical_is_bracketed(self):
        """
        The hierarchical count lies between the exclusive and the inclusive one.

        The ordering is structural **at a common window**: hierarchical vetoes on the
        subset of candidates that pass the louder-than gate, exclusive on all of them.
        The two defaults are different numbers -- PyCBC's ``--veto-window`` and
        ``--hierarchical-removal-window`` -- so the window is passed explicitly here. At
        the defaults the wider hierarchical window can reach background the narrower
        exclusive one left, and the bracket is not claimed.
        """
        plan = _plan(4)
        inclusive = self._background(plan)
        network = {"H1": _segments("H1"), "L1": _segments("L1")}
        zerolag = self._zerolag_pair()
        window = 1.0
        exclusive = exclusive_background(
            inclusive, zerolag, plan, GEOMETRY, network, window_s=window
        )
        hierarchical = hierarchical_removal(
            inclusive, zerolag, plan, GEOMETRY, network, window_s=window
        )
        for probe in (3.0, 5.0, 7.0, 9.0, 11.0, 13.0):
            assert (
                exclusive.n_above(probe)
                <= hierarchical.n_above(probe)
                <= inclusive.n_above(probe)
            )


class TestCountedRate:
    """
    The curve is the count, and nothing continues it.

    The generalised-Pareto continuation this class used to exercise is gone: it was used
    only by the p_astro noise density, where extrapolating it past the loudest background
    inverted the likelihood ratio. See SB-64.
    """

    def _curve(self):
        from sage.search.far import build_far_curve

        background = _background(np.random.default_rng(3).exponential(1.0, 4000))
        return build_far_curve(background, foreground_livetime_s=1000.0)

    def test_saturates_at_the_counting_floor(self):
        """
        Above the loudest background event every statistic reports the same rate.

        That floor is (1 + 1) / T_b: the counting is inclusive, so the loudest background
        event counts itself. Nothing separates candidates above it, and nothing pretends to.
        """
        curve = self._curve()
        loudest = float(curve.stat[-1])
        floor = 2.0 / curve.background_livetime_s * SECONDS_PER_JULIAN_YEAR
        assert curve.far_of(np.array([loudest + 1.0]))[0] == pytest.approx(floor)
        assert curve.far_of(np.array([loudest + 50.0]))[0] == pytest.approx(floor)

    def test_extrapolation_is_marked(self):
        """A reader must be able to tell the floor from a measurement."""
        curve = self._curve()
        loudest = float(curve.stat[-1])
        probe = np.array([loudest - 1.0, loudest + 1.0])
        assert curve.is_extrapolated(probe).tolist() == [False, True]

    def test_no_extrapolated_rate_exists(self):
        """The method is gone, not merely disabled."""
        curve = self._curve()
        assert not hasattr(curve, "far_extrapolated_of")
        assert not hasattr(curve, "ifar_extrapolated_of")
        assert not hasattr(curve, "tail")
