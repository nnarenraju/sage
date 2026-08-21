#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_slides.py
Description   : The time-slide ladder: stratification, all-pairs separation, measured livetime.

Created on 2026-08-13

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Three properties here are the ones an obvious test agrees with while the code is wrong.

Background livetime looks right for any implementation as long as the lags are short,
because retention is then close to one and ``n_slides * T_zerolag`` is close to the truth;
it is only at the depth a low FAR needs that the assumed value parts company with the
measured one. So the livetime is checked against an independently written interval
intersection, at ladder depths spanning three decades.

Lags look right as long as they are distinct, which a ladder packed against the minimum
also is; distinctness says nothing about whether the ladder samples the range. So the
lags are checked as a distribution, and the packed ladder is run through the same checks
as a negative control to show they can fail.

The separation floor looks right for two detectors whatever the implementation does about
pairs, because the only pair involves the reference. So the three-detector case is checked
with a lag vector that a reference-only implementation accepts.

Runs on synthetic segments; needs no data and no GPU. The light-travel floor needs pycbc,
which supplies the detector geometry.
"""

import dataclasses

import numpy as np
import pytest
from scipy import stats

from sage.search.geometry import SearchGeometry
from sage.search.segments import Segment
from sage.search import slides
from sage.search.slides import (
    SlidePlan,
    minimum_separation_s,
    pairwise_separations_ok,
    remeasure_livetimes,
    stratified_lags,
)

RATE = 2048.0
CHUNK_S = 512.0
OVERLAP_S = 15.5994

# Exact light-travel times between the LIGO and Virgo sites, in seconds.
H1L1 = 0.010012846152267725
H1V1 = 0.027287979933397113

# A 1 s stride rather than the production 205 samples: a thousand-slide ladder is
# measured window by window several times over in this file, and the cost of that is set
# by the window count. Nothing asserted here depends on the stride except through the lag
# lattice it defines, which is checked explicitly.
GEOMETRY = SearchGeometry(
    sample_rate=RATE,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=2048,
    tc_lower_s=5.0,
    tc_upper_s=7.0,
)

GPS0 = 1238166018.0

# Wide enough that the deepest ladder still retains about half the zero-lag livetime, so
# retention is measured where it differs from one rather than where it rounds to it.
TAU_MAX_S = 1050.0


def _segments(detector, gps0=GPS0, n_chunks=4, sub_sample_offset=0.0):
    """Overlapping chunks, optionally shifted off the reference sample grid."""
    step = CHUNK_S - OVERLAP_S
    nsamples = int(round(CHUNK_S * RATE))
    out = []
    for k in range(n_chunks):
        start = gps0 + k * step + sub_sample_offset
        out.append(
            Segment(
                segment_index=k, detector=detector, observing_run="O3a",
                gps_start=start, gps_end=start + CHUNK_S, sample_rate=RATE,
                nsamples=nsamples, sample_start_idx=k * nsamples,
                dyn_range_fac=1.0, noise_low_freq_cutoff=15.0,
            )
        )
    return out


def _network(detectors=("H1", "L1"), n_chunks=4, sub_sample_offset=0.0):
    segs = {detectors[0]: _segments(detectors[0], n_chunks=n_chunks)}
    for extra in detectors[1:]:
        segs[extra] = _segments(
            extra, n_chunks=n_chunks, sub_sample_offset=sub_sample_offset
        )
    return segs


def _plan(n_slides, detectors=("H1", "L1"), seed=0, **kwargs):
    """A plan on the standard fixture, with the physical floor left to assert itself."""
    return SlidePlan.build(
        GEOMETRY,
        _network(detectors=detectors),
        n_slides=n_slides,
        reference_detector=detectors[0],
        min_separation_s=kwargs.pop("min_separation_s", 0.0),
        tau_max_s=kwargs.pop("tau_max_s", TAU_MAX_S),
        seed=seed,
        **kwargs,
    )


def _brute_force_coincident(segments_by_detector, offsets_s, window_s):
    """
    Coincident hostable time under one slide, by counting coverage on a swept timeline.

    Deliberately not the module's interval algebra: every segment is turned into the span
    of times at which it could host a window start, pulled back through its detector's
    offset, and the timeline is cut at every endpoint. Each elementary piece is kept if
    every detector covers its midpoint. Sorting endpoints and testing each interval
    directly shares no code path with the merge-and-sweep intersection under test.

    Returns
    -------
    tuple
        ``(seconds, n_pieces)``.
    """
    spans = {}
    for detector, segments in segments_by_detector.items():
        lag = offsets_s.get(detector, 0.0)
        spans[detector] = [
            (s.gps_start - lag, s.gps_end - window_s - lag)
            for s in segments
            if s.gps_end - s.gps_start >= window_s
        ]
    edges = sorted({edge for held in spans.values() for span in held for edge in span})
    total = 0.0
    n_pieces = 0
    previous = False
    for low, high in zip(edges[:-1], edges[1:]):
        middle = 0.5 * (low + high)
        covered = all(
            any(start <= middle <= end for start, end in held)
            for held in spans.values()
        )
        if covered:
            total += high - low
            n_pieces += 0 if previous else 1
        previous = covered
    return total, n_pieces


class TestMinimumSeparation:
    """The floor is physical: the widest baseline in the network, not the reference one."""

    def test_floor_is_content_baseline_guard(self):
        """
        H1-L1 contributes 10.012846 ms on top of the analysis content and the guard.

        A lag below this leaves the slid window analysing part of the stretch its own
        zero-lag window analysed, so the same second of data appears on both sides of the
        coincidence.
        """
        got = minimum_separation_s(GEOMETRY, ("H1", "L1"), guard_s=4.0)
        assert got == pytest.approx(12.0 + H1L1 + 4.0, abs=1e-12)

    def test_virgo_raises_floor(self):
        """
        The HLV floor moves by exactly H1V1 - H1L1, so it used the maximum over pairs.

        The reference detector's own baseline is the one a two-detector implementation
        reaches for. Taking it here would leave the floor 17 ms too low, and a slide
        inside the physical coincidence window admits genuine coincidences as background.
        """
        two = minimum_separation_s(GEOMETRY, ("H1", "L1"), guard_s=4.0)
        three = minimum_separation_s(GEOMETRY, ("H1", "L1", "V1"), guard_s=4.0)
        assert three - two == pytest.approx(H1V1 - H1L1, abs=1e-12)
        assert three == pytest.approx(12.0 + H1V1 + 4.0, abs=1e-12)

    def test_floor_order_independent(self):
        """The floor is a property of the network, not of the reference choice."""
        floors = {
            minimum_separation_s(GEOMETRY, order, guard_s=4.0)
            for order in (("H1", "L1", "V1"), ("V1", "L1", "H1"), ("L1", "V1", "H1"))
        }
        assert len(floors) == 1

    def test_floor_exceeds_coincidence(self):
        """
        Any admissible lag is longer than light takes to cross the network.

        This is the whole point of the floor: below it, a real signal seen in both
        detectors survives the slide and is counted as a background coincidence.
        """
        for network in (("H1", "L1"), ("H1", "L1", "V1")):
            floor = minimum_separation_s(GEOMETRY, network, guard_s=0.0)
            assert floor > GEOMETRY.max_light_travel_s(network)

    def test_guard_adds_to_floor(self):
        """The guard is an additive margin, so it is auditable in the reported floor."""
        bare = minimum_separation_s(GEOMETRY, ("H1", "L1"), guard_s=0.0)
        assert minimum_separation_s(GEOMETRY, ("H1", "L1"), guard_s=7.5) == pytest.approx(
            bare + 7.5, abs=1e-12
        )

    def test_negative_guard_refused(self):
        """A negative guard would lower the floor below what the geometry allows."""
        with pytest.raises(ValueError):
            minimum_separation_s(GEOMETRY, ("H1", "L1"), guard_s=-1.0)


class TestPairwiseSeparations:
    """Every pair, not every detector against the reference."""

    def test_close_followers_rejected(self):
        """
        Lags of 30 s and 45 s clear the reference by 20 s but sit 15 s apart.

        This is the vector a two-detector implementation emits without noticing: both
        detectors are far from Hanford, and Livingston and Virgo are within a light-travel
        time of each other in slid time, so a real HLV coincidence survives into the
        background.
        """
        assert not pairwise_separations_ok(np.array([[30.0, 45.0]]), 20.0)[0]

    def test_separated_pairs_accepted(self):
        """The same vector spread so all three separations clear the floor."""
        assert pairwise_separations_ok(np.array([[30.0, 60.0]]), 20.0)[0]

    def test_two_detectors_one_pair(self):
        """With one lagged detector the constraint reduces to the lag itself."""
        lags = np.array([[19.9], [20.0], [500.0]])
        assert list(pairwise_separations_ok(lags, 20.0)) == [False, True, True]

    def test_single_vector_scalar_answer(self):
        """A bare lag vector is accepted as a single slide, per the documented shape."""
        got = pairwise_separations_ok(np.array([30.0, 45.0]), 20.0)
        assert got.shape == (1,)
        assert not got[0]

    def test_separation_uses_magnitude(self):
        """
        A lag of -30 s separates the detectors as well as +30 s does.

        Sign says which detector reads later, not how far apart they are, so a check
        written on the signed value would reject half the lattice for no reason.
        """
        assert pairwise_separations_ok(np.array([[-30.0]]), 20.0)[0]
        assert pairwise_separations_ok(np.array([[30.0, -30.0]]), 20.0)[0]

    def test_single_detector_no_pairs(self):
        """A one-detector network is unconstrained rather than an error."""
        assert pairwise_separations_ok(np.zeros((5, 0)), 20.0).tolist() == [True] * 5

    def test_floor_inclusive(self):
        """A lag exactly at the floor is admissible; the floor already carries a guard."""
        assert pairwise_separations_ok(np.array([[20.0, 40.0]]), 20.0)[0]

    def test_answer_per_slide(self):
        """One entry per row, so a whole ladder is screened in one call."""
        lags = np.array([[30.0, 45.0], [30.0, 60.0], [10.0, 60.0]])
        assert list(pairwise_separations_ok(lags, 20.0)) == [False, True, False]


class TestStratifiedLags:
    """The ladder samples the range; it does not pack against the minimum."""

    MIN_S = 20.0
    MAX_S = 8192.0

    def _lags(self, n_slides=500, n_lagged=1, seed=7, tau_max_s=None):
        return stratified_lags(
            n_slides,
            n_lagged,
            self.MIN_S,
            self.MAX_S if tau_max_s is None else tau_max_s,
            GEOMETRY.stride_samples,
            GEOMETRY.sample_rate,
            seed,
        )

    def test_ladder_reaches_tau_max(self):
        """
        The largest lag drawn is at least half the ceiling.

        A ladder packed at the minimum satisfies every weaker check -- lags distinct, lags
        above the floor -- while re-pairing one loud glitch against nearly the same stretch
        of the other detector on every slide.
        """
        assert np.abs(self._lags()).max() >= 0.5 * self.MAX_S

    def test_lags_uniform(self):
        """
        A KS test against the uniform distribution the strata are cut from.

        Stratification is what makes the background samples as independent as the lag
        scale allows; a ladder biased to one end concentrates them where detector noise is
        most correlated.
        """
        lags = self._lags(n_lagged=2)
        for column in range(lags.shape[1]):
            scaled = (lags[:, column] - self.MIN_S) / (self.MAX_S - self.MIN_S)
            assert stats.kstest(scaled, "uniform").pvalue > 0.01

    def test_no_oversized_gap(self):
        """
        One lag per stratum bounds the spacing, so no decade of lag is left unsampled.

        An unstratified draw of the same size leaves gaps several times the mean by
        chance, and a gap is a range of lag the background never probes.
        """
        sorted_lags = np.sort(self._lags()[:, 0])
        gaps = np.diff(sorted_lags)
        assert gaps.max() <= 5.0 * gaps.mean()

    def test_packed_ladder_fails_checks(self):
        """
        The negative control: 20, 40, 60, ... passes distinctness and fails the rest.

        Without this the three checks above could be vacuous, and the ladder they are
        meant to exclude is exactly the one a minimum-separation multiple produces.
        """
        packed = self.MIN_S * np.arange(1, 83, dtype=float)
        assert len(set(packed)) == packed.size
        assert packed.max() < 0.5 * self.MAX_S
        scaled = (packed - self.MIN_S) / (self.MAX_S - self.MIN_S)
        assert stats.kstest(scaled, "uniform").pvalue < 0.01

    def test_seed_reproduces_ladder(self):
        """A campaign's background must be reproducible from the recorded seed."""
        assert np.array_equal(self._lags(seed=3), self._lags(seed=3))

    def test_seed_changes_ladder(self):
        """
        The seed has to matter, or reproducibility is a property of nothing.

        A ladder fixed by the row count of an arbitrarily ordered sample list is
        reproducible in the same trivial sense while being uncontrolled.
        """
        assert not np.array_equal(self._lags(seed=3), self._lags(seed=4))

    def test_lags_on_stride_lattice(self):
        """
        A lag off the lattice would imply resampling a slid window.

        Window starts advance by an integer number of samples, so a lag that is not a
        whole number of strides has no window to land on.
        """
        multiples = self._lags(n_lagged=2) / GEOMETRY.stride_s
        assert np.allclose(multiples, np.round(multiples), atol=1e-9)

    def test_lags_in_range_nonzero(self):
        """Zero lag is the foreground, and the floor and ceiling are the contract."""
        lags = self._lags(n_lagged=2)
        assert np.all(np.abs(lags) >= self.MIN_S)
        assert np.all(np.abs(lags) <= self.MAX_S)
        assert np.all(lags != 0.0)

    def test_lags_distinct_per_detector(self):
        """Two slides at the same lag are one slide counted twice."""
        lags = self._lags(n_lagged=2)
        for column in range(lags.shape[1]):
            assert np.unique(lags[:, column]).size == lags.shape[0]

    def test_all_pairs_separated(self):
        """
        The three-detector draw is screened, not just the two-detector one.

        The pairwise constraint removes a band of the lattice; a sampler that ignores it
        emits slides in which two followers sit inside a light-travel time of each other.
        """
        lags = self._lags(n_slides=60, n_lagged=2, tau_max_s=120.0)
        assert pairwise_separations_ok(lags, self.MIN_S).all()

    def test_overfull_lattice_refused(self):
        """
        Asking for more distinct lags than exist fails rather than repeating one.

        Repeated lags would inflate the apparent background depth while adding no
        independent samples, which is the failure this whole module exists to avoid.
        """
        with pytest.raises(ValueError, match="distinct"):
            self._lags(n_slides=5000, tau_max_s=100.0)

    def test_low_ceiling_refused_hlv(self):
        """
        Two lagged detectors both need room above the floor, and 35 s does not hold it.

        A two-detector implementation accepts this configuration and then emits slides
        whose followers are 15 s apart.
        """
        with pytest.raises(ValueError, match="slid detectors"):
            self._lags(n_slides=10, n_lagged=2, tau_max_s=35.0)

    def test_nonpositive_floor_refused(self):
        """
        A floor of zero or less is refused rather than quietly admitting a zero lag.

        The stratum arithmetic clamps ``k_min`` to zero when the floor is not positive,
        so the lowest stratum can draw the zero multiple -- a "slide" that is the
        foreground, contributing real coincidences to the background. The floor is a
        physical quantity and is always positive, so a non-positive request is a
        configuration error and not a permissive setting.
        """
        for floor in (0.0, -1.0):
            with pytest.raises(ValueError, match="min_separation_s must be positive"):
                stratified_lags(8, 1, floor, TAU_MAX_S, GEOMETRY.stride_samples,
                                GEOMETRY.sample_rate, 0)

    def test_ceiling_below_floor_refused(self):
        """No lag is admissible, so an empty ladder would be silently returned instead."""
        with pytest.raises(ValueError, match="tau_max"):
            self._lags(n_slides=10, tau_max_s=self.MIN_S)

    @pytest.mark.parametrize("n_slides,n_lagged", [(0, 1), (5, 0), (0, 0)])
    def test_empty_request_shape(self, n_slides, n_lagged):
        """A zero-slide or single-detector request is legitimate and stays typed."""
        assert self._lags(n_slides=n_slides, n_lagged=n_lagged).shape == (
            n_slides,
            n_lagged,
        )


class TestPlanStructure:
    """What a plan asserts about itself before any livetime is read from it."""

    def test_zero_lag_present_once(self):
        """
        The foreground travels with the background it will be compared against.

        Two zero-lag slides would count the foreground twice in the background livetime;
        none would leave the comparison without its denominator.
        """
        plan = _plan(12)
        zero = [s for s in plan.slides if all(v == 0.0 for v in s.offsets_s.values())]
        assert len(zero) == 1
        assert zero[0].slide_id == 0

    def test_depth_plus_zero_lag(self):
        """``n_slides`` counts slid slides, so a plan of n has n+1 entries."""
        assert len(_plan(12).slides) == 13

    def test_iteration_in_id_order(self):
        """Shards are written per slide id, so iteration order is part of the contract."""
        ids = [slide.slide_id for slide in _plan(12)]
        assert ids == sorted(ids) == list(range(13))

    def test_id_order_is_imposed(self, tmp_path):
        """
        Iteration sorts, rather than happening to receive a sorted list.

        Built plans arrive in id order, so a version iterating storage order passes every
        other test here. The order matters because :meth:`save` writes rows in iteration
        order and consumers key shards on the row index; a plan whose list was rebuilt or
        reordered would then persist its livetimes against the wrong slide ids.
        """
        plan = _plan(9)
        plan.slides = list(reversed(plan.slides))
        assert [s.slide_id for s in plan] == list(range(10))

        path = tmp_path / "slides" / "slide_plan.h5"
        plan.save(path)
        reloaded = SlidePlan.load(path)
        assert [s.slide_id for s in reloaded.slides] == list(range(10))
        by_id = {s.slide_id: s.livetime_s for s in plan.slides}
        assert [s.livetime_s for s in reloaded.slides] == [
            by_id[i] for i in range(10)
        ]

    def test_slide_names_all_detectors(self):
        """
        A slide is self-describing, so no consumer has to know which detector was held.

        The reference appears with a zero offset rather than being absent, which is what
        lets a reader apply the offsets without knowing the network's convention.
        """
        plan = _plan(6, detectors=("H1", "L1", "V1"))
        for slide in plan:
            assert set(slide.offsets_s) == {"H1", "L1", "V1"}
            assert slide.offsets_s["H1"] == 0.0

    def test_livetime_is_windows_times_stride(self):
        """
        Livetime is counted in windows, so it is exact rather than accumulated.

        Analysed time is what the background actually scores; deriving it from interval
        lengths would credit the search with time no window covers.
        """
        for slide in _plan(20):
            assert slide.livetime_s == pytest.approx(
                slide.n_windows * GEOMETRY.stride_s, abs=1e-9
            )

    def test_plan_records_floor_used(self):
        """
        A request below the physical floor is raised to it, and the plan says so.

        Recording the requested value instead would leave the campaign unable to state
        the separation its background was built with.
        """
        plan = _plan(4, min_separation_s=0.5)
        assert plan.min_separation_s == pytest.approx(
            minimum_separation_s(GEOMETRY, ("H1", "L1"), guard_s=4.0), abs=1e-12
        )
        lags = np.array([s.offsets_s["L1"] for s in plan.slides[1:]])
        assert np.all(np.abs(lags) >= plan.min_separation_s)

    def test_wider_floor_honoured(self):
        """The floor is a minimum; a campaign may ask for more separation than physics."""
        plan = _plan(4, min_separation_s=200.0)
        assert plan.min_separation_s == 200.0
        lags = np.array([s.offsets_s["L1"] for s in plan.slides[1:]])
        assert np.all(np.abs(lags) >= 200.0)

    def test_hlv_plan_separates_pairs(self):
        """
        The all-pairs constraint survives into the emitted slides.

        Checked on the plan and not only on the sampler, because it is the plan that the
        background stage reads.
        """
        plan = _plan(40, detectors=("H1", "L1", "V1"), seed=2)
        floor = plan.min_separation_s
        for slide in plan.slides[1:]:
            lags = [slide.offsets_s[d] for d in ("L1", "V1")]
            assert pairwise_separations_ok(np.array([lags]), floor)[0], slide.slide_id

    def test_seed_reproduces_plan(self):
        """A background is only reproducible if its lags are."""
        first = [s.offsets_s["L1"] for s in _plan(10, seed=5)]
        second = [s.offsets_s["L1"] for s in _plan(10, seed=5)]
        assert first == second

    def test_seed_changes_plan(self):
        """The recorded seed is what distinguishes two backgrounds of the same depth."""
        first = [s.offsets_s["L1"] for s in _plan(10, seed=5)]
        assert first != [s.offsets_s["L1"] for s in _plan(10, seed=6)]

    def test_zero_depth_foreground_only(self):
        """A depth of zero is a legitimate request and must not fabricate a slide."""
        plan = _plan(0)
        assert len(plan.slides) == 1
        assert plan.slides[0].slide_id == 0

    def test_negative_depth_refused(self):
        """A depth is a count; a negative one is a configuration error, not an empty plan."""
        with pytest.raises(ValueError, match="n_slides"):
            _plan(-1)

    def test_foreign_reference_refused(self):
        """Slides measured against a detector the search never reads mean nothing."""
        with pytest.raises(ValueError, match="reference"):
            SlidePlan.build(GEOMETRY, _network(), n_slides=4, reference_detector="V1")

    def test_empty_network_refused(self):
        """With no detectors there is nothing to slide against, so a plan is meaningless."""
        with pytest.raises(ValueError, match="detector"):
            SlidePlan.build(GEOMETRY, {}, n_slides=4)

    def test_single_detector_no_background(self):
        """
        Requesting slides on a one-detector network is refused, not silently obliged.

        With nothing to slide, every lag vector is empty and every "slide" is another
        copy of zero lag. The plan would then report a background livetime of exactly
        ``n * T_zerolag`` -- the closed form this module exists to refute -- built
        entirely out of foreground, and every FAR taken from it would be meaningless
        while looking perfectly well formed.
        """
        with pytest.raises(ValueError, match="nothing to slide"):
            SlidePlan.build(
                GEOMETRY,
                {"H1": _segments("H1")},
                n_slides=4,
                reference_detector="H1",
                min_separation_s=0.0,
                tau_max_s=TAU_MAX_S,
            )

    def test_single_detector_foreground_ok(self):
        """Depth zero asks for no background, so one detector is a legitimate request."""
        plan = SlidePlan.build(
            GEOMETRY,
            {"H1": _segments("H1")},
            n_slides=0,
            reference_detector="H1",
            min_separation_s=0.0,
            tau_max_s=TAU_MAX_S,
        )
        assert len(plan.slides) == 1
        assert plan.background_livetime_s == 0.0
        assert plan.foreground_livetime_s > 0.0

    def test_slide_immutable_hashable(self):
        """
        A measured slide cannot be edited in place, offsets included.

        ``livetime_s`` was measured for one lag assignment; a slide whose ``offsets_s``
        could be reassigned afterwards would report a livetime belonging to a different
        slide, and nothing about the object would show it. ``frozen=True`` alone does not
        reach inside the mapping, so the mapping itself is a read-only view.
        """
        slide = _plan(3).slides[1]
        with pytest.raises(dataclasses.FrozenInstanceError):
            slide.livetime_s = 0.0
        with pytest.raises(TypeError):
            slide.offsets_s["L1"] = 0.0
        assert isinstance(hash(slide), int)
        assert len({slide, dataclasses.replace(slide)}) == 1

    def test_floor_wider_than_data_refused(self):
        """
        Asking for a floor longer than the run refuses rather than returning empty slides.

        Every slide would retain nothing, and a plan of empty slides still reports a
        slide count; the failure has to happen where the request is made, not where the
        FAR is divided by a livetime of zero.
        """
        with pytest.raises(ValueError, match="span"):
            _plan(4, min_separation_s=1e6, tau_max_s=2e6)

    def test_plan_ceiling_below_floor_refused(self):
        """A ceiling under the physical floor admits no lag at all."""
        with pytest.raises(ValueError, match="tau_max_s"):
            _plan(4, min_separation_s=0.0, tau_max_s=1.0)


class TestMeasuredLivetime:
    """The invariant the reference implementation could not state: how much time survived."""

    @pytest.mark.parametrize("n_slides", [1, 10, 100, 1000])
    def test_total_matches_brute_force(self, n_slides):
        """
        The measured background livetime against a brute-force coverage sweep.

        The oracle is written from the segments, not from the module: each detector's
        hostable spans are pulled back through its offset and the timeline is swept for
        pieces every detector covers. The window count can only fall short of that by the
        stride left over at each contiguous piece and at each segment where the stride
        phase restarts, which is the bound asserted per slide.
        """
        network = _network()
        plan = SlidePlan.build(
            GEOMETRY, network, n_slides=n_slides, min_separation_s=0.0,
            tau_max_s=TAU_MAX_S, seed=11,
        )
        overhead = len(network["H1"]) * GEOMETRY.stride_s
        expected_background = 0.0
        expected_foreground = 0.0
        for slide in plan:
            expected, n_pieces = _brute_force_coincident(
                network, slide.offsets_s, GEOMETRY.window_s
            )
            if slide.slide_id == 0:
                expected_foreground += expected
            else:
                expected_background += expected
            assert slide.livetime_s <= expected + 1e-9
            assert expected - slide.livetime_s <= n_pieces * GEOMETRY.stride_s + overhead
        # The zero-lag slide is the foreground and is accounted separately: the two
        # together are every slide the sweep covered, and neither counts the other's
        # seconds.
        assert plan.background_livetime_s == pytest.approx(expected_background, rel=5e-3)
        assert plan.foreground_livetime_s == pytest.approx(expected_foreground, rel=5e-3)

    @pytest.mark.parametrize("n_slides", [1, 10, 100, 1000])
    def test_closed_form_overstates(self, n_slides):
        """
        ``n_slides * T_zerolag`` is above the measured total at every depth.

        The reference implementation had no way to measure retention and so had to assume
        this value. It is wrong in the direction that matters: an overstated background
        livetime divides the false-alarm count by too much and reports a rate too low.

        Asserted against ``n_slides``, not ``len(plan.slides)``: the extra factor the
        stored zero-lag slide contributes is arithmetic, and a bound that leans on it
        would pass for an implementation whose slid slides retained everything.
        """
        plan = SlidePlan.build(
            GEOMETRY, _network(), n_slides=n_slides, min_separation_s=0.0,
            tau_max_s=TAU_MAX_S, seed=11,
        )
        zero_lag_s = plan.foreground_livetime_s
        assert plan.background_livetime_s < n_slides * zero_lag_s

    def test_retention_falls_with_lag(self):
        """
        Slid retention is a function of the lag drawn, measured against the lag itself.

        The property that forbids a closed form is that a slide keeps less the further it
        is slid. Testing it as a fall in mean retention with ladder *depth* looks like the
        same thing and is not: with the zero-lag slide in the numerator, ``(1 + n r) /
        (n + 1)`` falls with ``n`` for any constant ``r`` whatsoever, so that version
        passes an implementation whose slides ignore their lag entirely.

        Here the slid slides of one deep ladder are binned by ``|lag|`` and the binned
        mean retention must fall across the bins. Averaged over seeds because a single
        bin at the top of the range is a small sample of a wide interval.
        """
        network = _network()
        zero_lag_s = SlidePlan.build(
            GEOMETRY, network, n_slides=0, min_separation_s=0.0, tau_max_s=TAU_MAX_S
        ).slides[0].livetime_s

        lags, retentions = [], []
        for seed in range(8):
            plan = SlidePlan.build(
                GEOMETRY, network, n_slides=200, min_separation_s=0.0,
                tau_max_s=TAU_MAX_S, seed=seed,
            )
            for slide in plan.slides[1:]:
                lags.append(abs(slide.offsets_s["L1"]))
                retentions.append(slide.livetime_s / zero_lag_s)
        lags, retentions = np.asarray(lags), np.asarray(retentions)

        edges = np.quantile(lags, np.linspace(0.0, 1.0, 6))
        binned = [
            float(np.mean(retentions[(lags >= lo) & (lags <= hi)]))
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        assert all(a > b for a, b in zip(binned, binned[1:])), binned
        # And the fall is substantial, not a rounding-scale drift: the longest lags keep
        # markedly less than the shortest, which is what makes the measurement necessary.
        assert binned[-1] < 0.75 * binned[0]

    def test_background_excludes_zero_lag(self):
        """
        Foreground seconds are not also counted as background seconds.

        ``far.py`` forms ``(1 + n_b) / T_b`` with ``n_b`` counted over slid triggers
        alone, so a ``T_b`` that included the zero-lag slide would divide a slid count by
        an exposure partly made of foreground -- biasing every rate low, in the direction
        that makes a candidate look more significant. The bias is ``1 / (n + 1)``, which
        is small at depth and a factor of two at one slide, so a test at a single large
        depth would barely see it.
        """
        for n_slides in (1, 4, 82):
            plan = _plan(n_slides, seed=3)
            slid = [s for s in plan.slides if s.slide_id != 0]
            zero = [s for s in plan.slides if s.slide_id == 0]
            assert len(zero) == 1 and len(slid) == n_slides

            assert plan.background_livetime_s == pytest.approx(
                sum(s.livetime_s for s in slid), abs=1e-9
            )
            assert plan.foreground_livetime_s == pytest.approx(
                zero[0].livetime_s, abs=1e-9
            )
            # The two partition the plan: nothing is counted twice, nothing is dropped.
            assert plan.background_livetime_s + plan.foreground_livetime_s == (
                pytest.approx(sum(s.livetime_s for s in plan.slides), abs=1e-9)
            )
            assert plan.background_livetime_s < sum(s.livetime_s for s in plan.slides)

    def test_total_sums_stored_slides(self):
        """
        Editing one slide's livetime moves the total by exactly that much.

        This is what "measured, not derived" means operationally: there is no closed form
        behind the property, so the total cannot drift away from the slides it came from.
        """
        plan = _plan(8)
        before = plan.background_livetime_s
        removed = plan.slides[3].livetime_s
        plan.slides[3] = dataclasses.replace(plan.slides[3], livetime_s=0.0)
        assert plan.background_livetime_s == pytest.approx(before - removed, abs=1e-9)

    def test_longer_lag_retains_less(self):
        """
        Per-slide livetime falls with lag, which is why the ladder's depth costs livetime.

        Measured on the plan rather than argued: the slides are sorted by lag and their
        livetimes must not increase, since a longer lag can only move more of one
        detector's data past the end of the other's.
        """
        plan = _plan(40, seed=4)
        slid = sorted(plan.slides[1:], key=lambda s: abs(s.offsets_s["L1"]))
        livetimes = np.array([s.livetime_s for s in slid])
        assert np.all(np.diff(livetimes) <= 1e-9)
        assert livetimes[-1] < livetimes[0]


class TestPersistence:
    """A reloaded plan is the plan that ran, down to its measured livetime."""

    def test_round_trip_exact(self, tmp_path):
        """
        Every field survives the write, including the offsets that define each slide.

        The plan is the provenance of the background: a campaign quotes a livetime and a
        separation from it long after the run, so a lossy round trip is a silently wrong
        FAR rather than a missing file.
        """
        plan = _plan(15, detectors=("H1", "L1", "V1"), seed=9)
        path = tmp_path / "slides" / "slide_plan.h5"
        plan.save(path)
        loaded = SlidePlan.load(path)

        assert loaded.reference_detector == plan.reference_detector
        assert loaded.seed == plan.seed
        assert loaded.min_separation_s == plan.min_separation_s
        assert loaded.tau_max_s == plan.tau_max_s
        assert loaded.background_livetime_s == plan.background_livetime_s
        for original, restored in zip(plan, loaded):
            assert restored.slide_id == original.slide_id
            assert restored.n_windows == original.n_windows
            assert restored.livetime_s == original.livetime_s
            assert restored.offsets_s == original.offsets_s

    def test_save_creates_directory(self, tmp_path):
        """The plan is written before the campaign root has any other product in it."""
        path = tmp_path / "campaign" / "slides" / "slide_plan.h5"
        _plan(3).save(path)
        assert path.exists()

    def test_livetime_read_not_recomputed(self, tmp_path):
        """
        A livetime altered before the write comes back altered.

        If loading re-measured the slides, a plan reloaded against a different segment
        list would quietly report a livetime the campaign never ran with, and the FAR
        would change without the background changing.
        """
        plan = _plan(6)
        plan.slides[2] = dataclasses.replace(plan.slides[2], livetime_s=1234.5)
        path = tmp_path / "slides" / "slide_plan.h5"
        plan.save(path)
        loaded = SlidePlan.load(path)
        assert loaded.slides[2].livetime_s == 1234.5
        assert loaded.background_livetime_s == pytest.approx(
            plan.background_livetime_s, abs=1e-9
        )

    def test_total_mismatch_refused(self, tmp_path):
        """
        An interrupted write leaves a slide list that does not sum to the stored total.

        Reading it would supply a background livetime no slide supports, so the file is
        refused instead of being trusted or silently re-summed.
        """
        import h5py

        path = tmp_path / "slides" / "slide_plan.h5"
        _plan(5).save(path)
        with h5py.File(path, "a") as handle:
            handle["livetime_s"][2] = 0.0
        with pytest.raises(ValueError, match="incomplete"):
            SlidePlan.load(path)

    def test_keep_threshold_round_trips(self, tmp_path):
        """
        The threshold the campaign ran with is stored beside the slides it applies to.

        Every slide job writes only triggers above it, so the number is what makes the
        stored counts comparable to each other and to the zero-lag pass. Re-deriving it
        in a later job would tie it to whatever that job happened to see; carrying it in
        the plan is what freezes it.
        """
        plan = _plan(6)
        plan.keep_threshold = 8.125
        path = tmp_path / "slides" / "slide_plan.h5"
        plan.save(path)
        assert SlidePlan.load(path).keep_threshold == 8.125

    def test_absent_threshold_is_none(self, tmp_path):
        """
        A plan drawn before the zero-lag pass has no threshold, and says so.

        The ladder is built before the histogram that sets the threshold exists, so
        "not yet frozen" is a state the file has to be able to express. Reading it back
        as a number -- zero especially -- would silently keep every trigger.
        """
        path = tmp_path / "slides" / "slide_plan.h5"
        _plan(4).save(path)
        assert SlidePlan.load(path).keep_threshold is None

    def test_stored_depth_counts_slid(self, tmp_path):
        """
        ``n_slides`` in the file means what it means in :meth:`build`.

        The stored table has one more row than the ladder has slides, because the
        zero-lag slide is in it. Writing that row count under the name ``n_slides``
        makes the file disagree with the argument that produced it, and a campaign
        quoting "82 slides" from the file would be quoting 83.
        """
        import h5py

        path = tmp_path / "slides" / "slide_plan.h5"
        _plan(7).save(path)
        with h5py.File(path, "r") as handle:
            assert int(handle.attrs["n_slides"]) == 7
            assert int(handle.attrs["n_records"]) == 8
            assert handle["slide_id"].shape[0] == 8

    def test_truncated_file_refused(self, tmp_path):
        """
        A missing dataset is refused as a truncated write, not raised as a KeyError.

        The failure a campaign actually hits is a job killed mid-write, and what it needs
        back is the path and what is missing from it. A bare ``KeyError('n_windows')``
        from inside the reader names neither.
        """
        import h5py

        path = tmp_path / "slides" / "slide_plan.h5"
        _plan(5).save(path)
        with h5py.File(path, "a") as handle:
            del handle["n_windows"]
        with pytest.raises(ValueError, match="truncated"):
            SlidePlan.load(path)

    def test_missing_attribute_refused(self, tmp_path):
        """A plan without its separation cannot say what background it describes."""
        import h5py

        path = tmp_path / "slides" / "slide_plan.h5"
        _plan(5).save(path)
        with h5py.File(path, "a") as handle:
            del handle.attrs["min_separation_s"]
        with pytest.raises(ValueError, match="min_separation_s"):
            SlidePlan.load(path)

    def test_ragged_file_refused(self, tmp_path):
        """
        Datasets of unequal length are a partial write, and are caught before the sum is.

        Zipping them would silently drop slides off the end, and the total would then
        disagree for a reason the error message would attribute to the wrong thing.
        """
        import h5py

        path = tmp_path / "slides" / "slide_plan.h5"
        _plan(5).save(path)
        with h5py.File(path, "a") as handle:
            data = handle["livetime_s"][:-1]
            del handle["livetime_s"]
            handle.create_dataset("livetime_s", data=data)
        with pytest.raises(ValueError, match="truncated"):
            SlidePlan.load(path)


class TestVetoedLivetime:
    """Re-measuring the ladder with stretches of detector time removed."""

    def test_veto_reduces_every_slide(self):
        """
        Removing detector time removes coincident time from the whole ladder.

        A veto in the reference detector is seen by every slide, because every slide reads
        the same reference data; only the follower's pairing differs. A re-measurement
        that left any slide untouched would be measuring something other than the lattice
        the background is scored on.
        """
        plan = _plan(6)
        vetoed = remeasure_livetimes(
            plan, GEOMETRY, _network(), {"H1": [(GPS0 + 100.0, GPS0 + 400.0)]}
        )

        before = {s.slide_id: s.livetime_s for s in plan}
        after = {s.slide_id: s.livetime_s for s in vetoed}
        assert set(before) == set(after)
        assert all(after[k] < before[k] for k in before)

    def test_no_veto_is_identity(self):
        """
        An empty veto reproduces the original ladder exactly, slide for slide.

        The re-measurement must run the same lattice as the original build, so with
        nothing removed it has to return the same numbers -- not merely close ones. Any
        difference here would be a second, disagreeing definition of livetime.
        """
        plan = _plan(6)
        same = remeasure_livetimes(plan, GEOMETRY, _network(), {})

        for original, rebuilt in zip(plan, same):
            assert rebuilt.slide_id == original.slide_id
            assert rebuilt.n_windows == original.n_windows
            assert rebuilt.livetime_s == original.livetime_s

    def test_window_overlap_is_vetoed(self):
        """
        A window overlapping a vetoed stretch is removed, not just one starting inside it.

        A window carries ``window_s`` of data, so a start a full window before the veto
        still reads vetoed samples. The loss must therefore exceed the vetoed duration by
        about one window per veto edge; a naive implementation that subtracted only the
        interval itself would lose exactly the vetoed time and leave vetoed samples in the
        background.
        """
        plan = _plan(0)
        veto_s = 100.0
        vetoed = remeasure_livetimes(
            plan,
            GEOMETRY,
            _network(),
            {"H1": [(GPS0 + 200.0, GPS0 + 200.0 + veto_s)]},
        )
        lost = plan.foreground_livetime_s - vetoed.foreground_livetime_s

        assert lost > veto_s
        assert lost == pytest.approx(veto_s + GEOMETRY.window_s, abs=2.0 * GEOMETRY.stride_s)

    def test_overlapping_vetoes_cost_their_union(self):
        """
        Two overlapping vetoes cost what one merged veto costs, not the sum of the two.

        A cluster of nearby removals produces overlapping windows, and summing their
        durations would over-charge the livetime -- reporting a background shorter than it
        is, which raises every rate. PyCBC coalesces before summing for the same reason.
        """
        plan = _plan(0)
        overlapping = remeasure_livetimes(
            plan,
            GEOMETRY,
            _network(),
            {"H1": [(GPS0 + 200.0, GPS0 + 300.0), (GPS0 + 250.0, GPS0 + 350.0)]},
        )
        merged = remeasure_livetimes(
            plan, GEOMETRY, _network(), {"H1": [(GPS0 + 200.0, GPS0 + 350.0)]}
        )

        assert overlapping.foreground_livetime_s == merged.foreground_livetime_s

    def test_unknown_detector_refused(self):
        """A veto on a detector outside the network would silently remove nothing."""
        plan = _plan(4)
        with pytest.raises(ValueError, match="not in the network"):
            remeasure_livetimes(
                plan, GEOMETRY, _network(), {"V1": [(GPS0, GPS0 + 10.0)]}
            )

    def test_plan_identity_preserved(self):
        """
        The lags, seed and frozen threshold survive the re-measurement.

        Only the livetimes change. A re-measurement that redrew the ladder would give the
        exclusive background a different set of slides from the inclusive one, and the two
        would no longer be comparable.
        """
        plan = _plan(5, seed=3)
        plan.keep_threshold = 7.5
        vetoed = remeasure_livetimes(
            plan, GEOMETRY, _network(), {"L1": [(GPS0 + 50.0, GPS0 + 60.0)]}
        )

        assert vetoed.seed == plan.seed
        assert vetoed.keep_threshold == 7.5
        assert vetoed.reference_detector == plan.reference_detector
        assert [s.offsets_s for s in vetoed] == [s.offsets_s for s in plan]


class TestRolledPairing:
    """sgwc-1's background: shift along the lattice rather than in GPS."""

    def test_sgwc1_shape_at_two_slides(self):
        """
        sgwc-1 rolls by ``N//2`` and ``N//3`` and so takes exactly two slides. The
        generalisation places ``K`` shifts evenly over the admissible range, which at
        ``K = 2`` and no floor is thirds of the lattice -- the same construction, and
        every shift a large fraction of the run.
        """
        assert list(slides.rolled_shifts(2, 1, 1_000_000, min_shift=0).ravel()) == [
            333_333,
            666_667,
        ]

        # A floor narrows the range it spreads over; it does not change the shape.
        with_floor = slides.rolled_shifts(2, 1, 1_000_000, min_shift=1000).ravel()
        assert list(with_floor) == [333_667, 666_333]

    def test_shifts_are_distinct_per_follower(self):
        """
        Two followers rolled by the same amount stay at zero lag with *each other*, so a
        three-detector background would carry a real coincidence in two of its three
        detectors.
        """
        shifts = slides.rolled_shifts(4, 2, 1_000_000, min_shift=1000)

        assert shifts.shape == (4, 2)
        assert (shifts[:, 0] != shifts[:, 1]).all()

    def test_floor_refused_when_unreachable(self):
        """A lattice too short for the floor is a stated failure, not a clipped shift."""
        with pytest.raises(ValueError, match="admits no shift"):
            slides.rolled_shifts(2, 1, 100, min_shift=60)

    def test_more_slides_than_shifts_refused(self):
        """Distinct shifts are the resource; asking for more than exist is an error."""
        with pytest.raises(ValueError, match="admissible"):
            slides.rolled_shifts(500, 1, 1000, min_shift=400)

    def test_separation_is_the_shift_when_contiguous(self):
        """On an unbroken stretch a shift of k windows is exactly k strides."""
        contiguous = np.arange(1000, dtype=np.float64) * 0.1

        assert slides.shifted_separations_s(contiguous, 10, 0.1) == pytest.approx(1.0)

    def test_gaps_only_widen_the_separation(self):
        """
        The bound the floor rests on. A sorted lattice skips every gap between segments,
        and skipping one can only push the paired times further apart -- so ``k * stride``
        is a lower bound rather than an approximation, and the floor cannot be undercut by
        the data's own structure.
        """
        contiguous = np.arange(1000, dtype=np.float64) * 0.1
        gapped = np.concatenate([np.arange(500) * 0.1, np.arange(500) * 0.1 + 500.0])

        assert slides.shifted_separations_s(gapped, 10, 0.1) >= (
            slides.shifted_separations_s(contiguous, 10, 0.1) - 1e-9
        )

    def test_empty_lattice_refused(self):
        """Nothing to measure is a stated failure, not a separation of zero."""
        with pytest.raises(ValueError, match="no separations"):
            slides.shifted_separations_s(np.zeros(0), 5, 0.1)

    def test_no_livetime_lost(self):
        """
        The whole advantage over a lag ladder. Every lattice ordinal is hostable in every
        detector, so re-pairing ordinals cannot push anything into a gap; a ladder at the
        same depth loses whatever its lags shift out of the data.
        """
        rolled = slides.SlidePlan.build(
            GEOMETRY, _network(), n_slides=4, reference_detector="H1",
            min_separation_s=20.0, seed=3, method="roll",
        )
        background = [s for s in rolled if s.slide_id != 0]

        assert background
        for slide in background:
            assert slide.livetime_s == pytest.approx(rolled.foreground_livetime_s)
            assert slide.n_windows == rolled.slides[0].n_windows

    def test_ladder_loses_livetime(self):
        """The contrast that makes the comparison meaningful, not an assumption about it."""
        ladder = slides.SlidePlan.build(
            GEOMETRY, _network(), n_slides=4, reference_detector="H1",
            min_separation_s=20.0, tau_max_s=1024.0, seed=3, method="ladder",
        )
        background = [s for s in ladder if s.slide_id != 0]

        assert any(s.livetime_s < ladder.foreground_livetime_s for s in background)

    def test_reference_carried_at_zero(self):
        """
        As ``offsets_s`` carries it, so a slide is self-describing and a round trip
        through the file changes nothing.
        """
        plan = slides.SlidePlan.build(
            GEOMETRY, _network(), n_slides=2, reference_detector="H1",
            min_separation_s=20.0, seed=3, method="roll",
        )
        slide = [s for s in plan if s.slide_id != 0][0]

        assert slide.window_shift["H1"] == 0
        assert slide.window_shift["L1"] > 0

    def test_round_trip(self, tmp_path):
        """A reloaded plan must describe the same pairing, not an equivalent one."""
        plan = slides.SlidePlan.build(
            GEOMETRY, _network(), n_slides=3, reference_detector="H1",
            min_separation_s=20.0, seed=3, method="roll",
        )
        target = tmp_path / "plan.h5"
        plan.save(target)
        reloaded = slides.SlidePlan.load(target)

        assert reloaded.method == "roll"
        assert list(reloaded.slides) == list(plan.slides)

    def test_unknown_method_refused(self):
        """The background pairing is not something to fall back to a default on."""
        with pytest.raises(ValueError, match="unknown slide method"):
            slides.SlidePlan.build(
                GEOMETRY, _network(), n_slides=1,
                reference_detector="H1", method="shuffle",
            )

    def test_cache_bound_from_a_memory_budget(self):
        """
        The halo a frontend cache must hold is twice the largest shift, so the affordable
        shift is half of what the budget holds. Measured on the production network at
        20.5 KB per window per detector: 60 GB of an 80 GB card buys 0.83 days, against
        the ladder's 2.3 hours.
        """
        windows, seconds = slides.cache_bounded_shift(
            60e9, n_detectors=2, bytes_per_window_per_detector=int(20.5 * 1024),
            stride_s=0.100098,
        )

        assert windows == pytest.approx(714_557, rel=1e-3)
        assert seconds / 86400.0 == pytest.approx(0.83, abs=0.01)

    def test_bounded_shifts_stay_inside_the_halo(self):
        """
        Every shift has to fit, not just the typical one: a single slide reaching past
        the halo makes the cache miss for that slide alone, and the miss is a silent
        recomputation rather than an error.
        """
        bound = 500_000
        shifts = slides.rolled_shifts(
            12, 1, 12_532_817, min_shift=1000, max_shift=bound
        ).ravel()

        assert shifts.max() <= 1000 + bound
        assert shifts.min() >= 1000

    def test_unbounded_shifts_span_the_run(self):
        """
        Which is what makes an unbounded roll decorrelate so well, and exactly why no
        cache can serve it -- the working set becomes the whole run's features, 3.9 TB on
        the O3a lattice.
        """
        n_windows = 12_532_817
        shifts = slides.rolled_shifts(8, 1, n_windows, min_shift=1000).ravel()

        assert shifts.max() > n_windows // 2

    def test_bound_still_beats_the_ladder(self):
        """
        The bound is worth having only if it leaves the roll ahead, and it is the typical
        separation that decides that rather than the smallest.

        Measured on this campaign at eight slides: separations run 2.2 to 17.7 hours,
        median 10.0, against a ladder whose *largest* lag is 2.3 hours. The smallest
        bounded shift is comparable to that largest lag -- 8,047 s against 8,192 -- so a
        claim about the minimum would be false; it is the typical separation that decides
        whether a background is correlated, and there the roll is an order of magnitude
        ahead.
        """
        separations = (
            slides.rolled_shifts(
                8, 1, 12_532_817, min_shift=1000, max_shift=714_557
            ).ravel()
            * 0.100098
        )
        ladder_largest_lag_s = 8192.0

        assert np.median(separations) > 4 * ladder_largest_lag_s
        assert separations.max() > 7 * ladder_largest_lag_s


class TestBackgroundDepth:
    """A campaign states how much background it wants, not how many slides."""

    def test_depth_scales_with_the_foreground(self):
        """
        The reason the depth is the knob. O3a HL analyses 106.75 days and O3a HLV 80.81,
        so a fixed slide count gives the two different backgrounds and their false-alarm
        rates stop being comparable at the same ladder length.
        """
        hl = slides.slides_for_background(10.0, 106.75 * 86400)
        hlv = slides.slides_for_background(10.0, 80.81 * 86400)

        assert hl == 35
        assert hlv == 46

    def test_rounded_up(self):
        """
        A campaign that asked for ten years and got nine and a half has a false-alarm
        floor it did not ask for.
        """
        year = 365.25 * 86400.0

        assert slides.slides_for_background(10.0, year) == 10
        assert slides.slides_for_background(10.0, year * 1.01) == 10
        assert slides.slides_for_background(10.0, year * 0.99) == 11

    def test_retention_costs_slides(self):
        """
        A ladder keeps less of each slide than a roll, so it needs proportionally more of
        them for the same years. Ignoring that undershoots the depth silently.
        """
        full = slides.slides_for_background(10.0, 30 * 86400, retention=1.0)
        lossy = slides.slides_for_background(10.0, 30 * 86400, retention=0.5)

        assert lossy >= 2 * full - 1

    def test_never_zero(self):
        """
        A target smaller than one slide still needs one: no background at all makes every
        candidate's rate the same number.
        """
        assert slides.slides_for_background(0.01, 365.25 * 86400) == 1

    def test_refusals(self):
        """Each of these produces a plausible plan rather than an error if allowed."""
        with pytest.raises(ValueError, match="target_yr must be positive"):
            slides.slides_for_background(0.0, 86400.0)
        with pytest.raises(ValueError, match="no foreground livetime"):
            slides.slides_for_background(1.0, 0.0)
        with pytest.raises(ValueError, match="retention"):
            slides.slides_for_background(1.0, 86400.0, retention=1.5)
