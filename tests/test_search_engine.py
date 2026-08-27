#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_engine.py
Description   : The scoring stage, executed rather than described.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later

Every other search test exercises index arithmetic, accounting or statistics on arrays
someone put there. This one runs the forward path: a checkpoint is loaded into a
registered architecture, the training run's parameter prior builds the multirate binning
and the point-estimate decode, fiducial spectra whiten, and windows read off a memmap come
back as triggers in a shard.

That is worth its own file because assembly is where this stage's failures live. The
pieces are all borrowed from the training path and each one works there; what can be wrong
is the order they are put together in, and the arguments passed across the joins -- a
sampler built without its encoding buffers compiled, a config registered after the module
that reads it, a device taken from the checkpoint instead of the campaign. None of those
raise until the whole thing is run.

Synthetic throughout: a toy network, a synthetic release, synthetic fiducial spectra. The
production prior is real because it is configuration rather than data, and because the
binning it fixes is the one the search actually uses.
"""

import dataclasses
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("h5py")

from tests.test_search_drivers import _campaign  # noqa: E402

PRODUCTION_PRIOR = "runs/o3b/gwconfig.yaml"


@pytest.fixture(scope="module")
def toy_architecture():
    """Register the toy network under a name a spec can select, once per session."""
    from sage.search.checkpoint import ARCHITECTURES, register_architecture
    from tests.search_fixtures import ToyFrontendNet

    if "toy" not in ARCHITECTURES:
        register_architecture(
            "toy", lambda cfg, data_cfg: ToyFrontendNet(len(cfg.detectors), cfg.norm_type)
        )
    return "toy"


def _scored_campaign(tmp_path, toy_architecture, **engine_overrides):
    """A campaign with everything the forward path needs, on the CPU."""
    from pathlib import Path

    from tests.search_fixtures import (
        make_synthetic_checkpoint,
        make_synthetic_fiducial,
    )

    prior = Path(PRODUCTION_PRIOR).resolve()
    if not prior.is_file():
        pytest.skip(f"no parameter prior at {prior}")

    spec = _campaign(tmp_path)
    make_synthetic_checkpoint(spec.engine.checkpoint, detectors=("H1", "L1"))
    fiducial = make_synthetic_fiducial(tmp_path / "fiducial")
    return dataclasses.replace(
        spec,
        data=dataclasses.replace(spec.data, fiducial_dir=fiducial),
        engine=dataclasses.replace(
            spec.engine,
            device="cpu",
            architecture=toy_architecture,
            gwconfig=prior,
            **engine_overrides,
        ),
    )


class TestForwardPath:
    """The engine runs, and what it wrote describes what it scored."""

    def test_zerolag_scores_the_lattice(self, tmp_path, toy_architecture):
        """
        Every window the grid holds is scored exactly once. The count is the assertion
        that matters: a reader that stopped at a segment boundary, or one that ran off
        the end of a chunk, shows up here and nowhere else.
        """
        import sage.search.engine as engine
        import sage.search.grid as grid
        import sage.search.segments as segments

        spec = _scored_campaign(tmp_path, toy_architecture)
        segments.run(spec)
        lattice = grid.run(spec)
        report = engine.run_search(spec, stage="zerolag", slide_id=0)
        assert report["n_windows"] == lattice["n_windows"]
        assert report["n_lattice_windows"] == lattice["n_windows"]
        assert report["blocks_completed"] == lattice["n_blocks"]

    def test_shard_is_finalised(self, tmp_path, toy_architecture):
        """
        A finished shard says so. The collation tells "measured, empty" from "never ran"
        by this flag and the committed block count, and a shard that never sets them is
        counted as covered while its slide contributed nothing.
        """
        import h5py

        import sage.search.engine as engine
        import sage.search.grid as grid
        import sage.search.segments as segments

        spec = _scored_campaign(tmp_path, toy_architecture)
        segments.run(spec)
        lattice = grid.run(spec)
        report = engine.run_search(spec, stage="zerolag", slide_id=0)
        with h5py.File(report["shard"], "r") as handle:
            assert bool(handle.attrs["finalised"])
            assert int(handle.attrs["n_blocks"]) == lattice["n_blocks"]
            assert int(handle.attrs["slide_id"]) == 0

    def test_campaign_device_wins(self, tmp_path, toy_architecture):
        """
        The checkpoint records the device the *training* job used -- 'cuda:0' in every
        real one. Registered unaltered it reaches the sampler's generator, and a CPU
        campaign dies inside CUDA initialisation with an error about a driver version.
        """
        from sage.core.config import get_cfg

        import sage.search.engine as engine
        import sage.search.grid as grid
        import sage.search.segments as segments

        spec = _scored_campaign(tmp_path, toy_architecture)
        segments.run(spec)
        grid.run(spec)
        engine.run_search(spec, stage="zerolag", slide_id=0)
        assert get_cfg().device == "cpu"


class TestEmptySlide:
    """A lag that analyses no time is a measurement, not a failure."""

    def test_empty_lattice_still_finalises(self, tmp_path, toy_architecture):
        """
        A slide whose lag carries every window off the end of the run contributes zero
        events over zero livetime. Both are already true; what must not happen is the
        engine raising, because the collation would then see a slide the plan declares
        and no shard for it, and refuse to collate a ladder that is in fact complete.
        """
        import h5py

        import sage.search.engine as engine
        import sage.search.grid as grid
        import sage.search.segments as segments

        spec = _scored_campaign(tmp_path, toy_architecture)
        segments.run(spec)
        grid.run(spec)
        # Longer than the release, so no window survives the shift.
        report = engine.run_search(
            spec, stage="background", slide_id=1, offsets_s={"H1": 0.0, "L1": 1.0e6}
        )
        assert report["n_windows"] == 0
        with h5py.File(report["shard"], "r") as handle:
            assert bool(handle.attrs["finalised"])

    def test_stream_refused_for_a_slide(self, tmp_path, toy_architecture):
        """
        The per-window stream is one value per analysed window, so a ladder would write
        one copy of the whole run per rung. Requested campaign-wide it applies to the
        zero-lag pass and is dropped for the slides, rather than making the background
        stage unrunnable.
        """
        import h5py

        import sage.search.engine as engine
        import sage.search.grid as grid
        import sage.search.segments as segments

        spec = _scored_campaign(tmp_path, toy_architecture, keep_stream=True)
        segments.run(spec)
        grid.run(spec)
        zerolag = engine.run_search(spec, stage="zerolag", slide_id=0)
        slid = engine.run_search(
            spec, stage="background", slide_id=1, offsets_s={"H1": 0.0, "L1": 40.0}
        )
        with h5py.File(zerolag["shard"], "r") as handle:
            assert "stream" in handle
        with h5py.File(slid["shard"], "r") as handle:
            assert "stream" not in handle


class TestChain:
    """segments -> grid -> zerolag -> slides -> background -> far, executed."""

    def test_significance_chain_runs(self, tmp_path, toy_architecture):
        """
        The stages after the engine consume what it wrote. Run together they check the
        joins the individual driver tests cannot: that the shard the engine finalises is
        the one the collation finds, that the threshold background freezes into the plan
        is the one the slides were scored against, and that far reads a background whose
        livetime came from the plan.
        """
        import sage.search.background as background
        import sage.search.engine as engine
        import sage.search.far as far
        import sage.search.grid as grid
        import sage.search.segments as segments
        import sage.search.slides as slides
        from sage.search.slides import SlidePlan

        spec = _scored_campaign(tmp_path, toy_architecture)
        segments.run(spec)
        grid.run(spec)
        engine.run_search(spec, stage="zerolag", slide_id=0)
        slides.run(spec)
        frozen_at = spec.path(*background.KEEP_THRESHOLD_FILE)
        assert not frozen_at.exists()

        collated = background.run(spec)
        assert collated["collated"]
        held = json.loads(frozen_at.read_text())["keep_threshold"]
        assert held == collated["keep_threshold"]

        report = far.run(spec)
        assert "inclusive" in report["curves"]
        assert report["fingerprint"]

    def test_slides_rerun_leaves_the_threshold(self, tmp_path, toy_architecture):
        """
        The threshold is background's, and lives in its own file for that reason. Stamped
        into slide_plan.h5 it had two owners: re-running slides rebuilt the plan and wiped
        it, without moving the slides fingerprint, so the rungs already scored had been
        thresholded against a value nothing on disk still held.
        """
        import sage.search.background as background
        import sage.search.engine as engine
        import sage.search.grid as grid
        import sage.search.segments as segments
        import sage.search.slides as slides

        spec = _scored_campaign(tmp_path, toy_architecture)
        segments.run(spec)
        grid.run(spec)
        engine.run_search(spec, stage="zerolag", slide_id=0)
        before = slides.run(spec)["fingerprint"]
        frozen = background.run(spec)["keep_threshold"]

        assert slides.run(spec)["fingerprint"] == before
        held = json.loads(
            spec.path(*background.KEEP_THRESHOLD_FILE).read_text()
        )["keep_threshold"]
        assert held == frozen

    def test_threshold_freeze_is_idempotent(self, tmp_path, toy_architecture):
        """
        Every background array task calls the freeze. The first writes the value and the
        rest adopt it, with no ordering requirement between them -- the read-modify-write
        over slide_plan.h5 that this replaced killed nine of ten tasks released together.
        """
        import sage.search.background as background
        import sage.search.engine as engine
        import sage.search.grid as grid
        import sage.search.segments as segments
        import sage.search.slides as slides
        from sage.search.slides import SlidePlan

        spec = _scored_campaign(tmp_path, toy_architecture)
        segments.run(spec)
        grid.run(spec)
        engine.run_search(spec, stage="zerolag", slide_id=0)
        slides.run(spec)
        plan = SlidePlan.load(spec.path("slides", "slide_plan.h5"))
        values = {background.freeze_keep_threshold(spec, plan) for _ in range(5)}
        assert len(values) == 1


class TestBlockPartition:
    """The engine and the reader must agree on what a block id names."""

    class _Reader:
        def __init__(self, blocks, block_seconds=None):
            self.blocks = blocks
            if block_seconds is not None:
                self.block_seconds = block_seconds

    def _blocks(self, n, span):
        from sage.search.grid import Block

        return [
            Block(block_id=i, gps_start=i * span, gps_end=(i + 1) * span,
                  span_slice=(i, i + 1))
            for i in range(n)
        ]

    def test_readers_blocks_are_taken_not_rederived(self):
        """
        Taken, not recomputed. Deriving the partition from ``max(block.duration_s)`` reads
        a block's *wall span* -- gaps included -- as if it were the livetime budget it was
        built from. Measured on the O3a lattice: the largest wall span is 254,401 s
        against a 32,768 s budget, so the engine walked 5 blocks where the reader held 30.
        """
        from sage.search.engine import _blocks_of

        held = self._blocks(30, 32768.0)
        walked = _blocks_of(self._Reader(held), grid=None)

        assert [b.block_id for b in walked] == [b.block_id for b in held]

    def test_a_gappy_block_does_not_coarsen_the_partition(self):
        """
        The specific shape that produced the defect: one block spanning a long gap, so its
        wall duration dwarfs the others'. Re-partitioning at that number merges the rest.
        """
        from sage.search.engine import _blocks_of

        from sage.search.grid import Block

        held = self._blocks(4, 32768.0)
        held[2] = Block(block_id=2, gps_start=0.0, gps_end=254_401.0, span_slice=(2, 3))
        walked = _blocks_of(self._Reader(held), grid=None)

        assert len(walked) == 4

    def test_falls_back_to_the_stated_budget(self):
        """A reader that holds no blocks may still say what it was built with."""
        from sage.search.engine import _blocks_of

        class _Grid:
            def blocks(self, seconds):
                return [("asked", seconds)]

        walked = _blocks_of(self._Reader([], block_seconds=1024.0), _Grid())

        assert walked == [("asked", 1024.0)]

    def test_refuses_a_reader_that_says_neither(self):
        """
        Guessing a partition is what caused this; a reader that cannot state one is an
        error rather than an occasion to infer.
        """
        from sage.search.engine import _blocks_of

        with pytest.raises(ValueError, match="neither blocks nor the block_seconds"):
            _blocks_of(self._Reader([]), grid=None)

    def test_reader_records_its_block_seconds(self, tmp_path):
        """
        The attribute the engine reads. Absent, the engine had nothing to agree with and
        inferred one instead.
        """
        from sage.search.reader import StreamingStrainReader

        assert "block_seconds" in StreamingStrainReader.__init__.__code__.co_varnames


class TestRolledCache:
    """Scoring several rolled slides in one pass, sharing frontend features."""

    def _pieces(self, tmp_path, toy_architecture):
        """A spec, a zero-lag grid, and the engine, all on the CPU."""
        from pathlib import Path

        from sage.search.checkpoint import as_config, load_search_model
        from sage.search.engine import (
            SearchEngine,
            build_param_sampler,
            build_processor,
        )
        from sage.search.grid import AnalysisGrid
        from sage.search.segments import coincident_intervals, load_segments

        spec = _scored_campaign(tmp_path, toy_architecture)
        geometry = spec.geometry_object()
        segments = {
            detector: load_segments(
                Path(spec.data.release_dir)
                / f"data_{detector}_{spec.data.observing_run}_segments.json"
            )
            for detector in spec.data.detectors
        }
        grid = AnalysisGrid.build(
            geometry, segments, coincident_intervals(segments),
            reference_detector="H1", coverage=False,
        )
        model, ckpt = load_search_model(
            spec.engine.checkpoint, cfg=None, data_cfg=None, device="cpu",
            architecture=spec.engine.architecture,
        )
        cfg, data_cfg = as_config(ckpt.cfg), as_config(ckpt.data_cfg)
        spec.apply_shadow_overrides(cfg, data_cfg)
        sampler = build_param_sampler(
            cfg, data_cfg, spec.engine.gwconfig, seed=int(spec.engine.sampler_seed)
        )
        engine = SearchEngine(
            model, build_processor(sampler), geometry, device="cpu",
            amp_dtype="float32", autocast=False, keep_threshold=float("-inf"),
        )
        return spec, grid, geometry, engine, ckpt

    def _reader(self, spec, grid, geometry, batch_size=64):
        from sage.search.reader import StreamingStrainReader

        return StreamingStrainReader(
            spec.data.release_dir, grid, geometry,
            batch_size=batch_size, prefetch=0,
        )

    def _writer(self, path, spec, ckpt, slide_id=0):
        """A shard with the provenance block the writer requires of any real product."""
        from sage.search.manifest import provenance
        from sage.search.triggers import TriggerWriter

        attrs = dict(provenance(spec, ckpt))
        attrs.update(
            clustered=False, slide_id=int(slide_id), stage="background",
            keep_threshold=float("-inf"), n_blocks=1,
        )
        return TriggerWriter(path, attrs)

    def test_matches_the_uncached_path(self, tmp_path, toy_architecture):
        """
        The gate. A cached slide must score **exactly** what re-running the whole network
        on shifted strain scores -- the cache is an arrangement of the same arithmetic, so
        anything else means features are being attributed to the wrong windows, and the
        background would be subtly unlike the zero-lag it is compared against.
        """
        from pathlib import Path

        import numpy as np

        from sage.search.grid import AnalysisGrid
        from sage.search.segments import coincident_intervals, load_segments

        spec, grid, geometry, engine, ckpt = self._pieces(tmp_path, toy_architecture)
        shift = 37

        # Uncached: build the rolled lattice and score it the ordinary way.
        segments = {
            d: load_segments(
                Path(spec.data.release_dir) / f"data_{d}_{spec.data.observing_run}"
                "_segments.json"
            )
            for d in spec.data.detectors
        }
        rolled = AnalysisGrid.build(
            geometry, segments, coincident_intervals(segments),
            reference_detector="H1", coverage=False, slide_id=1,
            window_shift={"H1": 0, "L1": shift},
        )
        plain_path = tmp_path / "plain.h5"
        plain = self._writer(plain_path, spec, ckpt, 1)
        try:
            engine.run(self._reader(spec, rolled, geometry), rolled, plain)
        finally:
            plain.close()

        # Cached: one zero-lag pass, the pairing applied by gathering shifted ordinals.
        cached_path = tmp_path / "cached.h5"
        cached = self._writer(cached_path, spec, ckpt, 1)
        try:
            engine.run_rolled(
                self._reader(spec, grid, geometry),
                [(1, {"H1": 0, "L1": shift})],
                {1: cached},
            )
        finally:
            cached.close()

        import h5py

        with h5py.File(plain_path) as a, h5py.File(cached_path) as b:
            left = np.sort(np.asarray(a["triggers/stat"]))
            right = np.sort(np.asarray(b["triggers/stat"]))
            hist_a = np.asarray(a["histogram/counts"])
            hist_b = np.asarray(b["histogram/counts"])

        assert left.size == right.size
        np.testing.assert_array_equal(hist_a, hist_b)
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-5)

    def test_every_window_scored_for_every_slide(self, tmp_path, toy_architecture):
        """
        A slide that quietly scores fewer windows than the lattice holds reports a
        livetime it did not analyse, and the false-alarm denominator is that livetime.
        """
        spec, grid, geometry, engine, ckpt = self._pieces(tmp_path, toy_architecture)
        writers = {sid: self._writer(tmp_path / f"s{sid}.h5", spec, ckpt, sid)
                   for sid in (1, 2, 3)}
        try:
            reports = engine.run_rolled(
                self._reader(spec, grid, geometry),
                [(1, {"H1": 0, "L1": 11}), (2, {"H1": 0, "L1": 29}),
                 (3, {"H1": 0, "L1": 53})],
                writers,
            )
        finally:
            for writer in writers.values():
                writer.close()

        assert set(reports) == {1, 2, 3}
        for report in reports.values():
            assert report.n_windows == len(grid)

    def test_zero_shift_refused(self, tmp_path, toy_architecture):
        """
        Nothing to share. A cache costs memory and a pass over the run; spending both to
        rescore the zero-lag is a mistake worth naming rather than absorbing.
        """
        spec, grid, geometry, engine, ckpt = self._pieces(tmp_path, toy_architecture)

        with pytest.raises(ValueError, match="zero-lag pass repeated"):
            engine.run_rolled(
                self._reader(spec, grid, geometry), [(1, {"H1": 0, "L1": 0})], {}
            )

    def test_wrapped_tail_is_scored(self, tmp_path, toy_architecture):
        """
        The pairing wraps: a reference ordinal near the end pairs with a follower near the
        start, whose features were evicted. Those windows must still be scored -- dropping
        them would shorten the background by the shift, silently.
        """
        spec, grid, geometry, engine, ckpt = self._pieces(tmp_path, toy_architecture)
        # A shift large enough that the wrap covers a real share of the lattice.
        shift = max(1, len(grid) // 4)
        writers = {1: self._writer(tmp_path / "wrap.h5", spec, ckpt, 1)}
        try:
            reports = engine.run_rolled(
                self._reader(spec, grid, geometry),
                [(1, {"H1": 0, "L1": shift})],
                writers,
            )
        finally:
            writers[1].close()

        assert reports[1].n_windows == len(grid)

    def test_each_slide_stamps_its_own_id(self, tmp_path, toy_architecture):
        """
        The cached path reads the run once through a **zero-lag** lattice and applies each
        slide by gathering shifted features, so every batch it produces carries
        ``slide_id = 0``. Stamping that labels every background trigger as foreground --
        the shard's attribute still says otherwise, and only the rows are what the
        collation groups on, so the background collates to nothing and the failure
        surfaces hours later as a missing column.
        """
        import h5py
        import numpy as np

        spec, grid, geometry, engine, ckpt = self._pieces(tmp_path, toy_architecture)
        paths = {sid: tmp_path / f"stamp{sid}.h5" for sid in (1, 2)}
        writers = {
            sid: self._writer(paths[sid], spec, ckpt, sid) for sid in paths
        }
        try:
            engine.run_rolled(
                self._reader(spec, grid, geometry),
                [(1, {"H1": 0, "L1": 13}), (2, {"H1": 0, "L1": 31})],
                writers,
            )
        finally:
            for writer in writers.values():
                writer.close()

        for slide_id, path in paths.items():
            with h5py.File(path, "r") as handle:
                rows = np.asarray(handle["triggers/slide_id"])
                assert handle.attrs["slide_id"] == slide_id
                assert rows.size, f"slide {slide_id} wrote no triggers to check"
                assert set(np.unique(rows)) == {slide_id}
