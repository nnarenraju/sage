#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_campaign.py
Description   : The injection campaign's geometry, products and plumbing.

Created on 2026-08-22

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The injection campaign draws real search-era noise rather than generating it, so what it
draws from decides the noise behind ``p(x | signal)``. Two defects lived here undetected
because nothing constructed the class: it indexed a follower detector into a lattice that
stores only the reference detector's spans, and it opened strain as a flat ``.bin`` when
the search release is segmented HDF5. Both are geometry, so both are testable here without
a GPU or a checkpoint.
"""

from pathlib import Path

import numpy as np
import pytest

from sage.search.geometry import SearchGeometry
from sage.search.grid import AnalysisGrid
from sage.search.injection.campaign import NoiseSlices
from sage.search.segments import coincident_intervals, load_segments

RATE = 2048.0
GEOMETRY = SearchGeometry(
    sample_rate=RATE,
    signal_length_s=12.0,
    padding_length_s=2.0,
    stride_samples=205,
    tc_lower_s=5.0,
    tc_upper_s=7.0,
)


class _Spec:
    """The three fields NoiseSlices reads, without a whole campaign behind them."""

    def __init__(self, release, detectors, run="O3a"):
        self.data = type(
            "_Data", (), {"release_dir": release, "detectors": detectors,
                          "observing_run": run}
        )()
        self.seed = 3

    def geometry_object(self):
        return GEOMETRY


def _grid(release, detectors):
    segments = {
        d: load_segments(release / f"data_{d}_O3a_segments.json") for d in detectors
    }
    return AnalysisGrid.build(GEOMETRY, segments, coincident_intervals(segments))


@pytest.fixture(scope="module")
def segmented(tmp_path_factory):
    """A search-grade release: one HDF5 dataset per segment, as O3a's is."""
    from tests.search_fixtures import make_synthetic_release

    return make_synthetic_release(
        tmp_path_factory.mktemp("noise_release"),
        detectors=("H1", "L1", "V1"),
        chunk_s=64.0,
        fill="noise",
        layout="segmented",
    )


class TestNoiseSlices:
    """Where the noise comes from, and that every detector can supply it."""

    @pytest.mark.parametrize("detectors", [("H1", "L1"), ("H1", "L1", "V1")])
    def test_every_detector_covers_the_lattice(self, segmented, detectors):
        """
        Followers included -- only the reference detector's spans are stored.

        A follower's analysed windows sit on its own segments at its own local offsets and
        are derived from the reference lattice, so reading them out of the lattice's span
        table raised ``KeyError`` for every network of more than one detector.
        """
        grid = _grid(segmented, detectors)
        noise = NoiseSlices(_Spec(segmented, detectors), grid, seed=1)
        try:
            for detector in detectors:
                assert sum(n for _, _, n in noise._spans[detector]) == len(grid)
        finally:
            noise.close()

    def test_draws_from_the_segmented_release(self, segmented):
        """
        The strain is read through the layout the sidecar declares.

        The search release is one dataset per segment; opening it as a flat ``.bin``
        raised ``FileNotFoundError`` on a file the release never had.
        """
        detectors = ("H1", "L1")
        grid = _grid(segmented, detectors)
        noise = NoiseSlices(_Spec(segmented, detectors), grid, seed=1)
        try:
            drawn, segment_index, local_start = noise.draw(6)
        finally:
            noise.close()
        assert drawn.shape == (6, 2, GEOMETRY.window_samples)
        assert np.isfinite(drawn).all()
        assert (drawn != 0).any(axis=(1, 2)).all()
        assert segment_index.shape == (6,) and local_start.shape == (6,)

    def test_draw_is_seeded(self, segmented):
        """Same seed, same noise: a resumed campaign must not re-noise its injections."""
        detectors = ("H1", "L1")
        grid = _grid(segmented, detectors)
        spec = _Spec(segmented, detectors)
        out = []
        for seed in (7, 7, 8):
            noise = NoiseSlices(spec, grid, seed=seed)
            try:
                out.append(noise.draw(4)[0])
            finally:
                noise.close()
        assert np.array_equal(out[0], out[1])
        assert not np.array_equal(out[0], out[2])

    def test_drawn_windows_are_analysed_windows(self, segmented):
        """
        Every draw lands on the lattice, not merely inside a segment.

        Drawing from the raw segments would sample stretches the search never reads: a
        window straddling a chunk boundary, or one in the band no window start can reach.
        """
        detectors = ("H1", "L1")
        grid = _grid(segmented, detectors)
        noise = NoiseSlices(_Spec(segmented, detectors), grid, seed=2)
        try:
            stride = GEOMETRY.stride_samples
            for detector in detectors:
                starts = {
                    (run.segment.segment_index, int(s))
                    for run in grid.runs_for_detector(detector)
                    for s in run.starts_local()
                }
                for segment, first_local, n_windows in noise._spans[detector]:
                    for k in (0, n_windows // 2, n_windows - 1):
                        assert (
                            segment.segment_index,
                            first_local + stride * k,
                        ) in starts
        finally:
            noise.close()

    def test_silent_detector_is_refused(self, segmented):
        """A detector hosting no analysed window is named, not divided by zero."""
        detectors = ("H1", "L1")
        grid = _grid(segmented, detectors)
        empty = AnalysisGrid(
            geometry=grid.geometry,
            spans_by_detector={grid.reference_detector: []},
            reference_detector=grid.reference_detector,
            segments_by_detector=grid.segments_by_detector,
        )
        with pytest.raises(ValueError, match="hosts no analysed windows"):
            NoiseSlices(_Spec(segmented, detectors), empty, seed=1)


class TestScoredShards:
    """
    Who names the injection shards.

    They are written one per stream, and two readers each spelled the name themselves
    without the stream -- so p_astro and its figure both looked for a file no campaign has
    ever produced. The name is now written once, next to the writer.
    """

    class _InjSpec:
        def __init__(self, root, streams):
            self.injection = type("_Inj", (), {"streams": streams})()
            self._root = Path(root)

        def path(self, stage, name):
            return self._root / stage / name

    def _spec(self, tmp_path, streams):
        return self._InjSpec(tmp_path, streams)

    def test_one_shard_per_stream(self, tmp_path):
        from sage.search.injection.campaign import scored_shards

        names = [p.name for p in scored_shards(self._spec(tmp_path, (0, 1, 7)))]
        assert names == [
            "injection_triggers_00.h5",
            "injection_triggers_01.h5",
            "injection_triggers_07.h5",
        ]

    def test_names_match_what_the_campaign_writes(self):
        """
        The reader's spelling and the writer's are the same expression.

        Asserted on the source, because the writer runs only on a GPU: a divergence is a
        stage that cannot find a product that was written correctly.
        """
        import inspect

        from sage.search.injection import campaign

        source = inspect.getsource(campaign)
        assert source.count('f"injection_triggers_{int(stream):02d}.h5"') == 1
        assert source.count('f"injection_triggers_{stream:02d}.h5"') == 1
        assert source.count('f"injection_triggers_{r.stream:02d}.h5"') == 1

    def test_no_reader_spells_the_name_itself(self):
        """Every other module goes through ``scored_shards``."""
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[1] / "sage" / "search"
        offenders = [
            str(path.relative_to(root))
            for path in sorted(root.rglob("*.py"))
            if path.name != "campaign.py" and "injection_triggers" in path.read_text()
        ]
        assert not offenders, offenders

    def test_missing_stream_is_named(self, tmp_path):
        """A campaign short one stream says which, rather than fitting on the rest."""
        from sage.search.injection.campaign import scored_stats

        (tmp_path / "injections").mkdir()
        with pytest.raises(FileNotFoundError, match="injection_triggers_01.h5"):
            scored_stats(self._spec(tmp_path, (0, 1)))


class TestTabulatedSampler:
    """
    The pre-drawn table stands in for the training prior.

    Its contract is that everything except the values comes from the sampler the network
    was trained under -- which includes the dtype and device the approximant's coefficient
    tables are registered in.
    """

    class _Base:
        def __init__(self):
            import torch

            self.device = torch.device("cpu")
            self.dtype = torch.float32

    def test_rows_take_the_base_dtype(self):
        """
        The population is drawn in float64; the approximant multiplies against float32.

        ``IMRPhenomD.get_coeffs`` matmuls the parameter batch against a coefficient table
        registered in the sampler's dtype, so a float64 table raised there rather than
        anywhere near the code that chose the dtype.
        """
        import torch

        from sage.search.injection.waveforms import TabulatedSampler

        table = torch.zeros((8, 3), dtype=torch.float64)
        sampler = TabulatedSampler(self._Base(), table)
        rows = sampler(4)
        assert rows.dtype == torch.float32
        assert rows.shape == (4, 3)

    def test_values_survive_the_cast(self):
        """Only the dtype changes; a cast that reordered or truncated rows would not."""
        import torch

        from sage.search.injection.waveforms import TabulatedSampler

        table = torch.arange(12, dtype=torch.float64).reshape(4, 3)
        sampler = TabulatedSampler(self._Base(), table)
        assert torch.equal(sampler(2), table[:2].to(torch.float32))
        assert torch.equal(sampler(2), table[2:].to(torch.float32))

    def test_wrapping_is_refused(self):
        """A second pass over the table would enter p(x | signal) twice."""
        import torch

        from sage.search.injection.waveforms import TabulatedSampler

        sampler = TabulatedSampler(self._Base(), torch.zeros((4, 2), dtype=torch.float64))
        sampler(3)
        with pytest.raises(IndexError, match="Wrapping would score"):
            sampler(2)


class TestCommitCadence:
    """
    How often the shard is committed, which is not how often it is scored.

    A commit snapshots the whole shard before appending to it, so its cost grows with the
    shard while the work it protects does not. Committing every generator batch made the
    campaign quadratic in its own length: 4.4 M injections at a 2,048 batch is 2,129
    commits of a shard growing to 235 MB -- about 250 GB copied to protect 235 MB.
    """

    class _Writer:
        """Records what was appended and when it was committed."""

        def __init__(self):
            self.rows = []
            self.blocks = []

        def completed_blocks(self):
            return list(self.blocks)

        def append(self, table):
            self.rows.append(int(table.columns["stat"].size))

        def complete_block(self, block_id):
            self.blocks.append(int(block_id))

    class _Engine:
        decoder = None

        def forward_frequency(self, spectra):
            n = int(spectra.shape[0])
            return np.zeros(n), np.zeros((n, 0))

    class _Injections:
        stream = 0

        def __init__(self, total, batch):
            self.total = int(total)
            self.batch_size = int(batch)

        def __len__(self):
            return self.total

        def build(self, lo, hi, noise):
            n = int(hi) - int(lo)
            return (
                np.zeros((n, 2, 4)),
                {
                    "segment_index": np.zeros(n, dtype=np.int64),
                    "local_start": np.zeros(n, dtype=np.int64),
                },
            )

    def _run(self, total, batch):
        from sage.search.injection.campaign import COMMIT_ROWS, InjectionCampaign

        writer = self._Writer()
        campaign = InjectionCampaign(
            None, self._Engine(), self._Injections(total, batch), None, writer
        )
        report = campaign.run()
        return writer, report, COMMIT_ROWS

    def test_every_injection_is_scored_once(self):
        """The cadence changes when work is committed, never how much is done."""
        writer, report, _ = self._run(10_000, 256)
        assert sum(writer.rows) == 10_000
        assert report.n_scored == 10_000

    def test_commits_are_far_rarer_than_batches(self):
        """Otherwise the snapshot cost is paid once per batch, which is the defect."""
        writer, _, commit_rows = self._run(10_000, 256)
        batches = -(-10_000 // 256)
        assert len(writer.blocks) < batches
        assert len(writer.blocks) == -(-10_000 // ((commit_rows // 256) * 256))

    def test_block_ids_are_contiguous_from_zero(self):
        """They are the resume markers; a gap would replay work that was committed."""
        writer, _, _ = self._run(10_000, 256)
        assert writer.blocks == list(range(len(writer.blocks)))

    def test_a_batch_larger_than_the_commit_still_commits(self):
        """A generator batch above COMMIT_ROWS must not round the block size to zero."""
        from sage.search.injection.campaign import COMMIT_ROWS

        writer, report, _ = self._run(COMMIT_ROWS * 2, COMMIT_ROWS * 2)
        assert writer.blocks == [0]
        assert report.n_scored == COMMIT_ROWS * 2

    def test_completed_blocks_are_skipped(self):
        """A resumed campaign must not rescore what it already wrote."""
        from sage.search.injection.campaign import COMMIT_ROWS, InjectionCampaign

        writer = self._Writer()
        writer.blocks = [0]
        batch = 256
        total = COMMIT_ROWS * 2
        campaign = InjectionCampaign(
            None, self._Engine(), self._Injections(total, batch), None, writer
        )
        report = campaign.run()
        assert report.n_scored == COMMIT_ROWS


class TestMassFrame:
    """
    The injected masses are detector frame, which is PyCBC's convention.

    The population model states source-frame masses; a waveform generator is handed
    detector-frame ones. PyCBC keeps the two apart by name -- unqualified ``mass1`` is
    what the generator receives, while ``srcmass1``/``srcmchirp`` are separate parameters
    filed under "derived parameters (these are not used for waveform generation)"
    (pycbc/waveform/parameters.py) -- and relates them by ``msrc = mdet / (1 + z)``
    (pycbc/mchirp_area.py:134). The table previously spent the redshift on the distance
    and withheld it from the masses, placing a binary at its correct luminosity distance
    while leaving it too light for that distance. See SB-50.
    """

    class _Sampler:
        """The training sampler's column contract, without a checkpoint behind it."""

        COLUMNS = (
            "mass1", "mass2", "mchirp", "q", "distance",
            "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z",
            "spin1_a", "spin2_a", "spin1_polar", "spin2_polar",
            "spin1_azimuthal", "spin2_azimuthal",
        )

        def __init__(self):
            import torch

            self.param_index = {name: i for i, name in enumerate(self.COLUMNS)}
            self.bounds = {"mass1": (7.0, 50.0)}
            self.device = torch.device("cpu")
            self.dtype = torch.float64

        def __call__(self, n):
            import torch

            return torch.zeros((n, len(self.COLUMNS)), dtype=torch.float64)

    def _intrinsic(self, z):
        """One binary per redshift, all else fixed, so only the frame factor varies."""
        n = len(z)
        out = np.zeros((n, 7))
        out[:, 0] = 30.0            # m1 source
        out[:, 1] = 0.8             # q
        out[:, 2] = np.asarray(z)
        out[:, 5] = 1.0             # cos tilts, to keep spins aligned and finite
        out[:, 6] = 1.0
        return out

    def test_masses_are_redshifted(self):
        from sage.search.injection.waveforms import build_injection_table

        sampler = self._Sampler()
        z = np.array([0.0, 0.5, 1.0, 2.0])
        table = np.asarray(build_injection_table(sampler, self._intrinsic(z), seed=1))
        i = sampler.param_index
        assert table[:, i["mass1"]] == pytest.approx(30.0 * (1.0 + z))
        assert table[:, i["mass2"]] == pytest.approx(0.8 * 30.0 * (1.0 + z))

    def test_zero_redshift_is_the_identity(self):
        """The conversion must not perturb a source at z = 0."""
        from sage.search.injection.waveforms import build_injection_table

        sampler = self._Sampler()
        table = np.asarray(
            build_injection_table(sampler, self._intrinsic([0.0]), seed=1)
        )
        assert table[0, sampler.param_index["mass1"]] == pytest.approx(30.0)

    def test_mass_ratio_is_frame_invariant(self):
        """q survives the conversion, since both masses scale together."""
        from sage.search.injection.waveforms import build_injection_table

        sampler = self._Sampler()
        z = np.array([0.0, 1.0, 2.0])
        table = np.asarray(build_injection_table(sampler, self._intrinsic(z), seed=1))
        i = sampler.param_index
        assert table[:, i["q"]] == pytest.approx(0.8)
        assert table[:, i["mass2"]] / table[:, i["mass1"]] == pytest.approx(0.8)

    def test_derived_columns_follow_the_detector_frame_masses(self):
        """
        ``mchirp`` is what a recovered-versus-injected comparison reads.

        Left on the source-frame masses it would disagree with the ``mass1``/``mass2``
        beside it, and the network's own chirp-mass estimate is in the frame it generates
        waveforms in.
        """
        from sage.search.injection.waveforms import build_injection_table

        sampler = self._Sampler()
        z = np.array([0.0, 1.0, 2.0])
        table = np.asarray(build_injection_table(sampler, self._intrinsic(z), seed=1))
        i = sampler.param_index
        m1, m2 = table[:, i["mass1"]], table[:, i["mass2"]]
        assert table[:, i["mchirp"]] == pytest.approx(
            (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2, rel=1e-6
        )

    def test_distance_is_untouched_by_the_frame(self):
        """
        Luminosity distance already carries the redshift; it must not carry it twice.

        This is the half that was always right, and the asymmetry with the masses is what
        made the table internally inconsistent.
        """
        from astropy import units as u
        from astropy.cosmology import Planck15
        from astropy.cosmology import units as cu

        from sage.search.injection.waveforms import build_injection_table

        sampler = self._Sampler()
        z = np.array([0.5, 1.0])
        table = np.asarray(build_injection_table(sampler, self._intrinsic(z), seed=1))
        expected = (
            (z * cu.redshift)
            .to(u.Mpc, cu.redshift_distance(Planck15, kind="luminosity"))
            .value
        )
        assert table[:, sampler.param_index["distance"]] == pytest.approx(expected)
