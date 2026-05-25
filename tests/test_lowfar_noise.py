"""Unit tests for sage.data.noise.lowfar_noise — CPU-only, no GPU/pycbc/lal needed.

Strategy
--------
* pycbc is stubbed via sys.modules before any import that would pull it in.
* sage.data.noise.__init__ is bypassed (it imports lowfar_noise → pycbc/lal
  transitively), mirroring the technique in test_data_hard_mining.py.
* A lightweight FakeNoiseSampler and FixedScoreModel let every miner be
  exercised end-to-end with tiny parameters (≤ 10 forward passes total).
* get_cfg is monkeypatched wherever a miner or _MiningReader._empty_dataset
  calls it so no real config file is required.
"""

import json
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------

pytest.importorskip("h5py", reason="sage.core.logger requires h5py")
pytest.importorskip("tqdm", reason="lowfar_noise requires tqdm")

# Ensure pycbc is importable so "from pycbc import DYN_RANGE_FAC" in
# lowfar_noise.py succeeds.  Use the real package when it is installed;
# only fall back to a stub when it is genuinely absent.  A module-level stub
# that replaces a real package would poison sys.modules for every other test
# in the same session (e.g. test_dsp_heterodyning.py needs pycbc.waveform).
try:
    import pycbc as _pycbc  # noqa: F401  — triggers real package registration
except ImportError:
    _pycbc_stub = types.ModuleType("pycbc")
    _pycbc_stub.DYN_RANGE_FAC = 1.0
    sys.modules["pycbc"] = _pycbc_stub

# Bypass sage.data.noise.__init__ which would import lowfar_noise → pycbc/lal.
_SAGE_ROOT = Path(__file__).resolve().parents[1] / "sage"


def _bypass_pkg(name):
    if name not in sys.modules:
        parts = name.split(".")[1:]
        mod = types.ModuleType(name)
        mod.__path__ = [str(_SAGE_ROOT.joinpath(*parts))]
        mod.__package__ = name
        sys.modules[name] = mod


_bypass_pkg("sage.data.noise")

from sage.data.noise.lowfar_noise import (  # noqa: E402
    StartTimeDataset,
    _MAPElitesArchive,
    _MiningReader,
    BruteForceMiner,
    CEMRareEventMiner,
    MAPElitesMiner,
    StartTimeNoiseSampler,
)
# Retrieve the already-loaded module object without walking the sage.data chain.
_lnm = sys.modules["sage.data.noise.lowfar_noise"]

# pin_memory() requires an NVIDIA driver even for CPU tensors.  Replace it
# with a no-op for the duration of this test module so tests can run on
# machines without a GPU.
torch.Tensor.pin_memory = lambda self, device=None: self

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_SR = 4096.0
_SEQ_LEN = 64         # short window keeps tests fast
_N_SAMPLES = 512      # 2× largest safe start index + seq_len
_DETECTORS = ["H1", "L1"]
_D = len(_DETECTORS)
_ENDIAN = "<" if sys.byteorder == "little" else ">"

_SEG_DTYPE = np.dtype([
    ("idx", np.int64),
    ("start", np.int64),
    ("end", np.int64),
    ("nsamples", np.int64),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sidecar(gps_start=1_234_567_890.0):
    return [{
        "segment_index": 0,
        "gps_start": gps_start,
        "sample_start_idx": 0,
        "sample_rate": _SR,
        "dtype": "float32",
        "endianness": _ENDIAN,
    }]


def _seg_arr():
    return np.array([(0, 0, _N_SAMPLES, _N_SAMPLES)], dtype=_SEG_DTYPE)


def _fake_cfg():
    return types.SimpleNamespace(detectors=_DETECTORS)


class _FakeSampler:
    """Minimal stand-in for MemmapNoiseSampler, borrowable by _MiningReader."""

    def __init__(self, bin_files):
        self.bin_files = [Path(p) for p in bin_files]
        self.seq_len = _SEQ_LEN
        self.n_detectors = len(bin_files)
        self.device = "cpu"
        self.postprocess_fn = None
        seg = _seg_arr()
        self.seg_index = [seg] * self.n_detectors
        self.segment_probs = [np.array([1.0])] * self.n_detectors
        self.mmaps = [
            np.memmap(str(p), dtype=np.dtype(f"{_ENDIAN}f4"), mode="r")
            for p in self.bin_files
        ]


class _FixedModel(nn.Module):
    """Always returns the same ranking statistic, regardless of input."""

    def __init__(self, score: float = 5.0):
        super().__init__()
        self._score = score

    def forward(self, x):
        return (torch.full((x.shape[0], 1), self._score),)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_env(tmp_path):
    """Write a minimal .bin + _segments.json sidecar for each detector."""
    rng = np.random.default_rng(0)
    bin_files = []
    for i, det in enumerate(_DETECTORS):
        p = tmp_path / f"{det}.bin"
        rng.standard_normal(_N_SAMPLES).astype(np.float32).tofile(str(p))
        with open(tmp_path / f"{det}_segments.json", "w") as f:
            json.dump(_sidecar(gps_start=1_234_567_890.0 + i * 3600), f)
        bin_files.append(p)
    return bin_files


@pytest.fixture
def sampler(fake_env):
    return _FakeSampler(fake_env)


@pytest.fixture
def reader(sampler, monkeypatch):
    monkeypatch.setattr(_lnm, "get_cfg", _fake_cfg)
    return _MiningReader(sampler, seed=0)


@pytest.fixture
def patched_cfg(monkeypatch):
    """Patch get_cfg so miners can build their return datasets."""
    monkeypatch.setattr(_lnm, "get_cfg", _fake_cfg)


@pytest.fixture
def simple_ds(fake_env):
    """A small StartTimeDataset whose bin_files point to real fake files."""
    N = 10
    return StartTimeDataset(
        detectors=_DETECTORS,
        start_indices=np.zeros((N, _D), dtype=np.int64),
        segment_indices=np.zeros((N, _D), dtype=np.int64),
        gps_times=np.linspace(1_234_567_890.0, 1_234_567_890.0 + N, N),
        scores=np.ones(N, dtype=np.float32) * 5.0,
        bin_files=[str(p) for p in fake_env],
        sample_rate=_SR,
        seq_len=_SEQ_LEN,
    )


# ---------------------------------------------------------------------------
# TestStartTimeDataset
# ---------------------------------------------------------------------------


class TestStartTimeDataset:
    def _make(self, n=10):
        return StartTimeDataset(
            detectors=_DETECTORS,
            start_indices=np.arange(n * _D, dtype=np.int64).reshape(n, _D),
            segment_indices=np.zeros((n, _D), dtype=np.int64),
            gps_times=np.arange(n, dtype=np.float64),
            scores=np.arange(n, dtype=np.float32),
            bin_files=["/fake/H1.bin", "/fake/L1.bin"],
            sample_rate=_SR,
            seq_len=_SEQ_LEN,
        )

    def test_len(self):
        assert len(self._make(7)) == 7

    def test_repr_empty(self):
        ds = StartTimeDataset(
            detectors=_DETECTORS,
            start_indices=np.empty((0, _D), dtype=np.int64),
            segment_indices=np.empty((0, _D), dtype=np.int64),
            gps_times=np.empty(0),
            scores=np.empty(0, dtype=np.float32),
            bin_files=[],
            sample_rate=_SR,
            seq_len=_SEQ_LEN,
        )
        assert "0 samples" in repr(ds)

    def test_repr_nonempty_contains_count(self):
        r = repr(self._make(5))
        assert "5" in r
        assert "H1" in r

    def test_save_load_roundtrip(self, tmp_path):
        ds = self._make(8)
        ds.save(str(tmp_path / "ds.npz"))
        ds2 = StartTimeDataset.load(str(tmp_path / "ds.npz"))
        np.testing.assert_array_equal(ds2.start_indices, ds.start_indices)
        np.testing.assert_array_equal(ds2.scores, ds.scores)
        assert ds2.detectors == ds.detectors
        assert ds2.sample_rate == ds.sample_rate
        assert ds2.seq_len == ds.seq_len

    def test_save_auto_adds_npz_suffix(self, tmp_path):
        self._make(4).save(str(tmp_path / "out"))   # no suffix
        assert (tmp_path / "out.npz").exists()
        assert (tmp_path / "out.json").exists()

    def test_load_without_npz_suffix(self, tmp_path):
        self._make(4).save(str(tmp_path / "ds.npz"))
        ds2 = StartTimeDataset.load(str(tmp_path / "ds"))  # no .npz
        assert len(ds2) == 4

    def test_filter_keeps_correct_subset(self):
        ds = self._make(10)      # scores 0..9
        filtered = ds.filter(5.0)
        assert len(filtered) == 5
        assert (filtered.scores >= 5.0).all()

    def test_filter_to_empty(self):
        ds = self._make(5)
        assert len(ds.filter(100.0)) == 0

    def test_merge_lengths_add(self):
        merged = self._make(6).merge(self._make(4))
        assert len(merged) == 10

    def test_merge_scores_concatenated(self):
        ds1, ds2 = self._make(3), self._make(3)
        merged = ds1.merge(ds2)
        np.testing.assert_array_equal(
            merged.scores, np.concatenate([ds1.scores, ds2.scores])
        )

    def test_merge_start_indices_concatenated(self):
        ds1, ds2 = self._make(4), self._make(3)
        merged = ds1.merge(ds2)
        np.testing.assert_array_equal(
            merged.start_indices,
            np.concatenate([ds1.start_indices, ds2.start_indices], axis=0),
        )

    def test_merge_detector_mismatch_raises(self):
        ds1 = self._make(3)
        ds2 = StartTimeDataset(
            detectors=["V1"],
            start_indices=np.zeros((3, 1), dtype=np.int64),
            segment_indices=np.zeros((3, 1), dtype=np.int64),
            gps_times=np.zeros(3),
            scores=np.zeros(3, dtype=np.float32),
            bin_files=["/fake/V1.bin"],
            sample_rate=_SR,
            seq_len=_SEQ_LEN,
        )
        with pytest.raises(AssertionError):
            ds1.merge(ds2)


# ---------------------------------------------------------------------------
# TestMAPElitesArchive
# ---------------------------------------------------------------------------


class TestMAPElitesArchive:
    def _arc(self, n_cells=10, samples_per_cell=3):
        return _MAPElitesArchive(
            n_cells=n_cells,
            samples_per_cell=samples_per_cell,
            gps_t_min=0.0,
            gps_t_max=100.0,
            n_detectors=_D,
        )

    def test_gps_to_cell_below_range_clamps_to_zero(self):
        assert self._arc()._gps_to_cell(-50.0) == 0

    def test_gps_to_cell_above_range_clamps_to_last(self):
        arc = self._arc(n_cells=10)
        assert arc._gps_to_cell(999.0) == 9

    def test_gps_to_cell_midpoint(self):
        # cell_width = 100/10 = 10; gps=55 → cell 5
        assert self._arc(n_cells=10)._gps_to_cell(55.0) == 5

    def test_update_fills_below_capacity(self):
        arc = self._arc(n_cells=1, samples_per_cell=5)
        n = arc.update(
            np.array([1.0, 2.0, 3.0]),
            np.zeros((3, _D), dtype=np.int64),
            np.zeros((3, _D), dtype=np.int64),
            np.zeros(3),
        )
        assert n == 3
        assert arc.total_samples == 3

    def test_update_replaces_worse_score(self):
        arc = self._arc(n_cells=1, samples_per_cell=2)
        arc.update(
            np.array([1.0, 2.0]),
            np.zeros((2, _D), dtype=np.int64),
            np.zeros((2, _D), dtype=np.int64),
            np.zeros(2),
        )
        n = arc.update(
            np.array([10.0]),
            np.zeros((1, _D), dtype=np.int64),
            np.zeros((1, _D), dtype=np.int64),
            np.zeros(1),
        )
        assert n == 1
        arc._rebuild_flat()
        assert arc._flat_scores.max() == pytest.approx(10.0)

    def test_update_no_improvement_returns_zero(self):
        arc = self._arc(n_cells=1, samples_per_cell=2)
        arc.update(
            np.array([5.0, 6.0]),
            np.zeros((2, _D), dtype=np.int64),
            np.zeros((2, _D), dtype=np.int64),
            np.zeros(2),
        )
        n = arc.update(
            np.array([1.0]),
            np.zeros((1, _D), dtype=np.int64),
            np.zeros((1, _D), dtype=np.int64),
            np.zeros(1),
        )
        assert n == 0

    def test_total_samples_and_n_filled_cells(self):
        arc = self._arc(n_cells=5, samples_per_cell=3)
        arc.update(
            np.ones(5),
            np.zeros((5, _D), dtype=np.int64),
            np.zeros((5, _D), dtype=np.int64),
            np.array([10.0, 30.0, 50.0, 70.0, 90.0]),
        )
        assert arc.n_filled_cells == 5
        assert arc.total_samples == 5

    def test_rebuild_flat_shapes(self):
        arc = self._arc(n_cells=2, samples_per_cell=5)
        N = 4
        arc.update(
            np.ones(N),
            np.zeros((N, _D), dtype=np.int64),
            np.zeros((N, _D), dtype=np.int64),
            np.linspace(0, 99, N),
        )
        arc._rebuild_flat()
        assert arc._flat_starts.shape == (N, _D)
        assert arc._flat_scores.shape == (N,)
        assert arc._flat_segs.shape == (N, _D)

    def test_propose_mutations_shape(self, reader):
        arc = self._arc(n_cells=2, samples_per_cell=5)
        N = 6
        arc.update(
            np.ones(N) * 3.0,
            np.zeros((N, _D), dtype=np.int64),
            np.zeros((N, _D), dtype=np.int64),
            np.linspace(0, 99, N),
        )
        rng = np.random.default_rng(1)
        starts, segs = arc.propose_mutations(4, rng, sigma_samples=10, reader=reader)
        assert starts.shape == (4, _D)
        assert segs.shape == (4, _D)

    def test_propose_mutations_empty_archive_falls_back_to_random(self, reader):
        arc = self._arc()   # empty
        rng = np.random.default_rng(1)
        starts, segs = arc.propose_mutations(3, rng, sigma_samples=10, reader=reader)
        assert starts.shape == (3, _D)
        assert segs.shape == (3, _D)

    def test_propose_mutations_triggers_rebuild(self, reader):
        """propose_mutations must work even before _rebuild_flat is called."""
        arc = self._arc(n_cells=2, samples_per_cell=5)
        arc.update(
            np.ones(3) * 2.0,
            np.zeros((3, _D), dtype=np.int64),
            np.zeros((3, _D), dtype=np.int64),
            np.linspace(0, 99, 3),
        )
        assert not arc._flat_valid
        rng = np.random.default_rng(2)
        starts, _ = arc.propose_mutations(2, rng, sigma_samples=5, reader=reader)
        assert starts.shape == (2, _D)


# ---------------------------------------------------------------------------
# TestMiningReader
# ---------------------------------------------------------------------------


class TestMiningReader:
    def test_gps_range_returns_ordered_floats(self, reader):
        t_min, t_max = reader.gps_range()
        assert isinstance(t_min, float)
        assert isinstance(t_max, float)
        assert t_min < t_max

    def test_gps_from_starts_shape(self, reader):
        starts, segs = reader.random_starts(5)
        gps = reader.gps_from_starts(starts, segs)
        assert gps.shape == (5,)
        assert gps.dtype == np.float64

    def test_gps_from_starts_ballpark(self, reader):
        starts = np.zeros((1, _D), dtype=np.int64)
        segs = np.zeros((1, _D), dtype=np.int64)
        gps = reader.gps_from_starts(starts, segs)
        # H1 sidecar has gps_start = 1_234_567_890; index 0 → offset 0
        assert abs(gps[0] - 1_234_567_890.0) < 1.0

    def test_random_starts_shape(self, reader):
        starts, segs = reader.random_starts(8)
        assert starts.shape == (8, _D)
        assert segs.shape == (8, _D)
        assert starts.dtype == np.int64
        assert segs.dtype == np.int64

    def test_random_starts_in_memmap_bounds(self, reader):
        starts, segs = reader.random_starts(50)
        for d in range(_D):
            assert np.all(starts[:, d] >= 0)
            assert np.all(starts[:, d] + _SEQ_LEN <= _N_SAMPLES)

    def test_random_starts_custom_weights(self, reader):
        weights = [np.array([1.0])] * _D
        starts, segs = reader.random_starts(4, weights=weights)
        assert starts.shape == (4, _D)

    def test_mutate_starts_in_bounds(self, reader):
        starts, segs = reader.random_starts(10)
        new_starts, _ = reader.mutate_starts(starts, segs, sigma_samples=200)
        for d in range(_D):
            assert np.all(new_starts[:, d] >= 0)
            assert np.all(new_starts[:, d] + _SEQ_LEN <= _N_SAMPLES)

    def test_mutate_starts_preserves_segs(self, reader):
        starts, segs = reader.random_starts(5)
        _, new_segs = reader.mutate_starts(starts, segs, sigma_samples=10)
        np.testing.assert_array_equal(new_segs, segs)

    def test_read_batch_shape(self, reader):
        starts, segs = reader.random_starts(4)
        out = reader.read_batch(starts, segs)
        expected_f = _SEQ_LEN // 2 + 1
        assert out.shape == (4, _D, expected_f)

    def test_read_batch_is_complex(self, reader):
        starts, segs = reader.random_starts(2)
        out = reader.read_batch(starts, segs)
        assert out.is_complex()

    def test_empty_dataset_length_zero(self, reader, sampler):
        ds = reader._empty_dataset(sampler)
        assert len(ds) == 0

    def test_empty_dataset_metadata(self, reader, sampler):
        ds = reader._empty_dataset(sampler)
        assert ds.detectors == _DETECTORS
        assert ds.sample_rate == _SR
        assert ds.seq_len == _SEQ_LEN
        assert ds.start_indices.shape == (0, _D)

    def test_score_percentile_str_empty(self):
        assert _MiningReader._score_percentile_str(np.array([])) == "n/a"

    def test_score_percentile_str_nonempty(self):
        s = _MiningReader._score_percentile_str(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert "/" in s
        assert "50" in s


# ---------------------------------------------------------------------------
# TestBruteForceMiner
# ---------------------------------------------------------------------------


class TestBruteForceMiner:
    def test_mine_above_threshold_returns_nonempty(self, sampler, patched_cfg):
        miner = BruteForceMiner(threshold=0.0, batch_size=8, prune_every=5)
        result = miner.mine(
            _FixedModel(5.0), sampler, processor=lambda x: x,
            device="cpu", n_windows=32,
        )
        assert isinstance(result, StartTimeDataset)
        assert len(result) > 0

    def test_mine_below_threshold_returns_empty(self, sampler, patched_cfg):
        miner = BruteForceMiner(threshold=100.0, batch_size=8)
        result = miner.mine(
            _FixedModel(1.0), sampler, processor=lambda x: x,
            device="cpu", n_windows=16,
        )
        assert len(result) == 0

    def test_mine_respects_max_samples(self, sampler, patched_cfg):
        miner = BruteForceMiner(
            threshold=0.0, batch_size=8, max_samples=5, prune_every=1
        )
        result = miner.mine(
            _FixedModel(5.0), sampler, processor=lambda x: x,
            device="cpu", n_windows=80,
        )
        assert len(result) <= 5

    def test_mine_restores_training_mode(self, sampler, patched_cfg):
        model = _FixedModel(5.0)
        model.train()
        BruteForceMiner(threshold=0.0, batch_size=4).mine(
            model, sampler, processor=lambda x: x,
            device="cpu", n_windows=8,
        )
        assert model.training

    def test_mine_scores_sorted_descending(self, sampler, patched_cfg):
        miner = BruteForceMiner(threshold=0.0, batch_size=8)
        result = miner.mine(
            _FixedModel(5.0), sampler, processor=lambda x: x,
            device="cpu", n_windows=32,
        )
        if len(result) > 1:
            assert np.all(np.diff(result.scores) <= 0)

    def test_mine_gps_times_shape(self, sampler, patched_cfg):
        miner = BruteForceMiner(threshold=0.0, batch_size=4)
        result = miner.mine(
            _FixedModel(5.0), sampler, processor=lambda x: x,
            device="cpu", n_windows=16,
        )
        assert result.gps_times.shape == (len(result),)

    def test_mine_prune_triggers_mid_run(self, sampler, patched_cfg):
        # prune_every=1 so the top-K prune path is exercised on every batch
        miner = BruteForceMiner(
            threshold=0.0, batch_size=4, max_samples=3, prune_every=1
        )
        result = miner.mine(
            _FixedModel(5.0), sampler, processor=lambda x: x,
            device="cpu", n_windows=40,
        )
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# TestMAPElitesMiner
# ---------------------------------------------------------------------------


class TestMAPElitesMiner:
    def _miner(self, threshold=0.0):
        return MAPElitesMiner(
            n_cells=5, samples_per_cell=4,
            init_batches=2, n_iterations=3,
            batch_size=4, threshold=threshold,
        )

    def test_mine_returns_dataset(self, sampler, patched_cfg):
        result = self._miner().mine(
            _FixedModel(5.0), sampler, processor=lambda x: x, device="cpu"
        )
        assert isinstance(result, StartTimeDataset)
        assert len(result) > 0

    def test_mine_scores_above_threshold(self, sampler, patched_cfg):
        result = self._miner(threshold=0.0).mine(
            _FixedModel(5.0), sampler, processor=lambda x: x, device="cpu"
        )
        assert (result.scores >= 0.0).all()

    def test_mine_empty_when_all_below_threshold(self, sampler, patched_cfg):
        result = self._miner(threshold=100.0).mine(
            _FixedModel(1.0), sampler, processor=lambda x: x, device="cpu"
        )
        assert len(result) == 0

    def test_mine_restores_training_mode(self, sampler, patched_cfg):
        model = _FixedModel(5.0)
        model.train()
        self._miner().mine(model, sampler, processor=lambda x: x, device="cpu")
        assert model.training

    def test_mine_gps_times_present(self, sampler, patched_cfg):
        result = self._miner().mine(
            _FixedModel(5.0), sampler, processor=lambda x: x, device="cpu"
        )
        assert result.gps_times.shape == (len(result),)


# ---------------------------------------------------------------------------
# TestCEMRareEventMiner
# ---------------------------------------------------------------------------


class TestCEMRareEventMiner:
    def _miner(self, threshold=0.0, n_gen=3):
        return CEMRareEventMiner(
            n_generations=n_gen, batch_size=8,
            elite_fraction=0.5, threshold=threshold,
        )

    def test_mine_returns_dataset(self, sampler, patched_cfg):
        result = self._miner().mine(
            _FixedModel(5.0), sampler, processor=lambda x: x, device="cpu"
        )
        assert isinstance(result, StartTimeDataset)
        assert len(result) > 0

    def test_mine_scores_above_threshold(self, sampler, patched_cfg):
        result = self._miner(threshold=0.0).mine(
            _FixedModel(5.0), sampler, processor=lambda x: x, device="cpu"
        )
        assert (result.scores >= 0.0).all()

    def test_mine_empty_when_all_below_threshold(self, sampler, patched_cfg):
        result = self._miner(threshold=100.0).mine(
            _FixedModel(1.0), sampler, processor=lambda x: x, device="cpu"
        )
        assert len(result) == 0

    def test_mine_restores_training_mode(self, sampler, patched_cfg):
        model = _FixedModel(5.0)
        model.train()
        self._miner().mine(model, sampler, processor=lambda x: x, device="cpu")
        assert model.training

    def test_mine_logs_weight_entropy(self, sampler, patched_cfg, capsys):
        # log_every = max(1, n_gen // 10); with n_gen=10 it logs every gen
        self._miner(n_gen=10).mine(
            _FixedModel(5.0), sampler, processor=lambda x: x, device="cpu"
        )
        captured = capsys.readouterr()
        assert "entropy" in captured.out.lower()

    def test_mine_weight_update_changes_distribution(self, sampler, patched_cfg):
        # With a very high learning rate the weights should diverge from uniform.
        miner = CEMRareEventMiner(
            n_generations=5, batch_size=8,
            elite_fraction=0.5, learning_rate=0.9,
            diversity_floor=1e-6, threshold=0.0,
        )
        miner.mine(_FixedModel(5.0), sampler, processor=lambda x: x, device="cpu")
        # Smoke test — just confirming it runs without error

    def test_mine_scores_sorted_descending(self, sampler, patched_cfg):
        result = self._miner().mine(
            _FixedModel(5.0), sampler, processor=lambda x: x, device="cpu"
        )
        if len(result) > 1:
            assert np.all(np.diff(result.scores) <= 0)


# ---------------------------------------------------------------------------
# TestStartTimeNoiseSampler
# ---------------------------------------------------------------------------


class TestStartTimeNoiseSampler:
    def test_init_attributes(self, simple_ds):
        samp = StartTimeNoiseSampler(
            dataset=simple_ds, postprocess_fn=None, batch_size=4, device="cpu",
        )
        assert samp.n_samples == len(simple_ds)
        assert samp.seq_len == _SEQ_LEN
        assert samp.n_detectors == _D
        assert samp.batch_size == 4
        samp.shutdown()

    def test_forward_fd_shape(self, simple_ds):
        B = 3
        samp = StartTimeNoiseSampler(
            dataset=simple_ds, postprocess_fn=None, batch_size=B, device="cpu",
        )
        fd, target = samp()
        expected_f = _SEQ_LEN // 2 + 1
        assert fd.shape == (B, _D, expected_f)
        assert fd.is_complex()
        samp.shutdown()

    def test_forward_target_shape(self, simple_ds):
        B = 2
        samp = StartTimeNoiseSampler(
            dataset=simple_ds, postprocess_fn=None, batch_size=B, device="cpu",
        )
        _, target = samp()
        assert target.shape == (B, 1)
        assert target.dtype == torch.float32
        samp.shutdown()

    def test_forward_target_is_zeros(self, simple_ds):
        samp = StartTimeNoiseSampler(
            dataset=simple_ds, postprocess_fn=None, batch_size=4, device="cpu",
        )
        _, target = samp()
        assert (target == 0.0).all()
        samp.shutdown()

    def test_multiple_forward_calls(self, simple_ds):
        samp = StartTimeNoiseSampler(
            dataset=simple_ds, postprocess_fn=None,
            batch_size=2, device="cpu", prefetch=2,
        )
        for _ in range(4):
            fd, _ = samp()
            assert fd.shape[0] == 2
        samp.shutdown()

    def test_shutdown_stops_thread(self, simple_ds):
        samp = StartTimeNoiseSampler(
            dataset=simple_ds, postprocess_fn=None, batch_size=2, device="cpu",
        )
        thread = samp._thread
        assert thread.is_alive()
        samp.shutdown()
        thread.join(timeout=3.0)
        assert not thread.is_alive()

    def test_noise_target_on_correct_device(self, simple_ds):
        samp = StartTimeNoiseSampler(
            dataset=simple_ds, postprocess_fn=None, batch_size=2, device="cpu",
        )
        assert samp.noise_target.device.type == "cpu"
        samp.shutdown()
