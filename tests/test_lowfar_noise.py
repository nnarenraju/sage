"""Unit tests for sage.data.noise.lowfar_noise — CPU-only, no GPU/pycbc/lal needed.

Strategy
--------
* pycbc is stubbed via sys.modules before any import that would pull it in.
* sage.data.noise.__init__ is bypassed (it imports lowfar_noise → pycbc/lal
  transitively).
* A lightweight FakeNoiseSampler and FixedScoreModel let StartTimeDataset and
  _MiningReader be exercised end-to-end with tiny parameters.
* get_cfg is monkeypatched wherever _MiningReader._empty_dataset
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

# lowfar_noise.py does "from pycbc import DYN_RANGE_FAC" at import time.
# We need pycbc in sys.modules only for that one import; afterwards we remove
# the stub so that other tests in the session (e.g. test_dsp_heterodyning.py)
# see the correct state — pycbc absent — and are skipped via importorskip
# rather than running against a hollow stub that has no submodules.
_pycbc_stubbed = False
try:
    import pycbc as _pycbc  # noqa: F401  real package — nothing to clean up
except ImportError:
    _pycbc_stub = types.ModuleType("pycbc")
    _pycbc_stub.DYN_RANGE_FAC = 1.0
    sys.modules["pycbc"] = _pycbc_stub
    _pycbc_stubbed = True

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
    _MiningReader,
)
# Retrieve the already-loaded module object without walking the sage.data chain.
_lnm = sys.modules["sage.data.noise.lowfar_noise"]

# lowfar_noise reaches DYN_RANGE_FAC *lazily* (sage.data.noise._pycbc_lazy),
# importing pycbc only on the first dyn_range_fac() call — which happens inside
# read_batch(), well after this point.  Prime that cache from the live stub
# (=1.0) before removing it, otherwise the first read_batch() would trigger a
# `from pycbc import ...` against the now-absent stub.  Removing the hollow stub
# afterwards keeps pytest.importorskip("pycbc") skipping tests that need real pycbc.
if _pycbc_stubbed:
    _lnm.dyn_range_fac()             # cache DYN_RANGE_FAC (=1.0) while the stub is live
    del sys.modules["pycbc"]

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

    def _dup_ds(self, rows, scores):
        rows = np.asarray(rows, dtype=np.int64)
        return StartTimeDataset(
            detectors=_DETECTORS,
            start_indices=rows,
            segment_indices=np.zeros_like(rows),
            gps_times=np.arange(len(rows), dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float32),
            bin_files=["/fake/H1.bin", "/fake/L1.bin"],
            sample_rate=_SR,
            seq_len=_SEQ_LEN,
        )

    def test_dedup_keeps_unique_start_rows(self):
        # rows 0 and 2 are identical (same window); 1 and 3 are distinct.
        ds = self._dup_ds(
            rows=[[10, 20], [30, 40], [10, 20], [50, 60]],
            scores=[1.0, 2.0, 5.0, 3.0],
        )
        out = ds.dedup()
        assert len(out) == 3                              # one duplicate removed
        uniq = {tuple(r) for r in out.start_indices.tolist()}
        assert uniq == {(10, 20), (30, 40), (50, 60)}

    def test_dedup_keeps_highest_score_of_duplicates(self):
        ds = self._dup_ds(
            rows=[[10, 20], [10, 20], [10, 20]],
            scores=[1.0, 9.0, 4.0],
        )
        out = ds.dedup()
        assert len(out) == 1
        assert out.scores[0] == 9.0                       # strongest occurrence kept

    def test_dedup_accumulation_does_not_grow_on_repeat(self):
        # Re-mining the same tail windows every epoch must not grow the set.
        ds = self._dup_ds(rows=[[1, 2], [3, 4]], scores=[7.0, 8.0])
        acc = ds
        for _ in range(5):
            acc = acc.merge(ds).dedup()                   # same windows re-found
        assert len(acc) == 2

    def test_dedup_noop_when_all_unique(self):
        ds = self._make(6)                                # all start rows distinct
        assert len(ds.dedup()) == 6


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


