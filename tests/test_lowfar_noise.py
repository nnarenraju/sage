"""Unit tests for sage.data.noise.lowfar_noise, on CPU.

A FakeNoiseSampler and FixedScoreModel exercise StartTimeDataset and
_MiningReader end to end with tiny parameters.
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

import sage.data.noise.lowfar_noise as _lnm
from sage.data.noise.lowfar_noise import StartTimeDataset, _MiningReader

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
    ("run", np.int64),
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
    # (run, idx, start, end, nsamples) -- single run 0
    return np.array([(0, 0, 0, _N_SAMPLES, _N_SAMPLES)], dtype=_SEG_DTYPE)


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
        # mmaps are nested [detector][run] (5a); this stub is single-run.
        self.mmaps = [
            [np.memmap(str(p), dtype=np.dtype(f"{_ENDIAN}f4"), mode="r")]
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


    def test_save_load_roundtrip(self, tmp_path):
        ds = self._make(8)
        ds.save(str(tmp_path / "ds.npz"))
        ds2 = StartTimeDataset.load(str(tmp_path / "ds.npz"))
        np.testing.assert_array_equal(ds2.start_indices, ds.start_indices)
        np.testing.assert_array_equal(ds2.scores, ds.scores)
        assert ds2.detectors == ds.detectors
        assert ds2.sample_rate == ds.sample_rate
        assert ds2.seq_len == ds.seq_len


    def test_filter_keeps_correct_subset(self):
        ds = self._make(10)      # scores 0..9
        filtered = ds.filter(5.0)
        assert len(filtered) == 5
        assert (filtered.scores >= 5.0).all()


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


# ---------------------------------------------------------------------------
# TestMiningReader
# ---------------------------------------------------------------------------


class TestMiningReader:


    def test_random_starts_in_memmap_bounds(self, reader):
        starts, segs = reader.random_starts(50)
        for d in range(_D):
            assert np.all(starts[:, d] >= 0)
            assert np.all(starts[:, d] + _SEQ_LEN <= _N_SAMPLES)


    def test_mutate_starts_in_bounds(self, reader):
        starts, segs = reader.random_starts(10)
        new_starts, _ = reader.mutate_starts(starts, segs, sigma_samples=200)
        for d in range(_D):
            assert np.all(new_starts[:, d] >= 0)
            assert np.all(new_starts[:, d] + _SEQ_LEN <= _N_SAMPLES)


