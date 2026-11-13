"""Unit tests for sage.data.noise.lowfar_noise: StartTimeDataset and
_MiningReader on a fake two-detector sampler, on CPU."""

import json
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path

pytest.importorskip("h5py", reason="sage.core.logger requires h5py")
pytest.importorskip("tqdm", reason="lowfar_noise requires tqdm")

from sage.data.noise.lowfar_noise import StartTimeDataset, _MiningReader

_SR = 4096.0
_SEQ_LEN = 64         # short window keeps tests fast
_N_SAMPLES = 512      # 2x largest safe start index + seq_len
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


def _sidecar(gps_start=1_234_567_890.0):
    """A one-segment sidecar record list starting at gps_start."""
    return [{
        "segment_index": 0,
        "gps_start": gps_start,
        "sample_start_idx": 0,
        "sample_rate": _SR,
        "dtype": "float32",
        "endianness": _ENDIAN,
    }]


def _seg_arr():
    """The segment table: one segment of run 0."""
    return np.array([(0, 0, 0, _N_SAMPLES, _N_SAMPLES)], dtype=_SEG_DTYPE)


class _FakeSampler:
    """Minimal MemmapNoiseSampler stand-in for _MiningReader."""

    def __init__(self, bin_files):
        self.bin_files = [Path(p) for p in bin_files]
        self.seq_len = _SEQ_LEN
        self.n_detectors = len(bin_files)
        self.device = "cpu"
        self.postprocess_fn = None
        seg = _seg_arr()
        self.seg_index = [seg] * self.n_detectors
        self.segment_probs = [np.array([1.0])] * self.n_detectors
        # mmaps are nested [detector][run]; this stub has one run.
        self.mmaps = [
            [np.memmap(str(p), dtype=np.dtype(f"{_ENDIAN}f4"), mode="r")]
            for p in self.bin_files
        ]


class _FixedModel(nn.Module):
    """Returns the same ranking statistic for every input."""

    def __init__(self, score: float = 5.0):
        super().__init__()
        self._score = score

    def forward(self, x):
        """A constant score per sample."""
        return (torch.full((x.shape[0], 1), self._score),)



@pytest.fixture
def fake_env(tmp_path):
    """Write a .bin file and segment sidecar per detector."""
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
    """A _FakeSampler over the fake files."""
    return _FakeSampler(fake_env)


@pytest.fixture
def reader(sampler):
    """A seeded _MiningReader on the fake sampler."""
    return _MiningReader(sampler, seed=0)



class TestStartTimeDataset:
    """Tests of StartTimeDataset."""

    def _make(self, n=10):
        """A dataset of n windows with scores 0 to n - 1."""
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
        """save then load returns the same dataset."""
        ds = self._make(8)
        ds.save(str(tmp_path / "ds.npz"))
        ds2 = StartTimeDataset.load(str(tmp_path / "ds.npz"))
        np.testing.assert_array_equal(ds2.start_indices, ds.start_indices)
        np.testing.assert_array_equal(ds2.scores, ds.scores)
        assert ds2.detectors == ds.detectors
        assert ds2.sample_rate == ds.sample_rate
        assert ds2.seq_len == ds.seq_len


    def test_filter_keeps_correct_subset(self):
        """filter keeps windows scoring at or above the threshold."""
        ds = self._make(10)      # scores 0..9
        filtered = ds.filter(5.0)
        assert len(filtered) == 5
        assert (filtered.scores >= 5.0).all()


    def test_merge_detector_mismatch_raises(self):
        """Merging datasets with different detectors is refused."""
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
        """A dataset with the given start rows and scores."""
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
        """dedup drops repeated start rows."""
        # Rows 0 and 2 are the same window; 1 and 3 are distinct.
        ds = self._dup_ds(
            rows=[[10, 20], [30, 40], [10, 20], [50, 60]],
            scores=[1.0, 2.0, 5.0, 3.0],
        )
        out = ds.dedup()
        assert len(out) == 3                              # one duplicate removed
        uniq = {tuple(r) for r in out.start_indices.tolist()}
        assert uniq == {(10, 20), (30, 40), (50, 60)}

    def test_dedup_keeps_highest_score_of_duplicates(self):
        """dedup keeps the highest score of a repeated window."""
        ds = self._dup_ds(
            rows=[[10, 20], [10, 20], [10, 20]],
            scores=[1.0, 9.0, 4.0],
        )
        out = ds.dedup()
        assert len(out) == 1
        assert out.scores[0] == 9.0                       # strongest occurrence kept

    def test_dedup_accumulation_does_not_grow_on_repeat(self):
        """Repeated merge plus dedup of one set keeps its size."""
        ds = self._dup_ds(rows=[[1, 2], [3, 4]], scores=[7.0, 8.0])
        acc = ds
        for _ in range(5):
            acc = acc.merge(ds).dedup()                   # same windows re-found
        assert len(acc) == 2



class TestMiningReader:
    """Tests of _MiningReader."""


    def test_random_starts_in_memmap_bounds(self, reader):
        """Random starts keep each window inside the memmap."""
        starts, segs = reader.random_starts(50)
        for d in range(_D):
            assert np.all(starts[:, d] >= 0)
            assert np.all(starts[:, d] + _SEQ_LEN <= _N_SAMPLES)


    def test_mutate_starts_in_bounds(self, reader):
        """Mutated starts keep each window inside the memmap."""
        starts, segs = reader.random_starts(10)
        new_starts, _ = reader.mutate_starts(starts, segs, sigma_samples=200)
        for d in range(_D):
            assert np.all(new_starts[:, d] >= 0)
            assert np.all(new_starts[:, d] + _SEQ_LEN <= _N_SAMPLES)


