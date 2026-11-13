"""Tests for the MLGWSC-1 benchmark search: slicing and clustering."""

import h5py
import numpy as np
import pytest

from sage.benchmark.mlgwsc1.mlgwsc1 import (
    CLUSTER_IN_S,
    TIME_TOLERANCE_IN_S,
    Slicer,
    clusters,
)
from sage.core.geometry import window_from_data

WINDOW = window_from_data(
    {"sample_rate_in_hz": 2048, "sample_length_in_s": 1, "padding_length_in_s": 0.5}
)


def data_file(path, detectors=("H1", "L1")):
    """Two segments in MLGWSC-1's layout; sample i holds the value i."""
    with h5py.File(path, "w") as file:
        file.attrs["detectors"] = list(detectors)
        for detector in detectors:
            for start, samples in [(1000, 5000), (2000, 4400)]:
                strain = np.arange(samples, dtype=float)
                ds = file.create_dataset(f"{detector}/{start}", data=strain)
                ds.attrs["start_time"], ds.attrs["delta_t"] = float(start), 1 / 2048
    return path


def test_slicer_windows(tmp_path):
    """Windows step 0.1 s per segment; other detectors are refused."""
    slicer = Slicer(data_file(tmp_path / "data.hdf"), ["H1", "L1"], WINDOW, 1.3)
    step = 204  # int(0.1 s * 2048 Hz)
    assert len(slicer) == (5000 - 4096) // step + (4400 - 4096) // step == 5
    strain, time = slicer[4]  # the second segment's only window
    assert time.item() == 2000 + 1.3
    assert strain.shape == (2, 4096) and strain[1, -1].item() == 4095
    strain, time = slicer[3]
    assert time.item() == 1000 + 3 * step / 2048 + 1.3
    assert strain[0, 0].item() == 3 * step
    with pytest.raises(ValueError, match="detectors"):
        Slicer(data_file(tmp_path / "hv.hdf", ("H1", "V1")), ["H1", "L1"], WINDOW, 0)


def test_clusters():
    """Triggers within CLUSTER_IN_S keep only the loudest."""
    apart = 0.5 + CLUSTER_IN_S + 0.01
    found = np.array([[0.0, 1.0], [0.2, 3.0], [0.5, 2.0], [apart, 5.0]])
    time, stat, var = clusters(found)
    assert time.tolist() == [0.2, found[3, 0]] and stat.tolist() == [3.0, 5.0]
    assert var.tolist() == [TIME_TOLERANCE_IN_S] * 2
