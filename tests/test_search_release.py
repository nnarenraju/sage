"""Zero spans of the search strain release."""

import json

import h5py
import numpy as np

from sage.config.machine import zero_runs_file
from sage.search.dataprep.release import write_zero_runs, zero_spans


def test_zero_spans():
    """Runs of exact zeros become GPS spans [start, end)."""
    samples = np.array([1.0, 0.0, 0.0, 2.0, 0.0])
    assert zero_spans(samples, 10.0, 2.0) == [(10.5, 11.5), (12.0, 12.5)]


def test_write_zero_runs(tmp_path):
    """Recorded spans are kept; a record without them is scanned."""
    path = tmp_path / "data_H1_O3a.h5"
    with h5py.File(path, "w") as handle:
        handle["segments/000000"] = np.ones(4, dtype=np.float32)
        handle["segments/000001"] = np.array([1, 0, 0, 1], dtype=np.float32)
    records = [
        {
            "dataset": "segments/000000",
            "gps_start": 0.0,
            "sample_rate": 2.0,
            "zero_spans": [[0.5, 1.0]],
        },
        {"dataset": "segments/000001", "gps_start": 10.0, "sample_rate": 2.0},
    ]
    write_zero_runs(path, records)
    saved = json.loads(zero_runs_file(path).read_text())
    assert saved == {"intervals": [[0.5, 1.0], [10.5, 11.5]]}
