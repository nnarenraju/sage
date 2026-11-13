"""Zero-lag scoring: output columns, resume after a stop, fixed settings."""

import json

import h5py
import numpy as np
import pytest
import torch

from sage.config.run_dir import SettingsChanged
from sage.core.pipeline import Grid, GWBatch, ProcessingState
from sage.search.dataprep.windows import analysis_windows, load_segments
from sage.search.model.score import network_outputs
from sage.search.zerolag.zerolag import score_zerolag

RATE = 16.0
COLUMNS = ["ranking_statistic", "non_astrophysical_logit"] + [
    f"{target}_raw_{part}" for part in ("mean", "width") for target in ("a", "b")
]


class Identity:
    """Processor stand-in: frequency-domain input passed through."""

    input_state = ProcessingState(Grid.FD_UNIFORM)

    def __call__(self, batch: GWBatch) -> GWBatch:
        """Return the batch unchanged."""
        return batch


def model(x: torch.Tensor) -> tuple:
    """Network stand-in with distinct, input-dependent outputs."""
    statistic = x.sum(dim=(1, 2))[:, None]
    estimates = x[:, 0, :4]
    logit = x[:, 1, :1]
    return statistic, estimates, logit


NETWORK = {
    "model": model,
    "processor": Identity(),
    "detectors": ["H1", "L1"],
    "window_samples": 16,
    "columns": COLUMNS,
}


def write_release(release_dir):
    """Two segments per detector, 10 s and 5 s, at 16 Hz."""
    rng = np.random.default_rng(1)
    for detector in NETWORK["detectors"]:
        records = []
        with h5py.File(release_dir / f"data_{detector}_O3a.h5", "w") as handle:
            for index, (start, end) in enumerate([(0.0, 10.0), (20.0, 25.0)]):
                n = int((end - start) * RATE)
                handle[f"segments/{index:06d}"] = rng.standard_normal(n).astype("f4")
                records.append({"segment_index": index, "gps_start": start,
                                "gps_end": end, "sample_rate": RATE, "nsamples": n})
        sidecar = release_dir / f"data_{detector}_O3a_segments.json"
        sidecar.write_text(json.dumps(records))


def test_network_outputs():
    """Columns: statistic, non-astrophysical logit, then estimates."""
    spectra = torch.randn(3, 2, 9, dtype=torch.complex64)
    outputs = network_outputs(NETWORK, spectra)
    statistic, estimates, logit = model(GWBatch(spectra, Identity.input_state)
                                        .to_network_input())
    assert np.array_equal(outputs[:, 0], statistic[:, 0].numpy())
    assert np.array_equal(outputs[:, 1], logit[:, 0].numpy())
    assert np.array_equal(outputs[:, 2:], estimates.numpy())


def test_resume(tmp_path):
    """A stopped pass resumes to the same scores; settings are fixed."""
    write_release(tmp_path)
    directory = tmp_path / "zerolag"
    segments = {
        detector: load_segments(tmp_path, "O3a", detector)
        for detector in NETWORK["detectors"]
    }
    windows = analysis_windows(segments, NETWORK["window_samples"], 4)

    def run(attrs):
        score_zerolag(NETWORK, tmp_path, "O3a", windows, directory, 4, 5, "cpu", attrs)

    run({"seed": 1})
    scores = (directory / "scores.bin").read_bytes()
    assert len(scores) == 54 * len(COLUMNS) * 4  # 37 + 17 windows

    progress = json.loads((directory / "scores.json").read_text())
    progress["done"] = 10
    (directory / "scores.json").write_text(json.dumps(progress))
    with open(directory / "scores.bin", "r+b") as handle:
        handle.seek(10 * len(COLUMNS) * 4)
        handle.write(b"part of a batch")
    run({"seed": 1})
    assert (directory / "scores.bin").read_bytes() == scores

    with pytest.raises(SettingsChanged, match="seed"):
        run({"seed": 2})
