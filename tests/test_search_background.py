"""Tests for the roll background: planning rolls and scoring a file."""

import json

import h5py
import numpy as np
import torch
from pycbc.conversions import sec_to_year

from sage.core.pipeline import Grid, GWBatch, ProcessingState
from sage.factory.build import build_model
from sage.search.dataprep.windows import analysis_windows, load_segments
from sage.search.model.score import network_outputs
from sage.search.timeslides.roll import plan_rolls, score_roll_file

RATE = 16.0
WINDOW = 1024
STRIDE = 256


def test_plan_rolls():
    """Live time counts pairs clear in both; files close at a year."""
    clear = {"H1": np.ones(8, dtype=bool), "L1": np.ones(8, dtype=bool)}
    clear["H1"][0] = clear["L1"][3] = False
    rate = 1 / (sec_to_year(1.0) ** -1 / 6)  # one window is 1/6 year
    files = plan_rolls(clear, ["H1", "L1"], 2, 2 / rate, 2.5, 1.0, 1, rate)
    assert [plan["shifts"] for plan in files] == [[[2]], [[4]], [[6]]]
    years = [sec_to_year(plan["live_time_s"][0]) for plan in files]
    assert np.allclose(years, [1.0, 1.0, 1.0])  # 6 of 8 pairs clear in each roll


class InverseFFT:
    """Processor stand-in: back to the time domain."""

    input_state = ProcessingState(Grid.FD_UNIFORM)

    def __call__(self, batch: GWBatch) -> GWBatch:
        """Inverse rfft of each window."""
        data = torch.fft.irfft(batch.data, n=WINDOW, norm="forward").float()
        return GWBatch(data, state=ProcessingState(Grid.TD_UNIFORM))


def tiny_network() -> dict:
    """Small separable H1-L1 network with a non-astrophysical logit."""
    torch.manual_seed(0)
    settings = {"name": "MSCNN1Datt_2DResNetCBAM", "backend_resnet_size": "18",
                "frontend_filters": "16"}
    model = build_model(settings, 2, 2, True, "cpu", torch.float32).eval()
    columns = ["ranking_statistic", "non_astrophysical_logit", "a_raw_mean",
               "b_raw_mean", "a_raw_width", "b_raw_width"]
    return {"model": model, "processor": InverseFFT(), "detectors": ["H1", "L1"],
            "window_samples": WINDOW, "columns": columns}


def write_release(release_dir):
    """Two segments per detector, eight windows each."""
    rng = np.random.default_rng(1)
    n = WINDOW + 7 * STRIDE
    for detector in ["H1", "L1"]:
        records = []
        with h5py.File(release_dir / f"data_{detector}_O3a.h5", "w") as handle:
            for index, start in enumerate([0.0, 1000.0]):
                handle[f"segments/{index:06d}"] = rng.standard_normal(n).astype("f4")
                records.append({"segment_index": index, "gps_start": start,
                                "gps_end": start + n / RATE, "sample_rate": RATE,
                                "nsamples": n})
        sidecar = release_dir / f"data_{detector}_O3a_segments.json"
        sidecar.write_text(json.dumps(records))


def test_roll_file(tmp_path):
    """Scores match the full network; a stopped file resumes exactly."""
    write_release(tmp_path)
    network = tiny_network()
    segments = {detector: load_segments(tmp_path, "O3a", detector) for detector in ["H1", "L1"]}
    windows = analysis_windows(segments, WINDOW, STRIDE)
    n_windows = len(windows["gps"])
    shifts = [[3], [6]]

    def run():
        """Score the roll file, resuming from its progress record."""
        score_roll_file(network, tmp_path, "O3a", windows, tmp_path / "background",
                        "year_01", shifts, STRIDE, 5, "cpu", {"shifts": shifts})

    run()
    rows = np.fromfile(tmp_path / "background" / "year_01.bin", dtype="<f4")
    rows = rows.reshape(len(shifts), n_windows, len(network["columns"]))

    with h5py.File(tmp_path / "data_H1_O3a.h5") as h1, h5py.File(tmp_path / "data_L1_O3a.h5") as l1:
        def window(handle, detector, index):
            """Strain of one detector in one analysis window."""
            segment = windows[f"{detector}_segment"][index]
            sample = windows[f"{detector}_sample"][index]
            return handle[f"segments/{segment:06d}"][sample : sample + WINDOW]

        for roll, (shift,) in enumerate(shifts):
            strain = np.stack([
                np.stack([window(h1, "H1", i), window(l1, "L1", (i + shift) % n_windows)])
                for i in range(n_windows)
            ]) / 5.902958103587057e20
            spectra = torch.fft.rfft(torch.from_numpy(strain.astype("f4")), dim=-1, norm="forward")
            expected = network_outputs(network, spectra)
            assert np.allclose(rows[roll], expected, rtol=1e-5, atol=1e-6)

    scores = (tmp_path / "background" / "year_01.bin").read_bytes()
    progress = tmp_path / "background" / "year_01.json"
    recorded = json.loads(progress.read_text())
    recorded["done"] = 5
    progress.write_text(json.dumps(recorded))
    with open(tmp_path / "background" / "year_01.bin", "r+b") as handle:
        handle.seek(5 * len(network["columns"]) * 4)
        handle.write(b"\0" * 64)
    run()
    assert (tmp_path / "background" / "year_01.bin").read_bytes() == scores
