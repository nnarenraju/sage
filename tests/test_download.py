"""sage download without the network: strain on disk adopted, its ASDs, bank and fiducial built; GWOSC's answers
recorded and read back offline; damage or other flags refused; GWOSC unreachable."""

import json
from functools import partial

import numpy as np
import pytest
import requests
import torch
import yaml
from gwpy.timeseries import TimeSeries

from sage.cli import main
from sage.config import machine
from sage.data import download
from sage.data.download import DATASET, download_run
from sage.data.noise.data_quality import FLAGS, read_bad_time
from sage.data.noise.real_noise import MemmapSingleNoiseSampler
from sage.data.asd.smoothing import LogSplineSmoothing
from sage.data.primer import DataReleaseDownloader, EstimateASD
from sage.dsp.welch import TorchWelch


def write_strain(data_dir, lengths):
    path = machine.strain_file(data_dir, "o3b", "H1")
    path.parent.mkdir(parents=True)
    records, offset = [], 0
    for index, n in enumerate(lengths):
        records.append({
            "segment_index": index, "detector": "H1", "observing_run": "O3b", "gps_start": 1e9 + 600.0 * index,
            "gps_end": 1e9 + 600.0 * index + n / 2048.0, "sample_rate": 2048.0, "nsamples": n, "dtype": "float32",
            "endianness": "<", "sample_start_idx": offset // 4, "byte_offset": offset, "byte_length": 4 * n,
            "dyn_range_fac": DATASET["scale"],
            "noise_low_freq_cutoff": 15.0,
        })
        offset += 4 * n
    path.write_bytes(np.random.default_rng(0).standard_normal(offset // 4).astype("<f4").tobytes())
    machine.segments_file(path).write_text(json.dumps(records))
    machine.failed_file(path).write_text("[]")
    machine.bad_time_file(path).write_text(json.dumps({"flags": list(FLAGS), "intervals": [[1e9, 1e9 + 1.0]]}))
    return path


def test_zero_runs(tmp_path, monkeypatch):
    path = write_strain(tmp_path, [40960, 45056])
    machine.bad_time_file(path).write_text(json.dumps({"flags": list(FLAGS), "intervals": []}))
    samples = np.memmap(path, dtype="<f4", mode="r+")
    samples[40960 + 100:40960 + 110] = 0  # missing data in the second segment
    samples.flush()
    download_run(tmp_path, "o3b", ["H1"], 2, False, False)
    start = 1e9 + 600.0
    assert json.loads(machine.zero_runs_file(path).read_text())["intervals"] == [
        [start + 100 / 2048, start + 110 / 2048]
    ]
    assert read_bad_time(path) == [(start + 100 / 2048, start + 110 / 2048)]
    sampler = MemmapSingleNoiseSampler(path, read_bad_time(path))
    assert list(sampler.segments["start"]) == [0]  # the bank never draws the second segment
    monkeypatch.setitem(download.ASDS, "bank_size", 1)
    with pytest.raises(ValueError, match="zero or non-finite samples"):  # before the ASD is estimated
        download.estimator("H1", tmp_path / "asds").write_raw_asds(
            noise_sampler=lambda n: torch.zeros(n), duration=32768
        )


def test_no_interpolation(tmp_path):
    """EstimateASD on Welch's own grid (interpolate_asd=False): the per-segment ASDs and a bank."""
    path = write_strain(tmp_path, [40960, 45056])
    asd_dir = tmp_path / "asds"
    estimator = EstimateASD(
        detector="H1", sample_rate=2048.0, data_dir=asd_dir, fiducial_dir=asd_dir, num_samples=2,
        asd_method=TorchWelch(delta_t=1 / 2048, seg_len=8192, seg_stride=4096, avg_method="median"),
        store_asds_as_bin=True, low_frequency_cutoff=15.0, max_filter_len=4096, interpolate_asd=False,
        asd_smoothener=LogSplineSmoothing(smooth_factor=None, noise_low_frequency_cutoff=15.0),
    )
    estimator.estimate_segment_asds(noise_segments_file=path)
    sampler = MemmapSingleNoiseSampler(path, [], return_tensor=True)
    estimator.write_raw_asds(noise_sampler=partial(sampler.sample, rng=np.random.default_rng(0)), duration=32768)
    for asds in (machine.segment_asd_file(asd_dir, "H1"), machine.recolour_bank_file(asd_dir, "H1")):
        values = np.fromfile(asds, dtype="<f4").reshape(2, -1)
        assert values.shape[1] == 8192 // 2 + 1 and np.isfinite(values).all()


def test_one_worker(tmp_path, monkeypatch):
    """The single-worker download path, offline: GWOSC's strain replaced by white noise."""
    noise = lambda det, start, end, cache: TimeSeries(
        np.random.default_rng(0).standard_normal(int((end - start) * 4096)), sample_rate=4096, t0=start
    )
    monkeypatch.setattr(TimeSeries, "fetch_open_data", staticmethod(noise))
    pieces = [(1e9, 1e9 + 32.0), (1e9 + 100.0, 1e9 + 132.0)]
    DataReleaseDownloader(
        segments_metadata=[{"segments": pieces, "detector": "H1", "observing_run": "O3b"}],
        save_parent_dir=str(tmp_path), num_workers=1, proxy_reset_every=1, proxy_reset_sleep=0.0, save_bin=True,
        release_dirname="training.part",
    ).download()
    records = json.loads((tmp_path / "training.part" / "data_H1_O3b_segments.json").read_text())
    assert [record["gps_start"] for record in records] == [1e9 + 0.2, 1e9 + 100.2]


def unreachable(*args, **kwargs):
    raise requests.exceptions.ConnectionError("proxy 503")


def test_adopt_strain(tmp_path, monkeypatch):
    monkeypatch.setitem(download.ASDS, "bank_size", 3)
    path = write_strain(tmp_path, [40960, 45056])
    assert download_run(tmp_path, "o3b", ["H1"], 2, True, True)["H1"]["pieces"] == 2
    assert yaml.safe_load(machine.dataset_file(tmp_path, "o3b").read_text())["sample_rate_in_hz"] == 2048.0
    record = yaml.safe_load(machine.asds_file(tmp_path, "o3b").read_text())
    assert record["detectors"]["H1"] == {"segment_asds": 2, "bank_seed": download.BANK_SEED}
    monkeypatch.setattr(download, "gwosc_run", str.upper)
    monkeypatch.setattr(download, "find_datasets", lambda detector, type: ["O3B"] if detector == "H1" else [])
    assert machine.fiducial_file(download.download_fiducial(tmp_path, ["o3b"], ["H1"]), "H1").exists()
    gwosc = yaml.safe_load(machine.dataset_file(tmp_path, "o3b").read_text())["gwosc"]
    assert gwosc == {"name": "O3B", "detectors": ["H1"]}
    monkeypatch.setattr(download, "gwosc_run", unreachable)
    monkeypatch.setattr(download, "find_datasets", unreachable)
    download.download_fiducial(tmp_path, ["o3b"], ["H1"])  # checked from the record, offline
    machine.bad_time_file(path).write_text(json.dumps({"flags": ["CBC_CAT2"], "intervals": []}))
    with pytest.raises(ValueError, match="flags"):
        download_run(tmp_path, "o3b", ["H1"], 2, False, False)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="bytes"):
        download_run(tmp_path, "o3b", ["H1"], 2, False, False)


def test_unreachable(tmp_path, monkeypatch, capsys):
    machine_file = tmp_path / "machine.yaml"
    machine_file.write_text(f"data_dir: {tmp_path}\n")
    monkeypatch.setattr(machine, "MACHINE_FILE", machine_file)
    monkeypatch.setattr(download, "find_datasets", unreachable)
    assert main(["download", "o3b"]) == 1
    err = capsys.readouterr().err
    assert "GWOSC could not be reached from here: proxy 503" in err and "sage download o3b" in err
