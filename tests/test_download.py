"""sage download without the network: strain on disk adopted, its ASDs, bank and fiducial built; GWOSC's answers
recorded and read back offline; damage or other flags refused; GWOSC unreachable."""

import json

import numpy as np
import pytest
import requests
import yaml

from sage.cli import main
from sage.config import machine
from sage.data import download
from sage.data.download import DATASET, download_run
from sage.data.noise.data_quality import FLAGS


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
