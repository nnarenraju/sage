"""End-to-end test of sage.factory.train_run on synthetic noise (one tiny epoch)."""

import json

import h5py
import numpy as np
import yaml

from sage.config import machine
from sage.core.logger import setup_logging
from sage.factory.train_run import train_run

RUN = """
data:
  detectors: [H1, L1]
  sample_rate_in_hz: 2048
  sample_length_in_s: 12
  padding_length_in_s: 2
  dtype: float32
signal:
  prior: bbh_narrow
  low_frequency_cutoff_in_hz: 20
  reference_frequency_in_hz: 20
  waveform: {name: IMRPhenomPv2}
  projection: {name: constant}
  snr: {name: halfnorm, scale: 4.0, loc: 5.0}
  targets: [tc, mchirp]
noise:
  training: &synthetic
    name: synthetic
    asd: {H1: aLIGODesignSensitivityP1200087, L1: aLIGODesignSensitivityP1200087}
    low_frequency_cutoff_in_hz: 15
    filter_duration_in_s: 4
  validation: *synthetic
  fiducial: [o3a]
  prefetch_batches: 1
  read_workers: 1
preprocess:
  - name: whiten
  - name: multirate
    lowest_sample_rate_in_hz: 64
    nyquist_margin_in_hz: 20
    min_bin_length_in_s: 0.05
    pre_merger_leeway_in_s: 0
    post_merger_leeway_in_s: 0
model: {name: MSCNN1Datt_2DResNetCBAM, backend_resnet_size: 18, frontend_filters: 16}
loss: {name: BCE_PE_Sigma}
optimiser: {name: AdamW}
schedule: {name: WarmupPolynomialDecayFlatTail, warmup_steps: 1, flat_tail_epochs: 0}
train:
  epochs: 1
  samples_per_epoch: 4
  batch_size: 4
  signal_fraction: 0.5
  device: cpu
  precision: fp32
  clip_norm: 100.0
  compile: false
  seed: 1
  keep_checkpoints: 1
validation:
  samples: 4
  every_n_epochs: 1
  final_samples: 16
  seed: 2
  snr: {name: halfnorm, scale: 4.0, loc: 5.0}
mining: {enabled: false}
ema: {enabled: true, calibration_batches: 2}
"""


def fake_data_dir(root):
    """A data directory holding a flat fiducial for H1 and L1 on the run's grid."""
    fiducial = root / "o3a" / "fiducial"
    fiducial.mkdir(parents=True)
    bins = 16 * 2048 // 2 + 1
    for detector in ("H1", "L1"):
        np.full(bins, 1e-23, dtype=np.float32).tofile(fiducial / f"fiducial_{detector}_psd.bin")
        meta = {"dtype": "float32", "byte_order": "little", "sample_rate": 2048.0,
                "delta_f": 1 / 16, "num_freq_bins": bins, "detector": detector}
        (fiducial / f"fiducial_{detector}_psd.json").write_text(json.dumps(meta))
    return root


def test_train_run(tmp_path, monkeypatch):
    data_dir = fake_data_dir(tmp_path / "data")
    machine_file = tmp_path / "machine.yaml"
    machine_file.write_text(f"data_dir: {data_dir}\n")
    monkeypatch.setattr(machine, "MACHINE_FILE", machine_file)
    monkeypatch.setattr("sage.config.run_dir.code_differs", lambda commit: False)  # checkout may hold edits
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.yaml").write_text(RUN)
    try:
        train_run(run_dir)
        train_run(run_dir)  # resumes at epoch 1 = epochs: records a segment, trains nothing
    finally:
        setup_logging(None, console=False)

    resolved = run_dir / "resolved"
    derived = yaml.safe_load((resolved / "derived.yaml").read_text())
    assert set(derived) == {"window", "multirate_bins", "target_scaling", "flat_tail_start_epoch", "seeds"}
    assert [s["start_epoch"] for s in yaml.safe_load((resolved / "versions.yaml").read_text())["segments"]] == [0, 1]
    assert sorted(p.name for p in (resolved / "fiducial").iterdir()) == [
        "fiducial_H1_psd.bin", "fiducial_H1_psd.json", "fiducial_L1_psd.bin", "fiducial_L1_psd.json",
    ]
    assert sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt")) == [
        "best.pt", "ema.pt", "ema_calibrated.pt", "epoch_0000.pt", "latest.pt",
    ]
    results = run_dir / "results"
    with h5py.File(results / "losses.h5", "r") as file:
        for split in ("training", "validation"):
            assert np.isfinite(file[split]["loss"][0]).all() and file[split]["loss"][0][0] > 0
    with h5py.File(results / "validation.h5", "r") as file:
        assert sorted(file) == ["epoch_0000", "final_best", "final_ema"]
        for name in ("signal_params", "signal_idx", "network_target"):  # one final set for both
            assert np.array_equal(file["final_best"][name][:], file["final_ema"][name][:], equal_nan=True)
    final = json.loads((results / "final_check.json").read_text())
    assert final["samples"] == 16 and final["best.pt"]["epoch"] == 0 and set(final["best.pt"]["loss"]) >= {"total"}
    assert final["ema_calibrated.pt"]["batchnorm_layers_recalibrated"] > 0
    assert final["lower_total_loss"] in ("best.pt", "ema_calibrated.pt")
