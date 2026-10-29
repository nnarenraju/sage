"""End-to-end tests of sage.factory.train_run and ``sage check`` / ``train`` / ``status`` / ``log`` / ``clone`` / ``validate`` / ``compare`` on synthetic noise (tiny epochs)."""

import json
import os
import shutil

import h5py
import numpy as np
import torch
import yaml

from sage.cli import main
from sage.config import machine
from sage.core.logger import setup_logging
from sage.factory.callbacks import EMACallback
from sage.factory.compare import recovered_fractions
from sage.factory.train_run import train_run
from sage.utils.checkpoint import CheckpointManager

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


def use_machine(tmp_path, monkeypatch):
    """A machine file pointing at a fake data directory; the checkout may hold edits."""
    data_dir = fake_data_dir(tmp_path / "data")
    machine_file = tmp_path / "machine.yaml"
    machine_file.write_text(f"data_dir: {data_dir}\n")
    monkeypatch.setattr(machine, "MACHINE_FILE", machine_file)
    monkeypatch.setattr("sage.config.run_dir.code_differs", lambda commit: False)


def test_train_run(tmp_path, monkeypatch, capsys):
    use_machine(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.yaml").write_text(RUN)
    files = lambda: {path: path.stat().st_mtime_ns for path in run_dir.rglob("*")}
    bin_dir = tmp_path / "bin"  # stand-in sbatch and scontrol
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text('#!/bin/bash\necho "$@" > "$(dirname "$0")/call"\necho 101\n')
    (bin_dir / "scontrol").write_text('#!/bin/bash\necho "JobId=101 StdOut=/jobs/slurm-101.out"\n')
    for stand_in in ("sbatch", "scontrol"):
        (bin_dir / stand_in).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    (tmp_path / "job.sh").write_text("#!/bin/bash\n")
    try:
        assert main(["check", str(run_dir)]) == 0 and list(files()) == [run_dir / "run.yaml"]  # writes nothing
        assert main(["check", str(run_dir), "--in-job", str(tmp_path / "job.sh")]) == 0
        assert (bin_dir / "call").read_text().split() == ["--parsable", "--export=ALL,SAGE_CHECK_ONLY=1", str(tmp_path / "job.sh")]
        monkeypatch.setenv("SAGE_CHECK_ONLY", "1")  # the job's sage train only checks
        assert main(["train", str(run_dir)]) == 0 and list(files()) == [run_dir / "run.yaml"]
        monkeypatch.setenv("SAGE_CHECK_ONLY", "yes")
        assert main(["train", str(run_dir)]) == 1
        monkeypatch.delenv("SAGE_CHECK_ONLY")
        assert main(["train", str(run_dir), "--stop-after", "0s"]) == 0  # stops before epoch 0
        assert not (run_dir / "checkpoints" / "progress.json").exists()
        train_run(run_dir)
        assert main(["train", str(run_dir)]) == 0  # resumes at epoch 1 = epochs: records a segment, trains nothing
        before = files()
        assert main(["check", str(run_dir)]) == 0 and files() == before  # a started run, from its checkpoint
    finally:
        setup_logging(None, console=False)

    resolved = run_dir / "resolved"
    derived = yaml.safe_load((resolved / "derived.yaml").read_text())
    assert set(derived) == {"window", "multirate_bins", "target_scaling", "flat_tail_start_epoch", "seeds"}
    assert [s["start_epoch"] for s in yaml.safe_load((resolved / "versions.yaml").read_text())["segments"]] == [0, 0, 1]
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
    capsys.readouterr()
    assert main(["status", str(run_dir)]) == 0 and main(["log", str(run_dir), "--lines", "1"]) == 0
    out = capsys.readouterr().out
    assert "1 of 1 epochs trained" in out and "Final check, ema_calibrated.pt (epoch 0)" in out

    def group(name, file_name="validate.h5"):
        with h5py.File(results / file_name, "r") as file:
            return {key: file[name][key][:] for key in file[name]}

    try:
        assert main(["validate", str(run_dir)]) == 0
        assert "Validated best.pt (epoch 0) on 16 validation samples" in capsys.readouterr().out
        first = group("validate_best")
        final = group("final_best", "validation.h5")
        assert all(np.array_equal(first[key], final[key], equal_nan=True) for key in final)  # the final check's pass
        assert main(["validate", str(run_dir)]) == 0  # the same checkpoint, the same result
        assert all(np.array_equal(first[key], value, equal_nan=True) for key, value in group("validate_best").items())
        assert main(["validate", str(run_dir), "--ema-calibrated"]) == 0
        assert np.array_equal(group("validate_ema_calibrated")["signal_idx"], first["signal_idx"])
        assert main(["validate", str(run_dir), "--from-epoch", "3"]) == 1
        assert "stored epochs: 0\nfix: sage validate" in capsys.readouterr().err
    finally:
        setup_logging(None, console=False)


def same(a, b) -> bool:
    """Whether two checkpoint entries are identical, tensors and arrays bit for bit."""
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def test_clone(tmp_path, monkeypatch, capsys):
    use_machine(tmp_path, monkeypatch)
    old = tmp_path / "old"
    old.mkdir()
    (old / "run.yaml").write_text(RUN.replace("epochs: 1", "epochs: 2").replace("keep_checkpoints: 1", "keep_checkpoints: 2"))
    save = CheckpointManager.save

    def save_then_stop(self, epoch, validation_loss):
        save(self, epoch, validation_loss)
        (old / "STOP").touch()

    monkeypatch.setattr(CheckpointManager, "save", save_then_stop)
    try:
        train_run(old)  # epoch 0 of 2
    finally:
        setup_logging(None, console=False)
    monkeypatch.setattr(CheckpointManager, "save", save)
    (old / "STOP").unlink()

    new, weights = tmp_path / "new", tmp_path / "weights"
    assert main(["clone", str(old), str(new), "--full-state"]) == 0
    assert sorted(str(p.relative_to(new)) for p in new.rglob("*") if p.is_file()) == [
        "clone/best.pt", "clone/start.pt", "resolved/clone.yaml", "results/losses.h5", "results/validation.h5", "run.yaml",
    ]
    assert yaml.safe_load((new / "resolved" / "clone.yaml").read_text())["epoch"] == 0
    with h5py.File(new / "results" / "losses.h5", "r") as file:
        assert file["training"]["loss"][0][0] > 0 and not file["training"]["loss"][1].any()
    assert main(["clone", str(old), str(weights), "--weights-only"]) == 0
    assert "the same as that checkpoint's" in capsys.readouterr().out

    initial = []

    class Spy(EMACallback):  # built from the network's weights as training starts
        def __init__(self, model, decay):
            initial.append({key: value.clone() for key, value in model.state_dict().items()})
            super().__init__(model, decay)

    monkeypatch.setattr("sage.factory.train_run.EMACallback", Spy)
    try:
        train_run(weights, stop_after=0.0)
    finally:
        setup_logging(None, console=False)
    monkeypatch.setattr("sage.factory.train_run.EMACallback", EMACallback)
    assert same(initial[0], torch.load(weights / "clone" / "start.pt", weights_only=False)["model"])

    try:
        train_run(old)  # resumes at epoch 1
        train_run(new)  # the full-state clone of epoch 0 trains epoch 1
    finally:
        setup_logging(None, console=False)
    resumed, cloned = (torch.load(run / "checkpoints" / "latest.pt", weights_only=False) for run in (old, new))
    for key in ("epoch", "best_validation_loss", "model", "optimiser", "schedule", "ema", "rng"):
        assert same(resumed[key], cloned[key]), key
    for run in (old, new):
        with h5py.File(run / "results" / "losses.h5", "r") as file:
            assert (file["validation"]["loss"][:] != 0).all()
    final = [json.loads((run / "results" / "final_check.json").read_text()) for run in (old, new)]
    assert final[0] == final[1]

    last = tmp_path / "last"
    assert main(["clone", str(old), str(last), "--full-state"]) == 0  # the last epoch: only the final check is left
    try:
        train_run(last)
    finally:
        setup_logging(None, console=False)
    assert json.loads((last / "results" / "final_check.json").read_text()) == final[0]
    capsys.readouterr()
    assert main(["status", str(new)]) == 0 and f"Cloned from {old}, epoch 0 (full state)" in capsys.readouterr().out

    assert main(["clone", str(old), str(new), "--full-state"]) == 1  # not empty
    assert main(["clone", str(old), str(tmp_path / "other"), "--weights-only", "--from-epoch", "5"]) == 1
    (old / "run.yaml").write_text((old / "run.yaml").read_text().replace("frontend_filters: 16", "frontend_filters: 8"))
    assert main(["clone", str(old), str(tmp_path / "other"), "--weights-only"]) == 1
    assert main(["train", str(old)]) == 1
    err = capsys.readouterr().err
    assert "stored epochs: 0, 1" in err and "model.frontend_filters: 16 -> 8" in err and not (tmp_path / "other").exists()
    assert f"fix: sage clone {old} <new directory> --full-state" in err


def test_recovered_fractions():
    background, foreground = np.arange(1000.0), np.array([998.5, 999.5, 990.0, 1.0])
    rows = recovered_fractions(background, foreground, (1e-3, 2e-3, 1e-4))
    assert rows == [(1e-3, 1, 999.0, 0.25), (2e-3, 2, 998.0, 0.5), (1e-4, 0, 999.0, 0.25)]  # count 0: the loudest


def test_compare(tmp_path, monkeypatch, capsys):
    use_machine(tmp_path, monkeypatch)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.yaml").write_text(RUN.replace("epochs: 1", "epochs: 2"))
    try:
        train_run(run)  # two validation passes in one segment
        twin = tmp_path / "twin"
        shutil.copytree(run, twin)
        windows = ["--noise-windows", "8", "--signal-windows", "8"]
        assert main(["compare", str(run), str(twin), "--output", str(tmp_path / "both"), *windows]) == 0
        assert "Recovered fraction at fixed noise false-alarm probability per window" in capsys.readouterr().out
        assert main(["compare", str(run), "--output", str(tmp_path / "alone"), *windows]) == 0
        assert main(["compare", str(run), "--output", str(tmp_path / "alone"), *windows]) == 1  # never overwritten
        assert main(["compare", str(run), str(run), "--output", str(tmp_path / "same"), *windows]) == 1
        assert "these repeat: run, run" in capsys.readouterr().err
    finally:
        setup_logging(None, console=False)

    with h5py.File(tmp_path / "both" / "compare.h5", "r") as both, h5py.File(tmp_path / "alone" / "compare.h5", "r") as alone:
        assert both.attrs["complete"] and list(both.attrs["runs"]) == ["run", "twin"]
        assert both["foreground/run"].shape == (8, 5) and both["background/run"].shape == (8, 5)
        assert list(both["foreground/run"].attrs["columns"]) == ["rank", "mu_tc", "mu_mchirp", "sigma_tc", "sigma_mchirp"]
        assert both["foreground/target_snr"].shape == (8, 1) and both["foreground/signal_params"].attrs["names"][0]
        for group in ("foreground", "background"):
            assert np.array_equal(both[group]["run"][:], both[group]["twin"][:])  # one network, the same windows
            assert np.array_equal(both[group]["run"][:], alone[group]["run"][:])  # windows: first run and seed only
    assert (tmp_path / "both" / "report.txt").read_text().startswith("background windows 8, foreground signals 8")
    with h5py.File(run / "results" / "validation.h5", "r") as file:  # every pass on the same windows
        passes = [file[group] for group in ("epoch_0000", "epoch_0001", "final_best")]
        for key in ("signal_params", "signal_idx"):
            first = passes[0][key][:]
            assert all(np.array_equal(first, group[key][:len(first)], equal_nan=True) for group in passes[1:])
