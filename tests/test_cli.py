"""Tests for the sage command line: journey, errors, machine commands
and executables."""

import re
import shutil
from pathlib import Path

from sage.cli import main
from sage.cli.registry import COMMANDS
from sage.cli.training import epoch_seconds
from sage.config import loader, machine
from sage.config.prior import load_prior
from sage.config.run_dir import find_run_dir
from sage.factory.build import resolve_settings

RUN = """
data: {detectors: [H1, L1], sample_rate_in_hz: 2048, sample_length_in_s: 12, padding_length_in_s: 2, dtype: float32}
signal: {prior: bbh_narrow, low_frequency_cutoff_in_hz: 20, reference_frequency_in_hz: 20, waveform: {name: IMRPhenomPv2},
         projection: {name: constant}, snr: {name: halfnorm, scale: 4.0, loc: 5.0}, targets: [delta_tc, mchirp]}
noise: {training: {name: real, runs: [o3b]}, validation: {name: real, runs: [o3a]}, fiducial: [o3a, o3b],
        prefetch_batches: 8, read_workers: 16}
preprocess: [{name: whiten}]
model: {name: MSCNN1Datt_2DResNetCBAM}
loss: {name: BCE_PE_Sigma}
optimiser: {name: AdamW}
schedule: {name: WarmupPolynomialDecayFlatTail}
train: {epochs: 4, samples_per_epoch: 64, batch_size: 32, signal_fraction: 0.5, device: cpu, precision: fp32,
        clip_norm: 100.0, compile: false, seed: 1, keep_checkpoints: 1}
validation: {samples: 64, every_n_epochs: 1, final_samples: 64, seed: 2, snr: {name: halfnorm, scale: 4.0, loc: 5.0}}
mining: {enabled: false}
ema: {enabled: true}
"""


def test_journey(capsys, tmp_path, monkeypatch):
    """With no command, sage prints the journey, starting at sage setup."""
    monkeypatch.setattr(machine, "MACHINE_FILE", tmp_path / "machine.yaml")
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "1. sage setup" in out and out.rstrip().endswith("sage <command> --help")


def test_unknown_command(capsys):
    """An unknown command is refused with the closest known one."""
    assert main(["machin"]) == 1
    assert capsys.readouterr().err == "error: unknown command 'machin'; did you mean 'machine'?\nfix: sage machine\n"


def test_machine_file(capsys, tmp_path, monkeypatch):
    """sage setup writes the machine file and validates data_dir."""
    monkeypatch.setattr(machine, "MACHINE_FILE", tmp_path / "config" / "machine.yaml")
    assert main(["machine"]) == 1 and capsys.readouterr().err.endswith("fix: sage setup\n")
    assert main(["setup"]) == 0
    assert machine.MACHINE_FILE.read_text().endswith("data_dir:\n")
    assert main(["machine"]) == 1
    error = capsys.readouterr().err
    assert error.endswith("fill it in\nfix: sage setup --data-dir <directory>\n")
    assert main(["setup", "--data-dir", str(tmp_path / "missing")]) == 1
    assert "fix: mkdir -p" in capsys.readouterr().err
    assert main(["setup", "--data-dir", str(tmp_path)]) == 0
    assert machine.read_data_dir() == tmp_path.resolve() and main(["machine"]) == 0


def test_run_directory(capsys, tmp_path, monkeypatch):
    """init, copy-prior, prepare, stop and check on a new run."""
    presets = tmp_path / "presets"
    (presets / "runs").mkdir(parents=True)
    (presets / "priors").mkdir()
    (presets / "runs" / "tiny.yaml").write_text(RUN)
    shutil.copyfile(loader.PRESETS / "priors" / "bbh_narrow.yaml", presets / "priors" / "bbh_narrow.yaml")
    monkeypatch.setattr(loader, "PRESETS", presets)
    monkeypatch.setattr(machine, "MACHINE_FILE", tmp_path / "machine.yaml")
    machine.write_machine_file(tmp_path)
    run_dir = tmp_path / "run"
    assert main(["list", "presets"]) == 0 and "tiny" in capsys.readouterr().out
    assert main(["init", str(run_dir), "--preset", "tiny"]) == 0
    assert "extends: tiny" in (run_dir / "run.yaml").read_text()
    # A second init does not overwrite the run file.
    assert main(["init", str(run_dir), "--preset", "tiny"]) == 1
    assert main(["copy-prior", "bbh_narrow", str(run_dir)]) == 0
    assert (run_dir / "prior.yaml").read_bytes() == (presets / "priors" / "bbh_narrow.yaml").read_bytes()
    capsys.readouterr()
    assert main(["prepare", str(run_dir)]) == 1  # the data directory is empty
    assert capsys.readouterr().err.endswith(f"fix: sage download {run_dir}\n")
    assert main(["stop", str(run_dir)]) == 0 and main(["train", str(run_dir)]) == 1
    assert capsys.readouterr().err.endswith(f"fix: rm {run_dir / 'flags' / 'stop'} && sage train {run_dir}\n")
    assert main(["check", str(run_dir)]) == 1  # the data directory is empty
    assert capsys.readouterr().err.endswith(f"fix: sage download {run_dir}\n")


def test_run_dir_from_inside(capsys, tmp_path, monkeypatch):
    """A command run below a run directory acts on that run."""
    run_dir = tmp_path / "run"
    (run_dir / "logs" / "jobs").mkdir(parents=True)
    (run_dir / "run.yaml").write_text(RUN)
    monkeypatch.chdir(run_dir / "logs" / "jobs")
    assert main(["stop"]) == 0 and (run_dir / "flags" / "stop").exists()
    assert capsys.readouterr().out.rstrip().endswith("Next: sage status")
    assert find_run_dir(tmp_path) == tmp_path


def test_run_presets(tmp_path):
    """Every run preset initialises and its prior loads."""
    for name in loader.preset_names("runs"):
        assert main(["init", str(tmp_path / name), "--preset", name]) == 0
        settings = resolve_settings(loader.load_run(tmp_path / name / "run.yaml"))
        load_prior(settings["signal"]["prior"], tmp_path / name)


def test_epoch_seconds(tmp_path):
    """Epoch duration excludes the time between jobs."""
    log = tmp_path / "run.log"
    lines = [("00:00", "Run /r from epoch 0, seed 1"), ("00:01", "Epoch 0 | training: 2"),
             ("00:11", "Epoch 1 | training: 2"), ("00:31", "Epoch 2 | training: 2"),
             ("09:00", "Run /r from epoch 2, seed 1"), ("09:01", "Epoch 2 | training: 2"),
             ("09:41", "Epoch 3 | training: 2")]
    log.write_text("".join(f"2026-11-04 {at}:00 | INFO    | sage.x:1 | {text}\n" for at, text in lines))
    # The 8.5 h between jobs is excluded.
    assert epoch_seconds(log, 3) == (10 + 20 + 40) * 60 / 3
    assert epoch_seconds(log, 1) == 40 * 60


def test_executables():
    """pyproject lists sage, sgw, sage-help and one per command."""
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    scripts = set(re.findall(r'^(\S+) = "sage\.cli:main"$', pyproject, flags=re.M))
    assert scripts == {"sage", "sgw", "sage-help", *(f"sage-{name}" for name in COMMANDS)}


def test_observing_run_named_like_directory(tmp_path, monkeypatch):
    """A data flag names an observing run, even beside a same-named directory."""
    from sage.cli import data

    monkeypatch.setattr(machine, "MACHINE_FILE", tmp_path / "machine.yaml")
    assert main(["setup", "--data-dir", str(tmp_path)]) == 0
    (tmp_path / "o3a").mkdir()
    monkeypatch.chdir(tmp_path)
    called = []
    monkeypatch.setattr(data, "fetch_search", lambda options, data_dir: called.append("search"))
    monkeypatch.setattr(data, "build_runs", lambda options, data_dir: called.append("runs"))
    assert main(["download", "o3a", "--search"]) == 0
    assert main(["build-asds", "o3a", "--segments"]) == 0
    assert called == ["search", "runs"]
