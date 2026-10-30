"""Tests for sage.config: run-file reading, extends and value checking."""

from pathlib import Path

import pytest
import torch
import yaml
from torch.optim import lr_scheduler

from sage.config import load_run, prepare_run_dir
from sage.config import loader
from sage.config.prior import load_prior, load_prior_file
from sage.config.run_dir import record_derived, record_versions
from sage.factory.build import resolve_settings

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
  waveform:
    name: IMRPhenomPv2
  projection:
    name: constant
  snr:
    name: halfnorm
    scale: 4.0
    loc: 5.0
  targets: [tc, mchirp]
noise:
  training:
    name: real
    runs: [o3b]
  validation:
    name: real
    runs: [o3a]
  recolour:
    target: o3a
    probability: 0.37
    asd_jitter_gain: 0.5
    validation: false
  fiducial: [o3a, o3b]
  prefetch_batches: 8
  read_workers: 16
preprocess:
  - name: whiten
  - name: multirate
    lowest_sample_rate_in_hz: 64
    nyquist_margin_in_hz: 20
    min_bin_length_in_s: 0.05
    pre_merger_leeway_in_s: 0
    post_merger_leeway_in_s: 0
model:
  name: MSCNN1Datt_2DResNetCBAM
loss:
  name: BCE_PE_Sigma
optimiser:
  name: AdamW
schedule:
  name: WarmupPolynomialDecayFlatTail
train:
  epochs: 128
  samples_per_epoch: 2000000
  batch_size: 64
  signal_fraction: 0.5
  device: cuda:0
  precision: bf16
  clip_norm: 100.0
  compile:
    mode: max-autotune
    fullgraph: true
    dynamic: true
  seed: 150914
  keep_checkpoints: 2
validation:
  samples: 200000
  every_n_epochs: 5
  final_samples: 2000000
  seed: 170817
  snr:
    name: halfnorm
    scale: 4.0
    loc: 5.0
mining:
  enabled: true
ema:
  enabled: true
"""

PRIOR = load_prior("bbh_narrow", Path("."))


def write(directory: Path, name: str, text: str) -> Path:
    """Write ``text`` to ``directory/name`` and return the path."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def presets(tmp_path, monkeypatch):
    """Empty presets directory used by the loader."""
    monkeypatch.setattr(loader, "PRESETS", tmp_path / "presets")
    return tmp_path / "presets"


def test_types(tmp_path):
    settings = load_run(write(tmp_path, "run.yaml", RUN))
    assert settings["data"] == {
        "detectors": ["H1", "L1"],
        "sample_rate_in_hz": 2048.0,
        "sample_length_in_s": 12.0,
        "padding_length_in_s": 2.0,
        "dtype": "float32",
    }
    assert settings["signal"]["low_frequency_cutoff_in_hz"] == 20.0
    assert settings["signal"]["targets"] == ["tc", "mchirp"]


def test_component_parameters(tmp_path):
    snr = load_run(write(tmp_path, "run.yaml", RUN))["signal"]["snr"]
    assert snr == {"name": "halfnorm", "scale": "4.0", "loc": "5.0"}
    with pytest.raises(ValueError, match="signal.projection: missing setting 'name'"):
        load_run(write(tmp_path, "run.yaml", RUN.replace("name: constant", "kind: constant")))


def test_noise_section(tmp_path):
    noise = load_run(write(tmp_path, "run.yaml", RUN))["noise"]
    assert noise["training"] == {"name": "real", "runs": ["o3b"]}
    assert noise["recolour"]["validation"] is False
    assert noise["prefetch_batches"] == 8


def test_optional_group(tmp_path):
    start = RUN.index("  recolour:")
    end = RUN.index("  fiducial:")
    noise = load_run(write(tmp_path, "run.yaml", RUN[:start] + RUN[end:]))["noise"]
    assert "recolour" not in noise


def test_unknown_variant(tmp_path):
    text = RUN.replace("    name: real\n    runs: [o3a]", "    name: rael\n    runs: [o3a]")
    with pytest.raises(ValueError, match="noise.validation.name: unknown 'rael'; did you mean 'real'"):
        load_run(write(tmp_path, "run.yaml", text))


def test_preprocess_steps(tmp_path):
    steps = load_run(write(tmp_path, "run.yaml", RUN))["preprocess"]
    assert [step["name"] for step in steps] == ["whiten", "multirate"]
    assert steps[1]["min_bin_length_in_s"] == 0.05
    with pytest.raises(ValueError, match=r"preprocess\[0\].name: unknown 'whitten'; did you mean 'whiten'"):
        load_run(write(tmp_path, "run.yaml", RUN.replace("- name: whiten", "- name: whitten")))
    with pytest.raises(ValueError, match=r"preprocess\[1\]: missing setting 'post_merger_leeway_in_s'"):
        load_run(write(tmp_path, "run.yaml", RUN.replace("    post_merger_leeway_in_s: 0\n", "")))


def test_exponent_without_dot(tmp_path):
    text = RUN.replace("sample_rate_in_hz: 2048", "sample_rate_in_hz: 2e3")
    assert load_run(write(tmp_path, "run.yaml", text))["data"]["sample_rate_in_hz"] == 2000.0


def test_not_a_number(tmp_path):
    text = RUN.replace("sample_rate_in_hz: 2048", "sample_rate_in_hz: fast")
    with pytest.raises(ValueError, match=r"data.sample_rate_in_hz: expected float"):
        load_run(write(tmp_path, "run.yaml", text))


def test_unknown_setting(tmp_path):
    text = RUN.replace("sample_rate_in_hz", "sample_rate")
    with pytest.raises(ValueError, match="did you mean 'sample_rate_in_hz'"):
        load_run(write(tmp_path, "run.yaml", text))


def test_missing_setting(tmp_path):
    text = RUN.replace("  dtype: float32\n", "")
    with pytest.raises(ValueError, match="data: missing setting 'dtype'"):
        load_run(write(tmp_path, "run.yaml", text))


def test_bad_choice(tmp_path):
    text = RUN.replace("dtype: float32", "dtype: float16")
    with pytest.raises(ValueError, match=r"data.dtype: 'float16' is not one of"):
        load_run(write(tmp_path, "run.yaml", text))


def test_bad_detector(tmp_path):
    text = RUN.replace("[H1, L1]", "[H1, X1]")
    with pytest.raises(ValueError, match=r"data.detectors\[1\]: 'X1' is not one of"):
        load_run(write(tmp_path, "run.yaml", text))


def test_extends(tmp_path, presets):
    write(presets / "runs", "base.yaml", RUN)
    run = write(tmp_path, "run.yaml", "extends: base\ndata:\n  detectors: [H1]\n")
    data = load_run(run)["data"]
    assert data["detectors"] == ["H1"]
    assert data["sample_rate_in_hz"] == 2048.0


def test_unknown_preset(tmp_path, presets):
    write(presets / "runs", "base.yaml", RUN)
    run = write(tmp_path, "run.yaml", "extends: bse\n")
    with pytest.raises(ValueError, match="unknown run preset 'bse'; did you mean 'base'"):
        load_run(run)


def test_preset_chain(tmp_path, presets):
    write(presets / "runs", "base.yaml", RUN)
    write(presets / "runs", "child.yaml", "extends: base\n")
    run = write(tmp_path, "run.yaml", "extends: child\n")
    with pytest.raises(ValueError, match="run preset 'child' extends another preset"):
        load_run(run)


def test_merge_replaces_lists():
    merged = loader.merge({"a": {"b": [1, 2], "c": 1}}, {"a": {"b": [3]}})
    assert merged == {"a": {"b": [3], "c": 1}}


def record(run_dir, text, prior=PRIOR):
    """Resolve a run file and record it in ``run_dir``."""
    settings = resolve_settings(load_run(write(run_dir, "run.yaml", text)))
    prepare_run_dir(run_dir, settings, prior, resolve_settings)
    return settings


def test_resolved_round_trip(tmp_path):
    settings = record(tmp_path, RUN)
    path = tmp_path / "resolved" / "run.yaml"
    text = path.read_text()
    assert "detectors: [H1, L1]" in text
    assert "frontend_filters: 32" in text and "betas: [0.9, 0.999]" in text and "foreach: null" in text
    assert resolve_settings(load_run(path)) == settings
    assert load_prior_file(tmp_path / "resolved" / "prior.yaml") == PRIOR


def test_rewrite_before_training(tmp_path):
    record(tmp_path, RUN)
    changed = record(tmp_path, RUN.replace("2048", "4096"))
    assert resolve_settings(load_run(tmp_path / "resolved" / "run.yaml")) == changed


def test_resume_unchanged(tmp_path):
    record(tmp_path, RUN)
    write(tmp_path / "checkpoints", "latest.pt", "")
    record(tmp_path, RUN)


def test_resume_changed(tmp_path):
    record(tmp_path, RUN)
    write(tmp_path / "checkpoints", "latest.pt", "")
    with pytest.raises(ValueError, match=r"data.sample_rate_in_hz: 2048.0 -> 4096.0"):
        record(tmp_path, RUN.replace("2048", "4096"))


def test_resume_prior_changed(tmp_path):
    record(tmp_path, RUN)
    write(tmp_path / "checkpoints", "latest.pt", "")
    changed = {**PRIOR, "priors": {**PRIOR["priors"], "tc": {**PRIOR["priors"]["tc"], "max": 11.3}}}
    with pytest.raises(ValueError, match=r"prior.yaml differs .*priors.tc.max: 11.2 -> 11.3"):
        record(tmp_path, RUN, changed)


def test_checkpoints_without_record(tmp_path):
    write(tmp_path / "checkpoints", "latest.pt", "")
    with pytest.raises(ValueError, match="resolved/run.yaml is missing"):
        record(tmp_path, RUN)


def test_derived_changed(tmp_path):
    record_derived(tmp_path, {"bins": [[0, 10, 64]]})
    write(tmp_path / "checkpoints", "latest.pt", "")
    record_derived(tmp_path, {"bins": [[0, 10, 64]]})
    with pytest.raises(ValueError, match=r"bins: \[\[0, 10, 64\]\] -> \[\[0, 12, 64\]\]"):
        record_derived(tmp_path, {"bins": [[0, 12, 64]]})


def test_code_change_recorded(tmp_path, monkeypatch):
    differs = {"now": False}
    monkeypatch.setattr("sage.config.run_dir.code_differs", lambda commit: differs["now"])
    record_versions(tmp_path, 0, "cpu")
    write(tmp_path / "checkpoints", "latest.pt", "")
    differs["now"] = True
    record_versions(tmp_path, 1, "cpu")  # a started run on changed code goes on
    path = tmp_path / "resolved" / "versions.yaml"
    changed = [s["sage_uncommitted_changes"] for s in yaml.safe_load(path.read_text())["segments"]]
    assert changed == [False, True]


class Component:
    """Component with typed parameters, a default, and an untyped one."""

    def __init__(self, scale: float, enabled: bool, device, label: str = "x", raw=0):
        pass


def test_component_arguments():
    arguments = loader.component_arguments(
        Component, {"scale": "2e-1", "enabled": "false"}, {"device", "raw"}, "c"
    )
    assert arguments == {"scale": 0.2, "enabled": False}


def test_component_unknown_and_missing():
    with pytest.raises(ValueError, match="c: unknown parameter 'scal'; did you mean 'scale'"):
        loader.component_arguments(Component, {"scal": "1"}, {"device", "raw"}, "c")
    with pytest.raises(ValueError, match="c: missing parameter 'enabled'"):
        loader.component_arguments(Component, {"scale": "1"}, {"device", "raw"}, "c")


def test_component_bool_strict():
    with pytest.raises(ValueError, match="c.enabled: expected true or false, got 'yes'"):
        loader.component_arguments(
            Component, {"scale": "1", "enabled": "yes"}, {"device", "raw"}, "c"
        )


def test_component_untyped():
    with pytest.raises(ValueError, match="Component.raw has no type hint"):
        loader.component_arguments(
            Component, {"scale": "1", "enabled": "true", "raw": "1"}, {"device"}, "c"
        )


def test_torch_hints():
    adamw = {"lr": "3e-4", "betas": ["0.8", "0.99"], "foreach": "false"}
    assert loader.component_arguments(torch.optim.AdamW, adamw, {"params"}, "optimiser") == {
        "lr": 3e-4, "betas": [0.8, 0.99], "foreach": False,
    }
    cyclic = {"base_lr": "1e-4", "max_lr": ["1e-3", "2e-3"], "mode": "exp_range"}
    assert loader.component_arguments(lr_scheduler.CyclicLR, cyclic, {"optimizer"}, "schedule") == {
        "base_lr": 1e-4, "max_lr": [1e-3, 2e-3], "mode": "exp_range",
    }
    with pytest.raises(ValueError, match="schedule.mode: 'maximum' is not one of min, max"):
        loader.component_arguments(lr_scheduler.ReduceLROnPlateau, {"mode": "maximum"}, {"optimizer"}, "schedule")


def test_torch_hint_without_yaml_form():
    with pytest.raises(ValueError, match="schedule.lr_lambda: takes .*Callable.*cannot be written in a run file"):
        loader.component_arguments(lr_scheduler.LambdaLR, {"lr_lambda": "f"}, {"optimizer"}, "schedule")


def test_setting_default(tmp_path):
    synthetic = (
        "    name: synthetic\n"
        "    asd:\n      H1: aLIGODesignSensitivityP1200087\n"
        "      L1: aLIGODesignSensitivityP1200087\n"
        "    low_frequency_cutoff_in_hz: 15\n"
    )
    text = RUN.replace("    name: real\n    runs: [o3a]\n", synthetic)
    validation = load_run(write(tmp_path, "run.yaml", text))["noise"]["validation"]
    assert validation["filter_duration_in_s"] == 128.0
    text = text.replace("    low_frequency_cutoff_in_hz: 15\n", "    low_frequency_cutoff_in_hz: 15\n    filter_duration_in_s: 8\n")
    assert load_run(write(tmp_path, "run.yaml", text))["noise"]["validation"]["filter_duration_in_s"] == 8.0
