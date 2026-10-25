"""Tests for sage.factory.build: section builders."""

import logging
from pathlib import Path

import pytest
import torch

from sage.config.prior import load_prior
from sage.core.pipeline import Grid, ProcessingState
from sage.data.waveform import prior_sampler
from sage.dsp.multirate_sampling import MultirateSampler
from sage.dsp.normalise import MinMaxNormalise
from sage.dsp.whiten import FiducialWhitening
from sage.factory.build import (
    build_data,
    build_noise,
    build_optimiser,
    build_preprocess,
    build_schedule,
    build_signal,
    check_heads,
    check_mining,
    check_train,
    compile_arguments,
    mining_epochs,
    precision_setup,
    resolve_loss,
    resolve_model,
    resolve_optimiser,
    resolve_schedule,
)


def data(detectors: list, dtype: str) -> dict:
    """A checked production ``data`` section."""
    return {
        "detectors": detectors,
        "sample_rate_in_hz": 2048.0,
        "sample_length_in_s": 12.0,
        "padding_length_in_s": 2.0,
        "dtype": dtype,
    }


def test_duplicate_detector():
    with pytest.raises(ValueError, match="more than once: H1"):
        build_data(data(["H1", "H1"], "float32"))


def test_float64_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _, _, dtype = build_data(data(["H1", "L1"], "float64"))
    assert dtype is torch.float64
    assert "waveform models compute in float64 regardless" in caplog.text


def signal(**overrides) -> dict:
    """A checked production ``signal`` section."""
    section = {
        "prior": "bbh_narrow",
        "low_frequency_cutoff_in_hz": 20.0,
        "reference_frequency_in_hz": 20.0,
        "waveform": {"name": "IMRPhenomPv2"},
        "projection": {"name": "constant"},
        "snr": {"name": "halfnorm", "scale": "4.0", "loc": "5.0"},
        "targets": ["tc", "mchirp"],
    }
    return {**section, **overrides}


def build(section: dict):
    """Build a signal sampler for the production window."""
    window, detectors, dtype = build_data(data(["H1", "L1"], "float32"))
    asds = torch.ones(2, window.n_frequency_bins)
    prior = load_prior("bbh_narrow", Path("."))
    return build_signal(section, prior, window, detectors, asds, 2, 1, "cpu", dtype)


def test_signal_off_grid():
    with pytest.raises(ValueError, match="20.03 Hz is not on the 1/16 Hz grid"):
        build(signal(low_frequency_cutoff_in_hz=20.03))


def test_signal_unknown_target():
    with pytest.raises(ValueError, match="'mchrip' is not a prior parameter; did you mean 'mchirp'"):
        build(signal(targets=["tc", "mchrip"]))


def test_signal_unknown_waveform():
    with pytest.raises(ValueError, match="unknown 'IMRPhenomPV2'; did you mean 'IMRPhenomPv2'"):
        build(signal(waveform={"name": "IMRPhenomPV2"}))


def test_recolour_synthetic(tmp_path):
    window, detectors, dtype = build_data(data(["H1", "L1"], "float32"))
    synthetic = {
        "name": "synthetic",
        "asd": {"H1": "aLIGODesignSensitivityP1200087", "L1": "aLIGODesignSensitivityP1200087"},
        "low_frequency_cutoff_in_hz": 15.0,
        "filter_duration_in_s": 128.0,
    }
    noise = {
        "training": synthetic,
        "validation": synthetic,
        "recolour": {"target": "o3a", "probability": 0.37, "asd_jitter_gain": 0.5, "validation": False},
        "fiducial": ["o3a", "o3b"],
        "prefetch_batches": 2,
        "read_workers": 1,
    }
    with pytest.raises(ValueError, match="recolour applies to real noise; noise.training is synthetic"):
        build_noise(noise, tmp_path, window, detectors, 2, 1, 2, 2.0, 3.0, "cpu", dtype)


def preprocess(post_merger_leeway_in_s: float) -> list:
    """A checked ``preprocess`` section with every step."""
    multirate = {
        "name": "multirate",
        "lowest_sample_rate_in_hz": 64.0,
        "nyquist_margin_in_hz": 20.0,
        "min_bin_length_in_s": 0.05,
        "pre_merger_leeway_in_s": 0.0,
        "post_merger_leeway_in_s": post_merger_leeway_in_s,
    }
    return [{"name": "whiten"}, {"name": "minmax_normalise"}, multirate]


def test_preprocess_leeway():
    window, _, dtype = build_data(data(["H1", "L1"], "float32"))
    bounds = prior_sampler(load_prior("bbh_narrow", Path(".")), 1, "cpu", dtype).bounds
    asds = torch.ones(2, window.n_frequency_bins)
    full_rate = []
    for leeway in (0.0, 0.25):
        chain = build_preprocess(
            preprocess(leeway), asds, window, bounds, 20.0, ProcessingState(Grid.FD_UNIFORM),
            "cpu", dtype,
        )
        assert [type(step) for step in chain.seq] == [FiducialWhitening, MinMaxNormalise, MultirateSampler]
        full_rate.append(next(b for b in chain.seq[-1].bins.tolist() if b[2] == 2048))
    assert full_rate[1][1] - full_rate[0][1] == 512
    assert full_rate[1][0] == full_rate[0][0]


def test_model_defaults():
    production = resolve_model({"name": "MSCNN1Datt_2DResNetCBAM"})
    assert production == {
        "name": "MSCNN1Datt_2DResNetCBAM",
        "frontend_filters": 32,
        "frontend_kernel": 64,
        "backend_resnet_size": 50,
        "norm_type": "instancenorm",
        "dropout": 0.0,
        "use_blurpool": True,
        "use_resnet_cd": True,
    }
    changed = resolve_model({"name": "MSCNN1Datt_2DResNetCBAM", "backend_resnet_size": "18", "use_blurpool": "false"})
    assert changed == {**production, "backend_resnet_size": 18, "use_blurpool": False}


def test_model_counts_from_builder():
    with pytest.raises(ValueError, match="model: unknown parameter 'num_detectors'"):
        resolve_model({"name": "MSCNN1Datt_2DResNetCBAM", "num_detectors": "3"})


def test_loss_defaults():
    assert resolve_loss({"name": "BCE_PE_Sigma"}) == {
        "name": "BCE_PE_Sigma",
        "regression_weight": 0.005,
        "coupling_weight": 0.005,
        "beta": 0.5,
        "sigma_min": 0.001,
        "sigma_max": 10.0,
        "confidence_weighting": True,
    }


def test_heads_mismatch():
    message = (
        r"model MSCNN1D_2DResNetCBAM gives one estimate per target, but loss BCE_PE_Sigma needs an "
        r"estimate and a width per target; use a loss from \['BCE_PE_Huber', 'BCE_PE_MSE'\] or a "
        r"model from \['MSCNN1Datt_2DResNetCBAM'\]"
    )
    check_heads({"name": "MSCNN1Datt_2DResNetCBAM"}, {"name": "BCE_PE_Sigma"})
    with pytest.raises(ValueError, match=message):
        check_heads({"name": "MSCNN1D_2DResNetCBAM"}, {"name": "BCE_PE_Sigma"})


def test_optimiser_defaults():
    adamw = resolve_optimiser({"name": "AdamW"})
    assert (adamw["lr"], adamw["weight_decay"]) == (2e-4, 1e-4)
    assert "params" not in adamw and "fused" not in adamw
    sgd = resolve_optimiser({"name": "SGD", "lr": "0.1"})
    assert (sgd["lr"], sgd["weight_decay"], sgd["momentum"]) == (0.1, 0, 0)


def test_optimiser_groups():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
    decay, no_decay = build_optimiser({"name": "AdamW"}, model).param_groups
    assert [p.ndim for p in decay["params"]] == [2] and decay["weight_decay"] == 1e-4
    assert [p.ndim for p in no_decay["params"]] == [1, 1, 1] and no_decay["weight_decay"] == 0.0
    assert decay["fused"] is True


def schedule_factors(**settings) -> list:
    """Factors of a 5-epoch, 4-batch run (lr 2e-4, floor 1e-6 -> 0.005)."""
    optimiser = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
    section = {"name": "WarmupPolynomialDecayFlatTail", **{k: str(v) for k, v in settings.items()}}
    factor = build_schedule(section, optimiser, 5, 4)[0].lr_lambdas[0]
    return [factor(step) for step in range(21)]


def test_schedule_shape():
    factors = schedule_factors(warmup_steps=4, flat_tail_epochs=1)
    assert factors[0] == 1e-3 and factors[4] == 1.0              # warmup to the peak
    decay = factors[4:16]
    assert all(a > b for a, b in zip(decay, decay[1:]))           # polynomial decay
    assert factors[16:] == [1e-6 / 2e-4] * 5                      # flat tail at the floor


def test_schedule_refusals():
    with pytest.raises(ValueError, match="warmup_steps: 11 is more than half the run .20 batches.; use at most 10"):
        schedule_factors(warmup_steps=11)
    with pytest.raises(ValueError, match="warmup .8 batches. and flat tail .16 batches. are longer than the run"):
        schedule_factors(warmup_steps=8, flat_tail_epochs=4)


def test_mining_epochs():
    mining = {"start_epoch": 3, "every_n_epochs": 4}
    assert mining_epochs(mining, {"name": "WarmupPolynomialDecayFlatTail"}, 128) == list(range(3, 124, 4))
    assert mining_epochs(mining, {"name": "StepLR", "step_size": "1", "step_every": "epoch"}, 12) == [3, 7, 11]


def test_mining_checks():
    mining, model = {"start_epoch": 124, "every_n_epochs": 4}, {"name": "MSCNN1Datt_2DResNetCBAM"}
    schedule = {"name": "WarmupPolynomialDecayFlatTail"}
    with pytest.raises(ValueError, match="mining needs real training noise; noise.training is synthetic"):
        check_mining(mining, {"training": {"name": "synthetic"}}, model, schedule, 128)
    with pytest.raises(ValueError, match="mining.start_epoch 124 leaves no mining round: mining stops at epoch 124"):
        check_mining(mining, {"training": {"name": "real"}}, model, schedule, 128)


def train(**overrides) -> dict:
    """A checked ``train`` section for a tiny CPU run."""
    section = {
        "epochs": 2, "samples_per_epoch": 8, "batch_size": 4, "signal_fraction": 0.5,
        "device": "cpu", "precision": "fp32", "clip_norm": 100.0, "compile": False,
        "seed": 1, "keep_checkpoints": 2,
    }
    return {**section, **overrides}


def test_train_checks():
    check_train(train())
    with pytest.raises(ValueError, match="samples_per_epoch 10 is not a whole number of batches of 4; nearest is 8"):
        check_train(train(samples_per_epoch=10))
    with pytest.raises(ValueError, match=r"0.3 of a batch of 4 is 1.2 signals; nearest whole is 0.25 \(1 per batch\)"):
        check_train(train(signal_fraction=0.3))


def test_compile_settings():
    assert compile_arguments(False) is None
    settings = {"mode": "max-autotune", "fullgraph": "true", "dynamic": "true"}
    assert compile_arguments(settings) == {"mode": "max-autotune", "fullgraph": True, "dynamic": True}
    with pytest.raises(ValueError, match="train.compile.mode: required"):
        compile_arguments({"fullgraph": "true"})
    with pytest.raises(ValueError, match="unknown parameter 'full_graph'; did you mean 'fullgraph'"):
        compile_arguments({"mode": "default", "full_graph": "true"})


def test_precision():
    assert precision_setup(train(precision="fp32")) == (None, None)
    dtype, scaler = precision_setup(train(precision="fp16"))
    assert dtype is torch.float16 and isinstance(scaler, torch.amp.GradScaler)


def test_schedule_step_every():
    with pytest.raises(ValueError, match="step_every: required for CosineAnnealingWarmRestarts"):
        resolve_schedule({"name": "CosineAnnealingWarmRestarts", "T_0": "5"})
    with pytest.raises(ValueError, match="WarmupPolynomialDecayFlatTail steps every batch"):
        resolve_schedule({"name": "WarmupPolynomialDecayFlatTail", "step_every": "batch"})
