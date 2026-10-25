"""Tests for sage.config.prior: prior presets, extends and checks."""

from pathlib import Path

import pytest

from sage.config.prior import load_prior


def test_preset_numbers():
    prior = load_prior("bbh_narrow", Path("."))["priors"]
    assert prior["mass1"]["min"] == 7.0
    assert isinstance(prior["injection_time"]["min"], float)
    assert prior["sky"] == {"name": "uniform_sky", "ra": "ra", "dec": "dec"}


def test_file_extends_preset(tmp_path):
    (tmp_path / "prior.yaml").write_text(
        "extends: bbh_narrow\npriors:\n  mass1:\n    max: 60\n"
    )
    priors = load_prior("prior.yaml", tmp_path)["priors"]
    assert priors["mass1"] == {"name": "uniform", "min": 7.0, "max": 60.0}
    assert priors["mass2"]["max"] == 50.0


def test_min_not_number(tmp_path):
    (tmp_path / "prior.yaml").write_text(
        "extends: bbh_narrow\npriors:\n  tc:\n    min: early\n"
    )
    with pytest.raises(ValueError, match="prior 'tc': min must be a number, got 'early'"):
        load_prior("prior.yaml", tmp_path)


def test_variable_params_mismatch(tmp_path):
    (tmp_path / "prior.yaml").write_text(
        "extends: bbh_narrow\nvariable_params: [mass1, mass2]\n"
    )
    with pytest.raises(ValueError, match="variable_params does not match"):
        load_prior("prior.yaml", tmp_path)
