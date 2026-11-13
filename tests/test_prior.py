"""Tests for sage.config.prior: prior presets, extends and checks."""

from pathlib import Path

import pytest
import torch

from sage.config.prior import load_prior
from sage.data.waveform.sampler import prior_sampler


def test_preset_numbers():
    """The preset reads with float bounds and a named sky prior."""
    prior = load_prior("bbh_narrow", Path("."))["priors"]
    assert prior["mass1"]["min"] == 7.0
    assert isinstance(prior["delta_tc"]["min"], float) and "tc" not in prior
    assert prior["sky"] == {"name": "uniform_sky", "polar-angle": "dec", "azimuthal-angle": "ra"}


def test_file_extends_preset(tmp_path):
    """A prior file extending a preset overrides only what it sets."""
    (tmp_path / "prior.yaml").write_text(
        "extends: bbh_narrow\npriors:\n  mass1:\n    max: 60\n"
    )
    priors = load_prior("prior.yaml", tmp_path)["priors"]
    assert priors["mass1"] == {"name": "uniform", "min": 7.0, "max": 60.0}
    assert priors["mass2"]["max"] == 50.0


def test_min_not_number(tmp_path):
    """A non-numeric bound is refused."""
    (tmp_path / "prior.yaml").write_text(
        "extends: bbh_narrow\npriors:\n  delta_tc:\n    min: early\n"
    )
    with pytest.raises(ValueError, match="prior 'delta_tc': min must be a number, got 'early'"):
        load_prior("prior.yaml", tmp_path)


def test_variable_params_mismatch(tmp_path):
    """variable_params that differ from the priors are refused."""
    (tmp_path / "prior.yaml").write_text(
        "extends: bbh_narrow\nvariable_params: [mass1, mass2]\n"
    )
    with pytest.raises(ValueError, match="variable_params does not match"):
        load_prior("prior.yaml", tmp_path)


def test_sky_names(tmp_path):
    """Sky angles may be renamed; the sampler uses the new names."""
    renamed = {"dec": "declination", "ra": "right_ascension"}
    names = "".join(f"  - {renamed.get(n, n)}\n" for n in load_prior("bbh_narrow", tmp_path)["variable_params"])
    (tmp_path / "prior.yaml").write_text(
        f"extends: bbh_narrow\nvariable_params:\n{names}priors:\n  sky:\n"
        "    polar-angle: declination\n    azimuthal-angle: right_ascension\n"
    )
    sampler = prior_sampler(load_prior("prior.yaml", tmp_path), 0, "cpu", torch.float32)
    declination = sampler(1000)[:, sampler.param_index["declination"]]
    assert "dec" not in sampler.param_index and declination.abs().max() <= torch.pi / 2


def test_bounds_follow_constraints(tmp_path):
    """Bounds follow the constraints, mass ordering included."""
    (tmp_path / "prior.yaml").write_text("extends: bbh_narrow\nconstraints: []\n")
    bounds = prior_sampler(load_prior("prior.yaml", tmp_path), 0, "cpu", torch.float32).bounds
    # Without constraints there is no mass ordering.
    assert bounds["mass1"] == (7.0, 50.0) and bounds["q"] == (7.0 / 50.0, 50.0 / 7.0)
    ordered = prior_sampler(load_prior("bbh_narrow", tmp_path), 0, "cpu", torch.float32).bounds
    assert ordered["q"] == (1.0, 50.0 / 7.0) and ordered["spin1z"] == (-0.99, 0.99)


def test_mass_order_rejects(tmp_path):
    """Mass ordering rejects draws rather than swapping masses."""
    (tmp_path / "prior.yaml").write_text(
        "extends: bbh_narrow\npriors:\n  mass1: {min: 10, max: 30}\n  mass2: {min: 5, max: 50}\n"
    )
    sampler = prior_sampler(load_prior("prior.yaml", tmp_path), 0, "cpu", torch.float32)
    draws = sampler(10_000)
    mass1, mass2 = draws[:, sampler.param_index["mass1"]], draws[:, sampler.param_index["mass2"]]
    # Swapping would put mass1 up to 50.
    assert mass1.max() <= 30.0 and (mass1 >= mass2).all()


def test_broad_preset():
    """bbh_broad differs from bbh_narrow only in the masses."""
    narrow, broad = (load_prior(name, Path("."))["priors"] for name in ("bbh_narrow", "bbh_broad"))
    assert broad["mass1"] == broad["mass2"] == {"name": "uniform", "min": 5.0, "max": 100.0}
    assert {**broad, "mass1": narrow["mass1"], "mass2": narrow["mass2"]} == narrow

