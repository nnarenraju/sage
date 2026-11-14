"""Tests for sage.config.prior: prior presets, prior files and checks."""

from pathlib import Path

import pytest
import torch
import yaml

from sage.config.loader import merge, read_preset
from sage.config.prior import load_prior
from sage.data.waveform.sampler import prior_sampler


def write_prior(directory, changes):
    """Write bbh_narrow with ``changes`` merged in, as a whole prior file."""
    prior = merge(read_preset("priors", "bbh_narrow"), changes)
    (directory / "prior.yaml").write_text(yaml.safe_dump(prior, sort_keys=False))


def test_preset_numbers():
    """The preset reads with float bounds and a named sky prior."""
    prior = load_prior("bbh_narrow", Path("."))["priors"]
    assert prior["mass1"]["min"] == 7.0
    assert isinstance(prior["delta_tc"]["min"], float) and "tc" not in prior
    assert prior["sky"] == {"name": "uniform_sky", "polar-angle": "dec", "azimuthal-angle": "ra"}


def test_extends_refused(tmp_path):
    """A prior file is whole: it cannot extend a preset."""
    (tmp_path / "prior.yaml").write_text("extends: bbh_narrow\n")
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['extends'\]"):
        load_prior("prior.yaml", tmp_path)


def test_min_not_number(tmp_path):
    """A non-numeric bound is refused."""
    write_prior(tmp_path, {"priors": {"delta_tc": {"min": "early"}}})
    with pytest.raises(ValueError, match="prior 'delta_tc': min must be a number, got 'early'"):
        load_prior("prior.yaml", tmp_path)


def test_variable_params_mismatch(tmp_path):
    """variable_params that differ from the priors are refused."""
    write_prior(tmp_path, {"variable_params": ["mass1", "mass2"]})
    with pytest.raises(ValueError, match="variable_params does not match"):
        load_prior("prior.yaml", tmp_path)


def test_sky_names(tmp_path):
    """Sky angles may be renamed; the sampler uses the new names."""
    renamed = {"dec": "declination", "ra": "right_ascension"}
    names = [renamed.get(n, n) for n in load_prior("bbh_narrow", tmp_path)["variable_params"]]
    sky = {"polar-angle": "declination", "azimuthal-angle": "right_ascension"}
    write_prior(tmp_path, {"variable_params": names, "priors": {"sky": sky}})
    sampler = prior_sampler(load_prior("prior.yaml", tmp_path), 0, "cpu", torch.float32)
    declination = sampler(1000)[:, sampler.param_index["declination"]]
    assert "dec" not in sampler.param_index and declination.abs().max() <= torch.pi / 2


def test_bounds_follow_constraints(tmp_path):
    """Bounds follow the constraints, mass ordering included."""
    write_prior(tmp_path, {"constraints": []})
    bounds = prior_sampler(load_prior("prior.yaml", tmp_path), 0, "cpu", torch.float32).bounds
    # Without constraints there is no mass ordering.
    assert bounds["mass1"] == (7.0, 50.0) and bounds["q"] == (7.0 / 50.0, 50.0 / 7.0)
    ordered = prior_sampler(load_prior("bbh_narrow", tmp_path), 0, "cpu", torch.float32).bounds
    assert ordered["q"] == (1.0, 50.0 / 7.0) and ordered["spin1z"] == (-0.99, 0.99)


def test_mass_order_rejects(tmp_path):
    """Mass ordering rejects draws rather than swapping masses."""
    write_prior(tmp_path, {"priors": {"mass1": {"min": "10", "max": "30"}, "mass2": {"min": "5", "max": "50"}}})
    sampler = prior_sampler(load_prior("prior.yaml", tmp_path), 0, "cpu", torch.float32)
    draws = sampler(10_000)
    mass1, mass2 = draws[:, sampler.param_index["mass1"]], draws[:, sampler.param_index["mass2"]]
    # Swapping would put mass1 up to 50.
    assert mass1.max() <= 30.0 and (mass1 >= mass2).all()


def test_broad_preset():
    """bbh_broad differs from bbh_narrow only in the masses and delta_tc."""
    narrow, broad = (load_prior(name, Path("."))["priors"] for name in ("bbh_narrow", "bbh_broad"))
    assert broad["mass1"] == broad["mass2"] == {"name": "uniform", "min": 5.0, "max": 100.0}
    assert broad["delta_tc"] == {"name": "uniform", "min": 18.0, "max": 18.2}
    unchanged = {"mass1": narrow["mass1"], "mass2": narrow["mass2"], "delta_tc": narrow["delta_tc"]}
    assert {**broad, **unchanged} == narrow

