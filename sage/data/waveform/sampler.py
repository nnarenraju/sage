#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : distributions.py
Description     : Short description of the file

Created on 2026-02-16 10:35:39

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation:

    sampler = read_from_config("gw_config.yaml", device="cuda")

    batch = sampler.sample(4096)

    print(batch["mass1"].shape)      # (4096,)
    print(batch["spin1x"].shape)     # derived param
    print(batch["distance"].shape)   # transformed param

"""

# Packages
import yaml
import torch

from typing import Dict, Any, Callable


def spherical_to_cartesian(radial, polar, azimuthal):
    sin_theta = torch.sin(polar)
    return (
        radial * sin_theta * torch.cos(azimuthal),
        radial * sin_theta * torch.sin(azimuthal),
        radial * torch.cos(polar),
    )


def mass1_mass2_to_mchirp_q(m1, m2):
    q = m2 / m1
    mchirp = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    return mchirp, q


def chirp_distance_to_distance(chirp_distance, mchirp):
    return chirp_distance * (mchirp / 1.2) ** (5 / 6)


def read_from_config(path, device="cuda"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return DistributionSampler(config, device=device)


class ConstraintChecker:
    def __init__(self, expr: str):
        self.expr = expr

    def check(self, params: Dict[str, torch.Tensor]):
        return eval(self.expr, {}, params)


class DistributionSampler:

    def __init__(self, config: Dict[str, Any], device="cuda"):
        self.device = device
        self.cfg = config

        self.variable_params = config["variable_params"]
        self.static_params = config.get("static_params", {})

        self.distributions = {}
        self.transforms = []
        self.constraints = []

        self._build_distributions()
        self._build_transforms()
        self._build_constraints()

    def _make_dist(self, name, args):
        if name == "uniform":
            return Uniform(args["min"], args["max"])
        if name == "uniform_angle":
            return UniformAngle()
        if name == "sin_angle":
            return SinAngle()
        if name == "uniform_sky":
            return UniformSky()
        if name == "uniform_solidangle":
            return UniformSolidAngle(args["polar-angle"], args["azimuthal-angle"])
        if name == "uniform_radius":
            return UniformRadius(args["min"], args["max"])
        raise ValueError(f"Unknown distribution {name}")

    def _build_distributions(self):
        for pname, pcfg in self.cfg["priors"].items():
            name = pcfg["name"]
            args = {k: v for k, v in pcfg.items() if k != "name"}
            self.distributions[pname] = self._make_dist(name, args)

    def _build_transforms(self):
        for _, tcfg in self.cfg.get("waveform_transforms", {}).items():
            name = tcfg["name"]

            if name == "spherical_to_cartesian":
                self.transforms.append(("spin_cartesian", tcfg))

            elif name == "mass1_mass2_to_mchirp_q":
                self.transforms.append(("mass", tcfg))

            elif name == "chirp_distance_to_distance":
                self.transforms.append(("distance", tcfg))

    def _build_constraints(self):
        for c in self.cfg.get("constraints", []):
            self.constraints.append(ConstraintChecker(c["expr"]))

    def _sample_base(self, N):
        params = {}

        for name in self.variable_params:
            if name in self.distributions:
                params[name] = self.distributions[name].sample((N,), self.device)

        # add static params as tensors
        for k, v in self.static_params.items():
            params[k] = torch.full((N,), float(v), device=self.device)

        return params

    def _apply_transforms(self, params):

        for tname, cfg in self.transforms:

            if tname == "spin_cartesian":
                x, y, z = spherical_to_cartesian(
                    params[cfg["radial"]],
                    params[cfg["polar"]],
                    params[cfg["azimuthal"]],
                )
                params[cfg["x"]] = x
                params[cfg["y"]] = y
                params[cfg["z"]] = z

            elif tname == "mass":
                mchirp, q = mass1_mass2_to_mchirp_q(params["mass1"], params["mass2"])
                params["mchirp"] = mchirp
                params["q"] = q

            elif tname == "distance":
                params["distance"] = chirp_distance_to_distance(
                    params["chirp_distance"], params["mchirp"]
                )

    def _enforce_constraints(self, params, N):

        if not self.constraints:
            return params

        mask = torch.ones(N, dtype=torch.bool, device=self.device)

        for c in self.constraints:
            mask &= c.check(params)

        if mask.all():
            return params

        # resample only bad rows
        bad = ~mask
        new_params = self._sample_base(bad.sum().item())
        self._apply_transforms(new_params)

        for k in params:
            params[k][bad] = new_params[k]

        return params

    def sample(self, N: int):

        params = self._sample_base(N)
        self._apply_transforms(params)
        params = self._enforce_constraints(params, N)

        return params
