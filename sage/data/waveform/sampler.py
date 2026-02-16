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

# LOCAL
from sage.data.waveform.distributions import (
    angular,
    powerlaw,
    sky,
    uniform,
)

# Conversions
from sage.data.waveform.conversions import (
    mass1_mass2_to_mchirp_q,
    chirp_distance_to_distance,
)

# Transformation constraints
import sage.data.waveform.constraints as constraints

_NAMED_CONSTRAINTS = ["mass_order"]


def spherical_to_cartesian(radial, polar, azimuthal):
    sin_theta = torch.sin(polar)
    return (
        radial * sin_theta * torch.cos(azimuthal),
        radial * sin_theta * torch.sin(azimuthal),
        radial * torch.cos(polar),
    )


def read_from_config(path, device="cuda"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return DistributionSampler(config, device=device)


class ConstraintChecker:
    def __init__(self, expr: str):
        self.expr = expr

    def check(self, params: Dict[str, torch.Tensor]):
        return eval(self.expr, {}, params)


class NamedConstraint:
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or []


class ExpressionConstraint:
    def __init__(self, expr: str):
        self.name = "custom"
        self.expr = expr

    def check(self, params):
        return eval(self.expr, {}, params)


class DistributionSampler:

    def __init__(self, config: Dict[str, Any], device="cuda"):
        self.device = device
        self.cfg = config

        self.variable_params = config["variable_params"]

        self.distributions = {}
        self.transforms = []
        self.constraints = []

        self._build_distributions()
        self._build_transforms()
        self._build_constraints()

    @staticmethod
    def get_named_constraints():
        return _NAMED_CONSTRAINTS

    def _make_dist(self, name, args):
        if name == "uniform":
            return uniform.Uniform(args["min"], args["max"])
        if name == "uniform_angle":
            return angular.UniformAngle()
        if name == "sin_angle":
            return angular.SinAngle()
        if name == "uniform_sky":
            return sky.UniformSky()
        if name == "uniform_solidangle":
            return angular.UniformSolidAngle(
                args["polar-angle"], args["azimuthal-angle"]
            )
        if name == "uniform_radius":
            return powerlaw.UniformRadius(args["min"], args["max"])
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
        self.constraints = []

        for c in self.cfg.get("constraints", []):

            # deterministic projection constraint
            if c["name"] in constraints._NAMED_CONSTRAINTS:
                self.constraints.append(NamedConstraint(c["name"], c.get("params")))

            # rejection constraint
            elif c["name"] == "custom":
                self.constraints.append(ExpressionConstraint(c["expr"]))

            else:
                raise ValueError(
                    f"Unknown constraint type '{c['name']}'. "
                    f"Available named: {constraints._NAMED_CONSTRAINTS} or 'custom'"
                )

    def _sample_base(self, N):
        params = {}

        for name, dist in self.distributions.items():
            sampled = dist.sample((N,), self.device)

            # eg. if the distribution is a solid-angle type, it returns a dict
            # It should add polar/azimuthal keys as an update
            if isinstance(sampled, dict):
                params.update(sampled)
            else:
                params[name] = sampled

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

        # Apply named deterministic constraints (projections)
        for c in self.constraints:
            if c.name in constraints._NAMED_CONSTRAINTS:
                params = getattr(constraints, c.name)(params)
            else:
                raise ValueError(
                    f"Unknown named constraint '{c.name}'. "
                    f"Available: {constraints._NAMED_CONSTRAINTS}"
                )

        # Collect only boolean constraints
        bool_constraints = [
            c for c in self.constraints if c.name not in _NAMED_CONSTRAINTS
        ]

        if not bool_constraints:
            return params

        # Partial resampling loop
        while True:

            mask = torch.ones(N, dtype=torch.bool, device=self.device)

            for c in bool_constraints:
                mask &= c.check(params)

            if mask.all():
                return params

            # resample only failed rows
            bad = ~mask
            n_bad = bad.sum().item()

            new_params = self._sample_base(n_bad)
            self._apply_transforms(new_params)

            # deterministic constraints must ALSO apply to resampled values
            for c in self.constraints:
                if c.name in _NAMED_CONSTRAINTS:
                    transform = globals()[c.name]
                    new_params = transform(new_params)

            for k in params:
                params[k][bad] = new_params[k]

    def sample(self, N: int):

        params = self._sample_base(N)
        params = self._enforce_constraints(params, N)
        self._apply_transforms(params)

        return params
