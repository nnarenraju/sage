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

from typing import Any, Dict

# LOCAL
from sage.data.waveform.distributions import (
    angular,
    powerlaw,
    sky,
    uniform,
)

from sage.core.config import get_cfg, get_data_cfg

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


def read_from_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    sage_cfg = get_cfg()

    return DistributionSampler(
        config,
        device=sage_cfg.device,
        dtype=sage_cfg.dtype,
    )


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

    def __init__(self, config: Dict[str, Any], device, dtype):
        self.cfg = config
        self.device = device
        self.dtype = dtype

        self.variable_params = config["variable_params"]

        self.distributions = {}
        self.transforms = []
        self.constraints = []

        ## Fix parameters
        self.param_names = []

        # Priors
        for pname, pcfg in self.cfg["priors"].items():

            if pcfg["name"] == "uniform_solidangle":
                self.param_names.append(pcfg["polar-angle"])
                self.param_names.append(pcfg["azimuthal-angle"])

            elif pcfg["name"] == "uniform_sky":
                self.param_names.append(pcfg["ra"])
                self.param_names.append(pcfg["dec"])

            else:
                self.param_names.append(pname)

        # Transforms
        for _, tcfg in self.cfg.get("waveform_transforms", {}).items():

            name = tcfg["name"]

            if name == "spherical_to_cartesian":
                self.param_names.extend([tcfg["x"], tcfg["y"], tcfg["z"]])

            elif name == "mass1_mass2_to_mchirp_q":
                self.param_names.extend(["mchirp", "q"])

            elif name == "chirp_distance_to_distance":
                self.param_names.append("distance")

        self.param_names = sorted(self.param_names)
        self.param_index = {name: i for i, name in enumerate(self.param_names)}
        self.num_params = len(self.param_names)

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
        params = torch.empty(
            N,
            self.num_params,
            device=self.device,
            dtype=self.dtype,
        )

        for name, dist in self.distributions.items():
            sampled = dist.sample(
                (N,),
                device=self.device,
                dtype=self.dtype,
            )

            if isinstance(sampled, dict):
                for sub_name, value in sampled.items():
                    idx = self.param_index[sub_name]
                    params[:, idx] = value
            else:
                idx = self.param_index[name]
                params[:, idx] = sampled

        return params

    def _apply_transforms(self, params):

        for tname, cfg in self.transforms:
            if tname == "spin_cartesian":
                r_idx = self.param_index[cfg["radial"]]
                p_idx = self.param_index[cfg["polar"]]
                a_idx = self.param_index[cfg["azimuthal"]]

                x, y, z = spherical_to_cartesian(
                    params[:, r_idx],
                    params[:, p_idx],
                    params[:, a_idx],
                )

                params[:, self.param_index[cfg["x"]]] = x
                params[:, self.param_index[cfg["y"]]] = y
                params[:, self.param_index[cfg["z"]]] = z

            elif tname == "mass":
                m1 = params[:, self.param_index["mass1"]]
                m2 = params[:, self.param_index["mass2"]]

                mchirp, q = mass1_mass2_to_mchirp_q(m1, m2)

                params[:, self.param_index["mchirp"]] = mchirp
                params[:, self.param_index["q"]] = q

            elif tname == "distance":
                cd = params[:, self.param_index["chirp_distance"]]
                mc = params[:, self.param_index["mchirp"]]

                d = chirp_distance_to_distance(cd, mc)

                params[:, self.param_index["distance"]] = d

    def _enforce_constraints(self, params):

        if not self.constraints:
            return params

        for c in self.constraints:

            if c.name in constraints._NAMED_CONSTRAINTS:
                fn = getattr(constraints, c.name)

                params = fn(
                    params,
                    self.param_index,
                    c.params,  # optional extra config
                )

            else:
                raise ValueError(
                    f"Unknown named constraint '{c.name}'. "
                    f"Available: {constraints._NAMED_CONSTRAINTS}"
                )

        return params

    def sample(self, N: int):

        params = self._sample_base(N)
        params = self._enforce_constraints(params)
        self._apply_transforms(params)

        return params
