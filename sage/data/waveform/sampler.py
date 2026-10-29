#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : distributions.py
Description     : Short description of the file

Created on 2026-02-16 10:35:39

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation:

    sampler = prior_sampler(load_prior("bbh_narrow", run_dir), seed, "cuda", torch.float32)

    batch = sampler.sample(4096)

    print(batch["mass1"].shape)      # (4096,)
    print(batch["spin1x"].shape)     # derived param
    print(batch["distance"].shape)   # transformed param

"""

# Packages
import math

import torch

from typing import Any, Dict

# LOCAL
from sage.data.waveform.distributions import (
    angular,
    cosmology,
    powerlaw,
    sky,
    uniform,
)


# Conversions
from sage.core.conversions import distance_from_chirp_distance_mchirp, mchirp_from_mass1_mass2
from sage.data.waveform.conversions import mass1_mass2_to_mchirp_q

# Transformation constraints
import sage.data.waveform.constraints as constraints

def spherical_to_cartesian(radial, polar, azimuthal):
    """
    Convert spherical spin components to Cartesian coordinates.

    Parameters
    ----------
    radial : torch.Tensor
        Spin magnitude.
    polar : torch.Tensor
        Polar angle (inclination) in radians.
    azimuthal : torch.Tensor
        Azimuthal angle in radians.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(x, y, z)`` Cartesian spin components.
    """
    sin_theta = torch.sin(polar)
    return (
        radial * sin_theta * torch.cos(azimuthal),
        radial * sin_theta * torch.sin(azimuthal),
        radial * torch.cos(polar),
    )


def prior_sampler(prior: Dict[str, Any], seed: int, device, dtype) -> "DistributionSampler":
    """
    Build a seeded sampler for a checked prior.

    Parameters
    ----------
    prior : dict
        Checked prior, from :mod:`sage.config.prior`.
    seed : int
        Seed of the sampler's own generator.
    device : str or torch.device
        Device to sample on.
    dtype : torch.dtype
        Dtype of the samples.

    Returns
    -------
    DistributionSampler
        Sampler ready to draw waveform parameters.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return DistributionSampler(prior, device=device, dtype=dtype, generator=generator)


class NamedConstraint:
    """
    Reference to a named constraint function defined in
    :mod:`sage.data.waveform.constraints`.

    Parameters
    ----------
    name : str
        Name of the constraint function (must appear in
        ``constraints._NAMED_CONSTRAINTS``).
    params : list or None
        Additional parameter names passed to the constraint function.
    """

    def __init__(self, name, params=None):
        self.name = name
        self.params = params or []


class DistributionSampler(torch.nn.Module):
    """
    Batched waveform-parameter sampler driven by a YAML prior configuration.

    Reads a dictionary that describes:

    * **priors** — per-parameter distributions (uniform, sin-angle,
      solid-angle, sky, uniform-radius, …).
    * **waveform_transforms** — deterministic reparametrisations applied
      after sampling (spherical → Cartesian spins, m1/m2 → mchirp/q,
      chirp-distance → luminosity distance).
    * **constraints** — named masks; samples outside them are rejected and
      drawn again (e.g. ``mass_order`` for m1 ≥ m2).

    After construction, :meth:`compile_target_scaling` must be called once
    to set the target scaling used by :meth:`scale_targets` /
    :meth:`unscale_targets`.

    Parameters
    ----------
    config : dict
        Parsed YAML dictionary with keys ``"priors"``,
        ``"waveform_transforms"`` (optional), and ``"constraints"`` (optional).
    device : str
        Torch device for all sampled tensors.
    dtype : torch.dtype
        Floating-point dtype for all sampled tensors.
    generator : torch.Generator
        Seeded generator used for all random draws.

    Attributes
    ----------
    param_names : list[str]
        Sorted list of all parameter names produced by this sampler
        (includes derived quantities from transforms).
    param_index : dict[str, int]
        Mapping from parameter name to column index in the output tensor.
    num_params : int
        Total number of parameters in the output batch.
    bounds : dict[str, tuple]
        Theoretical ``(min, max)`` bounds for every parameter.
    """

    def __init__(self, config: Dict[str, Any], device, dtype, generator):

        super().__init__()

        # Generator
        self.generator = generator

        self.cfg = config
        self.device = device
        self.dtype = dtype

        self.variable_params = config["variable_params"]

        self.distributions = {}
        self._norm_cache = {}
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
                self.param_names.append(pcfg["polar-angle"])
                self.param_names.append(pcfg["azimuthal-angle"])

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
        self.bounds = self.theoretical_bounds()
        for name, (low, high) in self.bounds.items():
            if high <= low:
                raise ValueError(f"Invalid bounds for {name}: ({low}, {high})")

    @staticmethod
    def get_named_constraints():
        """Return the list of built-in named constraint identifiers."""
        return constraints._NAMED_CONSTRAINTS

    def _make_dist(self, name, args):
        if name == "uniform":
            return uniform.Uniform(args["min"], args["max"])
        if name == "uniform_angle":
            return angular.UniformAngle()
        if name == "sin_angle":
            return angular.SinAngle()
        if name == "uniform_sky":
            return sky.UniformSky(args["polar-angle"], args["azimuthal-angle"])
        if name == "uniform_solidangle":
            return angular.UniformSolidAngle(
                args["polar-angle"], args["azimuthal-angle"]
            )
        if name == "uniform_radius":
            return powerlaw.UniformRadius(args["min"], args["max"])
        if name == "uniform_comoving_volume":
            return cosmology.UniformComovingVolume(
                args["min"],
                args["max"],
                cosmology=args.get("cosmology", "Planck18"),
            )
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

            # a named constraint is a mask; samples outside it are rejected (_sample_accepted)
            if c["name"] in constraints._NAMED_CONSTRAINTS:
                self.constraints.append(NamedConstraint(c["name"], c.get("params")))

            else:
                raise ValueError(
                    f"Unknown constraint type '{c['name']}'. "
                    f"Available named: {constraints._NAMED_CONSTRAINTS}"
                )

    def _sample_base(self, N):
        """
        Draw ``N`` raw parameter samples from all priors.

        Parameters
        ----------
        N : int
            Number of samples to draw.

        Returns
        -------
        torch.Tensor, shape ``(N, num_params)``
            Raw samples before transforms and constraints.
        """
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
                generator=self.generator,
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

                d = distance_from_chirp_distance_mchirp(cd, mc)

                params[:, self.param_index["distance"]] = d

    def _within_constraints(self, params):
        """Mask of the samples that satisfy every constraint."""
        keep = torch.ones(params.shape[0], dtype=torch.bool, device=params.device)
        for c in self.constraints:
            keep &= getattr(constraints, c.name)(params, self.param_index, c.params)
        return keep

    def _sample_accepted(self, N):
        """
        ``N`` raw samples within every constraint.

        Rejection sampling as ``pycbc.distributions.joint.JointDistribution.rvs``:
        each round draws ``ndraw`` samples from every prior and keeps those
        within the constraints, in order.
        """
        if not self.constraints:
            return self._sample_base(N)
        out = torch.empty(N, self.num_params, device=self.device, dtype=self.dtype)
        remaining = ndraw = N
        while remaining:
            scratch = self._sample_base(ndraw)
            keep = self._within_constraints(scratch)
            nkeep = int(keep.sum())
            kmin = N - remaining
            kmax = min(nkeep, remaining)
            out[kmin:kmin + kmax] = scratch[keep][:kmax]
            remaining = max(0, remaining - nkeep)
            # to try to speed up next go around, we'll increase the draw
            # size by the fraction of values that were kept, but cap at 1e6
            ndraw = int(min(1e6, ndraw * math.ceil(ndraw / (nkeep + 1.0))))
        return out

    def theoretical_bounds(self):
        """
        Compute analytic lower/upper bounds for all parameters
        based on YAML priors + constraints + deterministic transforms.
        """

        bounds = {}

        priors = self.cfg["priors"]

        ## BASE PRIORS

        for pname, pcfg in priors.items():

            name = pcfg["name"]

            if name == "uniform":
                bounds[pname] = (pcfg["min"], pcfg["max"])

            elif name == "uniform_radius":
                bounds[pname] = (pcfg["min"], pcfg["max"])

            elif name == "uniform_comoving_volume":
                bounds[pname] = (pcfg["min"], pcfg["max"])

            elif name == "uniform_angle":
                bounds[pname] = (0.0, 2 * torch.pi)

            elif name == "sin_angle":
                bounds[pname] = (0.0, torch.pi)

            elif name == "uniform_sky":
                bounds[pcfg["polar-angle"]] = (-torch.pi / 2, torch.pi / 2)
                bounds[pcfg["azimuthal-angle"]] = (0.0, 2 * torch.pi)

            elif name == "uniform_solidangle":
                bounds[pcfg["polar-angle"]] = (0.0, torch.pi)
                bounds[pcfg["azimuthal-angle"]] = (0.0, 2 * torch.pi)

        ## MASS ORDER CONSTRAINT (configured only)

        constrained = {c["name"] for c in self.cfg.get("constraints", [])}
        transforms = list(self.cfg.get("waveform_transforms", {}).values())
        if "mass_order" in constrained:
            m1_min, m1_max = bounds["mass1"]
            m2_min, m2_max = bounds["mass2"]

            # enforce m1 >= m2
            bounds["mass1"] = (max(m1_min, m2_min), m1_max)
            bounds["mass2"] = (m2_min, min(m2_max, m1_max))

        ## DERIVED MASS PARAMETERS (the mass transform)

        if any(t["name"] == "mass1_mass2_to_mchirp_q" for t in transforms):

            m1_min, m1_max = bounds["mass1"]
            m2_min, m2_max = bounds["mass2"]

            # q = m1/m2; with m2 <= m1 it is at least 1
            q_min = m1_min / m2_max
            bounds["q"] = (max(1.0, q_min) if "mass_order" in constrained else q_min, m1_max / m2_min)

            # Extremes occur on boundary
            candidates = [
                mchirp_from_mass1_mass2(m1_min, m2_min),
                mchirp_from_mass1_mass2(m1_min, m2_max),
                mchirp_from_mass1_mass2(m1_max, m2_min),
                mchirp_from_mass1_mass2(m1_max, m2_max),
            ]

            bounds["mchirp"] = (min(candidates), max(candidates))

        ## DISTANCE (the distance transform)

        if any(t["name"] == "chirp_distance_to_distance" for t in transforms):

            cd_min, cd_max = bounds["chirp_distance"]
            mc_min, mc_max = bounds["mchirp"]

            # distance from chirp distance
            d_min = distance_from_chirp_distance_mchirp(cd_min, mc_min)
            d_max = distance_from_chirp_distance_mchirp(cd_max, mc_max)

            bounds["distance"] = (d_min, d_max)

        ## SPIN CARTESIAN COMPONENTS (each transform's own keys)

        for tcfg in transforms:
            if tcfg["name"] == "spherical_to_cartesian":
                a_max = bounds[tcfg["radial"]][1]
                # since spherical: x, y, z in [-a, a]
                for key in ("x", "y", "z"):
                    bounds[tcfg[key]] = (-a_max, a_max)

        ## Ensure ordering consistent with param_names

        ordered_bounds = {
            name: bounds[name] for name in self.param_names if name in bounds
        }

        return ordered_bounds

    def compile_target_scaling(self, targets, dtype):
        """
        Scale ``targets`` to [-1, 1] from their bounds: ``(x - offset) / scale``,
        offset the middle of the bounds and scale half their width.

        Parameters
        ----------
        targets : list of str
            Estimated parameters, in head order.
        dtype : torch.dtype
            Dtype of the buffers.

        Raises
        ------
        ValueError
            If a target has no bounds.
        """
        missing = [target for target in targets if target not in self.bounds]
        if missing:
            raise ValueError(f"targets {missing} have no bounds from the prior and its transforms")
        lows, highs = zip(*(self.bounds[target] for target in targets))
        self.register_buffer("_target_indices", torch.tensor([self.param_index[t] for t in targets], dtype=torch.long))
        self.register_buffer("_target_offsets", torch.tensor([(a + b) / 2 for a, b in zip(lows, highs)], dtype=dtype))
        self.register_buffer("_target_scales", torch.tensor([(b - a) / 2 for a, b in zip(lows, highs)], dtype=dtype))

    def scale_targets(self, batch: torch.Tensor):
        """The targets of a ``(B, num_params)`` batch, scaled to [-1, 1] (``compile_target_scaling``)."""
        return (batch.index_select(1, self._target_indices) - self._target_offsets) / self._target_scales

    def unscale_targets(self, scaled: torch.Tensor):
        """Physical values of a ``(B, targets)`` batch of scaled targets."""
        return scaled * self._target_scales + self._target_offsets

    def forward(self, N: int):
        """
        Sample a batch of ``N`` waveform parameters from the configured prior.

        Applies constraints (rejection sampling) and deterministic transforms
        (e.g. mchirp/q, Cartesian spins, luminosity distance) in order.

        Parameters
        ----------
        N : int
            Number of samples to draw.

        Returns
        -------
        torch.Tensor, shape ``(N, num_params)``
            Complete parameter batch with all transforms applied.
        """
        params = self._sample_accepted(N)
        # NOTE: Just so I remove panic for future me
        # Tensors are mutable and passed by reference
        # So we don't need to return params from appply_transforms
        self._apply_transforms(params)

        return params
