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
    cosmology,
    powerlaw,
    sky,
    uniform,
)

from sage.core.math import Normalise
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


def read_from_config(path, seed):
    """
    Construct a :class:`DistributionSampler` from a YAML prior configuration.

    Parameters
    ----------
    path : str
        Path to the YAML file describing the prior distributions, transforms,
        and constraints.
    seed : int
        Seed for the internal :class:`torch.Generator`.

    Returns
    -------
    DistributionSampler
        Configured sampler ready to draw waveform parameters.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    sage_cfg = get_cfg()

    # Create a generator with a specific seed
    gen = torch.Generator(device=sage_cfg.device)
    gen.manual_seed(seed)

    return DistributionSampler(
        config,
        device=sage_cfg.device,
        dtype=sage_cfg.dtype,
        generator=gen,
    )


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
    * **constraints** — named deterministic projections that mutate params in
      place (e.g. ``mass_order`` to enforce m1 ≥ m2).

    After construction, :meth:`_compile_batch_standardiser` must be called
    once to pre-compute mean and standard deviation buffers used by
    :meth:`standardise_from_batch` / :meth:`unstandardise_from_batch`.

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
    normalisers : dict[str, Normalise]
        Pre-built :class:`~sage.core.math.Normalise` objects for each param.
    """

    def __init__(self, config: Dict[str, Any], device, dtype, generator):

        super().__init__()

        # Shared config
        self.sage_cfg = get_cfg()

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
        self.req_idx = None

        self._build_distributions()
        self._build_transforms()
        self._build_constraints()
        self.bounds = self.theoretical_bounds()
        self.normalisers = self.build_normalisers()

    @staticmethod
    def get_named_constraints():
        """Return the list of built-in named constraint identifiers."""
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

            # deterministic projection constraint (the only supported kind: it
            # mutates params in place, e.g. mass ordering). Rejection-style
            # constraints are not implemented in the vectorised fixed-batch flow.
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
                bounds[pcfg["ra"]] = (0.0, 2 * torch.pi)
                bounds[pcfg["dec"]] = (-torch.pi / 2, torch.pi / 2)

            elif name == "uniform_solidangle":
                bounds[pcfg["polar-angle"]] = (0.0, torch.pi)
                bounds[pcfg["azimuthal-angle"]] = (0.0, 2 * torch.pi)

        ## MASS ORDER CONSTRAINT

        if "mass1" in bounds and "mass2" in bounds:
            m1_min, m1_max = bounds["mass1"]
            m2_min, m2_max = bounds["mass2"]

            # enforce m1 >= m2
            bounds["mass1"] = (max(m1_min, m2_min), m1_max)
            bounds["mass2"] = (m2_min, min(m2_max, m1_max))

        ## DERIVED MASS PARAMETERS

        if "mass1" in bounds and "mass2" in bounds:

            m1_min, m1_max = bounds["mass1"]
            m2_min, m2_max = bounds["mass2"]

            # q = m1/m2, with m2 <= m1
            q_min = 1.0
            q_max = m1_max / m2_min
            bounds["q"] = (q_min, q_max)

            def mchirp(m1, m2):
                """Compute chirp mass from component masses."""
                return ((m1 * m2) ** (3.0 / 5.0)) / ((m1 + m2) ** (1.0 / 5.0))

            # Extremes occur on boundary
            candidates = [
                mchirp(m1_min, m2_min),
                mchirp(m1_min, m2_max),
                mchirp(m1_max, m2_min),
                mchirp(m1_max, m2_max),
            ]

            bounds["mchirp"] = (min(candidates), max(candidates))

        ## DISTANCE

        if "chirp_distance" in bounds and "mchirp" in bounds:

            cd_min, cd_max = bounds["chirp_distance"]
            mc_min, mc_max = bounds["mchirp"]

            # distance from chirp distance
            d_min = chirp_distance_to_distance(cd_min, mc_min)
            d_max = chirp_distance_to_distance(cd_max, mc_max)

            bounds["distance"] = (d_min, d_max)

        ## SPIN CARTESIAN COMPONENTS

        for spin in ["spin1", "spin2"]:

            a_name = f"{spin}_a"
            if a_name in bounds:

                a_min, a_max = bounds[a_name]

                # since spherical:
                # x,y,z in [-a, a]
                bounds[f"{spin}x"] = (-a_max, a_max)
                bounds[f"{spin}y"] = (-a_max, a_max)
                bounds[f"{spin}z"] = (-a_max, a_max)

        ## Ensure ordering consistent with param_names

        ordered_bounds = {
            name: bounds[name] for name in self.param_names if name in bounds
        }

        return ordered_bounds

    def build_normalisers(self):
        """
        Construct Normalise objects for all parameters
        using theoretical bounds.

        Returns
        -------
        dict[str, Normalise]
            Mapping parameter name -> Normalise object
        """

        bounds = self.theoretical_bounds()
        normalisers = {}

        for name, (min_val, max_val) in bounds.items():

            if max_val <= min_val:
                raise ValueError(
                    f"Invalid bounds for {name}: " f"({min_val}, {max_val})"
                )

            normalisers[name] = Normalise(
                min_val=min_val,
                max_val=max_val,
            )

        return normalisers

    def _compile_batch_normaliser(self):
        """
        Precompute tensors used for fast batch normalisation,
        adjusted for the sliced parameter subset (self.req_idx).
        """

        selected_names = self.sage_cfg.do_point_estimate

        # Convert to tensor (register as buffer)
        idxs = [self.param_index[key] for key in selected_names]
        indices_tensor = torch.tensor(idxs, dtype=torch.long)

        # Get min/max for selected names
        mins = torch.tensor(
            [self.normalisers[name].min_val for name in selected_names],
            dtype=self.sage_cfg.dtype,
        )
        maxs = torch.tensor(
            [self.normalisers[name].max_val for name in selected_names],
            dtype=self.sage_cfg.dtype,
        )
        scales = maxs - mins

        # Register as buffers (safe and device-aware)
        self.register_buffer("_norm_indices", indices_tensor)
        self.register_buffer("_norm_mins", mins)
        self.register_buffer("_norm_scales", scales)

    def norm_from_batch(self, batch):
        """
        Min-max normalise selected parameters in a full parameter batch.

        Parameters
        ----------
        batch : torch.Tensor, shape ``(B, total_params)``
            Full parameter batch produced by :meth:`forward`.

        Returns
        -------
        torch.Tensor, shape ``(B, selected_params)``
            Selected columns normalised to ``[0, 1]`` using theoretical bounds.
        """
        if batch.ndim != 2:
            raise ValueError("batch must be 2D (B, total_params)")

        selected = batch.index_select(1, self._norm_indices)

        return (selected - self._norm_mins) / self._norm_scales

    def unnorm_from_batch(self, normed_batch):
        """
        Invert min-max normalisation to recover physical parameter values.

        Parameters
        ----------
        normed_batch : torch.Tensor, shape ``(B, selected_params)``
            Normalised batch from :meth:`norm_from_batch`.

        Returns
        -------
        torch.Tensor, shape ``(B, selected_params)``
            Parameters in their original physical units/range.
        """
        if normed_batch.ndim != 2:
            raise ValueError("normed_batch must be 2D")

        return normed_batch * self._norm_scales + self._norm_mins

    def _compile_batch_standardiser(self, N: int = 1_000_000):
        """
        Compute mean and std for standardisation from a large sample of parameters
        and register them as buffers for fast batch standardisation.

        Parameters
        ----------
        N : int
            Number of samples to use to estimate mean and stddev
            Default is set to 1_000_000
        """

        # Sample directly
        all_samples = self._sample_base(N)  # shape (N, total_params)
        all_samples = self._enforce_constraints(all_samples)
        self._apply_transforms(all_samples)

        selected_names = self.sage_cfg.do_point_estimate
        idxs = [self.param_index[key] for key in selected_names]
        indices_tensor = torch.tensor(idxs, dtype=torch.long)

        # Select only the parameters we care about
        selected_samples = all_samples[:, idxs]

        # Compute mean and stddev across samples (dim=0)
        means = selected_samples.mean(dim=0)
        stds = selected_samples.std(dim=0)

        # Avoid division by zero
        stds[stds == 0.0] = 1.0

        # Register as buffers (device-aware)
        self.register_buffer("_std_indices", indices_tensor)
        self.register_buffer("_std_means", means)
        self.register_buffer("_std_stds", stds)

    def standardise_from_batch(self, batch: torch.Tensor):
        """
        Standardise batch using precomputed mean/std buffers.

        Parameters
        ----------
        batch : torch.Tensor
            Shape (B, total_params)
        Returns
        -------
        torch.Tensor
            Standardised batch (B, selected_params)
        """
        if batch.ndim != 2:
            raise ValueError("batch must be 2D (B, total_params)")

        selected = batch.index_select(1, self._std_indices)
        return (selected - self._std_means) / self._std_stds

    def unstandardise_from_batch(self, standardised_batch: torch.Tensor):
        """
        Convert standardised batch back to original parameter scale.

        Parameters
        ----------
        standardised_batch : torch.Tensor
            Shape (B, selected_params)
        Returns
        -------
        torch.Tensor
            Unstandardised batch
        """
        if standardised_batch.ndim != 2:
            raise ValueError("standardised_batch must be 2D")

        return standardised_batch * self._std_stds + self._std_means

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
        params = self._sample_base(N)
        params = self._enforce_constraints(params)
        # NOTE: Just so I remove panic for future me
        # Tensors are mutable and passed by reference
        # So we don't need to return params from appply_transforms
        self._apply_transforms(params)

        return params
