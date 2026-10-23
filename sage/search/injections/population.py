"""Draw binary black hole parameters from the GWTC-3 Power-Law + Peak model.

Ported from sgwc-1 gwtc3_analyses/cpu_pp_intrinsic_params.py, with each
mass ratio from its own conditional and spin densities marginalised as in
gwpopulation_pipe; hyperposterior marginalisation from thyme.
"""

import math
from typing import Dict, Tuple

import gwpopulation
import numpy as np
import pandas as pd


def get_p_m1(hyperparameters: pd.Series, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed Power-Law + Peak primary-mass density on a grid.

    Parameters
    ----------
    hyperparameters : pandas.Series
        One hyperposterior sample.
    n : int
        Grid points.

    Returns
    -------
    density, masses : numpy.ndarray
        Normalised density and its mass grid, up to 100 solar masses.
    """
    alpha = hyperparameters["alpha"]
    mmin = hyperparameters["mmin"]
    mmax = hyperparameters["mmax"]
    lam = hyperparameters["lam"]
    mpp = hyperparameters["mpp"]
    sigpp = hyperparameters["sigpp"]
    delta_m = hyperparameters["delta_m"]

    masses = np.linspace(mmin, 100.0, n)
    qs = np.linspace(0.0, 1.0, n)
    param_dict = {"mass_1": masses, "mass_ratio": qs}

    mass_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    smoothing = mass_model.smoothing(masses, mmin=mmin, mmax=mmax, delta_m=delta_m)
    p_m1 = mass_model.p_m1(
        param_dict, alpha=alpha, mmin=mmin, mmax=mmax, lam=lam, mpp=mpp, sigpp=sigpp
    )

    smooth_p_m1 = smoothing * p_m1
    return smooth_p_m1 / np.trapezoid(smooth_p_m1, masses), masses


def get_p_q(
    masses: np.ndarray, hyperparameters: pd.Series, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Mass-ratio density conditional on each primary mass.

    Parameters
    ----------
    masses : numpy.ndarray
        Primary masses, shape ``(N,)``.
    hyperparameters : pandas.Series
        One hyperposterior sample.
    n : int
        Grid points in mass ratio.

    Returns
    -------
    density : numpy.ndarray
        Normalised density of each row, shape ``(N, n)``.
    qs : numpy.ndarray
        Mass-ratio grid, shape ``(n,)``.
    """
    qs = np.linspace(0.0, 1.0, n)
    m1_grid, q_grid = np.meshgrid(masses, qs, indexing="ij")

    q_model = gwpopulation.models.mass.SinglePeakSmoothedMassDistribution()
    p_q = q_model.p_q(
        {"mass_1": m1_grid.ravel(), "mass_ratio": q_grid.ravel()},
        hyperparameters["beta"],
        hyperparameters["mmin"],
        hyperparameters["delta_m"],
    ).reshape(len(masses), n)
    return p_q / np.trapezoid(p_q, qs, axis=-1)[:, None], qs


def get_p_z(
    hyperparameters: pd.Series, z_max: float, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Power-law merger-rate redshift density on a grid.

    Parameters
    ----------
    hyperparameters : pandas.Series
        One hyperposterior sample.
    z_max : float
        Largest redshift drawn.
    n : int
        Grid points.

    Returns
    -------
    density, z : numpy.ndarray
        Normalised density and its redshift grid.
    """
    z = np.linspace(0.0, z_max, n)
    z_model = gwpopulation.models.redshift.PowerLawRedshift()
    p_z = z_model.probability({"redshift": z}, lamb=hyperparameters["lamb"])
    return p_z / np.trapezoid(p_z, z), z


def get_p_chi(hyperparameters: pd.Series, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Spin-magnitude density of one black hole on a grid.

    Parameters
    ----------
    hyperparameters : pandas.Series
        One hyperposterior sample.
    n : int
        Grid points.

    Returns
    -------
    density, chi : numpy.ndarray
        Normalised density and its spin-magnitude grid.
    """
    chi = np.linspace(0.0, 1.0, n)
    alpha_chi, beta_chi, amax = gwpopulation.conversions.mu_var_max_to_alpha_beta_max(
        hyperparameters["mu_chi"], hyperparameters["sigma_chi"], hyperparameters["amax"]
    )

    # Marginal over the second spin, as in gwpopulation_pipe.
    a_1, a_2 = np.meshgrid(chi, chi)
    joint = gwpopulation.models.spin.iid_spin_magnitude_beta(
        {"a_1": a_1, "a_2": a_2}, amax=amax, alpha_chi=alpha_chi, beta_chi=beta_chi
    )
    p_chi = np.trapezoid(joint, chi, axis=0)
    return p_chi / np.trapezoid(p_chi, chi), chi


def get_p_costilt(
    hyperparameters: pd.Series, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Spin-tilt density of one black hole on a grid.

    Parameters
    ----------
    hyperparameters : pandas.Series
        One hyperposterior sample.
    n : int
        Grid points.

    Returns
    -------
    density, cos_tilt : numpy.ndarray
        Normalised density and its cosine-tilt grid.
    """
    cos_tilt = np.linspace(-1.0, 1.0, n)

    # Marginal over the second tilt, as in gwpopulation_pipe.
    cos_tilt_1, cos_tilt_2 = np.meshgrid(cos_tilt, cos_tilt)
    joint = gwpopulation.models.spin.iid_spin_orientation_gaussian_isotropic(
        {"cos_tilt_1": cos_tilt_1, "cos_tilt_2": cos_tilt_2},
        hyperparameters["xi_spin"],
        hyperparameters["sigma_spin"],
    )
    p_costilt = np.trapezoid(joint, cos_tilt, axis=0)
    return p_costilt / np.trapezoid(p_costilt, cos_tilt), cos_tilt


def cdf(distribution: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral of a density along its last axis.

    Parameters
    ----------
    distribution : numpy.ndarray
        Density values on the grid, one row per density.
    theta : numpy.ndarray
        Ascending grid.

    Returns
    -------
    numpy.ndarray
        Cumulative integral, starting at zero.
    """
    steps = np.diff(theta) * (distribution[..., 1:] + distribution[..., :-1]) / 2.0
    zero = np.zeros_like(steps[..., :1])
    return np.concatenate([zero, np.cumsum(steps, axis=-1)], axis=-1)


def sample_1d(
    distribution: np.ndarray,
    theta: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inverse-CDF samples from a density on a grid.

    Parameters
    ----------
    distribution : numpy.ndarray
        Density values on the grid.
    theta : numpy.ndarray
        Ascending grid.
    n_samples : int
        Number of samples.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    numpy.ndarray
        Samples, shape ``(n_samples,)``.
    """
    return np.interp(rng.random(n_samples), cdf(distribution, theta), theta)


def sample_intrinsic(
    hyperparameters: pd.Series,
    n_samples: int,
    z_max: float,
    rng: np.random.Generator,
    n_grid: int = 1000,
) -> Dict[str, np.ndarray]:
    """Draw masses, redshift and spins for one hyperposterior sample.

    Parameters
    ----------
    hyperparameters : pandas.Series
        One hyperposterior sample.
    n_samples : int
        Number of binaries.
    z_max : float
        Largest redshift drawn.
    rng : numpy.random.Generator
        Random generator.
    n_grid : int
        Grid points of each density.

    Returns
    -------
    dict of str to numpy.ndarray
        ``m1``, ``q``, ``z``, ``chi_1``, ``chi_2``, ``costilt_1`` and
        ``costilt_2`` of each binary.
    """
    p_m1, masses = get_p_m1(hyperparameters, n_grid)
    m1 = sample_1d(p_m1, masses, n_samples, rng)

    # Each binary's mass ratio is drawn from its own conditional density.
    p_q, qs = get_p_q(m1, hyperparameters, n_grid)
    uniform = rng.random(n_samples)
    q = np.array([np.interp(u, row, qs) for u, row in zip(uniform, cdf(p_q, qs))])

    p_z, zs = get_p_z(hyperparameters, z_max, n_grid)
    p_chi, chis = get_p_chi(hyperparameters, n_grid)
    p_costilt, costilts = get_p_costilt(hyperparameters, n_grid)
    return {
        "m1": m1,
        "q": q,
        "z": sample_1d(p_z, zs, n_samples, rng),
        "chi_1": sample_1d(p_chi, chis, n_samples, rng),
        "chi_2": sample_1d(p_chi, chis, n_samples, rng),
        "costilt_1": sample_1d(p_costilt, costilts, n_samples, rng),
        "costilt_2": sample_1d(p_costilt, costilts, n_samples, rng),
    }


def plan_marginalisation(n_samples: int, n_available: int) -> Tuple[int, int]:
    """Hyperposterior samples to draw from and binaries under each (thyme).

    Parameters
    ----------
    n_samples : int
        Binaries wanted.
    n_available : int
        Hyperposterior samples available.

    Returns
    -------
    n_hyper, n_per_hyper : int
        As many distinct hyperposterior samples as possible, and the fewest
        binaries per sample that reach ``n_samples``.
    """
    n_hyper = min(n_samples, n_available)
    return n_hyper, math.ceil(n_samples / n_hyper)


def sample_marginalised(
    hyperposterior: pd.DataFrame,
    n_samples: int,
    z_max: float,
    seed: int,
    n_grid: int = 1000,
) -> Dict[str, np.ndarray]:
    """Draw binaries marginalised over the hyperposterior, as in thyme.

    Each block of binaries comes from a different hyperposterior sample;
    the draws are shuffled before keeping ``n_samples``.

    Parameters
    ----------
    hyperposterior : pandas.DataFrame
        Hyperposterior samples, one per row.
    n_samples : int
        Number of binaries.
    z_max : float
        Largest redshift drawn.
    seed : int
        Seed of the sample choice, draws and shuffle.
    n_grid : int
        Grid points of each density.

    Returns
    -------
    dict of str to numpy.ndarray
        Output of ``sample_intrinsic`` for ``n_samples`` binaries.
    """
    n_hyper, n_per_hyper = plan_marginalisation(n_samples, len(hyperposterior))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(hyperposterior), size=n_hyper, replace=False)

    blocks = [
        sample_intrinsic(hyperposterior.iloc[index], n_per_hyper, z_max, rng, n_grid)
        for index in chosen
    ]
    order = rng.permutation(n_hyper * n_per_hyper)[:n_samples]
    return {
        name: np.concatenate([block[name] for block in blocks])[order]
        for name in blocks[0]
    }
