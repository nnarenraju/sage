"""Waveform parameters of injections, as in sgwc-1 injection_study.

The population gives masses, spins and distance; the training prior gives
sky position, orientation and tc. Masses are detector frame (SB-50).
"""

from pathlib import Path
from typing import Dict

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15
from astropy.cosmology import units as cu
from pycbc.conversions import mchirp_from_mass1_mass2

from sage.data.waveform import read_from_config


def injection_parameters(
    population: Dict[str, np.ndarray],
    gwconfig: Path,
    gps_start: float,
    gps_end: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """PyCBC-named waveform parameters of each binary.

    Parameters
    ----------
    population : dict of str to numpy.ndarray
        Output of ``sample_marginalised``.
    gwconfig : Path
        Parameter prior the network was trained under.
    gps_start, gps_end : float
        Span of the searched data; each injection's projection time is
        drawn uniformly within it.
    seed : int
        Seed of the spin azimuths, projection times and prior draws.

    Returns
    -------
    dict of str to numpy.ndarray
        Masses, spins, distance, sky position, orientation, ``tc`` and
        ``injection_time``.
    """
    n = len(population["m1"])
    mass1 = population["m1"] * (1.0 + population["z"])

    rng = np.random.default_rng(seed)
    phi_1 = rng.uniform(0.0, 2.0 * np.pi, n)
    phi_2 = rng.uniform(0.0, 2.0 * np.pi, n)
    injection_time = rng.uniform(gps_start, gps_end, n)
    sintilt_1 = np.sqrt(1 - population["costilt_1"] ** 2)
    sintilt_2 = np.sqrt(1 - population["costilt_2"] ** 2)

    z = population["z"] * cu.redshift
    distance = z.to(u.Mpc, cu.redshift_distance(Planck15, kind="luminosity"))
    parameters = {
        "mass1": mass1,
        "mass2": population["q"] * mass1,
        "spin1x": population["chi_1"] * sintilt_1 * np.cos(phi_1),
        "spin1y": population["chi_1"] * sintilt_1 * np.sin(phi_1),
        "spin1z": population["chi_1"] * population["costilt_1"],
        "spin2x": population["chi_2"] * sintilt_2 * np.cos(phi_2),
        "spin2y": population["chi_2"] * sintilt_2 * np.sin(phi_2),
        "spin2z": population["chi_2"] * population["costilt_2"],
        "distance": distance.value,
        "injection_time": injection_time,
    }

    prior = read_from_config(str(gwconfig), seed=seed)
    draws = prior(n).cpu().numpy()
    for name in ("ra", "dec", "inclination", "coa_phase", "polarization", "tc"):
        parameters[name] = draws[:, prior.param_index[name]].astype(np.float64)
    return parameters


def in_training_prior(
    parameters: Dict[str, np.ndarray], mass_low: float, mass_high: float
) -> np.ndarray:
    """Mask of binaries whose chirp mass lies within the training prior.

    The bounds are the chirp masses of equal-mass binaries at the ends of
    the training mass prior, as in sgwc-1.

    Parameters
    ----------
    parameters : dict of str to numpy.ndarray
        Output of ``injection_parameters``.
    mass_low, mass_high : float
        Bounds of the training ``mass1`` prior.

    Returns
    -------
    numpy.ndarray
        True for each binary inside the trained chirp-mass range.
    """
    mchirp = mchirp_from_mass1_mass2(parameters["mass1"], parameters["mass2"])
    low = mchirp_from_mass1_mass2(mass_low, mass_low)
    high = mchirp_from_mass1_mass2(mass_high, mass_high)
    return (mchirp >= low) & (mchirp <= high)
