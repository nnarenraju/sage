#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : conversions.py
Description     : Short description of the file

Created on 2026-01-20 16:26:05

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import torch


def seconds_to_samples(nseconds, sample_rate, approx_mode=int, rounding=True):
    if rounding:
        # No need to change the base for rounding
        return approx_mode(round(nseconds * sample_rate))
    else:
        return approx_mode(nseconds * sample_rate)


def samples_to_seconds(nsamples, sample_rate):
    return nsamples / sample_rate


def mchirp_eta_to_mass1_mass2(mchirp: torch.Tensor, eta: torch.Tensor):
    """
    Convert chirp mass and symmetric mass ratio to individual component masses.

    Args:
        mchirp (torch.Tensor): Chirp mass of the binary (any units)
        eta (torch.Tensor): Symmetric mass ratio (dimensionless, 0 < eta <= 0.25)

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - m1 : Mass of the heavier component
            - m2 : Mass of the lighter component

    Notes:
        - The returned masses satisfy m1 >= m2.
        - Component masses are in the same units as the input chirp mass.
    """
    M = mchirp / eta ** (3 / 5)
    mass2 = (M - torch.sqrt(M**2 - 4 * M**2 * eta)) / 2
    mass1 = M - mass2
    return mass1, mass2


def eta_from_mass1_mass2(mass1, mass2):
    """Returns the symmetric mass ratio from mass1 and mass2."""
    return mass1 * mass2 / (mass1 + mass2) ** 2.0


def mchirp_from_mass1_mass2(mass1, mass2):
    """Returns the chirp mass from mass1 and mass2."""
    return eta_from_mass1_mass2(mass1, mass2) ** (3.0 / 5) * (mass1 + mass2)


def mass1_mass2_to_mchirp_eta(mass1, mass2):
    mchirp = mchirp_from_mass1_mass2(mass1, mass2)
    eta = eta_from_mass1_mass2(mass1, mass2)
    return mchirp, eta
