#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit and mass conversions: seconds and samples, component masses,
chirp mass, symmetric mass ratio and chirp distance.

Filename        : conversions.py
Description     : Short description of the file

Created on 2026-01-20 16:26:05

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import torch


def seconds_to_samples(nseconds, sample_rate, approx_mode=int, rounding=True):
    """
    Convert a duration in seconds to a sample count.

    Parameters
    ----------
    nseconds : float
        Duration in seconds.
    sample_rate : float
        Sampling rate in Hz.
    approx_mode : callable
        Type coercion applied to the result (default :class:`int`).
    rounding : bool
        If ``True`` (default), the product is rounded before
        coercion.

    Returns
    -------
    int or approx_mode result
        Number of samples.
    """
    if rounding:
        return approx_mode(round(nseconds * sample_rate))
    else:
        return approx_mode(nseconds * sample_rate)


def samples_to_seconds(nsamples, sample_rate):
    """
    Convert a sample count to a duration in seconds.

    Parameters
    ----------
    nsamples : int
        Number of samples.
    sample_rate : float
        Sampling rate in Hz.

    Returns
    -------
    float
        Duration in seconds.
    """
    return nsamples / sample_rate


def mchirp_eta_to_mass1_mass2(mchirp: torch.Tensor, eta: torch.Tensor):
    """
    Component masses from chirp mass and symmetric mass ratio.

    Parameters
    ----------
    mchirp : torch.Tensor
        Chirp mass, in any unit.
    eta : torch.Tensor
        Symmetric mass ratio, ``0 < eta <= 0.25``.

    Returns
    -------
    mass1, mass2 : torch.Tensor
        Heavier and lighter component masses, in the unit of
        ``mchirp``.
    """
    M = mchirp / eta ** (3 / 5)
    mass2 = (M - torch.sqrt(M**2 - 4 * M**2 * eta)) / 2
    mass1 = M - mass2
    return mass1, mass2


def eta_from_mass1_mass2(mass1, mass2):
    """
    Symmetric mass ratio from the component masses.

    Parameters
    ----------
    mass1, mass2 : array-like
        Component masses.

    Returns
    -------
    array-like
        ``mass1 * mass2 / (mass1 + mass2) ** 2``.
    """
    return mass1 * mass2 / (mass1 + mass2) ** 2.0


def mchirp_from_mass1_mass2(mass1, mass2):
    """
    Chirp mass from the component masses.

    Parameters
    ----------
    mass1, mass2 : array-like
        Component masses.

    Returns
    -------
    array-like
        Chirp mass, in the unit of the masses.
    """
    return eta_from_mass1_mass2(mass1, mass2) ** (3.0 / 5) * (mass1 + mass2)


def chirp_distance(dist, mchirp, ref_mass=1.4):
    """
    Chirp distance, as ``pycbc.conversions.chirp_distance``.

    Parameters
    ----------
    dist : array-like
        Luminosity distance.
    mchirp : array-like
        Chirp mass.
    ref_mass : float
        Component mass of the equal-mass reference binary.

    Returns
    -------
    array-like
        Chirp distance, in the unit of ``dist``.
    """
    return dist * (2.0 ** (-1.0 / 5) * ref_mass / mchirp) ** (5.0 / 6)


def distance_from_chirp_distance_mchirp(chirp_distance, mchirp, ref_mass=1.4):
    """
    Luminosity distance from chirp distance.

    As ``pycbc.conversions.distance_from_chirp_distance_mchirp``.

    Parameters
    ----------
    chirp_distance : array-like
        Chirp distance.
    mchirp : array-like
        Chirp mass.
    ref_mass : float
        Component mass of the equal-mass reference binary.

    Returns
    -------
    array-like
        Luminosity distance, in the unit of ``chirp_distance``.
    """
    return chirp_distance * (2.0 ** (-1.0 / 5) * ref_mass / mchirp) ** (-5.0 / 6)


def mass1_mass2_to_mchirp_eta(mass1, mass2):
    """
    Convert component masses to chirp mass and symmetric mass ratio.

    Parameters
    ----------
    mass1, mass2 : array-like
        Component masses (any consistent units).

    Returns
    -------
    tuple
        ``(mchirp, eta)``: chirp mass and symmetric mass ratio.
    """
    mchirp = mchirp_from_mass1_mass2(mass1, mass2)
    eta = eta_from_mass1_mass2(mass1, mass2)
    return mchirp, eta
