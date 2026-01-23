#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : project.py
Description     : Short description of the file

Created on 2026-01-23 16:17:38

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

# LOCAL
from sage.core.constants import C


def compute_antenna_patterns(ra, dec, psi, detector_tensors):
    """
    Compute the constant antenna patterns F+, Fx for each detector.

    Args:
        ra: tensor, [batch] right ascension (radians)
        dec: tensor, [batch] declination (radians)
        psi: tensor, [batch] polarization angle (radians)
        detector_tensors: tensor, [num_ifos, 3, 3] detector tensors

    Returns:
        Fp: [batch, num_ifos]
        Fc: [batch, num_ifos]
    """
    batch = ra.shape[0]
    num_ifos = detector_tensors.shape[0]

    # Compute source frame vectors (natural polarization basis)
    sin_dec, cos_dec = torch.sin(dec), torch.cos(dec)
    sin_ra, cos_ra = torch.sin(ra), torch.cos(ra)
    sin_psi, cos_psi = torch.sin(psi), torch.cos(psi)

    # Wave propagation unit vector
    n = torch.stack([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec], dim=1)  # [batch, 3]

    # Basis vectors orthogonal to n
    u = torch.stack([sin_ra, -cos_ra, torch.zeros_like(ra)], dim=1)  # [batch, 3]
    v = torch.cross(n, u, dim=1)  # [batch, 3]

    # Polarization basis vectors
    m = -u * sin_psi[:, None] - v * cos_psi[:, None]  # [batch, 3]
    l = -u * cos_psi[:, None] + v * sin_psi[:, None]  # [batch, 3]

    # Compute F+ and Fx for each detector
    # detector_tensors: [num_ifos, 3, 3]
    Fp = torch.einsum(
        "bd,bk,lm->bl", m[:, None, :], m[:, None, :], detector_tensors
    ) - torch.einsum("bd,bk,lm->bl", l[:, None, :], l[:, None, :], detector_tensors)

    Fx = torch.einsum(
        "bd,bk,lm->bl", m[:, None, :], l[:, None, :], detector_tensors
    ) + torch.einsum("bd,bk,lm->bl", l[:, None, :], m[:, None, :], detector_tensors)

    return Fp, Fx  # [batch, num_ifos]


def project_to_detector(hp, hc, ra, dec, psi, detector_tensors):
    """
    Project plus/cross polarizations to detector frame (frequency domain)

    Args:
        hp, hc: [batch, nfreq] complex frequency-domain polarizations
        ra, dec, psi: [batch] sky and polarization angles
        detector_tensors: [num_ifos, 3, 3] detector tensors

    Returns:
        hdet: [batch, num_ifos, nfreq] complex frequency-domain detector strain
    """
    Fp, Fx = compute_antenna_patterns(
        ra, dec, psi, detector_tensors
    )  # [batch, num_ifos]

    # Apply antenna patterns: broadcasting over frequency axis
    # hp/hc: [batch, nfreq], Fp/Fx: [batch, num_ifos]
    hdet = Fp[:, :, None] * hp[:, None, :] + Fx[:, :, None] * hc[:, None, :]

    return hdet
