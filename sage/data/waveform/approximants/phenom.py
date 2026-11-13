#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Device-resident constants shared by the IMRPhenom ports, with the
IMRPhenomD QNM tables and coefficient table.

Filename        : helper.py
Description     : Device-resident constants of the IMRPhenom ports,
                  with the QNM tables of LALSimIMRPhenomD.h.

Created on 2026-01-23 03:24:33

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

from sage.core import constants
from sage.core.interpolation import torch_natural_cubic_coeffs

from .phenom_data import _QNMData_a, _QNMData_fdamp, _QNMData_fRD, PhenomD_coeff_table


class PhenomConstants:
    """
    Device-resident constants of the IMRPhenom waveform models.

    The QNM ringdown and damping tables carry the second derivatives of
    their natural cubic splines, for the gsl_interp_cspline evaluation
    of fring and fdamp.

    Parameters
    ----------
    device : str or torch.device
        Device of all tensors.
    batch_size : int or None
        Batch size of the ONES, ZEROS and PHI6LOG tensors.
    dtype : torch.dtype or None
        Floating-point dtype of the real tensors; ONE_J and TWO_J are
        complex64.
    **kwargs
        Ignored.
    """

    def __init__(self, device="cuda", batch_size=None, dtype=None, **kwargs):

        # Physical and mathematical constants
        for name in constants.CONST_METADATA:
            value = getattr(constants, name)
            setattr(
                self,
                name,
                torch.tensor(value, device=device, dtype=dtype),
            )

        # Natural numbers
        self.ZERO = torch.tensor(0.0, device=device, dtype=dtype)
        self.ONE = torch.tensor(1.0, device=device, dtype=dtype)
        self.THREE = torch.tensor(3.0, device=device, dtype=dtype)
        self.FIVE = torch.tensor(5.0, device=device, dtype=dtype)
        self.SIX = torch.tensor(6.0, device=device, dtype=dtype)
        self.FIFTEEN = torch.tensor(15.0, device=device, dtype=dtype)
        self.TWENTY_FOUR = torch.tensor(24.0, device=device, dtype=dtype)
        self.FORTY_EIGHT = torch.tensor(48.0, device=device, dtype=dtype)
        self.ONE_NINTY_TWO = torch.tensor(192.0, device=device, dtype=dtype)

        # Powers of two (except 1)
        self.TWO = torch.tensor(2.0, device=device, dtype=dtype)
        self.FOUR = torch.tensor(4.0, device=device, dtype=dtype)
        self.EIGHT = torch.tensor(8.0, device=device, dtype=dtype)
        self.SIXTEEN = torch.tensor(16.0, device=device, dtype=dtype)
        self.THIRTY_TWO = torch.tensor(32.0, device=device, dtype=dtype)
        self.SIXTY_FOUR = torch.tensor(64.0, device=device, dtype=dtype)
        self.ONE_TWENTY_EIGHT = torch.tensor(128.0, device=device, dtype=dtype)
        self.TWO_FIFTY_SIX = torch.tensor(256.0, device=device, dtype=dtype)
        self.FIVE_HUNDRED_AND_TWELVE = torch.tensor(512.0, device=device, dtype=dtype)
        self.ONE_THOUSAND_AND_TWENTY_FOUR = torch.tensor(
            1024.0, device=device, dtype=dtype
        )

        # Fractions
        self.HALF = torch.tensor(0.5, device=device, dtype=dtype)
        self.ONE_BY_THREE = torch.tensor(1.0 / 3.0, device=device, dtype=dtype)
        self.THREE_BY_TWO = torch.tensor(3.0 / 2.0, device=device, dtype=dtype)
        self.FIVE_BY_THREE = torch.tensor(5.0 / 3.0, device=device, dtype=dtype)

        # Precomputed
        self.SQRT_6 = torch.sqrt(self.SIX)

        # Complex
        self.ONE_J = torch.tensor(1j, dtype=torch.complex64, device=device)
        self.TWO_J = torch.tensor(2j, dtype=torch.complex64, device=device)
        self.TWOPI = 2.0 * self.PI

        # Dimensionless frequency (Mf) at which define the end of the
        # waveform (f_CUT)
        self.fM_CUT = torch.tensor(0.2, device=device, dtype=dtype)

        # QNM tables and their natural cubic spline second derivatives
        self.QNMData_a = _QNMData_a.to(device=device, dtype=dtype)
        self.QNMData_fRD = _QNMData_fRD.to(device=device, dtype=dtype)
        self.QNMData_fdamp = _QNMData_fdamp.to(device=device, dtype=dtype)
        self.QNMData_fRD_M = torch_natural_cubic_coeffs(
            _QNMData_a, _QNMData_fRD
        ).to(device=device, dtype=dtype)
        self.QNMData_fdamp_M = torch_natural_cubic_coeffs(
            _QNMData_a, _QNMData_fdamp
        ).to(device=device, dtype=dtype)

        # Ones and zeros of shape (B, 1)
        self.ONES = torch.ones(
            (batch_size, 1),
            device=device,
            dtype=dtype,
        )

        self.ZEROS = torch.zeros(
            (batch_size, 1),
            device=device,
            dtype=dtype,
        )

        # PhenomD coefficient table
        self.PhenomD_coeff_table = PhenomD_coeff_table.to(device=device, dtype=dtype)

        # 3PN log coefficient of the TaylorF2 phasing
        self.PHI6LOG = torch.full(
            (batch_size, 1),
            -684.8 / 2.1,
            device=device,
            dtype=dtype,
        )
