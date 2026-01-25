#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : helper.py
Description     : Short description of the file

Created on 2026-01-23 03:24:33

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

# There are too many constants in the Phenom files which need to be tensors
# Creating tensors during a hot-path iteration kills the torch graph
# Here, we store lots of these constants and allow for device setting
# Putting Phenom into a massive class is not torch.compile friendly (can't use self)
# Dataclass with frozen=True is fine too

# Packages
import torch

# LOCAL
from sage.core import constants
from sage.core.interpolation import torch_scipylike_cubic_interp

from .qnm_data import _QNMData_a, _QNMData_fdamp, _QNMData_fRD


class PhenomConstants:

    def __init__(self, device="cuda"):

        # Natural numbers
        self.ZERO = torch.tensor(0.0, device=device)
        self.ONE = torch.tensor(1.0, device=device)
        self.THREE = torch.tensor(3.0, device=device)
        self.FIVE = torch.tensor(5.0, device=device)
        self.SIX = torch.tensor(6.0, device=device)
        self.FIFTEEN = torch.tensor(15.0, device=device)
        self.TWENTY_FOUR = torch.tensor(24.0, device=device)
        self.FORTY_EIGHT = torch.tensor(48.0, device=device)
        self.ONE_NINTY_TWO = torch.tensor(192.0, device=device)

        # Powers of two (except 1)
        self.TWO = torch.tensor(2.0, device=device)
        self.FOUR = torch.tensor(4.0, device=device)
        self.EIGHT = torch.tensor(8.0, device=device)
        self.SIXTEEN = torch.tensor(16.0, device=device)
        self.THIRTY_TWO = torch.tensor(32.0, device=device)
        self.SIXTY_FOUR = torch.tensor(64.0, device=device)
        self.ONE_TWENTY_EIGHT = torch.tensor(128.0, device=device)
        self.TWO_FIFTY_SIX = torch.tensor(256.0, device=device)
        self.FIVE_HUNDRED_AND_TWELVE = torch.tensor(512.0, device=device)
        self.ONE_THOUSAND_AND_TWENTY_FOUR = torch.tensor(1024.0, device=device)

        # Fractions
        self.HALF = torch.tensor(0.5, device=device)
        self.ONE_BY_THREE = torch.tensor(1.0 / 3.0, device=device)
        self.THREE_BY_TWO = torch.tensor(3.0 / 2.0, device=device)
        self.FIVE_BY_THREE = torch.tensor(5.0 / 3.0, device=device)

        # Precomputed
        self.SQRT_6 = torch.sqrt(self.SIX)

        # Complex
        self.ONE_J = torch.tensor(1j, dtype=torch.complex64, device=device)
        self.TWO_J = torch.tensor(2j, dtype=torch.complex64, device=device)

        # Constants from sage.core
        for name in constants.CONST_METADATA:
            value = getattr(constants, name)
            setattr(
                self,
                name,
                torch.tensor(value, device=device),
            )

        ## Physical constants for Pv2
        self.fM_CUT = torch.tensor(0.2, device=device)

        # QNM Data
        self._QNMData_a = _QNMData_a.to(device=device)
        self._QNMData_fdamp = _QNMData_fdamp.to(device=device)
        self._QNMData_fRD = _QNMData_fRD.to(device=device)

        # Target grid
        self.QNMData_a = torch.linspace(-1, 1, 500_000, device=device)

        # Interpolate using your torch cubic function
        self.QNMData_fRD = torch_scipylike_cubic_interp(
            self.QNMData_a, self._QNMData_a, self._QNMData_fRD
        )
        self.QNMData_fdamp = torch_scipylike_cubic_interp(
            self.QNMData_a, self._QNMData_a, self._QNMData_fdamp
        )
