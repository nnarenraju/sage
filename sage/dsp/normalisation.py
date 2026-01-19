#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : normalisation.py
Description   : Short description of the file

Created on 2026-01-19 16:29:43

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


class Normalise(TransformWrapperPerChannel):
    def __init__(self, always_apply=True, factors=[1.0, 1.0], ignore_factors=False):
        super().__init__(always_apply)
        assert len(factors) == 2
        self.factors = factors
        self.ignore_factors = ignore_factors

    def apply(self, y: np.ndarray, channel: int, special: dict):
        if not self.ignore_factors:
            norm = y / self.factors[channel]
        else:
            norm = (y - np.min(y)) / (np.max(y) - np.min(y))  # varies from 0 to 1
            norm = norm - np.mean(norm)  # centering at 0
        return norm

