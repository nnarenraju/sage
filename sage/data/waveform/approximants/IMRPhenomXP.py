#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : IMRPhenomXP.py
Description   : Batched IMRPhenomXP precessing-spin waveform, the
                IMRPhenomXPHM port restricted to the (2,2) mode
                (LALSimIMRPhenomX.c, LALSimIMRPhenomX_precession.c).

Created on 2026-08-04

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from sage.data.waveform.approximants.IMRPhenomXPHM import IMRPhenomXPHM


class IMRPhenomXP(IMRPhenomXPHM):
    """
    Batched IMRPhenomXP precessing-spin waveform.

    IMRPhenomXPHM restricted to the (2,2) co-precessing mode; the l = 2
    twist-up sums only the inertial m in [-2, 2].

    Parameters
    ----------
    f : torch.Tensor, shape (B, F)
        Frequency grid in Hz.
    f_ref : torch.Tensor, shape (B, 1)
        Reference frequency in Hz.
    **kwargs
        Forwarded to IMRPhenomXPHM.
    """

    # Only the (2,2) co-precessing mode is twisted up
    HM_MODES = ()
    COPREC_MODES = {2: (2,)}
