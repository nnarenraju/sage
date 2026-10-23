#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : IMRPhenomXP.py
Description   : GPU-native batched IMRPhenomXP precessing-spin
                frequency-domain waveform model (22-mode twist only).

                In lalsimulation-v6.1.0 the XP core
                (IMRPhenomXPGenerateFD, LALSimIMRPhenomX.c:1885) evaluates
                the co-precessing XAS 22 inline and twists it with
                IMRPhenomXPTwistUp22 (precession.c:1711) — numerically
                equivalent to the XPHM path restricted to the (2,2)
                co-precessing mode with identical Euler-angle machinery,
                Ylm constants and polarization rotation
                (LALSimIMRPhenomXPHM.c doc, section 6 of the port map).
                This class therefore restricts IMRPhenomXPHM to the
                (2,2); no separate twist code is ported.

                NRTidal extensions of XP (LALSimIMRPhenomX.c:2234-2367)
                are not ported here.

                Parameters (theta columns): identical to IMRPhenomXPHM.

Created on 2026-08-04

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

References
----------
IMRPhenomXP   : Pratten et al. (2020), arXiv:2004.06503
"""

from sage.data.waveform.approximants.IMRPhenomXPHM import IMRPhenomXPHM


class IMRPhenomXP(IMRPhenomXPHM):
    """
    GPU-native batched IMRPhenomXP precessing BBH waveform.

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

    # Only the (2,2) co-precessing mode is twisted; the higher-multipole
    # calibration of IMRPhenomXHM is bypassed entirely.
    HM_MODES = ()
    COPREC_MODES = {2: (2,)}
