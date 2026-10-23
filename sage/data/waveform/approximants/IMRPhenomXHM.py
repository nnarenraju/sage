#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : IMRPhenomXHM.py
Description   : GPU-native batched IMRPhenomXHM aligned-spin frequency-domain
                waveform model with calibrated higher multipoles.

                Implements the IMRPhenomXHM multimode aligned-spin model
                entirely in PyTorch, following García-Quirós et al. (2020),
                arXiv:2001.10914.  The (2,2) mode is delegated to
                IMRPhenomXAS; this class adds the calibrated (2,1), (3,3),
                (3,2) and (4,4) modes, including the spheroidal->spherical
                mode mixing of the (3,2).

                Ported from lalsimulation-v6.1.0 (sources vendored in
                sage/data/waveform/lalsim_src/):
                  LALSimIMRPhenomXHM.c            entry points, mode sum
                  LALSimIMRPhenomXHM_internals.c  waveform variables,
                                                  coefficient builders,
                                                  per-frequency evaluation
                  LALSimIMRPhenomXHM_inspiral.c   inspiral amp/phase fits
                  LALSimIMRPhenomXHM_intermediate.c intermediate fits
                  LALSimIMRPhenomXHM_ringdown.c   ringdown fits, spheroidal
                  LALSimIMRPhenomXHM_qnm.c        per-mode QNM + mixing fits

                Version flags are fixed to the v6.1.0 defaults:
                release 122022 amplitudes, 122019 phases (the release flag
                changed only amplitudes; LALSimIMRPhenomXHM.c:2389).
                Multibanding (PhenomXHMThresholdMband) is NOT ported; the
                LAL reference must insert PhenomXHMThresholdMband = 0.

                Parameters (theta columns)
                --------------------------
                waveform_utils.WAVEFORM_PARAMETERS (PyCBC names): mass1,
                mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                distance, tc, coa_phase, inclination, lambda1, lambda2,
                polarization, ra, dec.

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
IMRPhenomXHM  : García-Quirós et al. (2020), arXiv:2001.10914
IMRPhenomXAS  : García-Quirós et al. (2020), arXiv:2001.10914
Mode mixing   : Berti & Klein (2014), arXiv:1408.1860
"""

import math

import torch

from sage.data.waveform.approximants.IMRPhenomXAS import IMRPhenomXAS
from sage.data.waveform import taper as _taper_mod
from sage.data.waveform import waveform_utils
from sage.data.waveform.approximants import phenomxhm_fits_inspiral as _fits_insp
from sage.data.waveform.approximants import phenomxhm_fits_intermediate_amp as _fits_iamp
from sage.data.waveform.approximants import phenomxhm_fits_intermediate_phase as _fits_iphase
from sage.data.waveform.approximants import phenomxhm_fits_ringdown_amp as _fits_ramp
from sage.data.waveform.approximants import phenomxhm_fits_ringdown_phase as _fits_rphase

# Parameter-space fit dispatch, mirroring the InspiralAmpFits /
# IntermediateAmpFits / RingdownAmpFits function-pointer tables
# (IMRPhenomXHM_FillAmpFitsArray, internals.c:580-709), restricted to the
# entries reachable on the default release-122022 path.
_INSP_AMP_FITS = {
    21: (_fits_insp.IMRPhenomXHM_Insp_Amp_21_iv1, _fits_insp.IMRPhenomXHM_Insp_Amp_21_iv2, _fits_insp.IMRPhenomXHM_Insp_Amp_21_iv3),
    33: (_fits_insp.IMRPhenomXHM_Insp_Amp_33_iv1, _fits_insp.IMRPhenomXHM_Insp_Amp_33_iv2, _fits_insp.IMRPhenomXHM_Insp_Amp_33_iv3),
    32: (_fits_insp.IMRPhenomXHM_Insp_Amp_32_iv1, _fits_insp.IMRPhenomXHM_Insp_Amp_32_iv2, _fits_insp.IMRPhenomXHM_Insp_Amp_32_iv3),
    44: (_fits_insp.IMRPhenomXHM_Insp_Amp_44_iv1, _fits_insp.IMRPhenomXHM_Insp_Amp_44_iv2, _fits_insp.IMRPhenomXHM_Insp_Amp_44_iv3),
}
_INTER_AMP_FITS = {
    21: {1: _fits_iamp.IMRPhenomXHM_Inter_Amp_21_int1, 2: _fits_iamp.IMRPhenomXHM_Inter_Amp_21_int2,
         3: _fits_iamp.IMRPhenomXHM_Inter_Amp_21_int3, 4: _fits_iamp.IMRPhenomXHM_Inter_Amp_21_int4},
    33: {1: _fits_iamp.IMRPhenomXHM_Inter_Amp_33_int1, 2: _fits_iamp.IMRPhenomXHM_Inter_Amp_33_int2,
         3: _fits_iamp.IMRPhenomXHM_Inter_Amp_33_int3, 4: _fits_iamp.IMRPhenomXHM_Inter_Amp_33_int4},
    32: {1: _fits_iamp.IMRPhenomXHM_Inter_Amp_32_int1, 2: _fits_iamp.IMRPhenomXHM_Inter_Amp_32_int2,
         3: _fits_iamp.IMRPhenomXHM_Inter_Amp_32_int3, 4: _fits_iamp.IMRPhenomXHM_Inter_Amp_32_int4},
    44: {1: _fits_iamp.IMRPhenomXHM_Inter_Amp_44_int1, 2: _fits_iamp.IMRPhenomXHM_Inter_Amp_44_int2,
         3: _fits_iamp.IMRPhenomXHM_Inter_Amp_44_int3, 4: _fits_iamp.IMRPhenomXHM_Inter_Amp_44_int4},
}
_RD_AMP_RDCP_FITS = {
    21: (_fits_ramp.IMRPhenomXHM_RD_Amp_21_rdcp1, _fits_ramp.IMRPhenomXHM_RD_Amp_21_rdcp2, _fits_ramp.IMRPhenomXHM_RD_Amp_21_rdcp3),
    33: (_fits_ramp.IMRPhenomXHM_RD_Amp_33_rdcp1, _fits_ramp.IMRPhenomXHM_RD_Amp_33_rdcp2, _fits_ramp.IMRPhenomXHM_RD_Amp_33_rdcp3),
    44: (_fits_ramp.IMRPhenomXHM_RD_Amp_44_rdcp1, _fits_ramp.IMRPhenomXHM_RD_Amp_44_rdcp2, _fits_ramp.IMRPhenomXHM_RD_Amp_44_rdcp3),
}
_RD_AMP_32_COEFF_FITS = (_fits_ramp.IMRPhenomXHM_RD_Amp_32_alambda,
                         _fits_ramp.IMRPhenomXHM_RD_Amp_32_lambda,
                         _fits_ramp.IMRPhenomXHM_RD_Amp_32_sigma)
_RD_AMP_32_RDAUX_FITS = (_fits_ramp.IMRPhenomXHM_RD_Amp_32_rdaux1,
                         _fits_ramp.IMRPhenomXHM_RD_Amp_32_rdaux2)

# Phase fit dispatch (IMRPhenomXHM_FillPhaseFitsArray, internals.c:525-577)
_INSP_PHASE_LAMBDA_FITS = {
    21: _fits_insp.IMRPhenomXHM_Insp_Phase_21_lambda,
    33: _fits_insp.IMRPhenomXHM_Insp_Phase_33_lambda,
    32: _fits_insp.IMRPhenomXHM_Insp_Phase_32_lambda,
    44: _fits_insp.IMRPhenomXHM_Insp_Phase_44_lambda,
}
_INTER_PHASE_FITS = {
    21: tuple(getattr(_fits_iphase, f"IMRPhenomXHM_Inter_Phase_21_p{i}") for i in range(1, 7)),
    33: tuple(getattr(_fits_iphase, f"IMRPhenomXHM_Inter_Phase_33_p{i}") for i in range(1, 7)),
    32: tuple(getattr(_fits_iphase, f"IMRPhenomXHM_Inter_Phase_32_p{i}") for i in range(1, 7)),
    44: tuple(getattr(_fits_iphase, f"IMRPhenomXHM_Inter_Phase_44_p{i}") for i in range(1, 7)),
}
_RD_PHASE_22_ALPHA2 = _fits_rphase.IMRPhenomXHM_RD_Phase_22_alpha2
_RD_PHASE_22_ALPHAL = _fits_rphase.IMRPhenomXHM_RD_Phase_22_alphaL
_RD_PHASE_32_FITS = tuple(getattr(_fits_rphase, f"IMRPhenomXHM_RD_Phase_32_p{i}")
                          for i in range(1, 5))
_RD_PHASE_32_TIMESHIFT = _fits_rphase.IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift
_RD_PHASE_32_PHASESHIFT = _fits_rphase.IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift

# FALSE_ZERO (LALSimIMRPhenomXHM_internals.h:59)
_FALSE_ZERO = 1.0e-15


class IMRPhenomXHM(IMRPhenomXAS):
    """
    GPU-native batched IMRPhenomXHM aligned-spin multimode BBH waveform.

    Extends IMRPhenomXAS with the calibrated higher multipoles.  Each mode
    is built from the same 3-region (inspiral / intermediate / ringdown)
    ansatz family as the 22, with per-mode collocation-point fits and, for
    the (3,2), a spheroidal-basis ringdown rotated to the spherical basis
    with the 22 mode.

    Mode convention (LALSimIMRPhenomXHM.c:456-470): the single-mode
    generator produces h_{l,-m}(f) on the positive-frequency axis;
    h_{l,+m} follows from equatorial symmetry,
    h_{l,+m}(f) = (-1)^l conj(h_{l,-m}(-f)).

    Parameters
    ----------
    f : torch.Tensor, shape (B, F)
        Frequency grid in Hz.
    f_ref : torch.Tensor, shape (B, 1)
        Reference frequency in Hz.
    **kwargs
        Forwarded to IMRPhenomXAS / PhenomConstants.
    """

    # Calibrated higher multipoles; (2,2) is delegated to IMRPhenomXAS.
    # N_HIGHERMODES_IMPLEMENTED = 4, modeInt ordering {21:0, 33:1, 32:2, 44:3}
    # (LALSimIMRPhenomXHM_internals.c:108-137).
    HM_MODES = ((2, 1), (3, 3), (3, 2), (4, 4))

    # v6.1.0 default version flags (LALSimIMRPhenomXHM_internals.c:295-370).
    # Keys are modeTag = 10*ell + emm.
    RELEASE_VERSION = 122022
    INSPIRAL_AMP_VERSION = {21: 123, 33: 123, 32: 123, 44: 123}
    INTERMEDIATE_AMP_VERSION = {21: 110102, 33: 211112, 32: 211112, 44: 211112}
    RINGDOWN_AMP_VERSION = {21: 2, 33: 2, 32: 1, 44: 2}
    # Threshold for extreme-mass-ratio phase treatment (internals.c:386).
    ETA_EMR = 0.05
    # FD step of the (3,2) carg(SpheroidalToSpherical) derivatives
    # (internals.c:2188); class attribute so tests can probe the FD noise.
    _S2S_FSTEP = 0.0000001

    def __init__(self, f, f_ref, **kwargs):
        super().__init__(f, f_ref, **kwargs)

    # ------------------------------------------------------------------
    # Spin-weighted spherical harmonics
    # Source: XLALSpinWeightedSphericalHarmonic (lal SphericalHarmonics.c)
    # ------------------------------------------------------------------

    @staticmethod
    def spin_weighted_Y(theta, phi, s, ell, emm):
        """
        Spin-weight -2 spherical harmonic Y_{lm}(theta, phi), l = 2..4.

        Closed forms per (l, m); `ell`/`emm` are compile-time Python ints
        (dispatch by Python branching, as in IMRPhenomPv2.SpinWeightedY).
        Mirrors XLALSpinWeightedSphericalHarmonic
        (lal SphericalHarmonics.c:53-423); expressions carried over
        verbatim, ans = cpolar(1, m*phi) * fac.

        Returns
        -------
        torch.Tensor, complex, broadcast shape of ``theta``.
        """
        if s != -2:
            raise ValueError("only s = -2 implemented")
        PI = math.pi
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ch = torch.cos(theta / 2.0)
        sh = torch.sin(theta / 2.0)
        if ell == 2:
            if emm == -2:
                fac = math.sqrt(5.0 / (64.0 * PI)) * (1.0 - ct) * (1.0 - ct)
            elif emm == -1:
                fac = math.sqrt(5.0 / (16.0 * PI)) * st * (1.0 - ct)
            elif emm == 0:
                fac = math.sqrt(15.0 / (32.0 * PI)) * st * st
            elif emm == 1:
                fac = math.sqrt(5.0 / (16.0 * PI)) * st * (1.0 + ct)
            elif emm == 2:
                fac = math.sqrt(5.0 / (64.0 * PI)) * (1.0 + ct) * (1.0 + ct)
            else:
                raise ValueError("require |m| <= l")
        elif ell == 3:
            if emm == -3:
                fac = math.sqrt(21.0 / (2.0 * PI)) * ch * sh.pow(5)
            elif emm == -2:
                fac = math.sqrt(7.0 / (4.0 * PI)) * (2.0 + 3.0 * ct) * sh.pow(4)
            elif emm == -1:
                fac = math.sqrt(35.0 / (2.0 * PI)) * (
                    torch.sin(theta) + 4.0 * torch.sin(2.0 * theta)
                    - 3.0 * torch.sin(3.0 * theta)
                ) / 32.0
            elif emm == 0:
                fac = (math.sqrt(105.0 / (2.0 * PI)) * ct * st.pow(2)) / 4.0
            elif emm == 1:
                fac = -math.sqrt(35.0 / (2.0 * PI)) * (
                    torch.sin(theta) - 4.0 * torch.sin(2.0 * theta)
                    - 3.0 * torch.sin(3.0 * theta)
                ) / 32.0
            elif emm == 2:
                fac = math.sqrt(7.0 / PI) * ch.pow(4) * (-2.0 + 3.0 * ct) / 2.0
            elif emm == 3:
                fac = -math.sqrt(21.0 / (2.0 * PI)) * ch.pow(5) * sh
            else:
                raise ValueError("require |m| <= l")
        elif ell == 4:
            if emm == -4:
                fac = 3.0 * math.sqrt(7.0 / PI) * ch.pow(2) * sh.pow(6)
            elif emm == -3:
                fac = 3.0 * math.sqrt(7.0 / (2.0 * PI)) * ch * (1.0 + 2.0 * ct) * sh.pow(5)
            elif emm == -2:
                fac = (3.0 * (9.0 + 14.0 * ct + 7.0 * torch.cos(2.0 * theta))
                       * sh.pow(4)) / (4.0 * math.sqrt(PI))
            elif emm == -1:
                fac = (3.0 * (3.0 * torch.sin(theta) + 2.0 * torch.sin(2.0 * theta)
                              + 7.0 * torch.sin(3.0 * theta) - 7.0 * torch.sin(4.0 * theta))
                       ) / (32.0 * math.sqrt(2.0 * PI))
            elif emm == 0:
                fac = (3.0 * math.sqrt(5.0 / (2.0 * PI))
                       * (5.0 + 7.0 * torch.cos(2.0 * theta)) * st.pow(2)) / 16.0
            elif emm == 1:
                fac = (3.0 * (3.0 * torch.sin(theta) - 2.0 * torch.sin(2.0 * theta)
                              + 7.0 * torch.sin(3.0 * theta) + 7.0 * torch.sin(4.0 * theta))
                       ) / (32.0 * math.sqrt(2.0 * PI))
            elif emm == 2:
                fac = (3.0 * ch.pow(4) * (9.0 - 14.0 * ct + 7.0 * torch.cos(2.0 * theta))
                       ) / (4.0 * math.sqrt(PI))
            elif emm == 3:
                fac = -3.0 * math.sqrt(7.0 / (2.0 * PI)) * ch.pow(5) * (-1.0 + 2.0 * ct) * sh
            elif emm == 4:
                fac = 3.0 * math.sqrt(7.0 / PI) * ch.pow(6) * sh.pow(2)
            else:
                raise ValueError("require |m| <= l")
        else:
            raise ValueError("only l in [2, 4] implemented")
        # ans = cpolar(1.0, m*phi) * fac; phi may be a compile-time float
        # (mode sum uses phi = pi/2, precession uses phi = 0).
        angle = emm * phi
        if isinstance(angle, torch.Tensor):
            return fac * torch.polar(torch.ones_like(fac), angle)
        return fac * complex(math.cos(angle), math.sin(angle))

    # ------------------------------------------------------------------
    # QNM fits: per-mode ringdown/damping frequencies
    # Source: LALSimIMRPhenomXHM_qnm.c (rational polynomials in af)
    # ------------------------------------------------------------------

    @staticmethod
    def _qnm_fring_21(af):
        """evaluate_QNMfit_fring21 (LALSimIMRPhenomXHM_qnm.c:30)."""
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        return (0.059471695665734674 - 0.07585416297991414*af + 0.021967909664591865*x2 - 0.0018964744613388146*x3 + 0.001164879406179587*x4 - 0.0003387374454044957*x5) / (1 - 1.4437415542456158*af + 0.49246920313191234*x2)

    @staticmethod
    def _qnm_fdamp_21(af):
        """evaluate_QNMfit_fdamp21 (LALSimIMRPhenomXHM_qnm.c:47)."""
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        return (2.0696914454467294 - 3.1358071947583093*af + 0.14456081596393977*x2 + 1.2194717985037946*x3 - 0.2947372598589144*x4 + 0.002943057145913646*x5) / (146.1779212636481 - 219.81790388304876*af + 17.7141194900164*x2 + 75.90115083917898*x3 - 18.975287709794745*x4)

    @staticmethod
    def _qnm_fring_33(af):
        """evaluate_QNMfit_fring33 (LALSimIMRPhenomXHM_qnm.c:64)."""
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.09540436245212061 - 0.22799517865876945*af + 0.13402916709362475*x2 + 0.03343753057911253*x3 - 0.030848060170259615*x4 - 0.006756504382964637*x5 + 0.0027301732074159835*x6) / (1 - 2.7265947806178334*af + 2.144070539525238*x2 - 0.4706873667569393*x4 + 0.05321818246993958*x6)

    @staticmethod
    def _qnm_fdamp_33(af):
        """evaluate_QNMfit_fdamp33 (LALSimIMRPhenomXHM_qnm.c:82)."""
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        return (0.014754148319335946 - 0.03124423610028678*af + 0.017192623913708124*x2 + 0.001034954865629645*x3 - 0.0015925124814622795*x4 - 0.0001414350555699256*x5) / (1 - 2.0963684630756894*af + 1.196809702382645*x2 - 0.09874113387889819*x4)

    @staticmethod
    def _qnm_fring_32(af):
        """evaluate_QNMfit_fring32 (LALSimIMRPhenomXHM_qnm.c:99)."""
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.09540436245212061 - 0.13628306966373951*af + 0.030099881830507727*x2 - 0.000673589757007597*x3 + 0.0118277880067919*x4 + 0.0020533816327907334*x5 - 0.0015206141948469621*x6) / (1 - 1.6531854335715193*af + 0.5634705514193629*x2 + 0.12256204148002939*x4 - 0.027297817699401976*x6)

    @staticmethod
    def _qnm_fdamp_32(af):
        """evaluate_QNMfit_fdamp32 (LALSimIMRPhenomXHM_qnm.c:117)."""
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        return (0.014754148319335946 - 0.03445752346074498*af + 0.02168855041940869*x2 + 0.0014945908223317514*x3 - 0.0034761714223258693*x4) / (1 - 2.320722660848874*af + 1.5096146036915865*x2 - 0.18791187563554512*x4)

    @staticmethod
    def _qnm_fring_44(af):
        """evaluate_QNMfit_fring44 (LALSimIMRPhenomXHM_qnm.c:133)."""
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.1287821193485683 - 0.21224284094693793*af + 0.0710926778043916*x2 + 0.015487322972031054*x3 - 0.002795401084713644*x4 + 0.000045483523029172406*x5 + 0.00034775290179000503*x6) / (1 - 1.9931645124693607*af + 1.0593147376898773*x2 - 0.06378640753152783*x4)

    @staticmethod
    def _qnm_fdamp_44(af):
        """evaluate_QNMfit_fdamp44 (LALSimIMRPhenomXHM_qnm.c:151)."""
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.014986847152355699 - 0.01722587715950451*af - 0.0016734788189065538*x2 + 0.0002837322846047305*x3 + 0.002510528746148588*x4 + 0.00031983835498725354*x5 + 0.000812185411753066*x6) / (1 - 1.1350205970682399*af - 0.0500827971270845*x2 + 0.13983808071522857*x4 + 0.051876225199833995*x6)

    _QNM_FITS = {
        21: ("_qnm_fring_21", "_qnm_fdamp_21"),
        33: ("_qnm_fring_33", "_qnm_fdamp_33"),
        32: ("_qnm_fring_32", "_qnm_fdamp_32"),
        44: ("_qnm_fring_44", "_qnm_fdamp_44"),
    }

    @classmethod
    def get_fring_fdamp_lm(cls, af, Mfinal, ell, emm):
        """
        Dimensionless (Mf) ringdown and damping frequency of mode (l, m).

        fRING = fring_lm(afinal) / Mfinal ; fDAMP = fdamp_lm(afinal) / Mfinal
        Source: IMRPhenomXHM_SetHMWaveformVariables,
        LALSimIMRPhenomXHM_internals.c:393-394.
        """
        fring_name, fdamp_name = cls._QNM_FITS[10 * ell + emm]
        fring = getattr(cls, fring_name)(af) / Mfinal
        fdamp = getattr(cls, fdamp_name)(af) / Mfinal
        return fring, fdamp

    @staticmethod
    def _qnm_mixing_fits(af):
        """
        Spheroidal<->spherical mixing coefficients for the (3,2)-(2,2) pair.

        Eight real fits evaluate_QNMfit_{re,im}_l{2,3}m2lp{2,3}(af)
        (LALSimIMRPhenomXHM_qnm.c:170-311) assembled into the complex
        mixingCoeffs[4] = {a222, a223, a322, a323}; the signs of a322 and
        a323 are flipped to match the hybrid conventions
        (IMRPhenomXHM_Initialize_MixingCoeffs, internals.c:62-80).

        Returns
        -------
        tuple of 4 complex torch.Tensor, each shaped like ``af``.
        """
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        # evaluate_QNMfit_re_l2m2lp2 (qnm.c:170)
        re_l2m2lp2 = (1 - 2.2956993576253635*af + 1.461988775298876*x2 + 0.0043296365593147035*x3 - 0.1695667458204109*x4 - 0.0006267849034466508*x5) / (1 - 2.2956977727459043*af + 1.4646339137818438*x2 - 0.16843226886562457*x4 - 0.00007150540890128118*x6)
        # evaluate_QNMfit_im_l2m2lp2 (qnm.c:188)
        im_l2m2lp2 = (af*(0.3826673013161342 - 0.47531267226013896*af - 0.05898102880105067*x2 + 0.0724525431346487*x3 + 0.054714637311702986*x4 + 0.024544862718252784*x5)) / (-38.70835035062785 + 69.82140084545878*af - 27.99036444363243*x2 - 4.152310472191899*x4 + 1.*x6)
        # evaluate_QNMfit_re_l3m2lp2 (qnm.c:206)
        re_l3m2lp2 = (af*(0.47513455283841244 - 0.9016636384605536*af + 0.3844811236426182*x2 + 0.0855565148647794*x3 - 0.03620067426672167*x4 - 0.006557249133752502*x5)) / (-6.76894063440646 + 15.170831931186493*af - 9.406169787571082*x2 + 1.*x4)
        # evaluate_QNMfit_im_l3m2lp2 (qnm.c:223)
        im_l3m2lp2 = (af*(-2.8704762147145533 + 4.436434016918535*af - 1.0115343326360486*x2 - 0.08965314412106505*x3 - 0.4236810894599512*x4 - 0.041787576033810676*x5)) / (-171.80908957903395 + 272.362882450877*af - 76.68544453077854*x2 - 25.14197656531123*x4 + 1.*x6)
        # evaluate_QNMfit_re_l2m2lp3 (qnm.c:241)
        re_l2m2lp3 = (af*(18.522563276099167 - 37.978140351289014*af + 19.030390708998894*x2 + 3.0355668591803386*x3 - 2.210028290847915*x4 - 0.37117112862247975*x5)) / (164.52480238697507 - 377.9093045285145*af + 243.3353695550844*x2 - 30.79738566181734*x4 + 1.*x6)
        # evaluate_QNMfit_im_l2m2lp3 (qnm.c:259)
        im_l2m2lp3 = (af*(-49.7688437256778 + 120.43773704442333*af - 82.95323455645332*x2 + 1.721453011852496*x3 + 11.540237244397877*x4 - 0.9819458637589314*x5)) / (2858.5790831181725 - 6305.619505422591*af + 3825.6742092829054*x2 - 377.7822297815406*x4 + 1.*x6)
        # evaluate_QNMfit_re_l3m2lp3 (qnm.c:277)
        re_l3m2lp3 = (1 - 2.107852425643677*af + 1.1906393634562715*x2 + 0.02244848864087732*x3 - 0.09593447799423722*x4 - 0.0021343381708933025*x5 - 0.005319515989331159*x6) / (1 - 2.1078515887706324*af + 1.2043484690080966*x2 - 0.08910191596778137*x4 - 0.005471749827809503*x6)
        # evaluate_QNMfit_im_l3m2lp3 (qnm.c:295)
        im_l3m2lp3 = (af*(12.45701482868677 - 29.398484595717147*af + 18.26221675782779*x2 + 1.9308599142669403*x3 - 3.159763242921214*x4 - 0.0910871567367674*x5)) / (345.52914639836257 - 815.4349339779621*af + 538.3888932415709*x2 - 69.3840921447381*x4 + 1.*x6)
        a222 = torch.complex(re_l2m2lp2, im_l2m2lp2)
        a223 = torch.complex(re_l2m2lp3, im_l2m2lp3)
        # a322, a323 multiplied by -1 to match the hybrid conventions
        # (IMRPhenomXHM_Initialize_MixingCoeffs, internals.c:70-71).
        a322 = -torch.complex(re_l3m2lp2, im_l3m2lp2)
        a323 = -torch.complex(re_l3m2lp3, im_l3m2lp3)
        return a222, a223, a322, a323

    # ------------------------------------------------------------------
    # Per-mode waveform variables (pWFHM)
    # Source: IMRPhenomXHM_SetHMWaveformVariables, internals.c:84-470
    # ------------------------------------------------------------------

    def set_hm_waveform_variables(self, ell, emm, theta, derived, afinal=None,
                                  Mfinal=None):
        """
        Build the per-mode waveform-variable dict (mirrors
        IMRPhenomXHM_SetHMWaveformVariables, internals.c:84-470, for the
        default 122022 release with all PNR toggles off).

        Keys are the LAL struct member names; version flags are Python
        ints, physics quantities are (B, 1) tensors.

        Parameters
        ----------
        afinal, Mfinal : torch.Tensor (B, 1), optional
            Final spin / final mass override (the precessing subclass
            passes the precessing final spin); default is the aligned-spin
            Jimenez-Forteza 2017 fit inherited from IMRPhenomXAS.
        """
        eta = derived[:, 3:4]
        delta = derived[:, 4:5]
        chiPNHat = derived[:, 8:9]
        STotR = derived[:, 9:10]
        dchi = derived[:, 10:11]
        chi1L = derived[:, 11:12]
        chi2L = derived[:, 12:13]
        M_s = derived[:, 2:3]

        modeTag = 10 * ell + emm
        modeInt = {21: 0, 33: 1, 32: 2, 44: 3}[modeTag]

        # Ampzero = 1 (true) for odd modes and equal black holes
        # (internals.c:100, 112-122): q == 1 and chi1L == chi2L.
        if emm % 2 == 1:
            Ampzero = (theta[:, 0:1] == theta[:, 1:2]) & (chi1L == chi2L)
        else:
            Ampzero = torch.zeros_like(eta, dtype=torch.bool)

        # wf->ampNorm = wf22->ampNorm ; wf->fMECOlm = wf22->fMECO*emm*0.5
        ampNorm = torch.sqrt(2.0 * eta / 3.0) * (math.pi ** (-1.0 / 6.0))
        fMECOlm = self.fMECO(eta, chiPNHat, dchi, delta) * (emm * 0.5)

        # wf->Amp0 = wf22->amp0 = M_meters * M_s / dist_m (internals.c:606)
        dist_m = theta[:, 4:5] * self.Mpc
        Amp0 = (M_s * self.C) * M_s / dist_m

        # Release 122022 version overrides (internals.c:295-370).
        pWFHM = {
            "ell": ell,
            "emm": emm,
            "modeTag": modeTag,
            "modeInt": modeInt,
            "MixingOn": 1 if modeTag == 32 else 0,
            "Ampzero": Ampzero,
            "ampNorm": ampNorm,
            "fMECOlm": fMECOlm,
            "Amp0": Amp0,
            "useFAmpPN": 0,
            "AmpEMR": 0,
            "InspiralAmpVeto": 0,
            "IntermediateAmpVeto": 0,
            "RingdownAmpVeto": 0,
            "IMRPhenomXHMReleaseVersion": self.RELEASE_VERSION,
            "IMRPhenomXHMInspiralPhaseVersion": 122019,
            "IMRPhenomXHMIntermediatePhaseVersion": 122019,
            "IMRPhenomXHMRingdownPhaseVersion": 122019,
            "IMRPhenomXHMInspiralAmpFreqsVersion": 122022,
            "IMRPhenomXHMIntermediateAmpFreqsVersion": 0,
            "IMRPhenomXHMRingdownAmpFreqsVersion": 122022,
            "IMRPhenomXHMInspiralAmpFitsVersion": 122022,
            "IMRPhenomXHMIntermediateAmpFitsVersion": 122022,
            "IMRPhenomXHMRingdownAmpFitsVersion": 122022,
            "IMRPhenomXHMInspiralAmpVersion": self.INSPIRAL_AMP_VERSION[modeTag],
            "IMRPhenomXHMIntermediateAmpVersion": self.INTERMEDIATE_AMP_VERSION[modeTag],
            "IMRPhenomXHMRingdownAmpVersion": self.RINGDOWN_AMP_VERSION[modeTag],
            "nCollocPtsRDPhase": 4 if modeTag == 32 else 0,
            "nCollocPtsInterPhase": 6 if modeTag == 32 else 5,
            "etaEMR": self.ETA_EMR,
        }
        pWFHM["nCollocPtsInspAmp"] = len(str(pWFHM["IMRPhenomXHMInspiralAmpVersion"]))
        pWFHM["nCollocPtsInterAmp"] = len(str(pWFHM["IMRPhenomXHMIntermediateAmpVersion"]))

        # Spin parameterisations for the higher modes (internals.c:389-390)
        pWFHM["chi_s"] = (chi1L + chi2L) * 0.5
        pWFHM["chi_a"] = (chi1L - chi2L) * 0.5

        # Ringdown and damping frequencies (internals.c:393-394)
        if afinal is None:
            afinal = self.final_spin_2017(eta, STotR, dchi, delta)
        if Mfinal is None:
            Mfinal = self.final_mass_2017(eta, STotR, dchi, delta)
        pWFHM["fRING"], pWFHM["fDAMP"] = self.get_fring_fdamp_lm(
            afinal, Mfinal, ell, emm
        )

        # (3,2): mixing coeffs to transform the spheroidal-harmonic
        # ringdown ansatz back to spherical-harmonic (internals.c:455-458)
        if modeTag == 32:
            pWFHM["mixingCoeffs"] = self._qnm_mixing_fits(afinal)

        # Linear part of the 22 phase. timeshift * ff + phaseshift.
        pWFHM["timeshift"] = 0.0
        pWFHM["phaseshift"] = 0.0
        # current time-alignment of the hybrids (internals.c:464-465)
        psi4tostrain = IMRPhenomXHM._psi4tostrain_fit(eta, STotR, dchi)
        pWFHM["DeltaT"] = -2.0 * math.pi * (500.0 + psi4tostrain)

        pWFHM["fPhaseRDflat"] = 0.0
        pWFHM["fAmpRDfalloff"] = 0.0
        return pWFHM

    # ------------------------------------------------------------------
    # Amplitude: region boundaries
    # ------------------------------------------------------------------

    def _amp_fcut_insp(self, pWFHM, derived):
        """
        Inspiral->intermediate amplitude cut fAmpMatchIN, version 122022.

        Return the end frequency of the inspiral region and the beginning
        of the intermediate for the amplitude of one mode.
        Source: IMRPhenomXHM_Amplitude_fcutInsp, internals.c:719-798
        (case 122022, internals.c:782-794).
        """
        fMECO = pWFHM["fMECOlm"]
        emm = 1.0 * pWFHM["emm"]
        eta = derived[:, 3:4]
        chi1 = derived[:, 11:12]
        q = derived[:, 0:1] / derived[:, 1:2]
        # fcutEMR is the cutting frequency for extreme mass ratios that is
        # given by a fit to the frequency of a particular geometrical
        # structure of the amplitude (internals.c:730).
        fcutEMR = 1.25 * emm * (
            (0.011671068725758493 - 0.0000858396080377194 * chi1
             + 0.000316707064291237 * chi1 * chi1)
            * (0.8447212540381764 + 6.2873167352395125 * eta)
        ) / (1.2857082764038923 - 0.9977728883419751 * chi1)
        # q < 20: fMECO; else smooth tanh transition to the EMR fit
        # (transition_eta = 0.0192234 i.e. q=50, sharpness = 0.004).
        funcs = 0.5 + 0.5 * torch.tanh((eta - 0.0192234) / 0.004)
        blend = funcs * fMECO + (1.0 - funcs) * fcutEMR
        return torch.where(q < 20.0, fMECO, blend)

    def _amp_fcut_rd(self, pWFHM, phase22_coeffs):
        """
        Intermediate->ringdown amplitude cut fAmpMatchIM, version 122022.

        MixingOn: fRING22 - 0.5*fDAMP22 (v8); else fRINGlm - fDAMPlm (v2).
        Source: IMRPhenomXHM_Amplitude_fcutRD, internals.c:801-858
        (case 122022, internals.c:847-854).
        """
        if pWFHM["MixingOn"] == 1:
            return phase22_coeffs["fRING"] - 0.5 * phase22_coeffs["fDAMP"]
        return pWFHM["fRING"] - pWFHM["fDAMP"]

    # ------------------------------------------------------------------
    # Amplitude: parameter-space fits (release 122022)
    # ------------------------------------------------------------------

    @staticmethod
    def _pwf_fit_dict(derived):
        """
        pWF dict consumed by the parameter-space fit modules; keys are the
        IMRPhenomXWaveformStruct member names, values (B, 1) tensors.
        """
        dchi = derived[:, 10:11]
        return {
            "eta": derived[:, 3:4],
            "delta": derived[:, 4:5],
            "chiEff": derived[:, 7:8],
            "chiPNHat": derived[:, 8:9],
            "STotR": derived[:, 9:10],
            "dchi": dchi,
            "dchi_half": dchi * 0.5,
            "chi1L": derived[:, 11:12],
            "chi2L": derived[:, 12:13],
        }

    # ------------------------------------------------------------------
    # Amplitude: PN prefactors and coefficient assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _initialize_powers(number):
        """
        Cached powers of a number, built by the exact multiplication chain
        of IMRPhenomX_Initialize_Powers (LALSimIMRPhenomX_internals.c:54-103)
        so that double rounding matches LAL bit for bit.  Works for Python
        floats and torch tensors alike; returns a dict keyed by the LAL
        struct member names.
        """
        sixth = number ** (1.0 / 6.0)
        p = {}
        p["one_sixth"] = sixth
        p["m_one_sixth"] = 1.0 / sixth
        p["m_one"] = 1.0 / number
        p["itself"] = number
        p["one_third"] = sixth * sixth
        p["m_one_third"] = 1.0 / p["one_third"]
        p["two_thirds"] = p["one_third"] * p["one_third"]
        p["m_two_thirds"] = 1.0 / p["two_thirds"]
        p["four_thirds"] = p["two_thirds"] * p["two_thirds"]
        p["m_four_thirds"] = 1.0 / p["four_thirds"]
        p["five_thirds"] = p["four_thirds"] * p["one_third"]
        p["m_five_thirds"] = 1.0 / p["five_thirds"]
        p["seven_thirds"] = p["four_thirds"] * number
        p["m_seven_thirds"] = 1.0 / p["seven_thirds"]
        p["eight_thirds"] = p["seven_thirds"] * p["one_third"]
        p["m_eight_thirds"] = 1.0 / p["eight_thirds"]
        p["two"] = number * number
        p["three"] = p["two"] * number
        p["four"] = p["three"] * number
        p["five"] = p["four"] * number
        p["m_two"] = 1.0 / p["two"]
        p["m_three"] = 1.0 / p["three"]
        p["m_four"] = 1.0 / p["four"]
        p["m_five"] = 1.0 / p["five"]
        p["seven_sixths"] = p["one_sixth"] * p["itself"]
        p["m_seven_sixths"] = p["m_one_sixth"] * p["m_one"]
        p["log"] = math.log(number) if isinstance(number, float) else torch.log(number)
        p["sqrt"] = p["one_sixth"] * p["one_sixth"] * p["one_sixth"]
        return p

    def _get_pn_amplitude_coeffs(self, pWFHM, derived):
        """
        Complex PN inspiral-amplitude prefactors pnInitial..pnSixThirds and
        PNglobalfactor for mode (l, m); eqs. E10-E14 of arXiv:2001.10914.

        The ansatz in Fourier Domain is built as follows: we multiply the
        Time-domain Post-Newtonian series up to 3PN by the phasing factor
        given by the Stationary-Phase-Approximation, and then reexpand in
        powers of f up to 3PN.  useFAmpPN is 0 in release 122022, so the
        21 uses the reexpanded power series like every other mode.

        Source: IMRPhenomXHM_GetPNAmplitudeCoefficients,
        internals.c:969-1057; expressions carried over verbatim with
        I -> 1j.
        """
        chiA = pWFHM["chi_a"]
        chiS = pWFHM["chi_s"]
        eta = derived[:, 3:4]
        delta = derived[:, 4:5]
        PI = math.pi
        pi_p = self._initialize_powers(math.pi)
        log = math.log

        # Global factors of each PN hlm (internals.c:986-987)
        prefactors = (math.sqrt(2) / 3., 0.75 * math.sqrt(5 / 7.),
                      math.sqrt(5 / 7.) / 3., 4 * math.sqrt(2) / 9 * math.sqrt(5 / 7.))
        pn = {
            "PNglobalfactor": (2. / pWFHM["emm"]) ** (-7 / 6.) * prefactors[pWFHM["modeInt"]],
            "pnInitial": 0.0,
        }

        tag = pWFHM["modeTag"]
        if tag == 21:
            p2 = self._initialize_powers(2.0)
            pn["pnOneThird"] = delta * pi_p["one_third"] * p2["one_third"]
            pn["pnTwoThirds"] = (-3 * (chiA + chiS * delta)) / 2. * pi_p["two_thirds"] * p2["two_thirds"]
            pn["pnThreeThirds"] = (335 * delta + 1404 * delta * eta) / 672. * pi_p["itself"] * p2["itself"]
            pn["pnFourThirds"] = (3427 * chiA - (1j * 672) * delta + 3427 * chiS * delta - 8404 * chiA * eta - 3860 * chiS * delta * eta - 1344 * delta * PI - (1j * 672) * delta * log(16)) / 1344. * pi_p["four_thirds"] * p2["four_thirds"]
            pn["pnFiveThirds"] = (-155965824 * chiA * chiS - 964357 * delta + 432843264 * chiA * chiS * eta - 23670792 * delta * eta + 24385536 * chiA * PI + 24385536 * chiS * delta * PI - 77982912 * delta * chiA * chiA + 81285120 * delta * eta * chiA * chiA - 77982912 * delta * chiS * chiS + 39626496 * delta * eta * chiS * chiS + 21535920 * delta * eta * eta) / 8.128512e6 * pi_p["five_thirds"] * p2["five_thirds"]
            pn["pnSixThirds"] = (143063173 * chiA - (1j * 1350720) * delta + 143063173 * chiS * delta - 546199608 * chiA * eta - (1j * 72043776) * delta * eta - 169191096 * chiS * delta * eta - 9898560 * delta * PI + 20176128 * delta * eta * PI - (1j * 5402880) * delta * log(2) - (1j * 17224704) * delta * eta * log(2) + 61725888 * chiS * delta * chiA * chiA - 81285120 * chiS * delta * eta * chiA * chiA + 20575296 * chiA ** 3 - 81285120 * eta * chiA ** 3 + 61725888 * chiA * chiS * chiS - 165618432 * chiA * eta * chiS * chiS + 20575296 * delta * chiS ** 3 - 1016064 * delta * eta * chiS * chiS * chiS + 128873808 * chiA * eta * eta - 3859632 * chiS * delta * eta * eta) / 5.419008e6 * pi_p["two"] * p2["two"]
        elif tag == 33:
            p2d3 = self._initialize_powers(2. / 3.)
            pn["pnOneThird"] = delta * pi_p["one_third"] * p2d3["one_third"]
            pn["pnTwoThirds"] = torch.zeros_like(eta)
            pn["pnThreeThirds"] = (-1945 * delta + 2268 * delta * eta) / 672. * pi_p["itself"] * p2d3["itself"]
            pn["pnFourThirds"] = (325 * chiA - (1j * 504) * delta + 325 * chiS * delta - 1120 * chiA * eta - 80 * chiS * delta * eta + 120 * delta * PI + (1j * 720) * delta * log(1.5)) / 120. * pi_p["four_thirds"] * p2d3["four_thirds"]
            pn["pnFiveThirds"] = (-2263282560 * chiA * chiS - 1077664867 * delta + 9053130240 * chiA * chiS * eta - 5926068792 * delta * eta - 1131641280 * delta * chiA * chiA + 4470681600 * delta * eta * chiA * chiA - 1131641280 * delta * chiS * chiS + 55883520 * delta * eta * chiS * chiS + 2966264784 * delta * eta * eta) / 4.4706816e8 * pi_p["five_thirds"] * p2d3["five_thirds"]
            pn["pnSixThirds"] = (22007835 * chiA + (1j * 26467560) * delta + 22007835 * chiS * delta - 80190540 * chiA * eta - (1j * 98774368) * delta * eta - 31722300 * chiS * delta * eta - 9193500 * delta * PI + 17826480 * delta * eta * PI - (1j * 37810800) * delta * log(1.5) + (1j * 37558080) * delta * eta * log(1.5) - 12428640 * chiA * eta * eta - 6078240 * chiS * delta * eta * eta) / 2.17728e6 * pi_p["two"] * p2d3["two"]
        elif tag == 32:
            pn["pnOneThird"] = torch.zeros_like(eta)
            pn["pnTwoThirds"] = (-1 + 3 * eta) * pi_p["two_thirds"]
            pn["pnThreeThirds"] = -4 * chiS * eta * pi_p["itself"]
            pn["pnFourThirds"] = (10471 - 61625 * eta + 82460 * eta * eta) / 10080. * pi_p["four_thirds"]
            pn["pnFiveThirds"] = ((1j * 2520) - 3955 * chiS - 3955 * chiA * delta - (1j * 11088) * eta + 10810 * chiS * eta + 11865 * chiA * delta * eta - 12600 * chiS * eta * eta) / 840. * pi_p["five_thirds"]
            pn["pnSixThirds"] = (824173699 + 2263282560 * chiA * chiS * delta - 26069649 * eta - 15209631360 * chiA * chiS * delta * eta + 3576545280 * chiS * eta * PI + 1131641280 * chiA * chiA - 7865605440 * eta * chiA * chiA + 1131641280 * chiS * chiS - 11870591040 * eta * chiS * chiS - 13202119896 * eta * eta + 13412044800 * chiA * chiA * eta * eta + 5830513920 * chiS * chiS * eta * eta + 5907445488 * eta ** 3) / 4.4706816e8 * pi_p["two"]
        elif tag == 44:
            p2d4 = self._initialize_powers(0.5)
            pn["pnOneThird"] = torch.zeros_like(eta)
            pn["pnTwoThirds"] = (1 - 3 * eta) * pi_p["two_thirds"] * p2d4["two_thirds"]
            pn["pnThreeThirds"] = torch.zeros_like(eta)
            pn["pnFourThirds"] = (-158383 + 641105 * eta - 446460 * eta * eta) / 36960. * pi_p["four_thirds"] * p2d4["four_thirds"]
            pn["pnFiveThirds"] = ((1j * -1008) + 565 * chiS + 565 * chiA * delta + (1j * 3579) * eta - 2075 * chiS * eta - 1695 * chiA * delta * eta + 240 * PI - 720 * eta * PI + (1j * 960) * log(2) - (1j * 2880) * eta * log(2) + 1140 * chiS * eta * eta) / 120. * pi_p["five_thirds"] * p2d4["five_thirds"]
            pn["pnSixThirds"] = (7888301437 - 147113366400 * chiA * chiS * delta - 745140957231 * eta + 441340099200 * chiA * chiS * delta * eta - 73556683200 * chiA * chiA + 511264353600 * eta * chiA * chiA - 73556683200 * chiS * chiS + 224302478400 * eta * chiS * chiS + 2271682065240 * eta * eta - 871782912000 * chiA * chiA * eta * eta - 10897286400 * chiS * chiS * eta * eta - 805075876080 * eta ** 3) / 2.90594304e10 * pi_p["two"] * p2d4["two"]
        else:
            raise ValueError(f"unsupported modeTag {tag}")
        return pn

    def get_hm_amp_coeffs(self, pWFHM, derived, amp22_coeffs=None,
                          phase22_coeffs=None, spheroidal=None):
        """
        Full amplitude-coefficient dict for one mode (122022 path).

        Mirrors IMRPhenomXHM_GetAmplitudeCoefficients
        (internals.c:1146-1274): region cuts, PN prefactors, inspiral
        InspiralCoefficient[0..2] (closed forms, inspiral.c:893-943),
        ringdown coefficients + exponential falloff pair + RDAux
        polynomial (ringdown.c:727-892), intermediate polynomial via the
        batched collocation linear solve (intermediate.c:2436-2629).

        Rescale factors are all 0 in 122022 (internals.c:1193-1195):
        every region ansatz returns strain directly.  PNdominantlm /
        PNdominantlmpower are consumed only by RescaleFactor flags != 0
        and are not computed.

        The mixing mode (3,2) additionally needs the spheroidal ringdown
        (its intermediate right boundary is |SpheroidalToSpherical|);
        pass the spheroidal-phase context via ``spheroidal`` together
        with amp22/phase22 coefficient dicts.
        """
        eta = derived[:, 3:4]
        amp = {"ampNorm": pWFHM["ampNorm"]}

        # PNdominant = ampNorm * (2/emm)^(-7/6); misses the f^(-7/6)
        # (internals.c:1150)
        amp["PNdominant"] = pWFHM["ampNorm"] * (2. / pWFHM["emm"]) ** (-7 / 6.)

        # IMRPhenomXHMIntermediateAmpVersion digits: 0 skip point, 1 value,
        # 2 value + derivative (boundaries only) (internals.c:1198-1208)
        digits = [int(c) for c in str(pWFHM["IMRPhenomXHMIntermediateAmpVersion"])]
        amp["VersionCollocPtsInter"] = digits
        amp["nCoefficientsInter"] = sum(digits)

        amp["nCoefficientsRDAux"] = 0
        if pWFHM["MixingOn"]:
            amp["nCollocPtsRDAux"] = 2
            amp["nCoefficientsRDAux"] = 4
            amp["fRDAux"] = pWFHM["fRING"] - pWFHM["fDAMP"]  # v2

        # Take the cutting frequencies at the inspiral and ringdown
        # (internals.c:1222-1224)
        amp["fAmpMatchIN"] = self._amp_fcut_insp(pWFHM, derived)
        amp["fAmpMatchIM"] = self._amp_fcut_rd(pWFHM, phase22_coeffs)
        pWFHM["fAmpRDfalloff"] = pWFHM["fRING"] + 2 * pWFHM["fDAMP"]

        # Take Frequency Domain Post-Newtonian Amplitude Coefficients
        amp.update(self._get_pn_amplitude_coeffs(pWFHM, derived))

        self._get_inspiral_amp_coefficients(amp, pWFHM, derived)
        self._rd_amp_coefficients(amp, pWFHM, derived)
        self._get_intermediate_amp_coefficients(
            amp, pWFHM, derived, amp22_coeffs, phase22_coeffs, spheroidal
        )
        return amp

    def _insp_pn_amp_series(self, powers_Mf, amp):
        """
        |PN power series| of the FD inspiral ansatz (no pseudo-PN terms,
        no rescale).  Source: IMRPhenomXHM_Inspiral_PNAmp_Ansatz,
        inspiral.c:677-684.
        """
        series = (amp["pnInitial"]
                  + powers_Mf["one_third"] * amp["pnOneThird"]
                  + powers_Mf["two_thirds"] * amp["pnTwoThirds"]
                  + powers_Mf["itself"] * amp["pnThreeThirds"]
                  + powers_Mf["four_thirds"] * amp["pnFourThirds"]
                  + powers_Mf["five_thirds"] * amp["pnFiveThirds"]
                  + powers_Mf["two"] * amp["pnSixThirds"])
        return torch.abs(series)

    def _get_inspiral_amp_coefficients(self, amp, pWFHM, derived):
        """
        Collocation points and pseudo-PN coefficients of the inspiral
        amplitude, version 123 (all three points; the InspiralAmpVeto
        that lowers the version is 122019-only).

        Source: IMRPhenomXHM_Get_Inspiral_Amp_Coefficients
        (inspiral.c:841-891) and IMRPhenomXHM_Inspiral_Amp_Coefficients
        case 123 (inspiral.c:936-940).
        """
        fits = _INSP_AMP_FITS[pWFHM["modeTag"]]
        pfit = self._pwf_fit_dict(derived)

        # CollocationPointsFreqsAmplitudeInsp = {0.5, 0.75, 1} * fAmpMatchIN
        # (inspiral.c:876-878)
        fIN = amp["fAmpMatchIN"]
        p1 = self._initialize_powers(0.5 * fIN)
        p2 = self._initialize_powers(0.75 * fIN)
        p3 = self._initialize_powers(fIN)
        pcut = p3

        amp["fcutInsp_seven_thirds"] = pcut["seven_thirds"]
        amp["fcutInsp_eight_thirds"] = pcut["eight_thirds"]
        amp["fcutInsp_three"] = pcut["three"]

        # Collocation values (fabs of the parameter-space fits,
        # inspiral.c:884-886) and PN values at the same frequencies
        # (strain; rescale factor 0)
        iv = [torch.abs(f(pfit)) for f in fits]
        pn = [self._insp_pn_amp_series(p, amp) * amp["PNglobalfactor"]
              * p["m_seven_sixths"] * pWFHM["ampNorm"] for p in (p1, p2, p3)]

        # v_i = (iv_i - PN_i) / PNdominant / f_i^(-7/6) (inspiral.c:904-906)
        v1 = (iv[0] - pn[0]) / amp["PNdominant"] / p1["m_seven_sixths"]
        v2 = (iv[1] - pn[1]) / amp["PNdominant"] / p2["m_seven_sixths"]
        v3 = (iv[2] - pn[2]) / amp["PNdominant"] / p3["m_seven_sixths"]

        # case 123 (inspiral.c:936-940)
        f1, f2, f3, finsp = p1, p2, p3, pcut
        denom = (f1["seven_thirds"] * (f1["one_third"] - f2["one_third"]) * f2["seven_thirds"]
                 * (f1["one_third"] - f3["one_third"]) * (f2["one_third"] - f3["one_third"])
                 * f3["seven_thirds"])
        amp["InspiralCoefficient0"] = (finsp["seven_thirds"] * (
            -(f1["three"] * f3["eight_thirds"] * v2) + f1["eight_thirds"] * f3["three"] * v2
            + f2["three"] * (f3["eight_thirds"] * v1 - f1["eight_thirds"] * v3)
            + f2["eight_thirds"] * (-(f3["three"] * v1) + f1["three"] * v3))) / denom
        amp["InspiralCoefficient1"] = (finsp["eight_thirds"] * (
            f1["three"] * f3["seven_thirds"] * v2 - f1["seven_thirds"] * f3["three"] * v2
            + f2["three"] * (-(f3["seven_thirds"] * v1) + f1["seven_thirds"] * v3)
            + f2["seven_thirds"] * (f3["three"] * v1 - f1["three"] * v3))) / denom
        amp["InspiralCoefficient2"] = (finsp["three"] * (
            f1["seven_thirds"] * (-f1["one_third"] + f3["one_third"]) * f3["seven_thirds"] * v2
            + f2["seven_thirds"] * (-(f3["eight_thirds"] * v1) + f1["eight_thirds"] * v3)
            + f2["eight_thirds"] * (f3["seven_thirds"] * v1 - f1["seven_thirds"] * v3))) / denom

    def _rd_amp_coefficients(self, amp, pWFHM, derived):
        """
        Ringdown amplitude coefficients RDCoefficient[0..4] (+ RDAux for
        the 32): version 2 builds a_lambda/lambda/sigma from the rdcp
        collocation values with the monotonicity vetoes; version 1 (32)
        takes the fitted coefficients directly; the exponential-falloff
        pair is imposed at fAmpRDfalloff = fRING + 2 fDAMP.

        Source: IMRPhenomXHM_RD_Amp_Coefficients, ringdown.c:727-816.
        """
        pfit = self._pwf_fit_dict(derived)
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]

        if pWFHM["IMRPhenomXHMRingdownAmpVersion"] == 1:
            alam, lam, sig = _RD_AMP_32_COEFF_FITS
            rd0 = torch.abs(alam(pfit))
            rd1 = lam(pfit)
            rd2 = sig(pfit)
        elif pWFHM["IMRPhenomXHMRingdownAmpVersion"] == 2:
            f1, f2, f3 = _RD_AMP_RDCP_FITS[pWFHM["modeTag"]]
            rdcp1 = torch.abs(f1(pfit))
            rdcp2 = torch.abs(f2(pfit))
            rdcp3 = torch.abs(f3(pfit))
            # Apply vetos to RDCP (sequential; ringdown.c:771-779)
            rdcp3 = torch.where(rdcp3 >= rdcp2 * rdcp2 / rdcp1,
                                0.5 * rdcp2 * rdcp2 / rdcp1, rdcp3)
            rdcp3 = torch.where(rdcp3 > rdcp2, 0.5 * rdcp2, rdcp3)
            rdcp3 = torch.where((rdcp1 < rdcp2) & (rdcp3 > rdcp1), rdcp1, rdcp3)
            deno = torch.sqrt(rdcp1 / rdcp3) - (rdcp1 / rdcp2)
            deno = torch.where(deno <= 0, torch.full_like(deno, 1e-16), deno)
            rd0 = rdcp1 * fda / deno
            rd2 = torch.sqrt(rd0 / (rdcp2 * fda))
            rd1 = 0.5 * rd2 * torch.log(rdcp1 / rdcp3)
        else:
            raise ValueError("IMRPhenomXHMRingdownAmpVersion is not valid")
        amp["RDCoefficient0"] = rd0
        amp["RDCoefficient1"] = rd1
        amp["RDCoefficient2"] = rd2

        # Exponential falloff pair at fAmpRDfalloff (ringdown.c:799-807);
        # the ansatz there is the plain Lorentzian branch.
        ff = pWFHM["fAmpRDfalloff"]
        dfr = ff - frd
        dfd = fda * rd2
        rd3 = rd0 * fda / (torch.exp(rd1 / dfd * dfr) * (dfr * dfr + dfd * dfd))
        amp["RDCoefficient3"] = rd3
        amp["RDCoefficient4"] = -1. * self._rd_amp_dansatz(ff, pWFHM, amp) / rd3

        if amp["nCoefficientsRDAux"] > 0:
            self._rdaux_amp_coefficients(amp, pWFHM, derived)

    def _rd_amp_dansatz(self, Mf, pWFHM, amp):
        """
        Analytic derivative of the ringdown Lorentzian (versions 1/2).
        Source: IMRPhenomXHM_RD_Amp_DAnsatz, ringdown.c:981-989.
        """
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]
        rd0, rd1, rd2 = amp["RDCoefficient0"], amp["RDCoefficient1"], amp["RDCoefficient2"]
        dfr = Mf - frd
        numerator = rd0 * (dfr * dfr * rd1 + 2 * fda * dfr * rd2
                           + fda * fda * rd1 * rd2 * rd2)
        denom = (dfr * dfr + fda * fda * rd2 * rd2)
        denom = rd2 * denom * denom * torch.exp(dfr * rd1 / (fda * rd2))
        return -numerator / denom

    def _rd_amp_lorentzian(self, Mf, pWFHM, amp):
        """
        Lorentzian with exponential decay (versions 1/2 core branch).
        Source: IMRPhenomXHM_RD_Amp_Ansatz, ringdown.c:934-937.
        """
        dfr = Mf - pWFHM["fRING"]
        dfd = pWFHM["fDAMP"] * amp["RDCoefficient2"]
        return amp["RDCoefficient0"] * pWFHM["fDAMP"] / (
            torch.exp(amp["RDCoefficient1"] / dfd * dfr) * (dfr * dfr + dfd * dfd))

    def _rdaux_amp_coefficients(self, amp, pWFHM, derived):
        """
        Polynomial patch of the (3,2) spheroidal ringdown on
        [fAmpMatchIM, fRDAux]: two fitted collocation values plus value
        and derivative continuity at fRDAux; 4x4 batched solve.
        Source: IMRPhenomXHM_RDAux_Amp_Coefficients, ringdown.c:818-892.
        """
        pfit = self._pwf_fit_dict(derived)
        fRDAux = amp["fRDAux"]
        vals = [torch.abs(f(pfit)) for f in _RD_AMP_32_RDAUX_FITS]
        vals.append(self._rd_amp_lorentzian(fRDAux, pWFHM, amp))
        vals.append(self._rd_amp_dansatz(fRDAux, pWFHM, amp))
        freqs = [amp["fAmpMatchIM"], 0.5 * (amp["fAmpMatchIM"] + fRDAux),
                 fRDAux, fRDAux]

        # A = (1, fi, fi^2, fi^3) rows + derivative row (0, 1, 2f, 3f^2)
        # (ringdown.c:851-878)
        n = amp["nCoefficientsRDAux"]
        B = derived.shape[0]
        A = torch.empty((B, n, n), dtype=derived.dtype, device=derived.device)
        b = torch.empty((B, n, 1), dtype=derived.dtype, device=derived.device)
        for i in range(n):
            b[:, i, :] = vals[i]
            fc = freqs[i]
            if i < n - 1:
                fpower = torch.ones_like(fc)
                for j in range(n):
                    A[:, i, j:j + 1] = fpower
                    fpower = fpower * fc
            else:
                fpower = torch.ones_like(fc)
                A[:, i, 0:1] = 0.
                for j in range(1, n):
                    A[:, i, j:j + 1] = j * fpower
                    fpower = fpower * fc
        x = torch.linalg.solve(A, b)
        for i in range(n):
            amp[f"RDAuxCoefficient{i}"] = x[:, i, :]

    def _insp_amp_ndansatz(self, Mf, pWFHM, amp):
        """
        Numerical derivative of the inspiral ansatz (4th order finite
        differences, df = 10e-10), used for reconstructing the
        intermediate region.
        Source: IMRPhenomXHM_Inspiral_Amp_NDAnsatz, inspiral.c:777-798.
        """
        df = 10e-10
        fun2R = self._insp_amp_ansatz(self._initialize_powers(Mf + 2 * df), pWFHM, amp)
        funR = self._insp_amp_ansatz(self._initialize_powers(Mf + df), pWFHM, amp)
        funL = self._insp_amp_ansatz(self._initialize_powers(Mf - df), pWFHM, amp)
        fun2L = self._insp_amp_ansatz(self._initialize_powers(Mf - 2 * df), pWFHM, amp)
        return (-fun2R + 8 * funR - 8 * funL + fun2L) / (12 * df)

    def _get_intermediate_amp_coefficients(self, amp, pWFHM, derived,
                                           amp22_coeffs, phase22_coeffs,
                                           spheroidal):
        """
        Intermediate amplitude coefficients via the batched collocation
        linear solve: equispaced frequencies between fAmpMatchIN and
        fAmpMatchIM (FreqsVersion 0), boundary values/derivatives from
        the inspiral and ringdown ansatze, inner values from the
        parameter-space fits; matrix rows are f^j * f^(-7/6) (values) and
        (j - 7/6) f^(j-1) f^(-7/6) (derivatives).

        Source: IMRPhenomXHM_Intermediate_Amp_CollocationPoints
        (intermediate.c:2436-2536) and
        IMRPhenomXHM_Intermediate_Amp_Coefficients
        (intermediate.c:2538-2629).
        """
        digits = amp["VersionCollocPtsInter"]
        n = amp["nCoefficientsInter"]
        npts = pWFHM["nCollocPtsInterAmp"]
        fIN = amp["fAmpMatchIN"]
        fIM = amp["fAmpMatchIM"]
        deltaf = (fIM - fIN) / (npts - 1)

        rows = []  # (freq, is_derivative, value)
        # Left boundary: inspiral ansatz value (+ 4th-order FD derivative)
        pIN = self._initialize_powers(fIN)
        if digits[0] >= 1:
            rows.append((fIN, False, self._insp_amp_ansatz(pIN, pWFHM, amp)))
        if digits[0] == 2:
            rows.append((fIN, True, self._insp_amp_ndansatz(fIN, pWFHM, amp)))

        # Inner points: |parameter-space fits| (intermediate.c:2489-2500);
        # i <= 2 -> int{i}, i >= 3 -> int{i} from the 122022 extra slots
        pfit = self._pwf_fit_dict(derived)
        for i in range(1, npts - 1):
            if digits[i] == 1:
                fit = _INTER_AMP_FITS[pWFHM["modeTag"]][i]
                rows.append((fIN + deltaf * i, False, torch.abs(fit(pfit))))

        # Right boundary: ringdown ansatz (spheroidal->spherical for the
        # 32) value + derivative (intermediate.c:2506-2530)
        if pWFHM["MixingOn"] == 0:
            rd_val = self._rd_amp_ansatz(self._initialize_powers(fIM), pWFHM, amp)
            rd_der = self._rd_amp_dansatz(fIM, pWFHM, amp)
        else:
            if spheroidal is None or amp22_coeffs is None or phase22_coeffs is None:
                raise NotImplementedError(
                    "(3,2) intermediate boundary needs the spheroidal context"
                )
            rd_val, rd_der = self._spheroidal_boundary_value_derivative(
                fIM, derived, pWFHM, amp, spheroidal, amp22_coeffs, phase22_coeffs
            )
        if digits[npts - 1] >= 1:
            rows.append((fIM, False, rd_val))
        if digits[npts - 1] == 2:
            rows.append((fIM, True, rd_der))

        assert len(rows) == n, "inconsistent number of collocation points"

        B = derived.shape[0]
        A = torch.empty((B, n, n), dtype=derived.dtype, device=derived.device)
        b = torch.empty((B, n, 1), dtype=derived.dtype, device=derived.device)
        seven_sixths = 7 / 6.
        for i, (fc, is_der, val) in enumerate(rows):
            b[:, i, :] = val
            fc_m76 = fc ** (-seven_sixths)
            if not is_der:
                fpower = torch.ones_like(fc)
                for j in range(n):
                    A[:, i, j:j + 1] = fpower * fc_m76
                    fpower = fpower * fc
            else:
                fpower = 1. / fc
                for j in range(n):
                    A[:, i, j:j + 1] = (j - seven_sixths) * fpower * fc_m76
                    fpower = fpower * fc
        x = torch.linalg.solve(A, b)
        for i in range(n):
            amp[f"InterCoefficient{i}"] = x[:, i, :]

    # ------------------------------------------------------------------
    # Phase: parameter-space fits
    # ------------------------------------------------------------------

    def _phase_colloc_freqs(self, pWFHM, derived, phase22_coeffs):
        """
        Intermediate-phase collocation frequencies and region boundaries.

        fcut = GetfcutInsp = (1 + 0.001 (0.25/eta - 1)) (m/2) fMECO;
        non-32: 6 nodes between fcut and fRING, fPhaseMatchIM =
        fRING - fDAMP; 32: nodes toward fEnd = fRING22 - 0.5 fDAMP22 with
        the last two duplicated at fEnd (first and second derivative),
        and the 122019 EMR negative-spin correction to fPhaseMatchIM.
        fPhaseMatchIN = fMECOlm.
        Source: GetfcutInsp (internals.c:867-872) and
        IMRPhenomXHM_Intermediate_CollocPtsFreqs (internals.c:876-927).
        """
        eta = derived[:, 3:4]
        chiPNHat = derived[:, 8:9]
        dchi = derived[:, 10:11]
        delta = derived[:, 4:5]
        fring = pWFHM["fRING"]
        fdamp = pWFHM["fDAMP"]

        fMECO = self.fMECO(eta, chiPNHat, dchi, delta)
        eta_factor = 1.0 + 0.001 * (0.25 / eta - 1.)
        fcut = eta_factor * pWFHM["emm"] * 0.5 * fMECO

        out = {"fPhaseMatchIN": pWFHM["fMECOlm"]}
        if pWFHM["modeTag"] == 32:
            fRD22 = phase22_coeffs["fRING"]
            fdamp22 = phase22_coeffs["fDAMP"]
            fEnd = fRD22 - 0.5 * fdamp22
            freqs = [fcut,
                     (math.sqrt(3) * (fcut - fEnd) + 2 * (fcut + fEnd)) / 4.,
                     (3 * fcut + fEnd) / 4.,
                     (fcut + fEnd) / 2.,
                     fEnd, fEnd]
            # correct cutting frequency for EMR with negative spins
            # (version 122019; internals.c:903-905)
            chi1L = derived[:, 11:12]
            fIM = torch.where((eta < 0.01) & (chi1L < 0),
                              fEnd * (1.2 - 0.25 * chi1L), fEnd)
            out["fPhaseMatchIM"] = fIM
        else:
            freqs = [fcut,
                     (math.sqrt(3) * (fcut - fring) + 2 * (fcut + fring)) / 4.,
                     (3 * fcut + fring) / 4.,
                     (fcut + fring) / 2.,
                     (fcut + 3 * fring) / 4.,
                     (fcut + 7 * fring) / 8.]
            out["fPhaseMatchIM"] = fring - fdamp
        out["CollocationPointsFreqsPhaseInter"] = freqs
        return out

    def get_hm_phase_coeffs(self, pWFHM, derived, theta, phase22_coeffs,
                            amp_coeffs=None, amp22_coeffs=None, spheroidal=None):
        """
        Full phase-coefficient dict for one mode.

        Mirrors IMRPhenomXHM_GetPhaseCoefficients (internals.c:1952-2461):
        - collocation values from the p1..p6 fits with DeltaT added
          (time-shift wf so that modes peak around t=0);
        - inspiral: XAS 22 coefficients rescaled,
          phi_lm(f) ~ m/2 phi_22(2/m f), plus LambdaPN (2013-2055);
        - the 21 high-spin inspiral extension of the first two collocation
          values (2079-2094);
        - non-mixing: per-sample choice of 5 of the 6 collocation points
          (2102-2122), ringdown alpha2/alphaL rescaling of the 22 fits
          with wlm = 2 (l == m) else m/3 (2139-2147), batched 5x5 solve
          on {1, Lorentzian, 1/f, 1/f^2, 1/f^4} (2164-2175, 2280-2287);
        - mixing (32): dphi/df and d2phi/df2 at fcutRD from FD schemes on
          carg(SpheroidalToSpherical) (fstep = 1e-7), 6x6 solve with the
          derivative row and the extra 1/f^3 coefficient c3, and the EMR
          c0 glue (2184-2308); requires amp_coeffs / amp22_coeffs /
          spheroidal, and timeshift/phiref22 already set by
          get_spheroidal_coeffs;
        - region connection constants C1INSP/CINSP, C1RD/CRD (2325-2332);
        - timeshift (= XAS linb), phiref22 and the mode alignment
          deltaphiLM with the (2,1) pi flip (2360-2407).
        """
        if pWFHM["MixingOn"] == 1 and (
                spheroidal is None or amp_coeffs is None or amp22_coeffs is None):
            raise ValueError("(3,2) phase needs spheroidal/amp/amp22 context")

        eta = derived[:, 3:4]
        delta = derived[:, 4:5]
        chiPNHat = derived[:, 8:9]
        STotR = derived[:, 9:10]
        dchi = derived[:, 10:11]
        M_s = derived[:, 2:3]
        phic = theta[:, 6:7]
        ell, emm = pWFHM["ell"], pWFHM["emm"]
        m_over_2 = emm * 0.5
        two_over_m = 1. / m_over_2
        pfit = self._pwf_fit_dict(derived)

        pPhase = self._phase_colloc_freqs(pWFHM, derived, phase22_coeffs)
        freqs = pPhase["CollocationPointsFreqsPhaseInter"]

        # for each collocation point, call fit giving the value of the
        # phase derivative at that point; DeltaT shifts the modes to peak
        # around t=0 (internals.c:1997-2003)
        fits = _INTER_PHASE_FITS[pWFHM["modeTag"]]
        values = [fits[i](pfit) + pWFHM["DeltaT"] for i in range(6)]

        fcutRD = pPhase["fPhaseMatchIM"]
        fcutInsp = pPhase["fPhaseMatchIN"]

        # ---- Inspiral: rescale PhenX and apply PN corrections ----
        # (internals.c:2013-2055)
        z = torch.zeros_like(eta)
        phenXnonLog = [phase22_coeffs[f"phi{i}"] for i in range(10)] + [z, z, z]
        phenXLog = [z, z, z, z, z, phase22_coeffs["phi5L"], phase22_coeffs["phi6L"],
                    z, phase22_coeffs["phi8L"], phase22_coeffs["phi9L"], z, z, z]
        pseudoPN = [z] * 8 + [phase22_coeffs[f"sigma{i}"] for i in range(1, 6)]
        fact = phase22_coeffs["phiNorm"] / eta
        log_m_over_2 = math.log(m_over_2)
        phi, phiL = [], []
        for i in range(13):
            scale = m_over_2 ** ((8 - i) / 3.)
            nl = phenXnonLog[i] * fact
            lg = phenXLog[i] * fact
            ps = pseudoPN[i] * fact
            phi.append((nl + ps - lg * log_m_over_2) * scale)
            phiL.append(lg * scale)

        # LambdaPN: analytic PN correction for eta > 0.01, else fit
        # (internals.c:2039-2055)
        lambda_analytic = _fits_insp.IMRPhenomXHM_Insp_Phase_LambdaPN(eta, pWFHM["modeTag"])
        lambda_fit = _INSP_PHASE_LAMBDA_FITS[pWFHM["modeTag"]](pfit)
        LambdaPN = torch.where(eta > 0.01, lambda_analytic, lambda_fit)
        phi[8] = phi[8] + LambdaPN
        pPhase["phi"] = phi
        pPhase["phiL"] = phiL

        fring = pWFHM["fRING"]
        fdamp = pWFHM["fDAMP"]
        n = pWFHM["nCollocPtsInterPhase"]
        B = derived.shape[0]
        A = torch.empty((B, n, n), dtype=derived.dtype, device=derived.device)
        b = torch.empty((B, n, 1), dtype=derived.dtype, device=derived.device)

        if pWFHM["MixingOn"] == 0:
            # ---- 21 high-spin: extend inspiral into the intermediate
            # region (internals.c:2079-2094) ----
            if pWFHM["modeTag"] == 21:
                insp_vals = []
                for i in range(3):
                    FF = two_over_m * freqs[i]
                    insp_vals.append((1. / eta) * self.dphase(FF, phase22_coeffs, derived))
                diff12 = insp_vals[0] - insp_vals[1]
                diff23 = insp_vals[1] - insp_vals[2]
                new1 = values[2] + diff23
                new0 = new1 + diff12
                mask = STotR >= 0.8
                values[1] = torch.where(mask, new1, values[1])
                values[0] = torch.where(mask, new0, values[0])

            # ---- choose collocation points according to spin/mass ratio
            # (internals.c:2102-2122): pick 5 out of 6 available fits ----
            idxA = torch.tensor([0, 1, 3, 4, 5], device=derived.device)
            idxB = torch.tensor([0, 1, 2, 4, 5], device=derived.device)
            idxC = torch.tensor([0, 1, 2, 3, 5], device=derived.device)
            condA = (eta < pWFHM["etaEMR"]) | ((STotR >= 0.8) if emm == ell else torch.zeros_like(eta, dtype=torch.bool))
            if pWFHM["modeTag"] == 33:
                condA = condA | (STotR < 0)
            if pWFHM["modeTag"] == 21:
                condB = STotR >= 0.8
            else:
                condB = torch.zeros_like(eta, dtype=torch.bool)
            idx = torch.where(condA, idxA.unsqueeze(0),
                              torch.where(condB, idxB.unsqueeze(0), idxC.unsqueeze(0)))

            freqs_all = torch.cat(freqs, dim=1)
            values_all = torch.cat(values, dim=1)
            f_sel = torch.gather(freqs_all, 1, idx)
            v_sel = torch.gather(values_all, 1, idx)

            # ---- Ringdown: rescale the 22 fits (internals.c:2139-2157) ----
            wlm = 2. if ell == emm else emm / 3.
            pPhase["alpha0"] = z
            pPhase["alpha2"] = 1. / (fring * fring) * wlm * _RD_PHASE_22_ALPHA2(pfit)
            pPhase["alphaL"] = (1. / eta) * _RD_PHASE_22_ALPHAL(pfit)
            pPhase["phi0RD"] = self._rd_phase_ansatz_int(fcutRD, pWFHM, pPhase)
            pPhase["dphi0RD"] = self._rd_phase_ansatz(fcutRD, pWFHM, pPhase)

            # ---- Intermediate: batched linear solve on
            # {1, Lorentzian, 1/f, 1/f^2, 1/f^4} (internals.c:2164-2175) ----
            for i in range(n):
                ff = f_sel[:, i:i + 1]
                ffm1 = 1. / ff
                ffm2 = ffm1 * ffm1
                basis = [torch.ones_like(ff),
                         fdamp / (fdamp * fdamp + (ff - fring) * (ff - fring)),
                         ffm1, ffm2, ffm2 * ffm2]
                b[:, i, :] = v_sel[:, i:i + 1]
                for j in range(n):
                    A[:, i, j:j + 1] = basis[j]
        else:
            # ---- mode-mixing on: dphi/df and d2phi/df2 at fcutRD from
            # rotation of the spheroidal ansatz enter the linear system
            # (internals.c:2184-2254) ----
            fstep = self._S2S_FSTEP
            phase_args = []
            for k in (-1, 0, 1):
                S = self._spheroidal_to_spherical(
                    fcutRD + k * fstep, derived, pWFHM, amp_coeffs,
                    spheroidal, amp22_coeffs, phase22_coeffs)
                # make sure that all the three points belong to the same
                # branch of mod (internals.c:2198-2206)
                a = torch.fmod(torch.angle(S), 2. * math.pi)
                phase_args.append(torch.where(a > 0, a - 2. * math.pi, a))
            # store spherical-harmonic phase at fcutRD; FD schemes give
            # first and second phase derivatives (internals.c:2209-2213)
            pPhase["phi0RD"] = phase_args[1]
            pPhase["dphi0RD"] = 0.5 / fstep * (phase_args[2] - phase_args[0])
            d2phi0RD = (phase_args[2] - 2. * phase_args[1] + phase_args[0]) / (fstep * fstep)

            # first derivative: FD value for non-extreme mass ratio, else
            # keep the fit (glued later); second derivative always FD
            # (internals.c:2225-2231)
            emr = eta > pWFHM["etaEMR"]
            row_freqs = list(freqs)
            row_vals = list(values)
            row_freqs[n - 2] = torch.where(emr, fcutRD, freqs[n - 2])
            row_vals[n - 2] = torch.where(emr, pPhase["dphi0RD"], values[n - 2])
            row_freqs[n - 1] = fcutRD
            row_vals[n - 1] = d2phi0RD

            # value rows on {1, Lorentzian, 1/f, 1/f^2, 1/f^4, 1/f^3};
            # last row = derivative of the ansatz (internals.c:2235-2254)
            for i in range(n):
                ff = row_freqs[i]
                ffm1 = 1. / ff
                ffm2 = ffm1 * ffm1
                if i < n - 1:
                    basis = [torch.ones_like(ff),
                             fdamp / (fdamp * fdamp + (ff - fring) * (ff - fring)),
                             ffm1, ffm2, ffm2 * ffm2, ffm1 * ffm2]
                else:
                    ffm3 = ffm2 * ffm1
                    ffm4 = ffm2 * ffm2
                    ffm5 = ffm3 * ffm2
                    lor_den = fdamp * fdamp + (ff - fring) * (ff - fring)
                    basis = [torch.zeros_like(ff),
                             -2 * fdamp * (ff - fring) / (lor_den * lor_den),
                             -ffm2, -2. * ffm3, -4. * ffm5, -3 * ffm4]
                b[:, i, :] = row_vals[i]
                for j in range(n):
                    A[:, i, j:j + 1] = basis[j]

        x = torch.linalg.solve(A, b)
        pPhase["c0"] = x[:, 0, :]
        pPhase["cL"] = x[:, 1, :]
        pPhase["c1"] = x[:, 2, :]
        pPhase["c2"] = x[:, 3, :]
        pPhase["c4"] = x[:, 4, :]

        # currently the 32 mode is calibrated using one extra point; for
        # extreme mass ratio, glue intermediate and ringdown regions
        # (internals.c:2298-2308)
        if pWFHM["modeTag"] == 32:
            pPhase["c3"] = x[:, 5, :]
            glue = pPhase["dphi0RD"] - self._inter_phase_ansatz(fcutRD, pWFHM, pPhase)
            pPhase["c0"] = torch.where(eta < pWFHM["etaEMR"],
                                       pPhase["c0"] + glue, pPhase["c0"])

        # ---- glue inspiral and intermediate; ringdown and intermediate
        # (internals.c:2325-2332) ----
        pPhase["C1INSP"] = (self._inter_phase_ansatz(fcutInsp, pWFHM, pPhase)
                            - self._insp_phase_ansatz(fcutInsp, pPhase))
        pPhase["CINSP"] = (-(pPhase["C1INSP"] * fcutInsp)
                           + self._inter_phase_ansatz_int(fcutInsp, pWFHM, pPhase)
                           - self._insp_phase_ansatz_int(fcutInsp, pPhase))
        pPhase["C1RD"] = self._inter_phase_ansatz(fcutRD, pWFHM, pPhase) - pPhase["dphi0RD"]
        pPhase["CRD"] = (-pPhase["C1RD"] * fcutRD
                         + self._inter_phase_ansatz_int(fcutRD, pWFHM, pPhase)
                         - pPhase["phi0RD"])

        # ---- 22 reference quantities: timeshift (= IMRPhenomX_TimeShift_22,
        # the XAS linb) and phiref22 (internals.c:2360-2366); for the
        # mixing mode both were already set in GetSpheroidalCoefficients ----
        if pWFHM["MixingOn"] == 0:
            linb_fit = self._linb_fit(eta, STotR, dchi, delta)
            psi4 = self._psi4tostrain_fit(eta, STotR, dchi)
            frefFit = phase22_coeffs["fRING"] - phase22_coeffs["fDAMP"]
            dphi22Ref = (1. / eta) * self.dphase(frefFit, phase22_coeffs, derived)
            timeshift = linb_fit - dphi22Ref - 2. * math.pi * (500. + psi4)
            MfRef = self.f_ref * M_s
            phiref22 = (-1. / eta * self.phase(MfRef, phase22_coeffs, derived)
                        - timeshift * MfRef - pWFHM["phaseshift"]
                        + 2.0 * phic + math.pi / 4.)
            pWFHM["timeshift"] = timeshift
            pWFHM["phiref22"] = phiref22
        else:
            timeshift = pWFHM["timeshift"]
            phiref22 = pWFHM["phiref22"]

        # ---- mode alignment deltaphiLM at falign, Eq. (4.13)
        # (internals.c:2337-2370) ----
        fMECO22 = self.fMECO(eta, chiPNHat, dchi, delta)
        falign = torch.where(eta > pWFHM["etaEMR"],
                             0.6 * m_over_2 * fMECO22, m_over_2 * fMECO22)
        deltaphiLM = (m_over_2 * (1. / eta * self.phase(two_over_m * falign, phase22_coeffs, derived)
                                  + pWFHM["phaseshift"] + phiref22)
                      + timeshift * falign - 3. * math.pi / 4. * (1 - m_over_2)
                      - (self._insp_phase_ansatz_int(falign, pPhase)
                         + pPhase["C1INSP"] * falign + pPhase["CINSP"]))
        deltaphiLM = torch.fmod(deltaphiLM, 2. * math.pi)

        # the sign of the 21 amplitude changes across the parameter space,
        # we add Pi to the phase to account for this (internals.c:2401-2407)
        if pWFHM["modeTag"] == 21:
            ampsign = self._pn21_amp_sign(0.008, derived)
            deltaphiLM = torch.where(ampsign >= 0, deltaphiLM + math.pi, deltaphiLM)
        pPhase["deltaphiLM"] = deltaphiLM
        return pPhase

    @staticmethod
    def _pn21_amp_sign(ff, derived):
        """
        Expansion in frequency of the real part of the 21 PN amplitude;
        its sign flags the pi offset of the 21 phase.
        Source: IMRPhenomXHM_PN21AmpSign, internals.c:1937-1950.
        """
        eta = derived[:, 3:4]
        chi1 = derived[:, 11:12]
        chi2 = derived[:, 12:13]
        delta = torch.sqrt(1 - 4 * eta)
        PI = math.pi
        return (-16 * delta * eta * ff * PI ** 1.5) / (3. * math.sqrt(5)) \
            + (4 * 2 ** 0.3333333333333333 * (chi1 - chi2 + delta * (chi1 + chi2))
               * eta * ff ** 1.3333333333333333 * PI ** 1.8333333333333333) / math.sqrt(5) \
            + (2 * 2 ** 0.6666666666666666 * eta * (306 * delta - 360 * delta * eta)
               * ff ** 1.6666666666666667 * PI ** 2.1666666666666665) / (189. * math.sqrt(5))

    def get_spheroidal_coeffs(self, pWFHM, derived, theta, phase22_coeffs):
        """
        Spheroidal-basis (3,2) ringdown-phase coefficients
        alpha0_S, alphaL_S, alpha2_S, alpha4_S, phi0_S (RingdownPhase
        version 122019), and the 22 reference quantities timeshift /
        phiref22 (stored in pWFHM; for the mixing mode they are computed
        here, not in the phase builder).

        4x4 collocation solve on {1, Lorentzian_lm, f^-2, f^-4} at
        {fRING22, fRINGlm - 1.5 fDAMPlm, fRINGlm - 0.5 fDAMPlm,
        fRINGlm + 0.5 fDAMPlm}; then the relative time shift is imposed
        at frefRD = fRING22 + fDAMP22 via the SpheroidalTimeShift fit and
        the relative phase at frefRD = fRING22 via SpheroidalPhaseShift.
        Source: GetSpheroidalCoefficients (internals.c:2466-2636) and
        IMRPhenomXHM_Ringdown_CollocPtsFreqs (internals.c:940-945).
        """
        eta = derived[:, 3:4]
        STotR = derived[:, 9:10]
        dchi = derived[:, 10:11]
        delta = derived[:, 4:5]
        M_s = derived[:, 2:3]
        phic = theta[:, 6:7]
        pfit = self._pwf_fit_dict(derived)
        fringlm = pWFHM["fRING"]
        fdamplm = pWFHM["fDAMP"]
        fring22 = phase22_coeffs["fRING"]
        fdamp22 = phase22_coeffs["fDAMP"]

        # Collocation freqs, version 122019 (internals.c:940-945)
        freqs = [fring22,
                 fringlm - 1.5 * fdamplm,
                 fringlm - 0.5 * fdamplm,
                 fringlm + 0.5 * fdamplm]
        values = [f(pfit) for f in _RD_PHASE_32_FITS]

        # Solve the linear system of Eq. (6.8) on
        # {1, Lorentzian, f^-2, f^-4} (internals.c:2495-2506)
        n = 4
        B = derived.shape[0]
        A = torch.empty((B, n, n), dtype=derived.dtype, device=derived.device)
        b = torch.empty((B, n, 1), dtype=derived.dtype, device=derived.device)
        for i in range(n):
            ff = freqs[i]
            ffm2 = 1. / (ff * ff)
            basis = [torch.ones_like(ff),
                     fdamplm / (fdamplm * fdamplm + (ff - fringlm) * (ff - fringlm)),
                     ffm2, ffm2 * ffm2]
            b[:, i, :] = values[i]
            for j in range(n):
                A[:, i, j:j + 1] = basis[j]
        x = torch.linalg.solve(A, b)
        sph = {"alpha0_S": x[:, 0, :], "alphaL_S": x[:, 1, :],
               "alpha2_S": x[:, 2, :], "alpha4_S": x[:, 3, :],
               "phi0_S": torch.zeros_like(eta)}

        # ---- time-shift of spheroidal ansatz (internals.c:2560-2575) ----
        frefRD = fring22 + fdamp22
        tshift = _RD_PHASE_32_TIMESHIFT(pfit)
        # timeshift = IMRPhenomX_TimeShift_22 (the XAS linb)
        linb_fit = self._linb_fit(eta, STotR, dchi, delta)
        psi4 = self._psi4tostrain_fit(eta, STotR, dchi)
        dphi22Ref = (1. / eta) * self.dphase(fring22 - fdamp22, phase22_coeffs, derived)
        timeshift = linb_fit - dphi22Ref - 2. * math.pi * (500. + psi4)
        pWFHM["timeshift"] = timeshift
        # impose that dphiS(fref) - dphi22(fref) has the value of the fit
        dphi22ref = (1. / eta) * self.dphase(frefRD, phase22_coeffs, derived) + timeshift
        sph["alpha0_S"] = sph["alpha0_S"] + dphi22ref + tshift \
            - self._rd_phase_ansatz_spheroidal(frefRD, pWFHM, sph)

        # ---- phase-shift of spheroidal ansatz (internals.c:2601-2624) ----
        frefRD = fring22
        MfRef = self.f_ref * M_s
        phiref22 = (-1. / eta * self.phase(MfRef, phase22_coeffs, derived)
                    - timeshift * MfRef - pWFHM["phaseshift"]
                    + 2.0 * phic + math.pi / 4.)
        pWFHM["phiref22"] = phiref22
        phi22ref = (1. / eta * self.phase(frefRD, phase22_coeffs, derived)
                    + timeshift * frefRD + pWFHM["phaseshift"] + phiref22)
        phishift = _RD_PHASE_32_PHASESHIFT(pfit)
        sph["phi0_S"] = phi22ref \
            - self._rd_phase_ansatz_int_spheroidal(frefRD, pWFHM, sph) + phishift
        return sph

    @staticmethod
    def _rd_phase_ansatz_spheroidal(Mf, pWFHM, sph):
        """
        Spheroidal (3,2) dphi/df, version 122019:
        alpha0_S + alpha2_S/f^2 + alpha4_S/f^4 + alphaL_S Lorentzian.
        Source: IMRPhenomXHM_RD_Phase_Ansatz, ringdown.c:1404-1407.
        """
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]
        invf2 = 1. / (Mf * Mf)
        return (sph["alpha0_S"] + sph["alpha2_S"] * invf2
                + sph["alpha4_S"] * invf2 * invf2
                + sph["alphaL_S"] * fda / (fda * fda + (Mf - frd) * (Mf - frd)))

    @staticmethod
    def _rd_phase_ansatz_int_spheroidal(Mf, pWFHM, sph):
        """
        Spheroidal (3,2) phase, version 122019: phi0_S + alpha0_S f
        - alpha2_S/f - alpha4_S/(3 f^3) + alphaL_S atan((f - fRING)/fDAMP).
        Source: IMRPhenomXHM_RD_Phase_AnsatzInt, ringdown.c:1450-1453.
        """
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]
        invf = 1. / Mf
        return (sph["phi0_S"] + sph["alpha0_S"] * Mf - sph["alpha2_S"] * invf
                - 1. / 3. * sph["alpha4_S"] * invf * invf * invf
                + sph["alphaL_S"] * torch.atan((Mf - frd) / fda))

    @staticmethod
    def _rd22_amp_analytical(Mf, amp22_coeffs):
        """
        Analytic 22 ringdown amplitude (NR-rescaled units) from the XAS
        cached combinations gammaD13 = fDAMP gamma1 gamma3,
        gammaD2 = (fDAMP gamma3)^2, gammaR = gamma2/(fDAMP gamma3).
        Source: XLALSimIMRPhenomXRingdownAmplitude22AnsatzAnalytical
        (LALSimIMRPhenomXUtilities.c).
        """
        dfr = Mf - amp22_coeffs["fRING"]
        return (amp22_coeffs["gammaD13"] / (dfr * dfr + amp22_coeffs["gammaD2"])
                * torch.exp(-dfr * amp22_coeffs["gammaR"]))

    def _spheroidal_to_spherical(self, Mf, derived, pWFHM, amp, sph,
                                 amp22_coeffs, phase22_coeffs):
        """
        h32_spherical = conj(a322) wf22R + conj(a323) A32S e^{i phi32S}.

        Since the 22 is the most dominant mode, it will not show a
        significant mixing with any other mode, so we can assume that
        h22_spheroidal = h22_spherical.  wf22R is the analytic 22
        ringdown (amp x phase incl. timeshift/phiref22), scaled to strain
        for RingdownAmpVersion != 0.
        Source: SpheroidalToSpherical, internals.c:2653-2670.
        """
        eta = derived[:, 3:4]
        amp22 = self._rd22_amp_analytical(Mf, amp22_coeffs)
        phi22 = (1. / eta * self.phase(Mf, phase22_coeffs, derived)
                 + pWFHM["timeshift"] * Mf + pWFHM["phaseshift"] + pWFHM["phiref22"])
        wf22R = amp22 * torch.polar(torch.ones_like(phi22), phi22)
        # RingdownAmpVersion 1: strain units (internals.c:2660-2662);
        # m_seven_sixths by the LAL powers chain (m_one_sixth * m_one)
        m76 = (1. / Mf ** (1.0 / 6.0)) * (1. / Mf)
        wf22R = wf22R * (pWFHM["ampNorm"] * m76)
        amplm = self._rd_amp_ansatz_core(Mf, pWFHM, amp)
        philm = self._rd_phase_ansatz_int_spheroidal(Mf, pWFHM, sph)
        a322 = pWFHM["mixingCoeffs"][2]
        a323 = pWFHM["mixingCoeffs"][3]
        return (torch.conj(a322) * wf22R
                + torch.conj(a323) * amplm * torch.polar(torch.ones_like(philm), philm))

    # ------------------------------------------------------------------
    # Per-frequency evaluation: region ansatze
    # ------------------------------------------------------------------

    def _insp_amp_ansatz(self, powers_Mf, pWFHM, amp):
        """
        Complete inspiral amplitude ansatz: PN ansatz + pseudo-PN terms,
        returned as strain (rescale factor 0).
        Source: IMRPhenomXHM_Inspiral_Amp_Ansatz, inspiral.c:742-773
        (release != 122019 branch).
        """
        pn = self._insp_pn_amp_series(powers_Mf, amp) * amp["PNglobalfactor"] \
            * powers_Mf["m_seven_sixths"] * pWFHM["ampNorm"]
        pseudoterms = (powers_Mf["seven_thirds"] / amp["fcutInsp_seven_thirds"] * amp["InspiralCoefficient0"]
                       + powers_Mf["eight_thirds"] / amp["fcutInsp_eight_thirds"] * amp["InspiralCoefficient1"]
                       + powers_Mf["three"] / amp["fcutInsp_three"] * amp["InspiralCoefficient2"])
        pseudoterms = pseudoterms * powers_Mf["m_seven_sixths"] * amp["PNdominant"]
        return pn + pseudoterms

    def _inter_amp_ansatz(self, powers_Mf, pWFHM, amp):
        """
        Intermediate amplitude ansatz (release 122022):
        f^(-7/6) * polynomial.
        Source: IMRPhenomXHM_Intermediate_Amp_Ansatz,
        intermediate.c:1900-1912.
        """
        ff = powers_Mf["itself"]
        result = torch.zeros_like(ff)
        fpower = torch.ones_like(ff)
        for i in range(amp["nCoefficientsInter"]):
            result = result + amp[f"InterCoefficient{i}"] * fpower
            fpower = fpower * ff
        return result * powers_Mf["m_seven_sixths"]

    def _rd_amp_ansatz_core(self, ff, pWFHM, amp):
        """
        Ringdown amplitude ansatz (versions 1/2): RDAux polynomial below
        fRDAux (32 only), exponential falloff at/above fAmpRDfalloff,
        else the sigma-broadened Lorentzian with exponential decay.
        Source: IMRPhenomXHM_RD_Amp_Ansatz, ringdown.c:920-942.
        """
        out = self._rd_amp_lorentzian(ff, pWFHM, amp)
        falloff = pWFHM["fAmpRDfalloff"]
        out = torch.where(
            ff >= falloff,
            amp["RDCoefficient3"] * torch.exp(-amp["RDCoefficient4"] * (ff - falloff)),
            out,
        )
        if amp["nCoefficientsRDAux"] > 0:
            poly = torch.zeros_like(ff)
            fpower = torch.ones_like(ff)
            for i in range(amp["nCoefficientsRDAux"]):
                poly = poly + fpower * amp[f"RDAuxCoefficient{i}"]
                fpower = fpower * ff
            out = torch.where(ff < amp["fRDAux"], poly, out)
        return out

    def _rd_amp_ansatz(self, powers_Mf, pWFHM, amp):
        """See _rd_amp_ansatz_core; takes the cached-powers dict."""
        return self._rd_amp_ansatz_core(powers_Mf["itself"], pWFHM, amp)

    def _spheroidal_boundary_value_derivative(self, Mf, derived, pWFHM, amp,
                                              spheroidal, amp22_coeffs,
                                              phase22_coeffs):
        """
        |SpheroidalToSpherical| and its 4th-order numerical derivative
        (df = 10e-10) at the (3,2) intermediate right boundary.
        Source: intermediate.c:2512/2523-2524 and
        IMRPhenomXHM_RD_Amp_NDAnsatz, ringdown.c:1002-1029.
        """
        def cabs_s2s(f):
            return torch.abs(self._spheroidal_to_spherical(
                f, derived, pWFHM, amp, spheroidal, amp22_coeffs, phase22_coeffs))

        val = cabs_s2s(Mf)
        df = 10e-10
        der = (-cabs_s2s(Mf + 2 * df) + 8 * cabs_s2s(Mf + df)
               - 8 * cabs_s2s(Mf - df) + cabs_s2s(Mf - 2 * df)) / (12 * df)
        return val, der

    @staticmethod
    def _insp_phase_ansatz_int(Mf, pPhase):
        """
        Inspiral phase from the rescaled phenX coefficients:
        sum_i (phi[i] + phiL[i] log Mf) * Mf^((i-5)/3), i = 0..12.
        Source: IMRPhenomXHM_Inspiral_Phase_AnsatzInt, inspiral.c:1091-1101.
        """
        logMf = torch.log(Mf)
        onethird = Mf ** (1.0 / 6.0)
        onethird = onethird * onethird
        philm = 0.
        fpow = 1. / (onethird ** 5)  # Mf^(-5/3)
        for i in range(13):
            philm = philm + (pPhase["phi"][i] + pPhase["phiL"][i] * logMf) * fpow
            fpow = fpow * onethird
        return philm

    @staticmethod
    def _insp_phase_ansatz(Mf, pPhase):
        """
        dphi/df of the inspiral ansatz.
        Source: IMRPhenomXHM_Inspiral_Phase_Ansatz, inspiral.c:1103-1114.
        """
        logMf = torch.log(Mf)
        onethird = Mf ** (1.0 / 6.0)
        onethird = onethird * onethird
        dphilm = 0.
        fpow = 1. / (onethird ** 8)  # Mf^(-8/3)
        for i in range(13):
            ci = (i - 5) / 3.
            dphilm = dphilm + ((pPhase["phi"][i] + pPhase["phiL"][i] * logMf) * ci
                               + pPhase["phiL"][i]) * fpow
            fpow = fpow * onethird
        return dphilm

    @staticmethod
    def _inter_phase_ansatz_int(Mf, pWFHM, pPhase):
        """
        Intermediate phase: c0 f + c1 log f - c2/f - c4/(3 f^3)
        [- c3/(2 f^2) for the 32] + cL atan((f - fRING)/fDAMP).
        Source: IMRPhenomXHM_Inter_Phase_AnsatzInt,
        intermediate.c:3258-3283.
        """
        invf = 1. / Mf
        invf3 = invf * invf * invf
        out = (pPhase["c0"] * Mf + pPhase["c1"] * torch.log(Mf)
               - pPhase["c2"] * invf - 1. / 3. * pPhase["c4"] * invf3
               + pPhase["cL"] * torch.atan((Mf - pWFHM["fRING"]) / pWFHM["fDAMP"]))
        if pWFHM["modeTag"] == 32:
            out = out - 0.5 * pPhase["c3"] * invf * invf
        return out

    @staticmethod
    def _inter_phase_ansatz(Mf, pWFHM, pPhase):
        """
        dphi/df of the intermediate ansatz:
        c0 + c1/f + c2/f^2 + c4/f^4 [+ c3/f^3] + cL Lorentzian.
        Source: IMRPhenomXHM_Inter_Phase_Ansatz, intermediate.c:3285-3312.
        """
        invf = 1. / Mf
        invf2 = invf * invf
        fda = pWFHM["fDAMP"]
        frd = pWFHM["fRING"]
        out = (pPhase["c0"] + pPhase["c1"] * invf + pPhase["c2"] * invf2
               + pPhase["c4"] * invf2 * invf2
               + pPhase["cL"] * fda / (fda * fda + (Mf - frd) * (Mf - frd)))
        if pWFHM["modeTag"] == 32:
            out = out + pPhase["c3"] * invf2 * invf
        return out

    @staticmethod
    def _rd_phase_ansatz(Mf, pWFHM, pPhase):
        """
        dphi/df of the ringdown ansatz, non-mixing rescaling of the 22:
        alpha0 + fRING^2 alpha2 / f^2 + alphaL Lorentzian.
        Source: IMRPhenomXHM_RD_Phase_Ansatz, ringdown.c:1383-1400.
        """
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]
        return (pPhase["alpha0"] + frd * frd * pPhase["alpha2"] / (Mf * Mf)
                + pPhase["alphaL"] * fda / (fda * fda + (Mf - frd) * (Mf - frd)))

    @staticmethod
    def _rd_phase_ansatz_int(Mf, pWFHM, pPhase):
        """
        Ringdown phase, non-mixing: alpha0 f - fRING^2 alpha2 / f
        + alphaL atan((f - fRING)/fDAMP).
        Source: IMRPhenomXHM_RD_Phase_AnsatzInt, ringdown.c:1427-1445.
        """
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]
        return (pPhase["alpha0"] * Mf - frd * frd * pPhase["alpha2"] / Mf
                + pPhase["alphaL"] * torch.atan((Mf - frd) / fda))

    # ------------------------------------------------------------------
    # Per-frequency evaluation: piecewise assembly
    # Source: IMRPhenomXHM_Amplitude_* / IMRPhenomXHM_Phase_*,
    #         internals.c:2708-2871
    # ------------------------------------------------------------------

    def amp_lm(self, Mf, derived, pWFHM, amp_coeffs, phase_coeffs=None,
               amp22_coeffs=None, phase22_coeffs=None, spheroidal=None):
        """
        Piecewise amplitude of mode (l, m) over the full grid (strain, in
        NR units): inspiral for Mf < fAmpMatchIN, ringdown for
        Mf >= fAmpMatchIM, intermediate polynomial between; negative
        amplitudes are clamped to FALSE_ZERO (internals.c:2741); Ampzero
        modes return zero.  The (3,2) ringdown region
        (|SpheroidalToSpherical|) is handled by the mode-mixing evaluator
        once the spheroidal context exists.
        Source: IMRPhenomXHM_Amplitude_noModeMixing /
        IMRPhenomXHM_Amplitude_ModeMixing, internals.c:2708-2806.
        """
        p = self._initialize_powers(Mf)
        insp = self._insp_amp_ansatz(p, pWFHM, amp_coeffs)
        inter = self._inter_amp_ansatz(p, pWFHM, amp_coeffs)
        if pWFHM["MixingOn"] == 1:
            rd = torch.abs(self._spheroidal_to_spherical(
                Mf, derived, pWFHM, amp_coeffs, spheroidal,
                amp22_coeffs, phase22_coeffs))
        else:
            rd = self._rd_amp_ansatz(p, pWFHM, amp_coeffs)
        out = torch.where(Mf < amp_coeffs["fAmpMatchIN"], insp,
                          torch.where(Mf >= amp_coeffs["fAmpMatchIM"], rd, inter))
        out = torch.where(out < 0, torch.full_like(out, _FALSE_ZERO), out)
        return torch.where(pWFHM["Ampzero"], torch.zeros_like(out), out)

    def phase_lm(self, Mf, derived, pWFHM, phase_coeffs, amp_coeffs=None,
                 amp22_coeffs=None, phase22_coeffs=None, spheroidal=None):
        """
        Piecewise phase of mode (l, m) over the full grid: inspiral for
        Mf < fPhaseMatchIN, ringdown for Mf >= fPhaseMatchIM,
        intermediate between; region connection constants and deltaphiLM
        included.  The (3,2) ringdown region is
        carg(SpheroidalToSpherical) + C1RD Mf + CRD + deltaphiLM.
        Source: IMRPhenomXHM_Phase_noModeMixing /
        IMRPhenomXHM_Phase_ModeMixing, internals.c:2809-2871.
        """
        pPhase = phase_coeffs
        insp = (self._insp_phase_ansatz_int(Mf, pPhase)
                + pPhase["C1INSP"] * Mf + pPhase["CINSP"] + pPhase["deltaphiLM"])
        inter = (self._inter_phase_ansatz_int(Mf, pWFHM, pPhase)
                 + pPhase["deltaphiLM"])
        if pWFHM["MixingOn"] == 1:
            rd = (torch.angle(self._spheroidal_to_spherical(
                      Mf, derived, pWFHM, amp_coeffs, spheroidal,
                      amp22_coeffs, phase22_coeffs))
                  + pPhase["C1RD"] * Mf + pPhase["CRD"] + pPhase["deltaphiLM"])
        else:
            rd = (self._rd_phase_ansatz_int(Mf, pWFHM, pPhase)
                  + pPhase["C1RD"] * Mf + pPhase["CRD"] + pPhase["deltaphiLM"])
        return torch.where(Mf < pPhase["fPhaseMatchIN"], insp,
                           torch.where(Mf >= pPhase["fPhaseMatchIM"], rd, inter))

    # ------------------------------------------------------------------
    # Single-mode generator
    # Source: IMRPhenomXHMGenerateFDOneMode, LALSimIMRPhenomXHM.c:511
    # ------------------------------------------------------------------

    def gethlm(self, theta, ell, emm, derived=None, amp22_coeffs=None,
               phase22_coeffs=None, afinal=None):
        """
        Aligned-spin mode h_{l,-m}(f) on the (B, F) grid (no DC padding;
        the caller pads/assembles).

        (2,2) delegates to the XAS amplitude/phase machinery
        (h22 = Amp0 A Mf^(-7/6) e^{i(Psi/eta + linb Mf + phifRef)}, the
        series XLALSimIMRPhenomXASGenerateFD fills); higher modes run the
        per-mode builders and evaluate amp_lm/phase_lm, with the (3,2)
        spheroidal chain built in order.  The odd-l sign is folded into
        the amplitude, Amp0 -> -Amp0 (LALSimIMRPhenomXHM.c:601-604), so
        the array is directly h_{l,-m}.  Modes with Ampzero (odd m,
        equal mass, equal spin) return zeros.
        The optional ``afinal`` override (precessing final spin) is
        threaded into every coefficient builder; the final mass stays
        aligned-spin, mirroring the pWF overwrite performed by
        IMRPhenomX_SetPrecessingRemnantParams.
        Source: IMRPhenomXHMGenerateFDOneMode, LALSimIMRPhenomXHM.c:511-750.
        """
        if derived is None:
            derived = self.compute_derived_parameters(theta)
        if phase22_coeffs is None:
            phase22_coeffs = self.get_phase_coeffs(derived, afinal=afinal)
        # Model cutoff: every mode is zero above Mf = fCutDef
        # (f_max_prime truncation; internals.c:503-513)
        fcutdef = torch.where(derived[:, 7:8] > 0.99,
                              torch.full_like(derived[:, 7:8], 0.33),
                              torch.full_like(derived[:, 7:8], 0.3))
        eta = derived[:, 3:4]
        delta = derived[:, 4:5]
        STotR = derived[:, 9:10]
        dchi = derived[:, 10:11]
        M_s = derived[:, 2:3]
        phic = theta[:, 6:7]
        Mf = self.f * M_s

        if (ell, emm) == (2, 2):
            if amp22_coeffs is None:
                amp22_coeffs = self.get_amp_coeffs(derived, afinal=afinal)
            # Amp0 = amp0 * ampNorm; linb and phifRef as in the XAS
            # assembly (IMRPhenomXAS.get_hphc steps 3-6)
            M_m = M_s * self.C
            dist_m = theta[:, 4:5] * self.Mpc
            ampNorm = torch.sqrt(2.0 * eta / 3.0) * (math.pi ** (-1.0 / 6.0))
            Amp0 = (M_m * M_s / dist_m) * ampNorm
            linb_fit = self._linb_fit(eta, STotR, dchi, delta)
            psi4 = self._psi4tostrain_fit(eta, STotR, dchi)
            frefFit = phase22_coeffs["fRING"] - phase22_coeffs["fDAMP"]
            dphi22Ref = (1. / eta) * self.dphase(frefFit, phase22_coeffs, derived)
            linb = linb_fit - dphi22Ref - 2. * math.pi * (500. + psi4)
            MfRef = self.f_ref * M_s
            phifRef = (-(self.phase(MfRef, phase22_coeffs, derived) / eta
                         + linb * MfRef) + 2.0 * phic + 0.25 * math.pi)
            phi_total = (self.phase(Mf, phase22_coeffs, derived) / eta
                         + linb * Mf + phifRef)
            A_total = Amp0 * Mf.pow(-7.0 / 6.0) * self.amp(Mf, amp22_coeffs, derived)
            h = A_total * torch.polar(torch.ones_like(phi_total), phi_total)
            return torch.where(Mf <= fcutdef, h, torch.zeros_like(h))

        pWFHM = self.set_hm_waveform_variables(ell, emm, theta, derived,
                                               afinal=afinal)
        if pWFHM["MixingOn"] == 1:
            if amp22_coeffs is None:
                amp22_coeffs = self.get_amp_coeffs(derived, afinal=afinal)
            sph = self.get_spheroidal_coeffs(pWFHM, derived, theta, phase22_coeffs)
        else:
            sph = None
        amp = self.get_hm_amp_coeffs(pWFHM, derived, amp22_coeffs,
                                     phase22_coeffs, spheroidal=sph)
        phase = self.get_hm_phase_coeffs(pWFHM, derived, theta, phase22_coeffs,
                                         amp_coeffs=amp, amp22_coeffs=amp22_coeffs,
                                         spheroidal=sph)
        A = self.amp_lm(Mf, derived, pWFHM, amp, amp22_coeffs=amp22_coeffs,
                        phase22_coeffs=phase22_coeffs, spheroidal=sph)
        P = self.phase_lm(Mf, derived, pWFHM, phase, amp_coeffs=amp,
                          amp22_coeffs=amp22_coeffs, phase22_coeffs=phase22_coeffs,
                          spheroidal=sph)
        # if(ell%2 != 0) Amp0 = -Amp0 (LALSimIMRPhenomXHM.c:601-604)
        Amp0 = -pWFHM["Amp0"] if ell % 2 != 0 else pWFHM["Amp0"]
        h = Amp0 * A * torch.polar(torch.ones_like(P), P)
        return torch.where(Mf <= fcutdef, h, torch.zeros_like(h))

    # ------------------------------------------------------------------
    # Mode sum
    # Source: IMRPhenomXHMFDAddMode, LALSimIMRPhenomXHM.c:2282
    # ------------------------------------------------------------------

    def _fd_add_mode(self, hp, hc, hlm, theta_incl, phi_azim, ell, emm):
        """
        Accumulate mode (l, m) into hp/hc; equatorial symmetry adds the
        -m and m mode at once:

        Ym      = Y_{l,-m}(theta, phi)
        factorp = 0.5 (Ym + (-1)^l conj(Y_{lm}))
        factorc = i 0.5 (Ym - (-1)^l conj(Y_{lm}))
        hp += factorp * hlm ; hc += factorc * hlm

        Source: IMRPhenomXHMFDAddMode, LALSimIMRPhenomXHM.c:2282-2328.
        """
        minus1l = -1 if ell % 2 != 0 else 1
        Ym = self.spin_weighted_Y(theta_incl, phi_azim, -2, ell, -emm)
        Ystar = torch.conj(self.spin_weighted_Y(theta_incl, phi_azim, -2, ell, emm))
        factorp = 0.5 * (Ym + minus1l * Ystar)
        factorc = 1j * 0.5 * (Ym - minus1l * Ystar)
        return hp + factorp * hlm, hc + factorc * hlm

    # ------------------------------------------------------------------
    # Public interface
    # Source: IMRPhenomXHM_MultiMode2, LALSimIMRPhenomXHM.c:1931
    # ------------------------------------------------------------------

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute FD plus and cross polarisations for a BBH parameter batch,
        summing (2,±2), (2,±1), (3,±3), (3,±2), (4,±4).

        Follows IMRPhenomXHM_MultiMode2 (no multibanding): the 22-mode
        coefficient structs are built once and recycled by every higher
        mode; modes are added with azimuthal angle pi/2
        (LALSimIMRPhenomXHM.c:1900-1908).

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 17)
            Parameters in the waveform_utils.WAVEFORM_PARAMETERS layout.
        reproduce_lal : bool
            If True, skip FD tapering, tc shift and df normalisation so the
            output can be compared directly with raw LALSim output (with
            PhenomXHMThresholdMband = 0) on the same frequency grid.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_pad + F), complex

        Raises
        ------
        ValueError
            If any transverse spin is non-zero.
        """
        theta = waveform_utils.aligned_layout(theta, waveform_utils.ALIGNED_COLUMNS)
        iota = theta[:, 7:8]
        tc = theta[:, 5:6]
        derived = self.compute_derived_parameters(theta)
        M_s = derived[:, 2:3]

        # 22 coefficient structs built once and recycled by every mode
        amp22_coeffs = self.get_amp_coeffs(derived)
        phase22_coeffs = self.get_phase_coeffs(derived)

        hp = torch.zeros((self.B, self.f_numel), dtype=torch.complex128,
                         device=self.f.device)
        hc = torch.zeros_like(hp)

        # Loop over only positive m is intentional: the single-mode
        # generator returns h_{l,-m} and the positive mode is added
        # automatically in FDAddMode; modes summed with azimuthal angle
        # pi/2 (LALSimIMRPhenomXHM.c:2060, 2116, 1900-1908).
        h22 = self.gethlm(theta, 2, 2, derived, amp22_coeffs, phase22_coeffs)
        hp, hc = self._fd_add_mode(hp, hc, h22, iota, math.pi / 2., 2, 2)
        for (ell, emm) in self.HM_MODES:
            hlm = self.gethlm(theta, ell, emm, derived, amp22_coeffs,
                              phase22_coeffs)
            hp, hc = self._fd_add_mode(hp, hc, hlm, iota, math.pi / 2., ell, emm)

        if not reproduce_lal:
            fcut = self.fM_CUT_XAS / M_s
            _win = _taper_mod.fd_taper(
                f=self.f,
                f_min=self.f[0, 0].item(),
                f_cut=fcut,
                df=self.df,
            )
            hp = hp * _win
            hc = hc * _win
            hp, hc = self.apply_tc(hp, hc, tc)
            hp = hp * self.df
            hc = hc * self.df

        return self.pad_missing_frequencies(hp, hc)
