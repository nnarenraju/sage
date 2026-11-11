#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : IMRPhenomXHM.py
Description   : Batched PyTorch port of LALSimulation's IMRPhenomXHM
                multimode waveform (LALSimIMRPhenomXHM.c and its
                _internals, _inspiral, _intermediate, _ringdown, _qnm).

Created on 2026-08-04

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
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

# Source: IMRPhenomXHM_FillAmpFitsArray
# (LALSimIMRPhenomXHM_internals.c).
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

# Source: IMRPhenomXHM_FillPhaseFitsArray
# (LALSimIMRPhenomXHM_internals.c).
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

# Source: FALSE_ZERO (LALSimIMRPhenomXHM_internals.h).
_FALSE_ZERO = 1.0e-15


def _cdiv(z, x):
    """
    C division of a complex value by a real one, part by part.

    Parameters
    ----------
    z : torch.Tensor, complex
        Numerator.
    x : float or torch.Tensor
        Real denominator.

    Returns
    -------
    torch.Tensor, complex
    """
    return torch.complex(z.real / x, z.imag / x)


class IMRPhenomXHM(IMRPhenomXAS):
    """
    Batched IMRPhenomXHM aligned-spin multimode waveform.

    Adds the (2,1), (3,3), (3,2) and (4,4) modes to IMRPhenomXAS. The
    single-mode generator returns h_{l,-m}; h_{l,m} follows from
    equatorial symmetry.

    Parameters
    ----------
    f : torch.Tensor, shape (B, F)
        Frequency grid in Hz.
    f_ref : torch.Tensor, shape (B, 1)
        Reference frequency in Hz.
    **kwargs
        Forwarded to IMRPhenomXAS.
    """

    # Higher modes, in the order of the loop over l = 2..4, m = 1..l.
    HM_MODES = ((2, 1), (3, 2), (3, 3), (4, 4))

    # Version flags of release 122022; keys are modeTag = 10*ell + emm.
    RELEASE_VERSION = 122022
    INSPIRAL_AMP_VERSION = {21: 123, 33: 123, 32: 123, 44: 123}
    INTERMEDIATE_AMP_VERSION = {21: 110102, 33: 211112, 32: 211112, 44: 211112}
    RINGDOWN_AMP_VERSION = {21: 2, 33: 2, 32: 1, 44: 2}
    # Limit between comparable and extreme mass ratios for the phase
    ETA_EMR = 0.05
    # Step of the finite-difference phase derivatives of the (3,2)
    _S2S_FSTEP = 0.0000001

    def __init__(self, f, f_ref, **kwargs):
        super().__init__(f, f_ref, **kwargs)

    # ------------------------------------------------------------------
    # Spin-weighted spherical harmonics
    # ------------------------------------------------------------------

    # Source: XLALSpinWeightedSphericalHarmonic (SphericalHarmonics.c).
    @staticmethod
    def spin_weighted_Y(theta, phi, s, ell, emm):
        """
        Spin-weight -2 spherical harmonic Y_lm(theta, phi), l = 2..4.

        Parameters
        ----------
        theta : torch.Tensor
            Polar angle in rad.
        phi : float or torch.Tensor
            Azimuthal angle in rad.
        s : int
            Spin weight; only -2 is supported.
        ell, emm : int
            Mode numbers l and m.

        Returns
        -------
        torch.Tensor, complex, broadcast shape of ``theta``

        Raises
        ------
        ValueError
            If s != -2, l is not in [2, 4] or |m| > l.
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
        # ans = cpolar(1.0, m*phi) * fac
        angle = emm * phi
        if isinstance(angle, torch.Tensor):
            return fac * torch.polar(torch.ones_like(fac), angle)
        return fac * complex(math.cos(angle), math.sin(angle))

    # ------------------------------------------------------------------
    # QNM fits
    # ------------------------------------------------------------------

    # Source: evaluate_QNMfit_fring21 (LALSimIMRPhenomXHM_qnm.c).
    @staticmethod
    def _qnm_fring_21(af):
        """
        Fit of the (2,1) ringdown frequency.

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Dimensionless final spin.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        return (0.059471695665734674 - 0.07585416297991414*af + 0.021967909664591865*x2 - 0.0018964744613388146*x3 + 0.001164879406179587*x4 - 0.0003387374454044957*x5) / (1 - 1.4437415542456158*af + 0.49246920313191234*x2)

    # Source: evaluate_QNMfit_fdamp21 (LALSimIMRPhenomXHM_qnm.c).
    @staticmethod
    def _qnm_fdamp_21(af):
        """
        Fit of the (2,1) damping frequency.

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Dimensionless final spin.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        return (2.0696914454467294 - 3.1358071947583093*af + 0.14456081596393977*x2 + 1.2194717985037946*x3 - 0.2947372598589144*x4 + 0.002943057145913646*x5) / (146.1779212636481 - 219.81790388304876*af + 17.7141194900164*x2 + 75.90115083917898*x3 - 18.975287709794745*x4)

    # Source: evaluate_QNMfit_fring33 (LALSimIMRPhenomXHM_qnm.c).
    @staticmethod
    def _qnm_fring_33(af):
        """
        Fit of the (3,3) ringdown frequency.

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Dimensionless final spin.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.09540436245212061 - 0.22799517865876945*af + 0.13402916709362475*x2 + 0.03343753057911253*x3 - 0.030848060170259615*x4 - 0.006756504382964637*x5 + 0.0027301732074159835*x6) / (1 - 2.7265947806178334*af + 2.144070539525238*x2 - 0.4706873667569393*x4 + 0.05321818246993958*x6)

    # Source: evaluate_QNMfit_fdamp33 (LALSimIMRPhenomXHM_qnm.c).
    @staticmethod
    def _qnm_fdamp_33(af):
        """
        Fit of the (3,3) damping frequency.

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Dimensionless final spin.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        return (0.014754148319335946 - 0.03124423610028678*af + 0.017192623913708124*x2 + 0.001034954865629645*x3 - 0.0015925124814622795*x4 - 0.0001414350555699256*x5) / (1 - 2.0963684630756894*af + 1.196809702382645*x2 - 0.09874113387889819*x4)

    # Source: evaluate_QNMfit_fring32 (LALSimIMRPhenomXHM_qnm.c).
    @staticmethod
    def _qnm_fring_32(af):
        """
        Fit of the (3,2) ringdown frequency.

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Dimensionless final spin.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.09540436245212061 - 0.13628306966373951*af + 0.030099881830507727*x2 - 0.000673589757007597*x3 + 0.0118277880067919*x4 + 0.0020533816327907334*x5 - 0.0015206141948469621*x6) / (1 - 1.6531854335715193*af + 0.5634705514193629*x2 + 0.12256204148002939*x4 - 0.027297817699401976*x6)

    # Source: evaluate_QNMfit_fdamp32 (LALSimIMRPhenomXHM_qnm.c).
    @staticmethod
    def _qnm_fdamp_32(af):
        """
        Fit of the (3,2) damping frequency.

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Dimensionless final spin.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        return (0.014754148319335946 - 0.03445752346074498*af + 0.02168855041940869*x2 + 0.0014945908223317514*x3 - 0.0034761714223258693*x4) / (1 - 2.320722660848874*af + 1.5096146036915865*x2 - 0.18791187563554512*x4)

    # Source: evaluate_QNMfit_fring44 (LALSimIMRPhenomXHM_qnm.c).
    @staticmethod
    def _qnm_fring_44(af):
        """
        Fit of the (4,4) ringdown frequency.

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Dimensionless final spin.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.1287821193485683 - 0.21224284094693793*af + 0.0710926778043916*x2 + 0.015487322972031054*x3 - 0.002795401084713644*x4 + 0.000045483523029172406*x5 + 0.00034775290179000503*x6) / (1 - 1.9931645124693607*af + 1.0593147376898773*x2 - 0.06378640753152783*x4)

    # Source: evaluate_QNMfit_fdamp44 (LALSimIMRPhenomXHM_qnm.c).
    @staticmethod
    def _qnm_fdamp_44(af):
        """
        Fit of the (4,4) damping frequency.

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Dimensionless final spin.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
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

    # Source: IMRPhenomXHM_SetHMWaveformVariables
    # (LALSimIMRPhenomXHM_internals.c).
    @classmethod
    def get_fring_fdamp_lm(cls, af, Mfinal, ell, emm):
        """
        Ringdown and damping frequencies of mode (l, m), in Mf.

        Parameters
        ----------
        af, Mfinal : torch.Tensor, shape (B, 1)
            Final spin and final mass.
        ell, emm : int
            Mode numbers.

        Returns
        -------
        fring, fdamp : torch.Tensor, shape (B, 1)
        """
        fring_name, fdamp_name = cls._QNM_FITS[10 * ell + emm]
        fring = getattr(cls, fring_name)(af) / Mfinal
        fdamp = getattr(cls, fdamp_name)(af) / Mfinal
        return fring, fdamp

    # Source: IMRPhenomXHM_Initialize_MixingCoeffs
    # (LALSimIMRPhenomXHM_internals.c), evaluate_QNMfit_re_l2m2lp2 to
    # evaluate_QNMfit_im_l3m2lp3 (LALSimIMRPhenomXHM_qnm.c).
    @staticmethod
    def _qnm_mixing_fits(af):
        """
        Complex mixing coefficients of the (3,2) mode.

        We assume mixing with just one mode, the (2,2).

        Parameters
        ----------
        af : torch.Tensor, shape (B, 1)
            Dimensionless final spin.

        Returns
        -------
        a222, a223, a322, a323 : torch.Tensor, complex, shape (B, 1)
        """
        x2 = af * af
        x3 = x2 * af
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        re_l2m2lp2 = (1 - 2.2956993576253635*af + 1.461988775298876*x2 + 0.0043296365593147035*x3 - 0.1695667458204109*x4 - 0.0006267849034466508*x5) / (1 - 2.2956977727459043*af + 1.4646339137818438*x2 - 0.16843226886562457*x4 - 0.00007150540890128118*x6)
        im_l2m2lp2 = (af*(0.3826673013161342 - 0.47531267226013896*af - 0.05898102880105067*x2 + 0.0724525431346487*x3 + 0.054714637311702986*x4 + 0.024544862718252784*x5)) / (-38.70835035062785 + 69.82140084545878*af - 27.99036444363243*x2 - 4.152310472191899*x4 + 1.*x6)
        re_l3m2lp2 = (af*(0.47513455283841244 - 0.9016636384605536*af + 0.3844811236426182*x2 + 0.0855565148647794*x3 - 0.03620067426672167*x4 - 0.006557249133752502*x5)) / (-6.76894063440646 + 15.170831931186493*af - 9.406169787571082*x2 + 1.*x4)
        im_l3m2lp2 = (af*(-2.8704762147145533 + 4.436434016918535*af - 1.0115343326360486*x2 - 0.08965314412106505*x3 - 0.4236810894599512*x4 - 0.041787576033810676*x5)) / (-171.80908957903395 + 272.362882450877*af - 76.68544453077854*x2 - 25.14197656531123*x4 + 1.*x6)
        re_l2m2lp3 = (af*(18.522563276099167 - 37.978140351289014*af + 19.030390708998894*x2 + 3.0355668591803386*x3 - 2.210028290847915*x4 - 0.37117112862247975*x5)) / (164.52480238697507 - 377.9093045285145*af + 243.3353695550844*x2 - 30.79738566181734*x4 + 1.*x6)
        im_l2m2lp3 = (af*(-49.7688437256778 + 120.43773704442333*af - 82.95323455645332*x2 + 1.721453011852496*x3 + 11.540237244397877*x4 - 0.9819458637589314*x5)) / (2858.5790831181725 - 6305.619505422591*af + 3825.6742092829054*x2 - 377.7822297815406*x4 + 1.*x6)
        re_l3m2lp3 = (1 - 2.107852425643677*af + 1.1906393634562715*x2 + 0.02244848864087732*x3 - 0.09593447799423722*x4 - 0.0021343381708933025*x5 - 0.005319515989331159*x6) / (1 - 2.1078515887706324*af + 1.2043484690080966*x2 - 0.08910191596778137*x4 - 0.005471749827809503*x6)
        im_l3m2lp3 = (af*(12.45701482868677 - 29.398484595717147*af + 18.26221675782779*x2 + 1.9308599142669403*x3 - 3.159763242921214*x4 - 0.0910871567367674*x5)) / (345.52914639836257 - 815.4349339779621*af + 538.3888932415709*x2 - 69.3840921447381*x4 + 1.*x6)
        # mixingCoeffs are COMPLEX8 in LAL
        def complex8(re, im):
            return torch.complex(re.float().to(af.dtype), im.float().to(af.dtype))

        a222 = complex8(re_l2m2lp2, im_l2m2lp2)
        a223 = complex8(re_l2m2lp3, im_l2m2lp3)
        # Adjust conventions so that they match the ones used for the
        # hybrids
        a322 = -complex8(re_l3m2lp2, im_l3m2lp2)
        a323 = -complex8(re_l3m2lp3, im_l3m2lp3)
        return a222, a223, a322, a323

    # ------------------------------------------------------------------
    # Mode waveform variables
    # ------------------------------------------------------------------

    # Source: IMRPhenomXHM_SetHMWaveformVariables
    # (LALSimIMRPhenomXHM_internals.c).
    def set_hm_waveform_variables(self, ell, emm, theta, derived, afinal=None,
                                  Mfinal=None):
        """
        Parameters specific to the higher mode (l, m).

        Keys are the IMRPhenomXHMWaveformStruct member names, for
        release 122022 with the PNR options off.

        Parameters
        ----------
        ell, emm : int
            Mode numbers.
        theta : torch.Tensor
            Aligned layout, with m1 >= m2.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        afinal, Mfinal : torch.Tensor, shape (B, 1), optional
            Final spin and mass; by default the aligned-spin fits.

        Returns
        -------
        dict
        """
        eta = derived[:, 3:4]
        chiPNHat = derived[:, 8:9]
        STotR = derived[:, 9:10]
        dchi = derived[:, 10:11]
        chi1L = derived[:, 11:12]
        chi2L = derived[:, 12:13]

        modeTag = 10 * ell + emm
        modeInt = {21: 0, 33: 1, 32: 2, 44: 3}[modeTag]

        m1, m2 = self.solar_masses(theta)
        q = m1 / m2

        # Odd mode for equal mass, equal spin is zero
        if emm % 2 == 1:
            Ampzero = (q == 1.) & (chi1L == chi2L)
        else:
            Ampzero = torch.zeros_like(eta, dtype=torch.bool)

        Amp0, ampNorm = self.amp_norms(m1 + m2, eta, theta[:, 4:5])
        fMECOlm = self.fMECO(eta, chiPNHat, dchi) * (emm * 0.5)

        # Release 122022; HM phase is unchanged from release 122019
        pWFHM = {
            "ell": ell,
            "emm": emm,
            "modeTag": modeTag,
            "modeInt": modeInt,
            "MixingOn": 1 if modeTag == 32 else 0,
            "q": q,
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

        # Spin parameterisations used for the higher modes
        pWFHM["chi_s"] = (chi1L + chi2L) * 0.5
        pWFHM["chi_a"] = (chi1L - chi2L) * 0.5

        # Ringdown and damping frequencies
        if afinal is None:
            afinal = self.final_spin_2017(eta, STotR, dchi)
        if Mfinal is None:
            Mfinal = self.final_mass_2017(eta, STotR, dchi)
        pWFHM["fRING"], pWFHM["fDAMP"] = self.get_fring_fdamp_lm(
            afinal, Mfinal, ell, emm
        )

        # If (l,m)=(3,2), load the mixing coeffs to transform the
        # spheroidal-harmonic ringdown ansatz back to spherical-harmonic
        if modeTag == 32:
            pWFHM["mixingCoeffs"] = self._qnm_mixing_fits(afinal)

        # Linear part of the 22 phase. timeshift * ff + phaseshift.
        pWFHM["timeshift"] = 0.0
        pWFHM["phaseshift"] = 0.0
        # current time-alignment of the hybrids
        psi4tostrain = IMRPhenomXHM._psi4tostrain_fit(eta, STotR, dchi)
        pWFHM["DeltaT"] = -2.0 * math.pi * (500.0 + psi4tostrain)

        pWFHM["fPhaseRDflat"] = 0.0
        pWFHM["fAmpRDfalloff"] = 0.0
        return pWFHM

    # ------------------------------------------------------------------
    # Amplitude: region boundaries
    # ------------------------------------------------------------------

    # Source: IMRPhenomXHM_Amplitude_fcutInsp
    # (LALSimIMRPhenomXHM_internals.c).
    def _amp_fcut_insp(self, pWFHM, derived):
        """
        Inspiral cutting frequency for the amplitude, version 122022.

        Return the end frequency of the inspiral region and the
        beginning of the intermediate for the amplitude of one mode.

        Parameters
        ----------
        pWFHM : dict
            Output of set_hm_waveform_variables.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        fMECO = pWFHM["fMECOlm"]
        emm = 1.0 * pWFHM["emm"]
        eta = derived[:, 3:4]
        chi1 = derived[:, 11:12]
        q = pWFHM["q"]
        # fcutEMR is the cutting frequency for extreme mass ratios that
        # is given by a fit to the frequency of a particular geometrical
        # structure of the amplitude
        fcutEMR = 1.25 * emm * (
            (0.011671068725758493 - 0.0000858396080377194 * chi1
             + 0.000316707064291237 * chi1 * chi1)
            * (0.8447212540381764 + 6.2873167352395125 * eta)
        ) / (1.2857082764038923 - 0.9977728883419751 * chi1)
        # transition_eta = 0.0192234 (q=50), sharpness = 0.004
        funcs = 0.5 + 0.5 * torch.tanh((eta - 0.0192234) / 0.004)
        blend = funcs * fMECO + (1.0 - funcs) * fcutEMR
        return torch.where(q < 20.0, fMECO, blend)

    # Source: IMRPhenomXHM_Amplitude_fcutRD
    # (LALSimIMRPhenomXHM_internals.c).
    def _amp_fcut_rd(self, pWFHM, phase22_coeffs):
        """
        Ringdown cutting frequency for the amplitude, version 122022.

        Returns the end of the intermediate region and the beginning of
        the ringdown for the amplitude of one mode.

        Parameters
        ----------
        pWFHM : dict
            Output of set_hm_waveform_variables.
        phase22_coeffs : dict
            Phase coefficients of the 22 mode, used for the (3,2).

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        if pWFHM["MixingOn"] == 1:
            return phase22_coeffs["fRING"] - 0.5 * phase22_coeffs["fDAMP"]
        return pWFHM["fRING"] - pWFHM["fDAMP"]

    # ------------------------------------------------------------------
    # Amplitude coefficients
    # ------------------------------------------------------------------

    @staticmethod
    def _pwf_fit_dict(derived):
        """
        Waveform variables read by the parameter-space fits.

        Parameters
        ----------
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.

        Returns
        -------
        dict of torch.Tensor, shape (B, 1)
            Keyed by IMRPhenomXWaveformStruct member names.
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

    # Source: IMRPhenomXHM_GetPNAmplitudeCoefficients
    # (LALSimIMRPhenomXHM_internals.c).
    def _get_pn_amplitude_coeffs(self, pWFHM, derived):
        """
        Coefficients of the FD Post-Newtonian inspiral amplitude ansatz.

        We multiply the Time-domain Post-Newtonian series up to 3PN by
        the phasing factor given by the SPA and reexpand in powers of f.

        Parameters
        ----------
        pWFHM : dict
            Output of set_hm_waveform_variables.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.

        Returns
        -------
        dict
            PNglobalfactor and pnInitial to pnSixThirds.

        Raises
        ------
        ValueError
            If the mode is not (2,1), (3,3), (3,2) or (4,4).
        """
        chiA = pWFHM["chi_a"]
        chiS = pWFHM["chi_s"]
        eta = derived[:, 3:4]
        delta = derived[:, 4:5]
        PI = math.pi
        pi_p = self._initialize_powers(math.pi)
        log = math.log

        # Global factors of each PN hlm
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
            pn["pnFourThirds"] = _cdiv(3427 * chiA - (1j * 672) * delta + 3427 * chiS * delta - 8404 * chiA * eta - 3860 * chiS * delta * eta - 1344 * delta * PI - (1j * 672) * delta * log(16), 1344.) * pi_p["four_thirds"] * p2["four_thirds"]
            pn["pnFiveThirds"] = (-155965824 * chiA * chiS - 964357 * delta + 432843264 * chiA * chiS * eta - 23670792 * delta * eta + 24385536 * chiA * PI + 24385536 * chiS * delta * PI - 77982912 * delta * chiA * chiA + 81285120 * delta * eta * chiA * chiA - 77982912 * delta * chiS * chiS + 39626496 * delta * eta * chiS * chiS + 21535920 * delta * eta * eta) / 8.128512e6 * pi_p["five_thirds"] * p2["five_thirds"]
            pn["pnSixThirds"] = _cdiv(143063173 * chiA - (1j * 1350720) * delta + 143063173 * chiS * delta - 546199608 * chiA * eta - (1j * 72043776) * delta * eta - 169191096 * chiS * delta * eta - 9898560 * delta * PI + 20176128 * delta * eta * PI - (1j * 5402880) * delta * log(2) - (1j * 17224704) * delta * eta * log(2) + 61725888 * chiS * delta * chiA * chiA - 81285120 * chiS * delta * eta * chiA * chiA + 20575296 * chiA ** 3 - 81285120 * eta * chiA ** 3 + 61725888 * chiA * chiS * chiS - 165618432 * chiA * eta * chiS * chiS + 20575296 * delta * chiS ** 3 - 1016064 * delta * eta * chiS * chiS * chiS + 128873808 * chiA * eta * eta - 3859632 * chiS * delta * eta * eta, 5.419008e6) * pi_p["two"] * p2["two"]
        elif tag == 33:
            p2d3 = self._initialize_powers(2. / 3.)
            pn["pnOneThird"] = delta * pi_p["one_third"] * p2d3["one_third"]
            pn["pnTwoThirds"] = torch.zeros_like(eta)
            pn["pnThreeThirds"] = (-1945 * delta + 2268 * delta * eta) / 672. * pi_p["itself"] * p2d3["itself"]
            pn["pnFourThirds"] = _cdiv(325 * chiA - (1j * 504) * delta + 325 * chiS * delta - 1120 * chiA * eta - 80 * chiS * delta * eta + 120 * delta * PI + (1j * 720) * delta * log(1.5), 120.) * pi_p["four_thirds"] * p2d3["four_thirds"]
            pn["pnFiveThirds"] = (-2263282560 * chiA * chiS - 1077664867 * delta + 9053130240 * chiA * chiS * eta - 5926068792 * delta * eta - 1131641280 * delta * chiA * chiA + 4470681600 * delta * eta * chiA * chiA - 1131641280 * delta * chiS * chiS + 55883520 * delta * eta * chiS * chiS + 2966264784 * delta * eta * eta) / 4.4706816e8 * pi_p["five_thirds"] * p2d3["five_thirds"]
            pn["pnSixThirds"] = _cdiv(22007835 * chiA + (1j * 26467560) * delta + 22007835 * chiS * delta - 80190540 * chiA * eta - (1j * 98774368) * delta * eta - 31722300 * chiS * delta * eta - 9193500 * delta * PI + 17826480 * delta * eta * PI - (1j * 37810800) * delta * log(1.5) + (1j * 37558080) * delta * eta * log(1.5) - 12428640 * chiA * eta * eta - 6078240 * chiS * delta * eta * eta, 2.17728e6) * pi_p["two"] * p2d3["two"]
        elif tag == 32:
            pn["pnOneThird"] = torch.zeros_like(eta)
            pn["pnTwoThirds"] = (-1 + 3 * eta) * pi_p["two_thirds"]
            pn["pnThreeThirds"] = -4 * chiS * eta * pi_p["itself"]
            pn["pnFourThirds"] = (10471 - 61625 * eta + 82460 * eta * eta) / 10080. * pi_p["four_thirds"]
            pn["pnFiveThirds"] = _cdiv((1j * 2520) - 3955 * chiS - 3955 * chiA * delta - (1j * 11088) * eta + 10810 * chiS * eta + 11865 * chiA * delta * eta - 12600 * chiS * eta * eta, 840.) * pi_p["five_thirds"]
            pn["pnSixThirds"] = (824173699 + 2263282560 * chiA * chiS * delta - 26069649 * eta - 15209631360 * chiA * chiS * delta * eta + 3576545280 * chiS * eta * PI + 1131641280 * chiA * chiA - 7865605440 * eta * chiA * chiA + 1131641280 * chiS * chiS - 11870591040 * eta * chiS * chiS - 13202119896 * eta * eta + 13412044800 * chiA * chiA * eta * eta + 5830513920 * chiS * chiS * eta * eta + 5907445488 * eta ** 3) / 4.4706816e8 * pi_p["two"]
        elif tag == 44:
            p2d4 = self._initialize_powers(0.5)
            pn["pnOneThird"] = torch.zeros_like(eta)
            pn["pnTwoThirds"] = (1 - 3 * eta) * pi_p["two_thirds"] * p2d4["two_thirds"]
            pn["pnThreeThirds"] = torch.zeros_like(eta)
            pn["pnFourThirds"] = (-158383 + 641105 * eta - 446460 * eta * eta) / 36960. * pi_p["four_thirds"] * p2d4["four_thirds"]
            pn["pnFiveThirds"] = _cdiv((1j * -1008) + 565 * chiS + 565 * chiA * delta + (1j * 3579) * eta - 2075 * chiS * eta - 1695 * chiA * delta * eta + 240 * PI - 720 * eta * PI + (1j * 960) * log(2) - (1j * 2880) * eta * log(2) + 1140 * chiS * eta * eta, 120.) * pi_p["five_thirds"] * p2d4["five_thirds"]
            pn["pnSixThirds"] = (7888301437 - 147113366400 * chiA * chiS * delta - 745140957231 * eta + 441340099200 * chiA * chiS * delta * eta - 73556683200 * chiA * chiA + 511264353600 * eta * chiA * chiA - 73556683200 * chiS * chiS + 224302478400 * eta * chiS * chiS + 2271682065240 * eta * eta - 871782912000 * chiA * chiA * eta * eta - 10897286400 * chiS * chiS * eta * eta - 805075876080 * eta ** 3) / 2.90594304e10 * pi_p["two"] * p2d4["two"]
        else:
            raise ValueError(f"unsupported modeTag {tag}")
        return pn

    # Source: IMRPhenomXHM_GetAmplitudeCoefficients
    # (LALSimIMRPhenomXHM_internals.c).
    def get_hm_amp_coeffs(self, pWFHM, derived, amp22_coeffs=None,
                          phase22_coeffs=None, spheroidal=None):
        """
        Amplitude coefficients of one mode across the three regions.

        The rescale factors are 0, so every ansatz returns strain.

        Parameters
        ----------
        pWFHM : dict
            Output of set_hm_waveform_variables; fAmpRDfalloff is set.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        amp22_coeffs, phase22_coeffs : dict, optional
            Coefficients of the 22 mode, needed for the (3,2).
        spheroidal : dict, optional
            Output of get_spheroidal_coeffs, needed for the (3,2).

        Returns
        -------
        dict
        """
        amp = {"ampNorm": pWFHM["ampNorm"]}

        # Miss the f^(-7/6). The pi power included in ampNorm
        amp["PNdominant"] = pWFHM["ampNorm"] * (2. / pWFHM["emm"]) ** (-7 / 6.)

        # 0: don't use coll point, 1: use point, 2: use point and
        # derivative (this only applies for boundaries)
        digits = [int(c) for c in str(pWFHM["IMRPhenomXHMIntermediateAmpVersion"])]
        amp["VersionCollocPtsInter"] = digits
        amp["nCoefficientsInter"] = sum(digits)

        amp["nCoefficientsRDAux"] = 0
        if pWFHM["MixingOn"]:
            amp["nCollocPtsRDAux"] = 2
            amp["nCoefficientsRDAux"] = 4
            amp["fRDAux"] = pWFHM["fRING"] - pWFHM["fDAMP"]  # v2

        # Take the cutting frequencies at the inspiral and ringdown
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

    # Source: IMRPhenomXHM_Inspiral_PNAmp_Ansatz
    # (LALSimIMRPhenomXHM_inspiral.c).
    def _insp_pn_amp_series(self, powers_Mf, amp):
        """
        Modulus of the PN power series, before the prefactors.

        Parameters
        ----------
        powers_Mf : dict
            Output of _initialize_powers.
        amp : dict
            Amplitude coefficients.

        Returns
        -------
        torch.Tensor
        """
        series = (amp["pnInitial"]
                  + powers_Mf["one_third"] * amp["pnOneThird"]
                  + powers_Mf["two_thirds"] * amp["pnTwoThirds"]
                  + powers_Mf["itself"] * amp["pnThreeThirds"]
                  + powers_Mf["four_thirds"] * amp["pnFourThirds"]
                  + powers_Mf["five_thirds"] * amp["pnFiveThirds"]
                  + powers_Mf["two"] * amp["pnSixThirds"])
        return torch.abs(series)

    # Source: IMRPhenomXHM_Inspiral_PNAmp_Ansatz
    # (LALSimIMRPhenomXHM_inspiral.c).
    def _insp_pn_amp(self, powers_Mf, pWFHM, amp):
        """
        FD Post-Newtonian ansatz up to 3PN without the pseudo-PN terms.

        Parameters
        ----------
        powers_Mf : dict
            Output of _initialize_powers.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp : dict
            Amplitude coefficients.

        Returns
        -------
        torch.Tensor
            Strain amplitude.
        """
        return self._insp_pn_amp_series(powers_Mf, amp) * (
            amp["PNglobalfactor"] * powers_Mf["m_seven_sixths"] * pWFHM["ampNorm"])

    # Source: IMRPhenomXHM_Get_Inspiral_Amp_Coefficients,
    # IMRPhenomXHM_Inspiral_Amp_CollocationPoints and
    # IMRPhenomXHM_Inspiral_Amp_Coefficients
    # (LALSimIMRPhenomXHM_inspiral.c).
    def _get_inspiral_amp_coefficients(self, amp, pWFHM, derived):
        """
        Pseudo-PN coefficients of the inspiral amplitude, version 123.

        Parameters
        ----------
        amp : dict
            Amplitude coefficients; updated in place.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        """
        fits = _INSP_AMP_FITS[pWFHM["modeTag"]]
        pfit = self._pwf_fit_dict(derived)

        fIN = amp["fAmpMatchIN"]
        p1 = self._initialize_powers(0.5 * fIN)
        p2 = self._initialize_powers(0.75 * fIN)
        p3 = self._initialize_powers(fIN)
        pcut = p3

        amp["fcutInsp_seven_thirds"] = pcut["seven_thirds"]
        amp["fcutInsp_eight_thirds"] = pcut["eight_thirds"]
        amp["fcutInsp_three"] = pcut["three"]

        # Get PN values at collocation points frequencies
        iv = [torch.abs(f(pfit)) for f in fits]
        pn = [self._insp_pn_amp(p, pWFHM, amp) for p in (p1, p2, p3)]

        v1 = (iv[0] - pn[0]) / amp["PNdominant"] / p1["m_seven_sixths"]
        v2 = (iv[1] - pn[1]) / amp["PNdominant"] / p2["m_seven_sixths"]
        v3 = (iv[2] - pn[2]) / amp["PNdominant"] / p3["m_seven_sixths"]

        # 123 means we use the three collocation points
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

    # Source: IMRPhenomXHM_RD_Amp_Coefficients
    # (LALSimIMRPhenomXHM_ringdown.c).
    def _rd_amp_coefficients(self, amp, pWFHM, derived):
        """
        Ringdown amplitude coefficients, versions 1 and 2.

        Version 2 uses three collocation points; version 1, for the 32,
        uses the fit of the coefficients.

        Parameters
        ----------
        amp : dict
            Amplitude coefficients; updated in place.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.

        Raises
        ------
        ValueError
            If the ringdown amplitude version is not 1 or 2.
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
            # Apply vetos to RDCP
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

        # Exponential falloff from fAmpRDfalloff, matched to Lorentzian
        ff = pWFHM["fAmpRDfalloff"]
        dfr = ff - frd
        dfd = fda * rd2
        rd3 = rd0 * fda / (torch.exp(rd1 / dfd * dfr) * (dfr * dfr + dfd * dfd))
        amp["RDCoefficient3"] = rd3
        amp["RDCoefficient4"] = -1. * self._rd_amp_dansatz(ff, pWFHM, amp) / rd3

        if amp["nCoefficientsRDAux"] > 0:
            self._rdaux_amp_coefficients(amp, pWFHM, derived)

    # Source: IMRPhenomXHM_RD_Amp_DAnsatz
    # (LALSimIMRPhenomXHM_ringdown.c).
    def _rd_amp_dansatz(self, Mf, pWFHM, amp):
        """
        Derivative of the ringdown amplitude ansatz, versions 1 and 2.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp : dict
            Amplitude coefficients.

        Returns
        -------
        torch.Tensor, same shape as Mf
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

    # Source: IMRPhenomXHM_RD_Amp_Ansatz
    # (LALSimIMRPhenomXHM_ringdown.c).
    def _rd_amp_lorentzian(self, Mf, pWFHM, amp):
        """
        Lorentzian with exponential falloff, ringdown versions 1 and 2.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp : dict
            Amplitude coefficients.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        dfr = Mf - pWFHM["fRING"]
        dfd = pWFHM["fDAMP"] * amp["RDCoefficient2"]
        return amp["RDCoefficient0"] * pWFHM["fDAMP"] / (
            torch.exp(amp["RDCoefficient1"] / dfd * dfr) * (dfr * dfr + dfd * dfd))

    # Source: IMRPhenomXHM_RDAux_Amp_Coefficients
    # (LALSimIMRPhenomXHM_ringdown.c).
    def _rdaux_amp_coefficients(self, amp, pWFHM, derived):
        """
        Coefficients of the auxiliary ringdown polynomial of the (3,2).

        Two fitted collocation points on [fAmpMatchIM, fRDAux], and the
        value and derivative of the Lorentzian at fRDAux.

        Parameters
        ----------
        amp : dict
            Amplitude coefficients; updated in place.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        """
        pfit = self._pwf_fit_dict(derived)
        fRDAux = amp["fRDAux"]
        vals = [torch.abs(f(pfit)) for f in _RD_AMP_32_RDAUX_FITS]
        vals.append(self._rd_amp_lorentzian(fRDAux, pWFHM, amp))
        vals.append(self._rd_amp_dansatz(fRDAux, pWFHM, amp))
        freqs = [amp["fAmpMatchIM"], 0.5 * (amp["fAmpMatchIM"] + fRDAux),
                 fRDAux, fRDAux]

        # Set system matrix: Polynomial at the collocation points
        # frequencies + derivative at the right boundary
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

    # Source: IMRPhenomXHM_Inspiral_Amp_NDAnsatz
    # (LALSimIMRPhenomXHM_inspiral.c).
    def _insp_amp_ndansatz(self, Mf, pWFHM, amp):
        """
        Numerical derivative of the inspiral amplitude.

        4th order finite differences; it is used for reconstructing the
        intermediate region.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp : dict
            Amplitude coefficients.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        df = 10e-10
        fun2R = self._insp_amp_ansatz(self._initialize_powers(Mf + 2 * df), pWFHM, amp)
        funR = self._insp_amp_ansatz(self._initialize_powers(Mf + df), pWFHM, amp)
        funL = self._insp_amp_ansatz(self._initialize_powers(Mf - df), pWFHM, amp)
        fun2L = self._insp_amp_ansatz(self._initialize_powers(Mf - 2 * df), pWFHM, amp)
        return (-fun2R + 8 * funR - 8 * funL + fun2L) / (12 * df)

    # Source: IMRPhenomXHM_Intermediate_Amp_CollocationPoints and
    # IMRPhenomXHM_Intermediate_Amp_Coefficients
    # (LALSimIMRPhenomXHM_intermediate.c).
    def _get_intermediate_amp_coefficients(self, amp, pWFHM, derived,
                                           amp22_coeffs, phase22_coeffs,
                                           spheroidal):
        """
        Intermediate amplitude coefficients from the collocation points.

        Equispaced points between fAmpMatchIN and fAmpMatchIM; the
        ansatz is f^(-7/6) times a polynomial.

        Parameters
        ----------
        amp : dict
            Amplitude coefficients; updated in place.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        amp22_coeffs, phase22_coeffs : dict or None
            Coefficients of the 22 mode, needed for the (3,2).
        spheroidal : dict or None
            Output of get_spheroidal_coeffs, needed for the (3,2).

        Raises
        ------
        NotImplementedError
            If the (3,2) inputs are missing.
        """
        digits = amp["VersionCollocPtsInter"]
        n = amp["nCoefficientsInter"]
        npts = pWFHM["nCollocPtsInterAmp"]
        fIN = amp["fAmpMatchIN"]
        fIM = amp["fAmpMatchIM"]
        deltaf = (fIM - fIN) / (npts - 1)

        rows = []  # (freq, is_derivative, value)
        # Left boundary: inspiral ansatz value and derivative
        pIN = self._initialize_powers(fIN)
        if digits[0] >= 1:
            rows.append((fIN, False, self._insp_amp_ansatz(pIN, pWFHM, amp)))
        if digits[0] == 2:
            rows.append((fIN, True, self._insp_amp_ndansatz(fIN, pWFHM, amp)))

        # Call parameter space fits for the in between points
        pfit = self._pwf_fit_dict(derived)
        for i in range(1, npts - 1):
            if digits[i] == 1:
                fit = _INTER_AMP_FITS[pWFHM["modeTag"]][i]
                rows.append((fIN + deltaf * i, False, torch.abs(fit(pfit))))

        # Right boundary: ringdown ansatz value and derivative
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
        # Row frequency fAmpMatchIN + deltaf * i; value at fAmpMatchIM
        f_right = fIN + deltaf * (npts - 1)
        if digits[npts - 1] >= 1:
            rows.append((f_right, False, rd_val))
        if digits[npts - 1] == 2:
            rows.append((f_right, True, rd_der))

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
    # Phase coefficients
    # ------------------------------------------------------------------

    # Source: GetfcutInsp and IMRPhenomXHM_Intermediate_CollocPtsFreqs
    # (LALSimIMRPhenomXHM_internals.c).
    def _phase_colloc_freqs(self, pWFHM, derived, phase22_coeffs):
        """
        Collocation points frequencies of the intermediate phase.

        Parameters
        ----------
        pWFHM : dict
            Output of set_hm_waveform_variables.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        phase22_coeffs : dict
            Phase coefficients of the 22 mode.

        Returns
        -------
        dict
            fPhaseMatchIN, fPhaseMatchIM and
            CollocationPointsFreqsPhaseInter.
        """
        eta = derived[:, 3:4]
        chiPNHat = derived[:, 8:9]
        dchi = derived[:, 10:11]
        fring = pWFHM["fRING"]
        fdamp = pWFHM["fDAMP"]

        fMECO = self.fMECO(eta, chiPNHat, dchi)
        eta_factor = 1.0 + 0.001 * (0.25 / eta - 1.)
        fcut = eta_factor * pWFHM["emm"] * 0.5 * fMECO

        out = {"fPhaseMatchIN": pWFHM["fMECOlm"]}
        if pWFHM["modeTag"] == 32:
            fRD22 = phase22_coeffs["fRING"]
            fdamp22 = phase22_coeffs["fDAMP"]
            fEnd = fRD22 - 0.5 * fdamp22
            # we use first and second derivative at fEnd, so this
            # frequency is duplicated here
            freqs = [fcut,
                     (math.sqrt(3) * (fcut - fEnd) + 2 * (fcut + fEnd)) / 4.,
                     (3 * fcut + fEnd) / 4.,
                     (fcut + fEnd) / 2.,
                     fEnd, fEnd]
            # correct cutting frequency for EMR with negative spins
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

    # Source: IMRPhenomXHM_GetPhaseCoefficients
    # (LALSimIMRPhenomXHM_internals.c).
    def get_hm_phase_coeffs(self, pWFHM, derived, theta, phase22_coeffs,
                            amp_coeffs=None, amp22_coeffs=None, spheroidal=None):
        """
        Phase coefficients of one mode across the three regions.

        For the (3,2), ``pWFHM`` already holds timeshift and phiref22
        from get_spheroidal_coeffs.

        Parameters
        ----------
        pWFHM : dict
            Output of set_hm_waveform_variables; timeshift and phiref22
            are set for the modes without mixing.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        theta : torch.Tensor
            Aligned layout, with m1 >= m2.
        phase22_coeffs : dict
            Phase coefficients of the 22 mode.
        amp_coeffs, amp22_coeffs : dict, optional
            Amplitude coefficients of the mode and of the 22, needed for
            the (3,2).
        spheroidal : dict, optional
            Output of get_spheroidal_coeffs, needed for the (3,2).

        Returns
        -------
        dict

        Raises
        ------
        ValueError
            If the (3,2) inputs are missing.
        """
        if pWFHM["MixingOn"] == 1 and (
                spheroidal is None or amp_coeffs is None or amp22_coeffs is None):
            raise ValueError("(3,2) phase needs spheroidal/amp/amp22 context")

        eta = derived[:, 3:4]
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
        # phase derivative at that point; time-shift wf so that modes
        # peak around t=0
        fits = _INTER_PHASE_FITS[pWFHM["modeTag"]]
        values = [fits[i](pfit) + pWFHM["DeltaT"] for i in range(6)]

        fcutRD = pPhase["fPhaseMatchIM"]
        fcutInsp = pPhase["fPhaseMatchIN"]

        # Inspiral: rescale PhenX and apply PN corrections
        z = torch.zeros_like(eta)
        phenXnonLog = [phase22_coeffs[f"phi{i}"] for i in range(10)] + [z, z, z]
        phenXLog = [z, z, z, z, z, phase22_coeffs["phi5L"], phase22_coeffs["phi6L"],
                    z, phase22_coeffs["phi8L"], phase22_coeffs["phi9L"], z, z, z]
        pseudoPN = [z] * 8 + [phase22_coeffs[f"sigma{i}"] for i in range(1, 6)]
        # rescale the coefficients of phenX by applying
        # phi_lm(f) ~ m/2 * phi_22(2/m f)
        fact = torch.div(phase22_coeffs["phiNorm"], eta)
        log_m_over_2 = math.log(m_over_2)
        phi, phiL = [], []
        for i in range(13):
            scale = m_over_2 ** ((8 - i) / 3.)
            nl = phenXnonLog[i] * fact
            lg = phenXLog[i] * fact
            ps = pseudoPN[i] * fact
            phi.append((nl + ps - lg * log_m_over_2) * scale)
            phiL.append(lg * scale)

        # if the mass-ratio is not extreme, use the complex PN
        # amplitudes to correct the orbital phase at linear order in f,
        # else use some phenomenological fits
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
            # for high-spin cases: avoid sharp transitions for 21 mode
            # by extending the inspiral region into the intermediate one
            if pWFHM["modeTag"] == 21:
                # C1Int, C2Int, C1MRD and C2MRD are zero at this point
                unconnected = {**phase22_coeffs, "C1Int": 0., "C2Int": 0.,
                               "C1MRD": 0., "C2MRD": 0.}
                insp_vals = []
                for i in range(3):
                    FF = two_over_m * freqs[i]
                    insp_vals.append((1. / eta) * self.dphase(FF, unconnected))
                diff12 = insp_vals[0] - insp_vals[1]
                diff23 = insp_vals[1] - insp_vals[2]
                new1 = values[2] + diff23
                new0 = new1 + diff12
                mask = STotR >= 0.8
                values[1] = torch.where(mask, new1, values[1])
                values[0] = torch.where(mask, new0, values[0])

            # choose collocation points according to spin/mass ratio;
            # we have to pick 5 out of 6 available fits
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

            # Ringdown: rescaling of the 22 ansatz
            wlm = 2. if ell == emm else emm / 3.
            pPhase["alpha0"] = z
            pPhase["alpha2"] = 1. / (fring * fring) * wlm * _RD_PHASE_22_ALPHA2(pfit)
            pPhase["alphaL"] = (1. / eta) * _RD_PHASE_22_ALPHAL(pfit)
            pPhase["phi0RD"] = self._rd_phase_ansatz_int(fcutRD, pWFHM, pPhase)
            pPhase["dphi0RD"] = self._rd_phase_ansatz(fcutRD, pWFHM, pPhase)

            # Intermediate: set up the linear system on
            # {1, Lorentzian, 1/f, 1/f^2, 1/f^4}
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
            # mode-mixing on: compute dphi/df and d2phi/df2 at fcutRD
            # starting from rotation of spheroidal ansatz
            fstep = self._S2S_FSTEP
            args, shifts = [], []
            for k in (-1, 0, 1):
                S = self._spheroidal_to_spherical(
                    fcutRD + k * fstep, derived, pWFHM, amp_coeffs,
                    spheroidal, amp22_coeffs, phase22_coeffs)
                # make sure that all the three points belong to the same
                # branch of mod
                a = torch.fmod(torch.angle(S), 2. * math.pi)
                args.append(a)
                shifts.append(torch.where(a > 0, torch.full_like(a, 2. * math.pi),
                                          torch.zeros_like(a)))
            # store spherical-harmonic phase at fcutRD and apply the FD
            # schemes to get first and second phase derivatives
            # long double phase_args: the 2 pi shifts are kept apart
            fstep_m1 = 1. / fstep
            pPhase["phi0RD"] = args[1] - shifts[1]
            pPhase["dphi0RD"] = 0.5 * fstep_m1 * ((args[2] - args[0]) - (shifts[2] - shifts[0]))
            d2phi0RD = fstep_m1 * fstep_m1 * ((args[2] - 2. * args[1] + args[0])
                                              - (shifts[2] - 2. * shifts[1] + shifts[0]))

            # first derivative: if the mass-ratio is not extreme, use
            # the FD value, else keep the original value of the fit
            emr = eta > pWFHM["etaEMR"]
            row_freqs = list(freqs)
            row_vals = list(values)
            row_freqs[n - 2] = torch.where(emr, fcutRD, freqs[n - 2])
            row_vals[n - 2] = torch.where(emr, pPhase["dphi0RD"], values[n - 2])
            row_freqs[n - 1] = fcutRD
            row_vals[n - 1] = d2phi0RD
            pPhase["CollocationPointsFreqsPhaseInter"] = row_freqs

            # set up the linear system; the last point equals the second
            # derivative of the rotated spheroidal ansatz
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

        # currently the 32 mode is calibrated using one extra point
        # if the mass-ratio is extreme, glue intermediate and ringdown
        if pWFHM["modeTag"] == 32:
            pPhase["c3"] = x[:, 5, :]
            glued = (pPhase["c0"] + pPhase["dphi0RD"]
                     - self._inter_phase_ansatz(fcutRD, pWFHM, pPhase))
            pPhase["c0"] = torch.where(eta < pWFHM["etaEMR"], glued, pPhase["c0"])

        # glue inspiral and intermediate, then ringdown and intermediate
        pPhase["C1INSP"] = (self._inter_phase_ansatz(fcutInsp, pWFHM, pPhase)
                            - self._insp_phase_ansatz(fcutInsp, pPhase))
        pPhase["CINSP"] = (-(pPhase["C1INSP"] * fcutInsp)
                           + self._inter_phase_ansatz_int(fcutInsp, pWFHM, pPhase)
                           - self._insp_phase_ansatz_int(fcutInsp, pPhase))
        pPhase["C1RD"] = self._inter_phase_ansatz(fcutRD, pWFHM, pPhase) - pPhase["dphi0RD"]
        pPhase["CRD"] = (-pPhase["C1RD"] * fcutRD
                         + self._inter_phase_ansatz_int(fcutRD, pWFHM, pPhase)
                         - pPhase["phi0RD"])

        # compute explicitly the phase normalization to be applied to
        # IMRPhenomX; when mode-mixing is on this will have been already
        # computed in GetSpheroidalCoefficients
        if pWFHM["MixingOn"] == 0:
            timeshift = self.time_shift_22(derived, phase22_coeffs)
            MfRef = self.f_ref * M_s
            phiref22 = (-1. / eta * self.phase(MfRef, phase22_coeffs)
                        - timeshift * MfRef - pWFHM["phaseshift"]
                        + 2.0 * phic + math.pi / 4.)
            pWFHM["timeshift"] = timeshift
            pWFHM["phiref22"] = phiref22
        else:
            timeshift = pWFHM["timeshift"]
            phiref22 = pWFHM["phiref22"]

        # align each mode so that at low-f its relative phase wrt the 22
        # agrees with PN
        fMECO22 = self.fMECO(eta, chiPNHat, dchi)
        falign = torch.where(eta > pWFHM["etaEMR"],
                             0.6 * m_over_2 * fMECO22, m_over_2 * fMECO22)
        deltaphiLM = (m_over_2 * (1. / eta * self.phase(two_over_m * falign, phase22_coeffs)
                                  + pWFHM["phaseshift"] + phiref22)
                      + timeshift * falign - 3. * math.pi / 4. * (1 - m_over_2)
                      - (self._insp_phase_ansatz_int(falign, pPhase)
                         + pPhase["C1INSP"] * falign + pPhase["CINSP"]))
        deltaphiLM = torch.fmod(deltaphiLM, 2. * math.pi)

        # the sign of the 21 amplitude changes across the parameter
        # space, we add Pi to the phase to account for this
        if pWFHM["modeTag"] == 21:
            ampsign = self._pn21_amp_sign(0.008, derived)
            deltaphiLM = torch.where(ampsign >= 0, deltaphiLM + math.pi, deltaphiLM)
        pPhase["deltaphiLM"] = deltaphiLM
        return pPhase

    # Source: IMRPhenomXHM_PN21AmpSign (LALSimIMRPhenomXHM_internals.c).
    @staticmethod
    def _pn21_amp_sign(ff, derived):
        """
        Expansion in frequency of the real part of the 21 amplitude.

        This function occasionally changes sign; its sign sets the pi
        offset of the 21 phase.

        Parameters
        ----------
        ff : float
            Geometric frequency.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.

        Returns
        -------
        torch.Tensor, shape (B, 1)
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

    # Source: GetSpheroidalCoefficients and
    # IMRPhenomXHM_Ringdown_CollocPtsFreqs
    # (LALSimIMRPhenomXHM_internals.c).
    def get_spheroidal_coeffs(self, pWFHM, derived, theta, phase22_coeffs):
        """
        Ringdown phase of the (3,2) in a spheroidal-harmonics basis.

        Ringdown phase version 122019; also sets timeshift and phiref22
        in ``pWFHM``.

        Parameters
        ----------
        pWFHM : dict
            Output of set_hm_waveform_variables; updated in place.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        theta : torch.Tensor
            Aligned layout, with m1 >= m2.
        phase22_coeffs : dict
            Phase coefficients of the 22 mode.

        Returns
        -------
        dict
            alpha0_S, alphaL_S, alpha2_S, alpha4_S and phi0_S.
        """
        eta = derived[:, 3:4]
        M_s = derived[:, 2:3]
        phic = theta[:, 6:7]
        pfit = self._pwf_fit_dict(derived)
        fringlm = pWFHM["fRING"]
        fdamplm = pWFHM["fDAMP"]
        fring22 = phase22_coeffs["fRING"]
        fdamp22 = phase22_coeffs["fDAMP"]

        # Collocation points frequencies, version 122019
        freqs = [fring22,
                 fringlm - 1.5 * fdamplm,
                 fringlm - 0.5 * fdamplm,
                 fringlm + 0.5 * fdamplm]
        values = [f(pfit) for f in _RD_PHASE_32_FITS]

        # solve the linear system on {1, Lorentzian, f^-2, f^-4}
        n = 4
        B = derived.shape[0]
        A = torch.empty((B, n, n), dtype=derived.dtype, device=derived.device)
        b = torch.empty((B, n, 1), dtype=derived.dtype, device=derived.device)
        for i in range(n):
            ff = freqs[i]
            ffm1 = 1. / ff
            ffm2 = ffm1 * ffm1
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

        # time-shift of spheroidal ansatz
        frefRD = fring22 + fdamp22
        tshift = _RD_PHASE_32_TIMESHIFT(pfit)
        timeshift = self.time_shift_22(derived, phase22_coeffs)
        pWFHM["timeshift"] = timeshift
        # impose that dphiS(fref)-dphi22(fref) has the value of the fit
        dphi22ref = (1. / eta) * self.dphase(frefRD, phase22_coeffs) + timeshift
        sph["alpha0_S"] = sph["alpha0_S"] + dphi22ref + tshift \
            - self._rd_phase_ansatz_spheroidal(frefRD, pWFHM, sph)

        # phase-shift of spheroidal ansatz
        frefRD = fring22
        MfRef = self.f_ref * M_s
        phiref22 = (-1. / eta * self.phase(MfRef, phase22_coeffs)
                    - timeshift * MfRef - pWFHM["phaseshift"]
                    + 2.0 * phic + math.pi / 4.)
        pWFHM["phiref22"] = phiref22
        phi22ref = (1. / eta * self.phase(frefRD, phase22_coeffs)
                    + timeshift * frefRD + pWFHM["phaseshift"] + phiref22)
        phishift = _RD_PHASE_32_PHASESHIFT(pfit)
        sph["phi0_S"] = phi22ref \
            - self._rd_phase_ansatz_int_spheroidal(frefRD, pWFHM, sph) + phishift
        return sph

    # Source: IMRPhenomXHM_RD_Phase_Ansatz
    # (LALSimIMRPhenomXHM_ringdown.c).
    @staticmethod
    def _rd_phase_ansatz_spheroidal(Mf, pWFHM, sph):
        """
        dphi/df of the spheroidal ringdown of the (3,2), version 122019.

        alpha0 + alpha2/f^2 + alpha4/f^4 + alphaL Lorentzian.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        sph : dict
            Output of get_spheroidal_coeffs.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]
        p = IMRPhenomXAS._initialize_powers(Mf)
        return (sph["alpha0_S"] + sph["alpha2_S"] * p["m_two"] + sph["alpha4_S"] * p["m_four"]
                + sph["alphaL_S"] * fda / (fda * fda + (Mf - frd) * (Mf - frd)))

    # Source: IMRPhenomXHM_RD_Phase_AnsatzInt
    # (LALSimIMRPhenomXHM_ringdown.c).
    @staticmethod
    def _rd_phase_ansatz_int_spheroidal(Mf, pWFHM, sph):
        """
        Phase of the spheroidal ringdown of the (3,2), version 122019.

        phi0 + f alpha0 - alpha4/(3 f^3) - alpha2/f
        + alphaL atan((f - fRDlm)/fdamplm).

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        sph : dict
            Output of get_spheroidal_coeffs.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]
        p = IMRPhenomXAS._initialize_powers(Mf)
        return (sph["phi0_S"] + sph["alpha0_S"] * Mf - sph["alpha2_S"] * p["m_one"]
                - 1. / 3. * sph["alpha4_S"] * p["m_three"]
                + sph["alphaL_S"] * torch.atan((Mf - frd) / fda))

    # Source: XLALSimIMRPhenomXRingdownAmplitude22AnsatzAnalytical
    # (LALSimIMRPhenomXUtilities.c).
    @staticmethod
    def _rd22_amp_analytical(Mf, amp22_coeffs):
        """
        Analytical ringdown amplitude of the 22 mode, in NR units.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        amp22_coeffs : dict
            Amplitude coefficients of the 22 mode, with gammaR,
            gammaD13 and gammaD2.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        dfr = Mf - amp22_coeffs["fRING"]
        return (torch.exp(-dfr * amp22_coeffs["gammaR"]) * amp22_coeffs["gammaD13"]
                / (dfr * dfr + amp22_coeffs["gammaD2"]))

    # Source: SpheroidalToSpherical (LALSimIMRPhenomXHM_internals.c).
    def _spheroidal_to_spherical(self, Mf, derived, pWFHM, amp, sph,
                                 amp22_coeffs, phase22_coeffs):
        """
        Spheroidal to spherical rotation of the (3,2) ringdown.

        h32_spherical = a1 * h22_spheroidal + a2 * h32_spheroidal, where
        we assume that h22_spheroidal = h22_spherical.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        pWFHM : dict
            Output of set_hm_waveform_variables, with timeshift and
            phiref22.
        amp : dict
            Amplitude coefficients of the (3,2).
        sph : dict
            Output of get_spheroidal_coeffs.
        amp22_coeffs, phase22_coeffs : dict
            Coefficients of the 22 mode.

        Returns
        -------
        torch.Tensor, complex, same shape as Mf
        """
        eta = derived[:, 3:4]
        amp22 = self._rd22_amp_analytical(Mf, amp22_coeffs)
        phi22 = (1. / eta * self.phase(Mf, phase22_coeffs)
                 + pWFHM["timeshift"] * Mf + pWFHM["phaseshift"] + pWFHM["phiref22"])
        wf22R = amp22 * torch.polar(torch.ones_like(phi22), phi22)
        # Strain units; m_seven_sixths built as m_one_sixth * m_one
        m76 = (1. / Mf ** (1.0 / 6.0)) * (1. / Mf)
        wf22R = wf22R * (pWFHM["ampNorm"] * m76)
        amplm = self._rd_amp_ansatz_core(Mf, pWFHM, amp)
        philm = self._rd_phase_ansatz_int_spheroidal(Mf, pWFHM, sph)
        a322 = pWFHM["mixingCoeffs"][2]
        a323 = pWFHM["mixingCoeffs"][3]
        return (torch.conj(a322) * wf22R
                + torch.conj(a323) * amplm * torch.polar(torch.ones_like(philm), philm))

    # Source: SpheroidalToSphericalRecycle
    # (LALSimIMRPhenomXHM_internals.c).
    def _spheroidal_to_spherical_recycle(self, Mf, pWFHM, amp, sph, h22):
        """
        Rotation of the (3,2) using the previously computed 22 mode.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp : dict
            Amplitude coefficients of the (3,2).
        sph : dict
            Output of get_spheroidal_coeffs.
        h22 : torch.Tensor, complex, same shape as Mf
            The 22 mode.

        Returns
        -------
        torch.Tensor, complex, same shape as Mf
        """
        wf22R = _cdiv(h22, pWFHM["Amp0"])
        amplm = self._rd_amp_ansatz_core(Mf, pWFHM, amp)
        philm = self._rd_phase_ansatz_int_spheroidal(Mf, pWFHM, sph)
        a322 = pWFHM["mixingCoeffs"][2]
        a323 = pWFHM["mixingCoeffs"][3]
        return (torch.conj(a322) * wf22R
                + torch.conj(a323) * amplm * torch.polar(torch.ones_like(philm), philm))

    # ------------------------------------------------------------------
    # Region ansatze
    # ------------------------------------------------------------------

    # Source: IMRPhenomXHM_Inspiral_Amp_Ansatz
    # (LALSimIMRPhenomXHM_inspiral.c).
    def _insp_amp_ansatz(self, powers_Mf, pWFHM, amp):
        """
        Complete inspiral amplitude ansatz: PN ansatz + pseudo-PN terms.

        Parameters
        ----------
        powers_Mf : dict
            Output of _initialize_powers.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp : dict
            Amplitude coefficients.

        Returns
        -------
        torch.Tensor
            Strain amplitude.
        """
        pn = self._insp_pn_amp(powers_Mf, pWFHM, amp)
        pseudoterms = (powers_Mf["seven_thirds"] / amp["fcutInsp_seven_thirds"] * amp["InspiralCoefficient0"]
                       + powers_Mf["eight_thirds"] / amp["fcutInsp_eight_thirds"] * amp["InspiralCoefficient1"]
                       + powers_Mf["three"] / amp["fcutInsp_three"] * amp["InspiralCoefficient2"])
        pseudoterms = pseudoterms * (powers_Mf["m_seven_sixths"] * amp["PNdominant"])
        return pn + pseudoterms

    # Source: IMRPhenomXHM_Intermediate_Amp_Ansatz
    # (LALSimIMRPhenomXHM_intermediate.c).
    def _inter_amp_ansatz(self, powers_Mf, pWFHM, amp):
        """
        Intermediate amplitude ansatz, f^(-7/6) * polynomial.

        Parameters
        ----------
        powers_Mf : dict
            Output of _initialize_powers.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp : dict
            Amplitude coefficients.

        Returns
        -------
        torch.Tensor
        """
        ff = powers_Mf["itself"]
        result = torch.zeros_like(ff)
        fpower = torch.ones_like(ff)
        for i in range(amp["nCoefficientsInter"]):
            result = result + amp[f"InterCoefficient{i}"] * fpower
            fpower = fpower * ff
        return result * powers_Mf["m_seven_sixths"]

    # Source: IMRPhenomXHM_RD_Amp_Ansatz
    # (LALSimIMRPhenomXHM_ringdown.c).
    def _rd_amp_ansatz_core(self, ff, pWFHM, amp):
        """
        Ringdown amplitude ansatz, versions 1 and 2.

        Polynomial below fRDAux (32 only), exponential falloff from
        fAmpRDfalloff, else Lorentzian. For the modes with mixing this
        is the ansatz of the spheroidal part.

        Parameters
        ----------
        ff : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp : dict
            Amplitude coefficients.

        Returns
        -------
        torch.Tensor, same shape as ff
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

    # Source: IMRPhenomXHM_RD_Amp_Ansatz
    # (LALSimIMRPhenomXHM_ringdown.c).
    def _rd_amp_ansatz(self, powers_Mf, pWFHM, amp):
        """
        Ringdown amplitude ansatz on the powers of Mf.

        Parameters
        ----------
        powers_Mf : dict
            Output of _initialize_powers.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp : dict
            Amplitude coefficients.

        Returns
        -------
        torch.Tensor
        """
        return self._rd_amp_ansatz_core(powers_Mf["itself"], pWFHM, amp)

    # Source: IMRPhenomXHM_Intermediate_Amp_CollocationPoints
    # (LALSimIMRPhenomXHM_intermediate.c), IMRPhenomXHM_RD_Amp_NDAnsatz
    # (LALSimIMRPhenomXHM_ringdown.c).
    def _spheroidal_boundary_value_derivative(self, Mf, derived, pWFHM, amp,
                                              spheroidal, amp22_coeffs,
                                              phase22_coeffs):
        """
        Amplitude of SpheroidalToSpherical and its derivative at Mf.

        The derivative of the RD ansatz for modes with mixing can not be
        obtained analytically, so we use finite difference of 4th order.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, 1)
            Right boundary of the intermediate amplitude region.
        derived, pWFHM, amp, spheroidal, amp22_coeffs, phase22_coeffs
            As for _spheroidal_to_spherical.

        Returns
        -------
        val, der : torch.Tensor, shape (B, 1)
        """
        def cabs_s2s(f):
            return torch.abs(self._spheroidal_to_spherical(
                f, derived, pWFHM, amp, spheroidal, amp22_coeffs, phase22_coeffs))

        val = cabs_s2s(Mf)
        df = 10e-10
        der = (-cabs_s2s(Mf + 2 * df) + 8 * cabs_s2s(Mf + df)
               - 8 * cabs_s2s(Mf - df) + cabs_s2s(Mf - 2 * df)) / (12 * df)
        return val, der

    # Source: IMRPhenomXHM_Inspiral_Phase_AnsatzInt
    # (LALSimIMRPhenomXHM_inspiral.c).
    @staticmethod
    def _insp_phase_ansatz_int(Mf, pPhase):
        """
        Inspiral phase from the rescaled phenX coefficients.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pPhase : dict
            Phase coefficients, with phi and phiL.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        p = IMRPhenomXAS._initialize_powers(Mf)
        freqs = (p["m_five_thirds"], p["m_four_thirds"], p["m_one"], p["m_two_thirds"],
                 p["m_one_third"], 1., p["one_third"], p["two_thirds"], Mf, p["four_thirds"],
                 p["five_thirds"], p["two"], p["seven_thirds"])
        philm = 0.
        for i in range(13):
            philm = philm + (pPhase["phi"][i] + pPhase["phiL"][i] * p["log"]) * freqs[i]
        return philm

    # Source: IMRPhenomXHM_Inspiral_Phase_Ansatz
    # (LALSimIMRPhenomXHM_inspiral.c).
    @staticmethod
    def _insp_phase_ansatz(Mf, pPhase):
        """
        dphi/df of the inspiral phase ansatz.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pPhase : dict
            Phase coefficients, with phi and phiL.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        p = IMRPhenomXAS._initialize_powers(Mf)
        coeffs = (-5. / 3, -4. / 3, -1., -2. / 3, -1. / 3, 0., 1. / 3, 2. / 3, 1., 4. / 3,
                  5. / 3, 2., 7. / 3)
        freqs = (p["m_eight_thirds"], p["m_seven_thirds"], p["m_two"], p["m_five_thirds"],
                 p["m_four_thirds"], p["m_one"], p["m_two_thirds"], p["m_one_third"], 1.,
                 p["one_third"], p["two_thirds"], Mf, p["four_thirds"])
        dphilm = 0.
        for i in range(13):
            dphilm = dphilm + ((pPhase["phi"][i] + pPhase["phiL"][i] * p["log"]) * coeffs[i]
                               + pPhase["phiL"][i]) * freqs[i]
        return dphilm

    # Source: IMRPhenomXHM_Inter_Phase_AnsatzInt
    # (LALSimIMRPhenomXHM_intermediate.c).
    @staticmethod
    def _inter_phase_ansatz_int(Mf, pWFHM, pPhase):
        """
        Intermediate phase ansatz.

        c0 f + c1 log f - c2/f - c4/(3 f^3) + cL atan((f - fRD)/fdamp),
        and - c3/(2 f^2) for the 32.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        pPhase : dict
            Phase coefficients.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        p = IMRPhenomXAS._initialize_powers(Mf)
        out = (pPhase["c0"] * Mf + pPhase["c1"] * p["log"]
               - pPhase["c2"] * p["m_one"] - 1. / 3. * pPhase["c4"] * p["m_three"])
        if pWFHM["modeTag"] == 32:
            out = out - 0.5 * pPhase["c3"] * p["m_two"]
        return out + pPhase["cL"] * torch.atan((Mf - pWFHM["fRING"]) / pWFHM["fDAMP"])

    # Source: IMRPhenomXHM_Inter_Phase_Ansatz
    # (LALSimIMRPhenomXHM_intermediate.c).
    @staticmethod
    def _inter_phase_ansatz(Mf, pWFHM, pPhase):
        """
        dphi/df of the intermediate phase ansatz.

        c0 + c1/f + c2/f^2 + c4/f^4 + cL fdamp/(fdamp^2 + (f - fRD)^2),
        and + c3/f^3 for the 32.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        pPhase : dict
            Phase coefficients.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        p = IMRPhenomXAS._initialize_powers(Mf)
        fda = pWFHM["fDAMP"]
        frd = pWFHM["fRING"]
        out = (pPhase["c0"] + pPhase["c1"] * p["m_one"] + pPhase["c2"] * p["m_two"]
               + pPhase["c4"] * p["m_four"])
        if pWFHM["modeTag"] == 32:
            out = out + pPhase["c3"] * p["m_three"]
        return out + pPhase["cL"] * fda / (fda * fda + (Mf - frd) * (Mf - frd))

    # Source: IMRPhenomXHM_RD_Phase_Ansatz
    # (LALSimIMRPhenomXHM_ringdown.c).
    @staticmethod
    def _rd_phase_ansatz(Mf, pWFHM, pPhase):
        """
        dphi/df of the ringdown, rescaling of the 22 ansatz.

        Used for (21), (33), (44): alpha0 + fRDlm^2 alpha2/f^2
        + alphaL Lorentzian.

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        pPhase : dict
            Phase coefficients.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]
        m_two = IMRPhenomXAS._initialize_powers(Mf)["m_two"]
        return (pPhase["alpha0"] + frd * frd * pPhase["alpha2"] * m_two
                + pPhase["alphaL"] * fda / (fda * fda + (Mf - frd) * (Mf - frd)))

    # Source: IMRPhenomXHM_RD_Phase_AnsatzInt
    # (LALSimIMRPhenomXHM_ringdown.c).
    @staticmethod
    def _rd_phase_ansatz_int(Mf, pWFHM, pPhase):
        """
        Ringdown phase, rescaling of the 22 ansatz.

        alpha0 f - fRDlm^2 alpha2/f + alphaL atan((f - fRDlm)/fdamplm).

        Parameters
        ----------
        Mf : torch.Tensor
            Geometric frequency.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        pPhase : dict
            Phase coefficients.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        frd = pWFHM["fRING"]
        fda = pWFHM["fDAMP"]
        m_one = IMRPhenomXAS._initialize_powers(Mf)["m_one"]
        return (pPhase["alpha0"] * Mf - frd * frd * pPhase["alpha2"] * m_one
                + pPhase["alphaL"] * torch.atan((Mf - frd) / fda))

    # ------------------------------------------------------------------
    # Inspiral, merger and ringdown reconstruction
    # ------------------------------------------------------------------

    # Source: IMRPhenomXHM_Amplitude_noModeMixing,
    # IMRPhenomXHM_Amplitude_ModeMixing and
    # IMRPhenomXHM_Amplitude_ModeMixingRecycle
    # (LALSimIMRPhenomXHM_internals.c).
    def amp_lm(self, Mf, derived, pWFHM, amp_coeffs, phase_coeffs=None,
               amp22_coeffs=None, phase22_coeffs=None, spheroidal=None, h22=None):
        """
        Amplitude of mode (l, m) through inspiral, merger and ringdown.

        In NR units, with negative values set to FALSE_ZERO; an Ampzero
        mode is zero. The (3,2) ringdown rotates ``h22`` when given.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F)
            Geometric frequencies.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        amp_coeffs : dict
            Output of get_hm_amp_coeffs.
        phase_coeffs : dict, optional
            Not used by the amplitude.
        amp22_coeffs, phase22_coeffs, spheroidal : dict, optional
            Coefficients of the 22 mode and spheroidal ringdown, for the
            (3,2).
        h22 : torch.Tensor, complex, shape (B, F), optional
            The 22 mode, recycled for the (3,2).

        Returns
        -------
        torch.Tensor, shape (B, F)
        """
        p = self._initialize_powers(Mf)
        insp = self._insp_amp_ansatz(p, pWFHM, amp_coeffs)
        inter = self._inter_amp_ansatz(p, pWFHM, amp_coeffs)
        if pWFHM["MixingOn"] == 1 and h22 is not None:
            rd = torch.abs(self._spheroidal_to_spherical_recycle(
                Mf, pWFHM, amp_coeffs, spheroidal, h22))
        elif pWFHM["MixingOn"] == 1:
            rd = torch.abs(self._spheroidal_to_spherical(
                Mf, derived, pWFHM, amp_coeffs, spheroidal,
                amp22_coeffs, phase22_coeffs))
        else:
            rd = self._rd_amp_ansatz(p, pWFHM, amp_coeffs)
        out = torch.where(Mf < amp_coeffs["fAmpMatchIN"], insp,
                          torch.where(Mf >= amp_coeffs["fAmpMatchIM"], rd, inter))
        out = torch.where(out < 0, torch.full_like(out, _FALSE_ZERO), out)
        return torch.where(pWFHM["Ampzero"], torch.zeros_like(out), out)

    # Source: IMRPhenomXHM_Phase_noModeMixing,
    # IMRPhenomXHM_Phase_ModeMixing and
    # IMRPhenomXHM_Phase_ModeMixingRecycle
    # (LALSimIMRPhenomXHM_internals.c).
    def phase_lm(self, Mf, derived, pWFHM, phase_coeffs, amp_coeffs=None,
                 amp22_coeffs=None, phase22_coeffs=None, spheroidal=None, h22=None):
        """
        Phase of mode (l, m) through inspiral, merger and ringdown.

        The (3,2) ringdown phase is carg(SpheroidalToSpherical),
        rotating ``h22`` when given.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F)
            Geometric frequencies.
        derived : torch.Tensor, shape (B, 13)
            Output of compute_derived_parameters.
        pWFHM : dict
            Output of set_hm_waveform_variables.
        phase_coeffs : dict
            Output of get_hm_phase_coeffs.
        amp_coeffs, amp22_coeffs, phase22_coeffs : dict, optional
            Amplitude coefficients of the mode and coefficients of the
            22 mode, for the (3,2).
        spheroidal : dict, optional
            Output of get_spheroidal_coeffs, for the (3,2).
        h22 : torch.Tensor, complex, shape (B, F), optional
            The 22 mode, recycled for the (3,2).

        Returns
        -------
        torch.Tensor, shape (B, F)
        """
        pPhase = phase_coeffs
        insp = (self._insp_phase_ansatz_int(Mf, pPhase)
                + pPhase["C1INSP"] * Mf + pPhase["CINSP"] + pPhase["deltaphiLM"])
        inter = (self._inter_phase_ansatz_int(Mf, pWFHM, pPhase)
                 + pPhase["deltaphiLM"])
        if pWFHM["MixingOn"] == 1:
            if h22 is not None:
                S = self._spheroidal_to_spherical_recycle(Mf, pWFHM, amp_coeffs, spheroidal, h22)
            else:
                S = self._spheroidal_to_spherical(Mf, derived, pWFHM, amp_coeffs, spheroidal,
                                                  amp22_coeffs, phase22_coeffs)
            rd = torch.angle(S) + pPhase["C1RD"] * Mf + pPhase["CRD"] + pPhase["deltaphiLM"]
        else:
            rd = (self._rd_phase_ansatz_int(Mf, pWFHM, pPhase)
                  + pPhase["C1RD"] * Mf + pPhase["CRD"] + pPhase["deltaphiLM"])
        return torch.where(Mf < pPhase["fPhaseMatchIN"], insp,
                           torch.where(Mf >= pPhase["fPhaseMatchIM"], rd, inter))

    # ------------------------------------------------------------------
    # Single-mode generator
    # ------------------------------------------------------------------

    # Source: IMRPhenomXHMGenerateFDOneMode (LALSimIMRPhenomXHM.c).
    def gethlm(self, theta, ell, emm, derived=None, amp22_coeffs=None,
               phase22_coeffs=None, afinal=None, h22=None, within=None):
        """
        Aligned-spin mode h_{l,-m}(f) on the (B, F) grid.

        The (3,2) ringdown rotates ``h22`` when given, otherwise the
        analytical 22 ringdown. ``afinal`` replaces the final spin; the
        final mass stays aligned-spin.

        Parameters
        ----------
        theta : torch.Tensor
            Aligned layout, with m1 >= m2.
        ell, emm : int
            Mode numbers, m > 0.
        derived : torch.Tensor, shape (B, 13), optional
            Output of compute_derived_parameters.
        amp22_coeffs, phase22_coeffs : dict, optional
            Coefficients of the 22 mode.
        afinal : torch.Tensor, shape (B, 1), optional
            Final spin.
        h22 : torch.Tensor, complex, shape (B, F), optional
            The 22 mode, recycled for the (3,2).
        within : torch.Tensor of bool, shape (B, F), optional
            Bins to keep; by default Mf <= fCutDef.

        Returns
        -------
        torch.Tensor, complex, shape (B, F)
        """
        if derived is None:
            derived = self.compute_derived_parameters(theta)
        if phase22_coeffs is None:
            phase22_coeffs = self.get_phase_coeffs(derived, afinal=afinal)
        Mf = self.f * derived[:, 2:3]
        if within is None:
            # Default end of the waveform, fCutDef
            fcutdef = torch.where(derived[:, 7:8] > 0.99,
                                  torch.full_like(derived[:, 7:8], 0.33),
                                  torch.full_like(derived[:, 7:8], 0.3))
            within = Mf <= fcutdef
        if amp22_coeffs is None and (ell, emm) in ((2, 2), (3, 2)):
            amp22_coeffs = self.get_amp_coeffs(derived, afinal=afinal)

        if (ell, emm) == (2, 2):
            h = self.get_h22(theta, derived, amp22_coeffs, phase22_coeffs)
            return torch.where(within, h, torch.zeros_like(h))

        pWFHM = self.set_hm_waveform_variables(ell, emm, theta, derived,
                                               afinal=afinal)
        if pWFHM["MixingOn"] == 1:
            sph = self.get_spheroidal_coeffs(pWFHM, derived, theta, phase22_coeffs)
        else:
            sph = None
        amp = self.get_hm_amp_coeffs(pWFHM, derived, amp22_coeffs,
                                     phase22_coeffs, spheroidal=sph)
        phase = self.get_hm_phase_coeffs(pWFHM, derived, theta, phase22_coeffs,
                                         amp_coeffs=amp, amp22_coeffs=amp22_coeffs,
                                         spheroidal=sph)
        A = self.amp_lm(Mf, derived, pWFHM, amp, amp22_coeffs=amp22_coeffs,
                        phase22_coeffs=phase22_coeffs, spheroidal=sph, h22=h22)
        P = self.phase_lm(Mf, derived, pWFHM, phase, amp_coeffs=amp,
                          amp22_coeffs=amp22_coeffs, phase22_coeffs=phase22_coeffs,
                          spheroidal=sph, h22=h22)
        # Multiply by (-1)^l to get the true h_l-m(f)
        Amp0 = -pWFHM["Amp0"] if ell % 2 != 0 else pWFHM["Amp0"]
        h = Amp0 * A * torch.polar(torch.ones_like(P), P)
        return torch.where(within, h, torch.zeros_like(h))

    # ------------------------------------------------------------------
    # Mode sum
    # ------------------------------------------------------------------

    # Source: IMRPhenomXHMFDAddMode (LALSimIMRPhenomXHM.c).
    def _fd_add_mode(self, hp, hc, hlm, theta_incl, phi_azim, ell, emm):
        """
        Sum one mode to hp and hc.

        Equatorial symmetry: add in -m and m mode.

        Parameters
        ----------
        hp, hc : torch.Tensor, complex, shape (B, F)
            Polarisations so far.
        hlm : torch.Tensor, complex, shape (B, F)
            The mode h_{l,-m}.
        theta_incl : torch.Tensor, shape (B, 1)
            Inclination in rad.
        phi_azim : float
            Azimuthal angle in rad.
        ell, emm : int
            Mode numbers, m > 0.

        Returns
        -------
        hp, hc : torch.Tensor, complex, shape (B, F)
        """
        minus1l = -1 if ell % 2 != 0 else 1
        Ym = self.spin_weighted_Y(theta_incl, phi_azim, -2, ell, -emm)
        Ystar = torch.conj(self.spin_weighted_Y(theta_incl, phi_azim, -2, ell, emm))
        factorp = 0.5 * (Ym + minus1l * Ystar)
        factorc = 1j * 0.5 * (Ym - minus1l * Ystar)
        return hp + factorp * hlm, hc + factorc * hlm

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    # Source: IMRPhenomXHM_MultiMode2 (LALSimIMRPhenomXHM.c).
    def get_hphc(self, theta, reproduce_lal=False):
        """
        FD plus and cross polarisations of a BBH parameter batch.

        The modes are 22, 21, 33, 32 and 44 with the respective negative
        ones, without multibanding.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 17)
            Parameters in the waveform_utils.WAVEFORM_PARAMETERS layout.
        reproduce_lal : bool
            If True, skip FD tapering, delta_tc shift and df
            normalisation so the output compares directly with LAL's,
            with PhenomXHMThresholdMband = 0.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_pad + F), complex

        Raises
        ------
        ValueError
            If any transverse spin is non-zero.
        """
        theta = waveform_utils.aligned_layout(
            theta, waveform_utils.ALIGNED_COLUMNS,
            waveform_utils.TRANSVERSE_SPIN_PARAMETERS + waveform_utils.TIDAL_PARAMETERS,
        )
        self.check_parameters(theta)
        theta = self.order_bodies(theta)
        iota = theta[:, 7:8]
        delta_tc = theta[:, 5:6]
        derived = self.compute_derived_parameters(theta)
        M_s = derived[:, 2:3]
        f_max_prime, f_last = self.checked_band(derived)
        within = self.within_band(M_s, f_max_prime, f_last)

        # 22 coefficients, computed once and used for all the modes
        amp22_coeffs = self.get_amp_coeffs(derived)
        phase22_coeffs = self.get_phase_coeffs(derived)

        hp = torch.zeros((self.B, self.f_numel), dtype=torch.complex128,
                         device=self.f.device)
        hc = torch.zeros_like(hp)

        # The 22 is added first and recycled by the (3,2). Loop over
        # only positive m is intentional: the single mode function
        # returns h_l-m, and the positive is added in FDAddMode.
        h22 = self.get_h22(theta, derived, amp22_coeffs, phase22_coeffs)
        h22 = torch.where(within, h22, torch.zeros_like(h22))
        hp, hc = self._fd_add_mode(hp, hc, h22, iota, math.pi / 2., 2, 2)
        for (ell, emm) in self.HM_MODES:
            hlm = self.gethlm(theta, ell, emm, derived, amp22_coeffs,
                              phase22_coeffs, h22=h22, within=within)
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
            hp, hc = self.apply_delta_tc(hp, hc, delta_tc)
            hp = hp * self.df
            hc = hc * self.df

        return self.pad_missing_frequencies(hp, hc)
