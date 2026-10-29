#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batched PyTorch port of LALSimulation's IMRPhenomXAS_NRTidalv3
aligned-spin BNS frequency-domain waveform (LALSimIMRPhenomX.c,
LALSimNRTunedTides.c).
"""

import math
import warnings

import torch

from sage.core.constants import GM, MSUN
from sage.data.waveform.approximants.IMRPhenomXAS import IMRPhenomXAS
from sage.data.waveform.approximants.IMRPhenomXHM import IMRPhenomXHM
from sage.data.waveform import taper as _taper_mod
from sage.data.waveform import waveform_utils
from sage.core.geometry import Window


class IMRPhenomXAS_NRTidalv3(IMRPhenomXAS, torch.nn.Module):
    """
    Batched IMRPhenomXAS with NRTidalv3 tidal corrections.

    With a ``param_sampler``, ``forward()`` returns detector-frame
    strain and standardised parameter targets; without one, the
    instance only generates waveforms.

    Parameters
    ----------
    param_sampler : DistributionSampler or None
        BNS parameter sampler (with ``lambda1`` and ``lambda2``).
    waveform_project : ConstantProjection or None
        Multi-detector projection module.
    augment : callable or None
        Optional SNR-rescaling augmentation.
    window : Window
        Window geometry; the grid runs from the low cutoff to Nyquist
        with ``df = 1 / padded_length_in_s``.
    signal_batch_size : int
        Signals per call.
    low_frequency_cutoff_in_hz, reference_frequency_in_hz : float
        Low cutoff and reference frequency.
    targets : list of str
        Estimated parameters, in head order.
    device : str or torch.device
        Device of the batches.
    dtype : torch.dtype
        Dtype of the data tensors.
    append_per_det_targets : bool
        Must be False: this model has no per-detector targets.
    """

    GRAPH_READY = True
    BODY_SWAP = [1, 0, 3, 2, 4, 5, 6, 7, 9, 8]

    def __init__(
        self,
        param_sampler,
        waveform_project,
        augment,
        window: Window,
        signal_batch_size: int,
        low_frequency_cutoff_in_hz: float,
        reference_frequency_in_hz: float,
        targets: list,
        device,
        dtype: torch.dtype,
        append_per_det_targets: bool = False,
    ):
        torch.nn.Module.__init__(self)
        if append_per_det_targets:
            raise ValueError("IMRPhenomXAS_NRTidalv3 has no per-detector targets")

        self.signal_batch_size = signal_batch_size
        self.set_band(
            low_frequency_cutoff_in_hz, reference_frequency_in_hz, window.sample_rate_in_hz / 2.0,
            window.delta_f_in_hz, window.n_frequency_bins, window.padding_length_in_s,
            signal_batch_size, device, dtype,
        )
        self.param_sampler = param_sampler
        self.waveform_project = waveform_project
        self.augment = augment

        self.extras = waveform_utils.TRANSVERSE_SPIN_PARAMETERS
        if self.param_sampler is not None:
            param_index = self.param_sampler.param_index
            parameters = waveform_utils.TIDAL_ALIGNED_PARAMETERS
            self.source, self.target = waveform_utils.layout_columns(param_index, parameters, type(self).__name__)
            self.extras = waveform_utils.given_extras(param_index, parameters)
            self.dist_col = param_index["distance"]

            self.param_sampler._compile_batch_normaliser(targets, dtype)
            self.param_sampler._compile_batch_standardiser(targets)
            self.param_sampler.to(device)

        if self.waveform_project is not None:
            self.waveform_project.to(device)

    @classmethod
    def from_band(cls, delta_f, low_frequency_cutoff_in_hz, f_max, reference_frequency_in_hz, batch_size, device):
        """
        The waveform alone, for LAL's deltaF, f_min, f_max and fRef.

        As IMRPhenomXAS.from_band.

        Returns
        -------
        IMRPhenomXAS_NRTidalv3
            Model with no sampler, projection or augmentation.
        """
        model = super().from_band(delta_f, low_frequency_cutoff_in_hz, f_max, reference_frequency_in_hz,
                                  batch_size, device)
        model.signal_batch_size = batch_size
        model.param_sampler = model.waveform_project = model.augment = None
        model.extras = waveform_utils.TRANSVERSE_SPIN_PARAMETERS
        return model

    @torch.no_grad()
    def forward(self, return_theta=False):
        """
        Sample BNS waveforms and return detector-frame strain.

        Parameters
        ----------
        return_theta : bool
            Also return the raw parameter batch.

        Returns
        -------
        hf : torch.Tensor, shape (B, D, F)
            Detector-frame strain for each detector.
        targets : torch.Tensor, shape (B, n_params + 1)
            Standardised parameter targets with a trailing signal-label
            column of ones.
        all_theta : torch.Tensor, shape (B, total_params)
            Full raw parameter batch (only when ``return_theta=True``).
        """
        all_theta = self.param_sampler(self.B)
        theta = waveform_utils.to_layout(all_theta, self.source, self.target)

        hp, hc = self.get_hphc(theta)
        hp = hp.to(self._out_cdtype)
        hc = hc.to(self._out_cdtype)

        hf = self.waveform_project(
            hp,
            hc,
            ra=theta[:, -2],
            dec=theta[:, -1],
            polarization=theta[:, -3],
        )

        if self.augment:
            hf, scale = self.augment(hf)
            all_theta[:, self.dist_col] = (
                all_theta[:, self.dist_col] / scale.to(all_theta.dtype)
            )

        normed_targets = self.param_sampler.standardise_from_batch(all_theta)
        targets = torch.cat(
            [normed_targets, torch.ones_like(normed_targets[:, :1])], dim=1
        )

        if return_theta:
            return hf, targets, all_theta
        return hf, targets


    # Source: XLALSimNRTunedTidesComputeKappa2T (LALSimNRTunedTides.c).
    @staticmethod
    def _kappa2T(Xa, Xb, lambda1, lambda2):
        """
        Tidal coupling constant kappa2T.

        Parameters
        ----------
        Xa, Xb : torch.Tensor, shape (B, 1)
            Masses of companions 1 and 2 divided by the total mass.
        lambda1, lambda2 : torch.Tensor, shape (B, 1)
            Dimensionless tidal deformabilities.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        term1 = (1.0 + 12.0*Xb/Xa) * Xa.pow(5) * lambda1
        term2 = (1.0 + 12.0*Xa/Xb) * Xb.pow(5) * lambda2
        return (3.0/13.0) * (term1 + term2)

    # Source: XLALSimNRTunedTidesSetFDTidalPhase_v3_Coeffs
    # (LALSimNRTunedTides.c).
    @staticmethod
    def _kappaAB(Xa, Xb, lambda1, lambda2):
        """
        Tidal coupling constants kappaA and kappaB of the two stars.

        Parameters
        ----------
        Xa, Xb : torch.Tensor, shape (B, 1)
            Masses of companions 1 and 2 divided by the total mass.
        lambda1, lambda2 : torch.Tensor, shape (B, 1)
            Dimensionless tidal deformabilities.

        Returns
        -------
        kappaA, kappaB : torch.Tensor, shape (B, 1)
        """
        kappaA = 3.0 * Xb * Xa.pow(4) * lambda1
        kappaB = 3.0 * Xa * Xb.pow(4) * lambda2
        return kappaA, kappaB

    # Source: XLALSimNRTunedTidesSetFDTidalPhase_PN_Coeffs
    # (LALSimNRTunedTides.c).
    @staticmethod
    def _PN_coeffs(Xa):
        """
        7.5PN tidal phase coefficients.

        Parameters
        ----------
        Xa : torch.Tensor, shape (B, 1)
            Mass of companion 1 divided by total mass.

        Returns
        -------
        torch.Tensor, shape (B, 10)
            c_Newt, c_1, c_3over2, c_2 and c_5over2 of companion 1, then
            of companion 2.
        """
        Xb   = 1.0 - Xa
        Xa2  = Xa*Xa;  Xa3 = Xa2*Xa;  Xa4 = Xa3*Xa;  Xa5 = Xa4*Xa
        Xb2  = Xb*Xb;  Xb3 = Xb2*Xb;  Xb4 = Xb3*Xb;  Xb5 = Xb4*Xb

        pi    = math.pi
        den_a = 11.0*Xa - 12.0
        den_b = 11.0*Xb - 12.0

        # 7.5PN Coefficients
        c_NewtA = -3.0*den_a / (16.0*Xa*Xb2)
        c_1A    = (-1300.0*Xa3 + 11430.0*Xa2 + 4595.0*Xa - 15895.0) / (672.0*den_a)
        c_3o2A  = torch.full_like(Xa, -pi)
        c_2A    = (22861440.0*Xa5 - 102135600.0*Xa4 + 791891100.0*Xa3
                   + 874828080.0*Xa2 + 216234195.0*Xa
                   - 1939869350.0) / (27433728.0*den_a)
        c_5o2A  = -pi*(10520.0*Xa3 - 7598.0*Xa2 + 22415.0*Xa - 27719.0) / (672.0*den_a)

        c_NewtB = -3.0*den_b / (16.0*Xb*Xa2)
        c_1B    = (-1300.0*Xb3 + 11430.0*Xb2 + 4595.0*Xb - 15895.0) / (672.0*den_b)
        c_3o2B  = torch.full_like(Xa, -pi)
        c_2B    = (22861440.0*Xb5 - 102135600.0*Xb4 + 791891100.0*Xb3
                   + 874828080.0*Xb2 + 216234195.0*Xb
                   - 1939869350.0) / (27433728.0*den_b)
        c_5o2B  = -pi*(10520.0*Xb3 - 7598.0*Xb2 + 22415.0*Xb - 27719.0) / (672.0*den_b)

        return torch.cat([c_NewtA, c_1A, c_3o2A, c_2A, c_5o2A,
                          c_NewtB, c_1B, c_3o2B, c_2B, c_5o2B], dim=-1)

    # Source: XLALSimNRTunedTidesSetFDTidalPhase_v3_Coeffs
    # (LALSimNRTunedTides.c).
    @staticmethod
    def _nrtv3_coeffs(Xa, Xb, kappa2T, kappaA, kappaB, PN):
        """
        NRTidalv3 effective Love number and phase coefficients.

        Parameters
        ----------
        Xa, Xb : torch.Tensor, shape (B, 1)
            Masses of companions 1 and 2 divided by the total mass.
        kappa2T, kappaA, kappaB : torch.Tensor, shape (B, 1)
            Tidal coupling constants.
        PN : torch.Tensor, shape (B, 10)
            7.5PN coefficients from ``_PN_coeffs``.

        Returns
        -------
        dict of torch.Tensor, shape (B, 1)
        """
        q = Xa / Xb

        # Coefficients for the effective enhancement factor:
        s10, s11, s12 = 1.273000423,    3.64169971e-3,  1.76144380e-3
        s20, s21, s22 = 2.78793291e+1,  1.18175396e-2, -5.39996790e-3
        s30, s31, s32 = 1.42449682e-1, -1.70505852e-5,  3.38040594e-5

        s1 = s10 + s11*kappa2T + s12*q*kappa2T
        s2 = s20 + s21*kappa2T + s22*q*kappa2T
        s3 = s30 + s31*kappa2T + s32*q*kappa2T

        s2s3    = s2 * s3
        exps2s3 = torch.cosh(s2s3) + torch.sinh(s2s3)

        # Exponent parameters:
        alpha, beta = -8.08155404e-3, -1.13695919e+0

        kappaA_alp = (kappaA + 1.0).pow(alpha)
        kappaB_alp = (kappaB + 1.0).pow(alpha)
        Xa_bet     = Xa.pow(beta)
        Xb_bet     = Xb.pow(beta)

        # Pade approximant coefficients:
        n_5o20, n_5o21, n_5o22, n_5o23 = -9.40654388e+2,  6.26517157e+2,  5.53629706e+2,  8.84823087e+1
        n_30,   n_31,   n_32,   n_33   =  4.05483848e+2, -4.25525054e+2, -1.92004957e+2, -5.10967553e+1
        d_10,   d_11,   d_12           =  3.80343306e+0, -2.52026996e+1, -3.08054443e+0

        n_5o2A = n_5o20 + n_5o21*Xa + n_5o22*kappaA_alp + n_5o23*Xa_bet
        n_3A   = n_30   + n_31*Xa   + n_32*kappaA_alp   + n_33*Xa_bet
        d_1A   = d_10   + d_11*Xa   + d_12*Xa_bet

        n_5o2B = n_5o20 + n_5o21*Xb + n_5o22*kappaB_alp + n_5o23*Xb_bet
        n_3B   = n_30   + n_31*Xb   + n_32*kappaB_alp   + n_33*Xb_bet
        d_1B   = d_10   + d_11*Xb   + d_12*Xb_bet

        # 7.5PN Coefficients
        c_1A   = PN[:, 1:2];  c_3o2A = PN[:, 2:3]
        c_2A   = PN[:, 3:4];  c_5o2A = PN[:, 4:5]
        c_1B   = PN[:, 6:7];  c_3o2B = PN[:, 7:8]
        c_2B   = PN[:, 8:9];  c_5o2B = PN[:, 9:10]

        # Pade Coefficients constrained with PN
        inv_c1_A = 1.0 / c_1A
        n_1A     = c_1A + d_1A
        n_3o2A   = (c_1A*c_3o2A - c_5o2A - c_3o2A*d_1A + n_5o2A) * inv_c1_A
        n_2A     = c_2A + c_1A*d_1A
        d_3o2A   = -(c_5o2A + c_3o2A*d_1A - n_5o2A) * inv_c1_A

        inv_c1_B = 1.0 / c_1B
        n_1B     = c_1B + d_1B
        n_3o2B   = (c_1B*c_3o2B - c_5o2B - c_3o2B*d_1B + n_5o2B) * inv_c1_B
        n_2B     = c_2B + c_1B*d_1B
        d_3o2B   = -(c_5o2B + c_3o2B*d_1B - n_5o2B) * inv_c1_B

        return dict(
            s1=s1, s2=s2, s3=s3, exps2s3=exps2s3,
            kappaA=kappaA,  kappaB=kappaB,
            n_5o2A=n_5o2A,  n_3A=n_3A,   d_1A=d_1A,
            n_5o2B=n_5o2B,  n_3B=n_3B,   d_1B=d_1B,
            n_1A=n_1A,  n_3o2A=n_3o2A,  n_2A=n_2A,  d_3o2A=d_3o2A,
            n_1B=n_1B,  n_3o2B=n_3o2B,  n_2B=n_2B,  d_3o2B=d_3o2B,
        )

    # Source: XLALSimNRTunedTidesMergerFrequency_v3
    # (LALSimNRTunedTides.c).
    @staticmethod
    def _merger_frequency_v3(mtot, lambda1, lambda2, q, chi1L, chi2L):
        """
        Merger frequency of a BNS system for NRTidalv3.

        Parameters
        ----------
        mtot : torch.Tensor, shape (B, 1)
            Total mass in solar masses.
        lambda1, lambda2 : torch.Tensor, shape (B, 1)
            Dimensionless tidal deformabilities.
        q : torch.Tensor, shape (B, 1)
            Mass ratio, q >= 1.
        chi1L, chi2L : torch.Tensor, shape (B, 1)
            Aligned spin components.

        Returns
        -------
        torch.Tensor, shape (B, 1)
            Merger frequency in Hz.
        """
        Xa   = q / (1.0 + q)
        Xb   = 1.0 - Xa
        nu   = Xa * Xb
        Xa2  = Xa*Xa;  Xa3 = Xa2*Xa
        Xb2  = Xb*Xb;  Xb3 = Xb2*Xb

        kappa2eff  = 3.0*nu*(Xa3*lambda1 + Xb3*lambda2)
        kappa2eff2 = kappa2eff*kappa2eff

        a_0 = 0.22
        a_1M = 0.80
        a_1S, b_1S = 0.25, -1.99
        a_1T, a_2T = 0.0485, 5.86e-6
        a_3T, a_4T = 0.10,   1.86e-4
        b_1T, b_2T = 1.80,   599.99
        b_3T, b_4T = 7.80,   84.76

        Xval = 1.0 - 4.0*nu

        p_1S = a_1S * (1.0 + b_1S*Xval)
        p_1T = a_1T * (1.0 + b_1T*Xval)
        p_2T = a_2T * (1.0 + b_2T*Xval)
        p_3T = a_3T * (1.0 + b_3T*Xval)
        p_4T = a_4T * (1.0 + b_4T*Xval)

        Sval = Xa2*chi1L + Xb2*chi2L

        QM  = 1.0 + a_1M*Xval
        QS  = 1.0 + p_1S*Sval
        QT  = (1.0 + p_1T*kappa2eff + p_2T*kappa2eff2) / \
              (1.0 + p_3T*kappa2eff + p_4T*kappa2eff2)

        Qfit = a_0 * QM * QS * QT
        Momega_merger = nu * Qfit * (2.0 * math.pi)
        return Momega_merger / (2.0 * math.pi) / (mtot * GM)

    # Source: SimNRTunedTidesFDTidalPhase_v3 (LALSimNRTunedTides.c),
    # IMRPhenomX_TidalPhase (LALSimIMRPhenomX_internals.c).
    @staticmethod
    def _phi_tidal_nrt(Mf, PN, tc):
        """
        NRTidalv3 tidal phase, without the minimum search and PN blend.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F) or (B, 1)
            Geometric frequencies.
        PN : torch.Tensor, shape (B, 10)
            7.5PN coefficients from ``_PN_coeffs``.
        tc : dict of torch.Tensor, shape (B, 1)
            Coefficients from ``_nrtv3_coeffs``.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        s1, s2     = tc['s1'], tc['s2']
        exps2s3    = tc['exps2s3']
        kappaA, kappaB = tc['kappaA'], tc['kappaB']

        n_1A,  n_3o2A, n_2A,  n_5o2A, n_3A,  d_1A,  d_3o2A = (
            tc['n_1A'], tc['n_3o2A'], tc['n_2A'],
            tc['n_5o2A'], tc['n_3A'], tc['d_1A'], tc['d_3o2A'])
        n_1B,  n_3o2B, n_2B,  n_5o2B, n_3B,  d_1B,  d_3o2B = (
            tc['n_1B'], tc['n_3o2B'], tc['n_2B'],
            tc['n_5o2B'], tc['n_3B'], tc['d_1B'], tc['d_3o2B'])

        c_NewtA = PN[:, 0:1]
        c_NewtB = PN[:, 5:6]

        s2Mf    = -2.0 * s2 * math.pi * Mf
        exps2Mf = torch.cosh(s2Mf) + torch.sinh(s2Mf)

        # expression for the effective love number enhancement factor
        inv_1p_exps2s3 = 1.0 / (1.0 + exps2s3)
        dynk2bar = (
            1.0
            + (s1 - 1.0) / (1.0 + exps2Mf * exps2s3)
            - (s1 - 1.0) * inv_1p_exps2s3
            - 2.0*math.pi*Mf * (s1 - 1.0) * s2 * exps2s3
            * inv_1p_exps2s3 * inv_1p_exps2s3
        )

        dynkappaA = kappaA * dynk2bar
        dynkappaB = kappaB * dynk2bar

        piMf   = math.pi * Mf
        piMf23 = piMf.pow(2.0/3.0)
        piMf1  = piMf
        piMf43 = piMf.pow(4.0/3.0)
        piMf53 = piMf.pow(5.0/3.0)
        piMf2  = piMf.pow(2.0)

        # Pade approximant
        numA = (1.0 + n_1A*piMf23 + n_3o2A*piMf1 + n_2A*piMf43
                + n_5o2A*piMf53 + n_3A*piMf2)
        denA = 1.0 + d_1A*piMf23 + d_3o2A*piMf1

        numB = (1.0 + n_1B*piMf23 + n_3o2B*piMf1 + n_2B*piMf43
                + n_5o2B*piMf53 + n_3B*piMf2)
        denB = 1.0 + d_1B*piMf23 + d_3o2B*piMf1

        phi_A = -c_NewtA * dynkappaA * piMf53 * (numA / denA)
        phi_B = -c_NewtB * dynkappaB * piMf53 * (numB / denB)
        return phi_A + phi_B

    # Source: IMRPhenomX_TidalPhaseDerivative
    # (LALSimIMRPhenomX_internals.c).
    @staticmethod
    def _dphi_tidal_nrt(Mf, PN, tc):
        """
        Derivative of the NRTidalv3 tidal phase with respect to Mf.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, 1)
            Geometric frequency.
        PN : torch.Tensor, shape (B, 10)
            7.5PN coefficients from ``_PN_coeffs``.
        tc : dict of torch.Tensor, shape (B, 1)
            Coefficients from ``_nrtv3_coeffs``.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        s1, s2, s3 = tc['s1'], tc['s2'], tc['s3']
        exps2s3 = tc['exps2s3']
        pi = math.pi
        piMf = pi * Mf
        x23, x43, x53 = piMf.pow(2.0/3.0), piMf.pow(4.0/3.0), piMf.pow(5.0/3.0)
        x2, x73, x83, x3 = piMf.pow(2.0), piMf.pow(7.0/3.0), piMf.pow(8.0/3.0), piMf.pow(3.0)

        s2Mf = -s2 * Mf * pi * 2.0
        exps2Mf = torch.cosh(s2Mf) + torch.sinh(s2Mf)
        dynk2bar = (1.0 + (s1 - 1.0) * (1.0 / (1.0 + exps2Mf * exps2s3))
                    - (s1 - 1.0) / (1.0 + exps2s3)
                    - 2.0 * piMf * (s1 - 1.0) * s2 * exps2s3 / ((1.0 + exps2s3) * (1.0 + exps2s3)))
        s2s3b, s2Mfb = -s2 * s3, s2 * Mf * pi * 2.0
        exps2s3b = torch.cosh(s2s3b) + torch.sinh(s2s3b)
        exps2Mfb = torch.cosh(s2Mfb) + torch.sinh(s2Mfb)
        dynk2bar_deriv = ((s1 - 1.0) * (2.0 * pi) * s2 * exps2Mfb * exps2s3b
                          * (1.0 / ((1.0 + exps2Mfb * exps2s3b) * (1.0 + exps2Mfb * exps2s3b)))
                          - 2.0 * pi * (s1 - 1.0) * s2 * exps2s3 / ((1.0 + exps2s3) * (1.0 + exps2s3)))

        def star(c_newt, kappa, n_1, n_3o2, n_2, n_5o2, n_3, d_1, d_3o2):
            num = 1.0 + n_1 * x23 + n_3o2 * piMf + n_2 * x43 + n_5o2 * x53 + n_3 * x2
            den = 1.0 + d_1 * x23 + d_3o2 * piMf
            dphase1 = -(c_newt * kappa * dynk2bar_deriv * x53 * num) / den
            dphase2 = (c_newt * kappa * dynk2bar * x23 * pi * (
                -5.0 - x23 * (3.0 * d_1 + 7.0 * n_1) - 2.0 * piMf * (d_3o2 + 4.0 * n_3o2)
                - x43 * (5.0 * d_1 * n_1 + 9.0 * n_2)
                - 2.0 * x53 * (2.0 * d_3o2 * n_1 + 3.0 * d_1 * n_3o2 + 5.0 * n_5o2)
                - x2 * (7.0 * d_1 * n_2 + 11.0 * n_3 + 5.0 * d_3o2 * n_3o2)
                - 2.0 * x73 * (3.0 * d_3o2 * n_2 + 4.0 * d_1 * n_5o2)
                - x83 * (9.0 * d_1 * n_3 + 7.0 * d_3o2 * n_5o2)
                - 2.0 * x3 * (4.0 * d_3o2 * n_3))) / (3.0 * den * den)
            return dphase1 + dphase2

        return (star(PN[:, 0:1], tc['kappaA'], tc['n_1A'], tc['n_3o2A'], tc['n_2A'],
                     tc['n_5o2A'], tc['n_3A'], tc['d_1A'], tc['d_3o2A'])
                + star(PN[:, 5:6], tc['kappaB'], tc['n_1B'], tc['n_3o2B'], tc['n_2B'],
                       tc['n_5o2B'], tc['n_3B'], tc['d_1B'], tc['d_3o2B']))

    # Source: SimNRTunedTidesFDTidalPhase_PN (LALSimNRTunedTides.c).
    @staticmethod
    def _phi_tidal_PN(Mf, PN, kappaA, kappaB):
        """
        PN tidal phase correction, at 7.5PN.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F) or (B, 1)
            Geometric frequencies.
        PN : torch.Tensor, shape (B, 10)
            7.5PN coefficients from ``_PN_coeffs``.
        kappaA, kappaB : torch.Tensor, shape (B, 1)
            Tidal coupling constants of the two stars.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        c_NewtA = PN[:, 0:1];  c_1A = PN[:, 1:2]
        c_3o2A  = PN[:, 2:3];  c_2A = PN[:, 3:4];  c_5o2A = PN[:, 4:5]
        c_NewtB = PN[:, 5:6];  c_1B = PN[:, 6:7]
        c_3o2B  = PN[:, 7:8];  c_2B = PN[:, 8:9];  c_5o2B = PN[:, 9:10]

        piMf   = math.pi * Mf
        piMf23 = piMf.pow(2.0/3.0)
        piMf43 = piMf.pow(4.0/3.0)
        piMf53 = piMf.pow(5.0/3.0)

        phi_PNA = -c_NewtA * kappaA * piMf53 * (
            1.0 + c_1A*piMf23 + c_3o2A*piMf + c_2A*piMf43 + c_5o2A*piMf53)
        phi_PNB = -c_NewtB * kappaB * piMf53 * (
            1.0 + c_1B*piMf23 + c_3o2B*piMf + c_2B*piMf43 + c_5o2B*piMf53)
        return phi_PNA + phi_PNB

    # Source: PlanckTaper (LALSimNRTunedTides.c).
    @staticmethod
    def _planck_taper(f, f1, f2, exp=torch.exp):
        """
        Planck taper window: 0 at f <= f1, 1 at f >= f2.

        Parameters
        ----------
        f : torch.Tensor, shape (B, F) or (B, 1)
            Frequencies.
        f1, f2 : torch.Tensor, shape (B, 1)
            Start and end of the taper.
        exp : callable
            The exponential as the LAL caller writes it: exp in
            PlanckTaper, cosh + sinh in IMRPhenomX_TidalPhase.

        Returns
        -------
        torch.Tensor, same shape as f
        """
        mid    = (f > f1) & (f < f2)
        f_safe = torch.where(mid, f, 0.5*(f1 + f2))
        arg    = (f2 - f1)/(f_safe - f1) + (f2 - f1)/(f_safe - f2)
        exp_fn = exp(arg)
        taper_mid = 1.0 / (exp_fn + 1.0)

        return torch.where(f <= f1, torch.zeros_like(f),
               torch.where(f >= f2, torch.ones_like(f), taper_mid))

    # Source: XLALSimNRTunedTidesFDTidalPhaseFrequencySeries
    # (LALSimNRTunedTides.c).
    def _tidal_phase_full(self, f, Mf, PN, tc, f_merger):
        """
        NRTidalv3 tidal phase with LAL's minimum search and PN blend.

        Parameters
        ----------
        f : torch.Tensor, shape (B, F)
            Frequencies in Hz.
        Mf : torch.Tensor, shape (B, F)
            Geometric frequencies.
        PN : torch.Tensor, shape (B, 10)
            7.5PN coefficients from ``_PN_coeffs``.
        tc : dict of torch.Tensor, shape (B, 1)
            Coefficients from ``_nrtv3_coeffs``.
        f_merger : torch.Tensor, shape (B, 1)
            Merger frequency in Hz.

        Returns
        -------
        torch.Tensor, shape (B, F)
        """
        kappaA = tc['kappaA']
        kappaB = tc['kappaB']

        phi_raw = self._phi_tidal_nrt(Mf, PN, tc)

        # start checking of minimum;
        check_region = (f >= 0.9 * f_merger)

        trigger = (phi_raw[:, 1:] >= phi_raw[:, :-1]) & check_region[:, 1:]

        freeze_flag_diff = (trigger.to(phi_raw.dtype).cumsum(dim=-1) > 0)

        freeze_flag = torch.zeros_like(check_region)
        freeze_flag[:, 1:] = freeze_flag_diff

        trigger_any = trigger.any(dim=-1)
        trigger_idx = trigger.long().argmax(dim=-1)

        freeze_vals = phi_raw.gather(
            -1, trigger_idx.unsqueeze(-1))

        freeze_vals = torch.where(
            trigger_any.unsqueeze(-1), freeze_vals,
            phi_raw[:, -1:])

        # If a minimum is found, the tidal phase will be constant at
        # that minimum value
        phi_clipped = torch.where(freeze_flag, freeze_vals, phi_raw)

        # We employ here the smooth connection between NRTidal and PN
        # post-merger
        planck = self._planck_taper(f, 1.15 * f_merger, 1.35 * f_merger)
        phi_PN = self._phi_tidal_PN(Mf, PN, kappaA, kappaB)

        return phi_clipped * (1.0 - planck) + phi_PN * planck

    # Source: SimNRTunedTidesFDTidalAmplitude (LALSimNRTunedTides.c).
    @staticmethod
    def _tidal_amp(Mf, kappa2T):
        """
        Tidal amplitude corrections.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F)
            Geometric frequencies.
        kappa2T : torch.Tensor, shape (B, 1)
            Tidal coupling constant.

        Returns
        -------
        torch.Tensor, shape (B, F)
        """
        x    = (math.pi * Mf).pow(2.0/3.0)
        n1   = 4.157407407407407
        n289 = 2519.111111111111
        d    = 13477.8073677
        poly = (1.0 + n1*x + n289*x.pow(2.89)) / (1.0 + d*x.pow(4.0))
        return -9.0 * kappa2T * x.pow(3.25) * poly

    # Source: XLALSimInspiralEOSQfromLambda (LALSimInspiralEOS.c).
    @staticmethod
    def _quadparam_from_lambda(lam):
        """
        Quadrupole-monopole parameter calculated from love number.

        Parameters
        ----------
        lam : torch.Tensor, shape (B, 1)
            Dimensionless tidal deformability.

        Returns
        -------
        torch.Tensor, shape (B, 1)
            Quadrupole parameter; 1.0 (BH value) for lam < 0.5.
        """
        loglam = torch.log(lam)
        q = 0.194 + 0.0936*loglam + 0.0474*loglam*loglam
        q = q - 0.00421*loglam*loglam*loglam
        q = q + 0.000123*loglam*loglam*loglam*loglam
        # Love number is (nearly) zero; cannot compute QM parameter.
        # Setting to 1.0 (BH value).
        return torch.where(lam < 0.5, torch.ones_like(lam), torch.exp(q))

    # Source:
    # XLALSimUniversalRelationSpinInducedOctupoleVSSpinInducedQuadrupole
    # (LALSimUniversalRelations.c), XLALSimInspiralGetHOSpinTerms
    # (LALSimNRTunedTides.c).
    @staticmethod
    def _octparam_from_quadparam(quadparam):
        """
        Spin-induced octupole parameter from the quadrupole parameter.

        Parameters
        ----------
        quadparam : torch.Tensor, shape (B, 1)
            Spin-induced quadrupole moment parameter.

        Returns
        -------
        torch.Tensor, shape (B, 1)
            Octupole parameter with 1 removed to account for the BBH
            baseline.
        """
        lnx = torch.log(quadparam)
        lny = (0.003131 + 2.071*lnx + (-0.7152)*lnx*lnx + 0.2458*lnx*lnx*lnx
               + (-0.03309)*lnx*lnx*lnx*lnx)
        return torch.exp(lny) - 1.0

    # Source: IMRPhenomXGetTidalPhaseCoefficients and
    # IMRPhenomX_TidalPhase (LALSimIMRPhenomX_internals.c),
    # XLALSimInspiralGetHOSpinTerms (LALSimNRTunedTides.c).
    @staticmethod
    def _spin_tidal_pn_phase(Mf, Xa, Xb, chi1L, chi2L,
                             quadparam1, quadparam2,
                             octparam1, octparam2):
        """
        Spin-induced quadrupole and octupole tidal phase, 2PN to 3.5PN.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F) or (B, 1)
            Geometric frequencies.
        Xa, Xb : torch.Tensor, shape (B, 1)
            Masses of companions 1 and 2 divided by the total mass.
        chi1L, chi2L : torch.Tensor, shape (B, 1)
            Aligned spin components.
        quadparam1, quadparam2 : torch.Tensor, shape (B, 1)
            Spin-induced quadrupole moment parameters.
        octparam1, octparam2 : torch.Tensor, shape (B, 1)
            Spin-induced octupole parameters minus 1.

        Returns
        -------
        torch.Tensor, same shape as Mf
        """
        pi = math.pi

        Xa2 = Xa * Xa
        Xb2 = Xb * Xb
        chi1sq = chi1L * chi1L
        chi2sq = chi2L * chi2L
        dqp1 = quadparam1 - 1.0
        dqp2 = quadparam2 - 1.0

        pfaN = 3.0 / (128.0 * Xa * Xb)

        # 2PN terms
        c2pn = -50.0 * (Xa2 * dqp1 * chi1sq + Xb2 * dqp2 * chi2sq)

        # 3PN terms
        coeff6A = (4703.5/8.4 + (2935.0/6.0)*Xa - 120.0*Xa2) * Xa2
        coeff6B = (4703.5/8.4 + (2935.0/6.0)*Xb - 120.0*Xb2) * Xb2
        c3pn = coeff6A * dqp1 * chi1sq + coeff6B * dqp2 * chi2sq

        # Get the PN SS-tail and SSS terms
        SS_3p5 = (-400.0*pi * (dqp1*chi1sq*Xa2 + dqp2*chi2sq*Xb2))
        SSS_3p5 = (
            10.0 * ((Xa2 + (308.0/3.0)*Xa)*chi1L + (Xb2 - (89.0/3.0)*Xb)*chi2L)
                 * dqp1 * Xa2 * chi1sq
            + 10.0 * ((Xb2 + (308.0/3.0)*Xb)*chi2L + (Xa2 - (89.0/3.0)*Xa)*chi1L)
                   * dqp2 * Xb2 * chi2sq
            - 440.0 * octparam1 * Xa * Xa2 * chi1sq * chi1L
            - 440.0 * octparam2 * Xb * Xb2 * chi2sq * chi2L
        )
        c3p5pn = SS_3p5 + SSS_3p5

        piMf = pi * Mf
        return pfaN * (
            c2pn  * piMf.pow(-1.0/3.0)
            + c3pn  * piMf.pow( 1.0/3.0)
            + c3p5pn * piMf.pow( 2.0/3.0)
        )

    # Source: IMRPhenomX_TidalPhaseDerivative
    # (LALSimIMRPhenomX_internals.c).
    @staticmethod
    def _spin_tidal_pn_dphase(Mf, Xa, Xb, chi1L, chi2L,
                               quadparam1, quadparam2,
                               octparam1, octparam2):
        """
        Derivative of the spin-induced tidal phase with respect to Mf.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, 1)
            Geometric frequency.
        Xa, Xb : torch.Tensor, shape (B, 1)
            Masses of companions 1 and 2 divided by the total mass.
        chi1L, chi2L : torch.Tensor, shape (B, 1)
            Aligned spin components.
        quadparam1, quadparam2 : torch.Tensor, shape (B, 1)
            Spin-induced quadrupole moment parameters.
        octparam1, octparam2 : torch.Tensor, shape (B, 1)
            Spin-induced octupole parameters minus 1.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        pi = math.pi

        Xa2 = Xa * Xa
        Xb2 = Xb * Xb
        chi1sq = chi1L * chi1L
        chi2sq = chi2L * chi2L
        dqp1 = quadparam1 - 1.0
        dqp2 = quadparam2 - 1.0

        pfaN = 3.0 / (128.0 * Xa * Xb)

        c2pn = -50.0 * (Xa2 * dqp1 * chi1sq + Xb2 * dqp2 * chi2sq)

        coeff6A = (4703.5/8.4 + (2935.0/6.0)*Xa - 120.0*Xa2) * Xa2
        coeff6B = (4703.5/8.4 + (2935.0/6.0)*Xb - 120.0*Xb2) * Xb2
        c3pn = coeff6A * dqp1 * chi1sq + coeff6B * dqp2 * chi2sq

        SS_3p5 = (-400.0*pi * (dqp1*chi1sq*Xa2 + dqp2*chi2sq*Xb2))
        SSS_3p5 = (
            10.0 * ((Xa2 + (308.0/3.0)*Xa)*chi1L + (Xb2 - (89.0/3.0)*Xb)*chi2L)
                 * dqp1 * Xa2 * chi1sq
            + 10.0 * ((Xb2 + (308.0/3.0)*Xb)*chi2L + (Xa2 - (89.0/3.0)*Xa)*chi1L)
                   * dqp2 * Xb2 * chi2sq
            - 440.0 * octparam1 * Xa * Xa2 * chi1sq * chi1L
            - 440.0 * octparam2 * Xb * Xb2 * chi2sq * chi2L
        )
        c3p5pn = SS_3p5 + SSS_3p5

        piMf = pi * Mf
        three_pn_dphi = pfaN * (-c2pn + c3pn * piMf.pow(2.0/3.0)) / (3.0 * piMf.pow(1.0/3.0) * Mf)
        c3p5_dphi = pfaN * c3p5pn * (2.0/3.0) * piMf.pow(2.0/3.0) / Mf

        return three_pn_dphi + c3p5_dphi

    # Source: XLALSimIMRPhenomXHM (LALSimIMRPhenomXHM.c),
    # IMRPhenomXSetWaveformVariables (LALSimIMRPhenomX_internals.c).
    @classmethod
    def check_parameters(cls, theta):
        """
        Raise LAL's error for parameters it rejects.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 10)
            Tidal aligned layout.

        Raises
        ------
        ValueError
            With LAL's message, for the first failing row.
        """
        m1, m2, chi1, chi2, distance = (theta[:, k] for k in range(5))
        lambda1, lambda2 = theta[:, 8], theta[:, 9]
        m1_SI, m2_SI = m1 * MSUN, m2 * MSUN
        mass_ratio = torch.where(m1_SI > m2_SI, m1_SI / m2_SI, m2_SI / m1_SI)
        cls.raise_first([
            (~(distance > 0), lambda i: "distance must be positive."),
            ((mass_ratio > 1000.0) & (torch.abs(mass_ratio - 1000.0) > 1e-12),
             lambda i: "ERROR: Model not valid at mass ratios beyond 1000."),
            (m1_SI <= 0.0, lambda i: "m1 must be positive."),
            (m2_SI <= 0.0, lambda i: "m2 must be positive."),
            ((lambda1 < 0) | (lambda2 < 0),
             lambda i: f"lambda1 = {lambda1[i]:f}, lambda2 = {lambda2[i]:f}. "
                       "Both should be greater than zero for NRTidalv2 and NRTidalv3"),
            ((chi1 > 1.0) | (chi1 < -1.0) | (chi2 > 1.0) | (chi2 < -1.0),
             lambda i: "Unphysical spins requested: must obey the Kerr bound [-1,1]."),
        ])

    def get_hphc(self, theta, reproduce_lal=False):
        """
        FD plus and cross polarisations of a BNS parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 17)
            Parameters in the waveform_utils.WAVEFORM_PARAMETERS layout.
        reproduce_lal : bool
            If True, skip FD tapering, tc shift and df normalisation so
            the output compares directly with LAL's.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_frequency_bins), complex128

        Raises
        ------
        ValueError
            If any transverse spin is non-zero, or LAL rejects the
            parameters.
        """
        theta = waveform_utils.aligned_layout(theta, waveform_utils.TIDAL_COLUMNS, self.extras)
        self.check_parameters(theta)
        theta = self.order_bodies(theta)
        m1       = theta[:, 0:1]
        m2       = theta[:, 1:2]
        dist_Mpc = theta[:, 4:5]
        tc_val   = theta[:, 5:6]
        phic     = theta[:, 6:7]
        iota     = theta[:, 7:8]
        lambda1  = theta[:, 8:9]
        lambda2  = theta[:, 9:10]

        derived  = self.compute_derived_parameters(theta)
        # Conversion factor Hz to geometric frequency, total mass in
        # seconds
        M_s   = derived[:, 2:3]
        eta   = derived[:, 3:4]    # Symmetric mass ratio
        delta = derived[:, 4:5]    # PN symmetry coefficient
        STotR = derived[:, 9:10]
        dchi  = derived[:, 10:11]
        chi1L = derived[:, 11:12]
        chi2L = derived[:, 12:13]

        # Source: IMRPhenomXSetWaveformVariables
        # (LALSimIMRPhenomX_internals.c).
        Mtot   = m1 + m2
        Xa     = m1 / Mtot
        Xb     = m2 / Mtot
        # Merger frequency from pWF->q (IMRPhenomXSetWaveformVariables,
        # LALSimIMRPhenomX_internals.c) for the alignment and the tidal
        # phase at fRef, and from the SI masses for the tidal series
        # (XLALSimNRTunedTidesFDTidalPhaseFrequencySeries,
        # LALSimNRTunedTides.c).
        q_wf = torch.where(eta == 0.25, torch.ones_like(m1), m1 / m2)
        m1_tides, m2_tides = (m1 * MSUN) / MSUN, (m2 * MSUN) / MSUN
        f_merger_wf = self._merger_frequency_v3(Mtot, lambda1, lambda2, q_wf, chi1L, chi2L)
        f_merger_tides = self._merger_frequency_v3(
            m1_tides + m2_tides, lambda1, lambda2, m1_tides / m2_tides, chi1L, chi2L)
        unequal = ((q_wf == 1.0) | (m1_tides == m2_tides)) & (lambda1 != lambda2)
        if unequal.any():
            i = int(unequal[:, 0].nonzero()[0])
            warnings.warn(f"m1 == m2 ({float(m1[i]):g}), but lambda1 != lambda2 "
                          f"({float(lambda1[i]):g}, {float(lambda2[i]):g}).")

        f_max_prime, f_last = self.checked_band(derived, [
            # Source: XLALSimNRTunedTidesFDTidalPhaseFrequencySeries
            # (LALSimNRTunedTides.c).
            (int(self.f_min / self.delta_f) * self.delta_f >= 0.9 * f_merger_tides[:, 0],
             lambda i: f"f_min = {self.f_min:g} Hz is too high."),
        ])

        amp_coeffs   = self.get_amp_coeffs(derived)
        phase_coeffs = self.get_phase_coeffs(derived)
        fRING = phase_coeffs['fRING']
        fDAMP = phase_coeffs['fDAMP']

        # Amplitude normalization
        M_m    = M_s * self.C
        dist_m = dist_Mpc * self.Mpc
        amp0   = M_m * M_s / dist_m
        ampNorm = torch.sqrt(2.0*eta/3.0) * (math.pi**(-1.0/6.0))

        kappa2T        = self._kappa2T(Xa, Xb, lambda1, lambda2)
        kappaA, kappaB = self._kappaAB(Xa, Xb, lambda1, lambda2)
        PN             = self._PN_coeffs(Xa)
        tc_d           = self._nrtv3_coeffs(Xa, Xb, kappa2T, kappaA, kappaB, PN)

        # Source: XLALSimInspiralSetQuadMonParamsFromLambdas
        # (LALSimInspiral.c), called in IMRPhenomXSetWaveformVariables
        # (LALSimIMRPhenomX_internals.c).
        quadparam1 = self._quadparam_from_lambda(lambda1)
        quadparam2 = self._quadparam_from_lambda(lambda2)
        # Remove -1 to account for BBH baseline
        octparam1  = self._octparam_from_quadparam(quadparam1)
        octparam2  = self._octparam_from_quadparam(quadparam2)

        # Linear time and phase shifts so that model peaks near t ~ 0
        # (IMRPhenomX_TimeShift_22, LALSimIMRPhenomX_internals.c)
        linb_fit_val = IMRPhenomXAS._linb_fit(eta, STotR, dchi, delta)
        psi4val      = IMRPhenomXAS._psi4tostrain_fit(eta, STotR, dchi)

        frefFit   = fRING - fDAMP
        dphi22Ref = (1.0/eta) * self.dphase(frefFit, phase_coeffs)
        linb_bbh  = linb_fit_val - dphi22Ref - 2.0*math.pi*(500.0 + psi4val)

        # correct for time and phase shifts due to tidal phase
        # (IMRPhenomXASGenerateFD, LALSimIMRPhenomX.c)
        Mf_final = torch.where(f_merger_wf < f_last, f_merger_wf, f_last) * M_s

        dphi_bbh_fmerger = (1.0/eta) * self.dphase(Mf_final, phase_coeffs)

        dphi_tid_fmerger = self._dphi_tidal_nrt(Mf_final, PN, tc_d)

        # Source: IMRPhenomX_TidalPhaseDerivative
        # (LALSimIMRPhenomX_internals.c).
        dphi_tid_fmerger = dphi_tid_fmerger + self._spin_tidal_pn_dphase(
            Mf_final, Xa, Xb, chi1L, chi2L, quadparam1, quadparam2,
            octparam1, octparam2)

        dphi_fmerger = dphi_bbh_fmerger + linb_bbh - dphi_tid_fmerger
        # This was adapted from the PhenomPv2 implementation; the
        # resulting BBH limit can then have a time-shift in its phase
        linb         = linb_bbh - dphi_fmerger

        # extra contribution to phi(fRef) due to tidal corrections
        # (IMRPhenomXASGenerateFD, LALSimIMRPhenomX.c;
        # IMRPhenomX_TidalPhase, LALSimIMRPhenomX_internals.c)
        MfRef    = self.f_ref * M_s
        phi22ref = self.phase(MfRef, phase_coeffs)
        Mfmerger_wf = f_merger_wf * M_s
        planck_ref = self._planck_taper(MfRef, 1.15 * Mfmerger_wf, 1.35 * Mfmerger_wf,
                                        exp=lambda x: torch.cosh(x) + torch.sinh(x))
        phiTfRef = -(self._phi_tidal_nrt(MfRef, PN, tc_d) * (1.0 - planck_ref)
                     + self._phi_tidal_PN(MfRef, PN, kappaA, kappaB) * planck_ref
                     + self._spin_tidal_pn_phase(
                         MfRef, Xa, Xb, chi1L, chi2L,
                         quadparam1, quadparam2, octparam1, octparam2))
        # Calculate phase at reference frequency: phifRef = 2.0*phi0 +
        # LAL_PI_4 + PhenomXPhase(fRef)
        phifRef  = (-((1.0/eta)*phi22ref + phiTfRef + linb*MfRef)
                    + 2.0*phic + 0.25*math.pi)

        Mf = self.f * M_s

        # Scale phase by 1/eta
        phi22   = self.phase(Mf, phase_coeffs, loop=True)
        phi_bbh = phi22*(1.0/eta) + (linb*Mf + phifRef)

        # Add NRTidal phase, if selected, code adapted from
        # LALSimIMRPhenomP.c
        # Add spin-induced quadrupole moment terms to tidal phasing
        phi_tidal = (self._tidal_phase_full(self.f, Mf, PN, tc_d, f_merger_tides)
                     + self._spin_tidal_pn_phase(
                         Mf, Xa, Xb, chi1L, chi2L,
                         quadparam1, quadparam2, octparam1, octparam2))
        phi_total = phi_bbh - phi_tidal

        # Reconstruct waveform with NRTidal terms included:
        # h(f) = [A(f) + A_tidal(f)] * Exp{I [phi(f) - phi_tidal(f)]}
        # * window(f) (IMRPhenomXASGenerateFD, LALSimIMRPhenomX.c)
        A_bbh  = self.amp(Mf, amp_coeffs, loop=True)
        ampT   = self._tidal_amp(Mf, kappa2T)

        coeff_tidal = 2.0 * math.sqrt(1.0/5.0) * math.sqrt(math.pi)
        A_total = amp0 * (ampNorm * Mf.pow(-7.0/6.0) * A_bbh
                          + coeff_tidal * ampT)

        # Prepare tapering of amplitude beyond merger frequency
        amp_win = 1.0 - self._planck_taper(self.f, f_merger_tides, 1.2*f_merger_tides)
        h22 = A_total * torch.polar(torch.ones_like(phi_total), phi_total) * amp_win
        # We do not want to generate the waveform at frequencies > f_max
        # (default = 0.3 Mf)
        h22 = torch.where((self.f <= f_last) & (Mf <= f_max_prime * M_s), h22, torch.zeros_like(h22))

        # Add the hl-m mode to hptilde and hctilde.
        # Equatorial symmetry: add in -m and m mode
        # (IMRPhenomXHM_MultiMode and IMRPhenomXHMFDAddMode,
        # LALSimIMRPhenomXHM.c)
        Ym = IMRPhenomXHM.spin_weighted_Y(iota, math.pi / 2, -2, 2, -2)
        Ystar = torch.conj(IMRPhenomXHM.spin_weighted_Y(iota, math.pi / 2, -2, 2, 2))
        hp = 0.5 * (Ym + Ystar) * h22
        hc = 1j * 0.5 * (Ym - Ystar) * h22

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
            hp, hc = self.apply_tc(hp, hc, tc_val)
            hp = hp * self.df
            hc = hc * self.df

        # Zeros from zero frequency to the starting frequency and zeros
        # beyond the cutoff frequency (XLALSimIMRPhenomXASGenerateFD,
        # LALSimIMRPhenomX.c)
        return self.pad_missing_frequencies(hp, hc)
