#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : IMRPhenomXAS_NRTidalv3.py
Description   : GPU-native batched IMRPhenomXAS_NRTidalv3 aligned-spin BNS/NSBH
                frequency-domain waveform model.

                Implements the NRTidalv3 tidal corrections on top of the
                IMRPhenomXAS BBH backbone (García-Quirós et al. 2020,
                arXiv:2001.10914), following Abac et al. 2023
                (arXiv:2311.07456) and the LALSim C reference implementation
                in LALSimNRTunedTides.c and LALSimIMRPhenomX_internals.c.

                Tidal contributions:
                  • NRTidalv3 tidal phase (per-star Padé approximant with
                    dynamic effective Love number, Eqs. 27-33 of 2311.07456)
                  • Smooth post-merger transition to 7.5PN tidal phase
                    (Planck taper in [1.15, 1.35] × f_merger, Eq. 45)
                  • Minimum-clamping of tidal phase after merger
                  • NRTidalv2 tidal amplitude correction (Eq. 24 of
                    arXiv:1905.06011, same formula reused in NRTidalv3)
                  • Planck-window tapering of amplitude at/beyond f_merger

                Spin-induced quadrupole/octupole moment (SIQM) tidal PN terms
                at 2PN, 3PN, and 3.5PN order in the tidal phase, following
                LALSimIMRPhenomX_internals.c IMRPhenomXGetTidalPhaseCoefficients
                and IMRPhenomX_TidalPhase (lines 2893-3085).  Quadrupole
                parameters are derived from lambda via the universal relation
                XLALSimInspiralEOSQfromLambda (LALSimInspiralEOS.c line 104),
                octupole parameters from XLALSimUniversalRelationSpinInduced-
                OctupoleVSSpinInducedQuadrupole (LALSimUniversalRelations.c).

                Parameters (theta columns)
                --------------------------
                waveform_utils.WAVEFORM_PARAMETERS (PyCBC names): mass1,
                mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                distance, tc, coa_phase, inclination, lambda1, lambda2,
                polarization, ra, dec.

Created on 2026-05-28

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

References
----------
NRTidalv3      : Abac et al. (2023), arXiv:2311.07456
Merger freq fit: Gonzalez et al. (2022), arXiv:2210.16366
7.5PN tidal    : Vines et al. (2011); Henry et al. (2020)
kappa2T        : Dietrich et al. (2017), arXiv:1706.02969
NRTidalv2 amp  : Dietrich et al. (2019), arXiv:1905.06011
BBH backbone   : Garcia-Quiros et al. (2020), arXiv:2001.10914
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
    GPU-native batched IMRPhenomXAS with NRTidalv3 tidal corrections.

    Inherits the full BBH backbone (amplitude, three-region phase, QNM tables,
    connection coefficients, time-alignment fit) from IMRPhenomXAS and adds
    per-star NRTidalv3 tidal phase/amplitude on top.

    When constructed with a ``param_sampler`` this class also acts as a
    signal-sampler ``torch.nn.Module`` (same pattern as ``IMRPhenomPv2``).
    Call ``forward()`` to obtain a batch of detector-frame strain tensors
    and normalised parameter targets ready for network training. Without a
    sampler the instance is pure waveform math on the signal-band grid.

    Parameters
    ----------
    param_sampler : DistributionSampler or None
        BNS parameter sampler (with ``lambda1`` and ``lambda2``).
    waveform_project : ConstantProjection or None
        Multi-detector projection module.
    augment : callable or None
        Optional SNR-rescaling augmentation.
    window : Window
        Window geometry; the grid runs from the low cutoff to Nyquist with
        ``df = 1 / padded_length_in_s``.
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

        # Without a sampler every parameter it does not take is checked.
        self.extras = waveform_utils.TRANSVERSE_SPIN_PARAMETERS
        if self.param_sampler is not None:
            param_index = self.param_sampler.param_index
            parameters = waveform_utils.TIDAL_ALIGNED_PARAMETERS
            self.source, self.target = waveform_utils.layout_columns(param_index, parameters, type(self).__name__)
            self.extras = waveform_utils.given_extras(param_index, parameters)
            # Column of "distance" in the full all_theta tensor — used
            # by the SNR-rescaling augmentation to keep distance consistent.
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

        The output has LAL's NextPow2(f_max/deltaF) + 1 bins
        (LALSimIMRPhenomX.c 286-303), zeros above the grid.

        Parameters
        ----------
        delta_f, low_frequency_cutoff_in_hz, f_max, reference_frequency_in_hz : float
            LAL's deltaF, f_min, f_max and fRef, in Hz.
        batch_size : int
            Signals per call.
        device : str or torch.device
            Device of the batches.

        Returns
        -------
        IMRPhenomXAS_NRTidalv3
            Model with no sampler, projection or augmentation.
        """
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        model.signal_batch_size = batch_size
        n_bins = int(2 ** math.ceil(math.log2(int(f_max / delta_f)))) + 1
        model.set_band(low_frequency_cutoff_in_hz, reference_frequency_in_hz, f_max, delta_f, n_bins, 0.0,
                       batch_size, device, torch.float64)
        model.param_sampler = model.waveform_project = model.augment = None
        model.extras = waveform_utils.TRANSVERSE_SPIN_PARAMETERS
        return model

    def set_band(self, f_min, f_ref, f_max, delta_f, n_bins, padding_length_in_s, batch_size, device, dtype):
        """
        Set LAL's frequency grid, reference frequency and output buffers.

        Parameters
        ----------
        f_min, f_ref, f_max, delta_f : float
            LAL's f_min, fRef, f_max and deltaF, in Hz.
        n_bins : int
            Bins of the output, from DC.
        padding_length_in_s : float
            Window padding, for the training-path tc placement.
        batch_size : int
            Signals per call.
        device : str or torch.device
            Device of the batches.
        dtype : torch.dtype
            Real dtype of the training-path output.

        Raises
        ------
        ValueError
            On LAL's frequency checks.
        """
        # LAL's frequency checks and fRef default (LALSimIMRPhenomX.c 209-239).
        if f_ref < 0.0:
            raise ValueError("fRef_In must be positive or set to 0 to ignore.")
        if delta_f <= 0.0:
            raise ValueError("deltaF must be positive.")
        if f_min <= 0.0:
            raise ValueError("f_min must be positive.")
        if f_max < 0.0:
            raise ValueError("f_max must be non-negative.")
        if f_ref == 0.0:
            f_ref = f_min

        # LAL's grid i * deltaF for i in [(size_t)(f_min/deltaF), (size_t)(f_max/deltaF)]
        # (LALSimIMRPhenomX.c 589-605).
        self.f_min = f_min
        self.f_max = f_max
        self.delta_f = delta_f
        i_start = int(f_min / delta_f)
        i_stop = int(f_max / delta_f) + 1
        f = torch.arange(i_start, i_stop, device=device, dtype=torch.float64) * delta_f
        f = f.unsqueeze(0).expand(batch_size, -1).clone()
        f_ref = waveform_utils.reference_frequencies(f_ref, batch_size, device)
        self._out_cdtype = torch.promote_types(dtype, torch.complex64)

        # Initialise the BBH backbone with the full-resolution grid.
        # IMRPhenomXAS.__init__ sets self.df via the endpoint formula
        # (f[-1]-f[0])/(n-1), which is exact for the production grid —
        # same pattern as IMRPhenomD does for IMRPhenomPv2.
        IMRPhenomXAS.__init__(self, f, f_ref, padding_length_in_s=padding_length_in_s)
        self.B = f.shape[0]
        self.f = f
        self.f_numel = self.f[0].numel()
        # n_pad: bins from 0 Hz to the first grid bin; bins above the grid stay zero.
        self.n_pad = i_start
        self.hp_buffer = torch.empty((self.B, n_bins), dtype=torch.complex128, device=device)
        self.hc_buffer = torch.empty_like(self.hp_buffer)

    @torch.no_grad()
    def forward(self, return_theta=False):
        """
        Sample a batch of BNS waveforms and return detector-frame strain.

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


    # ------------------------------------------------------------------
    # Static tidal helpers — all operate on (B, 1) tensors unless noted
    # ------------------------------------------------------------------

    # ----------------------------------------------------------------
    # 1. Tidal coupling constant  kappa2T
    #    Source: XLALSimNRTunedTidesComputeKappa2T
    #            LALSimNRTunedTides.c lines 133-162
    #    Reference: Eq. 2 of arXiv:1706.02969
    # ----------------------------------------------------------------
    @staticmethod
    def _kappa2T(Xa, Xb, lambda1, lambda2):
        """Effective tidal coupling constant kappa2T.  Returns (B, 1)."""
        term1 = (1.0 + 12.0*Xb/Xa) * Xa.pow(5) * lambda1
        term2 = (1.0 + 12.0*Xa/Xb) * Xb.pow(5) * lambda2
        return (3.0/13.0) * (term1 + term2)

    # ----------------------------------------------------------------
    # 2. Individual tidal coupling constants  kappaA, kappaB
    #    Source: NRTidalv3_coeffs[4:6] in
    #            XLALSimNRTunedTidesSetFDTidalPhase_v3_Coeffs
    #            LALSimNRTunedTides.c lines 504-505
    # ----------------------------------------------------------------
    @staticmethod
    def _kappaAB(Xa, Xb, lambda1, lambda2):
        """
        Individual tidal coupling constants kappaA and kappaB.

        Returns (kappaA, kappaB) each (B, 1).
        """
        kappaA = 3.0 * Xb * Xa.pow(4) * lambda1
        kappaB = 3.0 * Xa * Xb.pow(4) * lambda2
        return kappaA, kappaB

    # ----------------------------------------------------------------
    # 3. 7.5PN tidal phase coefficients  (10 per batch element)
    #    Source: XLALSimNRTunedTidesSetFDTidalPhase_PN_Coeffs
    #            LALSimNRTunedTides.c lines 433-465
    #    Reference: Eq. (45) of arXiv:2311.07456
    #
    #    Layout of returned (B, 10) tensor:
    #      cols 0..4 = [c_NewtA, c_1A, c_3o2A, c_2A, c_5o2A]  (primary)
    #      cols 5..9 = [c_NewtB, c_1B, c_3o2B, c_2B, c_5o2B]  (secondary)
    # ----------------------------------------------------------------
    @staticmethod
    def _PN_coeffs(Xa):
        """
        7.5PN tidal phase coefficients.

        Parameters: Xa (B, 1) — primary mass fraction m1/(m1+m2).
        Returns: PN (B, 10).
        """
        Xb   = 1.0 - Xa
        Xa2  = Xa*Xa;  Xa3 = Xa2*Xa;  Xa4 = Xa3*Xa;  Xa5 = Xa4*Xa
        Xb2  = Xb*Xb;  Xb3 = Xb2*Xb;  Xb4 = Xb3*Xb;  Xb5 = Xb4*Xb

        pi    = math.pi
        den_a = 11.0*Xa - 12.0
        den_b = 11.0*Xb - 12.0

        # --- star A (primary) ---
        c_NewtA = -3.0*den_a / (16.0*Xa*Xb2)
        c_1A    = (-1300.0*Xa3 + 11430.0*Xa2 + 4595.0*Xa - 15895.0) / (672.0*den_a)
        c_3o2A  = torch.full_like(Xa, -pi)
        c_2A    = (22861440.0*Xa5 - 102135600.0*Xa4 + 791891100.0*Xa3
                   + 874828080.0*Xa2 + 216234195.0*Xa
                   - 1939869350.0) / (27433728.0*den_a)
        c_5o2A  = -pi*(10520.0*Xa3 - 7598.0*Xa2 + 22415.0*Xa - 27719.0) / (672.0*den_a)

        # --- star B (secondary) ---
        c_NewtB = -3.0*den_b / (16.0*Xb*Xa2)
        c_1B    = (-1300.0*Xb3 + 11430.0*Xb2 + 4595.0*Xb - 15895.0) / (672.0*den_b)
        c_3o2B  = torch.full_like(Xa, -pi)
        c_2B    = (22861440.0*Xb5 - 102135600.0*Xb4 + 791891100.0*Xb3
                   + 874828080.0*Xb2 + 216234195.0*Xb
                   - 1939869350.0) / (27433728.0*den_b)
        c_5o2B  = -pi*(10520.0*Xb3 - 7598.0*Xb2 + 22415.0*Xb - 27719.0) / (672.0*den_b)

        return torch.cat([c_NewtA, c_1A, c_3o2A, c_2A, c_5o2A,
                          c_NewtB, c_1B, c_3o2B, c_2B, c_5o2B], dim=-1)  # (B, 10)

    # ----------------------------------------------------------------
    # 4. NRTidalv3 Pade and enhancement-factor coefficients
    #    Source: XLALSimNRTunedTidesSetFDTidalPhase_v3_Coeffs
    #            LALSimNRTunedTides.c lines 470-567
    # ----------------------------------------------------------------
    @staticmethod
    def _nrtv3_coeffs(Xa, Xb, kappa2T, kappaA, kappaB, PN):
        """
        NRTidalv3 Pade and dynamic-Love-number coefficients.

        Parameters: Xa, Xb, kappa2T, kappaA, kappaB — all (B, 1);
                    PN (B, 10) from _PN_coeffs.
        Returns: dict of (B, 1) tensors.
        """
        q = Xa / Xb   # mass ratio >= 1

        # --- effective Love-number enhancement parameters ---
        s10, s11, s12 = 1.273000423,    3.64169971e-3,  1.76144380e-3
        s20, s21, s22 = 2.78793291e+1,  1.18175396e-2, -5.39996790e-3
        s30, s31, s32 = 1.42449682e-1, -1.70505852e-5,  3.38040594e-5

        s1 = s10 + s11*kappa2T + s12*q*kappa2T
        s2 = s20 + s21*kappa2T + s22*q*kappa2T
        s3 = s30 + s31*kappa2T + s32*q*kappa2T

        s2s3    = s2 * s3
        exps2s3 = torch.cosh(s2s3) + torch.sinh(s2s3)   # LAL's form of exp(s2*s3)

        # --- exponent parameters for kappa^alpha and X^beta ---
        alpha, beta = -8.08155404e-3, -1.13695919e+0

        kappaA_alp = (kappaA + 1.0).pow(alpha)
        kappaB_alp = (kappaB + 1.0).pow(alpha)
        Xa_bet     = Xa.pow(beta)
        Xb_bet     = Xb.pow(beta)

        # --- raw Pade numerator/denominator fit coefficients ---
        n_5o20, n_5o21, n_5o22, n_5o23 = -9.40654388e+2,  6.26517157e+2,  5.53629706e+2,  8.84823087e+1
        n_30,   n_31,   n_32,   n_33   =  4.05483848e+2, -4.25525054e+2, -1.92004957e+2, -5.10967553e+1
        d_10,   d_11,   d_12           =  3.80343306e+0, -2.52026996e+1, -3.08054443e+0

        n_5o2A = n_5o20 + n_5o21*Xa + n_5o22*kappaA_alp + n_5o23*Xa_bet
        n_3A   = n_30   + n_31*Xa   + n_32*kappaA_alp   + n_33*Xa_bet
        d_1A   = d_10   + d_11*Xa   + d_12*Xa_bet

        n_5o2B = n_5o20 + n_5o21*Xb + n_5o22*kappaB_alp + n_5o23*Xb_bet
        n_3B   = n_30   + n_31*Xb   + n_32*kappaB_alp   + n_33*Xb_bet
        d_1B   = d_10   + d_11*Xb   + d_12*Xb_bet

        # --- PN-constrained Pade coefficients ---
        # Source: LALSimNRTunedTides.c lines 543-565
        c_1A   = PN[:, 1:2];  c_3o2A = PN[:, 2:3]
        c_2A   = PN[:, 3:4];  c_5o2A = PN[:, 4:5]
        c_1B   = PN[:, 6:7];  c_3o2B = PN[:, 7:8]
        c_2B   = PN[:, 8:9];  c_5o2B = PN[:, 9:10]

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

    # ----------------------------------------------------------------
    # 5. BNS merger frequency in Hz
    #    Source: XLALSimNRTunedTidesMergerFrequency_v3
    #            LALSimNRTunedTides.c lines 209-291
    #    Reference: Eq. (23) of arXiv:2210.16366
    # ----------------------------------------------------------------
    @staticmethod
    def _merger_frequency_v3(mtot, lambda1, lambda2, q, chi1L, chi2L):
        """
        BNS merger frequency in Hz, with LAL's arguments and arithmetic.

        Parameters: all (B, 1); mtot in solar masses, q >= 1.
        Returns f_merger (B, 1) in Hz.
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

        # spin combination S = Xa^2*chi1 + Xb^2*chi2
        Sval = Xa2*chi1L + Xb2*chi2L

        QM  = 1.0 + a_1M*Xval
        QS  = 1.0 + p_1S*Sval
        QT  = (1.0 + p_1T*kappa2eff + p_2T*kappa2eff2) / \
              (1.0 + p_3T*kappa2eff + p_4T*kappa2eff2)

        Qfit = a_0 * QM * QS * QT
        Momega_merger = nu * Qfit * (2.0 * math.pi)
        return Momega_merger / (2.0 * math.pi) / (mtot * GM)

    # ----------------------------------------------------------------
    # 6. NRTidalv3 raw tidal phase (no clipping, no Planck taper)
    #    Source: IMRPhenomX_TidalPhase (NRTidalv3 branch)
    #            LALSimIMRPhenomX_internals.c lines 2968-3075
    #    Reference: Eqs. 27-33 of arXiv:2311.07456
    #
    #    All frequency-dependent expressions use M_omega = pi * Mf.
    # ----------------------------------------------------------------
    @staticmethod
    def _phi_tidal_nrt(Mf, PN, tc):
        """
        NRTidalv3 Pade tidal phase at frequency Mf (unclipped, untapered).

        Parameters
        ----------
        Mf : (B, F) or (B, 1)
        PN : (B, 10) from _PN_coeffs
        tc : dict of (B, 1) tensors from _nrtv3_coeffs

        Returns
        -------
        phi_nrt : same shape as Mf  (negative — tidal phase < 0)
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

        # --- Dynamic effective Love-number enhancement (Eq. 27 of 2311.07456) ---
        # s2Mf = -2 * s2 * M_omega = -2 * s2 * pi * Mf
        # LAL's cosh(x) + sinh(x) form of exp(x) (LALSimNRTunedTides.c 588).
        s2Mf    = -2.0 * s2 * math.pi * Mf
        exps2Mf = torch.cosh(s2Mf) + torch.sinh(s2Mf)

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

        # --- Powers of (pi * Mf) ---
        piMf   = math.pi * Mf
        piMf23 = piMf.pow(2.0/3.0)
        piMf1  = piMf
        piMf43 = piMf.pow(4.0/3.0)
        piMf53 = piMf.pow(5.0/3.0)
        piMf2  = piMf.pow(2.0)

        # --- Per-star Pade approximant (Eqs. 30, 32 of 2311.07456) ---
        numA = (1.0 + n_1A*piMf23 + n_3o2A*piMf1 + n_2A*piMf43
                + n_5o2A*piMf53 + n_3A*piMf2)
        denA = 1.0 + d_1A*piMf23 + d_3o2A*piMf1

        numB = (1.0 + n_1B*piMf23 + n_3o2B*piMf1 + n_2B*piMf43
                + n_5o2B*piMf53 + n_3B*piMf2)
        denB = 1.0 + d_1B*piMf23 + d_3o2B*piMf1

        phi_A = -c_NewtA * dynkappaA * piMf53 * (numA / denA)
        phi_B = -c_NewtB * dynkappaB * piMf53 * (numB / denB)
        return phi_A + phi_B

    # ----------------------------------------------------------------
    # 6b. Derivative of the NRTidalv3 Pade tidal phase
    #     Source: IMRPhenomX_TidalPhaseDerivative (NRTidalv3 branch),
    #             LALSimIMRPhenomX_internals.c lines 3141-3227
    #     At Mf <= Mfmerger < 1.15*Mfmerger the Planck taper and its
    #     derivative are zero, so only NRTuned_dphaseNRT survives.
    # ----------------------------------------------------------------
    @staticmethod
    def _dphi_tidal_nrt(Mf, PN, tc):
        """
        d/dMf of the NRTidalv3 Pade tidal phase, LAL's analytic form.

        Parameters
        ----------
        Mf : (B, 1)
        PN : (B, 10) from _PN_coeffs
        tc : dict of (B, 1) tensors from _nrtv3_coeffs

        Returns
        -------
        dphi_nrt : (B, 1)
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

    # ----------------------------------------------------------------
    # 7. 7.5PN tidal phase (for post-merger Planck taper transition)
    #    Source: SimNRTunedTidesFDTidalPhase_PN
    #            LALSimNRTunedTides.c lines 655-698
    # ----------------------------------------------------------------
    @staticmethod
    def _phi_tidal_PN(Mf, PN, kappaA, kappaB):
        """
        7.5PN tidal phase at Mf.

        Parameters: Mf (B, F) or (B, 1); PN (B, 10); kappaA/B (B, 1).
        Returns: phi_PN (same shape as Mf).
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

    # ----------------------------------------------------------------
    # 8. Planck taper window (vectorized)
    #    Source: PlanckTaper in LALSimNRTunedTides.c lines 44-53
    # ----------------------------------------------------------------
    @staticmethod
    def _planck_taper(f, f1, f2, exp=torch.exp):
        """
        Planck taper: 0 at f <= f1, smooth rise, 1 at f >= f2.

        Parameters: f (B, F), f1 / f2 (B, 1); exp, the exponential as the
        LAL caller writes it (PlanckTaper uses exp, IMRPhenomX_TidalPhase
        cosh + sinh).
        Returns window (B, F).
        """
        mid    = (f > f1) & (f < f2)
        f_safe = torch.where(mid, f, 0.5*(f1 + f2))   # avoid /0 outside mid
        arg    = (f2 - f1)/(f_safe - f1) + (f2 - f1)/(f_safe - f2)
        exp_fn = exp(arg)
        taper_mid = 1.0 / (exp_fn + 1.0)

        return torch.where(f <= f1, torch.zeros_like(f),
               torch.where(f >= f2, torch.ones_like(f), taper_mid))

    # ----------------------------------------------------------------
    # 9. Full tidal phase: minimum-clamping + Planck taper
    #    Source: XLALSimNRTunedTidesFDTidalPhaseFrequencySeries
    #            (NRTidalv3_V branch), LALSimNRTunedTides.c lines 827-853
    #
    #    LAL procedure (exactly replicated here):
    #      a) Evaluate raw NRTidalv3 Pade phase at every frequency.
    #      b) From 0.9 * fmerger onward, find the FIRST frequency at which
    #         phi_raw[i] >= phi_raw[i-1] (i.e. the first non-decrease).
    #         Freeze phi at phi_raw[indexmin] = phi_raw[i-1] for all bins
    #         from that index onward.
    #      c) Smoothly transition to 7.5PN tidal phase in the post-merger
    #         Planck window [1.15 * fmerger, 1.35 * fmerger].
    #
    #    NOTE: torch.cummin is NOT used because it picks up the globally
    #    minimum value, which can be a post-Pade-singularity artefact
    #    (phi_raw → ±∞ at a Pade pole for heavy BNS with asymmetric
    #    Lambda).  LAL's sequential first-increase check fires before the
    #    singularity, correctly freezing at the last decreasing value.
    # ----------------------------------------------------------------
    def _tidal_phase_full(self, f, Mf, PN, tc, f_merger):
        """
        NRTidalv3 tidal phase with minimum-clamping and Planck taper.

        Parameters
        ----------
        f        : (B, F) — grid in Hz, where LAL places its boundaries
        Mf       : (B, F)
        PN       : (B, 10)
        tc       : dict of (B, 1) tensors
        f_merger : (B, 1) — merger frequency in Hz

        Returns
        -------
        phi_tidal : (B, F)  (negative; tidal phase is a negative correction)
        """
        kappaA = tc['kappaA']
        kappaB = tc['kappaB']

        # (a) Raw NRTidalv3 Pade phase
        phi_raw = self._phi_tidal_nrt(Mf, PN, tc)             # (B, F)

        # (b) LAL-equivalent minimum-clamping.
        #
        #     LAL iterates bins sequentially and breaks at the FIRST bin i
        #     (with f[i] >= 0.9*fmerger) where phi[i] >= phi[i-1], then
        #     freezes at phi[i-1] = phi[indexmin].  This naturally stops
        #     before any Pade singularity (where the raw phase shoots to
        #     +/-inf) because the phase starts increasing toward the pole.
        #
        #     Vectorised equivalent:
        #       1. Compare phi_raw[k+1] >= phi_raw[k], as LAL does.
        #       2. Within the check region (f >= 0.9*fmerger), mark the FIRST
        #          non-decrease.
        #       3. "Freeze" flag: True for every bin at or after that first
        #          non-decrease.
        #       4. Freeze value = phi_raw[first_non_decrease_index] (the last
        #          decreasing value, i.e. the minimum before the singularity).
        #
        check_region = (f >= 0.9 * f_merger)                  # (B, F)  bool

        # A "triggering" step: phi_raw[k+1] >= phi_raw[k] inside the check region.
        # check_region[:, 1:] aligns with the step from k → k+1.
        trigger = (phi_raw[:, 1:] >= phi_raw[:, :-1]) & check_region[:, 1:]  # (B, F-1) bool

        # Once triggered, the freeze flag stays True for all later bins.
        # cumsum > 0 means "trigger has been seen at or before this step".
        freeze_flag_diff = (trigger.to(phi_raw.dtype).cumsum(dim=-1) > 0)  # (B, F-1)

        # Extend to full-F shape: bin 0 is never frozen; freeze_flag[k] = flag for k.
        freeze_flag = torch.zeros_like(check_region)           # (B, F)
        freeze_flag[:, 1:] = freeze_flag_diff

        # Freeze value: phi_raw at the first non-decrease index.
        # For each batch element, find the first True in trigger within
        # the check region; the freeze value = phi_raw at that index
        # (= last decreasing value before the singularity).
        #
        # If no trigger is found, freeze_idx stays at the last valid bin
        # (the running minimum equals phi_raw throughout, so no clamping
        # is applied — freeze_flag remains False everywhere).
        trigger_any = trigger.any(dim=-1)                      # (B,) bool
        # argmax on the int cast returns the first True position.
        trigger_idx = (trigger.long().argmax(dim=-1)           # (B,) — diff index
                       .clamp(max=phi_raw.shape[-1] - 2))

        freeze_vals = phi_raw.gather(                          # (B, 1)
            -1, trigger_idx.unsqueeze(-1))

        # Where no trigger was found keep phi_raw (no clamping needed).
        freeze_vals = torch.where(
            trigger_any.unsqueeze(-1), freeze_vals,
            phi_raw[:, -1:])                                   # (B, 1)

        # Apply clamping: frozen bins use freeze_vals, others use phi_raw.
        phi_clipped = torch.where(freeze_flag, freeze_vals, phi_raw)  # (B, F)

        # (c) Smooth post-merger transition to 7.5PN tidal phase
        planck = self._planck_taper(f, 1.15 * f_merger, 1.35 * f_merger)  # (B, F)
        phi_PN = self._phi_tidal_PN(Mf, PN, kappaA, kappaB)   # (B, F)

        return phi_clipped * (1.0 - planck) + phi_PN * planck

    # ----------------------------------------------------------------
    # 10. NRTidalv2 tidal amplitude correction
    #     Source: SimNRTunedTidesFDTidalAmplitude
    #             LALSimNRTunedTides.c lines 344-366
    #     Reference: Eq. 24 of arXiv:1905.06011 (reused in NRTidalv3)
    # ----------------------------------------------------------------
    @staticmethod
    def _tidal_amp(Mf, kappa2T):
        """
        Tidal amplitude correction (dimensionless).

        Parameters: Mf (B, F); kappa2T (B, 1).
        Returns: ampT (B, F).
        """
        # x = (pi * Mf)^(2/3)
        x    = (math.pi * Mf).pow(2.0/3.0)
        n1   = 4.157407407407407
        n289 = 2519.111111111111
        d    = 13477.8073677
        poly = (1.0 + n1*x + n289*x.pow(2.89)) / (1.0 + d*x.pow(4.0))
        return -9.0 * kappa2T * x.pow(3.25) * poly

    # ----------------------------------------------------------------
    # 11. Spin-induced quadrupole/octupole moment universal relations
    #     Source: XLALSimInspiralEOSQfromLambda
    #             LALSimInspiralEOS.c lines 104-123
    #             XLALSimUniversalRelationSpinInducedOctupoleVSSpinInducedQuadrupole
    #             LALSimUniversalRelations.c lines 150-162
    # ----------------------------------------------------------------
    @staticmethod
    def _quadparam_from_lambda(lam):
        """
        Spin-induced quadrupole moment parameter from tidal deformability.

        Parameters: lam (B, 1) — tidal deformability Lambda.
        Returns: quadparam (B, 1) — quadrupole moment parameter.

        XLALSimInspiralEOSQfromLambda:
          loglam = ln(lambda)
          quadparam = exp(0.194 + 0.0936*loglam + 0.0474*loglam^2
                        - 0.00421*loglam^3 + 0.000123*loglam^4)
          Returns 1.0 for lambda < 0.5 (BH limit).
        """
        # Clip to avoid log(0) or negative lambda; values < 0.5 → quadparam=1
        loglam = torch.log(lam.clamp(min=0.5))
        q = (0.194
             + 0.0936 * loglam
             + 0.0474 * loglam * loglam
             - 0.00421 * loglam.pow(3)
             + 0.000123 * loglam.pow(4))
        quadparam = torch.exp(q)
        # For lambda < 0.5, set quadparam = 1 (BH)
        return torch.where(lam < 0.5, torch.ones_like(lam), quadparam)

    @staticmethod
    def _octparam_from_quadparam(quadparam):
        """
        Spin-induced octupole moment parameter from quadrupole moment parameter.

        Parameters: quadparam (B, 1).
        Returns: octparam (B, 1) with the BBH baseline removed (i.e., oct - 1).

        XLALSimUniversalRelationSpinInducedOctupoleVSSpinInducedQuadrupole:
          coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]
          lnq = ln(quadparam)
          octparam_raw = exp(coeffs[0] + coeffs[1]*lnq + ... + coeffs[4]*lnq^4)
          octparam = octparam_raw - 1  (remove BBH baseline)
        """
        lnq = torch.log(quadparam.clamp(min=1e-10))
        lny = (0.003131
               + 2.071   * lnq
               - 0.7152  * lnq * lnq
               + 0.2458  * lnq.pow(3)
               - 0.03309 * lnq.pow(4))
        return torch.exp(lny) - 1.0

    # ----------------------------------------------------------------
    # 12. 2PN+3PN+3.5PN spin-induced quadrupole/octupole tidal phase
    #     Source: IMRPhenomXGetTidalPhaseCoefficients and
    #             IMRPhenomX_TidalPhase (NRTidalv3 branch)
    #             LALSimIMRPhenomX_internals.c lines 2893-3085
    #
    #     Formulas:
    #       pfaN = 3/(128 * Xa * Xb)
    #
    #       c2PN = (4PNQM2SOCoeff(Xa) + 4PNQM2SCoeff(Xa)) * (qp1-1)*chi1^2
    #              + (same for b)
    #            where 4PNQM2SOCoeff(X) = -75 * X^2
    #                  4PNQM2SCoeff(X)  =  25 * X^2
    #            → c2PN = -50 * (Xa^2*(qp1-1)*chi1^2 + Xb^2*(qp2-1)*chi2^2)
    #
    #       c3PN = 6PNQM2SCoeff(Xa) * (qp1-1)*chi1^2 + (same for b)
    #            where 6PNQM2SCoeff(X) = (4703.5/8.4 + 2935/6*X - 120*X^2)*X^2
    #
    #       c3p5PN = SS_3p5PN + SSS_3p5PN  (from XLALSimInspiralGetHOSpinTerms)
    #            SS_3p5PN  = -400*pi*(qp1-1)*chi1^2*Xa^2 - 400*pi*(qp2-1)*chi2^2*Xb^2
    #            SSS_3p5PN = 10*((Xa^2+308/3*Xa)*chi1+(Xb^2-89/3*Xb)*chi2)*(qp1-1)*Xa^2*chi1^2
    #                       +10*((Xb^2+308/3*Xb)*chi2+(Xa^2-89/3*Xa)*chi1)*(qp2-1)*Xb^2*chi2^2
    #                       -440*oct1*Xa^3*chi1^3 - 440*oct2*Xb^3*chi2^3
    #
    #       phase contribution at frequency Mf:
    #         phi_spin = pfaN * (c2PN * pi^(-1/3) * Mf^(-1/3)
    #                          + c3PN * pi^(1/3)  * Mf^(1/3)
    #                          + c3p5PN * pi^(2/3) * Mf^(2/3))
    # ----------------------------------------------------------------
    @staticmethod
    def _spin_tidal_pn_phase(Mf, Xa, Xb, chi1L, chi2L,
                             quadparam1, quadparam2,
                             octparam1, octparam2):
        """
        2PN+3PN+3.5PN spin-induced quadrupole/octupole tidal phase.

        All parameters (B, 1) except Mf which may be (B, F) or (B, 1).
        Returns phi_spin with the same shape as Mf (negative for typical BNS).
        """
        pi = math.pi

        Xa2 = Xa * Xa
        Xb2 = Xb * Xb
        chi1sq = chi1L * chi1L
        chi2sq = chi2L * chi2L
        dqp1 = quadparam1 - 1.0   # zero for BBH
        dqp2 = quadparam2 - 1.0

        pfaN = 3.0 / (128.0 * Xa * Xb)

        # 2PN: (4PNQM2SOCoeff + 4PNQM2SCoeff)(X) = (-75 + 25)*X^2 = -50*X^2
        c2pn = -50.0 * (Xa2 * dqp1 * chi1sq + Xb2 * dqp2 * chi2sq)

        # 3PN: 6PNQM2SCoeff(X) = (4703.5/8.4 + 2935/6*X - 120*X^2) * X^2
        coeff6A = (4703.5/8.4 + (2935.0/6.0)*Xa - 120.0*Xa2) * Xa2
        coeff6B = (4703.5/8.4 + (2935.0/6.0)*Xb - 120.0*Xb2) * Xb2
        c3pn = coeff6A * dqp1 * chi1sq + coeff6B * dqp2 * chi2sq

        # 3.5PN: SS + SSS terms (XLALSimInspiralGetHOSpinTerms)
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

        # Phase: pfaN * (c2pn * (pi*Mf)^(-1/3) + c3pn * (pi*Mf)^(1/3)
        #               + c3p5pn * (pi*Mf)^(2/3))
        piMf = pi * Mf
        return pfaN * (
            c2pn  * piMf.pow(-1.0/3.0)
            + c3pn  * piMf.pow( 1.0/3.0)
            + c3p5pn * piMf.pow( 2.0/3.0)
        )

    @staticmethod
    def _spin_tidal_pn_dphase(Mf, Xa, Xb, chi1L, chi2L,
                               quadparam1, quadparam2,
                               octparam1, octparam2):
        """
        Derivative d(phi_spin)/dMf of the 2PN+3PN+3.5PN spin-tidal phase.

        d/dMf [pfaN * c2pn * (pi*Mf)^(-1/3)] = pfaN*c2pn*pi^(-1/3)*(-1/3)*Mf^(-4/3)
        d/dMf [pfaN * c3pn * (pi*Mf)^(1/3)]  = pfaN*c3pn*pi^(1/3)*(1/3)*Mf^(-2/3)
        d/dMf [pfaN*c3p5*(pi*Mf)^(2/3)]       = pfaN*c3p5*pi^(2/3)*(2/3)*Mf^(-1/3)

        = pfaN * (-c2pn + c3pn*(pi*Mf)^(2/3)) / (3*(pi*Mf)^(1/3)*Mf)
          + pfaN * (2/3) * c3p5pn * (pi*Mf)^(2/3) / Mf

        All parameters (B, 1). Returns (B, 1).
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
        # d/dMf [c2pn*(pi*Mf)^(-1/3)] = c2pn*pi^(-1/3)*(-1/3)*Mf^(-4/3)
        #   = c2pn * (-1/3) * pi^(-1/3) * Mf^(-4/3)
        # Combined with 3PN term (Eq. at LALSim internals.c line 3102):
        #   threePN_dphase = pfaN * (-c2pn + c3pn*(piMf)^(2/3)) / (3 * Mf^(4/3) * pi^(1/3))
        #   Note Mf^(-4/3) * pi^(-1/3) = (piMf)^(-1/3) / Mf
        three_pn_dphi = pfaN * (-c2pn + c3pn * piMf.pow(2.0/3.0)) / (3.0 * piMf.pow(1.0/3.0) * Mf)
        # 3.5PN derivative: pfaN * c3p5pn * (2/3) * pi^(2/3) * Mf^(-1/3)
        c3p5_dphi = pfaN * c3p5pn * (2.0/3.0) * piMf.pow(2.0/3.0) / Mf

        return three_pn_dphi + c3p5_dphi

    # ----------------------------------------------------------------
    # 13. LAL's parameter checks and body order
    #     Sources: XLALSimIMRPhenomXHM (LALSimIMRPhenomXHM.c 1424-1443),
    #              XLALSimIMRPhenomXASGenerateFD (LALSimIMRPhenomX.c 209-235),
    #              IMRPhenomXSetWaveformVariables (internals.c 326-385, 510-600),
    #              IMRPhenomXASGenerateFD (LALSimIMRPhenomX.c 589-595)
    # ----------------------------------------------------------------
    @staticmethod
    def raise_first(failures):
        """
        Raise the message of the first failing check of the first failing row.

        Parameters
        ----------
        failures : list of (torch.Tensor, callable)
            Per-row failure masks, shape (B,), in LAL's order, each with a
            function of the row index that returns LAL's message.

        Raises
        ------
        ValueError
            When any row fails a check.
        """
        failed = torch.stack([mask for mask, _ in failures])        # (checks, B)
        if not failed.any():
            return
        row = int(failed.any(dim=0).nonzero()[0])
        check = int(failed[:, row].nonzero()[0])
        raise ValueError(f"signal {row}: {failures[check][1](row)}")

    @classmethod
    def check_parameters(cls, theta):
        """
        Raise LAL's error for parameters it rejects before ordering the bodies.

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

    @staticmethod
    def order_bodies(theta):
        """
        Swap masses, spins and tidal deformabilities where mass1 < mass2, as LAL does.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 10)
            Tidal aligned layout.

        Returns
        -------
        torch.Tensor, shape (B, 10)
        """
        swap = ~(theta[:, 0:1] >= theta[:, 1:2])
        if swap.any():
            warnings.warn("Warning: m1 < m2, swapping the masses, spins, tidal deformabilities, "
                          "and spin-induced quadrupole parameters.")
        return torch.where(swap, theta[:, [1, 0, 3, 2, 4, 5, 6, 7, 9, 8]], theta)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute FD plus and cross polarisations for a BNS parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 17)
            Parameters in the waveform_utils.WAVEFORM_PARAMETERS layout.
        reproduce_lal : bool
            If True, skip FD tapering, tc shift, and df normalisation so
            the output can be compared directly with raw LALSim output.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_frequency_bins), complex128

        Raises
        ------
        ValueError
            If any transverse spin is non-zero, or LAL rejects the parameters.
        """
        # ----------------------------------------------------------------
        # 1. Extract intrinsic + extrinsic parameters
        # ----------------------------------------------------------------
        # [m1, m2, chi1, chi2, distance, tc, phic, inclination, lambda1,
        #  lambda2]
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

        # ----------------------------------------------------------------
        # 2. Derived BBH parameters
        # ----------------------------------------------------------------
        derived  = self.compute_derived_parameters(theta)
        M_s   = derived[:, 2:3]    # total mass in seconds          (B, 1)
        eta   = derived[:, 3:4]    # symmetric mass ratio
        delta = derived[:, 4:5]    # |m1 - m2| / (m1 + m2)
        STotR = derived[:, 9:10]   # effective spin for fits
        dchi  = derived[:, 10:11]  # chi1 - chi2
        chi1L = derived[:, 11:12]  # aligned spin body 1
        chi2L = derived[:, 12:13]  # aligned spin body 2

        # Tidal mass fractions are pWF->m1, pWF->m2 = m / Mtot (internals.c 422-423).
        Mtot   = m1 + m2
        Xa     = m1 / Mtot                                          # (B, 1)
        Xb     = m2 / Mtot
        # pWF->q, set to 1 when eta == 0.25 (internals.c 391-402), sets the
        # merger frequency of the alignment and of the tidal phase at fRef;
        # the tidal series uses m1/m2 from the SI masses (LALSimNRTunedTides.c 797-799).
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

        # LAL's cut and its checks on it (internals.c 503-540, 574-600; LALSimIMRPhenomX.c 589-595).
        f_cut = torch.where(derived[:, 7:8] > 0.99, torch.full_like(M_s, 0.33), torch.full_like(M_s, 0.3)) / M_s
        f_max_prime = torch.clamp(f_cut, max=self.f_max)
        for fails, message in ((f_cut <= self.f_min, "(fCut = {:g} Hz) <= f_min = {:g}"),
                               (f_max_prime <= self.f_min, "f_max <= f_min")):
            if fails.any():
                i = int(fails[:, 0].nonzero()[0])
                warnings.warn(message.format(float(f_cut[i]), self.f_min))
        Mfinal = self.final_mass_2017(eta, STotR, dchi)
        afinal = self.final_spin_2017(eta, STotR, dchi)
        i_start = int(self.f_min / self.delta_f)
        i_stop = (f_max_prime / self.delta_f).floor() + 1
        self.raise_first([
            (Mfinal[:, 0] > 1.0, lambda i: "IMRPhenomX_FinalMass2018: Final mass > 1.0 not physical."),
            (torch.abs(afinal[:, 0]) > 1.0, lambda i: "IMRPhenomX_FinalSpin2018: Final spin > 1.0 is not physical."),
            (self.fMECO(eta, derived[:, 8:9], dchi)[:, 0] > self.fISCO(afinal)[:, 0],
             lambda i: "Error: f_MECO cannot be greater than f_ISCO."),
            (i_start > i_stop[:, 0],
             lambda i: f"minimum freq index {i_start} and maximum freq index {int(i_stop[i])} do not fulfill "
                       f"0<=ind_min<=ind_max<=htilde->data>length={int(i_stop[i])}."),
            # LAL's tidal-phase minimum search compares phi_tidal[i] with phi_tidal[i-1] from
            # 0.9 f_merger on; at i = 0 the unsigned i-1 wraps and LAL reads outside the array
            # and crashes (LALSimNRTunedTides.c 834-836), so LAL has no result here.
            (self.f[:, 0] >= 0.9 * f_merger_tides[:, 0],
             lambda i: f"f_min = {self.f_min:g} Hz is too high."),
        ])

        # ----------------------------------------------------------------
        # 3. BBH amplitude and phase coefficients
        # ----------------------------------------------------------------
        amp_coeffs   = self.get_amp_coeffs(derived)
        phase_coeffs = self.get_phase_coeffs(derived)
        fRING = phase_coeffs['fRING']
        fDAMP = phase_coeffs['fDAMP']

        # ----------------------------------------------------------------
        # 4. Overall amplitude prefactors
        #    amp0    = M_metres * M_s / dist_m
        #    ampNorm = sqrt(2*eta/3) * pi^{-1/6}
        # ----------------------------------------------------------------
        M_m    = M_s * self.C
        dist_m = dist_Mpc * self.Mpc
        amp0   = M_m * M_s / dist_m                                # (B, 1)
        ampNorm = torch.sqrt(2.0*eta/3.0) * (math.pi**(-1.0/6.0))

        # ----------------------------------------------------------------
        # 5. Tidal coefficients
        # ----------------------------------------------------------------
        kappa2T        = self._kappa2T(Xa, Xb, lambda1, lambda2)
        kappaA, kappaB = self._kappaAB(Xa, Xb, lambda1, lambda2)
        PN             = self._PN_coeffs(Xa)                       # (B, 10)
        tc_d           = self._nrtv3_coeffs(Xa, Xb, kappa2T, kappaA, kappaB, PN)

        # Spin-induced quadrupole/octupole parameters (non-zero for BNS)
        # Source: XLALSimInspiralSetQuadMonParamsFromLambdas →
        #         XLALSimInspiralEOSQfromLambda (LALSimInspiralEOS.c)
        quadparam1 = self._quadparam_from_lambda(lambda1)          # (B, 1)
        quadparam2 = self._quadparam_from_lambda(lambda2)
        octparam1  = self._octparam_from_quadparam(quadparam1)     # oct - 1
        octparam2  = self._octparam_from_quadparam(quadparam2)

        # ----------------------------------------------------------------
        # 6. BBH linb (time-alignment shift)
        #    Identical to IMRPhenomXAS.get_hphc
        # ----------------------------------------------------------------
        linb_fit_val = IMRPhenomXAS._linb_fit(eta, STotR, dchi, delta)
        psi4val      = IMRPhenomXAS._psi4tostrain_fit(eta, STotR, dchi)

        frefFit   = fRING - fDAMP
        dphi22Ref = (1.0/eta) * self.dphase(frefFit, phase_coeffs, derived)
        linb_bbh  = linb_fit_val - dphi22Ref - 2.0*math.pi*(500.0 + psi4val)

        # ----------------------------------------------------------------
        # 7. Tidal correction to linb
        #    Source: LALSimIMRPhenomX.c lines 716-742
        #
        #    At Mf_final = min(Mfmerger, last grid frequency) the total
        #    group delay d phi_total/d Mf is forced to zero:
        #
        #      dphi_fmerger = (1/eta)*dphi_BBH(Mf_final) + linb_bbh
        #                     - dphi_tidal(Mf_final)
        #      linb         = linb_bbh - dphi_fmerger
        #
        #    The last grid frequency is LAL's floor(f_max_prime/deltaF)*deltaF,
        #    f_max_prime = min(f_max, fCutDef/M_s), fCutDef = 0.33 if
        #    chiEff > 0.99 else 0.3 (LALSimIMRPhenomX_internals.c 503-533,
        #    LALSimIMRPhenomX.c 590-605).
        #
        #    dphi_tidal(Mf_final) is LAL's analytic derivative (_dphi_tidal_nrt
        #    plus the spin-tidal PN terms).
        # ----------------------------------------------------------------
        f_last = (i_stop - 1) * self.delta_f
        Mf_final = torch.where(f_merger_wf < f_last, f_merger_wf, f_last) * M_s   # (B, 1)

        dphi_bbh_fmerger = (1.0/eta) * self.dphase(Mf_final, phase_coeffs, derived)

        dphi_tid_fmerger = self._dphi_tidal_nrt(Mf_final, PN, tc_d)

        # Add spin-tidal PN derivative: d(2PN+3PN+3.5PN spin phase)/dMf at Mf_final
        # Source: IMRPhenomX_TidalPhaseDerivative, internals.c line 3102, 3134
        dphi_tid_fmerger = dphi_tid_fmerger + self._spin_tidal_pn_dphase(
            Mf_final, Xa, Xb, chi1L, chi2L, quadparam1, quadparam2,
            octparam1, octparam2)

        dphi_fmerger = dphi_bbh_fmerger + linb_bbh - dphi_tid_fmerger
        linb         = linb_bbh - dphi_fmerger                     # (B, 1)

        # ----------------------------------------------------------------
        # 8. Reference-frequency tidal phase shift
        #    Source: LALSimIMRPhenomX.c line 740-748
        #
        #    phiTfRef = -phi_tidal(MfRef): the Pade phase blended with the
        #    7.5PN phase over [1.15, 1.35]*Mfmerger, without the minimum
        #    clamp (IMRPhenomX_TidalPhase, internals.c 3044-3070)
        #
        #    phifRef = -(1/eta * phi_BBH(MfRef) + phiTfRef + linb*MfRef)
        #              + 2*phic + pi/4
        # ----------------------------------------------------------------
        MfRef    = self.f_ref * M_s                                # (B, 1)
        phi22ref = self.phase(MfRef, phase_coeffs, derived)
        Mfmerger_wf = f_merger_wf * M_s
        planck_ref = self._planck_taper(MfRef, 1.15 * Mfmerger_wf, 1.35 * Mfmerger_wf,
                                        exp=lambda x: torch.cosh(x) + torch.sinh(x))
        phiTfRef = -(self._phi_tidal_nrt(MfRef, PN, tc_d) * (1.0 - planck_ref)
                     + self._phi_tidal_PN(MfRef, PN, kappaA, kappaB) * planck_ref
                     + self._spin_tidal_pn_phase(
                         MfRef, Xa, Xb, chi1L, chi2L,
                         quadparam1, quadparam2, octparam1, octparam2))
        phifRef  = (-((1.0/eta)*phi22ref + phiTfRef + linb*MfRef)
                    + 2.0*phic + 0.25*math.pi)

        # ----------------------------------------------------------------
        # 9. Full frequency grid waveform
        # ----------------------------------------------------------------
        Mf = self.f * M_s                                          # (B, F)

        # BBH phase
        phi22   = self.phase(Mf, phase_coeffs, derived)            # (B, F)
        phi_bbh = phi22*(1.0/eta) + linb*Mf + phifRef              # (B, F)

        # Tidal phase (with minimum-clamping and Planck taper)
        # phi_tidal < 0; subtracting it increases the total phase
        # matching LALSim's  cexp(I*(phi - phaseTidal)) with phaseTidal < 0
        phi_tidal = (self._tidal_phase_full(self.f, Mf, PN, tc_d, f_merger_tides)
                     + self._spin_tidal_pn_phase(
                         Mf, Xa, Xb, chi1L, chi2L,
                         quadparam1, quadparam2, octparam1, octparam2))  # (B, F)
        phi_total = phi_bbh - phi_tidal                             # (B, F)

        # ----------------------------------------------------------------
        # 10. Amplitude
        #     Source: LALSimIMRPhenomX.c line 888
        #
        #     h(f) = amp0 * [ampNorm * Mf^{-7/6} * A_BBH
        #                    + 2*sqrt(pi/5) * A_tidal] * planck_window
        # ----------------------------------------------------------------
        A_bbh  = self.amp(Mf, amp_coeffs, derived)                 # (B, F)
        ampT   = self._tidal_amp(Mf, kappa2T)                      # (B, F)

        coeff_tidal = 2.0 * math.sqrt(1.0/5.0) * math.sqrt(math.pi)
        A_total = amp0 * (ampNorm * Mf.pow(-7.0/6.0) * A_bbh
                          + coeff_tidal * ampT)                     # (B, F)

        # Planck amplitude taper: 1 below fmerger, tapers to 0 at 1.2*fmerger
        amp_win = 1.0 - self._planck_taper(self.f, f_merger_tides, 1.2*f_merger_tides)
        h22 = A_total * torch.polar(torch.ones_like(phi_total), phi_total) * amp_win
        # LAL leaves bins above its cut at zero (LALSimIMRPhenomX.c 806-807, 906).
        h22 = torch.where((self.f <= f_last) & (Mf <= f_max_prime * M_s), h22, torch.zeros_like(h22))

        # ----------------------------------------------------------------
        # 11. hp / hc: the (2,-2) mode added with its m = +-2 harmonics at
        #     phi = pi/2, as IMRPhenomXHM_MultiMode and IMRPhenomXHMFDAddMode
        #     do (LALSimIMRPhenomXHM.c 1880-1888, 2282-2328)
        # ----------------------------------------------------------------
        Ym = IMRPhenomXHM.spin_weighted_Y(iota, math.pi / 2, -2, 2, -2)
        Ystar = torch.conj(IMRPhenomXHM.spin_weighted_Y(iota, math.pi / 2, -2, 2, 2))
        hp = 0.5 * (Ym + Ystar) * h22
        hc = 1j * 0.5 * (Ym - Ystar) * h22

        # ----------------------------------------------------------------
        # 12. Optional: FD taper, tc shift, df normalisation
        # ----------------------------------------------------------------
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

        # Zero-pad from DC to f_low so the output matches the full 0→Nyquist
        # frequency grid expected by ConstantProjection.
        return self.pad_missing_frequencies(hp, hc)
