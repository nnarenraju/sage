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
                0  : m1          (solar masses, m1 >= m2)
                1  : m2          (solar masses)
                2  : chi1z       (dimensionless aligned spin, body 1)
                3  : chi2z       (dimensionless aligned spin, body 2)
                4  : distance    (Mpc)
                5  : tc          (s, time of coalescence)
                6  : phic        (rad, reference orbital phase)
                7  : inclination (rad)
                8  : lambda1     (dimensionless tidal deformability Lambda_1 >= 0)
                9  : lambda2     (dimensionless tidal deformability Lambda_2 >= 0)

Created on 2026-05-28

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
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

import torch

from sage.data.waveform.approximants.IMRPhenomXAS import IMRPhenomXAS
from sage.data.waveform import taper as _taper_mod
from sage.data.waveform import waveform_utils
from sage.data.waveform.multiband_selector import MultibandSelector
from sage.core.config import get_cfg, get_data_cfg
from sage.core.pipeline import Grid, ProcessingState


class IMRPhenomXAS_NRTidalv3(IMRPhenomXAS, torch.nn.Module):
    """
    GPU-native batched IMRPhenomXAS with NRTidalv3 tidal corrections.

    Inherits the full BBH backbone (amplitude, three-region phase, QNM tables,
    connection coefficients, time-alignment fit) from IMRPhenomXAS and adds
    per-star NRTidalv3 tidal phase/amplitude on top.

    When constructed with a ``param_sampler`` this class also acts as a
    signal-sampler ``torch.nn.Module`` (same pattern as ``IMRPhenomPv2``).
    Call ``forward()`` to obtain a batch of detector-frame strain tensors
    and normalised parameter targets ready for network training.

    Parameters (waveform math only)
    --------------------------------
    f : torch.Tensor, shape (B, F)
        Frequency grid in Hz.
    f_ref : torch.Tensor, shape (B, 1)
        Reference frequency in Hz.

    Parameters (signal-sampler mode)
    ---------------------------------
    param_sampler : DistributionSampler or None
        BNS parameter sampler built from ``runs/bns/gwconfig.yaml``.
        When ``None`` the instance can still be used as pure waveform math.
    waveform_project : ConstantProjection or None
        Multi-detector projection module.
    augment : callable or None
        Optional SNR-rescaling augmentation.
    """

    GRAPH_READY = True

    # Multibanding modes
    MULTIBAND_NONE       = 'none'        # full uniform FD grid (default)
    MULTIBAND_WORST_CASE = 'worst_case'  # single coarse grid, worst-case masses
    MULTIBAND_PER_SIGNAL = 'per_signal'  # per-signal LAL grid (future / not yet implemented)

    @property
    def output_state(self) -> ProcessingState:
        """
        Processing state of the waveform batch returned by forward().

        Used by :class:`~sage.factory.training.SageVanillaTraining` to
        automatically configure GWBatch tracking and noise multibanding.

        Returns
        -------
        ProcessingState
            ``Grid.FD_COARSE`` when ``multiband_mode='worst_case'``;
            ``Grid.FD_UNIFORM`` otherwise.
        """
        if self.multiband_mode == self.MULTIBAND_WORST_CASE:
            return ProcessingState(Grid.FD_COARSE)
        return ProcessingState(Grid.FD_UNIFORM)

    # Ordered parameter names fed to get_hphc (first 10) and to the
    # projection module (last 3).  Must match the theta column layout
    # documented in the class docstring of get_hphc.
    WAVEFORM_PARAM_NAMES = [
        "mass1",        # 0
        "mass2",        # 1
        "chi1z",        # 2
        "chi2z",        # 3
        "distance",     # 4
        "tc",           # 5
        "coa_phase",    # 6
        "inclination",  # 7
        "lambda1",      # 8
        "lambda2",      # 9
        "polarization", # 10 — projection only
        "ra",           # 11 — projection only
        "dec",          # 12 — projection only
    ]

    def __init__(
        self,
        param_sampler=None,
        waveform_project=None,
        augment=None,
        multiband_mode='none',
        m1_worst=None,
        m2_worst=None,
    ):
        """
        Parameters
        ----------
        param_sampler : DistributionSampler or None
        waveform_project : ConstantProjection or None
        augment : callable or None
        multiband_mode : str
            'none'        — full uniform FD grid, behaviour unchanged (default).
            'worst_case'  — waveforms generated directly at the worst-case
                            coarse grid; self.selector is available for noise.
            'per_signal'  — per-signal LAL grid (not yet implemented).
        m1_worst, m2_worst : float or None
            Component masses (M_sun) used to build the worst-case grid.
            When both are None (default) the worst-case pair is found
            automatically by scanning param_sampler.bounds at startup —
            no hardcoded values.  Provide explicit masses only to skip the
            scan (e.g. for reproducibility in tests).
            Only used when multiband_mode='worst_case'.
        """
        torch.nn.Module.__init__(self)

        self.cfg = get_cfg()
        self.data_cfg = get_data_cfg()
        self.multiband_mode = multiband_mode
        self.selector = None

        if multiband_mode == self.MULTIBAND_PER_SIGNAL:
            raise NotImplementedError(
                "multiband_mode='per_signal' is not yet implemented. "
                "Future: per-signal SimInspiralChooseFDWaveformSequence + interpolation. "
                "Use 'none' or 'worst_case'."
            )

        self.signal_batch_size = int(self.cfg.batch_size * self.cfg.class_balance)

        # Build the full-resolution uniform frequency grid.  In worst_case
        # mode we still need it briefly so that df and sample_length_in_s are
        # set from the base delta_f before we swap self.f for the coarse grid.
        f, f_ref = waveform_utils.get_freqs(
            self.data_cfg.signal_low_frequency_cutoff,
            self.data_cfg.sample_rate / 2.0,
            self.data_cfg.padded_length_in_s,
            self.signal_batch_size,
            self.cfg.device,
            self.cfg.dtype,
        )
        # df = base uniform spacing (= padded_delta_f = 1/padded_length_in_s).
        # Stored now before self.f is potentially replaced so normalisation is
        # always relative to the base grid, not to any coarse spacing.
        self.df = f[0][1] - f[0][0]
        self.sample_length_in_s = 1.0 / self.df
        self.f_ref = f_ref

        # Initialise the BBH backbone with the full-resolution grid.
        # IMRPhenomXAS stores internal coefficient tables that don't depend on
        # the frequency array itself, so this is always correct.
        IMRPhenomXAS.__init__(self, f, f_ref)
        self.B = f.shape[0]

        if multiband_mode == self.MULTIBAND_WORST_CASE:
            # Build the worst-case MultibandSelector.
            #
            # If explicit masses are provided, use them directly (useful for
            # reproducibility or when the caller has already run the scan).
            # Otherwise scan the mass prior at runtime to find the pair that
            # maximises N_coarse — no hardcoded masses, always consistent with
            # the gwconfig being used.
            if m1_worst is not None and m2_worst is not None:
                self.selector = MultibandSelector.from_prior(
                    m1_worst=m1_worst,
                    m2_worst=m2_worst,
                    data_cfg=self.data_cfg,
                    device=self.cfg.device,
                )
            elif param_sampler is not None:
                self.selector = MultibandSelector.from_prior_scan(
                    param_sampler=param_sampler,
                    data_cfg=self.data_cfg,
                    device=self.cfg.device,
                )
            else:
                raise ValueError(
                    "multiband_mode='worst_case' requires either a param_sampler "
                    "(so the prior mass bounds can be read) or explicit m1_worst "
                    "and m2_worst masses."
                )
            # Replace the uniform signal-band frequency grid with the coarse
            # grid so that all waveform computations run at N_coarse points and
            # the full-resolution waveform is never materialised.
            coarse_freqs = self.selector.coarse_freqs.to(self.cfg.dtype)  # (N_coarse,)
            self.f = coarse_freqs.unsqueeze(0).expand(self.signal_batch_size, -1)  # (B, N_coarse)
            self.f_numel = int(self.f[0].numel())   # N_coarse
            self.n_pad   = 0   # no DC-to-f_min region in the coarse representation
        else:
            self.f = f
            self.f_numel = self.f[0].numel()
            # n_pad: bins from 0 Hz to f_low on the full padded grid.
            # Computed via exact integer arithmetic to avoid float rounding.
            F_total = int(self.data_cfg.sample_rate * self.data_cfg.padded_length_in_s / 2) + 1
            self.n_pad = F_total - self.f_numel

        self.hp_buffer = torch.empty(
            (self.B, self.n_pad + self.f_numel),
            dtype=torch.complex64,
            device=self.cfg.device,
        )
        self.hc_buffer = torch.empty_like(self.hp_buffer)

        self.param_sampler = param_sampler
        self.waveform_project = waveform_project
        self.augment = augment

        # In worst_case multibanding the forward pass produces (B, D, N_coarse).
        # OptimalSNREstimator was built for (B, D, F_full); slice its ASD and
        # mask to the coarse indices so the SNR computation is dimensionally
        # consistent.  The SNR integral uses the base padded delta_f at each
        # coarse bin, which is a good approximation since all coarse bins are
        # in the signal band and the multibanding grid preserves the PN phase
        # accumulation rate — the SNR integral converges to the correct value.
        if self.augment is not None and multiband_mode == self.MULTIBAND_WORST_CASE:
            idx     = self.selector.coarse_indices        # (N_coarse,)
            snr_est = self.augment.snr_estimator
            snr_est.asds = snr_est.asds[:, :, idx]        # (1, D, N_coarse)
            if snr_est.mask is not None:
                snr_est.mask = snr_est.mask[:, :, idx]    # (1, 1, N_coarse)

        if self.param_sampler is not None:
            get_idx = self.param_sampler.param_index
            self.req_idx = torch.tensor(
                [get_idx[key] for key in self.WAVEFORM_PARAM_NAMES],
                device=self.cfg.device,
                dtype=torch.int32,
            )
            # Column of "distance" in the full all_theta tensor — used
            # by the SNR-rescaling augmentation to keep distance consistent.
            self.dist_col = int(self.req_idx[4].item())

            self.param_sampler.req_idx = self.req_idx
            self.param_sampler._compile_batch_normaliser()
            self.param_sampler._compile_batch_standardiser()
            self.param_sampler.to(self.cfg.device)

        if self.waveform_project is not None:
            self.waveform_project.to(self.cfg.device)

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
        req_theta = all_theta[:, self.req_idx]

        # First 10 columns are the waveform parameters; last 3 are for projection.
        hp, hc = self.get_hphc(req_theta[:, :10])

        if self.multiband_mode == self.MULTIBAND_WORST_CASE:
            # hp, hc are (B, N_coarse) at the coarse frequencies.
            # Pass the 1-D coarse freq array so the time-delay phase in
            # ConstantProjection is evaluated at the correct frequencies.
            hf = self.waveform_project(
                hp,
                hc,
                ra=req_theta[:, 11],
                dec=req_theta[:, 12],
                polarization=req_theta[:, 10],
                freqs=self.f[0],  # (N_coarse,) — 1D coarse freq array
            )
        else:
            hf = self.waveform_project(
                hp,
                hc,
                ra=req_theta[:, 11],
                dec=req_theta[:, 12],
                polarization=req_theta[:, 10],
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

    def apply_tc(self, hp, hc, tc):
        """
        Apply tc phase shift using the analysis-window convention.

        ``tc`` is measured from the START of the analysis window (same
        convention as IMRPhenomPv2).  Adding ``padding_length_in_s``
        converts it to the padded-segment frame before computing the FD
        phase ramp, so gwconfig tc values are directly interpretable as
        "seconds into the analysis window."
        """
        _tc = (tc + self.data_cfg.padding_length_in_s) - self.sample_length_in_s
        hp = torch.polar(torch.abs(hp), torch.angle(hp) - 2 * self.PI * self.f * _tc)
        hc = torch.polar(torch.abs(hc), torch.angle(hc) - 2 * self.PI * self.f * _tc)
        return hp, hc

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
        exps2s3 = torch.exp(s2s3)   # = exp(s2*s3)

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
    # 5. BNS merger frequency in dimensionless Mf
    #    Source: XLALSimNRTunedTidesMergerFrequency_v3
    #            LALSimNRTunedTides.c lines 209-291
    #    Reference: Eq. (23) of arXiv:2210.16366
    #
    #    Key identity: Mfmerger = nu * Qfit  (mass-independent!)
    # ----------------------------------------------------------------
    @staticmethod
    def _merger_freq_v3(Xa, Xb, lambda1, lambda2, chi1L, chi2L):
        """
        BNS merger frequency in dimensionless Mf = f * M_total.

        The result is mass-independent: Mfmerger = eta * Qfit.

        Parameters: all (B, 1).
        Returns Mfmerger (B, 1).
        """
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

        # Mfmerger = nu * a_0 * QM * QS * QT  (dimensionless)
        return nu * a_0 * QM * QS * QT

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
        # Use torch.exp directly: cosh(x)+sinh(x) overflows to NaN for
        # large |x| (cosh→+∞, sinh→−∞, inf−inf=NaN), whereas exp(x)
        # gracefully underflows to 0 for large negative x.
        s2Mf    = -2.0 * s2 * math.pi * Mf
        exps2Mf = torch.exp(s2Mf)

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
    def _planck_taper(f, f1, f2):
        """
        Planck taper: 0 at f <= f1, smooth rise, 1 at f >= f2.

        Parameters: f (B, F), f1 / f2 (B, 1).
        Returns window (B, F).
        """
        mid    = (f > f1) & (f < f2)
        f_safe = torch.where(mid, f, 0.5*(f1 + f2))   # avoid /0 outside mid
        arg    = (f2 - f1)/(f_safe - f1) + (f2 - f1)/(f_safe - f2)
        # Must use torch.exp, NOT cosh(arg)+sinh(arg).
        # When f → f2⁻, arg → −∞  (e.g. −10 918 at one grid step from the
        # upper edge for a 32 s segment).  exp(−10918) = 0 gracefully, but
        # cosh(−10918) = +∞ and sinh(−10918) = −∞  ⟹  ∞ + (−∞) = NaN.
        exp_fn = torch.exp(arg)
        taper_mid = 1.0 / (exp_fn + 1.0)

        return torch.where(f <= f1, torch.zeros_like(f),
               torch.where(f >= f2, torch.ones_like(f), taper_mid))

    # ----------------------------------------------------------------
    # 9. Full tidal phase: minimum-clamping + Planck taper
    #    Source: XLALSimNRTunedTidesFDTidalPhaseFrequencySeries
    #            (NRTidalv3_V branch), LALSimNRTunedTides.c lines 827-853
    #
    #    Procedure (mirrors LALSim):
    #      a) Evaluate raw NRTidalv3 Pade phase at every frequency.
    #      b) From 0.9 * fmerger onward, freeze the phase at its running
    #         minimum (torch.cummin handles this exactly).
    #      c) Smoothly transition to 7.5PN tidal phase in the post-merger
    #         Planck window [1.15 * fmerger, 1.35 * fmerger].
    # ----------------------------------------------------------------
    def _tidal_phase_full(self, Mf, PN, tc, Mfmerger):
        """
        NRTidalv3 tidal phase with minimum-clamping and Planck taper.

        Parameters
        ----------
        Mf       : (B, F)
        PN       : (B, 10)
        tc       : dict of (B, 1) tensors
        Mfmerger : (B, 1)

        Returns
        -------
        phi_tidal : (B, F)  (negative; tidal phase is a negative correction)
        """
        kappaA = tc['kappaA']
        kappaB = tc['kappaB']

        # (a) Raw NRTidalv3 Pade phase
        phi_raw    = self._phi_tidal_nrt(Mf, PN, tc)          # (B, F)

        # (b) Minimum-clamping via running minimum
        #     torch.cummin gives min(phi_raw[0..k]) at each index k.
        #     phi_raw is monotonically decreasing in the inspiral, so cummin
        #     tracks phi_raw exactly until the phase minimum, then stays
        #     constant — exactly the "freeze at minimum" behavior in LALSim.
        check_mask = (Mf >= 0.9 * Mfmerger)                   # (B, F)
        phi_cummin = torch.cummin(phi_raw, dim=-1).values       # (B, F)
        phi_clipped = torch.where(check_mask, phi_cummin, phi_raw)

        # (c) Smooth post-merger transition to 7.5PN tidal phase
        Mf_t1  = 1.15 * Mfmerger                              # (B, 1)
        Mf_t2  = 1.35 * Mfmerger
        planck = self._planck_taper(Mf, Mf_t1, Mf_t2)        # (B, F)
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute FD plus and cross polarisations for a BNS parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 10+)
            Columns: [m1, m2, chi1z, chi2z, distance, tc, phic,
                      inclination, lambda1, lambda2]
            Masses in solar masses, distance in Mpc, angles in radians,
            tidal deformabilities dimensionless (>= 0).
        reproduce_lal : bool
            If True, skip FD tapering, tc shift, and df normalisation so
            the output can be compared directly with raw LALSim output.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_pad + F), complex128
        """
        # ----------------------------------------------------------------
        # 1. Extract intrinsic + extrinsic parameters
        # ----------------------------------------------------------------
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
        delta = derived[:, 4:5]    # sqrt(1 - 4*eta)
        STotR = derived[:, 9:10]   # effective spin for fits
        dchi  = derived[:, 10:11]  # chi1 - chi2
        chi1L = derived[:, 11:12]  # aligned spin body 1
        chi2L = derived[:, 12:13]  # aligned spin body 2

        # For TIDAL computations use EXACT mass fractions from the
        # raw input masses — bypassing the eta nudge that biases kappa2T
        # and all NRTidalv3 coefficients for near-equal-mass binaries.
        # LALSim uses m1/(m1+m2) directly in all NRTidal functions.
        # (The BBH backbone goes through `derived` internally so it is
        #  unaffected by this separate Xa/Xb definition.)
        Mtot   = m1 + m2
        Xa     = m1 / Mtot                                          # (B, 1)
        Xb     = m2 / Mtot

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
        Mfmerger       = self._merger_freq_v3(Xa, Xb, lambda1, lambda2, chi1L, chi2L)

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
        #    At Mf_final = Mfmerger the total group delay d phi_total/d Mf
        #    is forced to zero:
        #
        #      dphi_fmerger = (1/eta)*dphi_BBH(Mf_final) + linb_bbh
        #                     - dphi_tidal(Mf_final)
        #      linb         = linb_bbh - dphi_fmerger
        #
        #    dphi_tidal(Mf_final) is computed by central finite difference
        #    on _phi_tidal_nrt (Planck taper = 0 at Mf_final = Mfmerger
        #    since Mfmerger < 1.15*Mfmerger, so unclipped formula applies).
        # ----------------------------------------------------------------
        Mf_final = Mfmerger                                        # (B, 1)

        dphi_bbh_fmerger = (1.0/eta) * self.dphase(Mf_final, phase_coeffs, derived)

        h_fd = Mf_final * 1e-6
        dphi_tid_fmerger = (
            self._phi_tidal_nrt(Mf_final + h_fd, PN, tc_d)
            - self._phi_tidal_nrt(Mf_final - h_fd, PN, tc_d)
        ) / (2.0 * h_fd)

        # Add spin-tidal PN derivative: d(2PN+3PN+3.5PN spin phase)/dMf at Mfmerger
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
        #    phiTfRef = -phi_tidal(MfRef)
        #             (MfRef << Mfmerger => Planck taper = 0,
        #              so raw _phi_tidal_nrt is the correct formula)
        #
        #    phifRef = -(1/eta * phi_BBH(MfRef) + phiTfRef + linb*MfRef)
        #              + 2*phic + pi/4
        # ----------------------------------------------------------------
        MfRef    = self.f_ref * M_s                                # (B, 1)
        phi22ref = self.phase(MfRef, phase_coeffs, derived)
        # phiTfRef = -IMRPhenomX_TidalPhase(MfRef, ...) which includes
        # NRTidalv3 Padé + 2PN + 3PN + 3.5PN spin-tidal terms.
        phiTfRef = -(self._phi_tidal_nrt(MfRef, PN, tc_d)
                     + self._spin_tidal_pn_phase(
                         MfRef, Xa, Xb, chi1L, chi2L,
                         quadparam1, quadparam2, octparam1, octparam2))
        phifRef  = (-(phi22ref/eta + phiTfRef + linb*MfRef)
                    + 2.0*phic + 0.25*math.pi)

        # ----------------------------------------------------------------
        # 9. Full frequency grid waveform
        # ----------------------------------------------------------------
        Mf = self.f * M_s                                          # (B, F)

        # BBH phase
        phi22   = self.phase(Mf, phase_coeffs, derived)            # (B, F)
        phi_bbh = phi22/eta + linb*Mf + phifRef                    # (B, F)

        # Tidal phase (with minimum-clamping and Planck taper)
        # phi_tidal < 0; subtracting it increases the total phase
        # matching LALSim's  cexp(I*(phi - phaseTidal)) with phaseTidal < 0
        phi_tidal = (self._tidal_phase_full(Mf, PN, tc_d, Mfmerger)
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
        amp_win = 1.0 - self._planck_taper(Mf, Mfmerger, 1.2*Mfmerger)
        A_total = A_total * amp_win

        # ----------------------------------------------------------------
        # 11. hp / hc from inclination (same projection as BBH baseline)
        # ----------------------------------------------------------------
        YLM      = math.sqrt(5.0 / (4.0 * math.pi)) / 2.0
        cos_iota = torch.cos(iota)

        hp = torch.polar(
            YLM * 0.5 * A_total * (1.0 + cos_iota*cos_iota),
            phi_total + math.pi,
        )
        hc = torch.polar(
            YLM * A_total * cos_iota,
            phi_total + 0.5*math.pi,
        )

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

        # In worst_case multibanding mode the waveform is already in the coarse
        # representation (N_coarse points from f_min to f_max).  No DC-to-f_low
        # zero-padding is needed or correct here — the coarse noise will be
        # selected from the full FD noise array using self.selector externally.
        if self.multiband_mode == self.MULTIBAND_WORST_CASE:
            return hp, hc

        # Zero-pad from DC to f_low so the output matches the full 0→Nyquist
        # frequency grid expected by ConstantProjection.
        return self.pad_missing_frequencies(hp, hc)
