#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : IMRPhenomXAS_NRTidalv3.py
Description   : GPU-native batched IMRPhenomXAS_NRTidalv3 frequency-domain
                waveform model for binary neutron stars (BNS) and
                neutron-star black-hole (NSBH) systems.

                Extends IMRPhenomXAS with NRTidalv3 tidal phase corrections
                (Abac et al. 2023, arXiv:2311.07456) applied on top of the
                aligned-spin BBH backbone.  Intended as a drop-in replacement
                for IMRPhenomPv2 in the Sage CBC detection pipeline when tidal
                effects are required.

                Parameters (theta columns)
                --------------------------
                0  : m1          (solar masses, m1 >= m2)
                1  : m2          (solar masses)
                2  : chi1z       (dimensionless aligned spin of body 1)
                3  : chi2z       (dimensionless aligned spin of body 2)
                4  : lambda1     (dimensionless tidal deformability of body 1)
                5  : lambda2     (dimensionless tidal deformability of body 2)
                6  : distance    (Mpc)
                7  : tc          (s, time of coalescence)
                8  : phic        (rad, reference orbital phase)
                9  : inclination (rad)
                --- extrinsic (used for detector projection) ---
                10 : polarization (rad)
                11 : ra           (rad)
                12 : dec          (rad)

Created on 2026-05-27

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

References
----------
IMRPhenomXAS  : García-Quirós et al. (2020), arXiv:2001.10914
NRTidalv3     : Abac et al. (2023), arXiv:2311.07456
kappa2T       : Dietrich et al. (2019), arXiv:1905.06011
"""

import torch
import torch.nn as nn

from sage.data.waveform.approximants.IMRPhenomXAS import IMRPhenomXAS
from sage.data.waveform.approximants.phenomx_data import (
    _NRTv3_s10, _NRTv3_s11, _NRTv3_s12,
    _NRTv3_s20, _NRTv3_s21, _NRTv3_s22,
    _NRTv3_s30, _NRTv3_s31, _NRTv3_s32,
    _NRTv3_alpha, _NRTv3_beta,
    _NRTv3_n_5over20, _NRTv3_n_5over21, _NRTv3_n_5over22, _NRTv3_n_5over23,
    _NRTv3_n_30, _NRTv3_n_31, _NRTv3_n_32, _NRTv3_n_33,
    _NRTv3_d_10, _NRTv3_d_11, _NRTv3_d_12,
)
from sage.data.waveform import taper
from sage.data.waveform import waveform_utils
from sage.core.config import get_cfg, get_data_cfg
from sage.core.torch import nudge_backward_


class IMRPhenomXAS_NRTidalv3(IMRPhenomXAS, nn.Module):
    """
    GPU-native batched IMRPhenomXAS_NRTidalv3 BNS/NSBH waveform generator.

    Adds NRTidalv3 tidal-phase corrections on top of the aligned-spin
    IMRPhenomXAS BBH backbone.  Implements the Sage ``forward()`` interface
    so it can be used as a drop-in replacement for IMRPhenomPv2 in the
    training loop.

    ``GRAPH_READY = True`` indicates that the full ``forward`` pass is
    compatible with ``torch.compile(fullgraph=True)``.

    Parameters
    ----------
    param_sampler : callable or None
    waveform_project : callable or None
    augment : callable or None
    """

    GRAPH_READY = True

    def __init__(self, param_sampler=None, waveform_project=None, augment=None):

        nn.Module.__init__(self)

        self.cfg      = get_cfg()
        self.data_cfg = get_data_cfg()

        self.signal_batch_size = int(self.cfg.batch_size * self.cfg.class_balance)

        f, f_ref = waveform_utils.get_freqs(
            self.data_cfg.signal_low_frequency_cutoff,
            self.data_cfg.sample_rate / 2.0,
            self.data_cfg.padded_length_in_s,
            self.signal_batch_size,
            self.cfg.device,
            self.cfg.dtype,
        )

        IMRPhenomXAS.__init__(self, f, f_ref)

        self.param_sampler    = param_sampler
        self.waveform_project = waveform_project
        self.augment          = augment

        self.param_names = [
            "mass1", "mass2",
            "spin1z", "spin2z",
            "lambda1", "lambda2",
            "distance", "tc", "coa_phase", "inclination",
            "polarization", "ra", "dec",
        ]

        if self.param_sampler is not None:
            get_idx = self.param_sampler.param_index
            self.req_idx = torch.tensor(
                [get_idx[key] for key in self.param_names],
                device=f.device,
                dtype=torch.int32,
            )
            self.param_sampler.req_idx = self.req_idx
            self.param_sampler._compile_batch_normaliser()
            self.param_sampler._compile_batch_standardiser()
            self.param_sampler.to(self.cfg.device)

        if self.waveform_project is not None:
            self.waveform_project.to(self.cfg.device)

    # ------------------------------------------------------------------
    # Sage training-loop interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, return_theta=False):
        all_theta  = self.param_sampler(self.B)
        req_theta  = all_theta[:, self.req_idx]

        hp, hc = self.get_hphc(req_theta)

        hf = self.waveform_project(
            hp, hc,
            ra=req_theta[:, -2],
            dec=req_theta[:, -1],
            polarization=req_theta[:, -3],
        )

        if self.augment:
            hf = self.augment(hf)

        normed_targets = self.param_sampler.standardise_from_batch(all_theta)
        targets = torch.cat(
            [normed_targets, torch.ones_like(normed_targets[:, :1])], dim=1
        )

        if return_theta:
            return hf, targets, all_theta
        return hf, targets

    # ------------------------------------------------------------------
    # Waveform generation
    # ------------------------------------------------------------------

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute FD plus and cross polarisations for a BNS/NSBH parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 13+)
            [m1, m2, chi1z, chi2z, lambda1, lambda2,
             distance, tc, phic, inclination, polarization, ra, dec]
        reproduce_lal : bool
            Skip tapering / tc / df normalisation to match raw LAL output.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, F), complex
        """
        raise NotImplementedError("get_hphc not yet implemented")

    # ------------------------------------------------------------------
    # NRTidalv3 — kappa2T (tidal coupling constant)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_kappa2T(m1_s, m2_s, lambda1, lambda2):
        """
        Dimensionless tidal coupling constant κ₂ᵀ.

        Source: XLALSimNRTunedTidesComputeKappa2T in LALSimNRTunedTides.c

        κ₂ᵀ = 3 [ Xb/Xa⁴ * Xa⁴*Xb * λ₁ + Xa/Xb⁴ * Xb⁴*Xa * λ₂ ]
            = 3 [ Xb * λ₁  *  Xa⁴  +  Xa * λ₂ * Xb⁴ ]

        (LAL form: kappa2T = 3/13 * [...] * (1 + 12*X/Y) * ... )
        Full expression below matches LALSimNRTunedTides.c line-for-line.

        Parameters
        ----------
        m1_s, m2_s : torch.Tensor, shape (B, 1)
            Component masses in seconds (m * G/c³).
        lambda1, lambda2 : torch.Tensor, shape (B, 1)
            Dimensionless tidal deformabilities.

        Returns
        -------
        kappa2T : torch.Tensor, shape (B, 1)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # NRTidalv3 — 7.5PN tidal phase coefficients
    # ------------------------------------------------------------------

    @staticmethod
    def compute_PN_tidal_coeffs(Xa, Xb):
        """
        Compute the 10 PN tidal coefficients used to constrain the NRTidalv3
        Padé approximant.

        Source: XLALSimNRTunedTidesSetFDTidalPhase_PN_Coeffs in
                LALSimNRTunedTides.c (7.5PN tidal phase coefficients,
                Vines, Flanagan & Hinderer 2011 + higher-order terms).

        Parameters
        ----------
        Xa, Xb : torch.Tensor, shape (B, 1)
            Mass fractions m1/M and m2/M.

        Returns
        -------
        PN_coeffs : torch.Tensor, shape (B, 10)
            Columns 0-4: body-A coefficients (c_Newt, c_1, c_3/2, c_2, c_5/2)
            Columns 5-9: body-B coefficients
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # NRTidalv3 — coefficient array
    # ------------------------------------------------------------------

    def compute_nrtv3_coeffs(self, Xa, M_s, lambda1, lambda2, PN_coeffs):
        """
        Build the 20-element NRTidalv3 coefficient array.

        Mirrors XLALSimNRTunedTidesSetFDTidalPhase_v3_Coeffs from
        LALSimNRTunedTides.c, vectorised over the batch dimension.

        Parameters
        ----------
        Xa : torch.Tensor, shape (B, 1)   — m1/M
        M_s : torch.Tensor, shape (B, 1)  — total mass in seconds
        lambda1, lambda2 : (B, 1)         — tidal deformabilities
        PN_coeffs : torch.Tensor, shape (B, 10)

        Returns
        -------
        coeffs : torch.Tensor, shape (B, 20)
            [s1, s2, s3, exp(s2*s3),
             kappaA, kappaB,
             n_5/2A, n_3A, d_1A,
             n_5/2B, n_3B, d_1B,
             n_1A, n_3/2A, n_2A, d_3/2A,
             n_1B, n_3/2B, n_2B, d_3/2B]
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # NRTidalv3 — tidal phase per frequency bin
    # ------------------------------------------------------------------

    @staticmethod
    def tidal_phase(f_Hz, M_s, nrtv3_coeffs, PN_coeffs):
        """
        Evaluate the NRTidalv3 tidal phase correction Ψ_tidal(f).

        Mirrors SimNRTunedTidesFDTidalPhase_v3 from LALSimNRTunedTides.c.
        Equation (27, 30) of Abac et al. (2023), arXiv:2311.07456.

        Includes the dynamic effective Love number enhancement (Eq. 27) and
        the per-body Padé approximant (Eq. 30).

        Parameters
        ----------
        f_Hz : torch.Tensor, shape (B, F)
            Frequency grid in Hz.
        M_s : torch.Tensor, shape (B, 1)
            Total mass in seconds.
        nrtv3_coeffs : torch.Tensor, shape (B, 20)
            Output of compute_nrtv3_coeffs.
        PN_coeffs : torch.Tensor, shape (B, 10)
            Output of compute_PN_tidal_coeffs.

        Returns
        -------
        psi_tidal : torch.Tensor, shape (B, F)
            Tidal phase correction in radians (negative, to be added to BBH phase).
        """
        raise NotImplementedError
