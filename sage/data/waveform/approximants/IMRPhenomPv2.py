#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : IMRPhenomPv2.py
Description     : Batched PyTorch port of the IMRPhenomPv2 precessing
                  frequency-domain waveform (LALSimIMRPhenomP.c),
                  built on the IMRPhenomD port.

Created on 2026-01-21 05:26:04

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

Licensing
---------
This file is part of Sage and is distributed under GPL-3.0-or-later.
Portions derive from the MIT-licensed JAX reference implementation by
Adam Coogan and Thomas Edwards; that notice is retained verbatim in
``THIRD_PARTY_LICENSE`` in this directory and must not be removed.  The MIT
terms cover those derived portions only.

"""


import torch

from sage.data.waveform.approximants import IMRPhenomD
from sage.core.interpolation import torch_natural_cubic_coeffs, torch_natural_cubic_interp
from sage.core.math import safe_sqrt

from sage.data.waveform import taper
from sage.data.waveform import waveform_utils
from sage.core.geometry import Window


class IMRPhenomPv2(IMRPhenomD.IMRPhenomD, torch.nn.Module):
    """
    Batched IMRPhenomPv2 precessing-spin waveform.

    forward() returns detector-frame strain and scaled parameter
    targets.

    Parameters
    ----------
    param_sampler : DistributionSampler
        Waveform parameter sampler.
    waveform_project : ConstantProjection
        Detector projection.
    augment : callable or None
        Augmentation applied to the projected strain (the SNR rescaler).
    window : Window
        Window geometry; sets the frequency grid and delta_tc placement.
    signal_batch_size : int
        Signals drawn per call.
    low_frequency_cutoff_in_hz : float
        Lowest frequency of the waveform grid.
    reference_frequency_in_hz : float
        Reference frequency of the waveform phase.
    targets : list of str
        Point-estimate targets, in head order.
    device : str or torch.device
        Device of the batches.
    dtype : torch.dtype
        Real dtype of the output; strain is its complex counterpart.
    append_per_det_targets : bool
        Append per-detector targets after the class column (masker
        only).
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

        self.signal_batch_size = signal_batch_size

        # Fixed frequency grid
        f = waveform_utils.get_freqs(
            low_frequency_cutoff_in_hz,
            window.sample_rate_in_hz / 2.0,
            window.padded_length_in_s,
            signal_batch_size,
            device,
            torch.float64,
        )
        f_ref = waveform_utils.reference_frequencies(
            reference_frequency_in_hz, signal_batch_size, device
        )
        IMRPhenomD.IMRPhenomD.__init__(self, f, f_ref, window.padding_length_in_s)
        self.sample_length_in_s = float(window.padded_length_in_s)
        self._out_cdtype = torch.promote_types(dtype, torch.complex64)

        self.param_sampler = param_sampler
        self.waveform_project = waveform_project
        self.augment = augment

        # Append per-detector targets after the class column
        self.append_per_det_targets = bool(append_per_det_targets)
        # Column of mchirp in the targets
        self.mc_pe_idx = list(targets).index("mchirp")

        param_index = self.param_sampler.param_index
        parameters = waveform_utils.PRECESSING_PARAMETERS
        self.source, self.target = waveform_utils.layout_columns(param_index, parameters, type(self).__name__)
        self.extras = waveform_utils.given_extras(param_index, parameters)
        # Columns of distance and delta_tc in all_theta
        self.dist_col = param_index["distance"]
        self.delta_tc_col = param_index["delta_tc"]

        self.param_sampler.compile_target_scaling(targets, dtype)

        self.param_sampler.to(device)
        self.waveform_project.to(device)

    @torch.no_grad()
    def forward(self, return_theta=False):
        """
        Sample precessing waveforms and return detector-frame strain.

        Parameters
        ----------
        return_theta : bool
            Also return the raw parameter batch.

        Returns
        -------
        hf : torch.Tensor, shape (B, D, F)
            Detector-frame strain for each detector.
        targets : torch.Tensor, shape (B, n_params + 1 [+ 2 D])
            Standardised parameter targets with a signal-label column of
            ones, then per-detector delta_tc and scaled mchirp if
            append_per_det_targets.
        all_theta : torch.Tensor, shape (B, total_params)
            Full raw parameter batch (only when return_theta=True).
        """
        all_theta = self.param_sampler(self.B)
        theta = waveform_utils.to_layout(all_theta, self.source, self.target)

        hp, hc = self.get_hphc(theta)
        hp = hp.to(self._out_cdtype)
        hc = hc.to(self._out_cdtype)

        proj = self.waveform_project(
            hp,
            hc,
            ra=theta[:, -2],
            dec=theta[:, -1],
            polarization=theta[:, -3],
            return_delay=self.append_per_det_targets,
        )
        if self.append_per_det_targets:
            hf, dt = proj  # dt: (B, D) projection delays in s
        else:
            hf = proj

        if self.augment:
            # hf * scale is the strain of a source at distance / scale
            hf, scale = self.augment(hf)
            all_theta[:, self.dist_col] = (
                all_theta[:, self.dist_col] / scale.to(all_theta.dtype)
            )

        normed_targets = self.param_sampler.scale_targets(all_theta)
        targets = torch.cat(
            [normed_targets, torch.ones_like(normed_targets[:, :1])], dim=1
        )

        # Per-detector targets after the class column: delta_tc plus the
        # projection delay in s, then the scaled mchirp of each detector
        if self.append_per_det_targets:
            D = dt.shape[1]
            per_det_delta_tc = all_theta[:, self.delta_tc_col].unsqueeze(1) + dt
            per_det_mc = normed_targets[:, self.mc_pe_idx : self.mc_pe_idx + 1].expand(
                -1, D
            )
            targets = torch.cat([targets, per_det_delta_tc, per_det_mc], dim=1)

        if return_theta:
            return hf, targets, all_theta
        else:
            return hf, targets

    # Source: PhenomPCore (LALSimIMRPhenomP.c).
    def get_hphc(self, theta, reproduce_lal=False):
        """
        FD plus and cross polarisations of a parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 17)
            Parameters in the waveform_utils.WAVEFORM_PARAMETERS layout.
        reproduce_lal : bool
            If True, skip FD tapering, delta_tc shift and df
            normalisation and keep LAL's band (lal_band), so the output
            compares directly with LAL's.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_frequency_bins), complex128
        """
        # m1=0, m2=1, s1x=2, s1y=3, s1z=4, s2x=5, s2y=6,
        # s2z=7, dist_mpc=8, delta_tc=9, phiRef=10, incl=11
        theta = self.order_bodies(waveform_utils.precessing_layout(theta, self.extras))

        derived = self.compute_derived_parameters(theta)

        # chi1_l=0, chi2_l=1, chip=2, thetaJN=3,
        # alpha0=4, phi_aligned=5, zeta_polariz=6
        converted_spins = self.convert_spins(theta, derived)

        # Twice the orbital phase phi_aligned
        phic = 2 * converted_spins[:, 5:6]

        angcoeffs, alphaNNLOoffset, epsilonNNLOoffset = self.compute_pv2_coeffs(
            theta, derived, converted_spins
        )

        Y2 = self.compute_spin_weighted_Y(converted_spins)

        # IMRPhenomD assumes that m1 >= m2
        theta_swapped = torch.cat(
            [
                theta[:, 0:1],
                theta[:, 1:2],
                converted_spins[:, 1:2],
                converted_spins[:, 0:1],
                theta[:, 8:9],
                phic,
            ],
            dim=1,
        )

        phd_derived = super().compute_derived_parameters(theta_swapped)

        coeffs = super().get_coeffs(
            converted_spins[:, 1:2], converted_spins[:, 0:1], phd_derived[:, 3:4]
        )

        f_Ms, fx_Ms, fcut_true, trans_fs = self.get_derived_freqs(
            theta_swapped,
            derived,
            phd_derived,
            coeffs,
            converted_spins,
        )

        hPhenomD, _ = self.PhenomPOneFrequency(
            self.f,
            f_Ms,
            fx_Ms,
            theta_swapped,
            phd_derived,
            coeffs,
            trans_fs,
            fcut_true,
        )

        hp, hc = self.PhenomPCoreTwistUp(
            f_Ms,
            hPhenomD,
            derived[:, 1:2],
            converted_spins[:, 0:1],  # chi1_l, smaller body
            converted_spins[:, 1:2],  # chi2_l, larger body
            converted_spins[:, 2:3],
            angcoeffs,
            Y2,
            alphaNNLOoffset - converted_spins[:, 4:5],
            epsilonNNLOoffset,
        )

        hp, hc = self.correct_time_and_phase(
            hp,
            hc,
            theta_swapped,
            derived,
            phd_derived,
            trans_fs,
            fx_Ms,
            coeffs,
            fcut_true,
        )

        # Rotate h+ and hx by 2 zeta
        c2z = torch.cos(2 * converted_spins[:, 6:7])
        s2z = torch.sin(2 * converted_spins[:, 6:7])
        final_hp = c2z * hp + s2z * hc
        final_hc = c2z * hc - s2z * hp

        if not reproduce_lal:
            # Frequency domain tapering
            _taper = taper.fd_taper(
                f=self.f,
                f_min=self.f[0, 0].item(),
                f_cut=fcut_true,
                df=self.df,
            )
            final_hp *= _taper
            final_hc *= _taper

            # Phase shift equivalent to a delta_tc time shift
            final_hp, final_hc = self.apply_delta_tc(final_hp, final_hc, theta[:, 9:10])

            # Continuous to discrete Fourier normalisation
            final_hp *= self.df
            final_hc *= self.df
        else:
            final_hp, final_hc = self.lal_band(final_hp, final_hc, fcut_true)

        # Zeros from zero frequency to the starting frequency
        final_hp, final_hc = self.pad_missing_frequencies(final_hp, final_hc)

        return final_hp, final_hc


    # Source: PhenomPCore (LALSimIMRPhenomP.c).
    @staticmethod
    def order_bodies(theta):
        """
        Swap bodies, masses and spin vectors, where mass1 < mass2.

        PhenomPCore enforces m2 >= m1; this port takes mass1 >= mass2
        and assigns the bodies in convert_spins.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 12)

        Returns
        -------
        torch.Tensor, shape (B, 12)
        """
        swapped = theta[:, [1, 0, 5, 6, 7, 2, 3, 4, 8, 9, 10, 11]]
        return torch.where(theta[:, 0:1] < theta[:, 1:2], swapped, theta)

    def pad_missing_frequencies(self, hp, hc):
        """
        Zero-pad hp and hc from zero frequency to the grid start.

        Parameters
        ----------
        hp, hc : torch.Tensor, shape (B, n_active)
            Polarisations on the frequency grid.

        Returns
        -------
        hp_pad, hc_pad : torch.Tensor, shape (B, n_pad + n_active)
            Zero-padded polarisations.
        """
        hp_pad = torch.zeros_like(self.hp_buffer)
        hc_pad = torch.zeros_like(self.hc_buffer)
        hp_pad[:, self.n_pad :] = hp
        hc_pad[:, self.n_pad :] = hc

        return hp_pad, hc_pad

    # Source: PhenomPCore (LALSimIMRPhenomP.c).
    def compute_derived_parameters(self, theta):
        """
        Mass-derived quantities of a parameter batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, N)
            Columns 0 and 1 are mass1 and mass2 in solar masses.

        Returns
        -------
        derived : torch.Tensor, shape (B, 4)
            Total mass M in solar masses, eta, q = m1 / m2 >= 1, and M
            in seconds.
        """
        M = theta[:, 1:2] + theta[:, 0:1]
        eta = theta[:, 1:2] * theta[:, 0:1] / (M * M)
        q = theta[:, 0:1] / theta[:, 1:2]  # q >= 1
        M_s = (theta[:, 1:2] + theta[:, 0:1]) * self.GM  # m_sec
        # if (eta > 0.25 || q < 1.0), nudge eta to 0.25 and q to 1
        eta.clamp_(max=0.25)
        q.clamp_(min=1.0)

        return torch.cat([M, eta, q, M_s], dim=1)

    # Source: PhenomPCore (LALSimIMRPhenomP.c).
    def compute_pv2_coeffs(self, theta, derived, converted_spins):
        """
        NNLO angle coefficients and their offsets at f_ref.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 12)
            Precessing layout; masses in columns 0 and 1.
        derived : torch.Tensor, shape (B, 4)
            Output of compute_derived_parameters.
        converted_spins : torch.Tensor, shape (B, 7)
            Output of convert_spins.

        Returns
        -------
        angcoeffs : torch.Tensor, shape (B, 10)
            alphacoeff1 to 5, then epsiloncoeff1 to 5.
        alphaNNLOoffset, epsilonNNLOoffset : torch.Tensor, shape (B, 1)
            Offsets of alpha and epsilon at f_ref.
        """
        # Effective aligned spin
        chi_eff = (
            theta[:, 1:2] * converted_spins[:, 0:1]
            + theta[:, 0:1] * converted_spins[:, 1:2]
        ) / derived[:, 0:1]

        # Dimensionless aligned spin of the largest BH
        chil = (1.0 + derived[:, 2:3]) / derived[:, 2:3] * chi_eff

        piM = self.PI * derived[:, 3:4]

        omega_ref = piM * self.f_ref
        logomega_ref = torch.log(omega_ref)
        omega_ref_cbrt = (piM * self.f_ref) ** self.ONE_BY_THREE
        omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

        # Next-to-next-to leading order PN coefficients for Euler angles
        # alpha and epsilon
        angcoeffs = self.ComputeNNLOanglecoeffs(
            derived[:, 2:3],
            chil,
            converted_spins[:, 2:3],
        )

        # Compute the offsets due to the choice of integration constant
        # in alpha and epsilon PN formula
        alphaNNLOoffset = (
            angcoeffs[:, 0:1] / omega_ref
            + angcoeffs[:, 1:2] / omega_ref_cbrt2
            + angcoeffs[:, 2:3] / omega_ref_cbrt
            + angcoeffs[:, 3:4] * logomega_ref
            + angcoeffs[:, 4:5] * omega_ref_cbrt
        )

        epsilonNNLOoffset = (
            angcoeffs[:, 5:6] / omega_ref
            + angcoeffs[:, 6:7] / omega_ref_cbrt2
            + angcoeffs[:, 7:8] / omega_ref_cbrt
            + angcoeffs[:, 8:9] * logomega_ref
            + angcoeffs[:, 9:10] * omega_ref_cbrt
        )

        return angcoeffs, alphaNNLOoffset, epsilonNNLOoffset

    # Source: PhenomPCore (LALSimIMRPhenomP.c).
    def compute_spin_weighted_Y(self, converted_spins):
        """
        l = 2 spherical harmonics of spin weight -2 at thetaJN.

        Parameters
        ----------
        converted_spins : torch.Tensor, shape (B, 7)
            Output of convert_spins; column 3 is thetaJN.

        Returns
        -------
        Y2 : torch.Tensor, shape (B, 5)
            Y2m2, Y2m1, Y20, Y21 and Y22.
        """
        Y2m2 = self.SpinWeightedY(converted_spins[:, 3:4], 0, -2, 2, -2)
        Y2m1 = self.SpinWeightedY(converted_spins[:, 3:4], 0, -2, 2, -1)
        Y20 = self.SpinWeightedY(converted_spins[:, 3:4], 0, -2, 2, -0)
        Y21 = self.SpinWeightedY(converted_spins[:, 3:4], 0, -2, 2, 1)
        Y22 = self.SpinWeightedY(converted_spins[:, 3:4], 0, -2, 2, 2)
        return torch.cat([Y2m2, Y2m1, Y20, Y21, Y22], dim=1)

    def get_derived_freqs(
        self,
        theta_swapped,
        derived,
        phd_derived,
        coeffs,
        converted_spins,
    ):
        """
        Geometric and physical frequencies of a parameter batch.

        Parameters
        ----------
        theta_swapped : torch.Tensor, shape (B, 6)
            Reduced parameters with m1 >= m2, for IMRPhenomD.
        derived : torch.Tensor, shape (B, 4)
            Output of compute_derived_parameters.
        phd_derived : torch.Tensor, shape (B, 4)
            Output of IMRPhenomD.compute_derived_parameters.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients from get_coeffs.
        converted_spins : torch.Tensor, shape (B, 7)
            Output of convert_spins; column 2 is chip.

        Returns
        -------
        f_Ms : torch.Tensor, shape (B, F)
            Geometric frequencies of the grid.
        fx_Ms : torch.Tensor, shape (B, 8)
            Geometric fRef, f1, f2, f3, f4, fRD, fdamp and fmid.
        fcut_true : torch.Tensor, shape (B, 1)
            Cutoff frequency fCut in Hz.
        trans_fs : torch.Tensor, shape (B, 6)
            f1, f2, f3, f4, fRD and fdamp in Hz.
        """
        # {f1, f2, f3, f4, f_RD, f_damp}
        trans_fs = self.phP_get_transition_frequencies(
            theta_swapped,
            coeffs[:, 5:6],
            coeffs[:, 6:7],
            converted_spins[:, 2:3],
            derived,
            phd_derived,
        )

        fcut_true = super().get_fcut_true(derived[:, 3:4])

        f1_Ms = trans_fs[:, 0:1] * derived[:, 3:4]
        f2_Ms = trans_fs[:, 1:2] * derived[:, 3:4]
        f3_Ms = trans_fs[:, 2:3] * derived[:, 3:4]
        f4_Ms = trans_fs[:, 3:4] * derived[:, 3:4]
        f_Ms = self.f * derived[:, 3:4]
        fref_Ms = self.f_ref * derived[:, 3:4]
        f_RD_Ms = trans_fs[:, 4:5] * derived[:, 3:4]
        f_damp_Ms = trans_fs[:, 5:6] * derived[:, 3:4]
        # Middle of the three amplitude collocation points
        fmid_Ms = ((trans_fs[:, 2:3] + trans_fs[:, 3:4]) / 2) * derived[:, 3:4]
        fx_Ms = torch.cat(
            [fref_Ms, f1_Ms, f2_Ms, f3_Ms, f4_Ms, f_RD_Ms, f_damp_Ms, fmid_Ms], dim=1
        )

        return f_Ms, fx_Ms, fcut_true, trans_fs

    # Source: PhenomPCore (LALSimIMRPhenomP.c).
    def correct_time_and_phase(
        self,
        hp,
        hc,
        theta_swapped,
        derived,
        phd_derived,
        trans_fs,
        fx_Ms,
        coeffs,
        fcut_true,
    ):
        """
        Correct phasing so we coalesce at t=0.

        The time correction is t(f_final) = 1/(2pi) dphi/df (f_final),
        from a spline of the phasing on 10 points in
        [0.8 f_final, min(1.2 f_final, fCut)].

        Parameters
        ----------
        hp, hc : torch.Tensor, shape (B, F)
            Polarisations before correction.
        theta_swapped : torch.Tensor, shape (B, 6)
            Reduced parameters with m1 >= m2.
        derived : torch.Tensor, shape (B, 4)
            Output of compute_derived_parameters.
        phd_derived : torch.Tensor, shape (B, 4)
            Output of IMRPhenomD.compute_derived_parameters.
        trans_fs : torch.Tensor, shape (B, 6)
            f1, f2, f3, f4, fRD and fdamp in Hz.
        fx_Ms : torch.Tensor, shape (B, 8)
            Geometric fRef, f1, f2, f3, f4, fRD, fdamp and fmid.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients.
        fcut_true : torch.Tensor, shape (B, 1)
            Cutoff frequency fCut in Hz.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, F)
            Corrected polarisations.
        """
        # Set up fixed frequency grid around ringdown frequency
        n_fixed = 10
        f_final = trans_fs[:, 4:5]
        freqs_fixed_start = 0.8 * f_final
        freqs_fixed_stop = torch.minimum(1.2 * f_final, self.fM_CUT / derived[:, 3:4])
        delta_freqs_fixed = (freqs_fixed_stop - freqs_fixed_start) / (n_fixed - 1)
        steps = torch.arange(n_fixed, device=self.f.device, dtype=self.f.dtype)
        freqs_fixed = freqs_fixed_start + steps * delta_freqs_fixed

        _, phase_fixed = self.PhenomPOneFrequency(
            freqs_fixed,
            freqs_fixed * derived[:, 3:4],
            fx_Ms,
            theta_swapped,
            phd_derived,
            coeffs,
            trans_fs,
            fcut_true,
        )

        # phasing = -phPhenom; ignore alpha and epsilon contributions
        phasing = -phase_fixed
        M = torch_natural_cubic_coeffs(freqs_fixed, phasing)
        dphi_df = torch_natural_cubic_interp(
            f_final, freqs_fixed, phasing, M, derivative=True
        )
        t_corr_fixed = dphi_df / (2 * self.PI)

        phase_corr = torch.polar(
            torch.ones_like(self.f), -2 * self.PI * self.f * t_corr_fixed
        )
        return hp * phase_corr, hc * phase_corr

    # Source: XLALSimIMRPhenomPCalculateModelParametersFromSourceFrame
    # (LALSimIMRPhenomP.c).
    def convert_spins(self, theta, derived):
        """
        Map the source-frame spins into IMRPhenomP intrinsic parameters.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 12)
            mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y,
            spin2z, distance, delta_tc, coa_phase and inclination,
            with mass1 >= mass2: chi1_l, of the smaller body, is
            spin2z.
        derived : torch.Tensor, shape (B, 4)
            Output of compute_derived_parameters.

        Returns
        -------
        converted : torch.Tensor, shape (B, 7)
            chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned and
            zeta_polariz.
        """
        m1_2 = theta[:, 1:2] * theta[:, 1:2]
        m2_2 = theta[:, 0:1] * theta[:, 0:1]

        # From the components in the source frame, determine chi1_l,
        # chi2_l, chip and phi_aligned, and the spherical angles of J

        # Aligned spins
        chi1_l = theta[:, 7:8]  # Dimensionless aligned spin on BH 1
        chi2_l = theta[:, 4:5]  # Dimensionless aligned spin on BH 2

        # Magnitude of the spin projections in the orbital plane
        S1_perp = m1_2 * torch.sqrt(theta[:, 5:6] ** 2 + theta[:, 6:7] ** 2)
        S2_perp = m2_2 * torch.sqrt(theta[:, 2:3] ** 2 + theta[:, 3:4] ** 2)

        A1 = self.TWO + (3 * theta[:, 0:1]) / (2 * theta[:, 1:2])
        A2 = self.TWO + (3 * theta[:, 1:2]) / (2 * theta[:, 0:1])
        ASp1 = A1 * S1_perp
        ASp2 = A2 * S2_perp
        num = torch.maximum(ASp1, ASp2)
        den = torch.where(theta[:, 0:1] > theta[:, 1:2], A2 * m2_2, A1 * m1_2)
        # chip = max(A1 Sp1, A2 Sp2) / (A_i m_i^2) for i index of
        # larger BH
        chip = num / den

        # Compute L, J0 and orientation angles
        m_sec = derived[:, 0:1] * self.GM
        piM = self.PI * m_sec
        v_ref = (piM * self.f_ref) ** self.ONE_BY_THREE
        L0 = (
            derived[:, 0:1]
            * derived[:, 0:1]
            * self.L2PNR(
                v_ref,
                derived[:, 1:2],
            )
        )
        J0x_sf = m1_2 * theta[:, 5:6] + m2_2 * theta[:, 2:3]
        J0y_sf = m1_2 * theta[:, 6:7] + m2_2 * theta[:, 3:4]
        J0z_sf = L0 + m1_2 * theta[:, 7:8] + m2_2 * theta[:, 4:5]
        J0 = torch.sqrt(J0x_sf * J0x_sf + J0y_sf * J0y_sf + J0z_sf * J0z_sf)

        thetaJ_sf = torch.arccos(J0z_sf / J0)
        # MAX_TOL_ATAN
        tol = 1.0e-15
        phiJ_sf = torch.where(
            (J0x_sf.abs() < tol) & (J0y_sf.abs() < tol),
            self.PI / 2.0 - theta[:, 10:11],  # aligned spin limit
            torch.arctan2(J0y_sf, J0x_sf),
        )
        phi_aligned = -phiJ_sf

        # First we determine kappa, from the components of N in the
        # source frame
        Nx_sf = torch.sin(theta[:, 11:12]) * torch.cos(self.PI / 2.0 - theta[:, 10:11])
        Ny_sf = torch.sin(theta[:, 11:12]) * torch.sin(self.PI / 2.0 - theta[:, 10:11])
        Nz_sf = torch.cos(theta[:, 11:12])

        tmp_x = Nx_sf
        tmp_y = Ny_sf
        tmp_z = Nz_sf

        tmp_x, tmp_y, tmp_z = self.ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)

        kappa = -torch.where(
            (tmp_y.abs() < tol) & (tmp_x.abs() < tol),
            self.ZERO,
            torch.arctan2(tmp_y, tmp_x),
        )

        # Then we determine alpha0, by rotating LN
        tmp_x, tmp_y, tmp_z = self.ZERO, self.ZERO, self.ONE
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

        alpha0 = torch.where(
            (tmp_x.abs() < tol) & (tmp_y.abs() < tol),
            self.PI,  # this is the aligned spin case
            torch.arctan2(tmp_y, tmp_x),
        )

        # Finally we determine thetaJ, by rotating N
        tmp_x, tmp_y, tmp_z = Nx_sf, Ny_sf, Nz_sf
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)
        Nx_Jf, Nz_Jf = tmp_x, tmp_z
        thetaJN = torch.arccos(Nz_Jf)

        # Finally, we need to redefine the polarizations: the P,Q,N
        # triad of PhenomP and the X,Y,N triad of LAL differ by a
        # rotation around N by an angle zeta

        Xx_sf = -torch.cos(theta[:, 11:12]) * torch.sin(theta[:, 10:11])
        Xy_sf = -torch.cos(theta[:, 11:12]) * torch.cos(theta[:, 10:11])
        Xz_sf = torch.sin(theta[:, 11:12])
        tmp_x, tmp_y, tmp_z = Xx_sf, Xy_sf, Xz_sf
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

        # Now the tmp_a are the components of X in the J frame; we need
        # the polar angle of that vector in the P,Q basis.
        # P = NxJ/|NxJ|, N in the (pos x)z half plane of the J frame
        PArunx_Jf = self.ZERO
        PAruny_Jf = -self.ONE
        PArunz_Jf = self.ZERO

        # Q = NxP
        QArunx_Jf = Nz_Jf
        QAruny_Jf = self.ZERO
        QArunz_Jf = -Nx_Jf

        XdotPArun = tmp_x * PArunx_Jf + tmp_y * PAruny_Jf + tmp_z * PArunz_Jf
        XdotQArun = tmp_x * QArunx_Jf + tmp_y * QAruny_Jf + tmp_z * QArunz_Jf

        zeta_polariz = torch.arctan2(XdotQArun, XdotPArun)

        return torch.cat(
            [
                chi1_l,
                chi2_l,
                chip,
                thetaJN,
                alpha0,
                phi_aligned,
                zeta_polariz,
            ],
            dim=1,
        )

    # Source: L2PNR (LALSimIMRPhenomP.c).
    def L2PNR(self, v, eta):
        """
        Simple 2PN version of the orbital angular momentum L.

        Without any spin terms, expressed as a function of v.

        Parameters
        ----------
        v : torch.Tensor, shape (B, 1) or (B, F)
            Cubic root of (Pi * Frequency (geometric)).
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.

        Returns
        -------
        L : torch.Tensor, same shape as v
            Orbital angular momentum for unit total mass.
        """
        eta2 = eta * eta
        x = v * v
        x2 = x * x
        return (
            eta
            * (
                self.ONE
                + (self.THREE_BY_TWO + eta / self.SIX) * x
                + (3.375 - (19.0 * eta) / self.EIGHT - eta2 / self.TWENTY_FOUR) * x2
            )
        ) / torch.sqrt(x)

    # Source: ROTATEZ (LALSimIMRPhenomP.c).
    @staticmethod
    def ROTATEZ(angle, x, y, z):
        """Rotate vector (x, y, z) about the z-axis by angle."""
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        return x * ca - y * sa, x * sa + y * ca, z

    # Source: ROTATEY (LALSimIMRPhenomP.c).
    @staticmethod
    def ROTATEY(angle, x, y, z):
        """Rotate vector (x, y, z) about the y-axis by angle."""
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        return x * ca + z * sa, y, -x * sa + z * ca

    # Source: ComputeNNLOanglecoeffs (LALSimIMRPhenomP.c).
    def ComputeNNLOanglecoeffs(self, q, chil, chip):
        """
        NNLO PN coefficients for Euler angles alpha and epsilon.

        Parameters
        ----------
        q : torch.Tensor, shape (B, 1)
            Mass-ratio (convention q>1).
        chil : torch.Tensor, shape (B, 1)
            Dimensionless aligned spin of the largest BH.
        chip : torch.Tensor, shape (B, 1)
            Dimensionless spin component in the orbital plane.

        Returns
        -------
        angcoeffs : torch.Tensor, shape (B, 10)
            alphacoeff1 to 5, then epsiloncoeff1 to 5.
        """
        m2 = q / (1.0 + q)
        m1 = self.ONE / (1.0 + q)
        dm = m1 - m2
        mtot = self.ONE
        eta = m1 * m2

        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        mtot2 = mtot * mtot
        mtot4 = mtot2 * mtot2
        mtot6 = mtot4 * mtot2
        mtot8 = mtot6 * mtot2
        chil2 = chil * chil
        chip2 = chip * chip
        chip4 = chip2 * chip2
        dm2 = dm * dm
        dm3 = dm2 * dm
        m2_2 = m2 * m2
        m2_3 = m2_2 * m2
        m2_4 = m2_3 * m2
        m2_5 = m2_4 * m2
        m2_6 = m2_5 * m2
        m2_7 = m2_6 * m2
        m2_8 = m2_7 * m2

        alphacoeff1 = -0.18229166666666666 - (5 * dm) / (64.0 * m2)

        alphacoeff2 = (-15 * dm * m2 * chil) / (128.0 * mtot2 * eta) - (
            35 * m2_2 * chil
        ) / (128.0 * mtot2 * eta)

        alphacoeff3 = (
            -1.7952473958333333
            - (4555 * dm) / (7168.0 * m2)
            - (15 * chip2 * dm * m2_3) / (128.0 * mtot4 * eta2)
            - (35 * chip2 * m2_4) / (128.0 * mtot4 * eta2)
            - (515 * eta) / 384.0
            - (15 * dm2 * eta) / (256.0 * m2_2)
            - (175 * dm * eta) / (256.0 * m2)
        )

        alphacoeff4 = (
            -(35 * self.PI) / 48.0
            - (5 * dm * self.PI) / (16.0 * m2)
            + (5 * dm2 * chil) / (16.0 * mtot2)
            + (5 * dm * m2 * chil) / (3.0 * mtot2)
            + (2545 * m2_2 * chil) / (1152.0 * mtot2)
            - (5 * chip2 * dm * m2_5 * chil) / (128.0 * mtot6 * eta3)
            - (35 * chip2 * m2_6 * chil) / (384.0 * mtot6 * eta3)
            + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
            + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
        )

        alphacoeff5 = (
            4.318908476114694
            + (27895885 * dm) / (2.1676032e7 * m2)
            - (15 * chip4 * dm * m2_7) / (512.0 * mtot8 * eta4)
            - (35 * chip4 * m2_8) / (512.0 * mtot8 * eta4)
            - (485 * chip2 * dm * m2_3) / (14336.0 * mtot4 * eta2)
            + (475 * chip2 * m2_4) / (6144.0 * mtot4 * eta2)
            + (15 * chip2 * dm2 * m2_2) / (256.0 * mtot4 * eta)
            + (145 * chip2 * dm * m2_3) / (512.0 * mtot4 * eta)
            + (575 * chip2 * m2_4) / (1536.0 * mtot4 * eta)
            + (39695 * eta) / 86016.0
            + (1615 * dm2 * eta) / (28672.0 * m2_2)
            - (265 * dm * eta) / (14336.0 * m2)
            + (955 * eta2) / 576.0
            + (15 * dm3 * eta2) / (1024.0 * m2_3)
            + (35 * dm2 * eta2) / (256.0 * m2_2)
            + (2725 * dm * eta2) / (3072.0 * m2)
            - (15 * dm * m2 * self.PI * chil) / (16.0 * mtot2 * eta)
            - (35 * m2_2 * self.PI * chil) / (16.0 * mtot2 * eta)
            + (15 * chip2 * dm * m2_7 * chil2) / (128.0 * mtot8 * eta4)
            + (35 * chip2 * m2_8 * chil2) / (128.0 * mtot8 * eta4)
            + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
            + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
            + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
        )

        epsiloncoeff1 = -0.18229166666666666 - (5 * dm) / (64.0 * m2)
        epsiloncoeff2 = (-15 * dm * m2 * chil) / (128.0 * mtot2 * eta) - (
            35 * m2_2 * chil
        ) / (128.0 * mtot2 * eta)
        epsiloncoeff3 = (
            -1.7952473958333333
            - (4555 * dm) / (7168.0 * m2)
            - (515 * eta) / 384.0
            - (15 * dm2 * eta) / (256.0 * m2_2)
            - (175 * dm * eta) / (256.0 * m2)
        )
        epsiloncoeff4 = (
            -(35 * self.PI) / 48.0
            - (5 * dm * self.PI) / (16.0 * m2)
            + (5 * dm2 * chil) / (16.0 * mtot2)
            + (5 * dm * m2 * chil) / (3.0 * mtot2)
            + (2545 * m2_2 * chil) / (1152.0 * mtot2)
            + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
            + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
        )
        epsiloncoeff5 = (
            4.318908476114694
            + (27895885 * dm) / (2.1676032e7 * m2)
            + (39695 * eta) / 86016.0
            + (1615 * dm2 * eta) / (28672.0 * m2_2)
            - (265 * dm * eta) / (14336.0 * m2)
            + (955 * eta2) / 576.0
            + (15 * dm3 * eta2) / (1024.0 * m2_3)
            + (35 * dm2 * eta2) / (256.0 * m2_2)
            + (2725 * dm * eta2) / (3072.0 * m2)
            - (15 * dm * m2 * self.PI * chil) / (16.0 * mtot2 * eta)
            - (35 * m2_2 * self.PI * chil) / (16.0 * mtot2 * eta)
            + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
            + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
            + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
        )

        angcoeffs = torch.cat(
            [
                alphacoeff1,
                alphacoeff2,
                alphacoeff3,
                alphacoeff4,
                alphacoeff5,
                epsiloncoeff1,
                epsiloncoeff2,
                epsiloncoeff3,
                epsiloncoeff4,
                epsiloncoeff5,
            ],
            dim=1,
        )

        return angcoeffs

    # Source: XLALSpinWeightedSphericalHarmonic (SphericalHarmonics.c).
    def SpinWeightedY(self, theta, phi, s, l, m):
        """
        Spin-weighted spherical harmonic, for s = -2 and l = 2 only.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 1)
            Polar angle in radians.
        phi : float or torch.Tensor
            Azimuthal angle in radians.
        s : int
            Spin weight; must be -2.
        l : int
            Degree; must be 2.
        m : int
            Order; must satisfy |m| <= l.

        Returns
        -------
        Y : torch.Tensor, shape (B, 1)
            Complex spin-weighted spherical harmonic.

        Raises
        ------
        ValueError
            If s = -2, l = 2 and |m| > 2.
        """
        if s == -2:
            if l == 2:
                if m == -2:
                    fac = (
                        torch.sqrt(self.FIVE / (64.0 * self.PI))
                        * (1.0 - torch.cos(theta))
                        * (1.0 - torch.cos(theta))
                    )
                elif m == -1:
                    fac = (
                        torch.sqrt(self.FIVE / (16.0 * self.PI))
                        * torch.sin(theta)
                        * (1.0 - torch.cos(theta))
                    )
                elif m == 0:
                    fac = (
                        torch.sqrt(self.FIFTEEN / (32.0 * self.PI))
                        * torch.sin(theta)
                        * torch.sin(theta)
                    )
                elif m == 1:
                    fac = (
                        torch.sqrt(self.FIVE / (16.0 * self.PI))
                        * torch.sin(theta)
                        * (1.0 + torch.cos(theta))
                    )
                elif m == 2:
                    fac = (
                        torch.sqrt(self.FIVE / (64.0 * self.PI))
                        * (1.0 + torch.cos(theta))
                        * (1.0 + torch.cos(theta))
                    )
                else:
                    raise ValueError(
                        f"Invalid mode s={s}, l={l}, m={m} require |m| <= l"
                    )

        return fac * torch.exp(self.ONE_J * m * phi)

    # Source: ComputeIMRPhenDPhaseConnectionCoefficients and fmaxCalc
    # (LALSimIMRPhenomD_internals.c).
    def phP_get_transition_frequencies(
        self,
        theta,
        gamma2,
        gamma3,
        chip,
        derived,
        phd_derived,
    ):
        """
        Transition, ringdown and damping frequencies, in Hz.

        The ringdown and damping frequencies use the final spin with the
        in-plane spin chip.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 6)
            Reduced parameters with m1 >= m2.
        gamma2, gamma3 : torch.Tensor, shape (B, 1)
            Phenom coefficients gamma2 and gamma3.
        chip : torch.Tensor, shape (B, 1)
            Dimensionless spin in the orbital plane.
        derived : torch.Tensor, shape (B, 4)
            Output of compute_derived_parameters.
        phd_derived : torch.Tensor, shape (B, 4)
            Output of IMRPhenomD.compute_derived_parameters.

        Returns
        -------
        trans_fs : torch.Tensor, shape (B, 6)
            f1, f2, f3, f4, fRD and fdamp.
        """
        # Convention m1 >= m2
        f_RD, f_damp = self.phP_get_fRD_fdamp(
            theta,
            derived,
            phd_derived,
            chip,
        )

        # Phase transition frequencies
        f1 = 0.018 / derived[:, 3:4]
        f2 = 0.5 * f_RD

        # Amplitude transition frequencies
        f3 = 0.014 / derived[:, 3:4]
        f4_gammaneg_gtr_1 = torch.abs(f_RD + (-f_damp * gamma3) / gamma2)
        f4_gammaneg_less_1 = torch.abs(
            f_RD
            + (f_damp * (-1 + torch.sqrt(self.ONE - (gamma2) ** 2.0)) * gamma3) / gamma2
        )

        # If gamma2 >= 1 then set the square root term to zero
        f4 = torch.where(gamma2 >= 1, f4_gammaneg_gtr_1, f4_gammaneg_less_1)

        return torch.cat([f1, f2, f3, f4, f_RD, f_damp], dim=1)

    # Source: fring and fdamp (LALSimIMRPhenomD_internals.c).
    def phP_get_fRD_fdamp(self, theta, derived, phd_derived, chip):
        """
        Ringdown and damping frequencies, in Hz.

        The QNM tables are interpolated at the final spin with the
        in-plane spin chip, and divided by 1 - Erad.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 6)
            Reduced parameters with m1 >= m2; columns 2 and 3 are chi1
            and chi2.
        derived : torch.Tensor, shape (B, 4)
            Output of compute_derived_parameters.
        phd_derived : torch.Tensor, shape (B, 4)
            Output of IMRPhenomD.compute_derived_parameters; column 3
            is eta.
        chip : torch.Tensor, shape (B, 1)
            Dimensionless spin in the orbital plane.

        Returns
        -------
        fRD, fdamp : torch.Tensor, shape (B, 1)
            Ringdown and damping frequencies in Hz.
        """
        # Convention m1 >= m2
        finspin = self.FinalSpin_inplane(theta, derived, chip)
        Erad = self.EradRational0815(
            phd_derived[:, 3:4],
            theta[:, 2:3],
            theta[:, 3:4],
        )

        fRD = torch_natural_cubic_interp(
            finspin, self.QNMData_a, self.QNMData_fRD, self.QNMData_fRD_M
        )
        fdamp = torch_natural_cubic_interp(
            finspin, self.QNMData_a, self.QNMData_fdamp, self.QNMData_fdamp_M
        )

        factor = 1.0 / (1.0 - Erad)
        fRD *= factor
        fdamp *= factor

        return fRD / derived[:, 3:4], fdamp / derived[:, 3:4]

    # Source: FinalSpinIMRPhenomD_all_in_plane_spin_on_larger_BH and
    # PhenomPCore (LALSimIMRPhenomP.c).
    def FinalSpin_inplane(self, theta, derived, chip):
        """
        Final spin with all in-plane spin on the larger BH.

        A final spin magnitude > 1 is set to 1, keeping its sign.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 6)
            Reduced parameters with m1 >= m2; column 0 is m1.
        derived : torch.Tensor, shape (B, 4)
            Output of compute_derived_parameters; column 0 is M.
        chip : torch.Tensor, shape (B, 1)
            Dimensionless spin in the orbital plane.

        Returns
        -------
        af : torch.Tensor, shape (B, 1)
            Final dimensionless spin.
        """
        # Convention m1 >= m2 of IMRPhenomD
        q_factor = theta[:, 0:1] / derived[:, 0:1]
        af_parallel = self.FinalSpin0815(
            derived[:, 1:2],
            theta[:, 2:3],
            theta[:, 3:4],
        )
        Sperp = chip * q_factor * q_factor
        af = torch.copysign(self.ONE, af_parallel) * torch.sqrt(
            Sperp * Sperp + af_parallel * af_parallel
        )
        # Final spin magnitude > 1: set final spin magnitude = 1
        af = af.clamp(min=-1.0, max=1.0)
        return af

    # Source: FinalSpin0815 (LALSimIMRPhenomD_internals.c).
    def FinalSpin0815(self, eta, chi1, chi2):
        """
        Wrapper function for FinalSpin0815_s.

        Parameters
        ----------
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        chi1 : torch.Tensor, shape (B, 1)
            Dimensionless aligned spin of the larger BH.
        chi2 : torch.Tensor, shape (B, 1)
            Dimensionless aligned spin of the smaller BH.

        Returns
        -------
        af_parallel : torch.Tensor, shape (B, 1)
            Aligned-spin final dimensionless spin.
        """
        Seta = torch.sqrt(self.ONE - 4.0 * eta)
        m1 = self.HALF * (self.ONE + Seta)
        m2 = self.HALF * (self.ONE - Seta)
        s = (m1 * m1) * chi1 + (m2 * m2) * chi2
        return self.FinalSpin0815_s(eta, s)

    # Source: FinalSpin0815_s (LALSimIMRPhenomD_internals.c).
    def FinalSpin0815_s(self, eta, S):
        """
        Formula to predict the final spin.

        Parameters
        ----------
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        S : torch.Tensor, shape (B, 1)
            Spin s = m1^2 chi1 + m2^2 chi2, for unit total mass.

        Returns
        -------
        af : torch.Tensor, shape (B, 1)
            Final dimensionless spin.
        """
        eta2 = eta * eta
        eta3 = eta2 * eta
        S2 = S * S
        S3 = S2 * S
        return eta * (
            3.4641016151377544
            - 4.399247300629289 * eta
            + 9.397292189321194 * eta2
            - 13.180949901606242 * eta3
            + S
            * (
                (self.ONE / eta - 0.0850917821418767 - 5.837029316602263 * eta)
                + (0.1014665242971878 - 2.0967746996832157 * eta) * S
                + (-1.3546806617824356 + 4.108962025369336 * eta) * S2
                + (-0.8676969352555539 + 2.064046835273906 * eta) * S3
            )
        )

    # Source: PhenomInternal_EradRational0815
    # (LALSimIMRPhenomInternalUtils.c).
    def EradRational0815(self, eta, chi1, chi2):
        """
        Wrapper function for EradRational0815_s.

        Parameters
        ----------
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        chi1 : torch.Tensor, shape (B, 1)
            Dimensionless aligned spin of the larger BH.
        chi2 : torch.Tensor, shape (B, 1)
            Dimensionless aligned spin of the smaller BH.

        Returns
        -------
        Erad : torch.Tensor, shape (B, 1)
            Fraction of total mass radiated as gravitational waves.
        """
        Seta = torch.sqrt(self.ONE - 4.0 * eta)
        m1 = self.HALF * (self.ONE + Seta)
        m2 = self.HALF * (self.ONE - Seta)
        m1s = m1 * m1
        m2s = m2 * m2
        s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)

        return self.EradRational0815_s(eta, s)

    # Source: EradRational0815_s (LALSimIMRPhenomInternalUtils.c).
    def EradRational0815_s(self, eta, s):
        """
        Formula to predict the total radiated energy.

        Parameters
        ----------
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        s : torch.Tensor, shape (B, 1)
            Mass-weighted effective spin.

        Returns
        -------
        Erad : torch.Tensor, shape (B, 1)
            Radiated energy over the total mass.
        """
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta

        return (
            (
                0.055974469826360077 * eta
                + 0.5809510763115132 * eta2
                - 0.9606726679372312 * eta3
                + 3.352411249771192 * eta4
            )
            * (
                self.ONE
                + (
                    -0.0030302335878845507
                    - 2.0066110851351073 * eta
                    + 7.7050567802399215 * eta2
                )
                * s
            )
        ) / (
            self.ONE
            + (
                -0.6714403054720589
                - 1.4756929437702908 * eta
                + 7.304676214885011 * eta2
            )
            * s
        )

    # Source: PhenomPCoreTwistUp (LALSimIMRPhenomP.c).
    def PhenomPCoreTwistUp(
        self,
        f_Ms,
        hPhenom,
        eta,
        chi1_l,
        chi2_l,
        chip,
        angcoeffs,
        Y2m,
        alphaoffset,
        epsilonoffset,
    ):
        """
        Twist up the IMRPhenomD waveform into precessing h+ and hx.

        The frequency domain l=2 plus and cross polarizations are
        calculated separately for each m = -2, ... , 2.

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, F)
            Geometric frequencies.
        hPhenom : torch.Tensor, shape (B, F)
            IMRPhenom waveform (before precession).
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        chi1_l : torch.Tensor, shape (B, 1)
            Aligned spin of the smaller BH.
        chi2_l : torch.Tensor, shape (B, 1)
            Aligned spin of the larger BH.
        chip : torch.Tensor, shape (B, 1)
            Dimensionless spin in the orbital plane.
        angcoeffs : torch.Tensor, shape (B, 10)
            PN coeffs for the NNLO angles, from ComputeNNLOanglecoeffs.
        Y2m : torch.Tensor, shape (B, 5)
            l=2 spherical harmonics of spin weight -2.
        alphaoffset : torch.Tensor, shape (B, 1)
            f_ref dependent offset for alpha angle.
        epsilonoffset : torch.Tensor, shape (B, 1)
            f_ref dependent offset for epsilon angle.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, F)
            Plus and cross polarisations.
        """
        q = (self.ONE + torch.sqrt(self.ONE - 4.0 * eta) - 2.0 * eta) / (2.0 * eta)
        # Mass of the smaller BH for unit total mass M=1
        m1 = 1.0 / (1.0 + q)
        # Mass of the larger BH for unit total mass M=1
        m2 = q / (1.0 + q)
        # Dimensionfull spin component in the orbital plane, S_2_perp
        Sperp = chip * (m2 * m2)
        # Dimensionfull aligned spin
        SL = chi1_l * m1 * m1 + chi2_l * m2 * m2

        # Compute PN NNLO angles
        omega = self.PI * f_Ms
        logomega = torch.log(omega)
        omega_cbrt = omega**self.ONE_BY_THREE
        omega_cbrt2 = omega_cbrt * omega_cbrt

        alpha = (
            angcoeffs[:, 0:1] / omega
            + angcoeffs[:, 1:2] / omega_cbrt2
            + angcoeffs[:, 2:3] / omega_cbrt
            + angcoeffs[:, 3:4] * logomega
            + angcoeffs[:, 4:5] * omega_cbrt
        ) - alphaoffset

        epsilon = (
            angcoeffs[:, 5:6] / omega
            + angcoeffs[:, 6:7] / omega_cbrt2
            + angcoeffs[:, 7:8] / omega_cbrt
            + angcoeffs[:, 8:9] * logomega
            + angcoeffs[:, 9:10] * omega_cbrt
        ) - epsilonoffset

        cBetah, sBetah = self.WignerdCoefficients(
            omega_cbrt,
            SL,
            eta,
            Sperp,
        )

        cBetah2 = cBetah * cBetah
        cBetah3 = cBetah2 * cBetah
        cBetah4 = cBetah3 * cBetah
        sBetah2 = sBetah * sBetah
        sBetah3 = sBetah2 * sBetah
        sBetah4 = sBetah3 * sBetah

        hp_sum = self.ZERO
        hc_sum = self.ZERO

        # Sum up contributions to h+ and hx; precompute powers of
        # e^{i m alpha}
        cexp_i_alpha = torch.polar(torch.ones_like(alpha), alpha)
        cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
        cexp_mi_alpha = 1.0 / cexp_i_alpha
        cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
        T2m = (
            cexp_2i_alpha * cBetah4 * Y2m[:, 0:1]
            - cexp_i_alpha * 2 * cBetah3 * sBetah * Y2m[:, 1:2]
            + 1 * self.SQRT_6 * sBetah2 * cBetah2 * Y2m[:, 2:3]
            - cexp_mi_alpha * 2 * cBetah * sBetah3 * Y2m[:, 3:4]
            + cexp_m2i_alpha * sBetah4 * Y2m[:, 4:5]
        )
        Tm2m = (
            cexp_m2i_alpha * sBetah4 * torch.conj(Y2m[:, 0:1])
            + cexp_mi_alpha * 2 * cBetah * sBetah3 * torch.conj(Y2m[:, 1:2])
            + 1 * self.SQRT_6 * sBetah2 * cBetah2 * torch.conj(Y2m[:, 2:3])
            + cexp_i_alpha * 2 * cBetah3 * sBetah * torch.conj(Y2m[:, 3:4])
            + cexp_2i_alpha * cBetah4 * torch.conj(Y2m[:, 4:5])
        )
        hp_sum = T2m + Tm2m
        hc_sum = self.ONE_J * (T2m - Tm2m)
        eps_phase_hP = torch.exp(-self.TWO_J * epsilon) * hPhenom / 2.0

        hp = eps_phase_hP * hp_sum
        hc = eps_phase_hP * hc_sum

        return hp, hc

    # Source: WignerdCoefficients (LALSimIMRPhenomP.c).
    def WignerdCoefficients(self, v, SL, eta, Sp):
        """
        cos(beta/2) and sin(beta/2) for the WignerD symbol.

        Parameters
        ----------
        v : torch.Tensor, shape (B, F)
            Cubic root of (Pi * Frequency (geometric)).
        SL : torch.Tensor, shape (B, 1)
            Dimensionfull aligned spin.
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass-ratio.
        Sp : torch.Tensor, shape (B, 1)
            Dimensionfull spin component in the orbital plane.

        Returns
        -------
        cos_beta_half, sin_beta_half : torch.Tensor, shape (B, F)
            cos(beta/2) and sin(beta/2).
        """
        # We define the shorthand s := Sp / (L + SL)
        L = self.L2PNR(
            v,
            eta,
        )
        s = Sp / (L + SL)
        s2 = s * s
        cos_beta = torch.sqrt(self.ONE / (1.0 + s2))
        # safe_sqrt keeps the gradient finite at cos_beta = 1
        cos_beta_half = safe_sqrt((1.0 + cos_beta) / self.TWO)
        sin_beta_half = safe_sqrt((1.0 - cos_beta) / self.TWO)

        return cos_beta_half, sin_beta_half

    # Source: PhenomPCoreOneFrequency (LALSimIMRPhenomP.c).
    def PhenomPOneFrequency(
        self,
        f,
        f_Ms,
        fx_Ms,
        theta,
        phd_derived,
        coeffs,
        trans_fs,
        fcut_true,
    ):
        """
        IMRPhenom waveform (before precession) and its phase.

        Parameters
        ----------
        f : torch.Tensor, shape (B, F)
            Frequencies in Hz.
        f_Ms : torch.Tensor, shape (B, F)
            Geometric frequencies.
        fx_Ms : torch.Tensor, shape (B, 8)
            Geometric fRef, f1, f2, f3, f4, fRD, fdamp and fmid.
        theta : torch.Tensor, shape (B, 6)
            Reduced parameters with m1 >= m2; column 5 is 2 phic.
        phd_derived : torch.Tensor, shape (B, 4)
            Output of IMRPhenomD.compute_derived_parameters.
        coeffs : torch.Tensor, shape (B, 19)
            Phenom coefficients.
        trans_fs : torch.Tensor, shape (B, 6)
            f1, f2, f3, f4, fRD and fdamp in Hz.
        fcut_true : torch.Tensor, shape (B, 1)
            Cutoff frequency fCut in Hz.

        Returns
        -------
        hPhenom : torch.Tensor, shape (B, F), complex
            IMRPhenom waveform (before precession).
        phase : torch.Tensor, shape (B, F)
            Phase phPhenom.
        """
        phase = super().phase(theta[:, :4], coeffs, phd_derived, f_Ms, fx_Ms)
        # Note: phic is orbital phase
        phase = phase - theta[:, 5:6]

        # amp0 without the 2 sqrt(5 / (64 pi)) factor of IMRPhenomD
        norm = 2.0 * torch.sqrt(self.FIVE / (64.0 * self.PI))
        Amp = (
            super().amp(
                f,
                theta[:, :5],
                coeffs,
                trans_fs,
                phd_derived,
                f_Ms,
                fx_Ms,
                fcut_true,
            )
            / norm
        )

        # Assemble IMRPhenom waveform
        hPhenom = Amp * (torch.exp(-self.ONE_J * phase))
        return hPhenom, phase
