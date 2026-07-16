#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : IMRPhenomPv2.py
Description     : Short description of the file

Created on 2026-01-21 05:26:04

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""


# Packages
import torch

# LOCAL
from sage.data.waveform.approximants import IMRPhenomD
from sage.core.torch import nudge_backward_, nudge_forward_

from sage.data.waveform import taper
from sage.data.waveform import waveform_utils
from sage.core.config import get_cfg, get_data_cfg


class IMRPhenomPv2(IMRPhenomD.IMRPhenomD, torch.nn.Module):
    """
    GPU-native batched IMRPhenomPv2 precessing-spin waveform generator.

    Extends :class:`~sage.data.waveform.approximants.IMRPhenomD.IMRPhenomD`
    with precessing-spin corrections to the polarisations (``hp``, ``hc``),
    then projects through :class:`~sage.data.waveform.project.ConstantProjection`
    to produce detector-frame strain.  Optional SNR rescaling and data
    augmentation are applied before the final output.

    ``GRAPH_READY = True`` indicates that the entire ``forward`` pass is
    compatible with ``torch.compile(fullgraph=True)``.

    Parameters
    ----------
    param_sampler : callable or None
        Waveform parameter sampler; if ``None``, a default
        :class:`~sage.data.waveform.sampler.DistributionSampler` is used.
    waveform_project : callable or None
        Detector projection module; if ``None``, defaults to
        :class:`~sage.data.waveform.project.ConstantProjection`.
    augment : callable or None
        Optional augmentation callable applied to the projected strain.
    """

    GRAPH_READY = True

    def __init__(
        self,
        param_sampler=None,
        waveform_project=None,
        augment=None,
        append_per_det_targets=False,
        extra_batch=0,
    ):

        torch.nn.Module.__init__(self)

        # Setup configs
        self.cfg = get_cfg()
        self.data_cfg = get_data_cfg()

        # Base signal count = class_balance * batch_size. ``extra_batch`` adds
        # more signals beyond that — used by the consistency loop to build
        # non-astrophysical (class-0) samples without eating into the coherent
        # (class-1) signal budget.
        self.signal_batch_size = (
            int(self.cfg.batch_size * self.cfg.class_balance) + int(extra_batch)
        )

        # Fixed frequency grid
        f, f_ref = waveform_utils.get_freqs(
            self.data_cfg.signal_low_frequency_cutoff,
            self.data_cfg.sample_rate / 2.0,
            self.data_cfg.padded_length_in_s,
            self.signal_batch_size,
            self.cfg.device,
            self.cfg.dtype,
        )
        self.f = f
        # Use padded_length_in_s directly — same source used by get_freqs,
        # avoids catastrophic cancellation from f[0][1] - f[0][0].
        _T = float(self.data_cfg.padded_length_in_s)
        self.df = torch.tensor(1.0 / _T, device=self.cfg.device, dtype=self.cfg.dtype)
        self.sample_length_in_s = _T
        self.f_numel = self.f[0].numel()
        self.f_ref = f_ref

        # Initialise PhenomD with freqs
        IMRPhenomD.IMRPhenomD.__init__(self, f, f_ref)

        # Batch size
        self.B = f.shape[0]

        # Tensor of zeroes for hp and hc
        # Accounts for freqs from DC to f_upper
        self.n_pad = int(torch.round((self.f[0][0] - self.df) / self.df)) + 1
        self.hp_buffer = torch.empty(
            (self.B, self.n_pad + self.f_numel),
            dtype=torch.complex64,
            device=f.device,
        )
        self.hc_buffer = torch.empty_like(self.hp_buffer)

        # Parameter sampler
        self.param_sampler = param_sampler
        self.waveform_project = waveform_project
        self.augment = augment

        # When True, append per-detector targets AFTER the class column:
        # [pe..., class, tc_det0.. (physical s), mc_det0.. (standardised)].
        # Consumed by the multi-detector consistency heads. Off by default so
        # the standard [pe..., class] target (and everything that reads
        # targets[:, -1] / [:, :-1]) is unchanged.
        self.append_per_det_targets = bool(append_per_det_targets)
        # Index of mchirp within do_point_estimate (its standardised column).
        self.mc_pe_idx = list(self.cfg.do_point_estimate).index("mchirp")

        # param names needed for Pv2
        self.param_names = [
            "mass1",
            "mass2",
            "spin1x",
            "spin1y",
            "spin1z",
            "spin2x",
            "spin2y",
            "spin2z",
            "distance",
            "tc",
            "coa_phase",
            "inclination",
            "polarization",
            "ra",
            "dec",
        ]

        if self.param_sampler is not None:
            get_idx = self.param_sampler.param_index
            self.req_idx = torch.tensor(
                [get_idx[key] for key in self.param_names],
                device=f.device,
                dtype=torch.int32,
            )

        # Column index of "distance" in the full all_theta parameter tensor.
        # req_idx maps self.param_names → columns of all_theta; "distance" is
        # self.param_names[8], so req_idx[8] gives its all_theta column.
        self.dist_col = int(self.req_idx[8].item())

        # Column of geocentric "tc" in the full all_theta tensor, used to build
        # per-detector arrival times (tc_det = tc_geocentric + projection delay).
        self.tc_col = int(self.param_sampler.param_index["tc"])

        # Target handling
        self.param_sampler.req_idx = self.req_idx
        self.param_sampler._compile_batch_normaliser()
        self.param_sampler._compile_batch_standardiser()

        # Move to correct device
        self.param_sampler.to(self.cfg.device)
        self.waveform_project.to(self.cfg.device)

    @torch.no_grad()
    def forward(self, return_theta=False):
        all_theta = self.param_sampler(self.B)
        req_theta = all_theta[:, self.req_idx]

        hp, hc = self.get_hphc(req_theta)

        proj = self.waveform_project(
            hp,
            hc,
            ra=req_theta[:, -2],
            dec=req_theta[:, -1],
            polarization=req_theta[:, -3],
            return_delay=self.append_per_det_targets,
        )
        if self.append_per_det_targets:
            hf, dt = proj  # dt: (B, D) per-detector geocentre delay (seconds)
        else:
            hf = proj

        if self.augment:
            # augment returns (hf_scaled, scale) where hf_new = hf_old * scale.
            # Since strain ∝ 1/distance, distance_new = distance_old / scale.
            hf, scale = self.augment(hf)
            all_theta[:, self.dist_col] = (
                all_theta[:, self.dist_col] / scale.to(all_theta.dtype)
            )

        # Target handling
        normed_targets = self.param_sampler.standardise_from_batch(all_theta)
        targets = torch.cat(
            [normed_targets, torch.ones_like(normed_targets[:, :1])], dim=1
        )

        # Per-detector targets appended AFTER the class column, so the standard
        # [pe..., class] view (targets[:, : num_pe + 1]) is unchanged. Layout:
        # [pe..., class, tc_det0..tc_det{D-1}, mc_det0..mc_det{D-1}].
        #   - tc_det = geocentric tc + projection delay, physical within-window
        #     seconds (amplitude rescaling above does not affect timing);
        #   - mc_det = the standardised mchirp, identical across detectors for a
        #     real coherent injection (broadcast). The masker overwrites these
        #     for non-astrophysical pairs (per-detector independent mchirp).
        if self.append_per_det_targets:
            D = dt.shape[1]
            per_det_tc = all_theta[:, self.tc_col].unsqueeze(1) + dt   # (B, D)
            per_det_mc = normed_targets[:, self.mc_pe_idx : self.mc_pe_idx + 1].expand(
                -1, D
            )                                                          # (B, D)
            targets = torch.cat([targets, per_det_tc, per_det_mc], dim=1)

        if return_theta:
            return hf, targets, all_theta
        else:
            return hf, targets

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute frequency-domain plus and cross polarisations with precessing-spin corrections.

        Calls :meth:`compute_derived_parameters`, :meth:`convert_spins`, and
        :meth:`PhenomPCoreTwistUp` to apply IMRPhenomPv2 precessing corrections on
        top of the aligned-spin IMRPhenomD backbone.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 15)
            Waveform parameters: mass1, mass2, spin1x, spin1y, spin1z,
            spin2x, spin2y, spin2z, distance, tc, coa_phase, inclination,
            polarization, ra, dec.
        reproduce_lal : bool, optional
            If ``True``, skip tapering, time-shifting, and df normalisation so
            output matches the raw LAL convention.  Default is ``False``.

        Returns
        -------
        hp : torch.Tensor, shape (B, n_freq)
            Plus polarisation (complex64).
        hc : torch.Tensor, shape (B, n_freq)
            Cross polarisation (complex64).
        """
        # m1=0, m2=1, s1x=2, s1y=3, s1z=4, s2x=5, s2y=6,
        # s2z=7, dist_mpc=8, tc=9, phiRef=10, incl=11
        # Pv2 requires m2 > m1; Swapping masses and spins done internally

        # Compute generic derived quantities from masses
        derived = self.compute_derived_parameters(theta)

        # Convert spins into derived quantities
        # TODO: Remove after completing code.
        # chi1_l=0, chi2_l=1, chip=2, thetaJN=3,
        # alpha0=4, phi_aligned=5, zeta_polariz=6
        converted_spins = self.convert_spins(theta, derived)

        # Converting to orbital phase
        phic = 2 * converted_spins[:, 5:6]

        # Get all required coefficients and offsets
        angcoeffs, alphaNNLOoffset, epsilonNNLOoffset = self.compute_pv2_coeffs(
            theta, derived, converted_spins
        )

        Y2 = self.compute_spin_weighted_Y(converted_spins)

        # Calling PhenomD functions which require swapped masses
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

        # This is an IMRPhenomD function and required m1 > m2
        # So we swap back before calling get_coeffs
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

        # Do PhenomD mass swapped operations (m1 > m2)
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

        # PhenomP get hp and hc
        # LALSim PhenomPCoreTwistUp convention (after internal mass swap where m1<=m2):
        #   chi1_l = spin of SMALLER body (m1 in LALSim after swap where m1 <= m2)
        #   chi2_l = spin of LARGER body (m2 in LALSim after swap)
        # Sage's convert_spins stores:
        #   converted_spins[:,0] = chi1_l = spin of SMALLER body  (theta[:,7] = spin1z_input with m2<m1)
        #   converted_spins[:,1] = chi2_l = spin of LARGER body   (theta[:,4] = spin2z_input with m1>m2)
        # No swap needed; pass directly.
        hp, hc = self.PhenomPCoreTwistUp(
            f_Ms,
            hPhenomD,
            derived[:, 1:2],
            converted_spins[:, 0:1],  # chi1_l for TwistUp = spin of SMALLER body
            converted_spins[:, 1:2],  # chi2_l for TwistUp = spin of LARGER body
            converted_spins[:, 2:3],
            angcoeffs,
            Y2,
            alphaNNLOoffset - converted_spins[:, 4:5],
            epsilonNNLOoffset,
        )

        # Do corrections for time shift and phase
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

        # final touches to hp and hc, stolen from Scott
        c2z = torch.cos(2 * converted_spins[:, 6:7])
        s2z = torch.sin(2 * converted_spins[:, 6:7])
        final_hp = c2z * hp + s2z * hc
        final_hc = c2z * hc - s2z * hp

        if not reproduce_lal:
            # Frequency domain tapering
            _taper = taper.fd_taper(
                f=self.f,
                f_min=20.0,
                f_cut=fcut_true,
                df=self.df,
            )
            final_hp *= _taper
            final_hc *= _taper

            # Apply phase shift equivalent to applying tc
            final_hp, final_hc = self.apply_tc(final_hp, final_hc, theta[:, 9:10])

            # Make hf consistent with the scale of other data
            # LAL works in continuous Fourier regime
            final_hp *= self.df
            final_hc *= self.df

        # Accounting for DC components and zero-padding below f_min
        # We start from 0 Hz, df Hz, 2df Hz; not including f_min
        # Assuming f_min included in fs
        final_hp, final_hc = self.pad_missing_frequencies(final_hp, final_hc)

        return final_hp, final_hc

    def apply_tc(self, hp, hc, tc):
        """
        Apply a time-of-coalescence phase shift to hp and hc.

        Converts ``tc`` from duration-space into a frequency-domain phase ramp
        and applies it to the plus and cross polarisations in polar form.

        Parameters
        ----------
        hp : torch.Tensor, shape (B, n_freq)
            Plus polarisation.
        hc : torch.Tensor, shape (B, n_freq)
            Cross polarisation.
        tc : torch.Tensor, shape (B, 1)
            Time of coalescence in seconds relative to the segment end.

        Returns
        -------
        hp : torch.Tensor, shape (B, n_freq)
            Phase-shifted plus polarisation.
        hc : torch.Tensor, shape (B, n_freq)
            Phase-shifted cross polarisation.
        """
        # Apply time shift to account for tc
        # Converting from tc in duration space to actual shift
        _tc = (tc + self.data_cfg.padding_length_in_s) - self.sample_length_in_s
        # We do this in polar as well without torch exp
        hp = torch.polar(torch.abs(hp), torch.angle(hp) - 2 * self.PI * self.f * _tc)
        hc = torch.polar(torch.abs(hc), torch.angle(hc) - 2 * self.PI * self.f * _tc)
        return hp, hc

    def pad_missing_frequencies(self, hp, hc):
        """
        Zero-pad hp and hc from DC to the low-frequency cutoff.

        The waveform is only computed above f_min; this method prefixes the
        required number of zero bins so the output spans [0, f_max] with
        uniform df spacing.

        Parameters
        ----------
        hp : torch.Tensor, shape (B, n_active)
            Plus polarisation on the active frequency grid.
        hc : torch.Tensor, shape (B, n_active)
            Cross polarisation on the active frequency grid.

        Returns
        -------
        hp_pad : torch.Tensor, shape (B, n_pad + n_active)
            Zero-padded plus polarisation.
        hc_pad : torch.Tensor, shape (B, n_pad + n_active)
            Zero-padded cross polarisation.
        """
        # Accounting for DC components and zero-padding below f_min
        # We start from 0 Hz, df Hz, 2df Hz; not including f_min
        # Assuming f_min included in fs
        # This accounts for LAL-like handlings of f
        hp_pad = torch.zeros_like(self.hp_buffer)
        hc_pad = torch.zeros_like(self.hc_buffer)
        # Fill empty buffer with hp and hc
        hp_pad[:, self.n_pad :] = hp
        hc_pad[:, self.n_pad :] = hc

        return hp_pad, hc_pad

    def compute_derived_parameters(self, theta):
        """
        Compute PhenomPv2-specific derived parameters from raw masses.

        Overrides the IMRPhenomD base method.  Internally, mass ordering
        is m1 ≤ m2 (Pv2 convention), whereas PhenomD expects m1 ≥ m2 and
        the swap is applied before calling any PhenomD helper.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 2+)
            Columns 0 and 1 are mass1 and mass2 in solar masses.

        Returns
        -------
        derived : torch.Tensor, shape (B, 4)
            Columns: M (total mass, M☉), eta (symmetric mass ratio),
            q = m1/m2 ≥ 1 (Pv2 convention), M_s (M in seconds).
        """
        # Overriding inherited method
        # Derived params different from PhenomD
        M = theta[:, 1:2] + theta[:, 0:1]
        eta = theta[:, 1:2] * theta[:, 0:1] / (M * M)
        q = theta[:, 0:1] / theta[:, 1:2]  # q>=1 due to swapped masses
        M_s = (theta[:, 1:2] + theta[:, 0:1]) * self.GM  # also called m_sec
        # Mirror LALSim PhenomPCore: "if (eta > 0.25 || q < 1.0) { nudge(&eta,0.25,...); nudge(&q,1.0,...); }"
        # LALSim's nudge() rounds TO the boundary; clamp_(max/min) matches that.
        eta.clamp_(max=0.25)
        q.clamp_(min=1.0)

        return torch.cat([M, eta, q, M_s], dim=1)

    def compute_pv2_coeffs(self, theta, derived, converted_spins):
        """
        Compute NNLO precession-angle coefficients and reference-frequency offsets.

        Evaluates the five alpha (precession) and five epsilon (rotation) PN
        coefficients via :meth:`ComputeNNLOanglecoeffs` and integrates them at
        ``f_ref`` to obtain the reference-frame offsets used in
        :meth:`PhenomPCoreTwistUp`.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 2+)
            Raw waveform parameters; masses are in columns 0–1.
        derived : torch.Tensor, shape (B, 4)
            Output of :meth:`compute_derived_parameters`.
        converted_spins : torch.Tensor, shape (B, 7)
            Output of :meth:`convert_spins`.

        Returns
        -------
        angcoeffs : torch.Tensor, shape (B, 10)
            Stacked alpha (cols 0–4) and epsilon (cols 5–9) PN coefficients.
        alphaNNLOoffset : torch.Tensor, shape (B, 1)
            Precession angle at ``f_ref`` for reference-frame subtraction.
        epsilonNNLOoffset : torch.Tensor, shape (B, 1)
            Rotation angle at ``f_ref`` for reference-frame subtraction.
        """
        # Other one-off derived quantities
        chi_eff = (
            theta[:, 1:2] * converted_spins[:, 0:1]
            + theta[:, 0:1] * converted_spins[:, 1:2]
        ) / derived[:, 0:1]

        chil = (1.0 + derived[:, 2:3]) / derived[:, 2:3] * chi_eff

        piM = self.PI * derived[:, 3:4]

        omega_ref = piM * self.f_ref
        logomega_ref = torch.log(omega_ref)
        omega_ref_cbrt = (piM * self.f_ref) ** self.ONE_BY_THREE
        omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

        # angcoeffs is a torch.cat with the following values in order
        # alphacoeff1, alphacoeff2, alphacoeff3, alphacoeff4, alphacoeff5,
        # epsiloncoeff1, epsiloncoeff2, epsiloncoeff3, epsiloncoeff4, epsiloncoeff5,
        angcoeffs = self.ComputeNNLOanglecoeffs(
            derived[:, 2:3],
            chil,
            converted_spins[:, 2:3],
        )

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

    def compute_spin_weighted_Y(self, converted_spins):
        """
        Evaluate the five l=2 spin-weight-(-2) spherical harmonics at thetaJN.

        Parameters
        ----------
        converted_spins : torch.Tensor, shape (B, 7)
            Output of :meth:`convert_spins`; column 3 is ``thetaJN``.

        Returns
        -------
        Y2 : torch.Tensor, shape (B, 5)
            Columns: Y₂₋₂, Y₂₋₁, Y₂₀, Y₂₁, Y₂₂ (complex, s=-2).
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
        Compute all dimensionful frequency quantities needed for PhenomPv2.

        Calls :meth:`phP_get_transition_frequencies` with the mass-swapped
        parameters and collects the full set of f × M_s scale products.

        Parameters
        ----------
        theta_swapped : torch.Tensor, shape (B, 6)
            Mass-swapped reduced parameter vector passed to PhenomD helpers.
        derived : torch.Tensor, shape (B, 4)
            PhenomPv2 derived parameters from :meth:`compute_derived_parameters`.
        phd_derived : torch.Tensor, shape (B, 4+)
            PhenomD derived parameters from the parent ``compute_derived_parameters``.
        coeffs : torch.Tensor, shape (B, 7+)
            PhenomD amplitude/phase coefficients from :meth:`get_coeffs`.
        converted_spins : torch.Tensor, shape (B, 7)
            Output of :meth:`convert_spins`; column 2 is chip.

        Returns
        -------
        f_Ms : torch.Tensor, shape (B, n_freq)
            Frequency grid scaled by M_s.
        fx_Ms : torch.Tensor, shape (B, 8)
            Special frequency scale products: fref, f1, f2, f3, f4, fRD, fdamp, fmid.
        fcut_true : torch.Tensor, shape (B, 1)
            Physical frequency cutoff in Hz.
        trans_fs : torch.Tensor, shape (B, 6)
            Transition frequencies: f1, f2, f3, f4, fRD, fdamp.
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

        # Precomputing required parameters
        f1_Ms = trans_fs[:, 0:1] * derived[:, 3:4]
        f2_Ms = trans_fs[:, 1:2] * derived[:, 3:4]
        f3_Ms = trans_fs[:, 2:3] * derived[:, 3:4]
        f4_Ms = trans_fs[:, 3:4] * derived[:, 3:4]
        f_Ms = self.f * derived[:, 3:4]
        fref_Ms = self.f_ref * derived[:, 3:4]
        f_RD_Ms = trans_fs[:, 4:5] * derived[:, 3:4]
        f_damp_Ms = trans_fs[:, 5:6] * derived[:, 3:4]
        # Central frequency point (used f_RD and f_damp)
        fmid_Ms = ((trans_fs[:, 2:3] + trans_fs[:, 3:4]) / 2) * derived[:, 3:4]
        fx_Ms = torch.cat(
            [fref_Ms, f1_Ms, f2_Ms, f3_Ms, f4_Ms, f_RD_Ms, f_damp_Ms, fmid_Ms], dim=1
        )

        return f_Ms, fx_Ms, fcut_true, trans_fs

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
        Apply time-shift and phase corrections so the PhenomPv2 waveform coalesces at t=0.

        Evaluates the PhenomD phase on a fixed frequency grid near the ringdown
        frequency, estimates d(phase)/df via central difference, and calls
        :meth:`apply_time_shift_phase_correction`.

        Parameters
        ----------
        hp : torch.Tensor, shape (B, n_freq)
            Plus polarisation before correction.
        hc : torch.Tensor, shape (B, n_freq)
            Cross polarisation before correction.
        theta_swapped : torch.Tensor, shape (B, 6)
            Mass-swapped reduced parameters.
        derived : torch.Tensor, shape (B, 4)
            PhenomPv2 derived parameters.
        phd_derived : torch.Tensor, shape (B, 4+)
            PhenomD derived parameters.
        trans_fs : torch.Tensor, shape (B, 6)
            Transition frequencies (f1, f2, f3, f4, fRD, fdamp).
        fx_Ms : torch.Tensor, shape (B, 8)
            Special frequency scale products.
        coeffs : torch.Tensor, shape (B, 7+)
            PhenomD coefficients.
        fcut_true : torch.Tensor, shape (B, 1)
            Physical frequency cutoff in Hz.

        Returns
        -------
        hp : torch.Tensor, shape (B, n_freq)
            Corrected plus polarisation.
        hc : torch.Tensor, shape (B, n_freq)
            Corrected cross polarisation.
        """
        ## ** This is where we do the corrections to phase and time shift **
        # Fixed frequency grid around ringdown frequency for Pv2
        # 10 points should be enough for cubic interpolation
        # Same n_fixed used in LAL version
        n_fixed = 1000

        fcut = self.fM_CUT / derived[:, 3:4]
        f_final = trans_fs[:, 4:5]

        freqs_fixed_start = 0.8 * f_final
        freqs_fixed_stop = torch.minimum(1.2 * f_final, fcut)

        # Create linspace weights once
        t = torch.linspace(
            0.0,
            1.0,
            n_fixed,
            device=self.f.device,
            dtype=self.f.dtype,
        )

        # Broadcast to (B, n_fixed)
        freqs_fixed = freqs_fixed_start + (freqs_fixed_stop - freqs_fixed_start) * t
        ff_Ms = freqs_fixed * derived[:, 3:4]

        # Compute phase on fixed grid
        # We have inverted m1 and m2 back to the convention m1 > m2 for PhenomD call
        phase_fixed = torch.empty(n_fixed, device=self.f.device, dtype=self.f.dtype)

        _, phase_fixed = self.PhenomPOneFrequency(
            freqs_fixed,
            ff_Ms,
            fx_Ms,
            theta_swapped,
            phd_derived,
            coeffs,
            trans_fs,
            fcut_true,
        )

        hp, hc = self.apply_time_shift_phase_correction(
            hptilde=hp,
            hctilde=hc,
            freqs_fixed=freqs_fixed,
            phase_fixed=phase_fixed,
            f_final=f_final,
        )

        return hp, hc

    def convert_spins(self, theta, derived):
        """
        Convert Cartesian spin components to the PhenomPv2 spin parameterisation.

        Maps (spin1x/y/z, spin2x/y/z) plus masses and inclination into the
        seven quantities used throughout PhenomPv2: aligned spins (chi1_l,
        chi2_l), precessing-plane spin magnitude (chip), tilt of J w.r.t.
        line-of-sight (thetaJN), precession reference angle (alpha0),
        aligned-frame orbital phase offset (phi_aligned), and polarisation
        rotation (zeta_polariz).

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 12+)
            Columns: mass1[0], mass2[1], spin1x[2], spin1y[3], spin1z[4],
            spin2x[5], spin2y[6], spin2z[7], distance[8], tc[9],
            coa_phase[10], inclination[11].
            Sage convention: mass1 >= mass2 (enforced by mass_order constraint),
            so spin1 belongs to the LARGER body and spin2 to the SMALLER body.
            Therefore theta[:,7] (spin2z) = aligned spin of the SMALLER body
            and theta[:,4] (spin1z) = aligned spin of the LARGER body, which
            is why chi1_l reads from index 7 and chi2_l from index 4 below.
        derived : torch.Tensor, shape (B, 4)
            Output of :meth:`compute_derived_parameters`.

        Returns
        -------
        converted : torch.Tensor, shape (B, 7)
            Columns: chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz.
        """
        m1_2 = theta[:, 1:2] * theta[:, 1:2]
        m2_2 = theta[:, 0:1] * theta[:, 0:1]

        # From the components in the source frame, we can easily determine
        # chi1_l, chi2_l, chip and phi_aligned, which we need to return.
        # We also compute the spherical angles of J,
        # which we need to transform to the J frame

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
        # Adding this for safety (we shouldn't need it)
        # const REAL8 den = (m2 > m1) ? A2*m2_2 : A1*m1_2;
        den = torch.where(theta[:, 0:1] > theta[:, 1:2], A2 * m2_2, A1 * m1_2)
        chip = num / den

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
        phiJ_sf = torch.arctan2(J0y_sf, J0x_sf)
        phi_aligned = -phiJ_sf

        # First we determine kappa
        # in the source frame, the components of N are given in Eq (35c) of T1500606-v6
        Nx_sf = torch.sin(theta[:, 11:12]) * torch.cos(self.PI / 2.0 - theta[:, 10:11])
        Ny_sf = torch.sin(theta[:, 11:12]) * torch.sin(self.PI / 2.0 - theta[:, 10:11])
        Nz_sf = torch.cos(theta[:, 11:12])

        tmp_x = Nx_sf
        tmp_y = Ny_sf
        tmp_z = Nz_sf

        tmp_x, tmp_y, tmp_z = self.ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)

        kappa = -torch.arctan2(tmp_y, tmp_x)

        # Then we determine alpha0, by rotating LN
        tmp_x, tmp_y, tmp_z = self.ZERO, self.ZERO, self.ONE
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

        alpha0 = torch.arctan2(tmp_y, tmp_x)

        # Finally we determine thetaJ, by rotating N
        tmp_x, tmp_y, tmp_z = Nx_sf, Ny_sf, Nz_sf
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)
        Nx_Jf, Nz_Jf = tmp_x, tmp_z
        thetaJN = torch.arccos(Nz_Jf)

        # Finally, we need to redefine the polarizations:
        # PhenomP's polarizations are defined following Arun et al (arXiv:0810.5336)
        # i.e. projecting the metric onto the P,Q,N triad defined with P=NxJ/|NxJ|
        # (see (2.6) in there).
        # By contrast, the triad X,Y,N used in LAL
        # ("waveframe" in the nomenclature of T1500606-v6)
        # is defined in e.g. eq (35) of this document
        # (via its components in the source frame; note we use the defautl Omega=Pi/2).
        # Both triads differ from each other by a rotation around N by an angle \zeta
        # and we need to rotate the polarizations accordingly by 2\zeta

        Xx_sf = -torch.cos(theta[:, 11:12]) * torch.sin(theta[:, 10:11])
        Xy_sf = -torch.cos(theta[:, 11:12]) * torch.cos(theta[:, 10:11])
        Xz_sf = torch.sin(theta[:, 11:12])
        tmp_x, tmp_y, tmp_z = Xx_sf, Xy_sf, Xz_sf
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
        tmp_x, tmp_y, tmp_z = self.ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

        # Now the tmp_a are the components of X in the J frame
        # We need the polar angle of that vector in the P,Q basis of Arun et al
        # P = NxJ/|NxJ| and since we put N in the (pos x)z half plane of the J frame
        PArunx_Jf = self.ZERO
        PAruny_Jf = -self.ONE
        PArunz_Jf = self.ZERO

        # Q = NxP
        QArunx_Jf = Nz_Jf
        QAruny_Jf = self.ZERO
        QArunz_Jf = -Nx_Jf

        # Calculate the dot products XdotPArun and XdotQArun
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

    def L2PNR(self, v, eta):
        """
        Compute the 2PN orbital angular momentum magnitude L (reduced units).

        Parameters
        ----------
        v : torch.Tensor, shape (B, 1) or (B, n_freq)
            Orbital velocity (πM f_ref)^(1/3).
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.

        Returns
        -------
        L : torch.Tensor
            2PN orbital angular momentum in units of M² (G=c=1).
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

    @staticmethod
    def ROTATEZ(angle, x, y, z):
        """Rotate vector (x, y, z) about the z-axis by ``angle`` radians."""
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        return x * ca - y * sa, x * sa + y * ca, z

    @staticmethod
    def ROTATEY(angle, x, y, z):
        """Rotate vector (x, y, z) about the y-axis by ``angle`` radians."""
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        return x * ca + z * sa, y, -x * sa + z * ca

    def ComputeNNLOanglecoeffs(self, q, chil, chip):
        """
        Compute the ten NNLO PN precession-angle coefficients.

        Returns the five alpha (precession angle) and five epsilon (rotation
        angle) post-Newtonian coefficients as stacked columns.  See Appendix A
        of arXiv:1408.1810 (Hannam et al.) for the analytic expressions.

        Parameters
        ----------
        q : torch.Tensor, shape (B, 1)
            Mass ratio q = m1/m2 ≥ 1.
        chil : torch.Tensor, shape (B, 1)
            Effective aligned spin χ_eff weighted by (1+q)/q.
        chip : torch.Tensor, shape (B, 1)
            In-plane spin magnitude parameter.

        Returns
        -------
        angcoeffs : torch.Tensor, shape (B, 10)
            Columns 0–4: alphacoeff1…5; columns 5–9: epsiloncoeff1…5.
        """
        # Precompute
        m2 = q / (1.0 + q)
        m1 = self.ONE / (1.0 + q)
        dm = m1 - m2
        mtot = self.ONE
        eta = m1 * m2
        # This should prevent NaNs
        nudge_backward_(eta, 0.25, 1e-6)

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

    def SpinWeightedY(self, theta, phi, s, l, m):
        """
        Evaluate a spin-weighted spherical harmonic Y^s_{lm}(theta, phi).

        Currently supports only s=-2, l=2, m in {-2,-1,0,1,2} (dominant GW
        modes).  Ported from ``SphericalHarmonics.c`` in LALSuite.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 1)
            Polar angle in radians.
        phi : float or torch.Tensor
            Azimuthal angle in radians; typically 0 in the J-frame.
        s : int
            Spin weight; must be -2.
        l : int
            Degree; must be 2.
        m : int
            Order; must satisfy |m| ≤ l.

        Returns
        -------
        Y : torch.Tensor, shape (B, 1)
            Complex spin-weighted spherical harmonic value.
        """
        # Copied from SphericalHarmonics.c in LAL
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

        # TODO: Replacing with polar since here it might be more efficient
        return fac * torch.exp(self.ONE_J * m * phi)

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
        Compute PhenomPv2 phase and amplitude transition frequencies.

        Differs from the parent PhenomD method by using
        :meth:`phP_get_fRD_fdamp` (which incorporates the in-plane spin chip)
        rather than the aligned-spin ringdown frequency.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 6)
            Mass-swapped reduced parameters.
        gamma2 : torch.Tensor, shape (B, 1)
            PhenomD amplitude Lorentzian width coefficient.
        gamma3 : torch.Tensor, shape (B, 1)
            PhenomD amplitude Lorentzian damping coefficient.
        chip : torch.Tensor, shape (B, 1)
            In-plane spin magnitude.
        derived : torch.Tensor, shape (B, 4)
            PhenomPv2 derived parameters.
        phd_derived : torch.Tensor, shape (B, 4+)
            PhenomD derived parameters.

        Returns
        -------
        trans_fs : torch.Tensor, shape (B, 6)
            Transition frequencies: f1, f2, f3, f4, fRD, fdamp in Hz.
        """
        # m1 > m2 should hold here (masses swapped before calling)
        # get_fRD_fdamp is different; so we had to rewrite this function again
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

        # Replacing heaviside with where;
        # Boundary will not reach exactly due to machine precision
        f4 = torch.where(gamma2 >= 1, f4_gammaneg_gtr_1, f4_gammaneg_less_1)

        return torch.cat([f1, f2, f3, f4, f_RD, f_damp], dim=1)

    def phP_get_fRD_fdamp(self, theta, derived, phd_derived, chip):
        """
        Compute ringdown and damping frequencies for PhenomPv2.

        Uses the precessing final spin from :meth:`FinalSpin_inplane` (which
        includes the in-plane chip contribution) and the radiated energy from
        :meth:`EradRational0815` to look up fRD and fdamp from QNM tables.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 6)
            Mass-swapped reduced parameters; columns 2–3 are chi1, chi2.
        derived : torch.Tensor, shape (B, 4)
            PhenomPv2 derived parameters.
        phd_derived : torch.Tensor, shape (B, 4+)
            PhenomD derived parameters; column 3 is eta.
        chip : torch.Tensor, shape (B, 1)
            In-plane spin magnitude.

        Returns
        -------
        fRD : torch.Tensor, shape (B, 1)
            Ringdown frequency in Hz.
        fdamp : torch.Tensor, shape (B, 1)
            Damping frequency in Hz.
        """
        # m1 > m2 should hold here
        finspin = self.FinalSpin_inplane(theta, derived, chip)
        Erad = self.EradRational0815(
            phd_derived[:, 3:4],
            theta[:, 2:3],
            theta[:, 3:4],
        )

        rel_idx = (finspin - self.QNMData_a[0]) / (
            self.QNMData_a[1] - self.QNMData_a[0]
        )
        idx_lower = rel_idx.floor().long().clamp(0, len(self.QNMData_a) - 2)
        frac = rel_idx - idx_lower.float()

        fRD = (
            self.QNMData_fRD[idx_lower] * (1.0 - frac)
            + self.QNMData_fRD[idx_lower + 1] * frac
        )
        fdamp = (
            self.QNMData_fdamp[idx_lower] * (1.0 - frac)
            + self.QNMData_fdamp[idx_lower + 1] * frac
        )

        factor = 1.0 / (1.0 - Erad)
        fRD *= factor
        fdamp *= factor

        return fRD / derived[:, 3:4], fdamp / derived[:, 3:4]

    def FinalSpin_inplane(self, theta, derived, chip):
        """
        Compute the final dimensionless spin including in-plane spin contribution.

        Combines the aligned final spin from :meth:`FinalSpin0815` with the
        perpendicular component S_perp = chip × (m2/M)² to produce the total
        final spin magnitude, preserving the sign from the aligned component.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 6)
            Mass-swapped reduced parameters; column 0 is m1 (larger mass).
        derived : torch.Tensor, shape (B, 4)
            PhenomPv2 derived parameters; column 0 is M.
        chip : torch.Tensor, shape (B, 1)
            In-plane spin magnitude.

        Returns
        -------
        af : torch.Tensor, shape (B, 1)
            Final dimensionless spin.
        """
        # This is without GM and swapped (equivalent to original M, eta)
        # Swapping does not change M or eta value
        # Here we assume m1 > m2, the convention used in phenomD
        # (not the convention of internal phenomP)
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
        # Match LAL (LALSimIMRPhenomP.c:798-801): the in-plane quadrature add can
        # push |af| > 1 in extreme corners (very high mass ratio + large aligned
        # AND in-plane spin), which the QNM table (nodes at |a| <= 1) cannot take
        # -- Sage would otherwise extrapolate past the endpoint, giving a wrong
        # (even negative) fdamp. Clamp the magnitude to 1, preserving sign. This
        # is a no-op on the whole validated prior (max reachable |af| ~ 0.98).
        af = af.clamp(min=-1.0, max=1.0)
        return af

    def FinalSpin0815(self, eta, chi1, chi2):
        """
        Compute the aligned final spin using the Barkett et al. (0815) fit.

        Delegates to :meth:`FinalSpin0815_s` after forming the mass-weighted
        effective spin S = m1²·chi1 + m2²·chi2.

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

    def FinalSpin0815_s(self, eta, S):
        """
        Evaluate the Barkett et al. (arXiv:0815) final-spin rational fit given S.

        Parameters
        ----------
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        S : torch.Tensor, shape (B, 1)
            Mass-weighted effective spin S = (m1²·chi1 + m2²·chi2) / M².

        Returns
        -------
        af : torch.Tensor, shape (B, 1)
            Final dimensionless spin from the rational fit.
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

    def EradRational0815(self, eta, chi1, chi2):
        """
        Compute radiated energy fraction using the Barkett et al. (0815) rational fit.

        Delegates to :meth:`EradRational0815_s` after forming the mass-weighted
        effective spin.

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

    def EradRational0815_s(self, eta, s):
        """
        Evaluate the Barkett et al. (arXiv:0815) radiated-energy rational fit.

        Parameters
        ----------
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        s : torch.Tensor, shape (B, 1)
            Mass-weighted effective spin.

        Returns
        -------
        Erad : torch.Tensor, shape (B, 1)
            Radiated energy fraction E_rad / M_total.
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
        Apply the PhenomPv2 "twist-up" to convert aligned-spin PhenomD into
        precessing hp and hc polarisations.

        Evaluates the Wigner d-matrix coefficients, computes the precessing-frame
        alpha and epsilon angles, and assembles the l=2 mode sum following
        arXiv:1408.1810 (Hannam et al.), eqs. (A1)–(A4).

        Parameters
        ----------
        f_Ms : torch.Tensor, shape (B, n_freq)
            Frequency grid scaled by M_s.
        hPhenom : torch.Tensor, shape (B, n_freq)
            Complex aligned-spin PhenomD waveform (amplitude × phase).
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        chi1_l : torch.Tensor, shape (B, 1)
            Aligned spin of the larger BH.
        chi2_l : torch.Tensor, shape (B, 1)
            Aligned spin of the smaller BH.
        chip : torch.Tensor, shape (B, 1)
            In-plane spin parameter.
        angcoeffs : torch.Tensor, shape (B, 10)
            NNLO precession coefficients from :meth:`ComputeNNLOanglecoeffs`.
        Y2m : torch.Tensor, shape (B, 5)
            Spin-weighted spherical harmonics from :meth:`compute_spin_weighted_Y`.
        alphaoffset : torch.Tensor, shape (B, 1)
            Reference-frame alpha offset.
        epsilonoffset : torch.Tensor, shape (B, 1)
            Reference-frame epsilon offset.

        Returns
        -------
        hp : torch.Tensor, shape (B, n_freq)
            Plus polarisation (complex).
        hc : torch.Tensor, shape (B, n_freq)
            Cross polarisation (complex).
        """
        q = (self.ONE + torch.sqrt(self.ONE - 4.0 * eta) - 2.0 * eta) / (2.0 * eta)
        # Mass of the smaller BH for unit total mass M=1
        m1 = 1.0 / (1.0 + q)
        # Mass of the larger BH for unit total mass M=1
        m2 = q / (1.0 + q)
        # Dimensionfull spin component in the orbital plane. S_perp = S_2_perp
        Sperp = chip * (m2 * m2)
        # Dimensionfull aligned spin
        SL = chi1_l * m1 * m1 + chi2_l * m2 * m2

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

        # Replacing complex ops with real ops and complex label
        # cexp_i_alpha = torch.exp(self.ONE_J * alpha)
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
        # Doing polar here will be less efficient since it requires abs and angle ops
        # torch.polar(torch.abs(hPhenom) / 2.0, torch.angle(hPhenom) - 2.0 * epsilon)
        eps_phase_hP = torch.exp(-self.TWO_J * epsilon) * hPhenom / 2.0

        hp = eps_phase_hP * hp_sum
        hc = eps_phase_hP * hc_sum

        return hp, hc

    def WignerdCoefficients(self, v, SL, eta, Sp):
        """
        Compute the half-angle Wigner d-matrix coefficients cos(β/2) and sin(β/2).

        Estimates the precession opening angle β from the ratio of the total
        in-plane spin to the total angular momentum, using the 2PN orbital
        angular momentum from :meth:`L2PNR`.

        Parameters
        ----------
        v : torch.Tensor, shape (B, n_freq)
            Orbital velocity (πMf)^(1/3).
        SL : torch.Tensor, shape (B, 1)
            Aligned dimensionful spin S_L = chi1_l·m1² + chi2_l·m2².
        eta : torch.Tensor, shape (B, 1)
            Symmetric mass ratio.
        Sp : torch.Tensor, shape (B, 1)
            In-plane spin S_perp = chip · m2².

        Returns
        -------
        cos_beta_half : torch.Tensor
            cos(β/2) for the Wigner d-matrix.
        sin_beta_half : torch.Tensor
            sin(β/2) for the Wigner d-matrix.
        """
        # CL: jnp to torch; x**0.5 to sqrt; powers expanded
        # We define the shorthand s := Sp / (L + SL)
        L = self.L2PNR(
            v,
            eta,
        )
        s = Sp / (L + SL)
        s2 = s * s
        cos_beta = torch.sqrt(self.ONE / (1.0 + s2))
        cos_beta_half = torch.sqrt(((1.0 + cos_beta) / self.TWO))
        sin_beta_half = torch.sqrt(((1.0 - cos_beta) / self.TWO))

        return cos_beta_half, sin_beta_half

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
        m1, m2: in solar masses
        phic: Orbital phase at the peak of the underlying non precessing model (rad)
        M: Total mass (Solar masses)
        """
        ## PHASE
        phase = super().phase(theta[:, :4], coeffs, phd_derived, f_Ms, fx_Ms)
        phase = phase - theta[:, 5:6]

        ## AMPLITUDE
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

        # phase -= 2. * phic on line 1316
        # LAL assumed orbital phase and we have already accounted for this
        # Similar reason; no abs or angle if not using polar
        hPhenom = Amp * (torch.exp(-self.ONE_J * phase))
        return hPhenom, phase

    def apply_time_shift_phase_correction(
        self,
        hptilde,
        hctilde,
        freqs_fixed,
        phase_fixed,
        f_final,
        offset: int = 0,
    ):
        """
        Apply time shift correction so the waveform coalesces at t=0.

        Args:
            hptilde: Tensor of shape (n_freq,) with plus polarization.
            hctilde: Tensor of shape (n_freq,) with cross polarization.
            freqs: Tensor of frequencies corresponding to hptilde/hctilde.
            freqs_fixed: Fixed frequency grid used for spline interpolation.
            phase_fixed: Phase values on freqs_fixed.
            f_final: Final frequency (fRD or f_merger) to evaluate derivative.
            offset: Index offset if freqs does not start at zero.
        Returns:
            Tuple of corrected (hptilde, hctilde)
        """

        # Compute relative index (uniform grid assumption)
        rel_idx = (f_final - freqs_fixed[:, :1]) / (
            freqs_fixed[:, 1:2] - freqs_fixed[:, :1]
        )

        # Fast local estimate of dphi/df at f_final using a 3-point central difference
        # on the reduced (n_fixed) frequency grid. This avoids constructing a full
        # cubic spline, which is unnecessary when only a single derivative per batch
        # is required and the phase is smooth in the matching region.
        idx = rel_idx.floor().long().clamp(1, freqs_fixed.shape[1] - 2)

        f_prev = freqs_fixed.gather(1, idx - 1)
        f_next = freqs_fixed.gather(1, idx + 1)
        p_prev = phase_fixed.gather(1, idx - 1)
        p_next = phase_fixed.gather(1, idx + 1)

        # LALSim stores phase_fixed = *phasing = -phPhenom (negated PhenomD phase).
        # Sage stores phase_fixed = +phPhenom - 2*phic (positive).
        # Therefore: d(Sage phase_fixed)/df = -d(LALSim phase_fixed)/df
        # => negate here so Sage t_corr matches LALSim t_corr.
        t_corr_fixed = -(p_next - p_prev) / (f_next - f_prev) / (2 * self.PI)

        # Compute phase correction factor
        # LALSim line 1157: exp(-i*2pi*f*t_corr)  — negative sign
        phase_corr = torch.exp(-self.TWO_J * self.PI * self.f * t_corr_fixed)

        # Apply to waveform, respecting offset
        hptilde[..., offset : offset + self.f_numel] *= phase_corr
        hctilde[..., offset : offset + self.f_numel] *= phase_corr

        return hptilde, hctilde
