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


class IMRPhenomPv2(IMRPhenomD.IMRPhenomD):

    def __init__(self, f, f_ref):
        super().__init__(f, f_ref)
        # Fixed frequency grid
        self.f = f
        self.df = f[0][1] - f[0][0]
        self.sample_length_in_s = 1.0 / self.df
        self.f_numel = self.f[0].numel()
        self.f_ref = f_ref
        # Batch size
        self.B = f.shape[0]
        # Tensor of zeroes for hp and hc
        # Accounts for freqs from DC to f_upper
        self.n_pad = int(torch.round((self.f[0][0] - self.df) / self.df)) + 1
        self.hp_buffer = torch.empty(
            (self.B, self.n_pad + self.f_numel),
            dtype=torch.complex128,
            device=f.device,
        )
        self.hc_buffer = torch.empty_like(self.hp_buffer)

    @torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)
    def __call__(self, theta, reproduce_lal=False):
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
        hp, hc = self.PhenomPCoreTwistUp(
            f_Ms,
            hPhenomD,
            derived[:, 1:2],
            converted_spins[:, 0:1],
            converted_spins[:, 1:2],
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
        hp = c2z * hp + s2z * hc
        hc = c2z * hc - s2z * hp

        if not reproduce_lal:
            # Frequency domain tapering
            _taper = taper.fd_taper(
                f=self.f,
                f_min=20.0,
                f_cut=fcut_true,
                df=self.df,
            )
            hp *= _taper
            hc *= _taper

            # Apply phase shift equivalent to applying tc
            hp, hc = self.apply_tc(hp, hc, theta[:, 9:10])

            # Make hf consistent with the scale of other data
            # LAL works in continuous Fourier regime
            hp *= self.df
            hc *= self.df

        # Accounting for DC components and zero-padding below f_min
        # We start from 0 Hz, df Hz, 2df Hz; not including f_min
        # Assuming f_min included in fs
        hp, hc = self.pad_missing_frequencies(hp, hc)

        return hp, hc

    def apply_tc(self, hp, hc, tc):
        # Apply time shift to account for tc
        # Converting from tc in duration space to actual shift
        _tc = tc - self.sample_length_in_s
        # We do this in polar as well without torch exp
        hp = torch.polar(torch.abs(hp), torch.angle(hp) - 2 * self.PI * self.f * _tc)
        hc = torch.polar(torch.abs(hc), torch.angle(hc) - 2 * self.PI * self.f * _tc)
        return hp, hc

    def pad_missing_frequencies(self, hp, hc):
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
        # Overriding inherited method
        # Derived params different from PhenomD
        M = theta[:, 1:2] + theta[:, 0:1]
        eta = theta[:, 1:2] * theta[:, 0:1] / (M * M)
        q = theta[:, 0:1] / theta[:, 1:2]  # q>=1 due to swapped masses
        M_s = (theta[:, 1:2] + theta[:, 0:1]) * self.GM  # also called m_sec
        # This should prevent NaNs
        nudge_backward_(eta, 0.25, 1e-6)
        nudge_forward_(q, 1.0, 1e-6)

        return torch.cat([M, eta, q, M_s], dim=1)

    def compute_pv2_coeffs(self, theta, derived, converted_spins):
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
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        return x * ca - y * sa, x * sa + y * ca, z

    @staticmethod
    def ROTATEY(angle, x, y, z):
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        return x * ca + z * sa, y, -x * sa + z * ca

    def ComputeNNLOanglecoeffs(self, q, chil, chip):
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
        return af

    def FinalSpin0815(self, eta, chi1, chi2):
        Seta = torch.sqrt(self.ONE - 4.0 * eta)
        m1 = self.HALF * (self.ONE + Seta)
        m2 = self.HALF * (self.ONE - Seta)
        s = (m1 * m1) * chi1 + (m2 * m2) * chi2
        return self.FinalSpin0815_s(eta, s)

    def FinalSpin0815_s(self, eta, S):
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
        Seta = torch.sqrt(self.ONE - 4.0 * eta)
        m1 = self.HALF * (self.ONE + Seta)
        m2 = self.HALF * (self.ONE - Seta)
        m1s = m1 * m1
        m2s = m2 * m2
        s = (m1s * chi1 + m2s * chi2) / (m1s + m2s)

        return self.EradRational0815_s(eta, s)

    def EradRational0815_s(self, eta, s):
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

        t_corr_fixed = (p_next - p_prev) / (f_next - f_prev) / (2 * self.PI)

        # Compute phase correction factor
        phase_corr = torch.exp(self.TWO_J * self.PI * self.f * t_corr_fixed)

        # Apply to waveform, respecting offset
        hptilde[..., offset : offset + self.f_numel] *= phase_corr
        hctilde[..., offset : offset + self.f_numel] *= phase_corr

        return hptilde, hctilde
