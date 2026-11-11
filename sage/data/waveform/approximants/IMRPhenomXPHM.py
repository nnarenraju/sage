#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : IMRPhenomXPHM.py
Description   : Batched PyTorch port of LALSimulation's IMRPhenomXPHM
                precessing frequency-domain waveform with higher modes
                (LALSimIMRPhenomXPHM.c, LALSimIMRPhenomX_precession.c).

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

from sage.core.constants import C, G, MSUN
from sage.core.geometry import Window
from sage.core.math import cbrt, detach_where, safe_sqrt, strong_zero_sqrt
from sage.data.waveform import taper as _taper_mod
from sage.data.waveform import waveform_utils
from sage.data.waveform.approximants.IMRPhenomXHM import IMRPhenomXHM


def _cinv(z):
    """
    Reciprocal 1.0 / z as computed by C complex division (__divdc3).

    Parameters
    ----------
    z : torch.Tensor
        Complex input.

    Returns
    -------
    torch.Tensor
    """
    c, d = z.real, z.imag
    small = torch.abs(c) < torch.abs(d)
    num = torch.where(small, c, d)
    den = torch.where(small, d, c)
    ratio = num / den
    denom = num * ratio + den
    re = torch.where(small, ratio, torch.ones_like(ratio)) / denom
    im = torch.where(small, -torch.ones_like(ratio), -ratio) / denom
    return torch.complex(re, im)


def _detach_rows(mask: torch.Tensor, prec: dict) -> dict:
    """
    Copy of a precession dict with every tensor detached on ``mask``.

    Parameters
    ----------
    mask : torch.Tensor, shape (B, 1)
        Rows to detach.
    prec : dict
        Precession dict; nested dicts and lists of tensors are copied
        the same way.

    Returns
    -------
    dict
        Same keys and forward values as ``prec``.
    """
    out = {}
    for key, value in prec.items():
        if isinstance(value, dict):
            out[key] = _detach_rows(mask, value)
        elif isinstance(value, list):
            out[key] = [detach_where(mask, item) for item in value]
        elif torch.is_tensor(value) and value.is_floating_point():
            out[key] = detach_where(mask, value)
        else:
            out[key] = value
    return out


class IMRPhenomXPHM(IMRPhenomXHM, torch.nn.Module):
    """
    Batched IMRPhenomXPHM precessing multimode BBH waveform.

    ``forward()`` returns detector-frame strain and scaled parameter
    targets; ``from_grid`` builds the model on an explicit grid.

    Parameters
    ----------
    param_sampler : DistributionSampler
        Parameter sampler.
    waveform_project : ConstantProjection
        Multi-detector projection module.
    augment : callable or None
        SNR-rescaling augmentation.
    window : Window
        Window geometry; the grid runs from the low cutoff to Nyquist
        with ``df = 1 / padded_length_in_s``.
    signal_batch_size : int
        Signals per call.
    low_frequency_cutoff_in_hz, reference_frequency_in_hz : float
        Low cutoff and reference frequency.
    targets : list of str
        Estimated parameters, in head order (must include ``mchirp``).
    device : str or torch.device
        Device of the batches.
    dtype : torch.dtype
        Dtype of the data tensors (the waveform interior is float64).
    append_per_det_targets : bool
        Append per-detector targets after the class column.
    """

    # LAL default settings.
    PREC_VERSION = 223          # MSA angles; default 300 maps to 223
    NNLO_FALLBACK_VERSION = 102 # NNLO angles where the MSA system fails
    EXPANSION_ORDER = 5         # Expansion order of corrections to retain
    CONVENTION = 1              # PhenomXPConvention
    FINAL_SPIN_MOD = 3          # Default 4 maps to 3 outside SpinTaylor

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

        f = waveform_utils.get_freqs(
            low_frequency_cutoff_in_hz,
            window.sample_rate_in_hz / 2.0,
            window.padded_length_in_s,
            signal_batch_size,
            device,
            torch.float64,
        )
        f_ref = waveform_utils.reference_frequencies(reference_frequency_in_hz, signal_batch_size, device)
        IMRPhenomXHM.__init__(self, f, f_ref, padding_length_in_s=window.padding_length_in_s)
        self.sample_length_in_s = float(window.padded_length_in_s)
        self._out_cdtype = torch.promote_types(dtype, torch.complex64)

        self.param_sampler = param_sampler
        self.waveform_project = waveform_project
        self.augment = augment
        self.append_per_det_targets = bool(append_per_det_targets)
        self.mc_pe_idx = list(targets).index("mchirp")

        param_index = self.param_sampler.param_index
        parameters = waveform_utils.PRECESSING_PARAMETERS
        self.source, self.target = waveform_utils.layout_columns(param_index, parameters, type(self).__name__)
        self.extras = waveform_utils.given_extras(param_index, parameters)
        self.dist_col = param_index["distance"]
        self.delta_tc_col = param_index["delta_tc"]

        self.param_sampler.compile_target_scaling(targets, dtype)

        self.param_sampler.to(device)
        self.waveform_project.to(device)

    @classmethod
    def from_grid(cls, f, f_ref, **kwargs):
        """
        The waveform alone, on an explicit frequency grid.

        Parameters
        ----------
        f : torch.Tensor, shape (B, F)
            Frequency grid in Hz.
        f_ref : torch.Tensor, shape (B, 1)
            Reference frequency in Hz.
        **kwargs
            Passed to IMRPhenomXHM.__init__.

        Returns
        -------
        IMRPhenomXPHM
        """
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        IMRPhenomXHM.__init__(model, f, f_ref, **kwargs)
        model.extras = waveform_utils.TIDAL_PARAMETERS
        return model

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
        targets : torch.Tensor
            Standardised parameter targets with a trailing signal-label
            column of ones, then the per-detector targets if enabled.
        all_theta : torch.Tensor, shape (B, total_params)
            Full raw parameter batch (only when ``return_theta=True``).
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
            hf, dt = proj
        else:
            hf = proj

        if self.augment:
            # Strain scales as 1 / distance
            hf, scale = self.augment(hf)
            all_theta[:, self.dist_col] = (
                all_theta[:, self.dist_col] / scale.to(all_theta.dtype)
            )

        normed_targets = self.param_sampler.scale_targets(all_theta)
        targets = torch.cat(
            [normed_targets, torch.ones_like(normed_targets[:, :1])], dim=1
        )

        # Per-detector targets follow the class column
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

    # ------------------------------------------------------------------
    # Input ordering
    # ------------------------------------------------------------------

    # Source: XLALIMRPhenomXPCheckMassesAndSpins
    # (LALSimIMRPhenomXUtilities.c).
    @staticmethod
    def check_masses_and_spins(theta):
        """
        Swap the masses and spin vectors where m1 > m2 does not hold.

        m1 == m2 also takes the swap branch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 12)
            Precessing layout.

        Returns
        -------
        torch.Tensor, shape (B, 12)
        """
        swap = ~(theta[:, 0:1] > theta[:, 1:2])
        if not bool(swap.any()):
            return theta
        out = theta.clone()
        out[:, 0:1] = torch.where(swap, theta[:, 1:2], theta[:, 0:1])
        out[:, 1:2] = torch.where(swap, theta[:, 0:1], theta[:, 1:2])
        out[:, 2:5] = torch.where(swap, theta[:, 5:8], theta[:, 2:5])
        out[:, 5:8] = torch.where(swap, theta[:, 2:5], theta[:, 5:8])
        return out

    # ------------------------------------------------------------------
    # Precession variables
    # ------------------------------------------------------------------

    # Source: IMRPhenomXGetAndSetPrecessionVariables
    # (LALSimIMRPhenomX_precession.c).
    def get_and_set_precession_variables(self, theta, derived, version=None):
        """
        Populate the IMRPhenomXPrecessionStruct, as a dict.

        Rows where the MSA system fails to initialize take the NNLO
        angles (version 102) throughout.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 12)
            Precessing layout.
        derived : torch.Tensor
            Aligned-spin derived parameters.
        version : int or None
            IMRPhenomXPrecVersion; None selects PREC_VERSION.

        Returns
        -------
        dict
            (B, 1) tensors keyed by IMRPhenomXPrecessionStruct member
            names, with the spin-weighted Y_lm at (thetaJN, 0).
        """
        version = self.PREC_VERSION if version is None else version
        if version not in (220, 221, 222, 223, 224):
            prec = self._precession_variables(theta, derived, version)
        else:
            with torch.no_grad():
                msa_error = self._precession_variables(theta, derived, version)[
                    "MSA_ERROR"
                ]
            if not bool(msa_error.any()):
                prec = self._precession_variables(theta, derived, version)
            else:
                # If the MSA system fails to initialize we default to
                # the NNLO angles; each build is detached on the rows
                # it does not serve
                prec = self._precession_variables(
                    detach_where(msa_error, theta),
                    detach_where(msa_error, derived),
                    version,
                )
                prec_nnlo = self._precession_variables(
                    detach_where(~msa_error, theta),
                    detach_where(~msa_error, derived),
                    self.NNLO_FALLBACK_VERSION,
                )
                for key, val in prec_nnlo.items():
                    if key in ("version", "MSA_ERROR"):
                        continue
                    if key == "L_coeffs":
                        prec["L_coeffs"] = {
                            k: torch.where(msa_error, val[k], prec["L_coeffs"][k])
                            for k in val
                        }
                    elif key in prec and torch.is_tensor(val):
                        prec[key] = torch.where(msa_error, val, prec[key])
                    elif key not in prec:
                        # NNLO coefficients, used on MSA_ERROR rows
                        prec[key] = val

        # Spin-weighted spherical harmonics at (thetaJN, 0), l = 2..4
        for ell in (2, 3, 4):
            for emm in range(-ell, ell + 1):
                prec[f"Y{ell}{'m' if emm < 0 else ''}{abs(emm)}"] = \
                    self.spin_weighted_Y(prec["thetaJN"], 0.0, -2, ell, emm)
        return prec

    # Source: IMRPhenomXGetAndSetPrecessionVariables
    # (LALSimIMRPhenomX_precession.c).
    def _precession_variables(
        self, theta: torch.Tensor, derived: torch.Tensor, version: int
    ) -> dict:
        """
        Precession variables for one angle version.

        Excludes the MSA failure fallback and the Y_lm table.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 12)
            Precessing layout.
        derived : torch.Tensor
            Aligned-spin derived parameters.
        version : int
            IMRPhenomXPrecVersion (223 MSA or 102 NNLO).

        Returns
        -------
        dict
            (B, 1) tensors keyed by IMRPhenomXPrecessionStruct member
            names.
        """
        eta = derived[:, 3:4]
        # Normalized masses m_i_SI / Mtot_SI, with pWF->Mtot_SI summed
        # from the masses in solar units
        m1_SI = theta[:, 0:1] * MSUN
        m2_SI = theta[:, 1:2] * MSUN
        Mtot_SI = (m1_SI / MSUN) * MSUN + (m2_SI / MSUN) * MSUN
        m1 = m1_SI / Mtot_SI
        m2 = m2_SI / Mtot_SI
        # pWF->m1 and pWF->m2 in solar units, used by the MSA system
        Mtot_sol = m1_SI / MSUN + m2_SI / MSUN
        M = m1 + m2   # Total mass in solar units
        chi1x = theta[:, 2:3]
        chi1y = theta[:, 3:4]
        chi1z = theta[:, 4:5]
        chi2x = theta[:, 5:6]
        chi2y = theta[:, 6:7]
        chi2z = theta[:, 7:8]
        phiRef = theta[:, 10:11]
        inclination = theta[:, 11:12]

        # Powers of eta
        eta2 = eta * eta
        eta3 = eta * eta2
        eta4 = eta * eta3
        prec = {
            "eta": eta, "eta2": eta2, "eta3": eta3, "eta4": eta4,
            "inveta": 1.0 / eta, "inveta2": 1.0 / eta2,
            "inveta3": 1.0 / eta3, "inveta4": 1.0 / eta4,
            "sqrt_inveta": 1.0 / torch.sqrt(eta),
            "m1": m1, "m2": m2, "M": M,
            "m1_wf": (m1_SI / MSUN) / Mtot_sol, "m2_wf": (m2_SI / MSUN) / Mtot_sol,
            "chi1x": chi1x, "chi1y": chi1y, "chi1z": chi1z,
            "chi2x": chi2x, "chi2y": chi2y, "chi2z": chi2z,
        }
        m1_2 = m1 * m1
        m2_2 = m2 * m2

        prec["piGM"] = math.pi * (m1_SI + m2_SI) * (G / C) / (C * C)
        prec["fRef"] = self.f_ref

        # Dimensionful spin norms, used only squared
        chi1_norm = safe_sqrt(chi1x * chi1x + chi1y * chi1y + chi1z * chi1z)
        chi2_norm = safe_sqrt(chi2x * chi2x + chi2y * chi2y + chi2z * chi2z)
        S1_norm = torch.abs(chi1_norm) * m1_2
        S2_norm = torch.abs(chi2_norm) * m2_2
        prec["S1_norm_2"] = S1_norm * S1_norm
        prec["S2_norm_2"] = S2_norm * S2_norm

        # Get spin projections; chi_p is used only squared
        chi1_perp = safe_sqrt(chi1x * chi1x + chi1y * chi1y)
        chi2_perp = safe_sqrt(chi2x * chi2x + chi2y * chi2y)
        S1_perp = m1_2 * chi1_perp
        S2_perp = m2_2 * chi2_perp
        STot_perpx = m1_2 * chi1x + m2_2 * chi2x
        STot_perpy = m1_2 * chi1y + m2_2 * chi2y
        prec["STot_perp"] = torch.sqrt(STot_perpx ** 2 + STot_perpy ** 2)
        prec["chiTot_perp"] = prec["STot_perp"] * (M * M) / m1_2
        A1 = 2.0 + (3.0 * m2) / (2.0 * m1)
        A2 = 2.0 + (3.0 * m1) / (2.0 * m2)
        ASp1 = A1 * S1_perp
        ASp2 = A2 * S2_perp
        # S_p = max(A1 S1_perp, A2 S2_perp)
        num = torch.where(ASp2 > ASp1, ASp2, ASp1)
        den = A1 * m1_2   # m1 > m2
        prec["chi_p"] = num / den
        prec["SL"] = chi1z * m1_2 + chi2z * m2_2
        prec["Sperp"] = prec["chi_p"] * m1_2

        # Cache the orbital angular momentum coefficients
        chi1L, chi2L = chi1z, chi2z
        delta = derived[:, 4:5]
        eta2 = prec["eta2"]
        pi2 = math.pi * math.pi
        prec["version"] = version
        common_L = {
            "L0": torch.ones_like(eta),
            "L1": torch.zeros_like(eta),
            "L2": 3.0 / 2. + eta / 6.0,
            "L4": (81 + (-57 + eta) * eta) / 24.,
            "L6": (10935 + eta * (-62001 + 1674 * eta + 7 * eta2
                                  + 2214 * pi2)) / 1296.,
            "L7": torch.zeros_like(eta),
            "L8": torch.zeros_like(eta),
            "L8L": torch.zeros_like(eta),
        }
        if version in (222, 223):
            # 3PN orbital angular momentum, non-conserved spin norms
            common_L["L3"] = (-7 * (chi1L + chi2L + chi1L * delta - chi2L * delta)
                              + 5 * (chi1L + chi2L) * eta) / 6.
            common_L["L5"] = (-1650 * (chi1L + chi2L + chi1L * delta - chi2L * delta)
                              + 1336 * (chi1L + chi2L) * eta
                              + 511 * (chi1L - chi2L) * delta * eta
                              + 28 * (chi1L + chi2L) * eta2) / 600.
        else:
            # 3PN orbital angular momentum
            common_L["L3"] = (5 * (chi1L * (-2 - 2 * delta + eta)
                                   + chi2L * (-2 + 2 * delta + eta))) / 6.
            common_L["L5"] = (-7 * (chi1L * (72 + delta * (72 - 31 * eta)
                                             + eta * (-121 + 2 * eta))
                                    + chi2L * (72 + eta * (-121 + 2 * eta)
                                               + delta * (-72 + 31 * eta)))) / 144.
        prec["L_coeffs"] = common_L

        # Reference orbital angular momentum
        v_ref = cbrt(math.pi * derived[:, 2:3] * self.f_ref)
        prec["LRef"] = prec["M"] * prec["M"] * self._lpn_ansatz(v_ref, eta / v_ref, prec["L_coeffs"])

        # Get source frame (*_Sf) J = L + S1 + S2
        J0x = m1_2 * chi1x + m2_2 * chi2x
        J0y = m1_2 * chi1y + m2_2 * chi2y
        J0z = m1_2 * chi1z + m2_2 * chi2z + prec["LRef"]
        J0 = torch.sqrt(J0x * J0x + J0y * J0y + J0z * J0z)
        prec["J0"] = J0
        thetaJ_Sf = torch.where(J0 < 1e-10, torch.zeros_like(J0),
                                torch.acos(J0z / J0))
        prec["thetaJ_Sf"] = thetaJ_Sf

        # Get azimuthal angle of J0 in the source frame
        MAX_TOL_ATAN = 1.0e-15
        aligned_J = (torch.abs(J0x) < MAX_TOL_ATAN) & (torch.abs(J0y) < MAX_TOL_ATAN)
        # Impose the aligned spin limit; atan2 operands replaced there
        phiJ_Sf = torch.where(
            aligned_J,
            torch.zeros_like(J0),
            torch.atan2(
                torch.where(aligned_J, torch.zeros_like(J0y), J0y),
                torch.where(aligned_J, torch.ones_like(J0x), J0x),
            ),
        )
        prec["phiJ_Sf"] = phiJ_Sf
        prec["phi0_aligned"] = -phiJ_Sf
        # Convention 1 sets pWF->phi0 = 0

        # Determine kappa via rotations of N
        Nx_Sf = torch.sin(inclination) * torch.cos(math.pi / 2.0 - phiRef)
        Ny_Sf = torch.sin(inclination) * torch.sin(math.pi / 2.0 - phiRef)
        Nz_Sf = torch.cos(inclination)

        def rot_z(ang, x, y, z):
            c, s = torch.cos(ang), torch.sin(ang)
            return x * c - y * s, x * s + y * c, z

        def rot_y(ang, x, y, z):
            c, s = torch.cos(ang), torch.sin(ang)
            return x * c + z * s, y, -x * s + z * c

        tx, ty, tz = rot_z(-phiJ_Sf, Nx_Sf, Ny_Sf, Nz_Sf)
        tx, ty, tz = rot_y(-thetaJ_Sf, tx, ty, tz)
        # XLALSimIMRPhenomXatan2tol; operands replaced below tolerance
        aligned_N = (torch.abs(tx) < MAX_TOL_ATAN) & (torch.abs(ty) < MAX_TOL_ATAN)
        kappa = torch.where(
            aligned_N,
            torch.zeros_like(tx),
            torch.atan2(
                torch.where(aligned_N, torch.zeros_like(ty), ty),
                torch.where(aligned_N, torch.ones_like(tx), tx),
            ),
        )
        prec["kappa"] = kappa

        # Convention 1: alpha0 = pi - kappa in both branches
        prec["alpha0"] = math.pi - kappa

        # Convention 1: thetaJN from J0 . N
        J0dotN = J0x * Nx_Sf + J0y * Ny_Sf + J0z * Nz_Sf
        thetaJN = torch.acos(J0dotN / J0)
        prec["thetaJN"] = thetaJN
        prec["Nz_Jf"] = torch.cos(thetaJN)
        prec["Nx_Jf"] = torch.sin(thetaJN)

        # Convention 1
        prec["epsilon0"] = phiJ_Sf - math.pi

        # Polar angle zeta of X in the J frame, in the P, Q basis with
        # P = (Nz, 0, -Nx) and Q = (0, 1, 0)
        Xx_Sf = -torch.cos(inclination) * torch.sin(phiRef)
        Xy_Sf = -torch.cos(inclination) * torch.cos(phiRef)
        Xz_Sf = torch.sin(inclination)
        xx, xy, xz = rot_z(-phiJ_Sf, Xx_Sf, Xy_Sf, Xz_Sf)
        xx, xy, xz = rot_y(-thetaJ_Sf, xx, xy, xz)
        xx, xy, xz = rot_z(-kappa, xx, xy, xz)
        XdotPArun = xx * prec["Nz_Jf"] + xz * (-prec["Nx_Jf"])
        XdotQArun = xy
        prec["zeta_polarization"] = torch.atan2(XdotQArun, XdotPArun)

        if version in (220, 221, 222, 223, 224):
            self._initialize_msa_system(theta, derived, prec)
            # Compute and set final spin and RD frequency
            self._set_precessing_remnant_params(prec, derived)
            # alpha and epsilon offsets at omega = pi M fRef, mprime = 2
            vangles = self.msa_phi_zeta_costhetaL(v_ref, derived, prec)
            prec["alpha_offset"] = vangles[0] - prec["alpha0"]
            prec["epsilon_offset"] = vangles[1] - prec["epsilon0"]
        else:
            # Single spin PN Euler angles as per IMRPhenomPv2
            prec["MSA_ERROR"] = torch.zeros_like(eta, dtype=torch.bool)
            self._set_precessing_remnant_params(prec, derived)
            self._nnlo_angle_coeffs(derived, prec)
            # alpha and epsilon offsets at omega = pi M fRef, mprime = 2
            omega_ref = math.pi * derived[:, 2:3] * self.f_ref * 2. / 2
            logw = torch.log(omega_ref)
            wcbrt = cbrt(omega_ref)
            wcbrt2 = wcbrt * wcbrt
            prec["alpha_offset"] = (prec["alpha1"] / omega_ref
                                    + prec["alpha2"] / wcbrt2
                                    + prec["alpha3"] / wcbrt
                                    + prec["alpha4L"] * logw
                                    + prec["alpha5"] * wcbrt - prec["alpha0"])
            prec["epsilon_offset"] = (prec["epsilon1"] / omega_ref
                                      + prec["epsilon2"] / wcbrt2
                                      + prec["epsilon3"] / wcbrt
                                      + prec["epsilon4L"] * logw
                                      + prec["epsilon5"] * wcbrt - prec["epsilon0"])

        return prec

    # Source: IMRPhenomX_SetPrecessingRemnantParams
    # (LALSimIMRPhenomX_precession.c).
    def _set_precessing_remnant_params(self, prec, derived):
        """
        Set remnant quantities related to the precessing final spin.

        Final spin version 3, or version 0 where the MSA system fails;
        fRING and fDAMP are updated at the final spin.

        Parameters
        ----------
        prec : dict
            Precession variables; updated in place.
        derived : torch.Tensor
            Aligned-spin derived parameters.

        Returns
        -------
        dict
            ``prec`` with afinal_prec, afinal, Mfinal, fRING and fDAMP.
        """
        eta = prec["eta"]
        chi1z = prec["chi1z"]
        chi2z = prec["chi2z"]
        m1_2 = prec["m1"] * prec["m1"]
        m2_2 = prec["m2"] * prec["m2"]
        # XLALSimIMRPhenomXFinalSpin2017(eta, chi1z, chi2z)
        STotR = self._STotR(derived[:, 5:6], derived[:, 6:7], chi1z, chi2z)
        af_parallel = self.final_spin_2017(eta, STotR, chi1z - chi2z)
        # pWF->Mfinal keeps its aligned-spin value
        Mfinal = self.final_mass_2017(eta, derived[:, 9:10], derived[:, 10:11])

        M = prec["M"]
        Lfinal = M * M * af_parallel - m1_2 * chi1z - m2_2 * chi2z

        sign = torch.copysign(torch.ones_like(af_parallel), af_parallel)
        # Version 0: XLALSimIMRPhenomXPrecessingFinalSpin2017, with
        # chi_p weighted by q_factor^2, q_factor = m1 / M from eta
        m1_eta = 0.5 * (1.0 + safe_sqrt(1 - 4.0 * eta))
        m2_eta = 0.5 * (1.0 - safe_sqrt(1 - 4.0 * eta))
        q_factor = m1_eta / (m1_eta + m2_eta)
        Sperp0 = prec["chi_p"] * q_factor * q_factor
        af0 = sign * torch.sqrt(Sperp0 * Sperp0 + af_parallel * af_parallel)
        if prec["version"] in (220, 221, 222, 223, 224):
            # Version 3; version 0 where the MSA system fails
            af3 = sign * torch.sqrt(prec["SAv2"] + Lfinal * Lfinal
                                    + 2.0 * Lfinal * (prec["S1L_pav"] + prec["S2L_pav"])) / (M * M)
            afinal = torch.where(prec["MSA_ERROR"], af0, af3)
        else:
            # Version 3 requires an MSA version; defaulting to version 0
            afinal = af0

        afinal = torch.clamp(afinal, -1.0, 1.0)
        prec["afinal_prec"] = afinal
        prec["afinal"] = afinal
        prec["Mfinal"] = Mfinal

        # Update ringdown and damping frequency
        prec["fRING"], prec["fDAMP"] = self.get_fRD_fdamp(afinal, Mfinal)
        return prec

    # ------------------------------------------------------------------
    # Orbital angular momentum
    # ------------------------------------------------------------------

    # Source: XLALSimIMRPhenomXLPNAnsatz
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _lpn_ansatz(v, l_norm, L):
        """
        PN orbital angular momentum.

        L(v) = LNorm (sum_a L_a x^(a/2) + L8L x^4 log(x)), x = v^2.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        l_norm : torch.Tensor
            Orbital angular momentum normalization.
        L : dict of torch.Tensor, shape (B, 1)
            Coefficients L0 to L8 and L8L.

        Returns
        -------
        torch.Tensor, same shape as v
        """
        x = v * v
        x2 = x * x
        x3 = x * x2
        x4 = x * x3
        sqx = torch.sqrt(x)
        return l_norm * (L["L0"] + L["L1"] * sqx + L["L2"] * x + L["L3"] * (x * sqx)
                         + L["L4"] * x2 + L["L5"] * (x2 * sqx) + L["L6"] * x3
                         + L["L7"] * (x3 * sqx) + L["L8"] * x4
                         + L["L8L"] * x4 * torch.log(x))

    # Source: IMRPhenomX_L_norm_3PN_of_v
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _l_norm_3pn_of_v(v, v2, l_norm, prec):
        """
        3PN accurate orbital angular momentum, versions 222 and 223.

        Parameters
        ----------
        v, v2 : torch.Tensor
            Orbital velocity and its square.
        l_norm : torch.Tensor
            Newtonian orbital angular momentum.
        prec : dict
            Precession variables with ``constants_L``.

        Returns
        -------
        torch.Tensor, same shape as v
        """
        cl = prec["constants_L"]
        return l_norm * (1. + v2 * (cl[0] + v * cl[1]
                                    + v2 * (cl[2] + v * cl[3] + v2 * cl[4])))

    # Source: IMRPhenomX_Get_PN_beta (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _pn_beta(a, b, prec):
        """
        PN spin-orbit coupling.

        Parameters
        ----------
        a, b : float or torch.Tensor
            Coupling constants.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        return (prec["dotS1L"] * (a + b * prec["qq"])
                + prec["dotS2L"] * (a + torch.div(b, prec["qq"])))

    # Source: IMRPhenomX_Get_PN_sigma (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _pn_sigma(a, b, prec):
        """
        PN spin-spin coupling sigma.

        Parameters
        ----------
        a, b : float
            Coupling constants.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        return prec["inveta"] * (a * prec["dotS1S2"]
                                 - b * prec["dotS1L"] * prec["dotS2L"])

    # Source: IMRPhenomX_Get_PN_tau (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _pn_tau(a, b, prec):
        """
        PN spin-spin coupling tau.

        Parameters
        ----------
        a, b : float
            Coupling constants.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        return (prec["qq"] * (prec["S1_norm_2"] * a - b * prec["dotS1L"] * prec["dotS1L"])
                + (a * prec["S2_norm_2"] - b * prec["dotS2L"] * prec["dotS2L"]) / prec["qq"]
                ) / prec["eta"]

    # ------------------------------------------------------------------
    # GSL special functions
    # ------------------------------------------------------------------

    # Source: gsl_sf_ellint_RF_e (GSL specfunc/ellint.c).
    @staticmethod
    def _ellint_RF(x, y, z):
        """
        Carlson symmetric elliptic integral R_F(x, y, z).

        Duplication algorithm with errtol = 0.001 (GSL_PREC_DOUBLE).

        Parameters
        ----------
        x, y, z : torch.Tensor
            Arguments, broadcast together.

        Returns
        -------
        torch.Tensor
        """
        errtol = 0.001
        xn, yn, zn = x.clone(), y.clone(), z.clone()
        B = torch.broadcast_tensors(xn, yn, zn)
        xn, yn, zn = (b.clone() for b in B)
        mu_out = torch.zeros_like(xn)
        e2_out = torch.zeros_like(xn)
        e3_out = torch.zeros_like(xn)
        active = torch.ones_like(xn, dtype=torch.bool)
        for _ in range(200):
            mu = (xn + yn + zn) / 3.0
            xndev = 2.0 - (mu + xn) / mu
            yndev = 2.0 - (mu + yn) / mu
            zndev = 2.0 - (mu + zn) / mu
            epslon = torch.maximum(torch.abs(xndev),
                                   torch.maximum(torch.abs(yndev), torch.abs(zndev)))
            newly_done = active & (epslon < errtol)
            mu_out = torch.where(newly_done, mu, mu_out)
            e2_out = torch.where(newly_done, xndev * yndev - zndev * zndev, e2_out)
            e3_out = torch.where(newly_done, xndev * yndev * zndev, e3_out)
            active = active & ~newly_done
            if not active.any():
                break
            xnroot = torch.sqrt(xn)
            ynroot = torch.sqrt(yn)
            znroot = torch.sqrt(zn)
            lamda = xnroot * (ynroot + znroot) + ynroot * znroot
            xn = torch.where(active, (xn + lamda) * 0.25, xn)
            yn = torch.where(active, (yn + lamda) * 0.25, yn)
            zn = torch.where(active, (zn + lamda) * 0.25, zn)
        c1 = 1.0 / 24.0
        c2 = 3.0 / 44.0
        c3 = 1.0 / 14.0
        s = 1.0 + (c1 * e2_out - 0.1 - c2 * e3_out) * e2_out + c3 * e3_out
        return s / torch.sqrt(mu_out)

    # Source: gsl_sf_ellint_F_e (GSL specfunc/ellint.c).
    @classmethod
    def _ellint_F(cls, phi, k):
        """
        Incomplete elliptic integral of the first kind F(phi, k).

        Angular reduction to (-pi/2, pi/2] plus the periodicity terms,
        with the GSL modulus convention.

        Parameters
        ----------
        phi : torch.Tensor
            Amplitude.
        k : torch.Tensor
            Modulus.

        Returns
        -------
        torch.Tensor
        """
        nc = torch.floor(phi / math.pi + 0.5)
        phi_red = phi - nc * math.pi
        sin_phi = torch.sin(phi_red)
        sin2_phi = sin_phi * sin_phi
        x = 1.0 - sin2_phi
        y = 1.0 - k * k * sin2_phi
        one = torch.ones_like(x)
        val = sin_phi * cls._ellint_RF(x, y, one)
        # Add extra terms from periodicity, Kcomp = RF(0, 1 - k^2, 1)
        need_k = nc != 0
        if bool(need_k.any()):
            # GSL adds it only where nc != 0; k is detached elsewhere
            k_need = detach_where(~need_k, k)
            rk = cls._ellint_RF(torch.zeros_like(x), 1.0 - k_need * k_need, one)
            val = torch.where(need_k, val + 2 * nc * rk, val)
        return val

    # Source: gsl_sf_elljac_e (GSL specfunc/elljac.c).
    @staticmethod
    def _elljac_sn(u, m):
        """
        Jacobi elliptic function sn(u | m).

        AGM descent, with the reflected branch where |sin| < |cos|; each
        element starts the descent at its own AGM level.

        Parameters
        ----------
        u, m : torch.Tensor
            Argument and parameter, broadcast together.

        Returns
        -------
        torch.Tensor
        """
        eps = torch.finfo(u.dtype).eps
        # Limits |m| < 2 eps and |m - 1| < 2 eps; inputs detached there
        small_m = torch.abs(m) < 2.0 * eps
        unit_m = torch.abs(m - 1.0) < 2.0 * eps
        u_limit = u
        u = detach_where(small_m | unit_m, u)
        m = detach_where(small_m | unit_m, m)
        N = 16
        mu = [torch.ones_like(m)]
        nu = [torch.sqrt(1.0 - m)]
        # Per-element AGM stopping level n
        n_elem = torch.zeros_like(m, dtype=torch.long)
        done = torch.abs(mu[0] - nu[0]) <= 4.0 * eps * torch.abs(mu[0] + nu[0])
        for k in range(N - 1):
            mu.append(0.5 * (mu[k] + nu[k]))
            nu.append(torch.sqrt(mu[k] * nu[k]))
            newly = (~done) & (torch.abs(mu[k + 1] - nu[k + 1])
                               <= 4.0 * eps * torch.abs(mu[k + 1] + nu[k + 1]))
            n_elem = torch.where(newly, torch.full_like(n_elem, k + 1), n_elem)
            done = done | newly
        mu_st = torch.stack(mu, dim=0)
        mu_n = torch.gather(mu_st, 0, n_elem.unsqueeze(0)).squeeze(0)

        sin_umu = torch.sin(u * mu_n)
        cos_umu = torch.cos(u * mu_n)
        refl = torch.abs(sin_umu) < torch.abs(cos_umu)
        # Only the selected quotient; divisors replaced elsewhere
        t = torch.where(
            refl,
            sin_umu / torch.where(refl, cos_umu, torch.ones_like(cos_umu)),
            cos_umu / torch.where(refl, torch.ones_like(sin_umu), sin_umu),
        )

        # Descent c[k] = d[k+1] c[k+1], d[k] = (r + nu[k])/(r + mu[k]),
        # r = c[k+1]^2 / mu[k+1], started at each element's own level
        c = torch.zeros_like(u + m)
        d = torch.ones_like(u + m)
        for k in range(N - 1, -1, -1):
            init = n_elem == k
            step = n_elem > k
            c_new = d * c
            r = (c * c) / mu[k + 1] if k + 1 < N else torch.zeros_like(c)
            d_new = (r + nu[k]) / (r + mu[k])
            c = torch.where(step, c_new, c)
            d = torch.where(step, d_new, d)
            c = torch.where(init, mu_n * t, c)
            d = torch.where(init, torch.ones_like(d), d)
        sqrt1m = torch.sqrt(1.0 - m)
        sgn = lambda v: torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
        # gsl_hypot(1.0, c) = max sqrt(1 + (min/max)^2)
        big = torch.abs(c) > 1.0
        hmax = torch.where(big, torch.abs(c), torch.ones_like(c))
        hmin = torch.where(big, torch.ones_like(c), torch.abs(c))
        hu = hmin / hmax
        hyp = hmax * torch.sqrt(1 + hu * hu)
        # Reflected branch: dn = sqrt(1-m)/d, cn = dn sign(cos)/hyp,
        # sn = cn c/sqrt(1-m); direct branch: sn = sign(sin)/hyp
        dn_r = sqrt1m / d
        cn_r = dn_r * sgn(cos_umu) / hyp
        sn_r = cn_r * c / sqrt1m
        sn_d = sgn(sin_umu) / hyp
        sn = torch.where(refl, sn_r, sn_d)
        # Limits |m| -> 0 and m -> 1
        sn = torch.where(small_m, torch.sin(u_limit), sn)
        sn = torch.where(unit_m, torch.tanh(u_limit), sn)
        return sn

    # ------------------------------------------------------------------
    # MSA system
    # ------------------------------------------------------------------

    # Source: IMRPhenomX_Initialize_MSA_System
    # (LALSimIMRPhenomX_precession.c).
    def _initialize_msa_system(self, theta, derived, prec):
        """
        Initialize the MSA system at the reference frequency.

        Sets MSA_ERROR where |Omegaz5| > 1000.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 12)
            Precessing layout.
        derived : torch.Tensor
            Aligned-spin derived parameters.
        prec : dict
            Precession variables; updated in place with (B, 1) tensors.
        """
        eta = prec["eta"]
        eta2 = prec["eta2"]
        m1 = prec["m1_wf"]
        m2 = prec["m2_wf"]
        PI = math.pi

        # PN coefficients for d omega / d t
        LN2 = math.log(2.0)
        GAMMA = 0.577215664901532860606512090082402431
        dNS = (96. / 5., -1486. / 35., -264. / 5., 384. * PI / 5., 34103. / 945.,
               13661. / 105., 944. / 15., PI * (-4159. / 35.), PI * (-2268. / 5.),
               (16447322263. / 7276500. + PI * PI * 512. / 5. - LN2 * 109568. / 175.
                - GAMMA * 54784. / 175.),
               (-56198689. / 11340. + PI * PI * 902. / 5.), 1623. / 140.,
               -1121. / 27., -54784. / 525., -PI * 883. / 42., PI * 71735. / 63.,
               PI * 73196. / 63.)
        dSO = (-904. / 5., -120., -62638. / 105., 4636. / 5., -6472. / 35.,
               3372. / 5., -PI * 720., -PI * 2416. / 5., -208520. / 63.,
               796069. / 105., -100019. / 45., -1195759. / 945., 514046. / 105.,
               -8709. / 5., -PI * 307708. / 105., PI * 44011. / 7.,
               -PI * 7992. / 7., PI * 151449. / 35.)
        dSS = (-494. / 5., -1442. / 5., -233. / 5., -719. / 5.)
        Lns = (3. / 2., 1. / 6., 27. / 8., -19. / 8., 1. / 24., 135. / 16.,
               -6889 / 144. + 41. / 24. * PI * PI, 31. / 24., 7. / 1296.)
        Lso = (-14. / 6., -3. / 2., -11. / 2., 133. / 72., -33. / 8., 7. / 4.)

        # q = m2 / m1 < 1, m1 > m2
        q = m2 / m1
        invq = 1.0 / q
        prec["qq"] = q
        prec["invqq"] = invq
        mu = (m1 * m2) / (m1 + m2)
        prec["delta_qq"] = (1.0 - q) / (1.0 + q)
        prec["delta2_qq"] = prec["delta_qq"] ** 2
        prec["delta3_qq"] = prec["delta_qq"] * prec["delta2_qq"]
        prec["delta4_qq"] = prec["delta_qq"] * prec["delta3_qq"]

        # Dimensionful spin vectors, eta / q = m1^2 and eta * q = m2^2
        S1x = prec["chi1x"] * eta / q
        S1y = prec["chi1y"] * eta / q
        S1z = prec["chi1z"] * eta / q
        S2x = prec["chi2x"] * eta * q
        S2y = prec["chi2y"] * eta * q
        S2z = prec["chi2z"] * eta * q
        S1_0_norm = torch.sqrt(S1x * S1x + S1y * S1y + S1z * S1z)
        S2_0_norm = torch.sqrt(S2x * S2x + S2y * S2y + S2z * S2z)

        # Reference velocity and orbital angular momentum along z
        v_0 = cbrt(prec["piGM"] * prec["fRef"])
        v_0_2 = v_0 * v_0
        prec["v_0"] = v_0
        prec["v_0_2"] = v_0_2
        L0z = eta / v_0

        # Inner products used in MSA system
        dotS1L = S1z
        dotS2L = S2z
        dotS1S2 = S1x * S2x + S1y * S2y + S1z * S2z
        prec["dotS1L"] = dotS1L
        prec["dotS2L"] = dotS2L
        prec["dotS1S2"] = dotS1S2
        prec["dotS1Ln"] = dotS1L / S1_0_norm
        prec["dotS2Ln"] = dotS2L / S2_0_norm

        # Coefficients for PN orbital angular momentum at 3PN
        prec["constants_L"] = [
            Lns[0] + eta * Lns[1],
            self._pn_beta(Lso[0], Lso[1], prec),
            Lns[2] + eta * Lns[3] + eta * eta * Lns[4],
            self._pn_beta(Lso[2] + Lso[3] * eta, Lso[4] + Lso[5] * eta, prec),
            Lns[5] + Lns[6] * eta + Lns[7] * eta * eta + Lns[8] * eta * eta * eta,
        ]

        # Effective total spin
        Seff = (1.0 + q) * dotS1L + (1 + invq) * dotS2L
        prec["Seff"] = Seff
        prec["Seff2"] = Seff * Seff

        # Initial total spin S = S1 + S2 and J = L + S1 + S2
        S0x, S0y, S0z = S1x + S2x, S1y + S2y, S1z + S2z
        prec["S_0_norm"] = safe_sqrt(S0x * S0x + S0y * S0y + S0z * S0z)
        prec["S_0_norm_2"] = prec["S_0_norm"] ** 2
        prec["L_0_norm"] = L0z
        J0x, J0y, J0z = S0x, S0y, L0z + S0z
        prec["J_0_norm"] = torch.sqrt(J0x * J0x + J0y * J0y + J0z * J0z)
        L0norm = prec["L_0_norm"]
        J0norm = prec["J_0_norm"]

        # Get roots to S^2 equation: S^2_+, S^2_-, S^2_3
        S32, Smi2, Spl2 = self._msa_roots(L0norm, J0norm, prec)
        prec["Spl2"] = Spl2
        prec["Smi2"] = Smi2
        prec["S32"] = S32
        prec["Spl2pSmi2"] = Spl2 + Smi2
        prec["Spl2mSmi2"] = Spl2 - Smi2
        # S_+ and S_-; S_+ is used only by the corrections
        prec["Spl"] = strong_zero_sqrt(Spl2)
        prec["Smi"] = torch.sqrt(Smi2)
        prec["SAv2"] = 0.5 * prec["Spl2pSmi2"]
        prec["SAv"] = torch.sqrt(prec["SAv2"])
        prec["invSAv2"] = 1.0 / prec["SAv2"]
        prec["invSAv"] = 1.0 / prec["SAv"]

        c_1 = 0.5 * (J0norm * J0norm - L0norm * L0norm - prec["SAv2"]) / L0norm * eta
        c1_2 = c_1 * c_1
        prec["c1"] = c_1
        prec["c12"] = c1_2
        prec["c1_over_eta"] = c_1 / eta

        # Precession averaged spin couplings, at the initial values
        omqsq = (1.0 - q) * (1.0 - q) + 1e-16
        omq2 = (1.0 - q * q) + 1e-16
        S1L_pav = (c_1 * (1.0 + q) - q * eta * Seff) / (eta * omq2)
        S2L_pav = -q * (c_1 * (1.0 + q) - eta * Seff) / (eta * omq2)
        S1S2_pav = 0.5 * prec["SAv2"] - 0.5 * (prec["S1_norm_2"] + prec["S2_norm_2"])
        RmSq_v02 = prec["Spl2mSmi2"] * prec["Spl2mSmi2"] * v_0_2
        S1Lsq_pav = S1L_pav * S1L_pav + RmSq_v02 / (32.0 * eta2 * omqsq)
        S2Lsq_pav = S2L_pav * S2L_pav + q * q * RmSq_v02 / (32.0 * eta2 * omqsq)
        prec["S1L_pav"] = S1L_pav
        prec["S2L_pav"] = S2L_pav

        # Spin couplings
        beta3 = ((113. / 12.) + (25. / 4.) * (m2 / m1)) * S1L_pav \
            + ((113. / 12.) + (25. / 4.) * (m1 / m2)) * S2L_pav
        beta5 = (((31319. / 1008.) - (1159. / 24.) * eta)
                 + (m2 / m1) * ((809. / 84) - (281. / 8.) * eta)) * S1L_pav \
            + (((31319. / 1008.) - (1159. / 24.) * eta)
               + (m1 / m2) * ((809. / 84) - (281. / 8.) * eta)) * S2L_pav
        sigma4 = (1.0 / mu) * ((247. / 48.) * S1S2_pav - (721. / 48.) * S1L_pav * S2L_pav) \
            + (1.0 / (m1 * m1)) * ((233. / 96.) * prec["S1_norm_2"] - (719. / 96.) * S1Lsq_pav) \
            + (1.0 / (m2 * m2)) * ((233. / 96.) * prec["S2_norm_2"] - (719. / 96.) * S2Lsq_pav)

        # PN coefficients from precession-averaged spin couplings, then
        # from initial spin couplings for versions 222 and 223
        a0 = 96.0 * eta / 5.0
        a2 = (-(743. / 336.) - (11.0 / 4.) * eta) * a0
        a3 = (4.0 * PI - beta3) * a0
        a4 = ((34103. / 18144.) + (13661. / 2016.) * eta + (59. / 18.) * eta2 - sigma4) * a0
        a5 = (-(4159. / 672.) * PI - (189. / 8.) * PI * eta - beta5) * a0
        a0 = eta * dNS[0]
        a2 = eta * (dNS[1] + eta * dNS[2])
        a3 = eta * (dNS[3] + self._pn_beta(dSO[0], dSO[1], prec))
        a4 = eta * (dNS[4] + eta * (dNS[5] + eta * dNS[6])
                    + self._pn_sigma(dSS[0], dSS[1], prec)
                    + self._pn_tau(dSS[2], dSS[3], prec))
        a5 = eta * (dNS[7] + eta * dNS[8]
                    + self._pn_beta(dSO[2] + eta * dSO[3], dSO[4] + eta * dSO[5], prec))

        a0_2 = a0 * a0
        a0_3 = a0_2 * a0
        a2_2 = a2 * a2
        g0 = 1. / a0
        g2 = -(a2 / a0_2)
        g3 = -(a3 / a0_2)
        g4 = -(a4 * a0 - a2_2) / a0_3
        g5 = -(a5 * a0 - 2.0 * a3 * a2) / a0_3
        prec["g0"] = g0
        prec["g2"] = g2
        prec["g3"] = g3
        prec["g4"] = g4
        prec["g5"] = g5

        delta = prec["delta_qq"]
        delta2 = delta * delta
        prec["psi1"] = 3.0 * (2.0 * eta2 * Seff - c_1) / (eta * delta2)

        c_1_over_nu = prec["c1_over_eta"]
        c_1_over_nu_2 = c_1_over_nu * c_1_over_nu
        one_p_q_sq = (1. + q) * (1. + q)
        Seff_2 = Seff * Seff
        q_2 = q * q
        one_m_q_sq = (1. - q) * (1. - q)
        one_m_q2_2 = (1. - q_2) * (1. - q_2)
        one_m_q_4 = one_m_q_sq * one_m_q_sq

        # Delta term as in LALSimInspiralFDPrecAngles.c
        Del1 = 4. * c_1_over_nu_2 * one_p_q_sq
        Del2 = 8. * c_1_over_nu * q * (1. + q) * Seff
        Del3 = 4. * (one_m_q2_2 * prec["S1_norm_2"] - q_2 * Seff_2)
        Del4 = 4. * c_1_over_nu_2 * q_2 * one_p_q_sq
        Del5 = 8. * c_1_over_nu * q_2 * (1. + q) * Seff
        Del6 = 4. * (one_m_q2_2 * prec["S2_norm_2"] - q_2 * Seff_2)
        # Zero gradient where Delta is unused, e.g. a non-spinning body
        Delta = strong_zero_sqrt(torch.abs((Del1 - Del2 - Del3) * (Del4 - Del5 - Del6)))
        prec["Delta"] = Delta

        # psi2 as in LALSimInspiralFDPrecAngles.c
        u1 = 3. * g2 / g0
        u2 = 0.75 * one_p_q_sq / one_m_q_4
        u3 = -20. * c_1_over_nu_2 * q_2 * one_p_q_sq
        u4 = 2. * one_m_q2_2 * (q * (2. + q) * prec["S1_norm_2"]
                                + (1. + 2. * q) * prec["S2_norm_2"] - 2. * q * prec["SAv2"])
        u5 = 2. * q_2 * (7. + 6. * q + 7. * q_2) * 2. * c_1_over_nu * Seff
        u6 = 2. * q_2 * (3. + 4. * q + 3. * q_2) * Seff_2
        u7 = q * Delta
        prec["psi2"] = u1 + u2 * (u3 + u4 + u5 - u6 + u7)

        Rm = Spl2 - Smi2
        Rm_2 = Rm * Rm
        cp = Spl2 * eta2 - c1_2
        cm = Smi2 * eta2 - c1_2
        # fabs is here to help enforce positive definite cpcm
        cpcm = torch.abs(cp * cm)
        sqrt_cpcm = torch.sqrt(cpcm)
        a1dD = 0.5 + torch.div(0.75, eta)
        a2dD = -0.75 * Seff / eta
        D2RmSq = (cp - sqrt_cpcm) / eta2
        eta4 = prec["eta4"]
        D4RmSq = -0.5 * Rm * sqrt_cpcm / eta2 - cp / eta4 * (sqrt_cpcm - cp)
        S0m = prec["S1_norm_2"] - prec["S2_norm_2"]
        aw = (-3. * (1. + q) / q * (2. * (1. + q) * eta2 * Seff * c_1
                                    - (1. + q) * c1_2 + (1. - q) * eta2 * S0m))
        cw = torch.div(3. / 32., eta) * Rm_2
        dw = 4.0 * cp - 4.0 * D2RmSq * eta2
        hw = -2.0 * (2.0 * D2RmSq - Rm) * c_1
        fw = Rm * D2RmSq - D4RmSq - 0.25 * Rm_2
        adD = aw / dw
        hdD = hw / dw
        cdD = cw / dw
        fdD = fw / dw
        gw = torch.div(3. / 16., eta2) / eta * Rm_2 * (c_1 - eta2 * Seff)
        gdD = gw / dw
        hdD_2 = hdD * hdD
        adDfdD = adD * fdD
        adDfdDhdD = adDfdD * hdD
        adDhdD_2 = adD * hdD_2

        Omegaz0 = a1dD + adD
        Omegaz1 = a2dD - adD * Seff - adD * hdD
        Omegaz2 = adD * hdD * Seff + cdD - adD * fdD + adD * hdD_2
        Omegaz3 = (adDfdD - cdD - adDhdD_2) * (Seff + hdD) + adDfdDhdD
        Omegaz4 = (cdD + adDhdD_2 - 2.0 * adDfdD) * (hdD * Seff + hdD_2 - fdD) \
            - adD * fdD * fdD
        Omegaz5 = (cdD - adDfdD + adDhdD_2) * fdD * (Seff + 2.0 * hdD) \
            - (cdD + adDhdD_2 - 2.0 * adDfdD) * hdD_2 * (Seff + hdD) \
            - adDfdD * fdD * hdD

        # If |Omegaz5| > 1000 the system may be pathological
        prec["MSA_ERROR"] = torch.abs(Omegaz5) > 1000.0

        # Coefficients of phiz
        prec["Omegaz0_coeff"] = 3.0 * g0 * Omegaz0
        prec["Omegaz1_coeff"] = 3.0 * g0 * Omegaz1
        prec["Omegaz2_coeff"] = 3.0 * (g0 * Omegaz2 + g2 * Omegaz0)
        prec["Omegaz3_coeff"] = 3.0 * (g0 * Omegaz3 + g2 * Omegaz1 + g3 * Omegaz0)
        prec["Omegaz4_coeff"] = 3.0 * (g0 * Omegaz4 + g2 * Omegaz2 + g3 * Omegaz1
                                       + g4 * Omegaz0)
        prec["Omegaz5_coeff"] = 3.0 * (g0 * Omegaz5 + g2 * Omegaz3 + g3 * Omegaz2
                                       + g4 * Omegaz1 + g5 * Omegaz0)

        # Coefficients of zeta
        c1oveta2 = c_1 / eta2
        Omegazeta0 = Omegaz0
        Omegazeta1 = Omegaz1 + Omegaz0 * c1oveta2
        Omegazeta2 = Omegaz2 + Omegaz1 * c1oveta2
        Omegazeta3 = Omegaz3 + Omegaz2 * c1oveta2 + gdD
        Omegazeta4 = Omegaz4 + Omegaz3 * c1oveta2 - gdD * Seff - gdD * hdD
        Omegazeta5 = Omegaz5 + Omegaz4 * c1oveta2 + gdD * hdD * Seff \
            + gdD * (hdD_2 - fdD)
        prec["Omegazeta0_coeff"] = -g0 * Omegazeta0
        prec["Omegazeta1_coeff"] = -1.5 * g0 * Omegazeta1
        prec["Omegazeta2_coeff"] = -3.0 * (g0 * Omegazeta2 + g2 * Omegazeta0)
        prec["Omegazeta3_coeff"] = 3.0 * (g0 * Omegazeta3 + g2 * Omegazeta1
                                          + g3 * Omegazeta0)
        prec["Omegazeta4_coeff"] = 3.0 * (g0 * Omegazeta4 + g2 * Omegazeta2
                                          + g3 * Omegazeta1 + g4 * Omegazeta0)
        prec["Omegazeta5_coeff"] = 1.5 * (g0 * Omegazeta5 + g2 * Omegazeta3
                                          + g3 * Omegazeta2 + g4 * Omegazeta1
                                          + g5 * Omegazeta0)

        # Expansion order of corrections to retain; -1 keeps all orders
        if self.EXPANSION_ORDER != -1:
            for n in range(self.EXPANSION_ORDER, 6):
                if n >= 1:
                    prec[f"Omegaz{n}_coeff"] = torch.zeros_like(eta)
                    prec[f"Omegazeta{n}_coeff"] = torch.zeros_like(eta)

        # Get psi0 term; inputs detached where it is not evaluated
        near_equal = torch.abs(Smi2 - Spl2) < 1.0e-5
        with torch.no_grad():
            tmpB = (prec["S_0_norm"] ** 2 - Spl2) / (Smi2 - Spl2)
            accepted = (
                ((tmpB >= 0.0) & (tmpB <= 1.0))
                | ((tmpB > 1.0) & (tmpB - 1.0 < 0.00001))
                | ((tmpB < 0.0) & (tmpB > -0.00001))
            )
        psi0_skip = near_equal | ~accepted
        Smi2_g, Spl2_g, S32_g, S_0_g, v_0_g = (
            detach_where(psi0_skip, t)
            for t in (Smi2, Spl2, S32, prec["S_0_norm"], v_0)
        )
        prec_g = {
            key: detach_where(psi0_skip, prec[key])
            for key in ("psi1", "psi2", "g0", "delta_qq")
        }
        mm = torch.sqrt((Smi2_g - Spl2_g) / (S32_g - Spl2_g))
        tmpB = (S_0_g ** 2 - Spl2_g) / (Smi2_g - Spl2_g)
        volume_element = (L0z * (S1x * S2y - S1y * S2x))
        vol_sign = torch.where(volume_element > 0, torch.ones_like(eta),
                               torch.where(volume_element < 0,
                                           -torch.ones_like(eta),
                                           torch.zeros_like(eta)))
        prec["psi0"] = torch.zeros_like(eta)
        psi_of_v0 = self._msa_psiofv(v_0_g, v_0_g * v_0_g, prec["psi0"],
                                     prec_g["psi1"], prec_g["psi2"], prec_g)
        # tmpB clamped to [0, 1] within tolerance 1e-5
        tmpB_cl = torch.where((tmpB > 1.0) & (tmpB - 1. < 0.00001),
                              torch.ones_like(tmpB), tmpB)
        tmpB_cl = torch.where((tmpB < 0.0) & (tmpB > -0.00001),
                              torch.zeros_like(tmpB), tmpB_cl)
        tmpB_valid = (tmpB_cl >= 0.) & (tmpB_cl <= 1.)
        arg = torch.where(tmpB_valid, tmpB_cl, torch.zeros_like(tmpB_cl))
        psi0_val = self._ellint_F(torch.asin(vol_sign * torch.sqrt(arg)), mm) - psi_of_v0
        prec["psi0"] = torch.where(near_equal | ~tmpB_valid,
                                   torch.zeros_like(eta), psi0_val)

        # Initial phiz and zeta; corrections if |Spl2 - Smi2| > 1e-5
        prec["phiz_0"] = torch.zeros_like(eta)
        prec["zeta_0"] = torch.zeros_like(eta)
        corr_skip = ~(torch.abs(Spl2 - Smi2) > 1.e-5)
        corr_x, corr_y = self._msa_corrections(v_0, L0norm, J0norm, prec, corr_skip)
        phiz_0 = self._msa_phiz(v_0, J0norm, prec)
        zeta_0 = self._msa_zeta(v_0, prec)
        prec["phiz_0"] = -phiz_0 - corr_x
        prec["zeta_0"] = -zeta_0 - corr_y

    # -- MSA angles per frequency --------------------------------------

    # Source: IMRPhenomX_JNorm_MSA (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_j_norm(l_norm, prec):
        """
        Norm of J, sqrt(L^2 + 2 L c1 / eta + SAv2).

        Parameters
        ----------
        l_norm : torch.Tensor
            Norm of L.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor, same shape as l_norm
        """
        return torch.sqrt(l_norm * l_norm + 2.0 * l_norm * prec["c1_over_eta"]
                          + prec["SAv2"])

    # Source: IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_spin_evolution_coefficients(l_norm, j_norm, prec):
        """
        Coefficients B, C and D of the cubic in S^2.

        Parameters
        ----------
        l_norm, j_norm : torch.Tensor
            Norms of L and J.
        prec : dict
            Precession variables.

        Returns
        -------
        B, C, D : torch.Tensor
        """
        JNorm2 = j_norm * j_norm
        LNorm2 = l_norm * l_norm
        S1Norm2 = prec["S1_norm_2"]
        S2Norm2 = prec["S2_norm_2"]
        q = prec["qq"]
        eta = prec["eta"]
        J2mL2 = JNorm2 - LNorm2
        J2mL2Sq = J2mL2 * J2mL2
        delta = prec["delta_qq"]
        Seff = prec["Seff"]
        B = (LNorm2 + S1Norm2) * q + 2.0 * l_norm * Seff - 2.0 * JNorm2 \
            - S1Norm2 - S2Norm2 + (LNorm2 + S2Norm2) / q
        C = J2mL2Sq - 2.0 * l_norm * Seff * J2mL2 \
            - 2.0 * ((1.0 - q) / q) * LNorm2 * (S1Norm2 - q * S2Norm2) \
            + 4.0 * eta * LNorm2 * Seff * Seff \
            - 2.0 * delta * (S1Norm2 - S2Norm2) * Seff * l_norm \
            + 2.0 * ((1.0 - q) / q) * (q * S1Norm2 - S2Norm2) * JNorm2
        D = ((1.0 - q) / q) * (S2Norm2 - q * S1Norm2) * J2mL2Sq \
            + delta * delta * (S1Norm2 - S2Norm2) * (S1Norm2 - S2Norm2) * LNorm2 / eta \
            + 2.0 * delta * l_norm * Seff * (S1Norm2 - S2Norm2) * J2mL2
        return B, C, D

    # Source: IMRPhenomX_Return_Roots_MSA
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_cubic_angle(
        B: torch.Tensor, C: torch.Tensor, D: torch.Tensor
    ) -> tuple:
        """
        Terms sqrt(-p/3) and acos(arg)/3 of the cubic in S^2.

        The acos argument is clamped to [-1, 1].

        Parameters
        ----------
        B, C, D : torch.Tensor
            Cubic coefficients from _msa_spin_evolution_coefficients.

        Returns
        -------
        sqrtarg, theta : torch.Tensor
        """
        B2 = B * B
        B3 = B2 * B
        BC = B * C
        p = C - B2 / 3
        qc = (2.0 / 27.0) * B3 - BC / 3.0 + D
        sqrtarg = torch.sqrt(-p / 3.0)
        acosarg = torch.clamp(1.5 * qc / p / sqrtarg, -1.0, 1.0)
        return sqrtarg, torch.acos(acosarg) / 3.0

    # Source: IMRPhenomX_Return_Roots_MSA
    # (LALSimIMRPhenomX_precession.c).
    @classmethod
    def _msa_roots(cls, l_norm, j_norm, prec):
        """
        Roots S32, Smi2 and Spl2 of the cubic in S^2.

        Where the solution is NaN or a spin is aligned or zero: S32 = 0,
        Smi2 = S0^2 and Spl2 = Smi2 + 1e-9, with detached inputs.

        Parameters
        ----------
        l_norm, j_norm : torch.Tensor
            Norms of L and J.
        prec : dict
            Precession variables.

        Returns
        -------
        S32, Smi2, Spl2 : torch.Tensor
        """
        B, C, D = cls._msa_spin_evolution_coefficients(l_norm, j_norm, prec)
        with torch.no_grad():
            sqrtarg, theta = cls._msa_cubic_angle(B, C, D)
        # Degenerate cases
        degen = (torch.isnan(theta) | torch.isnan(sqrtarg)
                 | (prec["dotS1Ln"] == 1) | (prec["dotS2Ln"] == 1)
                 | (prec["dotS1Ln"] == -1) | (prec["dotS2Ln"] == -1)
                 | (prec["S1_norm_2"] == 0) | (prec["S2_norm_2"] == 0))
        B, C, D = (detach_where(degen, t) for t in (B, C, D))
        sqrtarg, theta = cls._msa_cubic_angle(B, C, D)
        cos_theta = torch.cos(theta)
        two_pi_2_3 = 2.0 * 2.0 * math.pi / 3.0
        pi_2_3 = 2.0 * math.pi / 3.0
        tmp1 = 2.0 * sqrtarg * torch.cos(theta - two_pi_2_3) - B / 3.0
        tmp2 = 2.0 * sqrtarg * torch.cos(theta - pi_2_3) - B / 3.0
        tmp3 = 2.0 * sqrtarg * cos_theta - B / 3.0
        tmp4 = torch.maximum(torch.maximum(tmp1, tmp2), tmp3)
        tmp5 = torch.minimum(torch.minimum(tmp1, tmp2), tmp3)
        # Find the remaining root, neither the max nor the min
        m3 = ((tmp4 - tmp3) > 0.0) & ((tmp5 - tmp3) < 0.0)
        m1_ = ((tmp4 - tmp1) > 0.0) & ((tmp5 - tmp1) < 0.0)
        tmp6 = torch.where(m3, tmp3, torch.where(m1_, tmp1, tmp2))
        Spl2 = torch.abs(tmp4)
        S32 = tmp5
        Smi2 = torch.abs(tmp6)
        S32 = torch.where(degen, torch.zeros_like(S32), S32)
        Smi2 = torch.where(degen, prec["S_0_norm_2"], Smi2)
        Spl2 = torch.where(degen, prec["S_0_norm_2"] + 1e-9, Spl2)
        return S32, Smi2, Spl2

    # Source: IMRPhenomX_psiofv (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_psiofv(v, v2, psi0, psi1, psi2, prec):
        """
        MSA phase psi at velocity v.

        psi = psi0 - 0.75 g0 delta_qq (1 + psi1 v + psi2 v^2) / v^3.

        Parameters
        ----------
        v, v2 : torch.Tensor
            Orbital velocity and its square.
        psi0, psi1, psi2 : torch.Tensor
            Phase coefficients.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor, same shape as v
        """
        return psi0 - 0.75 * prec["g0"] * prec["delta_qq"] \
            * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)

    # Source: IMRPhenomX_Return_SNorm_MSA
    # (LALSimIMRPhenomX_precession.c).
    @classmethod
    def _msa_s_norm(cls, v, roots, prec):
        """
        Norm of S, sqrt(Spl2 + (Smi2 - Spl2) sn^2(psi | m)).

        sn = 0 where |Smi2 - Spl2| < 1e-5, with detached inputs there.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        roots : tuple of torch.Tensor
            S32, Smi2 and Spl2.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor
        """
        S32, Smi2, Spl2 = roots
        sn_skip = torch.abs(Smi2 - Spl2) < 1.0e-5
        S32_g, Smi2_g, Spl2_g, v_g = (
            detach_where(sn_skip, t) for t in (S32, Smi2, Spl2, v)
        )
        prec_g = {
            key: detach_where(sn_skip, prec[key])
            for key in ("psi0", "psi1", "psi2", "g0", "delta_qq")
        }
        m = (Smi2_g - Spl2_g) / (S32_g - Spl2_g)
        psi = cls._msa_psiofv(v_g, v_g * v_g, prec_g["psi0"], prec_g["psi1"],
                              prec_g["psi2"], prec_g)
        sn = cls._elljac_sn(psi, m)
        sn = torch.where(sn_skip, torch.zeros_like(sn), sn)
        SNorm2 = Spl2 + (Smi2 - Spl2) * sn * sn
        # Used only squared
        return safe_sqrt(SNorm2)

    # Precession variables read by _msa_phiz_terms
    MSA_PHIZ_KEYS = (
        "eta", "eta2", "eta4", "inveta", "inveta2", "inveta3", "inveta4", "c1",
        "SAv", "SAv2", "invSAv", "invSAv2", "Omegaz0_coeff", "Omegaz1_coeff",
        "Omegaz2_coeff", "Omegaz3_coeff", "Omegaz4_coeff", "Omegaz5_coeff",
        "phiz_0",
    )

    # Source: IMRPhenomX_Return_phiz_MSA
    # (LALSimIMRPhenomX_precession.c).
    @classmethod
    def _msa_phiz(cls, v, j_norm, prec):
        """
        Azimuthal precession angle phiz at velocity v.

        NaN entries are set to 0, with detached inputs there.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        j_norm : torch.Tensor
            Norm of J.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor
        """
        phiz_out = cls._msa_phiz_terms(v, j_norm, prec)
        nan = torch.isnan(phiz_out)
        if bool(nan.any()):
            phiz_out = cls._msa_phiz_terms(
                detach_where(nan, v),
                detach_where(nan, j_norm),
                {key: detach_where(nan, prec[key]) for key in cls.MSA_PHIZ_KEYS},
            )
        return torch.where(nan, torch.zeros_like(phiz_out), phiz_out)

    # Source: IMRPhenomX_Return_phiz_MSA
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_phiz_terms(v, j_norm, prec):
        """
        Azimuthal precession angle phiz, before NaN handling.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        j_norm : torch.Tensor
            Norm of J.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor
        """
        invv = 1.0 / v
        invv2 = invv * invv
        eta = prec["eta"]
        LNewt = eta / v
        c1 = prec["c1"]
        c12 = c1 * c1
        SAv2 = prec["SAv2"]
        SAv = prec["SAv"]
        invSAv = prec["invSAv"]
        invSAv2 = prec["invSAv2"]
        inveta = prec["inveta"]
        inveta2 = prec["inveta2"]
        inveta3 = prec["inveta3"]
        inveta4 = prec["inveta4"]
        eta2 = prec["eta2"]
        log1 = torch.log(torch.abs(c1 + j_norm * eta + eta * LNewt))
        log2 = torch.log(torch.abs(c1 + j_norm * SAv * v + SAv2 * v))
        phiz_0_coeff = (j_norm * inveta4) * (0.5 * c12 - c1 * eta2 * invv / 6.0
                                             - SAv2 * eta2 / 3.0 - prec["eta4"] * invv2 / 3.0) \
            - (c1 * 0.5 * inveta) * (c12 * inveta4 - SAv2 * inveta2) * log1
        phiz_1_coeff = -0.5 * j_norm * inveta2 * (c1 + eta * LNewt) \
            + 0.5 * inveta3 * (c12 - eta2 * SAv2) * log1
        phiz_2_coeff = -j_norm + SAv * log2 - c1 * log1 * inveta
        phiz_3_coeff = j_norm * v - eta * log1 + c1 * log2 * invSAv
        phiz_4_coeff = (0.5 * j_norm * invSAv2 * v) * (c1 + v * SAv2) \
            - (0.5 * invSAv2 * invSAv) * (c12 - eta2 * SAv2) * log2
        phiz_5_coeff = -j_norm * v * ((0.5 * c12 * invSAv2 * invSAv2)
                                      - (c1 * v * invSAv2 / 6.0) - v * v / 3.0
                                      - eta2 * invSAv2 / 3.0) \
            + (0.5 * c1 * invSAv2 * invSAv2 * invSAv) * (c12 - eta2 * SAv2) * log2
        phiz_out = (phiz_0_coeff * prec["Omegaz0_coeff"]
                    + phiz_1_coeff * prec["Omegaz1_coeff"]
                    + phiz_2_coeff * prec["Omegaz2_coeff"]
                    + phiz_3_coeff * prec["Omegaz3_coeff"]
                    + phiz_4_coeff * prec["Omegaz4_coeff"]
                    + phiz_5_coeff * prec["Omegaz5_coeff"]
                    + prec["phiz_0"])
        return phiz_out

    # Precession variables read by _msa_zeta_terms
    MSA_ZETA_KEYS = (
        "eta", "Omegazeta0_coeff", "Omegazeta1_coeff", "Omegazeta2_coeff",
        "Omegazeta3_coeff", "Omegazeta4_coeff", "Omegazeta5_coeff", "zeta_0",
    )

    # Source: IMRPhenomX_Return_zeta_MSA
    # (LALSimIMRPhenomX_precession.c).
    @classmethod
    def _msa_zeta(cls, v, prec):
        """
        Precession angle zeta at velocity v.

        NaN entries are set to 0, with detached inputs there.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor
        """
        zeta_out = cls._msa_zeta_terms(v, prec)
        nan = torch.isnan(zeta_out)
        if bool(nan.any()):
            zeta_out = cls._msa_zeta_terms(
                detach_where(nan, v),
                {key: detach_where(nan, prec[key]) for key in cls.MSA_ZETA_KEYS},
            )
        return torch.where(nan, torch.zeros_like(zeta_out), zeta_out)

    # Source: IMRPhenomX_Return_zeta_MSA
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_zeta_terms(v, prec):
        """
        Precession angle zeta, before NaN handling.

        Note factor of log(v) as per LALSimInspiralFDPrecAngles.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor
        """
        invv = 1.0 / v
        invv2 = invv * invv
        invv3 = invv * invv2
        v2 = v * v
        logv = torch.log(v)
        zeta_out = prec["eta"] * (prec["Omegazeta0_coeff"] * invv3
                                  + prec["Omegazeta1_coeff"] * invv2
                                  + prec["Omegazeta2_coeff"] * invv
                                  + prec["Omegazeta3_coeff"] * logv
                                  + prec["Omegazeta4_coeff"] * v
                                  + prec["Omegazeta5_coeff"] * v2) \
            + prec["zeta_0"]
        return zeta_out

    # Source: IMRPhenomX_Return_Constants_c_MSA
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_constants_c(v, j_norm, prec):
        """
        Constants c0, c2 and c4, for versions other than 220.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        j_norm : torch.Tensor
            Norm of J.
        prec : dict
            Precession variables.

        Returns
        -------
        c0, c2, c4 : torch.Tensor
        """
        v2 = v * v
        v3 = v * v2
        v4 = v2 * v2
        v6 = v3 * v3
        JNorm2 = j_norm * j_norm
        Seff = prec["Seff"]
        eta = prec["eta"]
        eta3 = prec["eta3"]
        c0 = j_norm * (0.75 * (1.0 - Seff * v) * v2 * (eta3 + 4.0 * eta3 * Seff * v
             - 2.0 * eta * (JNorm2 - prec["Spl2"]
                            + 2.0 * (prec["S1_norm_2"] - prec["S2_norm_2"]) * prec["delta_qq"]) * v2
             - 4.0 * eta * Seff * (JNorm2 - prec["Spl2"]) * v3
             + (JNorm2 - prec["Spl2"]) * (JNorm2 - prec["Spl2"]) * v4 * prec["inveta"]))
        c2 = j_norm * (-1.5 * eta * (prec["Spl2"] - prec["Smi2"])
                       * (1.0 + 2.0 * Seff * v
                          - (JNorm2 - prec["Spl2"]) * v2 * prec["inveta2"])
                       * (1.0 - Seff * v) * v4)
        c4 = j_norm * (0.75 * prec["inveta"] * (prec["Spl2"] - prec["Smi2"])
                       * (prec["Spl2"] - prec["Smi2"]) * (1.0 - Seff * v) * v6)
        return c0, c2, c4

    # Source: IMRPhenomX_Return_Constants_d_MSA
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_constants_d(l_norm, j_norm, prec):
        """
        Constants d0, d2 and d4.

        Parameters
        ----------
        l_norm, j_norm : torch.Tensor
            Norms of L and J.
        prec : dict
            Precession variables.

        Returns
        -------
        d0, d2, d4 : torch.Tensor
        """
        LNorm2 = l_norm * l_norm
        JNorm2 = j_norm * j_norm
        d0 = -(JNorm2 - (l_norm + prec["Spl"]) * (l_norm + prec["Spl"])) \
            * (JNorm2 - (l_norm - prec["Spl"]) * (l_norm - prec["Spl"]))
        d2 = -2.0 * (prec["Spl2"] - prec["Smi2"]) * (JNorm2 + LNorm2 - prec["Spl2"])
        d4 = -(prec["Spl2"] - prec["Smi2"]) * (prec["Spl2"] - prec["Smi2"])
        return d0, d2, d4

    # Source: IMRPhenomX_Return_Psi_MSA (LALSimIMRPhenomX_precession.c).
    @classmethod
    def _msa_psi(cls, v, v2, prec):
        """
        MSA phase psi without the psi0 term.

        Parameters
        ----------
        v, v2 : torch.Tensor
            Orbital velocity and its square.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor, same shape as v
        """
        return -0.75 * prec["g0"] * prec["delta_qq"] \
            * (1.0 + prec["psi1"] * v + prec["psi2"] * v2) / (v2 * v)

    # Source: IMRPhenomX_Return_Psi_dot_MSA
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_psi_dot(v, prec):
        """
        Time derivative of the MSA phase psi.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor, same shape as v
        """
        v2 = v * v
        A_coeff = -1.5 * (v2 * v2 * v2) * (1.0 - v * prec["Seff"]) * prec["sqrt_inveta"]
        return 0.5 * A_coeff * torch.sqrt(prec["Spl2"] - prec["S32"])

    # Precession variables read by _msa_correction_terms and its helpers
    MSA_CORRECTION_KEYS = (
        "Seff", "eta", "eta3", "inveta", "inveta2", "sqrt_inveta", "delta_qq",
        "S1_norm_2", "S2_norm_2", "S32", "Smi2", "Spl", "Spl2", "Spl2mSmi2",
        "g0", "psi0", "psi1", "psi2",
    )

    # Source: IMRPhenomX_Return_MSA_Corrections_MSA
    # (LALSimIMRPhenomX_precession.c).
    @classmethod
    def _msa_corrections(
        cls,
        v: torch.Tensor,
        l_norm: torch.Tensor,
        j_norm: torch.Tensor,
        prec: dict,
        skip: torch.Tensor,
    ) -> tuple:
        """
        MSA corrections to phiz and zeta.

        Zero where ``skip`` or NaN, with detached inputs on branches
        that are not evaluated.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        l_norm, j_norm : torch.Tensor
            Norms of L and J.
        prec : dict
            Precession variables.
        skip : torch.Tensor of bool
            Entries where the corrections are not evaluated.

        Returns
        -------
        corr_x, corr_y : torch.Tensor
            Corrections to phiz and zeta.
        """
        with torch.no_grad():
            corr_x, corr_y = cls._msa_correction_terms(v, l_norm, j_norm, prec)
        skip_x = skip | torch.isnan(corr_x)
        skip_y = skip | torch.isnan(corr_y)
        gated = {
            key: detach_where(skip_x, prec[key]) for key in cls.MSA_CORRECTION_KEYS
        }
        corr_x, corr_y = cls._msa_correction_terms(
            detach_where(skip_x, v),
            detach_where(skip_x, l_norm),
            detach_where(skip_x, j_norm),
            gated,
            skip_y,
        )
        corr_x = torch.where(skip_x, torch.zeros_like(corr_x), corr_x)
        corr_y = torch.where(skip_y, torch.zeros_like(corr_y), corr_y)
        return corr_x, corr_y

    # Source: IMRPhenomX_Return_MSA_Corrections_MSA
    # (LALSimIMRPhenomX_precession.c).
    @classmethod
    def _msa_correction_terms(
        cls,
        v: torch.Tensor,
        l_norm: torch.Tensor,
        j_norm: torch.Tensor,
        prec: dict,
        skip_y: torch.Tensor = None,
    ) -> tuple:
        """
        MSA corrections to phiz and zeta, before NaN handling.

        A term whose n is 1, and zeta where ``skip_y``, gets detached
        inputs.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        l_norm, j_norm : torch.Tensor
            Norms of L and J.
        prec : dict
            Precession variables.
        skip_y : torch.Tensor of bool or None
            Entries where the zeta correction is not used.

        Returns
        -------
        corr_x, corr_y : torch.Tensor
            Corrections to phiz and zeta.
        """
        v2 = v * v
        c0, c2, c4 = cls._msa_constants_c(v, j_norm, prec)
        d0, d2, d4 = cls._msa_constants_d(l_norm, j_norm, prec)
        two_d0 = 2.0 * d0
        sd = torch.sqrt(torch.abs(d2 * d2 - 4.0 * d0 * d4))
        A_theta_L = 0.5 * ((j_norm / l_norm) + (l_norm / j_norm)
                           - (prec["Spl2"] / (j_norm * l_norm)))
        B_theta_L = 0.5 * prec["Spl2mSmi2"] / (j_norm * l_norm)
        nc_num = 2.0 * (d0 + d2 + d4)
        nc_denom = two_d0 + d2 + sd
        nc = nc_num / nc_denom
        nd = nc_denom / two_d0
        sqrt_nc = torch.sqrt(torch.abs(nc))
        sqrt_nd = torch.sqrt(torch.abs(nd))
        psi = cls._msa_psi(v, v2, prec) + prec["psi0"]
        psi_dot = cls._msa_psi_dot(v, prec)
        tan_psi = torch.tan(psi)
        atan_psi = torch.atan(tan_psi)
        C2den = 2.0 * d0 * sd * (d0 + d2 + d4)

        # A term is 0 where its n is 1, with detached inputs there
        shared = (c0, c2, c4, d0, d2, d4, sd, C2den, tan_psi, atan_psi, psi_dot)
        c_skip = nc == 1.0
        d_skip = nd == 1.0
        Cphi_term = cls._msa_cphi_term(
            *(detach_where(c_skip, t) for t in (nc, sqrt_nc) + shared)
        )
        Dphi_term = cls._msa_dphi_term(
            *(detach_where(d_skip, t) for t in (nd, sqrt_nd) + shared)
        )
        Cphi_term = torch.where(c_skip, torch.zeros_like(Cphi_term), Cphi_term)
        Dphi_term = torch.where(d_skip, torch.zeros_like(Dphi_term), Dphi_term)
        corr_x = Cphi_term + Dphi_term

        zeta_inputs = (A_theta_L, B_theta_L, d0, d2, sd, corr_x, Cphi_term, Dphi_term)
        if skip_y is not None:
            zeta_inputs = tuple(detach_where(skip_y, t) for t in zeta_inputs)
        A_y, B_y, d0_y, d2_y, sd_y, corr_x_y, Cphi_y, Dphi_y = zeta_inputs
        corr_y = A_y * corr_x_y + 2. * B_y * d0_y \
            * (Cphi_y / (sd_y - d2_y) - Dphi_y / (sd_y + d2_y))
        return corr_x, corr_y

    # Source: IMRPhenomX_Return_MSA_Corrections_MSA
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_cphi_term(
        nc: torch.Tensor, sqrt_nc: torch.Tensor, c0: torch.Tensor,
        c2: torch.Tensor, c4: torch.Tensor, d0: torch.Tensor,
        d2: torch.Tensor, d4: torch.Tensor, sd: torch.Tensor,
        C2den: torch.Tensor, tan_psi: torch.Tensor,
        atan_psi: torch.Tensor, psi_dot: torch.Tensor,
    ) -> torch.Tensor:
        """
        C_phi term, as implemented in LALSimInspiralFDPrecAngles.c.

        Parameters
        ----------
        nc, sqrt_nc : torch.Tensor
            nc and sqrt(|nc|).
        c0, c2, c4, d0, d2, d4, sd, C2den : torch.Tensor
            MSA constants and their combinations.
        tan_psi, atan_psi, psi_dot : torch.Tensor
            tan(psi), atan(tan(psi)) and psi_dot.

        Returns
        -------
        torch.Tensor
        """
        return torch.abs((c4 * d0 * ((2 * d0 + d2) + sd)
                          - c2 * d0 * ((d2 + 2. * d4) - sd)
                          - c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - sd))) / C2den) \
            * (sqrt_nc / (nc - 1.)
               * (atan_psi - torch.atan(sqrt_nc * tan_psi))) / psi_dot

    # Source: IMRPhenomX_Return_MSA_Corrections_MSA
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_dphi_term(
        nd: torch.Tensor, sqrt_nd: torch.Tensor, c0: torch.Tensor,
        c2: torch.Tensor, c4: torch.Tensor, d0: torch.Tensor,
        d2: torch.Tensor, d4: torch.Tensor, sd: torch.Tensor,
        C2den: torch.Tensor, tan_psi: torch.Tensor,
        atan_psi: torch.Tensor, psi_dot: torch.Tensor,
    ) -> torch.Tensor:
        """
        D_phi term, as implemented in LALSimInspiralFDPrecAngles.c.

        Parameters
        ----------
        nd, sqrt_nd : torch.Tensor
            nd and sqrt(|nd|).
        c0, c2, c4, d0, d2, d4, sd, C2den : torch.Tensor
            MSA constants and their combinations.
        tan_psi, atan_psi, psi_dot : torch.Tensor
            tan(psi), atan(tan(psi)) and psi_dot.

        Returns
        -------
        torch.Tensor
        """
        return torch.abs(-c4 * d0 * ((2 * d0 + d2) - sd)
                         + c2 * d0 * ((d2 + 2. * d4) + sd)
                         - c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd))) / C2den \
            * (sqrt_nd / (nd - 1.)
               * (atan_psi - torch.atan(sqrt_nd * tan_psi))) / psi_dot

    # Source: IMRPhenomX_costhetaLJ (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _msa_costhetaLJ(l_norm, j_norm, s_norm):
        """
        Cosine of the angle between L and J, clamped to [-1, 1].

        cos(thetaLJ) = 0.5 (J^2 + L^2 - S^2) / (L J).

        Parameters
        ----------
        l_norm, j_norm, s_norm : torch.Tensor
            Norms of L, J and S.

        Returns
        -------
        torch.Tensor
        """
        c = 0.5 * (j_norm * j_norm + l_norm * l_norm - s_norm * s_norm) \
            / (l_norm * j_norm)
        return torch.clamp(c, -1.0, 1.0)

    # Source: IMRPhenomX_Return_phi_zeta_costhetaL_MSA
    # (LALSimIMRPhenomX_precession.c).
    def msa_phi_zeta_costhetaL(self, v, derived, prec):
        """
        Generate phiz, zeta and cos(thetaL) at a given velocity.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        derived : torch.Tensor
            Aligned-spin derived parameters.
        prec : dict
            Precession variables.

        Returns
        -------
        phiz, zeta, cos_theta_L : torch.Tensor
        """
        eta = prec["eta"]
        L_norm = eta / v
        J_norm = self._msa_j_norm(L_norm, prec)
        L_norm3PN = self._l_norm_3pn_of_v(v, v * v, L_norm, prec)
        J_norm3PN = self._msa_j_norm(L_norm3PN, prec)
        roots = self._msa_roots(L_norm, J_norm, prec)
        S32, Smi2, Spl2 = roots
        # Roots at v, used by the corrections and the norm of S
        loc = dict(prec)
        loc["S32"] = S32
        loc["Smi2"] = Smi2
        loc["Spl2"] = Spl2
        # S_+, used only by the corrections
        loc["Spl"] = strong_zero_sqrt(Spl2)
        loc["Spl2mSmi2"] = Spl2 - Smi2
        SNorm = self._msa_s_norm(v, roots, loc)
        # Corrections only when |Smi2 - Spl2| > 1e-5
        corr_skip = ~(torch.abs(Smi2 - Spl2) > 1.0e-5)
        corr_x, corr_y = self._msa_corrections(v, L_norm, J_norm, loc, corr_skip)
        phiz = self._msa_phiz(v, J_norm, loc)
        zeta = self._msa_zeta(v, loc)
        cos_theta_L = self._msa_costhetaLJ(L_norm3PN, J_norm3PN, SNorm)
        return phiz + corr_x, zeta + corr_y, cos_theta_L

    # ------------------------------------------------------------------
    # NNLO angles
    # ------------------------------------------------------------------

    # Source: IMRPhenomXGetAndSetPrecessionVariables
    # (LALSimIMRPhenomX_precession.c).
    def _nnlo_angle_coeffs(self, derived, prec):
        """
        Single spin PN Euler angle coefficients, as per IMRPhenomPv2.

        Sets alpha1 to alpha5, alpha4L, epsilon1 to epsilon5 and
        epsilon4L in ``prec``.

        Parameters
        ----------
        derived : torch.Tensor
            Aligned-spin derived parameters.
        prec : dict
            Precession variables; updated in place.
        """
        eta = prec["eta"]
        eta2 = prec["eta2"]
        eta3 = prec["eta3"]
        eta4 = prec["eta4"]
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        m1 = prec["m1"]
        m1_2 = m1 * m1
        m1_3 = m1_2 * m1
        m1_4 = m1_3 * m1
        m1_5 = m1_4 * m1
        m1_6 = m1_5 * m1
        m1_7 = m1 * m1_6
        m1_8 = m1 * m1_7
        delta = derived[:, 4:5]
        delta2 = delta * delta
        delta3 = delta2 * delta
        # chi_eff = pWF->chiEff
        chi_eff = derived[:, 7:8]
        chip = prec["chi_p"]
        chip2 = chip * chip
        q = prec["m1"] / prec["m2"]   # q = m1 / m2 > 1.0
        PI = math.pi

        chiL = (1.0 + q) * (chi_eff / q)
        chiL2 = chiL * chiL

        prec["alpha1"] = -35 / 192. + (5 * delta) / (64. * m1)
        prec["alpha2"] = ((15 * chiL * delta * m1) / 128.
                          - (35 * chiL * m1_2) / 128.) / eta
        prec["alpha3"] = (-5515 / 3072. + eta * (-515 / 384. - (15 * delta2) / (256. * m1_2)
                          + (175 * delta) / (256. * m1)) + (4555 * delta) / (7168. * m1)
                          + ((15 * chip2 * delta * m1_3) / 128.
                             - (35 * chip2 * m1_4) / 128.) / eta2)
        prec["alpha4L"] = ((5 * chiL * delta2) / 16. - (5 * chiL * delta * m1) / 3.
                           + (2545 * chiL * m1_2) / 1152.
                           + ((-2035 * chiL * delta * m1) / 21504.
                              + (2995 * chiL * m1_2) / 9216.) / eta
                           + ((5 * chiL * chip2 * delta * m1_5) / 128.
                              - (35 * chiL * chip2 * m1_6) / 384.) / eta3
                           - (35 * PI) / 48. + (5 * delta * PI) / (16. * m1))
        prec["alpha5"] = (5 * (-190512 * delta3 * eta6
                          + 2268 * delta2 * eta3 * m1 * (eta2 * (323 + 784 * eta)
                          + 336 * (25 * chiL2 + chip2) * m1_4)
                          + 7 * m1_3 * (8024297 * eta4 + 857412 * eta5
                          + 3080448 * eta6 + 143640 * chip2 * eta2 * m1_4
                          - 127008 * chip2 * (-4 * chiL2 + chip2) * m1_8
                          + 6048 * eta3 * ((2632 * chiL2 + 115 * chip2) * m1_4
                          - 672 * chiL * m1_2 * PI))
                          + 3 * delta * m1_2 * (-5579177 * eta4 + 80136 * eta5
                          - 3845520 * eta6 + 146664 * chip2 * eta2 * m1_4
                          + 127008 * chip2 * (-4 * chiL2 + chip2) * m1_8
                          - 42336 * eta3 * ((726 * chiL2 + 29 * chip2) * m1_4
                          - 96 * chiL * m1_2 * PI)))) / (6.5028096e7 * eta4 * m1_3)

        prec["epsilon1"] = -35 / 192. + (5 * delta) / (64. * m1)
        prec["epsilon2"] = ((15 * chiL * delta * m1) / 128.
                            - (35 * chiL * m1_2) / 128.) / eta
        prec["epsilon3"] = (-5515 / 3072. + eta * (-515 / 384.
                            - (15 * delta2) / (256. * m1_2)
                            + (175 * delta) / (256. * m1))
                            + (4555 * delta) / (7168. * m1))
        prec["epsilon4L"] = ((5 * chiL * delta2) / 16. - (5 * chiL * delta * m1) / 3.
                             + (2545 * chiL * m1_2) / 1152.
                             + ((-2035 * chiL * delta * m1) / 21504.
                                + (2995 * chiL * m1_2) / 9216.) / eta
                             - (35 * PI) / 48. + (5 * delta * PI) / (16. * m1))
        prec["epsilon5"] = (5 * (-190512 * delta3 * eta3
                            + 2268 * delta2 * m1 * (eta2 * (323 + 784 * eta)
                            + 8400 * chiL2 * m1_4)
                            - 3 * delta * m1_2 * (eta * (5579177
                            + 504 * eta * (-159 + 7630 * eta))
                            + 254016 * chiL * m1_2 * (121 * chiL * m1_2 - 16 * PI))
                            + 7 * m1_3 * (eta * (8024297
                            + 36 * eta * (23817 + 85568 * eta))
                            + 338688 * chiL * m1_2 * (47 * chiL * m1_2 - 12 * PI)))) \
            / (6.5028096e7 * eta * m1_3)

    # Source: IMRPhenomX_PN_Euler_alpha_NNLO
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _pn_euler_alpha_nnlo(prec, omega, omega_cbrt, omega_cbrt2, logomega):
        """
        Calculate alpha using pre-cached NNLO PN expressions.

        Note that alpha_offset already includes the term alpha0.

        Parameters
        ----------
        prec : dict
            Precession variables with the NNLO coefficients.
        omega, omega_cbrt, omega_cbrt2, logomega : torch.Tensor
            omega, omega^(1/3), omega^(2/3) and log(omega).

        Returns
        -------
        torch.Tensor
        """
        return (prec["alpha1"] / omega
                + prec["alpha2"] / omega_cbrt2
                + prec["alpha3"] / omega_cbrt
                + prec["alpha4L"] * logomega
                + prec["alpha5"] * omega_cbrt
                - prec["alpha_offset"])

    # Source: IMRPhenomX_PN_Euler_epsilon_NNLO
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _pn_euler_epsilon_nnlo(prec, omega, omega_cbrt, omega_cbrt2, logomega):
        """
        Calculate epsilon using pre-cached NNLO PN expressions.

        Note that epsilon_offset already includes the term epsilon0.

        Parameters
        ----------
        prec : dict
            Precession variables with the NNLO coefficients.
        omega, omega_cbrt, omega_cbrt2, logomega : torch.Tensor
            omega, omega^(1/3), omega^(2/3) and log(omega).

        Returns
        -------
        torch.Tensor
        """
        return (prec["epsilon1"] / omega
                + prec["epsilon2"] / omega_cbrt2
                + prec["epsilon3"] / omega_cbrt
                + prec["epsilon4L"] * logomega
                + prec["epsilon5"] * omega_cbrt
                - prec["epsilon_offset"])

    # ------------------------------------------------------------------
    # Wigner-d half-angle coefficients
    # ------------------------------------------------------------------

    # Source: IMRPhenomXWignerdCoefficients_cosbeta
    # (LALSimIMRPhenomX_precession.c).
    @staticmethod
    def _wignerd_coefficients_cosbeta(cos_beta):
        """
        cos(beta/2) and sin(beta/2) from cos(beta).

        The results are always non-negative; safe_sqrt keeps the
        gradient finite at |cos(beta)| = 1.

        Parameters
        ----------
        cos_beta : torch.Tensor
            cos(beta).

        Returns
        -------
        cBetah, sBetah : torch.Tensor
        """
        cBetah = safe_sqrt(torch.abs(1.0 + cos_beta) / 2.0)
        sBetah = safe_sqrt(torch.abs(1.0 - cos_beta) / 2.0)
        return cBetah, sBetah

    # Source: IMRPhenomXWignerdCoefficients
    # (LALSimIMRPhenomX_precession.c).
    def _nnlo_cosbeta(self, v, derived, prec):
        """
        NNLO cos(beta) from the orbital angular momentum.

        s = Sperp / (L + SL) and
        cos(beta) = copysign(1, L + SL) / sqrt(1 + s^2).

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        derived : torch.Tensor
            Aligned-spin derived parameters.
        prec : dict
            Precession variables.

        Returns
        -------
        torch.Tensor, same shape as v
        """
        L = self._lpn_ansatz(v, prec["eta"] / v, prec["L_coeffs"])
        s = prec["Sperp"] / (L + prec["SL"])
        s2 = s * s
        sgn = torch.copysign(torch.ones_like(s), L + prec["SL"])
        return sgn / torch.sqrt(1.0 + s2)

    # Source: IMRPhenomXWignerdCoefficients
    # (LALSimIMRPhenomX_precession.c).
    def _wignerd_coefficients(self, v, derived, prec):
        """
        NNLO cos(beta/2) and sin(beta/2) at velocity v.

        Parameters
        ----------
        v : torch.Tensor
            Orbital velocity.
        derived : torch.Tensor
            Aligned-spin derived parameters.
        prec : dict
            Precession variables.

        Returns
        -------
        cBetah, sBetah : torch.Tensor
        """
        return self._wignerd_coefficients_cosbeta(self._nnlo_cosbeta(v, derived, prec))

    # ------------------------------------------------------------------
    # Euler angles over the frequency grid
    # ------------------------------------------------------------------

    # Source: IMRPhenomXPHMTwistUp and Get_alpha_beta_epsilon
    # (LALSimIMRPhenomXPHM.c).
    def get_alpha_epsilon_cosbeta(self, Mf, mprime, derived, prec):
        """
        Euler angles alpha, epsilon and beta over the frequency grid.

        MSA angles, with the NNLO angles on rows where the MSA system
        failed; the offsets are the mprime = 2 values for every mprime.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F)
            Geometric frequencies.
        mprime : int
            Second index of the non-precessing mode (l, mprime).
        derived : torch.Tensor
            Aligned-spin derived parameters.
        prec : dict
            Precession variables.

        Returns
        -------
        alpha, epsilon, cBetah, sBetah : torch.Tensor, shape (B, F)
        """
        omega = math.pi * Mf * (2.0 / mprime)
        version = prec["version"]

        # NNLO omega in the C order LAL_PI * Mf * 2./mprime
        omega_nnlo = math.pi * Mf * 2. / mprime

        def nnlo_angles(omega, prec):
            # The series subtract alpha_offset and epsilon_offset
            logomega = torch.log(omega)
            omega_cbrt = cbrt(omega)
            omega_cbrt2 = omega_cbrt * omega_cbrt
            alpha = self._pn_euler_alpha_nnlo(prec, omega, omega_cbrt,
                                              omega_cbrt2, logomega)
            epsilon = self._pn_euler_epsilon_nnlo(prec, omega, omega_cbrt,
                                                  omega_cbrt2, logomega)
            cosbeta = self._nnlo_cosbeta(omega_cbrt, derived, prec)
            return alpha, epsilon, cosbeta

        if version in (220, 221, 222, 223, 224):
            msa_error = prec["MSA_ERROR"]
            fallback = bool(msa_error.any())
            # NNLO angles on MSA_ERROR rows; each model is detached on
            # the rows it does not serve
            prec_msa = _detach_rows(msa_error, prec) if fallback else prec
            omega_msa = detach_where(msa_error, omega) if fallback else omega
            v = cbrt(omega_msa)
            phiz, zeta, ctl = self.msa_phi_zeta_costhetaL(v, derived, prec_msa)
            alpha = phiz - prec_msa["alpha_offset"]
            epsilon = zeta - prec_msa["epsilon_offset"]
            cosbeta = ctl
            if fallback:
                a_n, e_n, c_n = nnlo_angles(
                    detach_where(~msa_error, omega_nnlo), _detach_rows(~msa_error, prec)
                )
                alpha = torch.where(msa_error, a_n, alpha)
                epsilon = torch.where(msa_error, e_n, epsilon)
                cosbeta = torch.where(msa_error, c_n, cosbeta)
        else:
            alpha, epsilon, cosbeta = nnlo_angles(omega_nnlo, prec)

        cBetah, sBetah = self._wignerd_coefficients_cosbeta(cosbeta)
        return alpha, epsilon, cBetah, sBetah

    # ------------------------------------------------------------------
    # Wigner-d rows
    # ------------------------------------------------------------------

    # Source: IMRPhenomXPHMTwistUp and IMRPhenomXPHMTwistUpOneMode
    # (LALSimIMRPhenomXPHM.c).
    @staticmethod
    def _wigner_d_rows(cBetah, sBetah, ell, mprime):
        """
        Wigner-d rows d^l_{m,m'} and d^l_{m,-m'} for m = -l..l.

        The d^l_{m,-m'} row follows from d^l_{m,m'} by symmetry.

        Parameters
        ----------
        cBetah, sBetah : torch.Tensor, shape (B, F)
            cos(beta/2) and sin(beta/2).
        ell, mprime : int
            Mode indices (l, m').

        Returns
        -------
        d, dm : list of torch.Tensor, shape (B, F)

        Raises
        ------
        ValueError
            For an unsupported (l, m').
        """
        cBetah2 = cBetah * cBetah
        cBetah3 = cBetah * cBetah2
        cBetah4 = cBetah * cBetah3
        sBetah2 = sBetah * sBetah
        sBetah3 = sBetah * sBetah2
        sBetah4 = sBetah * sBetah3
        # LAL's pPrec->sqrt* literals, kept as written
        sqrt2 = 1.4142135623730951
        sqrt5 = 2.23606797749978981
        sqrt6 = 2.44948974278317788
        sqrt7 = 2.64575131106459072
        sqrt10 = 3.16227766016838
        sqrt14 = 3.74165738677394133
        sqrt15 = 3.87298334620741702
        sqrt70 = 8.36660026534075563
        sqrt30 = 5.477225575051661

        if (ell, mprime) == (2, 2):
            d = [sBetah4, 2.0 * cBetah * sBetah3, sqrt6 * sBetah2 * cBetah2,
                 2.0 * cBetah3 * sBetah, cBetah4]
            dm = [d[4], -d[3], d[2], -d[1], d[0]]
        elif (ell, mprime) == (2, 1):
            d = [2.0 * cBetah * sBetah3, 3.0 * cBetah2 * sBetah2 - sBetah4,
                 sqrt6 * (cBetah3 * sBetah - cBetah * sBetah3),
                 cBetah2 * (cBetah2 - 3.0 * sBetah2), -2.0 * cBetah3 * sBetah]
            dm = [-d[4], d[3], -d[2], d[1], -d[0]]
        elif (ell, mprime) == (3, 3):
            cBetah5 = cBetah * cBetah4
            cBetah6 = cBetah * cBetah5
            sBetah5 = sBetah * sBetah4
            sBetah6 = sBetah * sBetah5
            d = [sBetah6, sqrt6 * cBetah * sBetah5, sqrt15 * cBetah2 * sBetah4,
                 2.0 * sqrt5 * cBetah3 * sBetah3, sqrt15 * cBetah4 * sBetah2,
                 sqrt6 * cBetah5 * sBetah, cBetah6]
            dm = [d[6], -d[5], d[4], -d[3], d[2], -d[1], d[0]]
        elif (ell, mprime) == (3, 2):
            cBetah5 = cBetah * cBetah4
            sBetah5 = sBetah * sBetah4
            d = [sqrt6 * cBetah * sBetah5, sBetah4 * (5.0 * cBetah2 - sBetah2),
                 sqrt10 * sBetah3 * (2.0 * cBetah3 - cBetah * sBetah2),
                 sqrt30 * cBetah2 * (cBetah2 - sBetah2) * sBetah2,
                 sqrt10 * cBetah3 * (cBetah2 * sBetah - 2.0 * sBetah3),
                 cBetah4 * (cBetah2 - 5.0 * sBetah2),
                 -1.0 * sqrt6 * cBetah5 * sBetah]
            dm = [-d[6], d[5], -d[4], d[3], -d[2], d[1], -d[0]]
        elif (ell, mprime) == (4, 4):
            cBetah5 = cBetah * cBetah4
            cBetah6 = cBetah * cBetah5
            cBetah7 = cBetah * cBetah6
            cBetah8 = cBetah * cBetah7
            sBetah5 = sBetah * sBetah4
            sBetah6 = sBetah * sBetah5
            sBetah7 = sBetah * sBetah6
            sBetah8 = sBetah * sBetah7
            d = [sBetah8, 2.0 * sqrt2 * cBetah * sBetah7,
                 2.0 * sqrt7 * cBetah2 * sBetah6,
                 2.0 * sqrt14 * cBetah3 * sBetah5, sqrt70 * cBetah4 * sBetah4,
                 2.0 * sqrt14 * cBetah5 * sBetah3,
                 2.0 * sqrt7 * cBetah6 * sBetah2,
                 2.0 * sqrt2 * cBetah7 * sBetah, cBetah8]
            dm = [d[8], -d[7], d[6], -d[5], d[4], -d[3], d[2], -d[1], d[0]]
        else:
            raise ValueError(f"unsupported twist mode ({ell}, {mprime})")
        return d, dm

    # ------------------------------------------------------------------
    # Twist-up
    # ------------------------------------------------------------------

    # Source: IMRPhenomXPHMTwistUp (LALSimIMRPhenomXPHM.c).
    def twist_up(self, Mf, hlm_coprec, derived, prec, ell, mprime):
        """
        Twist one co-precessing mode up to the inertial J-frame.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F)
            Geometric frequencies.
        hlm_coprec : torch.Tensor, shape (B, F)
            Co-precessing mode h_{l,-m'}.
        derived : torch.Tensor
            Aligned-spin derived parameters.
        prec : dict
            Precession variables.
        ell, mprime : int
            Mode indices (l, m'), m' > 0.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, F)
            Contributions of the mode to h_+ and h_x.
        """
        alpha, epsilon, cBetah, sBetah = self.get_alpha_epsilon_cosbeta(
            Mf, mprime, derived, prec)
        d, dm = self._wigner_d_rows(cBetah, sBetah, ell, mprime)

        cexp_i_alpha = torch.polar(torch.ones_like(alpha), alpha)
        Y = [prec[f"Y{ell}{'m' if m < 0 else ''}{abs(m)}"]
             for m in range(-ell, ell + 1)]

        # Precompute powers of e^{i m alpha}
        pows = [None] * (2 * ell + 1)
        pows[ell] = torch.ones_like(cexp_i_alpha)
        pows[ell + 1] = cexp_i_alpha
        for k in range(2, ell + 1):
            pows[ell + k] = cexp_i_alpha * pows[ell + k - 1]
        cexp_mi_alpha = _cinv(cexp_i_alpha)
        pows[ell - 1] = cexp_mi_alpha
        for k in range(2, ell + 1):
            pows[ell - k] = cexp_mi_alpha * pows[ell - k + 1]

        # l = 3 flips the sign of the conjugated transfer function
        conj_sign = -1.0 if ell == 3 else 1.0

        hp_sum = torch.zeros_like(cexp_i_alpha)
        hc_sum = torch.zeros_like(cexp_i_alpha)
        for m in range(-ell, ell + 1):
            A = pows[ell - m] * dm[m + ell] * Y[m + ell]
            Astar = pows[ell + m] * d[m + ell] * torch.conj(Y[m + ell])
            hp_sum = hp_sum + (A + conj_sign * Astar)
            hc_sum = hc_sum + 1j * (A - conj_sign * Astar)

        eps_phase = torch.polar(torch.ones_like(epsilon),
                                -mprime * epsilon) * hlm_coprec / 2.0
        return eps_phase * hp_sum, eps_phase * hc_sum

    # ------------------------------------------------------------------
    # J-frame single modes
    # ------------------------------------------------------------------

    # Source: IMRPhenomXPHMTwistUpOneMode (LALSimIMRPhenomXPHM.c).
    def _twist_up_one_mode(self, Mf, hlm_coprec, derived, prec, ell, mprime, m):
        """
        Twist one co-precessing mode into one inertial J-frame mode.

        Parameters
        ----------
        Mf : torch.Tensor, shape (B, F)
            Geometric frequencies.
        hlm_coprec : torch.Tensor, shape (B, F)
            Co-precessing mode h_{l,-m'}.
        derived : torch.Tensor
            Aligned-spin derived parameters.
        prec : dict
            Precession variables.
        ell, mprime, m : int
            Co-precessing mode (l, m') and inertial mode index m.

        Returns
        -------
        hlm_pos, hlm_neg : torch.Tensor, shape (B, F)
            Positive and negative frequency series of mode (l, m).
        """
        alpha, epsilon, cBetah, sBetah = self.get_alpha_epsilon_cosbeta(
            Mf, mprime, derived, prec)
        d, dm = self._wigner_d_rows(cBetah, sBetah, ell, mprime)
        minus1l = -1.0 if ell % 2 != 0 else 1.0

        cexp_im_alpha = torch.polar(torch.ones_like(alpha), -m * alpha)
        exp_imprime_epsilon = torch.polar(torch.ones_like(epsilon),
                                          mprime * epsilon)

        hlm_pos = (1.0 / exp_imprime_epsilon) * hlm_coprec \
            * cexp_im_alpha * dm[m + ell]
        hlm_neg = exp_imprime_epsilon * minus1l * torch.conj(hlm_coprec) \
            * cexp_im_alpha * d[m + ell]
        return hlm_pos, hlm_neg

    # Co-precessing modes per l, positive mprime in LAL's loop order
    COPREC_MODES = {2: (1, 2), 3: (2, 3), 4: (4,)}

    # Source: XLALSimIMRPhenomXPHMOneMode and IMRPhenomXPHM_OneMode
    # (LALSimIMRPhenomXPHM.c).
    def get_hlm_jframe(self, theta, ell, emm):
        """
        Individual precessing mode hlm in the inertial J-frame.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 12)
            Precessing layout.
        ell, emm : int
            Mode indices (l, m).

        Returns
        -------
        hlm_pos, hlm_neg : torch.Tensor, shape (B, F), complex128
            Positive and negative frequency series.
        """
        theta = self.check_masses_and_spins(theta)
        theta_aligned = torch.cat(
            [theta[:, 0:2], theta[:, 4:5], theta[:, 7:8], theta[:, 8:9],
             torch.zeros_like(theta[:, 0:3])], dim=1)
        derived = self.compute_derived_parameters(theta_aligned)
        prec = self.get_and_set_precession_variables(theta, derived)
        afinal = prec["afinal"]
        amp22_coeffs = self.get_amp_coeffs(derived, afinal=afinal)
        phase22_coeffs = self.get_phase_coeffs(derived, afinal=afinal)

        M_s = derived[:, 2:3]
        Mf = self.f * M_s
        hlm_pos = torch.zeros_like(Mf, dtype=torch.complex128)
        hlm_neg = torch.zeros_like(Mf, dtype=torch.complex128)
        for mprime in self.COPREC_MODES[ell]:
            htildelm = self.gethlm(theta_aligned, ell, mprime,
                                   derived=derived, amp22_coeffs=amp22_coeffs,
                                   phase22_coeffs=phase22_coeffs,
                                   afinal=afinal)
            pos, neg = self._twist_up_one_mode(Mf, htildelm, derived, prec,
                                               ell, mprime, emm)
            hlm_pos = hlm_pos + pos
            hlm_neg = hlm_neg + neg
        return hlm_pos, hlm_neg

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    # Source: XLALSimIMRPhenomXPHM and IMRPhenomXPHM_hplushcross
    # (LALSimIMRPhenomXPHM.c).
    def get_hphc(self, theta, reproduce_lal=False):
        """
        FD plus and cross polarisations of a precessing BBH batch.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 17)
            Parameters in the waveform_utils.WAVEFORM_PARAMETERS layout.
        reproduce_lal : bool
            If True, skip FD tapering, delta_tc shift and df
            normalisation, to compare with LAL called with
            PhenomXHMThresholdMband = PhenomXPHMThresholdMband = 0.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_pad + F), complex
        """
        # [m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, distance,
        #  delta_tc, phic, inclination]
        theta = waveform_utils.precessing_layout(theta, self.extras)
        swapped = self.check_masses_and_spins(theta)
        # pWF keeps equal-mass bodies as given, pPrec swaps them
        body = torch.where(theta[:, 0:1] == theta[:, 1:2], theta, swapped)
        theta = swapped
        delta_tc = theta[:, 9:10]
        theta_aligned = torch.cat(
            [body[:, 0:2], body[:, 4:5], body[:, 7:8], body[:, 8:9],
             torch.zeros_like(theta[:, 0:3])], dim=1)
        derived = self.compute_derived_parameters(theta_aligned)
        M_s = derived[:, 2:3]
        f_max_prime, f_last = self.checked_band(derived)
        # Do not generate waveform above Mf_max
        within = self.within_band(M_s, f_max_prime, f_last)
        prec = self.get_and_set_precession_variables(theta, derived)
        afinal = prec["afinal"]
        amp22_coeffs = self.get_amp_coeffs(derived, afinal=afinal)
        phase22_coeffs = self.get_phase_coeffs(derived, afinal=afinal)

        Mf = self.f * M_s
        hp = torch.zeros((self.B, self.f_numel), dtype=torch.complex128,
                         device=self.f.device)
        hc = torch.zeros_like(hp)
        for ell in sorted(self.COPREC_MODES):
            for mprime in self.COPREC_MODES[ell]:
                htildelm = self.gethlm(theta_aligned, ell, mprime,
                                       derived=derived,
                                       amp22_coeffs=amp22_coeffs,
                                       phase22_coeffs=phase22_coeffs,
                                       afinal=afinal, within=within)
                hp_c, hc_c = self.twist_up(Mf, htildelm, derived, prec,
                                           ell, mprime)
                hp = hp + torch.where(within, hp_c, torch.zeros_like(hp_c))
                hc = hc + torch.where(within, hc_c, torch.zeros_like(hc_c))

        # Rotate waveform by 2 zeta
        cos2z = torch.cos(2.0 * prec["zeta_polarization"])
        sin2z = torch.sin(2.0 * prec["zeta_polarization"])
        hp, hc = cos2z * hp + sin2z * hc, cos2z * hc - sin2z * hp

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
