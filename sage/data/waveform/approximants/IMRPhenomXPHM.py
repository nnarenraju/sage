#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : IMRPhenomXPHM.py
Description   : GPU-native batched IMRPhenomXPHM precessing-spin
                frequency-domain waveform model with higher multipoles.

                Implements the IMRPhenomXPHM model entirely in PyTorch,
                following Pratten et al. (2020), arXiv:2004.06503
                ("Precessing paper").  Co-precessing-frame modes come from
                IMRPhenomXHM (and IMRPhenomXAS for the 22); this class adds
                the precessing-frame machinery: reference-frame geometry,
                MSA/NNLO Euler angles, per-mode Wigner-d twist-up and the
                final polarization rotation.

                Ported from lalsimulation-v6.1.0 (sources vendored in
                sage/data/waveform/lalsim_src/):
                  LALSimIMRPhenomXPHM.c          entry points, twist-up,
                                                 mode loop, assembly
                  LALSimIMRPhenomX_precession.c  precession variables,
                                                 MSA system, NNLO angles,
                                                 Wigner-d, final spin

                Fixed to the v6.1.0 default configuration:
                  PhenomXPrecVersion 300 -> 223 (MSA; precession.c:103)
                  with automatic per-sample fallback to NNLO 102 when the
                  MSA initialization fails (|Omegaz5| > 1000),
                  PhenomXPConvention 1, PhenomXPFinalSpinMod 4 -> 3,
                  ExpansionOrder 5, no PNR tuned angles, no antisymmetric
                  contribution, no mode/angle multibanding.
                The LAL reference must therefore insert
                PhenomXHMThresholdMband = 0 and PhenomXPHMThresholdMband = 0.
                SpinTaylor angle versions (310-321) are not ported.

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
IMRPhenomXPHM : Pratten et al. (2020), arXiv:2004.06503
MSA angles    : Chatziioannou et al. (2017), arXiv:1703.03967
NNLO angles   : Bohe et al., LIGO-T1500602; Schmidt et al. (2014),
                arXiv:1408.1810
IMRPhenomXHM  : García-Quirós et al. (2020), arXiv:2001.10914
"""

import math

import torch

from sage.core.config import get_cfg, get_data_cfg
from sage.core.math import safe_sqrt
from sage.data.waveform import taper as _taper_mod
from sage.data.waveform import waveform_utils
from sage.data.waveform.approximants.IMRPhenomXHM import IMRPhenomXHM


class IMRPhenomXPHM(IMRPhenomXHM, torch.nn.Module):
    """
    GPU-native batched IMRPhenomXPHM precessing multimode BBH waveform.

    Assembly (default path of XLALSimIMRPhenomXPHM ->
    IMRPhenomXPHM_hplushcross, LALSimIMRPhenomXPHM.c:167/740):

    1. aligned-spin waveform variables from (m1, m2, chi1z, chi2z);
    2. precession variables: chi_p, source-frame J0 geometry, thetaJN,
       alpha0/epsilon0, zeta_polarization, precessing final spin
       (fRING/fDAMP recomputed), MSA system initialization with NNLO
       fallback, Euler-angle offsets at fRef;
    3. per positive m' co-precessing mode h_{l,-m'} from XHM/XAS, built
       with the precessing-final-spin ringdown;
    4. per-frequency twist-up (eqs. 3.5-3.7 / E18-E19): Wigner-d matrices
       from cos(beta), transfer sum over inertial m in [-l, l], epsilon
       phase e^{-i m' epsilon}; accumulation into hp/hc;
    5. polarization rotation by 2*zeta_polarization.

    All precession state lives in a plain dict of (B, 1) tensors keyed by
    the IMRPhenomXPrecessionStruct member names; every per-frequency MSA
    function is stateless and vectorized over (B, F) (the C code mutates
    struct fields per frequency; the port re-derives them locally).

    Construction (two modes, dispatched on the first argument):

    IMRPhenomXPHM(f, f_ref)
        Research mode: explicit frequency grid, as IMRPhenomXAS/XHM.

    IMRPhenomXPHM(param_sampler, waveform_project, augment=...)
        Production signal-sampler mode, mirroring IMRPhenomPv2: the grid
        is built from the registered configs and :meth:`forward` yields
        (projected strain, targets) batches.

    Parameters (research mode)
    --------------------------
    f : torch.Tensor, shape (B, F)
        Frequency grid in Hz.
    f_ref : torch.Tensor, shape (B, 1)
        Reference frequency in Hz.
    **kwargs
        Forwarded to IMRPhenomXHM / IMRPhenomXAS / PhenomConstants.
    """

    # v6.1.0 defaults (empirically verified against the installed
    # lalsimulation 6.1.0 bindings; see lalsim_src provenance).
    PREC_VERSION = 223          # MSA (LALDict default 300 -> 223, precession.c:103)
    NNLO_FALLBACK_VERSION = 102 # MSA_ERROR fallback (precession.c:410-415)
    EXPANSION_ORDER = 5         # MSA truncation; zeroes the n=5 coefficients
                                # (precession.c:2652-2705)
    CONVENTION = 1              # PhenomXPConvention
    FINAL_SPIN_MOD = 3          # default 4 remapped for non-SpinTaylor
                                # (precession.c:1160)

    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        if args and torch.is_tensor(args[0]):
            IMRPhenomXHM.__init__(self, *args, **kwargs)
        else:
            self._init_production(*args, **kwargs)

    def _init_production(self, param_sampler=None, waveform_project=None,
                         augment=None, append_per_det_targets=False,
                         extra_batch=0):
        """
        Production signal-sampler construction, mirroring IMRPhenomPv2:
        frequency grid from the registered configs, parameter-column
        wiring into the waveform layout, and target-normaliser
        compilation.
        """
        self.cfg = get_cfg()
        self.data_cfg = get_data_cfg()

        self.signal_batch_size = (
            int(self.cfg.batch_size * self.cfg.class_balance) + int(extra_batch)
        )

        f, f_ref = waveform_utils.get_freqs(
            self.data_cfg.signal_low_frequency_cutoff,
            self.data_cfg.sample_rate / 2.0,
            self.data_cfg.padded_length_in_s,
            self.signal_batch_size,
            self.cfg.device,
            torch.float64,
        )
        IMRPhenomXHM.__init__(self, f, f_ref)
        self.sample_length_in_s = float(self.data_cfg.padded_length_in_s)
        self._out_cdtype = torch.promote_types(self.cfg.dtype, torch.complex64)

        self.param_sampler = param_sampler
        self.waveform_project = waveform_project
        self.augment = augment
        self.append_per_det_targets = bool(append_per_det_targets)
        self.mc_pe_idx = list(self.cfg.do_point_estimate).index("mchirp")

        param_index = self.param_sampler.param_index
        self.source, self.target = waveform_utils.layout_columns(param_index)
        self.dist_col = param_index["distance"]
        self.tc_col = param_index["tc"]

        self.param_sampler._compile_batch_normaliser()
        self.param_sampler._compile_batch_standardiser()

        self.param_sampler.to(self.cfg.device)
        self.waveform_project.to(self.cfg.device)

    @torch.no_grad()
    def forward(self, return_theta=False):
        """
        Production sampling step, as IMRPhenomPv2.forward: draw a
        parameter batch, evaluate the waveform (float64 internally, cast
        to the pipeline complex dtype), project onto the detector
        network, apply the SNR augmentation, and build the normalised
        target tensor.
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
            # augment returns (hf_scaled, scale) where hf_new = hf_old * scale.
            # Since strain ∝ 1/distance, distance_new = distance_old / scale.
            hf, scale = self.augment(hf)
            all_theta[:, self.dist_col] = (
                all_theta[:, self.dist_col] / scale.to(all_theta.dtype)
            )

        if getattr(self.cfg, "pe_target_minmax", False):
            normed_targets = self.param_sampler.norm_from_batch(all_theta)
        else:
            normed_targets = self.param_sampler.standardise_from_batch(all_theta)
        targets = torch.cat(
            [normed_targets, torch.ones_like(normed_targets[:, :1])], dim=1
        )

        # Per-detector targets appended AFTER the class column, layout as
        # in IMRPhenomPv2.forward.
        if self.append_per_det_targets:
            D = dt.shape[1]
            per_det_tc = all_theta[:, self.tc_col].unsqueeze(1) + dt
            per_det_mc = normed_targets[:, self.mc_pe_idx : self.mc_pe_idx + 1].expand(
                -1, D
            )
            targets = torch.cat([targets, per_det_tc, per_det_mc], dim=1)

        if return_theta:
            return hf, targets, all_theta
        else:
            return hf, targets

    # ------------------------------------------------------------------
    # Input ordering
    # Source: XLALIMRPhenomXPCheckMassesAndSpins,
    #         LALSimIMRPhenomXUtilities.c
    # ------------------------------------------------------------------

    @staticmethod
    def check_masses_and_spins(theta):
        """
        Enforce m1 > m2 by swapping bodies (masses and all six spin
        components) where needed, via torch.where on the mass ordering.
        As in C, m1 == m2 also takes the swap branch, so the spin labels
        exchange there too.  Sage samplers already emit m1 >= m2; the
        swap is kept for exact LAL parity on arbitrary inputs.
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
    # Precession variables (pPrec dict)
    # Source: IMRPhenomXGetAndSetPrecessionVariables, precession.c:67-1097
    # ------------------------------------------------------------------

    def get_and_set_precession_variables(self, theta, derived, version=None):
        """
        Build the precession dict (mirrors IMRPhenomXPrecessionStruct for
        the default MSA-223 / convention-1 path; version 102 selects the
        NNLO single-spin machinery as used by the fallback and the
        PN-angles diagnostic hook).  Ordered as in C:

        1.  spins, norms, perpendicular projections, chiTot_perp
            (precession.c:270-314);
        2.  chi_p (Schmidt et al. arXiv:1408.1810; precession.c:331-346);
        3.  MSA system initialization (precession.c:405) with the
            per-sample MSA_ERROR mask -> NNLO 102 fallback
            (precession.c:407-420);
        4.  precessing remnant: afinal, fRING/fDAMP recomputed
            (_set_precessing_remnant_params);
        5.  orbital angular momentum coefficients, version-223 branch
            with non-conserved spin norms (precession.c:497-513), and
            LRef (precession.c:569);
        6.  source-frame J0, thetaJ_Sf, phiJ_Sf, kappa, alpha0, thetaJN,
            zeta_polarization, epsilon0, phi0_aligned — convention 1
            (_compute_frame_angles);
        7.  NNLO alpha/epsilon coefficients for the fallback samples
            (precession.c:862-951; the cached in-struct convention, not
            the external XLALSimIMRPhenomXPNEuler* wrappers);
        8.  Euler-angle offsets at fRef, mprime = 2 (all per-mprime
            offsets are identical in v6.1.0; precession.c:989-1012);
        9.  spin-weighted Y_lm table at (thetaJN, 0), l = 2..4
            (precession.c:1058-1080).

        Returns
        -------
        dict of (B, 1) tensors keyed by LAL struct member names.

        Notes
        -----
        Current coverage: everything the MSA Euler angles need (spins,
        norms, 223 L-coefficients, LRef, convention-1 frame geometry,
        alpha0/epsilon0, MSA initialization, fRef offsets, thetaJN, Ylm
        table).  The precessing remnant (fRING/fDAMP override), the NNLO
        fallback coefficients and zeta_polarization are wired in the
        Stage C assembly passes.
        """
        eta = derived[:, 3:4]
        M_s = derived[:, 2:3]
        m1s = derived[:, 0:1]
        m2s = derived[:, 1:2]
        Mtot = m1s + m2s
        m1 = m1s / Mtot   # mass fractions, as stored in pWF (m1 > m2)
        m2 = m2s / Mtot
        chi1x = theta[:, 2:3]
        chi1y = theta[:, 3:4]
        chi1z = theta[:, 4:5]
        chi2x = theta[:, 5:6]
        chi2y = theta[:, 6:7]
        chi2z = theta[:, 7:8]
        phiRef = theta[:, 10:11]
        inclination = theta[:, 11:12]

        prec = {
            "eta": eta, "eta2": eta * eta, "eta3": eta ** 3, "eta4": eta ** 4,
            "inveta": 1.0 / eta, "inveta2": 1.0 / (eta * eta),
            "inveta3": 1.0 / eta ** 3, "inveta4": 1.0 / eta ** 4,
            "sqrt_inveta": torch.sqrt(1.0 / eta),
            "m1": m1, "m2": m2,
            "chi1x": chi1x, "chi1y": chi1y, "chi1z": chi1z,
            "chi2x": chi2x, "chi2y": chi2y, "chi2z": chi2z,
        }
        m1_2 = m1 * m1
        m2_2 = m2 * m2

        # SI frequency -> v conversions (precession.c:266-267)
        prec["piGM"] = math.pi * M_s
        prec["fRef"] = self.f_ref

        # In-plane spins and chi_p (Schmidt et al arXiv:1408.1810;
        # precession.c:301-349)
        chi1_perp = torch.sqrt(chi1x * chi1x + chi1y * chi1y)
        chi2_perp = torch.sqrt(chi2x * chi2x + chi2y * chi2y)
        S1_perp = m1_2 * chi1_perp
        S2_perp = m2_2 * chi2_perp
        STot_perpx = m1_2 * chi1x + m2_2 * chi2x
        STot_perpy = m1_2 * chi1y + m2_2 * chi2y
        prec["STot_perp"] = torch.sqrt(STot_perpx ** 2 + STot_perpy ** 2)
        prec["chiTot_perp"] = prec["STot_perp"] * (Mtot / Mtot) / m1_2
        A1 = 2.0 + (3.0 * m2) / (2.0 * m1)
        A2 = 2.0 + (3.0 * m1) / (2.0 * m2)
        ASp1 = A1 * S1_perp
        ASp2 = A2 * S2_perp
        num = torch.maximum(ASp1, ASp2)
        den = A1 * m1_2   # m1 > m2
        prec["chi_p"] = num / den
        prec["SL"] = chi1z * m1_2 + chi2z * m2_2
        prec["Sperp"] = prec["chi_p"] * m1_2

        # Orbital angular momentum coefficients: versions 222/223 use the
        # non-conserved-spin-norm 3PN L (precession.c:497-513); the NNLO
        # family 102 (and 220/221/224) uses the linear-in-spin 3PN L
        # (precession.c:472-495)
        chi1L, chi2L = chi1z, chi2z
        delta = derived[:, 4:5]
        eta2 = prec["eta2"]
        pi2 = math.pi * math.pi
        version = self.PREC_VERSION if version is None else version
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
            common_L["L3"] = (-7 * (chi1L + chi2L + chi1L * delta - chi2L * delta)
                              + 5 * (chi1L + chi2L) * eta) / 6.
            common_L["L5"] = (-1650 * (chi1L + chi2L + chi1L * delta - chi2L * delta)
                              + 1336 * (chi1L + chi2L) * eta
                              + 511 * (chi1L - chi2L) * delta * eta
                              + 28 * (chi1L + chi2L) * eta2) / 600.
        else:
            common_L["L3"] = (5 * (chi1L * (-2 - 2 * delta + eta)
                                   + chi2L * (-2 + 2 * delta + eta))) / 6.
            common_L["L5"] = (-7 * (chi1L * (72 + delta * (72 - 31 * eta)
                                             + eta * (-121 + 2 * eta))
                                    + chi2L * (72 + eta * (-121 + 2 * eta)
                                               + delta * (-72 + 31 * eta)))) / 144.
        prec["L_coeffs"] = common_L

        # LRef = M^2 LPN(v_ref), M = 1 in mass fractions (precession.c:569)
        v_ref = torch.pow(prec["piGM"] * self.f_ref, 1.0 / 3.0)
        prec["LRef"] = self._lpn_ansatz(v_ref, eta / v_ref, prec["L_coeffs"])

        # Source frame J0 (precession.c:578-593)
        J0x = m1_2 * chi1x + m2_2 * chi2x
        J0y = m1_2 * chi1y + m2_2 * chi2y
        J0z = m1_2 * chi1z + m2_2 * chi2z + prec["LRef"]
        J0 = torch.sqrt(J0x * J0x + J0y * J0y + J0z * J0z)
        prec["J0"] = J0
        thetaJ_Sf = torch.where(J0 < 1e-10, torch.zeros_like(J0),
                                torch.acos(J0z / J0))
        prec["thetaJ_Sf"] = thetaJ_Sf

        # Azimuthal angle of J0; aligned-spin limit is 0 for convention 1
        # (precession.c:608-638)
        MAX_TOL_ATAN = 1.0e-15
        aligned_J = (torch.abs(J0x) < MAX_TOL_ATAN) & (torch.abs(J0y) < MAX_TOL_ATAN)
        phiJ_Sf = torch.where(aligned_J, torch.zeros_like(J0),
                              torch.atan2(J0y, J0x))
        prec["phiJ_Sf"] = phiJ_Sf
        prec["phi0_aligned"] = -phiJ_Sf
        # convention 1: pWF->phi0 = 0 (precession.c:647-650); phiRef still
        # enters the frame geometry below

        # kappa via rotations RZ(-phiJ).RY(-thetaJ) of the line of sight
        # (precession.c:677-689)
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
        kappa = torch.where((torch.abs(tx) < MAX_TOL_ATAN) & (torch.abs(ty) < MAX_TOL_ATAN),
                            torch.zeros_like(tx), torch.atan2(ty, tx))
        prec["kappa"] = kappa

        # alpha0 = pi - kappa for convention 1, both aligned and generic
        # branches (precession.c:706-741)
        prec["alpha0"] = math.pi - kappa

        # thetaJN from J0 . N (convention 1; precession.c:766-776)
        J0dotN = J0x * Nx_Sf + J0y * Ny_Sf + J0z * Nz_Sf
        thetaJN = torch.acos(J0dotN / J0)
        prec["thetaJN"] = thetaJN
        prec["Nz_Jf"] = torch.cos(thetaJN)
        prec["Nx_Jf"] = torch.sin(thetaJN)

        # epsilon0 = phiJ_Sf - pi (convention 1; precession.c:974-987)
        prec["epsilon0"] = phiJ_Sf - math.pi

        # Polarization angle zeta: rotate the waveframe X vector into the
        # J frame and take its polar angle in the P, Q basis of Arun et
        # al (convention 1 triad P = (Nz, 0, -Nx), Q = (0, 1, 0));
        # the polarizations are later rotated by 2 zeta
        # (precession.c:788-855)
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
            # MSA system (precession.c:398-421)
            self._initialize_msa_system(theta, derived, prec)
            # Precessing remnant: afinal_prec and fRING/fDAMP override
            # (precession.c:434, 1101-1345)
            self._set_precessing_remnant_params(prec, derived)
            # Euler angle offsets at fRef, mprime = 2; all mprime share
            # these in v6.1.0 (Get_alphaepsilon_atfref, precession.c:1356-1363)
            vangles = self.msa_phi_zeta_costhetaL(v_ref, derived, prec)
            prec["alpha_offset"] = vangles[0] - prec["alpha0"]
            prec["epsilon_offset"] = vangles[1] - prec["epsilon0"]
            if bool(prec["MSA_ERROR"].any()):
                # Samples for which the MSA system failed (|Omegaz5| >
                # 1000) are downgraded to the NNLO 102 path.  In C the
                # downgrade happens before the L-coefficient caching
                # (precession.c:407-420), so those samples get the
                # 102-family chain throughout: L3/L5, LRef, frame
                # geometry, remnant, offsets and Ylm table.  Rebuild with
                # version 102 and select per sample.
                prec_nnlo = self.get_and_set_precession_variables(
                    theta, derived, version=102)
                mask = prec["MSA_ERROR"]
                for key, val in prec_nnlo.items():
                    if key in ("version", "MSA_ERROR"):
                        continue
                    if key == "L_coeffs":
                        prec["L_coeffs"] = {
                            k: torch.where(mask, val[k], prec["L_coeffs"][k])
                            for k in val
                        }
                    elif key in prec and torch.is_tensor(val) \
                            and val.is_floating_point():
                        prec[key] = torch.where(mask, val, prec[key])
                    elif key in prec and torch.is_tensor(val) \
                            and val.is_complex():
                        prec[key] = torch.where(mask, val, prec[key])
                    elif key not in prec:
                        # NNLO angle coefficients, consumed only on the
                        # masked samples by get_alpha_epsilon_cosbeta
                        prec[key] = val
        else:
            # NNLO single-spin PN Euler angle coefficients
            # (precession.c:862-925)
            prec["MSA_ERROR"] = torch.zeros_like(eta, dtype=torch.bool)
            self._set_precessing_remnant_params(prec, derived)
            self._nnlo_angle_coeffs(derived, prec)
            # NNLO offsets at omega_ref (Get_alphaepsilon_atfref,
            # precession.c:1366-1386)
            omega_ref = prec["piGM"] * self.f_ref
            logw = torch.log(omega_ref)
            wcbrt = torch.pow(omega_ref, 1.0 / 3.0)
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

        # Spin-weighted spherical harmonics at (thetaJN, 0), l = 2..4;
        # read thetaJN back from the dict so fallback samples use their
        # merged 102-frame value (precession.c:1058-1080)
        for ell in (2, 3, 4):
            for emm in range(-ell, ell + 1):
                prec[f"Y{ell}{'m' if emm < 0 else ''}{abs(emm)}"] = \
                    self.spin_weighted_Y(prec["thetaJN"], 0.0, -2, ell, emm)
        return prec

    def _chi_p(self, theta, derived):
        """
        Effective precessing-spin parameter chi_p.
        Source: precession.c:331-346.
        """
        raise NotImplementedError("port pending: LALSimIMRPhenomX_precession.c:331-346")

    def _set_precessing_remnant_params(self, prec, derived):
        """
        Precessing final spin and remnant quantities (default fsflag 3):

        af_parallel = XLALSimIMRPhenomXFinalSpin2017(eta, chi1z, chi2z)
        Lfinal      = M^2 af_parallel - m1^2 chi1z - m2^2 chi2z
        afinal_prec = sign(af_parallel) *
                      sqrt(SAv2 + Lfinal^2 + 2 Lfinal (S1L_pav + S2L_pav)) / M^2

        with fallback to XLALSimIMRPhenomXPrecessingFinalSpin2017(eta,
        chi1L, chi2L, chi_p) for MSA_ERROR samples, |afinal| clamped to 1,
        and fRING/fDAMP re-evaluated at afinal (precession.c:1287-1288)
        via the inherited IMRPhenomXAS.get_fRD_fdamp.
        Source: IMRPhenomX_SetPrecessingRemnantParams,
        precession.c:1101-1345.
        """
        eta = prec["eta"]
        delta = derived[:, 4:5]
        chi1z = prec["chi1z"]
        chi2z = prec["chi2z"]
        m1_2 = prec["m1"] * prec["m1"]
        m2_2 = prec["m2"] * prec["m2"]
        # aligned-spin final spin/mass via the (eta, STotR, dchi, delta)
        # reparametrization of the Jimenez-Forteza fits (binary-verified)
        STotR = (m1_2 * chi1z + m2_2 * chi2z) / (m1_2 + m2_2)
        dchi = chi1z - chi2z
        af_parallel = self.final_spin_2017(eta, STotR, dchi, delta)
        Mfinal = self.final_mass_2017(eta, STotR, dchi, delta)

        # Lfinal = M^2 afinal - m1^2 chi1z - m2^2 chi2z, M = 1
        # (precession.c:1157)
        Lfinal = af_parallel - m1_2 * chi1z - m2_2 * chi2z

        sign = torch.where(af_parallel >= 0,
                           torch.ones_like(af_parallel),
                           -torch.ones_like(af_parallel))
        # version 0: XLALSimIMRPhenomXPrecessingFinalSpin2017
        # (Utilities.c:225-253): af = sign(af_par) sqrt(Sperp^2 + af_par^2),
        # Sperp = chi_p q_factor^2, q_factor = m1/M
        Sperp0 = prec["chi_p"] * m1_2
        af0 = sign * torch.sqrt(Sperp0 * Sperp0 + af_parallel * af_parallel)
        if prec["version"] in (220, 221, 222, 223, 224):
            # default fsflag 4 -> 3: SAv^2-based precessing final spin
            # with sign(af_parallel) (TransPrecessionMethod 1;
            # precession.c:1204-1208); MSA failure -> version 0
            af3 = sign * torch.sqrt(prec["SAv2"] + Lfinal * Lfinal
                                    + 2.0 * Lfinal * (prec["S1L_pav"] + prec["S2L_pav"]))
            afinal = torch.where(prec["MSA_ERROR"], af0, af3)
        else:
            # fsflag 3 requires an MSA version; NNLO runs default to
            # version 0 (precession.c:1212-1216)
            afinal = af0

        # |afinal| <= 1 (precession.c:1280-1284)
        afinal = torch.clamp(afinal, -1.0, 1.0)
        prec["afinal_prec"] = afinal
        prec["afinal"] = afinal
        prec["Mfinal"] = Mfinal

        # fRING/fDAMP re-evaluated at the precessing final spin
        # (precession.c:1287-1288)
        prec["fRING"], prec["fDAMP"] = self.get_fRD_fdamp(afinal, Mfinal)
        return prec

    def _compute_frame_angles(self, theta, derived, prec):
        """
        Reference-frame geometry, convention 1 (default):

        J0_Sf = (m1^2 chi1 + m2^2 chi2) + LRef z_hat ;
        thetaJ_Sf, phiJ_Sf, kappa via the rotation composition
        RZ(-kappa) RY(-thetaJ_Sf) RZ(-phiJ_Sf);
        alpha0 = pi - kappa ; epsilon0 = phiJ_Sf - pi ; pWF phi0 = 0 ;
        thetaJN = acos(J0 . N / |J0|) ;
        zeta_polarization = atan2(X.QArun, X.PArun) with the convention-1
        triad P = (Nz, 0, -Nx), Q = (0, 1, 0).
        Source: precession.c:578-1012 (steps 20-26 of
        IMRPhenomXGetAndSetPrecessionVariables).
        """
        raise NotImplementedError("port pending: LALSimIMRPhenomX_precession.c:578-1012")

    def _get_alphaepsilon_atfref(self, derived, prec, mprime):
        """
        Euler-angle offsets at fRef:
        omega_ref = piM fRef 2/mprime ; MSA: angles(v(omega_ref)) - alpha0
        / - epsilon0 ; NNLO: closed-form series - alpha0 / - epsilon0.
        Source: Get_alphaepsilon_atfref, precession.c:1348-1389.
        """
        raise NotImplementedError("port pending: LALSimIMRPhenomX_precession.c:1348-1389")

    # ------------------------------------------------------------------
    # Orbital angular momentum (PN)
    # ------------------------------------------------------------------

    @staticmethod
    def _lpn_ansatz(v, l_norm, L):
        """
        L(v) = LNorm * sum_n L_n v^n + L8L v^8 log(v^2).
        Source: XLALSimIMRPhenomXLPNAnsatz, precession.c:1395-1424.
        """
        v2 = v * v
        v3 = v2 * v
        v4 = v3 * v
        v5 = v4 * v
        v6 = v5 * v
        v7 = v6 * v
        v8 = v7 * v
        return l_norm * (L["L0"] + L["L1"] * v + L["L2"] * v2 + L["L3"] * v3
                         + L["L4"] * v4 + L["L5"] * v5 + L["L6"] * v6
                         + L["L7"] * v7 + L["L8"] * v8
                         + L["L8L"] * v8 * torch.log(v2))

    @staticmethod
    def _l_norm_3pn_of_v(v, v2, l_norm, prec):
        """
        3PN orbital angular momentum with the constants_L[5] coefficients
        (versions 222/223 only).
        Source: IMRPhenomX_L_norm_3PN_of_v, precession.c:1954-1957.
        """
        cl = prec["constants_L"]
        return l_norm * (1. + v2 * (cl[0] + v * cl[1]
                                    + v2 * (cl[2] + v * cl[3] + v2 * cl[4])))

    @staticmethod
    def _pn_beta(a, b, prec):
        """Source: IMRPhenomX_Get_PN_beta, precession.c:3432-3435."""
        return (prec["dotS1L"] * (a + b * prec["qq"])
                + prec["dotS2L"] * (a + b / prec["qq"]))

    @staticmethod
    def _pn_sigma(a, b, prec):
        """Source: IMRPhenomX_Get_PN_sigma, precession.c:3441-3444."""
        return prec["inveta"] * (a * prec["dotS1S2"]
                                 - b * prec["dotS1L"] * prec["dotS2L"])

    @staticmethod
    def _pn_tau(a, b, prec):
        """Source: IMRPhenomX_Get_PN_tau, precession.c:3449-3452."""
        return (prec["qq"] * (prec["S1_norm_2"] * a - b * prec["dotS1L"] * prec["dotS1L"])
                + (a * prec["S2_norm_2"] - b * prec["dotS2L"] * prec["dotS2L"]) / prec["qq"]
                ) / prec["eta"]

    # ------------------------------------------------------------------
    # Special functions (GSL equivalents; exact, no approximations)
    # ------------------------------------------------------------------

    @staticmethod
    def _ellint_RF(x, y, z):
        """
        Carlson symmetric integral R_F(x, y, z) by the duplication
        algorithm, mirroring gsl_sf_ellint_RF_e (GSL specfunc/ellint.c)
        with the double-precision error tolerance errtol = 0.001.
        Elementwise on tensors; init-time only.
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

    @classmethod
    def _ellint_F(cls, phi, k):
        """
        Incomplete elliptic integral of the first kind F(phi, k) with the
        GSL modulus convention, mirroring gsl_sf_ellint_F_e
        (GSL specfunc/ellint.c): angular reduction to (-pi/2, pi/2],
        R_F(1 - sin^2 phi, 1 - k^2 sin^2 phi, 1) plus complete-integral
        periodicity terms.  Needed once per batch at MSA initialization
        (psi0, precession.c:2751-2765).
        """
        nc = torch.floor(phi / math.pi + 0.5)
        phi_red = phi - nc * math.pi
        sin_phi = torch.sin(phi_red)
        sin2_phi = sin_phi * sin_phi
        x = 1.0 - sin2_phi
        y = 1.0 - k * k * sin2_phi
        one = torch.ones_like(x)
        val = sin_phi * cls._ellint_RF(x, y, one)
        # add extra terms from periodicity (Kcomp = RF(0, 1 - k^2, 1))
        need_k = nc != 0
        if bool(need_k.any()):
            rk = cls._ellint_RF(torch.zeros_like(x), 1.0 - k * k, one)
            val = torch.where(need_k, val + 2 * nc * rk, val)
        return val

    @staticmethod
    def _elljac_sn(u, m):
        """
        Jacobi elliptic function sn(u | m), mirroring gsl_sf_elljac_e
        (GSL specfunc/elljac.c; Bulirsch AGM with the reflected
        sn(K-u)/cn(K-u)/dn(K-u) branch when |sin| < |cos|).  Vectorized
        over (B, F): the AGM level arrays depend on m only ((B, 1)); the
        descent runs top-down with per-element start level.
        """
        eps = torch.finfo(u.dtype).eps
        N = 16
        mu = [torch.ones_like(m)]
        nu = [torch.sqrt(1.0 - m)]
        # per-element AGM stopping level n
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
        nu_st = torch.stack(nu, dim=0)
        mu_n = torch.gather(mu_st, 0, n_elem.unsqueeze(0)).squeeze(0)

        sin_umu = torch.sin(u * mu_n)
        cos_umu = torch.cos(u * mu_n)
        refl = torch.abs(sin_umu) < torch.abs(cos_umu)
        t = torch.where(refl, sin_umu / cos_umu, cos_umu / sin_umu)

        # descent c[k] = d[k+1] c[k+1]; d[k] = (r + nu[k])/(r + mu[k]),
        # r = c[k+1]^2 / mu[k+1], initialized at each element's own level
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
        hyp = torch.hypot(torch.ones_like(c), c)
        # reflected branch: dn = sqrt(1-m)/d ; cn = dn sign(cos)/hypot(1,c);
        # sn = cn c / sqrt(1-m).  direct branch: sn = sign(sin)/hypot(1,c)
        dn_r = sqrt1m / d
        cn_r = dn_r * sgn(cos_umu) / hyp
        sn_r = cn_r * c / sqrt1m
        sn_d = sgn(sin_umu) / hyp
        sn = torch.where(refl, sn_r, sn_d)
        # small-|m| and m->1 limits (elljac.c)
        sn = torch.where(torch.abs(m) < 2.0 * eps, torch.sin(u), sn)
        sn = torch.where(torch.abs(m - 1.0) < 2.0 * eps, torch.tanh(u), sn)
        return sn

    # ------------------------------------------------------------------
    # MSA system (default angle model; version 223)
    # Source: IMRPhenomX_Initialize_MSA_System, precession.c:2029-2829
    # ------------------------------------------------------------------

    def _initialize_msa_system(self, theta, derived, prec):
        """
        One-time MSA initialization (Chatziioannou et al. arXiv:1703.03967,
        LALSimInspiralFDPrecAngles flavor for 222/223):

        - q-flip to the MSA convention qq = m2/m1 < 1 (precession.c:2069);
        - initial vectors S1_0, S2_0, L_0, J_0 at v_0 = cbrt(piGM fRef)
          and their norms; Seff;
        - S^2 cubic roots at v_0 -> Spl2, Smi2, S32; SAv2 = (Spl2+Smi2)/2;
        - c1 (Eq. 41), precession-averaged couplings (Eqs. A9-A14),
          spin-orbit beta3/5/6/7 and spin-spin sigma4;
        - TaylorT2 a-coefficients with the 222/223 initial-coupling
          override from the domegadt tables (precession.c:2360-2367);
          g-coefficients (Eqs. A1-A5);
        - psi1, Delta and psi2 in the 222/223 (FDPrecAngles) forms;
        - Omegaz0..5 / Omegazeta0..5 and their _coeff convolutions
          (Eqs. D10-D21, Appendix E), truncated by ExpansionOrder = 5
          (which zeroes the n=5 coefficients);
        - MSA_ERROR mask: |Omegaz5| > 1000 (precession.c:2610-2614), the
          only trigger in v6.1.0; flagged samples use NNLO 102;
        - psi0 via the incomplete elliptic integral (precession.c:2723);
        - initial angles phiz_0, zeta_0 (precession.c:2774-2800).

        All outputs are written into ``prec`` as (B, 1) tensors.
        """
        eta = prec["eta"]
        eta2 = prec["eta2"]
        eta3 = prec["eta3"]
        m1 = prec["m1"]
        m2 = prec["m2"]
        PI = math.pi

        # PN Coefficients for d omega / d t as per
        # LALSimInspiralFDPrecAngles_internals.c (precession.c:2056-2061)
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

        # q = m2/m1 < 1 (Chatziioannou et al convention; precession.c:2069)
        q = m2 / m1
        invq = 1.0 / q
        prec["qq"] = q
        prec["invqq"] = invq
        mu = (m1 * m2) / (m1 + m2)
        prec["delta_qq"] = (1.0 - q) / (1.0 + q)
        prec["delta2_qq"] = prec["delta_qq"] ** 2
        prec["delta3_qq"] = prec["delta_qq"] * prec["delta2_qq"]
        prec["delta4_qq"] = prec["delta_qq"] * prec["delta3_qq"]

        # Dimensionful spin vectors; eta/q = m1^2, eta*q = m2^2
        # (precession.c:2101-2107)
        S1x = prec["chi1x"] * eta / q
        S1y = prec["chi1y"] * eta / q
        S1z = prec["chi1z"] * eta / q
        S2x = prec["chi2x"] * eta * q
        S2y = prec["chi2y"] * eta * q
        S2z = prec["chi2z"] * eta * q
        S1_0_norm = torch.sqrt(S1x * S1x + S1y * S1y + S1z * S1z)
        S2_0_norm = torch.sqrt(S2x * S2x + S2y * S2y + S2z * S2z)

        # Reference velocity and orbital angular momentum, Lhat = (0,0,1)
        v_0 = torch.pow(prec["piGM"] * prec["fRef"], 1.0 / 3.0)
        v_0_2 = v_0 * v_0
        prec["v_0"] = v_0
        prec["v_0_2"] = v_0_2
        L0z = eta / v_0

        # Inner products used in MSA system (precession.c:2157-2168)
        dotS1L = S1z
        dotS2L = S2z
        dotS1S2 = S1x * S2x + S1y * S2y + S1z * S2z
        prec["dotS1L"] = dotS1L
        prec["dotS2L"] = dotS2L
        prec["dotS1S2"] = dotS1S2
        prec["dotS1Ln"] = dotS1L / S1_0_norm
        prec["dotS2Ln"] = dotS2L / S2_0_norm
        prec["S1_norm_2"] = S1_0_norm * S1_0_norm
        prec["S2_norm_2"] = S2_0_norm * S2_0_norm

        # 3PN orbital angular momentum coefficients as per
        # LALSimInspiralFDPrecAngles_internals.c (precession.c:2183-2187)
        prec["constants_L"] = [
            Lns[0] + eta * Lns[1],
            self._pn_beta(Lso[0], Lso[1], prec),
            Lns[2] + eta * Lns[3] + eta * eta * Lns[4],
            self._pn_beta(Lso[2] + Lso[3] * eta, Lso[4] + Lso[5] * eta, prec),
            Lns[5] + Lns[6] * eta + Lns[7] * eta * eta + Lns[8] * eta * eta * eta,
        ]

        # Effective total spin (precession.c:2190-2194)
        Seff = (1.0 + q) * dotS1L + (1 + invq) * dotS2L
        prec["Seff"] = Seff
        prec["Seff2"] = Seff * Seff

        # S_0 = S1 + S2 ; J_0 = L_0 + S_0 (precession.c:2201-2241)
        S0x, S0y, S0z = S1x + S2x, S1y + S2y, S1z + S2z
        prec["S_0_norm"] = torch.sqrt(S0x * S0x + S0y * S0y + S0z * S0z)
        prec["S_0_norm_2"] = prec["S_0_norm"] ** 2
        prec["L_0_norm"] = L0z
        J0x, J0y, J0z = S0x, S0y, L0z + S0z
        prec["J_0_norm"] = torch.sqrt(J0x * J0x + J0y * J0y + J0z * J0z)
        L0norm = prec["L_0_norm"]
        J0norm = prec["J_0_norm"]

        # Roots at v_0 -> SAv2 (Eq. 45; precession.c:2259-2282)
        S32, Smi2, Spl2 = self._msa_roots(L0norm, J0norm, prec)
        prec["Spl2"] = Spl2
        prec["Smi2"] = Smi2
        prec["S32"] = S32
        prec["Spl2pSmi2"] = Spl2 + Smi2
        prec["Spl2mSmi2"] = Spl2 - Smi2
        prec["Spl"] = torch.sqrt(Spl2)
        prec["Smi"] = torch.sqrt(Smi2)
        prec["SAv2"] = 0.5 * prec["Spl2pSmi2"]
        prec["SAv"] = torch.sqrt(prec["SAv2"])
        prec["invSAv2"] = 1.0 / prec["SAv2"]
        prec["invSAv"] = 1.0 / prec["SAv"]

        # c_1, Eq. 41 (precession.c:2294-2300)
        c_1 = 0.5 * (J0norm * J0norm - L0norm * L0norm - prec["SAv2"]) / L0norm * eta
        c1_2 = c_1 * c_1
        prec["c1"] = c_1
        prec["c12"] = c1_2
        prec["c1_over_eta"] = c_1 / eta

        # Precession averaged spin couplings, Eq. A9-A14 (precession.c:2303-2312)
        omqsq = (1.0 - q) * (1.0 - q) + 1e-16
        omq2 = (1.0 - q * q) + 1e-16
        S1L_pav = (c_1 * (1.0 + q) - q * eta * Seff) / (eta * omq2)
        S2L_pav = -q * (c_1 * (1.0 + q) - eta * Seff) / (eta * omq2)
        S1S2_pav = 0.5 * prec["SAv2"] - 0.5 * (prec["S1_norm_2"] + prec["S2_norm_2"])
        RmSq_v02 = prec["Spl2mSmi2"] * prec["Spl2mSmi2"] * v_0_2
        S1Lsq_pav = S1L_pav * S1L_pav + RmSq_v02 / (32.0 * eta2 * omqsq)
        S2Lsq_pav = S2L_pav * S2L_pav + q * q * RmSq_v02 / (32.0 * eta2 * omqsq)
        S1LS2L_pav = S1L_pav * S2L_pav - q * RmSq_v02 / (32.0 * eta2 * omqsq)
        prec["S1L_pav"] = S1L_pav
        prec["S2L_pav"] = S2L_pav

        # Spin couplings (precession.c:2315-2329)
        beta3 = ((113. / 12.) + (25. / 4.) * (m2 / m1)) * S1L_pav \
            + ((113. / 12.) + (25. / 4.) * (m1 / m2)) * S2L_pav
        beta5 = (((31319. / 1008.) - (1159. / 24.) * eta)
                 + (m2 / m1) * ((809. / 84) - (281. / 8.) * eta)) * S1L_pav \
            + (((31319. / 1008.) - (1159. / 24.) * eta)
               + (m1 / m2) * ((809. / 84) - (281. / 8.) * eta)) * S2L_pav
        beta6 = PI * (((75. / 2.) + (151. / 6.) * (m2 / m1)) * S1L_pav
                      + ((75. / 2.) + (151. / 6.) * (m1 / m2)) * S2L_pav)
        beta7 = (((130325. / 756) - (796069. / 2016) * eta + (100019. / 864.) * eta2)
                 + (m2 / m1) * ((1195759. / 18144) - (257023. / 1008.) * eta
                                + (2903 / 32.) * eta2) * S1L_pav
                 + ((130325. / 756) - (796069. / 2016) * eta + (100019. / 864.) * eta2)
                 + (m1 / m2) * ((1195759. / 18144) - (257023. / 1008.) * eta
                                + (2903 / 32.) * eta2) * S2L_pav)
        sigma4 = (1.0 / mu) * ((247. / 48.) * S1S2_pav - (721. / 48.) * S1L_pav * S2L_pav) \
            + (1.0 / (m1 * m1)) * ((233. / 96.) * prec["S1_norm_2"] - (719. / 96.) * S1Lsq_pav) \
            + (1.0 / (m2 * m2)) * ((233. / 96.) * prec["S2_norm_2"] - (719. / 96.) * S2Lsq_pav)

        # TaylorT2 dw/dt coefficients (precession.c:2332-2349), then the
        # 222/223 initial-coupling override (precession.c:2360-2367)
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

        # psi1, Eq. C1 (precession.c:2417-2422)
        delta = prec["delta_qq"]
        delta2 = delta * delta
        delta4 = delta2 * delta2
        prec["psi1"] = 3.0 * (2.0 * eta2 * Seff - c_1) / (eta * delta2)

        c_1_over_nu = prec["c1_over_eta"]
        c_1_over_nu_2 = c_1_over_nu * c_1_over_nu
        one_p_q_sq = (1. + q) * (1. + q)
        Seff_2 = Seff * Seff
        q_2 = q * q
        one_m_q_sq = (1. - q) * (1. - q)
        one_m_q2_2 = (1. - q_2) * (1. - q_2)
        one_m_q_4 = one_m_q_sq * one_m_q_sq

        # Delta, FDPrecAngles form for 222/223 (precession.c:2438-2447)
        Del1 = 4. * c_1_over_nu_2 * one_p_q_sq
        Del2 = 8. * c_1_over_nu * q * (1. + q) * Seff
        Del3 = 4. * (one_m_q2_2 * prec["S1_norm_2"] - q_2 * Seff_2)
        Del4 = 4. * c_1_over_nu_2 * q_2 * one_p_q_sq
        Del5 = 8. * c_1_over_nu * q_2 * (1. + q) * Seff
        Del6 = 4. * (one_m_q2_2 * prec["S2_norm_2"] - q_2 * Seff_2)
        Delta = torch.sqrt(torch.abs((Del1 - Del2 - Del3) * (Del4 - Del5 - Del6)))
        prec["Delta"] = Delta

        # psi2, Eq. C2, FDPrecAngles form for 222/223 (precession.c:2472-2484)
        u1 = 3. * g2 / g0
        u2 = 0.75 * one_p_q_sq / one_m_q_4
        u3 = -20. * c_1_over_nu_2 * q_2 * one_p_q_sq
        u4 = 2. * one_m_q2_2 * (q * (2. + q) * prec["S1_norm_2"]
                                + (1. + 2. * q) * prec["S2_norm_2"] - 2. * q * prec["SAv2"])
        u5 = 2. * q_2 * (7. + 6. * q + 7. * q_2) * 2. * c_1_over_nu * Seff
        u6 = 2. * q_2 * (3. + 4. * q + 3. * q_2) * Seff_2
        u7 = q * Delta
        prec["psi2"] = u1 + u2 * (u3 + u4 + u5 - u6 + u7)

        # D-appendix constants (precession.c:2508-2557)
        Rm = Spl2 - Smi2
        Rm_2 = Rm * Rm
        cp = Spl2 * eta2 - c1_2
        cm = Smi2 * eta2 - c1_2
        cpcm = torch.abs(cp * cm)
        sqrt_cpcm = torch.sqrt(cpcm)
        a1dD = 0.5 + 0.75 / eta
        a2dD = -0.75 * Seff / eta
        D2RmSq = (cp - sqrt_cpcm) / eta2
        eta4 = eta2 * eta2
        D4RmSq = -0.5 * Rm * sqrt_cpcm / eta2 - cp / eta4 * (sqrt_cpcm - cp)
        S0m = prec["S1_norm_2"] - prec["S2_norm_2"]
        aw = (-3. * (1. + q) / q * (2. * (1. + q) * eta2 * Seff * c_1
                                    - (1. + q) * c1_2 + (1. - q) * eta2 * S0m))
        cw = 3. / 32. / eta * Rm_2
        dw = 4.0 * cp - 4.0 * D2RmSq * eta2
        hw = -2.0 * (2.0 * D2RmSq - Rm) * c_1
        fw = Rm * D2RmSq - D4RmSq - 0.25 * Rm_2
        adD = aw / dw
        hdD = hw / dw
        cdD = cw / dw
        fdD = fw / dw
        gw = 3. / 16. / eta2 / eta * Rm_2 * (c_1 - eta2 * Seff)
        gdD = gw / dw
        hdD_2 = hdD * hdD
        adDfdD = adD * fdD
        adDfdDhdD = adDfdD * hdD
        adDhdD_2 = adD * hdD_2

        # Omegaz, Eqs. D10-D15 (precession.c:2580-2595)
        Omegaz0 = a1dD + adD
        Omegaz1 = a2dD - adD * Seff - adD * hdD
        Omegaz2 = adD * hdD * Seff + cdD - adD * fdD + adD * hdD_2
        Omegaz3 = (adDfdD - cdD - adDhdD_2) * (Seff + hdD) + adDfdDhdD
        Omegaz4 = (cdD + adDhdD_2 - 2.0 * adDfdD) * (hdD * Seff + hdD_2 - fdD) \
            - adD * fdD * fdD
        Omegaz5 = (cdD - adDfdD + adDhdD_2) * fdD * (Seff + 2.0 * hdD) \
            - (cdD + adDhdD_2 - 2.0 * adDfdD) * hdD_2 * (Seff + hdD) \
            - adDfdD * fdD * hdD

        # If |Omegaz5| > 1000 the system may be pathological: MSA_ERROR
        # (precession.c:2610-2614)
        prec["MSA_ERROR"] = torch.abs(Omegaz5) > 1000.0

        # Eq. 65 coefficients, Eqs. D16-D21 (precession.c:2619-2624)
        prec["Omegaz0_coeff"] = 3.0 * g0 * Omegaz0
        prec["Omegaz1_coeff"] = 3.0 * g0 * Omegaz1
        prec["Omegaz2_coeff"] = 3.0 * (g0 * Omegaz2 + g2 * Omegaz0)
        prec["Omegaz3_coeff"] = 3.0 * (g0 * Omegaz3 + g2 * Omegaz1 + g3 * Omegaz0)
        prec["Omegaz4_coeff"] = 3.0 * (g0 * Omegaz4 + g2 * Omegaz2 + g3 * Omegaz1
                                       + g4 * Omegaz0)
        prec["Omegaz5_coeff"] = 3.0 * (g0 * Omegaz5 + g2 * Omegaz3 + g3 * Omegaz2
                                       + g4 * Omegaz1 + g5 * Omegaz0)

        # zeta coefficients, Appendix E (precession.c:2627-2649)
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

        # ExpansionOrder truncation: default 5 ZEROES the n=5 coefficients
        # (fallthrough switch, precession.c:2652-2705; only -1 keeps them)
        if self.EXPANSION_ORDER != -1:
            for n in range(self.EXPANSION_ORDER, 6):
                if n >= 1:
                    prec[f"Omegaz{n}_coeff"] = torch.zeros_like(eta)
                    prec[f"Omegazeta{n}_coeff"] = torch.zeros_like(eta)

        # psi0 via the incomplete elliptic integral (precession.c:2736-2766)
        near_equal = torch.abs(Smi2 - Spl2) < 1.0e-5
        mm = torch.sqrt((Smi2 - Spl2) / (S32 - Spl2))
        tmpB = (prec["S_0_norm"] ** 2 - Spl2) / (Smi2 - Spl2)
        volume_element = (L0z * (S1x * S2y - S1y * S2x))
        vol_sign = torch.where(volume_element > 0, torch.ones_like(eta),
                               torch.where(volume_element < 0,
                                           -torch.ones_like(eta),
                                           torch.zeros_like(eta)))
        prec["psi0"] = torch.zeros_like(eta)
        psi_of_v0 = self._msa_psiofv(v_0, v_0_2, prec["psi0"], prec["psi1"],
                                     prec["psi2"], prec)
        # tmpB clamped to [0, 1] within tolerance 1e-5 (precession.c:2751-2765)
        tmpB_cl = torch.where((tmpB > 1.0) & (tmpB - 1. < 0.00001),
                              torch.ones_like(tmpB), tmpB)
        tmpB_cl = torch.where((tmpB < 0.0) & (tmpB > -0.00001),
                              torch.zeros_like(tmpB), tmpB_cl)
        tmpB_valid = (tmpB_cl >= 0.) & (tmpB_cl <= 1.)
        arg = torch.where(tmpB_valid, tmpB_cl, torch.zeros_like(tmpB_cl))
        psi0_val = self._ellint_F(torch.asin(vol_sign * torch.sqrt(arg)), mm) - psi_of_v0
        prec["psi0"] = torch.where(near_equal | ~tmpB_valid,
                                   torch.zeros_like(eta), psi0_val)

        # Initial angles (precession.c:2774-2800); corrections only when
        # |Spl2 - Smi2| > 1e-5
        prec["phiz_0"] = torch.zeros_like(eta)
        prec["zeta_0"] = torch.zeros_like(eta)
        corr_x, corr_y = self._msa_corrections(v_0, L0norm, J0norm, prec)
        big = torch.abs(Spl2 - Smi2) > 1.e-5
        corr_x = torch.where(big, corr_x, torch.zeros_like(corr_x))
        corr_y = torch.where(big, corr_y, torch.zeros_like(corr_y))
        phiz_0 = self._msa_phiz(v_0, J0norm, prec)
        zeta_0 = self._msa_zeta(v_0, prec)
        prec["phiz_0"] = -phiz_0 - corr_x
        prec["zeta_0"] = -zeta_0 - corr_y

    # -- MSA per-frequency chain (stateless, vectorized over (B, F)) ----

    @staticmethod
    def _msa_j_norm(l_norm, prec):
        """
        JNorm = sqrt(L^2 + 2 L c1/eta + SAv2)  (Eq. 41).
        Source: IMRPhenomX_JNorm_MSA, precession.c:2962-2966.
        """
        return torch.sqrt(l_norm * l_norm + 2.0 * l_norm * prec["c1_over_eta"]
                          + prec["SAv2"])

    @staticmethod
    def _msa_spin_evolution_coefficients(l_norm, j_norm, prec):
        """
        Coefficients B, C, D of the S^2 evolution cubic (Eqs. B2-B4).
        Source: IMRPhenomX_Return_Spin_Evolution_Coefficients_MSA,
        precession.c:3009-3059.
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

    @classmethod
    def _msa_roots(cls, l_norm, j_norm, prec):
        """
        Roots (S32, Smi2, Spl2) of the S^2 cubic via the trigonometric
        solve, with the degeneracy guard (NaN or exactly aligned spins ->
        S32 = 0, Smi2 = S0^2, Spl2 = Smi2 + 1e-9) and fabs enforcement.
        Source: IMRPhenomX_Return_Roots_MSA, precession.c:2844-2957.
        """
        B, C, D = cls._msa_spin_evolution_coefficients(l_norm, j_norm, prec)
        B2 = B * B
        B3 = B2 * B
        BC = B * C
        p = C - B2 / 3
        qc = (2.0 / 27.0) * B3 - BC / 3.0 + D
        sqrtarg = torch.sqrt(-p / 3.0)
        acosarg = torch.clamp(1.5 * qc / p / sqrtarg, -1.0, 1.0)
        theta = torch.acos(acosarg) / 3.0
        cos_theta = torch.cos(theta)
        two_pi_2_3 = 2.0 * 2.0 * math.pi / 3.0
        pi_2_3 = 2.0 * math.pi / 3.0
        tmp1 = 2.0 * sqrtarg * torch.cos(theta - two_pi_2_3) - B / 3.0
        tmp2 = 2.0 * sqrtarg * torch.cos(theta - pi_2_3) - B / 3.0
        tmp3 = 2.0 * sqrtarg * cos_theta - B / 3.0
        tmp4 = torch.maximum(torch.maximum(tmp1, tmp2), tmp3)
        tmp5 = torch.minimum(torch.minimum(tmp1, tmp2), tmp3)
        # remaining root: the one that is neither max nor min
        m3 = ((tmp4 - tmp3) > 0.0) & ((tmp5 - tmp3) < 0.0)
        m1_ = ((tmp4 - tmp1) > 0.0) & ((tmp5 - tmp1) < 0.0)
        tmp6 = torch.where(m3, tmp3, torch.where(m1_, tmp1, tmp2))
        Spl2 = torch.abs(tmp4)
        S32 = tmp5
        Smi2 = torch.abs(tmp6)
        # Degenerate guard (precession.c:2899-2913)
        degen = (torch.isnan(theta) | torch.isnan(sqrtarg)
                 | (prec["dotS1Ln"] == 1) | (prec["dotS2Ln"] == 1)
                 | (prec["dotS1Ln"] == -1) | (prec["dotS2Ln"] == -1)
                 | (prec["S1_norm_2"] == 0) | (prec["S2_norm_2"] == 0))
        S32 = torch.where(degen, torch.zeros_like(S32), S32)
        Smi2 = torch.where(degen, prec["S_0_norm_2"], Smi2)
        Spl2 = torch.where(degen, prec["S_0_norm_2"] + 1e-9, Spl2)
        return S32, Smi2, Spl2

    @staticmethod
    def _msa_psiofv(v, v2, psi0, psi1, psi2, prec):
        """
        psi(v) = psi0 - 0.75 g0 delta_qq (1 + psi1 v + psi2 v^2)/v^3
        (Eq. 51).  Source: IMRPhenomX_psiofv, precession.c:2831-2835.
        """
        return psi0 - 0.75 * prec["g0"] * prec["delta_qq"] \
            * (1.0 + psi1 * v + psi2 * v2) / (v2 * v)

    @classmethod
    def _msa_s_norm(cls, v, roots, prec):
        """
        S(v) = sqrt(Spl2 + (Smi2 - Spl2) sn^2(psi | m)) (Eqs. 23-25),
        with sn = 0 when |Smi2 - Spl2| < 1e-5.
        Source: IMRPhenomX_Return_SNorm_MSA, precession.c:2971-3003.
        """
        S32, Smi2, Spl2 = roots
        v2 = v * v
        m = (Smi2 - Spl2) / (S32 - Spl2)
        psi = cls._msa_psiofv(v, v2, prec["psi0"], prec["psi1"], prec["psi2"], prec)
        sn = cls._elljac_sn(psi, m)
        sn = torch.where(torch.abs(Smi2 - Spl2) < 1.0e-5, torch.zeros_like(sn), sn)
        SNorm2 = Spl2 + (Smi2 - Spl2) * sn * sn
        return torch.sqrt(SNorm2)

    @staticmethod
    def _msa_phiz(v, j_norm, prec):
        """
        phiz(v), Eq. 66, with the phiz_n coefficients of Eqs. D22-D27;
        NaN -> 0.  Source: IMRPhenomX_Return_phiz_MSA,
        precession.c:3181-3240.
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
                                             - SAv2 * eta2 / 3.0 - eta2 * eta2 * invv2 / 3.0) \
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
        return torch.where(torch.isnan(phiz_out), torch.zeros_like(phiz_out), phiz_out)

    @staticmethod
    def _msa_zeta(v, prec):
        """
        zeta(v), Eq. F5 (log(v) term as per
        LALSimInspiralFDPrecAngles_internals.c); NaN -> 0.
        Source: IMRPhenomX_Return_zeta_MSA, precession.c:3245-3277.
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
        return torch.where(torch.isnan(zeta_out), torch.zeros_like(zeta_out), zeta_out)

    @staticmethod
    def _msa_constants_c(v, j_norm, prec):
        """
        Eqs. B6-B8, version != 220 branch (default 223).
        Source: IMRPhenomX_Return_Constants_c_MSA, precession.c:3066-3091.
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

    @staticmethod
    def _msa_constants_d(l_norm, j_norm, prec):
        """
        Eqs. B9-B11.  Source: IMRPhenomX_Return_Constants_d_MSA,
        precession.c:3118-3132.
        """
        LNorm2 = l_norm * l_norm
        JNorm2 = j_norm * j_norm
        d0 = -(JNorm2 - (l_norm + prec["Spl"]) * (l_norm + prec["Spl"])) \
            * (JNorm2 - (l_norm - prec["Spl"]) * (l_norm - prec["Spl"]))
        d2 = -2.0 * (prec["Spl2"] - prec["Smi2"]) * (JNorm2 + LNorm2 - prec["Spl2"])
        d4 = -(prec["Spl2"] - prec["Smi2"]) * (prec["Spl2"] - prec["Smi2"])
        return d0, d2, d4

    @classmethod
    def _msa_psi(cls, v, v2, prec):
        """Source: IMRPhenomX_Return_Psi_MSA, precession.c:3155-3158."""
        return -0.75 * prec["g0"] * prec["delta_qq"] \
            * (1.0 + prec["psi1"] * v + prec["psi2"] * v2) / (v2 * v)

    @staticmethod
    def _msa_psi_dot(v, prec):
        """Source: IMRPhenomX_Return_Psi_dot_MSA, precession.c:3163-3175."""
        v2 = v * v
        A_coeff = -1.5 * (v2 * v2 * v2) * (1.0 - v * prec["Seff"]) * prec["sqrt_inveta"]
        return 0.5 * A_coeff * torch.sqrt(prec["Spl2"] - prec["S32"])

    @classmethod
    def _msa_corrections(cls, v, l_norm, j_norm, prec):
        """
        First-order MSA corrections (phiz_MSA, zeta_MSA): the 222/223
        FDPrecAngles fabs forms of the Cphi/Dphi terms and the
        222/223/224 zeta variant; NaN -> 0.
        Source: IMRPhenomX_Return_MSA_Corrections_MSA,
        precession.c:3283-3425.
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
        # As implemented in LALSimInspiralFDPrecAngles.c (222/223 branch)
        Cphi_term = torch.abs((c4 * d0 * ((2 * d0 + d2) + sd)
                               - c2 * d0 * ((d2 + 2. * d4) - sd)
                               - c0 * ((2 * d0 * d4) - (d2 + d4) * (d2 - sd))) / C2den) \
            * (sqrt_nc / (nc - 1.) * (atan_psi - torch.atan(sqrt_nc * tan_psi))) / psi_dot
        Dphi_term = torch.abs(-c4 * d0 * ((2 * d0 + d2) - sd)
                              + c2 * d0 * ((d2 + 2. * d4) + sd)
                              - c0 * (-(2 * d0 * d4) + (d2 + d4) * (d2 + sd))) / C2den \
            * (sqrt_nd / (nd - 1.) * (atan_psi - torch.atan(sqrt_nd * tan_psi))) / psi_dot
        # Limits implied by Eq. 67
        Cphi_term = torch.where(nc == 1.0, torch.zeros_like(Cphi_term), Cphi_term)
        Dphi_term = torch.where(nd == 1.0, torch.zeros_like(Dphi_term), Dphi_term)
        corr_x = Cphi_term + Dphi_term
        corr_y = A_theta_L * corr_x + 2. * B_theta_L * d0 \
            * (Cphi_term / (sd - d2) - Dphi_term / (sd + d2))
        corr_x = torch.where(torch.isnan(corr_x), torch.zeros_like(corr_x), corr_x)
        corr_y = torch.where(torch.isnan(corr_y), torch.zeros_like(corr_y), corr_y)
        return corr_x, corr_y

    @staticmethod
    def _msa_costhetaLJ(l_norm, j_norm, s_norm):
        """
        cos(thetaLJ) = 0.5 (J^2 + L^2 - S^2)/(L J), clamped to [-1, 1].
        Source: IMRPhenomX_costhetaLJ, precession.c:3139-3148.
        """
        c = 0.5 * (j_norm * j_norm + l_norm * l_norm - s_norm * s_norm) \
            / (l_norm * j_norm)
        return torch.clamp(c, -1.0, 1.0)

    def msa_phi_zeta_costhetaL(self, v, derived, prec):
        """
        Per-frequency MSA angle triple (phiz, zeta, cos_thetaL) over the
        (B, F) grid: L_norm = eta/v, JNorm, 3PN L (223 branch), roots,
        SNorm, corrections when |Smi2 - Spl2| > 1e-5, phiz/zeta/costhetaLJ.
        Made stateless: the roots at v are computed locally, never stored
        (the C mutates the struct per call).
        Source: IMRPhenomX_Return_phi_zeta_costhetaL_MSA,
        precession.c:1961-2026.
        """
        eta = prec["eta"]
        L_norm = eta / v
        J_norm = self._msa_j_norm(L_norm, prec)
        L_norm3PN = self._l_norm_3pn_of_v(v, v * v, L_norm, prec)
        J_norm3PN = self._msa_j_norm(L_norm3PN, prec)
        roots = self._msa_roots(L_norm, J_norm, prec)
        S32, Smi2, Spl2 = roots
        # the per-frequency roots update the struct state in C; the
        # correction and SNorm functions consume them
        loc = dict(prec)
        loc["S32"] = S32
        loc["Smi2"] = Smi2
        loc["Spl2"] = Spl2
        loc["Spl"] = torch.sqrt(Spl2)
        loc["Spl2mSmi2"] = Spl2 - Smi2
        SNorm = self._msa_s_norm(v, roots, loc)
        corr_x, corr_y = self._msa_corrections(v, L_norm, J_norm, loc)
        big = torch.abs(Smi2 - Spl2) > 1.0e-5
        corr_x = torch.where(big, corr_x, torch.zeros_like(corr_x))
        corr_y = torch.where(big, corr_y, torch.zeros_like(corr_y))
        phiz = self._msa_phiz(v, J_norm, loc)
        zeta = self._msa_zeta(v, loc)
        cos_theta_L = self._msa_costhetaLJ(L_norm3PN, J_norm3PN, SNorm)
        return phiz + corr_x, zeta + corr_y, cos_theta_L

    # ------------------------------------------------------------------
    # NNLO angle model (fallback; version 102)
    # ------------------------------------------------------------------

    def _nnlo_angle_coeffs(self, derived, prec):
        """
        NNLO single-spin PN Euler angle coefficients as per IMRPhenomPv2,
        from chiL = (1 + q)(chi_eff/q) and chi_p; written into prec as
        alpha1..alpha5, alpha4L, epsilon1..epsilon5, epsilon4L.
        Source: precession.c:862-925 (the cached in-struct convention;
        the external XLALSimIMRPhenomXPNEuler* wrappers carry an
        opposite-delta convention and are not used by the waveform path).
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
        m1_8 = m1_6 * m1_2
        delta = derived[:, 4:5]
        delta2 = delta * delta
        delta3 = delta2 * delta
        # chi_eff = Xa chi1z + Xb chi2z from the precession-frame spins
        # (the XAS derived tensor reads aligned spins from the 8-column
        # layout and must not be used here)
        chi_eff = prec["m1"] * prec["chi1z"] + prec["m2"] * prec["chi2z"]
        chip = prec["chi_p"]
        chip2 = chip * chip
        q = prec["m1"] / prec["m2"]   # q = m1/m2 > 1
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

    @staticmethod
    def _pn_euler_alpha_nnlo(prec, omega, omega_cbrt, omega_cbrt2, logomega):
        """
        alpha = alpha1/omega + alpha2/omega^{2/3} + alpha3/omega^{1/3}
              + alpha4L log(omega) + alpha5 omega^{1/3} - alpha_offset.
        Note that alpha_offset already includes the term alpha0.
        Source: IMRPhenomX_PN_Euler_alpha_NNLO, precession.c:1663-1684.
        """
        return (prec["alpha1"] / omega
                + prec["alpha2"] / omega_cbrt2
                + prec["alpha3"] / omega_cbrt
                + prec["alpha4L"] * logomega
                + prec["alpha5"] * omega_cbrt
                - prec["alpha_offset"])

    @staticmethod
    def _pn_euler_epsilon_nnlo(prec, omega, omega_cbrt, omega_cbrt2, logomega):
        """
        Same series with the epsilon coefficients.
        Source: IMRPhenomX_PN_Euler_epsilon_NNLO, precession.c:1687-1708.
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

    @staticmethod
    def _wignerd_coefficients_cosbeta(cos_beta):
        """
        cos(beta/2) = sqrt(|1 + cos beta| / 2),
        sin(beta/2) = sqrt(|1 - cos beta| / 2)  (both non-negative).
        Source: IMRPhenomXWignerdCoefficients_cosbeta,
        precession.c:1870-1881.

        The square roots are evaluated through safe_sqrt: at |cos beta|
        = 1 (aligned-spin limit) the argument vanishes and the sqrt
        derivative diverges, which would put NaN in every gradient.
        Forward values are unchanged.
        """
        cBetah = safe_sqrt(torch.abs(1.0 + cos_beta) / 2.0)
        sBetah = safe_sqrt(torch.abs(1.0 - cos_beta) / 2.0)
        return cBetah, sBetah

    def _nnlo_cosbeta(self, v, derived, prec):
        """
        NNLO cos(beta) from L(v), SL and Sperp:
        s = Sperp/(L + SL); cos beta = copysign(1, L + SL)/sqrt(1 + s^2).
        Source: IMRPhenomXWignerdCoefficients, precession.c:1883-1906.
        """
        L = self._lpn_ansatz(v, prec["eta"] / v, prec["L_coeffs"])
        s = prec["Sperp"] / (L + prec["SL"])
        s2 = s * s
        sgn = torch.where(L + prec["SL"] >= 0,
                          torch.ones_like(s), -torch.ones_like(s))
        return sgn / torch.sqrt(1.0 + s2)

    def _wignerd_coefficients(self, v, derived, prec):
        """
        NNLO cos(beta) followed by the half-angle map.
        Source: IMRPhenomXWignerdCoefficients, precession.c:1883-1906.
        """
        return self._wignerd_coefficients_cosbeta(self._nnlo_cosbeta(v, derived, prec))

    # ------------------------------------------------------------------
    # Euler angles over the frequency grid
    # ------------------------------------------------------------------

    def get_alpha_epsilon_cosbeta(self, Mf, mprime, derived, prec):
        """
        (alpha, epsilon, cos_beta) over the (B, F) grid for co-precessing
        mode index m':

        omega = pi Mf 2/m' ; v = cbrt(omega)
        MSA (default): (phiz, zeta, costhetaL) - offsets;
        NNLO (fallback samples): closed-form series + cos(beta) from L.
        The two are blended per sample with torch.where on the MSA_ERROR
        mask.  Offsets are the mprime = 2 values for every m' (v6.1.0
        behaviour; precession.c:993-1011, XPHM.c:3585-3625).
        Source: IMRPhenomXPHMTwistUp angle block,
        LALSimIMRPhenomXPHM.c:1829-1972; Get_alpha_beta_epsilon,
        LALSimIMRPhenomXPHM.c:3632-3676.

        Returns
        -------
        (alpha, epsilon, cBetah, sBetah) : (B, F) tensors
        """
        omega = math.pi * Mf * (2.0 / mprime)
        version = prec["version"]

        def nnlo_angles():
            # the series helpers already subtract alpha/epsilon_offset
            logomega = torch.log(omega)
            omega_cbrt = torch.pow(omega, 1.0 / 3.0)
            omega_cbrt2 = omega_cbrt * omega_cbrt
            alpha = self._pn_euler_alpha_nnlo(prec, omega, omega_cbrt,
                                              omega_cbrt2, logomega)
            epsilon = self._pn_euler_epsilon_nnlo(prec, omega, omega_cbrt,
                                                  omega_cbrt2, logomega)
            cosbeta = self._nnlo_cosbeta(omega_cbrt, derived, prec)
            return alpha, epsilon, cosbeta

        if version in (220, 221, 222, 223, 224):
            v = torch.pow(omega, 1.0 / 3.0)
            phiz, zeta, ctl = self.msa_phi_zeta_costhetaL(v, derived, prec)
            alpha = phiz - prec["alpha_offset"]
            epsilon = zeta - prec["epsilon_offset"]
            cosbeta = ctl
            if bool(prec["MSA_ERROR"].any()):
                mask = prec["MSA_ERROR"]
                a_n, e_n, c_n = nnlo_angles()
                alpha = torch.where(mask, a_n, alpha)
                epsilon = torch.where(mask, e_n, epsilon)
                cosbeta = torch.where(mask, c_n, cosbeta)
        else:
            alpha, epsilon, cosbeta = nnlo_angles()

        cBetah, sBetah = self._wignerd_coefficients_cosbeta(cosbeta)
        return alpha, epsilon, cBetah, sBetah

    # ------------------------------------------------------------------
    # Wigner-d rows
    # Source: IMRPhenomXPHMTwistUp / IMRPhenomXPHMTwistUpOneMode tables,
    #         LALSimIMRPhenomXPHM.c:1996-2183 / 3305-3395 (appendix A of
    #         arXiv:2004.06503)
    # ------------------------------------------------------------------

    @staticmethod
    def _wigner_d_rows(cBetah, sBetah, ell, mprime):
        """
        Rows d^l_{m,m'} and d^l_{m,-m'} for m = -l..l as lists of (B, F)
        tensors, transcribed verbatim from the C tables; the -m' row uses
        the symmetry d^l_{-m,-m'} = (-1)^(l+m') d^l_{m,m'} noted per
        block (eq. A2 of the Precessing paper).
        """
        cBetah2 = cBetah * cBetah
        cBetah3 = cBetah * cBetah2
        cBetah4 = cBetah * cBetah3
        sBetah2 = sBetah * sBetah
        sBetah3 = sBetah * sBetah2
        sBetah4 = sBetah * sBetah3
        sqrt2 = math.sqrt(2.0)
        sqrt5 = math.sqrt(5.0)
        sqrt6 = math.sqrt(6.0)
        sqrt7 = math.sqrt(7.0)
        sqrt10 = math.sqrt(10.0)
        sqrt14 = math.sqrt(14.0)
        sqrt15 = math.sqrt(15.0)
        sqrt30 = math.sqrt(30.0)
        sqrt70 = math.sqrt(70.0)

        if (ell, mprime) == (2, 2):
            d = [sBetah4, 2.0 * cBetah * sBetah3, sqrt6 * sBetah2 * cBetah2,
                 2.0 * cBetah3 * sBetah, cBetah4]
            dm = [d[4], -d[3], d[2], -d[1], d[0]]
        elif (ell, mprime) == (2, 1):
            d = [2.0 * cBetah * sBetah3, 3.0 * sBetah2 * cBetah2 - sBetah4,
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
    # Source: IMRPhenomXPHMTwistUp, LALSimIMRPhenomXPHM.c:1806-2203
    # ------------------------------------------------------------------

    def twist_up(self, Mf, hlm_coprec, derived, prec, ell, mprime):
        """
        Twist one co-precessing mode h_{l,-m'} into inertial-frame
        (hp, hc) contributions over the (B, F) grid (eqs. 3.5-3.7 /
        E18-E19 of arXiv:2004.06503):

        - Wigner-d rows d^l_{m,m'} and d^l_{m,-m'} from cBetah/sBetah
          powers, written out literally per (l, m') as in the C tables
          (XPHM.c:1996-2183);
        - transfer sum over inertial m in [-l, l]:
          A     = e^{-i m alpha} d^l_{m,-m'} Y_{lm}
          Astar = e^{+i m alpha} d^l_{m,m'}  conj(Y_{lm})
          hp_sum += A ± Astar ; hc_sum += i (A ∓ Astar)
          with the (-1)^l sign pattern (+- for l = 2, 4; -+ for l = 3);
        - eps_phase = e^{-i m' epsilon} hlm_coprec / 2;
          hp += eps_phase * hp_sum ; hc += eps_phase * hc_sum.

        Vectorized: no loop over frequency, the m-sum unrolled literally
        (IMRPhenomPv2.PhenomPCoreTwistUp style).  PolarizationSymmetry is
        1 on the default path (the -1 branch belongs to the PNR
        antisymmetric waveform).
        """
        alpha, epsilon, cBetah, sBetah = self.get_alpha_epsilon_cosbeta(
            Mf, mprime, derived, prec)
        d, dm = self._wigner_d_rows(cBetah, sBetah, ell, mprime)

        cexp_i_alpha = torch.polar(torch.ones_like(alpha), alpha)
        Y = [prec[f"Y{ell}{'m' if m < 0 else ''}{abs(m)}"]
             for m in range(-ell, ell + 1)]

        # e^{i m alpha} for m = -l..l via the same multiplication chain
        # as C (negative powers from the reciprocal, XPHM.c:2004-2008)
        pows = [None] * (2 * ell + 1)
        pows[ell] = torch.ones_like(cexp_i_alpha)
        pows[ell + 1] = cexp_i_alpha
        for k in range(2, ell + 1):
            pows[ell + k] = pows[ell + k - 1] * cexp_i_alpha
        cexp_mi_alpha = 1.0 / cexp_i_alpha
        pows[ell - 1] = cexp_mi_alpha
        for k in range(2, ell + 1):
            pows[ell - k] = pows[ell - k + 1] * cexp_mi_alpha

        # l = 3 flips the relative sign of the conjugated transfer term
        # (XPHM.c:2085-2090, 2119-2124)
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
    # J-frame single mode (verification hook)
    # Source: IMRPhenomXPHM_OneMode / IMRPhenomXPHMTwistUpOneMode,
    #         LALSimIMRPhenomXPHM.c:2797/3142
    # ------------------------------------------------------------------

    def _twist_up_one_mode(self, Mf, hlm_coprec, derived, prec, ell, mprime, m):
        """
        Contribution of co-precessing mode (l, m') to the inertial
        J-frame mode (l, m), positive and negative frequency series
        (eqs. E3-E4 of arXiv:2004.06503):

        hlm_pos = e^{-i m' epsilon} hlm_coprec  e^{-i m alpha} d^l_{m,-m'}
        hlm_neg = e^{+i m' epsilon} (-1)^l conj(hlm_coprec)
                  e^{-i m alpha} d^l_{m,m'}

        Source: IMRPhenomXPHMTwistUpOneMode, LALSimIMRPhenomXPHM.c:3142-3400
        (antisymmetric contribution zero on the default path).
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

    # Co-precessing modes per l, positive m' (IMRPhenomXPHM default
    # ModeArray; XPHM.c:2920 loops m' = 1..l over the active modes)
    COPREC_MODES = {2: (2, 1), 3: (3, 2), 4: (4,)}

    def get_hlm_jframe(self, theta, ell, emm):
        """
        Single precessing (l, m) mode in the inertial J-frame, returned as
        the (hlm_pos, hlm_neg) frequency series pair (precessing modes
        have support on both frequency signs).  Mirrors
        XLALSimIMRPhenomXPHMOneMode for direct per-mode verification
        against lalsimulation.SimIMRPhenomXPHMOneMode.

        The co-precessing modes are generated with the aligned-spin
        8-column theta [m1, m2, chi1z, chi2z, dist, 0, 0, 0] (pWF->phi0
        is 0 under convention 1; tc and inclination do not enter single
        modes) and the precessing final spin threaded into every
        coefficient builder.
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
    # Source: XLALSimIMRPhenomXPHM -> IMRPhenomXPHM_hplushcross,
    #         LALSimIMRPhenomXPHM.c:167/740 (no-multibanding branch)
    # ------------------------------------------------------------------

    def get_hphc(self, theta, reproduce_lal=False):
        """
        Compute FD plus and cross polarisations for a precessing BBH
        parameter batch.

        Steps (per-frequency twist branch, MBandPrecVersion = 0):

        1. mass/spin ordering (check_masses_and_spins);
        2. aligned-spin derived parameters from (m1, m2, chi1z, chi2z);
        3. precession dict (get_and_set_precession_variables) — replaces
           fRING/fDAMP with the precessing-final-spin values consumed by
           every co-precessing mode;
        4. loop over positive m' co-precessing modes (2,2), (2,1), (3,3),
           (3,2), (4,4): h_{l,-m'} via the inherited XHM/XAS machinery,
           skipping Ampzero modes; the co-precessing 22 uses phi0 = 0
           (convention 1) with the reference phase entering through the
           angle offsets;
        5. twist_up accumulation into hp/hc;
        6. polarization rotation by 2 zeta_polarization:
           hp' = cos(2 zeta) hp + sin(2 zeta) hc,
           hc' = cos(2 zeta) hc - sin(2 zeta) hp
           (XPHM.c:1591-1607);
        7. tail as in IMRPhenomXAS.get_hphc: FD taper, tc shift, df
           normalisation (skipped when reproduce_lal), DC zero-padding.

        Parameters
        ----------
        theta : torch.Tensor, shape (B, 17)
            Parameters in the waveform_utils.WAVEFORM_PARAMETERS layout.
        reproduce_lal : bool
            If True, compare against LAL called with
            PhenomXHMThresholdMband = 0 and PhenomXPHMThresholdMband = 0.

        Returns
        -------
        hp, hc : torch.Tensor, shape (B, n_pad + F), complex
        """
        # [m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, distance, tc,
        #  phic, inclination]
        theta = waveform_utils.fill_defaults(theta)[:, :12]
        theta = self.check_masses_and_spins(theta)
        tc = theta[:, 9:10]
        theta_aligned = torch.cat(
            [theta[:, 0:2], theta[:, 4:5], theta[:, 7:8], theta[:, 8:9],
             torch.zeros_like(theta[:, 0:3])], dim=1)
        derived = self.compute_derived_parameters(theta_aligned)
        M_s = derived[:, 2:3]
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
                                       afinal=afinal)
                hp_c, hc_c = self.twist_up(Mf, htildelm, derived, prec,
                                           ell, mprime)
                hp = hp + hp_c
                hc = hc + hc_c

        # Rotate the polarizations by 2 zeta (eq. C26 of arXiv:2004.06503;
        # XPHM.c:1586-1607)
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
            hp, hc = self.apply_tc(hp, hc, tc)
            hp = hp * self.df
            hc = hc * self.df

        return self.pad_missing_frequencies(hp, hc)
