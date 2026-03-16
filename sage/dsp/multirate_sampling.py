#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Mar 29 23:41:28 2022

__author__      = nnarenraju
__copyright__   = Copyright 2022, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

# Modules
import math
import torch
import numpy as np
import torch.nn.functional as F

from typing import List, Tuple, Iterable

# LAL and PyCBC
import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import lalsimulation as lalsim

from pycbc.conversions import tau_from_final_mass_spin, get_final_from_initial
from pycbc.detector import Detector
from pycbc import pnutils

# LOCAL
from sage.core.config import get_cfg, get_data_cfg


class MultirateSampler(torch.nn.Module):
    """
    Multi-rate decimator for batched GW time-domain data.

    Input shape: (B, D, L)
    Output shape: (B, D, L_compressed)

    B = batch
    D = detectors
    L = original sequence length
    """

    GRAPH_READY = True

    def __init__(
        self,
        binning_method,
        reflect_pad: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup configs
        self.cfg = get_cfg()
        data_cfg = get_data_cfg()

        # Get bins and fs from binning method
        self.bins = binning_method.detailed_bins
        self.num_bins = len(self.bins)

        self.register_buffer(
            "bins_tensor",
            torch.tensor(self.bins.tolist(), dtype=torch.int64),
            persistent=False,
        )

        self.original_fs = data_cfg.sample_rate
        self.min_fs = np.min(self.bins[:, 2])

        max_dec_factor = int(self.original_fs // self.min_fs)
        max_power = max_dec_factor.bit_length() - 1
        self.max_power = max_power

        # Precompute decimation power per bin
        dec_factors = self.original_fs // self.bins_tensor[:, 2]
        dec_powers = torch.log2(dec_factors.float()).long()
        self.register_buffer("dec_powers", dec_powers, persistent=False)

        # Precompute FIR kernels for powers of 2
        self._build_kernels()
        kernel_len = self.kernel.shape[-1]

        # Reflect Padding
        # Largest FIR kernel size if 2 * dec factor + 1
        # if max factor = 64 -> kernel size = 129 -> half = 64
        # So using 128 to 256 is very conservative
        # NOTE: Adjust if needed (and remove conservative assertion)
        if reflect_pad is not None:
            self.pad = reflect_pad
            assert self.pad >= 2 * max_dec_factor
            assert self.pad >= kernel_len
        else:
            kernel_len = self.kernel.shape[-1]

            group_delay = (kernel_len - 1) // 2

            self.pad = max(
                4 * kernel_len,
                2 * max_dec_factor,
                2 * group_delay * max_dec_factor,
            )

        # Get sorted decimation factors and remember original order
        self.power_to_bins = self.order_decimations(dec_powers)

        # Sanity check
        # TODO: This isn't compile friendly; we can include this outside
        # factors = [2**i for i in range(1, self.max_power + 1)]
        # assert all(factors > self.original_fs // self.min_fs)

    ## Decimation ordering ##
    def order_decimations(self, dec_powers):
        # Unique powers sorted (for sequential decimation reuse)
        unique_powers = sorted(set(dec_powers.tolist()))
        self.powers = unique_powers
        self.max_required_power = max(unique_powers)

        # Map power -> list of bins
        power_to_bins = {p: [] for p in unique_powers}

        for i in range(self.num_bins):

            start = int(self.bins[i, 0] + self.pad)
            end = int(self.bins[i, 1] + self.pad)
            power = int(dec_powers[i].item())

            div = 1 << power

            sidx = start // div
            eidx = end // div

            power_to_bins[power].append((sidx, eidx, i))

        return power_to_bins

    ## FIR kernel construction ##
    def _build_legacy_kernels(self):
        factor = 2
        kernel_size = 2 * factor + 1
        t = torch.arange(-factor, factor + 1, dtype=torch.float32)

        h = torch.sinc(t / factor)
        h = h * torch.hamming_window(kernel_size, periodic=False)
        h = h / h.sum()

        self.register_buffer(
            "kernel",
            h.view(1, 1, -1).to(self.cfg.device, dtype=self.cfg.dtype),
            persistent=False,
        )

    def _build_kernels(self):

        # choose taps based on safety target
        # GOOD SAFE VALUES:
        # 31 taps → ~70 dB
        # 47 taps → ~85 dB
        # 63 taps → ~95 dB
        NTAPS = 63

        M = NTAPS - 1
        n = torch.arange(NTAPS, dtype=torch.float32)

        # ideal halfband impulse response
        h = torch.sinc((n - M / 2) / 2)

        # window (Kaiser safer than Hamming)
        beta = 9  # A ~ 8.7 + 9.08 * beta ~ 90.4 dB attenuation
        h *= torch.kaiser_window(NTAPS, periodic=False, beta=beta)

        # normalize DC gain
        # h /= h.sum() --> This does not preserve L2 energy
        h /= torch.sqrt((h**2).sum())

        self.register_buffer(
            "kernel",
            h.view(1, 1, -1).to(self.cfg.device, dtype=self.cfg.dtype),
            persistent=False,
        )

    ## Single stage decimation ##
    def _decimate_once(self, x: torch.Tensor) -> torch.Tensor:

        pad = self.kernel.shape[-1] // 2
        x = F.pad(x, (pad, pad), mode="reflect")

        B, D, L = x.shape
        x = x.reshape(B * D, 1, L)

        y = F.conv1d(x, self.kernel, stride=2)

        return y.reshape(B, D, y.shape[-1])

    ## Forward ##
    def forward(self, noisy_signals: torch.Tensor) -> torch.Tensor:

        x = F.pad(
            noisy_signals,
            (self.pad, self.pad),
            mode="reflect",
        )

        outputs = [None] * self.num_bins

        current = x
        current_power = 0

        for target_power in self.powers:

            # Incrementally decimate
            steps = target_power - current_power

            for _ in range(steps):
                current = self._decimate_once(current)

            current_power = target_power

            # Extract bins belonging to this power
            for sidx, eidx, original_idx in self.power_to_bins[target_power]:
                chunk = current[:, :, sidx:eidx]
                outputs[original_idx] = chunk

        return torch.cat(outputs, dim=-1)


class DyadicPyramidBinning:
    MTSUN_SI = 4.92549102554e-06
    MSUN_SI = 1.989e30

    def __init__(
        self,
        param_bounds,
        lowest_allowed_fs=64.0,
        safe_nyquist_gap=20.0,
        min_bin_duration=0.05,
        verify_nyquist=False,
    ):
        # Setup configs
        data_cfg = get_data_cfg()

        self.prior_low_mass, self.prior_high_mass = param_bounds["mass1"]
        self.signal_low_freq_cutoff = data_cfg.signal_low_frequency_cutoff
        self.sample_rate = data_cfg.sample_rate
        self.signal_length = data_cfg.sample_length_in_s
        self.lowest_allowed_fs = lowest_allowed_fs
        self.tc_inject_lower, self.tc_inject_upper = param_bounds["tc"]
        self.safe_nyquist_gap = safe_nyquist_gap
        self.min_bin_duration = min_bin_duration

        # Precompute tf for lowest mass system
        self.t, self.f = self._get_tf_evolution_before_tc()

        # Physics driven starting frequency
        self.f_isco = self._f_schwarzschild_isco(2.0 * self.prior_low_mass)
        self.fs_anchor = self._next_pow2((2.0 * self.f_isco) + self.safe_nyquist_gap)

        # Pre/Post fudge
        # We don't necessarily need to generalise this to used detector
        # We can pick the two ifos farthest apart for a conservative estimate
        self.light_travel_time = (
            Detector("H1").light_travel_time_to_detector(Detector("V1")) * 1.1
        )
        self.pre_fudge_factor = self._get_pre_fudge_factor()
        self.post_fudge_factor = self._get_post_fudge_factor()

        # Construct bins immediately
        self.detailed_bins = self._construct_multirate_bins()
        # Verify Nyquist for all bins
        # This is an option because sometimes we leave things out deliberately
        if verify_nyquist:
            self.verify_nyquist_condition()

    def _velocity_to_frequency(self, v, M):
        return v**3 / (M * self.MTSUN_SI * np.pi)

    def _f_schwarzschild_isco(self, M):
        return self._velocity_to_frequency((1.0 / 6.0) ** 0.5, M)

    def _get_freq_at_time(self, search_time):
        idx = (np.abs(self.t - search_time)).argmin()
        return self.f[idx]

    def _get_time_at_freq(self, search_freq):
        idx = (np.abs(self.f - search_freq)).argmin()
        return self.t[idx]

    def _get_tf_evolution_before_tc(self):
        # Get npoints from tau
        npoints = (
            self.get_imr_chirp_time(
                self.prior_low_mass,
                self.prior_low_mass,
                0.99,
                0.99,
                self.signal_low_freq_cutoff,
            )
            * self.sample_rate
        )
        # Get tf of given waveform
        t, f = pnutils.get_inspiral_tf(
            tc=0.0,
            mass1=self.prior_low_mass,
            mass2=self.prior_low_mass,
            spin1=0.99,
            spin2=0.99,
            f_low=self.signal_low_freq_cutoff,
            n_points=int(npoints),
            pn_2order=7,
            approximant="IMRPhenomD",
        )
        return (t, f)

    def _get_pre_fudge_factor(self):
        # Calculate fudge factor at left end of the waveform injection
        # Get t, f from lowest mass binary system
        # The times should vary from 0.0 to -tau starting at tc
        time_at_decim_start_freq = self._get_time_at_freq(search_freq=self.f_isco)
        pre_fudge_factor = (
            self.light_travel_time + time_at_decim_start_freq
        ) * 1.1  # just in case
        return pre_fudge_factor

    def _get_post_fudge_factor(self):
        # Get fudge factor that accounts for wrap around from PyCBC
        # This can be used to estimate the merger+ringdown leeway for MR sampling
        # This should account for waveform content after tc
        m_final, spin_final = get_final_from_initial(
            mass1=self.prior_high_mass,
            mass2=self.prior_high_mass,
            spin1z=0.99,
            spin2z=0.99,
        )
        post_fudge_factor = (
            tau_from_final_mass_spin(m_final, spin_final) * 10 * 1.5
        )  # just in case
        # Adding light travel time between detectors H1 and V1 (We use H1 and L1, but just in case)
        light_travel_time = (
            Detector("H1").light_travel_time_to_detector(Detector("V1")) * 1.1
        )
        post_fudge_factor += light_travel_time
        return post_fudge_factor

    def _next_pow2(self, x):
        return 1 << int(np.ceil(np.log2(x)))

    def get_imr_chirp_time(self, m1, m2, s1z, s2z, fl):
        return 1.1 * lalsim.SimIMRPhenomDChirpTime(
            m1 * 1.989e30, m2 * 1.989e30, s1z, s2z, fl
        )

    def plot_multirate_tf(self):

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.cm as cm
        import numpy as np

        fontsize = 16

        fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=300)

        # Plot prior tf manifold (optional but useful sanity check)
        for m1 in np.linspace(self.prior_low_mass + 0.1, self.prior_high_mass, 256):

            m2 = m1 - 0.1

            t, f = pnutils.get_inspiral_tf(
                tc=0.0,
                mass1=m1,
                mass2=m2,
                spin1=0.99,
                spin2=0.99,
                f_low=self.signal_low_freq_cutoff,
                n_points=512,
                pn_2order=7,
                approximant="IMRPhenomD",
            )

            ax.plot(t, f, linewidth=0.6, alpha=0.2, c="gray")

        # Longest waveform used for bin construction
        ax.plot(self.t, self.f, linestyle="dashed", linewidth=2.0, color="k")

        # tc placement inside segment
        tc = (self.tc_inject_upper + self.tc_inject_lower) / 2.0 + 0.05

        # Build Nyquist curve exactly like your example
        x = []
        y = []

        for foo in self.detailed_bins:
            x.extend([foo[0] / self.sample_rate - tc, foo[1] / self.sample_rate - tc])
            y.extend([foo[2] / 2.0, foo[2] / 2.0])

        ax.plot(
            x, y, linestyle="dashed", c=np.array([191, 44, 35]) / 255.0, linewidth=2.0
        )

        # Padding
        lpad = self.tc_inject_lower - self.get_imr_chirp_time(
            self.prior_low_mass,
            self.prior_low_mass,
            0.99,
            0.99,
            self.signal_low_freq_cutoff,
        )

        rpad = self.signal_length - (self.tc_inject_upper + self.post_fudge_factor)

        # Secondary x-axis
        x_to_altx = lambda x: x + tc
        altx_to_x = lambda altx: altx - tc

        secax = ax.secondary_xaxis("top", functions=(x_to_altx, altx_to_x))
        secax.set_xlabel("Time [seconds]", fontsize=fontsize)

        ax.set_ylabel("Frequency [Hertz]", fontsize=fontsize)
        ax.set_xlabel(
            "Relative Time (tc = 0.0) [seconds]", labelpad=7.5, fontsize=fontsize
        )
        ax.set_yscale("log")

        ax.set_xlim(
            -self.get_imr_chirp_time(
                self.prior_low_mass,
                self.prior_low_mass,
                0.99,
                0.99,
                self.signal_low_freq_cutoff,
            )
            - lpad,
            rpad,
        )

        ax.set_ylim(self.signal_low_freq_cutoff, self.sample_rate)
        plt.tick_params(axis="both", which="major", labelsize=fontsize)
        secax.tick_params(axis="both", which="major", labelsize=fontsize)

        plt.tight_layout()
        plt.show()

    def _construct_multirate_bins(self):

        total_samples = int(self.signal_length * self.sample_rate)
        min_fs = self._next_pow2(self.lowest_allowed_fs)

        ## Construct unchanged region
        start_unchanged = int(
            (self.tc_inject_lower - self.pre_fudge_factor) * self.sample_rate
        )
        end_unchanged = int(
            (self.tc_inject_upper + self.post_fudge_factor) * self.sample_rate
        )

        # Clamp to segment bounds
        start_unchanged = max(0, start_unchanged)
        end_unchanged = min(total_samples, end_unchanged)

        bins = []

        ## Build backward frequency ladder

        # We march backward in time/frequency from f_ISCO
        f_current = self.f_isco

        # Enforce signal low-frequency cutoff
        f_floor = self.signal_low_freq_cutoff

        # Frequency ladder (monotonic decreasing)
        freqs = []

        while f_current > f_floor:
            freqs.append(f_current)
            f_current /= 2.0

        freqs.append(f_floor)

        ## Convert ladder to sample boundaries

        boundaries = []

        for f_k in freqs:

            t_k = self._get_time_at_freq(f_k)

            # Time is negative before merger
            start_idx = int((self.tc_inject_lower + t_k) * self.sample_rate)

            # Clamp to segment
            start_idx = max(0, start_idx)

            # If inside unchanged region, clamp to unchanged start
            if start_idx >= start_unchanged:
                start_idx = start_unchanged

            boundaries.append(start_idx)

            # If we hit segment start, stop
            if start_idx == 0:
                break

        # Ensure strictly decreasing boundaries
        boundaries = np.unique(boundaries)[::-1]

        ## Build bins BEFORE unchanged region
        prev_end = start_unchanged

        for start_idx in boundaries:

            if start_idx >= prev_end:
                continue

            # Convert index -> time relative to merger
            t_latest = (prev_end / self.sample_rate) - self.tc_inject_lower

            # Only consider inspiral region (t <= 0)
            if t_latest > 0:
                f_latest = self.f_isco
            else:
                f_latest = np.interp(t_latest, self.t, self.f)

            required_fs = 2.0 * f_latest + self.safe_nyquist_gap
            fs_k = max(self._next_pow2(required_fs), min_fs)

            bins.append([start_idx, prev_end, int(fs_k)])

            prev_end = start_idx

            if start_idx == 0:
                break

        ## Prepend lowest rate if needed
        if prev_end > 0:
            bins.append([0, prev_end, int(min_fs)])

        ## Insert unchanged region
        bins.append([start_unchanged, end_unchanged, int(self.sample_rate)])

        ## Trailing region -> lowest allowed fs
        if end_unchanged < total_samples:
            bins.append([end_unchanged, total_samples, int(min_fs)])

        ## Sort and merge adjacent identical fs bins
        bins = sorted(bins, key=lambda x: x[0])

        merged = []
        for b in bins:
            if not merged:
                merged.append(b)
                continue

            last = merged[-1]

            # contiguous and same fs → merge
            if last[1] == b[0] and last[2] == b[2]:
                last[1] = b[1]
            else:
                merged.append(b)

        bins = np.array(merged)

        ## Refine: eliminate tiny bins by raising fs
        min_bin_samples = int(self.min_bin_duration * self.sample_rate)

        bins = bins.tolist()

        for i in range(len(bins)):

            start, end, fs = bins[i]
            width = end - start

            if width >= min_bin_samples:
                continue

            # Determine neighboring fs values
            left_fs = bins[i - 1][2] if i > 0 else None
            right_fs = bins[i + 1][2] if i < len(bins) - 1 else None

            # Choose highest neighboring fs
            candidate_fs = fs

            if left_fs is not None:
                candidate_fs = max(candidate_fs, left_fs)

            if right_fs is not None:
                candidate_fs = max(candidate_fs, right_fs)

            bins[i][2] = candidate_fs

        # Re-merge after adjustments
        bins = sorted(bins, key=lambda x: x[0])

        merged = []
        for b in bins:
            if not merged:
                merged.append(b)
                continue

            last = merged[-1]

            if last[1] == b[0] and last[2] == b[2]:
                last[1] = b[1]
            else:
                merged.append(b)

        bins = np.array(merged)

        ## Final sanity checks
        assert bins[0, 0] == 0
        assert bins[-1, 1] == total_samples
        assert np.all(bins[:-1, 1] == bins[1:, 0])
        assert np.all(bins[:, 2] >= min_fs)

        return bins

    def verify_nyquist_condition(self, verbose=True):
        """
        Verifies that for every multirate bin:

            f(t) <= fs/2 - safe_nyquist_gap/2

        across the entire bin.

        Raises AssertionError if violated.
        """

        violations = []

        for start, end, fs in self.detailed_bins:

            if end <= start:
                continue

            # Convert bin sample indices to times relative to merger
            t_start = (start / self.sample_rate) - self.tc_inject_lower
            t_end = (end / self.sample_rate) - self.tc_inject_lower

            # We only care about times before merger (t <= 0)
            if t_start > 0:
                continue

            # Clip to inspiral domain
            t_bin_min = max(t_start, self.t[0])
            t_bin_max = min(t_end, 0.0)

            if t_bin_max <= t_bin_min:
                continue

            # Sample densely inside bin
            t_dense = np.linspace(t_bin_min, t_bin_max, 500)

            # Interpolate frequencies
            f_dense = np.interp(t_dense, self.t, self.f)

            nyquist_limit = (fs / 2.0) - (self.safe_nyquist_gap / 2.0)

            max_freq = np.max(f_dense)

            if max_freq > nyquist_limit:

                violations.append(
                    {
                        "bin": (start, end, fs),
                        "max_freq": max_freq,
                        "nyquist_limit": nyquist_limit,
                    }
                )

        if violations:
            if verbose:
                print("Nyquist violations detected:")
                for v in violations:
                    print(v)
            raise AssertionError("Nyquist condition violated in one or more bins.")

        if verbose:
            print("Nyquist condition verified: all bins safe.")

        return True
