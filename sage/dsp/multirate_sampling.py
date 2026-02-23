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
from sage.core.manager import SharedConfig


class TorchMultiRateSampler(SharedConfig, torch.nn.Module):
    """
    Multi-rate decimator for batched GW time-domain data.

    Input shape: (B, D, L)
    Output shape: (B, D, L_compressed)

    B = batch
    D = detectors
    L = original sequence length
    """

    def __init__(
        self,
        bins: Iterable[Tuple[int, int, int]],
        original_fs: int,
        min_fs: int,
        reflect_pad: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(bins, np.ndarray):
            bins = bins.tolist()

        self.register_buffer(
            "bins_tensor",
            torch.tensor(bins, dtype=torch.int64),
            persistent=False,
        )

        self.original_fs = int(original_fs)
        self.min_fs = int(min_fs)

        max_dec_factor = int(original_fs // min_fs)
        max_power = max_dec_factor.bit_length() - 1
        self.max_power = max_power

        # Reflect Padding
        # Largest FIR kernel size if 2 * dec factor + 1
        # if max factor = 64 -> kernel size = 129 -> half = 64
        # So using 128 to 256 is very conservative
        # NOTE: Adjust if needed (and remove conservative assertion)
        if reflect_pad is not None:
            assert self.pad >= 2 * max_dec_factor
            self.pad = reflect_pad
        else:
            self.pad = 2 * max_dec_factor

        # Precompute FIR kernels for powers of 2
        self.kernels = torch.nn.ModuleDict()
        self._build_kernels()

    ## FIR kernel construction ##

    def _build_kernels(self):
        for i in range(1, self.max_power + 1):
            factor = 2**i

            kernel_size = 2 * factor + 1
            t = torch.arange(-factor, factor + 1, dtype=torch.float32)

            h = torch.sinc(t / factor)
            h = h * torch.hamming_window(kernel_size, periodic=False)
            h = h / h.sum()

            kernel = h.view(1, 1, -1)
            self.register_buffer(f"kernel_{factor}", kernel, persistent=False)

    ## Single stage decimation ##
    def _decimate_once(self, x: torch.Tensor, factor: int) -> torch.Tensor:

        kernel = getattr(self, f"kernel_{factor}")
        kernel = kernel.to(dtype=x.dtype, device=x.device)

        pad = kernel.shape[-1] // 2

        # reflect padding is better than zero padding
        x = F.pad(x, (pad, pad), mode="reflect")

        B, D, L = x.shape
        x = x.view(B * D, 1, L)

        y = F.conv1d(x, kernel, stride=1)
        y = y[:, :, ::factor]

        return y.view(B, D, y.shape[-1])

    ## Build decimation pyramid ##
    def _build_pyramid(self, x: torch.Tensor):

        pyramid = {1: x}

        current = x
        current_factor = 1

        for i in range(1, self.max_power + 1):
            factor = 2**i

            if factor > self.original_fs // self.min_fs:
                break

            current = self._decimate_once(current, 2)
            current_factor *= 2

            pyramid[current_factor] = current

        return pyramid

    ## Forward ##

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        signals: (B, D, L)
        """

        B, D, L = signals.shape

        signals = F.pad(signals, (self.pad, self.pad), mode="reflect")

        pyramid = self._build_pyramid(signals)
        chunks = []

        for start_idx, end_idx, new_fs in self.bins_tensor:

            start_idx = start_idx + self.pad
            end_idx = end_idx + self.pad

            dec_factor = self.original_fs // new_fs

            if dec_factor == 1:
                chunk = signals[:, :, start_idx:end_idx]
            else:
                decimated = pyramid[int(dec_factor)]

                sidx = start_idx // dec_factor
                eidx = end_idx // dec_factor

                chunk = decimated[:, :, sidx:eidx]

            chunks.append(chunk)

        out = torch.cat(chunks, dim=-1)

        return out


class TDMultirateSampler:
    MTSUN_SI = 4.92549102554e-06
    MSUN_SI = 1.989e30

    def __init__(
        self,
        prior_low_mass,
        prior_high_mass,
        signal_low_freq_cutoff,
        sample_rate,
        signal_length,
        lowest_allowed_fs,
        tc_inject_lower,
        tc_inject_upper,
        safe_nyquist_gap=8.0,
        min_bin_duration=0.05,
        verify_nyquist=False,
    ):

        self.prior_low_mass = prior_low_mass
        self.prior_high_mass = prior_high_mass
        self.signal_low_freq_cutoff = signal_low_freq_cutoff
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.lowest_allowed_fs = lowest_allowed_fs
        self.tc_inject_lower = tc_inject_lower
        self.tc_inject_upper = tc_inject_upper
        self.safe_nyquist_gap = safe_nyquist_gap
        self.min_bin_duration = min_bin_duration

        # Precompute tf for lowest mass system
        self.t, self.f = self._get_tf_evolution_before_tc()

        # Physics driven starting frequency
        self.f_isco = self._f_schwarzschild_isco(2.0 * self.prior_low_mass)
        self.fs_anchor = self._next_pow2((2.0 * self.f_isco) + self.safe_nyquist_gap)

        # Pre/Post fudge
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
