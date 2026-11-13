#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Physics-informed multirate sampling: a dyadic bin layout from the
inspiral time-frequency track, and the half-band FIR decimator for it.

Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Mar 29 23:41:28 2022

__author__      = nnarenraju
__copyright__   = Copyright 2022, ProjectName
__credits__     = nnarenraju
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

import math
import torch
import numpy as np
import torch.nn.functional as F

from typing import List, Tuple, Iterable

import warnings

# Silences the swiglal stdio-redirection warning raised on importing LAL
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import lalsimulation as lalsim

from pycbc.conversions import tau_from_final_mass_spin, get_final_from_initial
from pycbc.detector import Detector
from pycbc import pnutils

from sage.core.checks import check_power_of_two
from sage.core.constants import GM, MSUN
from sage.core.geometry import Window
from sage.core.pipeline import GWBatch, ProcessingState


class MultirateSampler(torch.nn.Module):
    """
    Physics-informed multirate decimator for batched time-domain strain.

    The reflect-padded input is halved by cascaded half-band FIR passes
    (63 taps, Kaiser beta=9); each bin of the layout is cut at its own
    rate, and the bins are concatenated in order.

    Parameters
    ----------
    bins : np.ndarray, shape ``(N_bins, 3)``
        Bin layout from :class:`DyadicPyramidBinning`: start sample, end
        sample and target sample rate per bin.
    window : Window
        Window geometry; sets the input sample rate.
    device : str or torch.device
        Device of the FIR kernel.
    dtype : torch.dtype
        Dtype of the FIR kernel.

    Attributes
    ----------
    bins : np.ndarray, shape ``(N_bins, 3)``
        Columns: ``[start_sample, end_sample, target_fs]``.
    kernel : torch.Tensor, shape ``(1, 1, 63)``
        Unit-L2-norm half-band FIR filter, a buffer.
    """

    def __init__(self, bins: np.ndarray, window: Window, device, dtype):
        super().__init__()

        self.bins = bins
        self.num_bins = len(self.bins)

        self.register_buffer(
            "bins_tensor",
            torch.tensor(self.bins.tolist(), dtype=torch.int64),
            persistent=False,
        )

        self.original_fs = window.sample_rate_in_hz
        self.min_fs = np.min(self.bins[:, 2])

        max_dec_factor = int(self.original_fs // self.min_fs)
        max_power = max_dec_factor.bit_length() - 1
        self.max_power = max_power

        # Precompute decimation power per bin
        dec_factors = self.original_fs // self.bins_tensor[:, 2]
        dec_powers = torch.log2(dec_factors.float()).long()
        self.register_buffer("dec_powers", dec_powers, persistent=False)

        # Half-band FIR kernel
        self._build_kernels(device, dtype)
        kernel_len = self.kernel.shape[-1]

        # Reflect padding covering the group delay at the largest factor
        group_delay = (kernel_len - 1) // 2
        self.pad = max(
            4 * kernel_len,
            2 * max_dec_factor,
            2 * group_delay * max_dec_factor,
        )

        # Bins grouped by decimation power
        self.power_to_bins = self.order_decimations(dec_powers)

    def order_decimations(self, dec_powers):
        """
        Group the bins by decimation power, with their decimated slices.

        Also sets ``powers``, the sorted unique powers, and
        ``max_required_power``.

        Parameters
        ----------
        dec_powers : torch.Tensor, shape ``(N_bins,)`` int64
            ``log2(original_fs / target_fs)`` for each bin.

        Returns
        -------
        dict[int, list[tuple[int, int, int]]]
            Each power to its ``(start_idx, end_idx, bin_index)``
            slices of the padded input decimated by ``2**power``.
        """
        unique_powers = sorted(set(dec_powers.tolist()))
        self.powers = unique_powers
        self.max_required_power = max(unique_powers)

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

    def _build_kernels(self, device, dtype):
        """
        Register the 63-tap Kaiser half-band FIR kernel as ``kernel``.

        Parameters
        ----------
        device : str or torch.device
            Device of the kernel.
        dtype : torch.dtype
            Dtype of the kernel.
        """

        NTAPS = 63

        M = NTAPS - 1
        n = torch.arange(NTAPS, dtype=torch.float32)

        # Ideal half-band impulse response
        h = torch.sinc((n - M / 2) / 2)

        beta = 9  # Kaiser: A ~ 8.7 + 9.07 * beta ~ 90 dB attenuation
        h *= torch.kaiser_window(NTAPS, periodic=False, beta=beta)

        # Unit L2 norm keeps white-noise variance per 2x stage; the DC
        # gain is then ~1.439 per stage
        h /= torch.sqrt((h**2).sum())

        self.register_buffer(
            "kernel",
            h.view(1, 1, -1).to(device, dtype=dtype),
            persistent=False,
        )

    def _decimate_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        One half-band FIR low-pass and downsample by 2.

        The input is reflect-padded by half the kernel length first.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, L)``
            Time series.

        Returns
        -------
        torch.Tensor, shape ``(B, D, ceil(L/2))``
            Decimated series.
        """

        pad = self.kernel.shape[-1] // 2
        x = F.pad(x, (pad, pad), mode="reflect")

        B, D, L = x.shape
        x = x.reshape(B * D, 1, L)

        y = F.conv1d(x, self.kernel, stride=2)

        return y.reshape(B, D, y.shape[-1])

    @staticmethod
    def next_state(state: ProcessingState) -> ProcessingState:
        """
        State after multirate sampling.

        Parameters
        ----------
        state : ProcessingState
            State of the incoming batch.

        Returns
        -------
        ProcessingState
            TD_MULTIRATE.

        Raises
        ------
        PipelineError
            If the batch is not uniform time-domain data.
        """
        return state.after_multirate()

    @torch.no_grad()
    def forward(self, batch: GWBatch) -> GWBatch:
        """
        Decimate time-domain strain by the bin layout.

        Parameters
        ----------
        batch : GWBatch
            Whitened ``TD_UNIFORM`` strain ``(B, D, L)``.

        Returns
        -------
        GWBatch
            ``TD_MULTIRATE`` strain ``(B, D, L_compressed)``.

        Raises
        ------
        PipelineError
            If the batch is not uniform time-domain data.
        """
        new_state = self.next_state(batch.state)
        return GWBatch(self._decimate(batch.data), new_state)

    def _decimate(self, noisy_signals: torch.Tensor) -> torch.Tensor:
        """
        Decimate each bin to its rate and concatenate the bins.

        Parameters
        ----------
        noisy_signals : torch.Tensor, shape ``(B, D, L)``
            Uniformly sampled strain.

        Returns
        -------
        torch.Tensor, shape ``(B, D, L_compressed)``
            Multirate strain.
        """
        x = F.pad(
            noisy_signals,
            (self.pad, self.pad),
            mode="reflect",
        )

        outputs = [None] * self.num_bins

        current = x
        current_power = 0

        for target_power in self.powers:

            # Continue from the previous power
            steps = target_power - current_power

            for _ in range(steps):
                current = self._decimate_once(current)

            current_power = target_power

            for sidx, eidx, original_idx in self.power_to_bins[target_power]:
                chunk = current[:, :, sidx:eidx]
                outputs[original_idx] = chunk

        return torch.cat(outputs, dim=-1)   # (B, D, L_compressed)


class DyadicPyramidBinning:
    """
    Multirate bin layout from the inspiral time-frequency track.

    Boundaries come from a frequency ladder halving from f_ISCO, timed
    on the longest inspiral of the prior; each bin gets the lowest
    power-of-2 rate it needs, and the merger region keeps the full rate.

    Parameters
    ----------
    mass_range : tuple of float
        Lowest and highest component mass of the prior (solar masses).
        The lowest sets the frequency ladder, the highest the
        post-merger time.
    delta_tc_range : tuple of float
        Earliest and latest merger time in the window (seconds); sets
        the full-rate region.
    window : Window
        Window geometry; sets the sample rate and segment length.
    low_frequency_cutoff_in_hz : float
        Signal low-frequency cutoff; the bottom of the frequency ladder.
    lowest_sample_rate_in_hz : float
        Minimum sample rate of any bin; a power of two.
    nyquist_margin_in_hz : float
        Margin added to twice the highest frequency of a bin when
        choosing its sample rate.
    min_bin_length_in_s : float
        Bins shorter than this are raised to their neighbours' highest
        rate and merged.
    pre_merger_leeway_in_s, post_merger_leeway_in_s : float
        Extra time kept at the full rate before the earliest and after
        the latest merger time, on top of the fixed margins.

    Attributes
    ----------
    detailed_bins : np.ndarray, shape ``(N_bins, 3)``
        Columns: ``[start_sample, end_sample, sample_rate]``, indices
        from the start of the full-rate segment.

    Raises
    ------
    ValueError
        If the sample rate or lowest sample rate is not a power of two,
        or a leeway is negative.
    """
    def __init__(
        self,
        mass_range,
        delta_tc_range,
        window: Window,
        low_frequency_cutoff_in_hz: float,
        lowest_sample_rate_in_hz: float,
        nyquist_margin_in_hz: float,
        min_bin_length_in_s: float,
        pre_merger_leeway_in_s: float,
        post_merger_leeway_in_s: float,
    ):
        check_power_of_two("data.sample_rate_in_hz", window.sample_rate_in_hz)
        check_power_of_two("multirate lowest_sample_rate_in_hz", lowest_sample_rate_in_hz)
        for name, leeway in (
            ("pre_merger_leeway_in_s", pre_merger_leeway_in_s),
            ("post_merger_leeway_in_s", post_merger_leeway_in_s),
        ):
            if leeway < 0:
                raise ValueError(f"multirate {name} must be >= 0, got {leeway}")

        self.prior_low_mass, self.prior_high_mass = mass_range
        self.signal_low_freq_cutoff = low_frequency_cutoff_in_hz
        self.sample_rate = window.sample_rate_in_hz
        self.signal_length = window.sample_length_in_s
        self.lowest_allowed_fs = lowest_sample_rate_in_hz
        self.tc_inject_lower, self.tc_inject_upper = delta_tc_range
        self.safe_nyquist_gap = nyquist_margin_in_hz
        self.min_bin_duration = min_bin_length_in_s
        self.extra_pre_tc_leeway = pre_merger_leeway_in_s
        self.extra_post_tc_leeway = post_merger_leeway_in_s

        # Track of the lowest-mass system
        self.t, self.f = self._get_tf_evolution_before_tc()

        # Top of the frequency ladder: ISCO of the lowest total mass
        self.f_isco = self._f_schwarzschild_isco(2.0 * self.prior_low_mass)
        self.fs_anchor = self._next_pow2((2.0 * self.f_isco) + self.safe_nyquist_gap)

        # H1-V1 light travel time with a 10% margin
        self.light_travel_time = (
            Detector("H1").light_travel_time_to_detector(Detector("V1")) * 1.1
        )
        self.pre_fudge_factor = (
            self._get_pre_fudge_factor() + self.extra_pre_tc_leeway
        )
        self.post_fudge_factor = (
            self._get_post_fudge_factor() + self.extra_post_tc_leeway
        )

        self.detailed_bins = self._construct_multirate_bins()

    def _velocity_to_frequency(self, v, M):
        """
        GW frequency of a PN velocity.

        Parameters
        ----------
        v : float
            PN velocity parameter.
        M : float
            Total mass [Msun].

        Returns
        -------
        float
            GW frequency [Hz].
        """
        return v**3 / (M * GM * np.pi)

    def _f_schwarzschild_isco(self, M):
        """
        GW frequency at the Schwarzschild ISCO.

        Parameters
        ----------
        M : float
            Total mass [Msun].

        Returns
        -------
        float
            GW frequency [Hz].
        """
        return self._velocity_to_frequency((1.0 / 6.0) ** 0.5, M)

    def _get_freq_at_time(self, search_time):
        """
        Frequency of the track sample nearest a time.

        Parameters
        ----------
        search_time : float
            Time from merger [s].

        Returns
        -------
        float
            Frequency [Hz].
        """
        idx = (np.abs(self.t - search_time)).argmin()
        return self.f[idx]

    def _get_time_at_freq(self, search_freq):
        """
        Time of the track sample nearest a frequency.

        Parameters
        ----------
        search_freq : float
            Frequency [Hz].

        Returns
        -------
        float
            Time from merger [s], negative before merger.
        """
        idx = (np.abs(self.f - search_freq)).argmin()
        return self.t[idx]

    def _get_tf_evolution_before_tc(self):
        """
        Time-frequency track of the longest inspiral of the prior.

        Equal lowest masses, aligned spins 0.99, IMRPhenomD timing in
        ``pnutils.get_inspiral_tf`` with ``tc = 0``; one point per
        sample over :meth:`get_imr_chirp_time`.

        Returns
        -------
        t, f : numpy.ndarray
            Times from merger [s] and frequencies [Hz].
        """
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
        """
        Full-rate time kept before the earliest merger.

        Returns
        -------
        float
            1.1 x (light travel time + track time at f_ISCO), in s.
        """
        time_at_decim_start_freq = self._get_time_at_freq(search_freq=self.f_isco)
        pre_fudge_factor = (
            self.light_travel_time + time_at_decim_start_freq
        ) * 1.1  # 10% margin
        return pre_fudge_factor

    def _get_post_fudge_factor(self):
        """
        Full-rate time kept after the latest merger.

        Fifteen damping times of the heaviest remnant (both masses at
        the prior maximum, aligned spins 0.99), plus the H1-V1 light
        travel time with a 10% margin.

        Returns
        -------
        float
            Time in s.
        """
        m_final, spin_final = get_final_from_initial(
            mass1=self.prior_high_mass,
            mass2=self.prior_high_mass,
            spin1z=0.99,
            spin2z=0.99,
        )
        post_fudge_factor = (
            tau_from_final_mass_spin(m_final, spin_final) * 10 * 1.5
        )  # 10 damping times, 50% margin
        light_travel_time = (
            Detector("H1").light_travel_time_to_detector(Detector("V1")) * 1.1
        )
        post_fudge_factor += light_travel_time
        return post_fudge_factor

    def _next_pow2(self, x):
        """
        Smallest power of 2 at or above ``x``.

        Parameters
        ----------
        x : float
            Positive value.

        Returns
        -------
        int
            Power of 2.
        """
        return 1 << int(np.ceil(np.log2(x)))

    def get_imr_chirp_time(self, m1, m2, s1z, s2z, fl):
        """
        1.1 times the IMRPhenomD chirp time.

        Parameters
        ----------
        m1, m2 : float
            Component masses [Msun].
        s1z, s2z : float
            Aligned spins.
        fl : float
            Starting frequency [Hz].

        Returns
        -------
        float
            Time in s.
        """
        return 1.1 * lalsim.SimIMRPhenomDChirpTime(
            m1 * MSUN, m2 * MSUN, s1z, s2z, fl
        )

    def plot_multirate_tf(self):
        """Plot the bins' Nyquist frequencies over the prior tracks."""

        # Deferred: matplotlib is needed only for this plot
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.cm as cm
        import numpy as np

        fontsize = 16

        fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=300)

        # Tracks of near-equal-mass systems across the prior
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

        # Plot origin: mid merger time plus 0.05 s
        tc = (self.tc_inject_upper + self.tc_inject_lower) / 2.0 + 0.05

        # Nyquist frequency of each bin
        x = []
        y = []

        for foo in self.detailed_bins:
            x.extend([foo[0] / self.sample_rate - tc, foo[1] / self.sample_rate - tc])
            y.extend([foo[2] / 2.0, foo[2] / 2.0])

        ax.plot(
            x, y, linestyle="dashed", c=np.array([191, 44, 35]) / 255.0, linewidth=2.0
        )

        # Margins of the time axis
        lpad = self.tc_inject_lower - self.get_imr_chirp_time(
            self.prior_low_mass,
            self.prior_low_mass,
            0.99,
            0.99,
            self.signal_low_freq_cutoff,
        )

        rpad = self.signal_length - (self.tc_inject_upper + self.post_fudge_factor)

        # Top axis: time from the window start
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
        """
        Bin layout covering the segment.

        Returns
        -------
        numpy.ndarray, shape ``(N_bins, 3)``
            Contiguous ``[start_sample, end_sample, sample_rate]`` rows.
        """

        total_samples = int(self.signal_length * self.sample_rate)
        min_fs = self._next_pow2(self.lowest_allowed_fs)

        # Full-rate region around the merger times
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

        # Frequency ladder halving from f_ISCO to the cutoff
        f_current = self.f_isco
        f_floor = self.signal_low_freq_cutoff
        freqs = []

        while f_current > f_floor:
            freqs.append(f_current)
            f_current /= 2.0

        freqs.append(f_floor)

        # Ladder frequencies to sample boundaries
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

        # Bins before the full-rate region
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

        # Lowest rate from the segment start
        if prev_end > 0:
            bins.append([0, prev_end, int(min_fs)])

        bins.append([start_unchanged, end_unchanged, int(self.sample_rate)])

        # Lowest rate after the full-rate region
        if end_unchanged < total_samples:
            bins.append([end_unchanged, total_samples, int(min_fs)])

        # Merge contiguous bins of equal rate
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

        # Raise short bins to the highest neighbouring rate
        min_bin_samples = int(self.min_bin_duration * self.sample_rate)

        bins = bins.tolist()

        for i in range(len(bins)):

            start, end, fs = bins[i]
            width = end - start

            if width >= min_bin_samples:
                continue

            left_fs = bins[i - 1][2] if i > 0 else None
            right_fs = bins[i + 1][2] if i < len(bins) - 1 else None

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

        # The bins tile the segment at or above the lowest rate
        assert bins[0, 0] == 0
        assert bins[-1, 1] == total_samples
        assert np.all(bins[:-1, 1] == bins[1:, 0])
        assert np.all(bins[:, 2] >= min_fs)

        return bins

    def verify_nyquist_condition(self, verbose=True):
        """
        Check ``f(t) <= fs/2 - safe_nyquist_gap/2`` in every bin.

        The track is sampled at 500 points over each bin's pre-merger
        part.

        Parameters
        ----------
        verbose : bool
            Print the outcome and any violations.

        Returns
        -------
        bool
            True when every bin passes.

        Raises
        ------
        AssertionError
            If any bin violates the condition.
        """

        violations = []

        for start, end, fs in self.detailed_bins:

            if end <= start:
                continue

            # Convert bin sample indices to times relative to merger
            t_start = (start / self.sample_rate) - self.tc_inject_lower
            t_end = (end / self.sample_rate) - self.tc_inject_lower

            # Pre-merger times only (t <= 0)
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
