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
import warnings
import numpy as np
from scipy.signal import decimate
from operator import itemgetter

from scipy.signal import butter, sosfiltfilt

from pycbc.conversions import tau_from_final_mass_spin, get_final_from_initial
from pycbc.detector import Detector

from pycbc import waveform
from pycbc import pnutils

import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lalsimulation as lalsim

import torch
import torch.nn.functional as F
from typing import List, Tuple

import numpy as np
from typing import Optional, List, Tuple
import lalsimulation as lalsim
from lal import MSUN_SI
from pycbc.detector import Detector

import matplotlib.pyplot as plt


class TDMultirateSampler:
    def __init__(self, data_cfg):
        """
        Multi-rate sampler for batched signals (B, D, seq_len) using multi-stage decimation.
        data_cfg.dbins: list of tuples (start_idx, end_idx, new_sample_rate)
        data_cfg.sample_rate: original sampling rate
        data_cfg.corrupted_len: int or [left, right]
        """
        self.dbins = getattr(data_cfg, "dbins", None)
        self.sample_rate = data_cfg.sample_rate
        self.corrupted_len = getattr(data_cfg, "corrupted_len", 0)

        # Precompute FIR kernels for powers-of-2 decimation
        self.max_power = 16  # max decimation 2**16
        self.fir_kernels = {}
        self._precompute_fir_kernels()

    def _precompute_fir_kernels(self):
        """Precompute low-pass FIR kernels for powers-of-2 decimation factors"""
        for i in range(1, self.max_power + 1):
            factor = 2**i
            kernel_size = 2 * factor + 1
            t = torch.arange(-factor, factor + 1, dtype=torch.float32)
            h = torch.sinc(t / factor)
            h *= torch.hamming_window(kernel_size, periodic=False)
            h /= h.sum()
            self.fir_kernels[factor] = h.view(1, 1, -1)

    @staticmethod
    def _power_of_two_factors(n: int) -> List[int]:
        """Return list of powers-of-2 factors for multi-stage decimation"""
        factors = []
        while n > 1:
            p = 2 ** (n.bit_length() - 1)
            factors.append(p)
            n //= p
        return factors

    @staticmethod
    def _slice_indices(
        start_idx: int, end_idx: int, dec_factor: int, seq_len: int
    ) -> Tuple[int, int]:
        """Compute decimated start/end indices for a bin"""
        num_dec = seq_len // dec_factor
        sidx = int(start_idx / seq_len * num_dec)
        eidx = int(end_idx / seq_len * num_dec)
        return sidx, eidx

    def _decimate_stage(self, signal: torch.Tensor, factor: int) -> torch.Tensor:
        """Single-stage decimation by a power-of-2 factor"""
        kernel = self.fir_kernels[factor].to(signal.device, signal.dtype)
        pad = kernel.shape[-1] // 2
        sig_padded = F.pad(signal, (pad, pad), mode="replicate")
        B, D, L = sig_padded.shape
        sig_reshaped = sig_padded.view(B * D, 1, L)
        filtered = F.conv1d(sig_reshaped, kernel, stride=1)
        decimated = filtered[:, :, ::factor]
        return decimated.view(B, D, decimated.shape[-1])

    def _multi_stage_decimate(
        self, signal: torch.Tensor, dec_factor: int
    ) -> torch.Tensor:
        """Multi-stage decimation via powers-of-2"""
        stages = self._power_of_two_factors(dec_factor)
        dec_sig = signal
        for f in stages:
            dec_sig = self._decimate_stage(dec_sig, f)
        return dec_sig

    def multirate_sample(self, signals: torch.Tensor) -> torch.Tensor:
        """
        signals: (B, D, seq_len)
        Returns: (B, D, new_seq_len)
        """
        if self.dbins is None:
            raise ValueError("dbins must be set in data_cfg.")

        B, D, seq_len = signals.shape
        chunks = []

        for start_idx, end_idx, new_fs in self.dbins:
            if new_fs == self.sample_rate:
                # No decimation
                chunk = signals[:, :, start_idx:end_idx]
            else:
                dec_factor = int(round(self.sample_rate / new_fs))
                dec_sig = self._multi_stage_decimate(signals, dec_factor)
                sidx, eidx = self._slice_indices(
                    start_idx, end_idx, dec_factor, seq_len
                )
                chunk = dec_sig[:, :, sidx:eidx]

            chunks.append(chunk)

        multirate_signal = torch.cat(chunks, dim=-1)

        # Remove corrupted regions
        if isinstance(self.corrupted_len, list):
            lcor, rcor = self.corrupted_len
        else:
            lcor = rcor = self.corrupted_len

        if lcor != 0 or rcor != 0:
            multirate_signal = multirate_signal[
                :, :, lcor : -rcor if rcor != 0 else None
            ]

        return multirate_signal


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

        # Checks
        if not (sample_rate != 0 and ((sample_rate & (sample_rate - 1)) == 0)):
            raise ValueError("sample_rate must be a power of 2")

        # Precompute tf for lowest mass system
        self.t, self.f = self._get_tf_evolution_before_tc()

        # Physics driven starting frequency
        self.f_isco = self._f_schwarzschild_isco(2.0 * self.prior_low_mass)
        self.fs_anchor = self._next_pow2((2.0 * self.f_isco) + self.safe_nyquist_gap)

        # Pre/Post fudge
        self.light_travel_time = (
            Detector("H1").light_travel_time_to_detector(Detector("V1")) * 1.1
        )
        self.pre_fudge_factor = (
            self._get_time_at_freq(self.f_isco) * 1.1 + self.light_travel_time
        )
        self.post_fudge_factor = self._get_post_fudge_factor()

        # Construct bins immediately
        self.detailed_bins = self._construct_multirate_bins()

    def _velocity_to_frequency(self, v, M):
        return v**3 / (M * self.MTSUN_SI * np.pi)

    def _f_schwarzschild_isco(self, M):
        return self._velocity_to_frequency((1.0 / 6.0) ** 0.5, M)

    def _get_time_at_freq(self, search_freq):
        idx = (np.abs(self.f - search_freq)).argmin()
        return self.t[idx]

    def _get_freq_at_time(self, search_time):
        idx = (np.abs(self.t - search_time)).argmin()
        return self.f[idx]

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

        bins = []

        # Unchanged region around tc (original fs)

        start_unchanged = int(
            (self.tc_inject_lower - self.pre_fudge_factor) * self.sample_rate
        )

        len_unchanged = int(
            (
                self.pre_fudge_factor
                + (self.tc_inject_upper - self.tc_inject_lower)
                + self.post_fudge_factor
            )
            * self.sample_rate
        )

        end_unchanged = start_unchanged + len_unchanged
        bins.append([start_unchanged, end_unchanged, int(self.sample_rate)])

        min_fs = self._next_pow2(self.lowest_allowed_fs)

        # STEP 1: Compute segment-limited f_min_segment
        t_available = -(self.tc_inject_lower - self.pre_fudge_factor)
        t_low = self.t[0]

        # Remember: values are negative
        # We ask if available time is enough to cover cutoff
        if t_low >= t_available:
            f_min_segment = self.signal_low_freq_cutoff
        else:
            f_min_segment = self._get_freq_at_time(t_available)

        # STEP 2: Build frequency ladder
        freqs = []
        f = self.f_isco

        while f > f_min_segment:
            freqs.append(f)
            f /= 2.0

        freqs.append(f_min_segment)

        # STEP 3: Convert freq -> time -> sample index
        starts = []

        for f_k in freqs:
            t_k = self._get_time_at_freq(f_k)
            start = int((self.tc_inject_lower + t_k) * self.sample_rate)
            starts.append(start)

        print(bins)
        print()
        print(f_min_segment)
        print(freqs)
        print(starts)
        print(self._get_time_at_freq(self.signal_low_freq_cutoff))
        print(self._get_time_at_freq(freqs[-1]))
        raise

        # -------------------------------------------------
        # STEP 4: Build bins between ladder boundaries
        # -------------------------------------------------

        ends = [start_unchanged]

        for k in range(len(starts)):

            f_k = freqs[k]
            required_fs = 2.0 * f_k + self.safe_nyquist_gap
            fs_k = max(self._next_pow2(required_fs), min_fs)

            bins.append([starts[k], ends[-1], int(fs_k)])

            ends.append(starts[k])

        # -------------------------------------------------
        # If ladder didn't reach segment start → prepend
        # -------------------------------------------------

        if ends[-1] > 0:
            bins.append([0, ends[-1], int(min_fs)])

        # -------------------------------------------------
        # Trailing noise
        # -------------------------------------------------

        bins.append(
            [end_unchanged, int(self.signal_length * self.sample_rate), int(min_fs)]
        )

        bins = np.array(bins)
        bins = bins[np.argsort(bins[:, 0])]

        # assert np.all(bins[:-1, 1] == bins[1:, 0]), "Bins are not contiguous in time"

        return bins

    def _construct_multirate_bins_(self):

        bins = {}
        ends = []

        # Unchanged region around tc (at original sampling frequency)
        start_unchanged = int(
            (self.tc_inject_lower - self.pre_fudge_factor) * self.sample_rate
        )

        len_unchanged = int(
            (
                self.pre_fudge_factor
                + (self.tc_inject_upper - self.tc_inject_lower)
                + self.post_fudge_factor
            )
            * self.sample_rate
        )

        end_unchanged = start_unchanged + len_unchanged
        bins["unchanged"] = [start_unchanged, end_unchanged, int(self.sample_rate)]
        ends.append(start_unchanged)

        # Build octave ladder from fISCO downward
        k = 1
        # Bit shift operator moves lowest_allowed_fs to nearest power of 2
        # If it is already a power of 2, nothing happens
        min_fs = self._next_pow2(self.lowest_allowed_fs)
        injstart = self.tc_inject_lower - self.light_travel_time
        start = int(injstart * self.sample_rate)

        while start > 0:

            # Instance of the frequency ladder
            f_k = self.f_isco / (2.0**k)
            f_kminus1 = self.f_isco / (2.0 ** (k - 1))
            required_fs = 2.0 * f_kminus1 + self.safe_nyquist_gap
            fs_k = max(self._next_pow2(required_fs), min_fs)

            if f_k >= self.signal_low_freq_cutoff:
                f_k_start = f_k
            else:
                # We can't go below signal low frequency cutoff
                # _get_time_at_freq is only valid till cutoff
                f_k_start = self.signal_low_freq_cutoff

            start = int(
                (injstart - self._get_time_at_freq(f_k_start)) * self.sample_rate
            )

            # We likely cutoff some very low f part of signals given sample length
            # Start will become negative in this case and we break
            if start < 0:
                start = 0

            bins[f"block_{k}"] = [
                start,
                ends[-1],
                int(fs_k),
            ]

            ends.append(start)
            k += 1

        # Trailing noise and presignal
        bins["noise"] = [
            end_unchanged,
            int(self.signal_length * self.sample_rate),
            int(self.lowest_allowed_fs),
        ]

        bins["pre_signal"] = [
            0,
            ends[-1],
            int(min_fs),
        ]

        bins = dict(reversed(bins.items()))
        bins = np.array([v for v in bins.values()])
        detailed_bins = bins[np.argsort(bins[:, 0])]

        # Check contiguity
        assert np.all(
            detailed_bins[:-1, 1] == detailed_bins[1:, 0]
        ), "Bins are not contiguous in time"

        return detailed_bins


####################################################################################


def multirate_sampling(signal, data_cfg, check=False):
    # Downsample the data into required sampling rates and slice intervals
    # These intervals are stitched together to for a sample with MRsampling
    # Get data bins (pre-calculated for given problem in dataset object)
    dbins = data_cfg.dbins

    multirate_chunks = []
    new_sample_rates = []

    # Now downsample the signals from both detectors based on dbins
    for start_idx, end_idx, new_sample_rate in dbins:
        new_sample_rates.append(new_sample_rate)
        if new_sample_rate != data_cfg.sample_rate:
            # Calculate decimation factor
            decimation_factor = int(round(data_cfg.sample_rate / new_sample_rate))
            # Decimation of signals based on decimation factor
            """
            Downsample the signal after applying an anti-aliasing filter.
            By default, an order 8 Chebyshev type I filter is used. 
            A 30 point FIR filter with Hamming window is used if ftype is ‘fir’.
            
            Decimation factor, specified as a positive integer. 
            For better results when 'r' is greater than 13, divide 'r' into 
            smaller factors and call decimate several times.
            
            """

            # Sanity check
            if decimation_factor > 13:
                # tmp_signals = signals[:] # --> many signals
                tmp_signal = np.copy(signal)

                # The decimation factor should always be of type 2**n
                # So factorisation should be quite straight-forward (depricated on April 1st, 2022)

                # Prime-factorisation
                factors = prime_factors(decimation_factor)
                factors = np.array(factors)
                if len(factors) == 1:
                    raise ValueError(
                        "The decimation factor is prime and > 13. There are no factors."
                    )
                if len(factors[factors > 13]) > 0:
                    raise ValueError(
                        "One or more prime factors > 13. Edit buffer_factor to try get rid of this."
                    )

                # Decimate the signal 'nfactor' times using the prime factors
                for factor in factors:
                    # Sequential decimation
                    # --> many signals
                    # tmp_signals = [decimate(tmp_signal, factor) for tmp_signal in tmp_signals]
                    tmp_signal = decimate(tmp_signal, factor)

                # Store the final decimated signal
                decimated_signal = tmp_signal

            else:
                # Sequential decimation
                # --> many signals
                # decimated_signals = [decimate(signal, decimation_factor) for signal in signals]
                decimated_signal = decimate(signal, decimation_factor)

            # Now slice the appropriate parts of the decimated signals using bin idx
            # Note than the bin idx was made using the original sampling rate
            num_samples_original = len(signal)
            num_samples_decimated = int(num_samples_original / decimation_factor)

            ## Convert the bin idxs to decimated idxs
            # Normalise the bin idxs
            start_idx_norm = start_idx / num_samples_original
            end_idx_norm = end_idx / num_samples_original
            # Using the normalised bins idxs, get the decimated idxs
            sidx_dec = int(start_idx_norm * num_samples_decimated)
            eidx_dec = int(end_idx_norm * num_samples_decimated)

            # Slice the decimated signals using the start and end decimated idx
            chunk = decimated_signal[sidx_dec:eidx_dec]
            # Rescale the decimated chunk using a mean based factor
            # Change in mean^2 amplitude
            # This doesn't make sense since the signal is not rescaled when decimated
            # func = np.mean
            # mean_sample = np.sqrt(func(signal**2.))
            # mean_decimated = np.sqrt(func(decimated_signal**2.))
            # factor = mean_sample/mean_decimated
            # chunk = chunk * factor
        else:
            # No decimation done, original sample rate is used
            chunk = signal[int(start_idx) : int(end_idx)]

        # Append the decimated chunk together
        # --> many signals
        # multirate_chunks.append(np.stack(chunk, axis=0))
        multirate_chunks.append(chunk)

    # Now properly concatenate all the decimated chunks together using numpy
    # --> many signals
    # multirate_signals = np.column_stack(tuple(multirate_chunks))
    # Get the idxs of each chunk edge for glitch veto
    # start = 0
    # Save the start and end idx of chunks
    # Remove corrupted samples and update indices
    # save_idxs = []
    # for chunk in multirate_chunks:
    #     save_idxs.append([start, start+len(chunk)-data_cfg.corrupted_len])
    #     start = start + len(chunk)
    # save_idxs[-1][1] -= data_cfg.corrupted_len

    multirate_signal = np.concatenate(tuple(multirate_chunks))
    # Remove regions corrupted by high decimation (if required)
    if isinstance(data_cfg.corrupted_len, list):
        lcorrupted_len = data_cfg.corrupted_len[0]
        rcorrupted_len = data_cfg.corrupted_len[1]
    elif isinstance(data_cfg.corrupted_len, int):
        lcorrupted_len = data_cfg.corrupted_len
        rcorrupted_len = data_cfg.corrupted_len

    if lcorrupted_len != 0 and rcorrupted_len != 0:
        multirate_signal = multirate_signal[lcorrupted_len : -1 * rcorrupted_len]
    else:
        multirate_signal = multirate_signal

    if check:
        return None, None
    else:
        return multirate_signal
