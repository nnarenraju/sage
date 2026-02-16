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


class TFUtils:
    """
    Utility class for waveform time-frequency evolution and
    multi-rate sampling bin calculation.
    """

    def __init__(self, data_cfg):
        self.cfg = data_cfg
        self.sample_rate = data_cfg.sample_rate
        self.prior_low_mass = data_cfg.prior_low_mass
        self.prior_high_mass = data_cfg.prior_high_mass
        self.signal_low_freq_cutoff = data_cfg.signal_low_freq_cutoff
        self.signal_length = data_cfg.signal_length
        self.decimation_start_freq = data_cfg.decimation_start_freq
        self.num_blocks = data_cfg.num_blocks
        self.lowest_allowed_fs = data_cfg.lowest_allowed_fs
        self.gap_bw_nyquist_and_fs = data_cfg.gap_bw_nyquist_and_fs
        self.override_freqs = data_cfg.override_freqs
        self.split_with_freqs = data_cfg.split_with_freqs
        self.split_with_times = data_cfg.split_with_times
        self.tc_inject_lower = data_cfg.tc_inject_lower
        self.tc_inject_upper = data_cfg.tc_inject_upper
        self.post_fudge_factor = data_cfg.post_fudge_factor
        self.noise_pad = getattr(data_cfg, "noise_pad", 0)

        # Precompute inspiral time-frequency evolution
        self.t, self.f = self._get_tf_evolution_before_tc(self.prior_low_mass)

    # -----------------------
    # Time-Frequency helpers
    # -----------------------
    @staticmethod
    def get_time_at_freq(t: np.ndarray, f: np.ndarray, search_freq: float) -> float:
        idx = (np.abs(f - search_freq)).argmin()
        return -t[idx]

    @staticmethod
    def get_freq_at_time(t: np.ndarray, f: np.ndarray, search_time: float) -> float:
        idx = (np.abs(-t - search_time)).argmin()
        return f[idx]

    @staticmethod
    def get_imr_chirp_time(
        m1: float, m2: float, s1z: float, s2z: float, fl: float
    ) -> float:
        # Multiply masses by solar mass to get SI units
        return 1.1 * lalsim.SimIMRPhenomDChirpTime(
            m1 * 1.989e30, m2 * 1.989e30, s1z, s2z, fl
        )

    def _get_tf_evolution_before_tc(self, mass: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute inspiral time-frequency evolution for lowest mass binary."""
        npoints = int(
            self.get_imr_chirp_time(mass, mass, 0.99, 0.99, self.signal_low_freq_cutoff)
            * self.sample_rate
        )
        t, f = pnutils.get_inspiral_tf(
            tc=0.0,
            mass1=mass,
            mass2=mass,
            spin1=0.99,
            spin2=0.99,
            f_low=self.signal_low_freq_cutoff,
            n_points=npoints,
            pn_2order=7,
            approximant="IMRPhenomD",
        )
        return t, f

    # -----------------------
    # Multi-rate bin calculation
    # -----------------------
    def get_sampling_rate_bins_type2(self) -> np.ndarray:
        """Compute multi-rate sampling bins for type2 strategy."""
        t, f = self.t, self.f

        # Compute pre-fudge factor
        time_at_decim_start_freq = self.get_time_at_freq(
            t, f, self.decimation_start_freq
        )
        light_travel_time = (
            Detector("H1").light_travel_time_to_detector(Detector("V1")) * 1.1
        )
        pre_fudge_factor = (light_travel_time + time_at_decim_start_freq) * 1.1

        bins = {}
        bins["noise"] = []
        bins["unchanged"] = []

        # Determine block frequencies
        if self.split_with_times:
            block_times = np.linspace(
                -time_at_decim_start_freq, min(t), self.num_blocks
            )
            block_freqs = np.array(
                [self.get_freq_at_time(t, f, -st) for st in block_times]
            )[::-1]
            block_freqs = np.floor(block_freqs)
        elif self.split_with_freqs:
            block_freqs = np.linspace(
                self.signal_low_freq_cutoff,
                self.decimation_start_freq,
                self.num_blocks,
                dtype=int,
            )
            block_freqs = block_freqs // 10 * 10

        if len(self.override_freqs) != 0:
            block_freqs = self.override_freqs

        ends = []
        start_unchanged = int(
            (self.tc_inject_lower - pre_fudge_factor) * self.sample_rate
        )
        len_unchanged = int(
            (
                pre_fudge_factor
                + (self.tc_inject_upper - self.tc_inject_lower)
                + self.post_fudge_factor
            )
            * self.sample_rate
        )
        end_unchanged = start_unchanged + len_unchanged

        bins["unchanged"].extend(
            [start_unchanged, end_unchanged, int(self.sample_rate)]
        )
        ends.append(start_unchanged)

        # Remaining blocks
        for n, bfq in enumerate(block_freqs[-2::-1]):
            bname = f"block_{n}"
            bins[bname] = []

            injstart = self.tc_inject_lower - light_travel_time
            start = int(
                (injstart - self.get_time_at_freq(t, f, bfq)) * self.sample_rate
            )
            start = start if bfq != self.signal_low_freq_cutoff else 0
            bins[bname].extend(
                [
                    start,
                    ends[-1],
                    max(
                        int(block_freqs[-(n + 1)] * 2.0 + self.gap_bw_nyquist_and_fs),
                        self.lowest_allowed_fs,
                    ),
                ]
            )
            ends.append(start)

        # Noise block after ringdown
        bins["noise"].extend(
            [
                end_unchanged,
                int(self.signal_length * self.sample_rate),
                self.lowest_allowed_fs,
            ]
        )

        # Convert to ordered array
        bins = dict(reversed(bins.items()))
        detailed_bins = np.array([v for v in bins.values()])
        return detailed_bins


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


####################################################################################


def prime_factors(n):
    # Return the prime factors, to be used in decimation
    i = 2
    factors = []
    while i**2 <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def get_time_at_freq(t, f, search_freq):
    idx = (np.abs(f - search_freq)).argmin()
    time_at_search_freq = -t[idx]
    return time_at_search_freq


def get_freq_at_time(t, f, search_time):
    t = -t
    idx = (np.abs(t - search_time)).argmin()
    freq_at_search_time = f[idx]
    return freq_at_search_time


def get_imr_chirp_time(m1, m2, s1z, s2z, fl):
    return 1.1 * lalsim.SimIMRPhenomDChirpTime(
        m1 * 1.989e30, m2 * 1.989e30, s1z, s2z, fl
    )


def get_tf_evolution_before_tc(prior_low_mass, signal_low_freq_cutoff, sample_rate):
    # Get npoints from tau
    npoints = (
        get_imr_chirp_time(
            prior_low_mass, prior_low_mass, 0.99, 0.99, signal_low_freq_cutoff
        )
        * sample_rate
    )
    # Get tf of given waveform
    t, f = pnutils.get_inspiral_tf(
        tc=0.0,
        mass1=prior_low_mass,
        mass2=prior_low_mass,
        spin1=0.99,
        spin2=0.99,
        f_low=signal_low_freq_cutoff,
        n_points=int(npoints),
        pn_2order=7,
        approximant="IMRPhenomD",
    )
    return (t, f)


def get_sampling_rate_bins_type2(data_cfg):
    # Get data_cfg input params
    signal_low_freq_cutoff = data_cfg.signal_low_freq_cutoff
    sample_rate = data_cfg.sample_rate
    prior_low_mass = data_cfg.prior_low_mass
    prior_high_mass = data_cfg.prior_high_mass
    signal_length = data_cfg.signal_length

    decimation_start_freq = data_cfg.decimation_start_freq
    noise_pad = data_cfg.noise_pad
    num_blocks = data_cfg.num_blocks
    lowest_allowed_fs = data_cfg.lowest_allowed_fs
    gap_bw_nyquist_and_fs = data_cfg.gap_bw_nyquist_and_fs
    override_freqs = data_cfg.override_freqs

    split_with_freqs = data_cfg.split_with_freqs
    split_with_times = data_cfg.split_with_times

    post_fudge_factor = data_cfg.post_fudge_factor
    tc_inject_lower = data_cfg.tc_inject_lower
    tc_inject_upper = data_cfg.tc_inject_upper

    """ Pre Fudge Factor """
    # Calculate fudge factor at left end of the waveform injection
    # Get t, f from lowest mass binary system
    # The times should vary from 0.0 to -tau starting at tc
    t, f = get_tf_evolution_before_tc(
        prior_low_mass, signal_low_freq_cutoff, sample_rate
    )
    time_at_decim_start_freq = get_time_at_freq(t, f, search_freq=decimation_start_freq)
    light_travel_time = (
        Detector("H1").light_travel_time_to_detector(Detector("V1")) * 1.1
    )
    pre_fudge_factor = (
        light_travel_time + time_at_decim_start_freq
    ) * 1.1  # just in case
    # print('Pre fudge duration = {} s'.format(pre_fudge_factor))

    """ MR Sampling params """
    bins = {}
    # Noise block after ringdown
    bins["noise"] = []
    # Block for unchanged sampling rate
    bins["unchanged"] = []
    # Get block start freqs
    if split_with_times:
        block_times = np.linspace(
            -get_time_at_freq(t, f, decimation_start_freq), min(t), num_blocks
        )
        block_freqs = np.array(
            [get_freq_at_time(t, f, -search_t) for search_t in block_times]
        )[::-1]
        block_freqs = block_freqs // 1 * 1
    if split_with_freqs:
        block_freqs = np.linspace(
            signal_low_freq_cutoff, decimation_start_freq, num_blocks, dtype=int
        )
        block_freqs = block_freqs // 10 * 10

    if len(override_freqs) != 0:
        block_freqs = override_freqs

    ## Get start and stop of all blocks
    ends = []
    # 2048 Hz sampling rate bin (unchanged sampling rate)
    start_unchanged = int((tc_inject_lower - pre_fudge_factor) * sample_rate)
    len_unchanged = int(
        (pre_fudge_factor + (tc_inject_upper - tc_inject_lower) + post_fudge_factor)
        * sample_rate
    )
    end_unchanged = start_unchanged + len_unchanged
    bins["unchanged"].append(start_unchanged)
    bins["unchanged"].append(end_unchanged)
    bins["unchanged"].append(int(sample_rate))
    # Ends will contain end idxs of all other blocks
    ends.append(start_unchanged)
    # Iterate through all other blocks and get start, end times
    for n, bfq in enumerate(block_freqs[-2::-1]):
        bname = "block_{}".format(n)
        bins[bname] = []
        # Get start and end times
        injstart = tc_inject_lower - light_travel_time
        start = int((injstart - get_time_at_freq(t, f, bfq)) * sample_rate)
        bins[bname].append(start if bfq != signal_low_freq_cutoff else 0)
        bins[bname].append(ends[-1])
        block_fs = (block_freqs[-(n + 1)] * 2.0) + gap_bw_nyquist_and_fs
        block_fs = int(block_fs) if block_fs >= lowest_allowed_fs else lowest_allowed_fs
        bins[bname].append(block_fs)
        # Add the start idx of this block as end idx for next block in iter
        ends.append(start)

    # Add noise pad after ringdown as lowest fs
    bins["noise"].append(end_unchanged)
    bins["noise"].append(int(signal_length * sample_rate))
    bins["noise"].append(lowest_allowed_fs)

    # Prepare bins to be used by mrsampling function
    bins = dict(reversed(bins.items()))
    detailed_bins = np.array([foo for foo in bins.values()])

    return detailed_bins


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
