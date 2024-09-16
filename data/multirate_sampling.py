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



def prime_factors(n):
    # Return the prime factors, to be used in decimation
    i = 2
    factors = []
    while i**2<=n:
        if n%i:
            i+=1
        else:
            n//=i
            factors.append(i)
    if n>1:
        factors.append(n)
    return factors


def velocity_to_frequency(v, M):
    """ Calculate the gravitational-wave frequency from the
    total mass and invariant velocity.
    Taken from:
        https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html

    Parameters
    ----------
    v : float
        Invariant velocity
    M : float
        Binary total mass

    Returns
    -------
    f : float
        Gravitational-wave frequency
    """
    MTSUN_SI = 4.92549102554e-06
    return v**(3.0) / (M * MTSUN_SI * np.pi)


def f_schwarzchild_isco(M):
    """
    Innermost stable circular orbit (ISCO) for a test particle
    orbiting a Schwarzschild black hole.
    Taken from:
        https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html

    Parameters
    ----------
    M : float or numpy.array
        Total mass in solar mass units

    Returns
    -------
    f : float or numpy.array
        Frequency in Hz
    """
    return velocity_to_frequency((1.0/6.0)**(0.5), M)


def get_sampling_rate_bins_type1(data_cfg):
    
    # Get data_cfg input params
    signal_low_freq_cutoff=data_cfg.signal_low_freq_cutoff
    sample_rate=data_cfg.sample_rate
    low_mass=data_cfg.prior_low_mass
    max_signal_length=data_cfg.max_signal_length
    tc_inject_lower=data_cfg.tc_inject_lower 
    tc_inject_upper=data_cfg.tc_inject_upper
    ringdown_leeway=data_cfg.ringdown_leeway
    merger_leeway=data_cfg.merger_leeway
    start_freq_factor=data_cfg.start_freq_factor
    fs_reduction_factor=data_cfg.fs_reduction_factor
    fbin_reduction_factor=data_cfg.fbin_reduction_factor
    
    # Signal low freq cutoff is taken to happen at max_signal_length for worst case
    ## Approximate value for f_ISCO with BBH system both with lowest mass in priors
    # f_ISCO = 4400./(low_mass+low_mass) # Hz
    ## Closer value of f_ISCO
    f_ISCO = f_schwarzchild_isco(low_mass + low_mass)
    # Get necessary constants
    C = signal_low_freq_cutoff/f_ISCO # used to obtain upper fval when freq>=f_ISCO
    # Function to obtain frequencies wrt time
    f_check = lambda t_: signal_low_freq_cutoff/((t_/20.0)**(3./8.) + C * delta(t_))
    # delta function at dt_merg_old == 0.0, to activate 'C'
    delta = lambda t_: 1.0 if t_==0 else 0.0
    # Clip the frequency response for all values greater than f_ISCO
    clip_f = lambda f_: f_ISCO if f_>f_ISCO else f_
    
    # Times given as input to get the frequencies must take into account tc_inject_lower
    offset = max_signal_length - tc_inject_lower
    # Getting the bins with decaying sampling frequencies
    offset_signal_length = max_signal_length - offset
    t = np.linspace(-1.0*offset, offset_signal_length, int(max_signal_length*sample_rate))
    
    # Get check frequencies
    f_edge = f_ISCO
    f_hqual = 750.0
    f_bad = 50.0
    bad_chunk = True
    hqual_chunk = True
    bins = [0] # bin always starts at 0
    check_f = []
    for n, t_ in enumerate(t):
        # The last bin end val will be the last value in signal
        if f_edge < signal_low_freq_cutoff:
            break
        # Adding a tc_upper and ringdown leeway
        # TODO: Add 2s to end_time to account for ringdown and light-travel delay
        leeway = tc_inject_upper - tc_inject_lower + ringdown_leeway
        if (t_ < 0.0 and t_ >= -1*leeway) or (t_ >= 0.0 and t_ < merger_leeway):
            f = f_hqual
        elif t_ > 0 and t_ > merger_leeway:
            f = clip_f(f_check(t_))
        else:
            f = f_bad # bad value check
        
        # Save output response for plotting
        check_f.append(f)
            
        # Adding the freq edges
        if f == f_hqual and bad_chunk:
            # this is where the bad chunk ends and ringdown+merger phase should start
            bins.append(n)
            bad_chunk = False
        elif f < f_hqual and f > f_ISCO/2.0 and hqual_chunk:
            # if freq is f_ISCO, we transition into inspiral phase from ringdown+merger phase
            # this should correspond to the ringdown phase and add a merger leeway
            bins.append(n)
            hqual_chunk = False
        elif f < f_edge/fbin_reduction_factor and f != f_bad:
            # get the time when frequency of the inspiral reduces by a factor of 2.0 (or fbin_reduction_factor)
            bins.append(n)
            f_edge = f_edge/fbin_reduction_factor
            
    bins.append(len(t))
    check_f = np.array(check_f)
    
    # Add the bins and sampling frequency for the pure noise chunks
    bad_bin = [[bins[0], bins[1], 64.0]] # using low sampling freq for bad bin
    hqual_bin = [[bins[1], bins[2], sample_rate]] # using highest sampling freq for ringdown+merger phase
    # Starting freq. at time of merger is given based on f_ISCO
    # We use a sampling freq. a factor of 4 higher than f_ISCO
    if start_freq_factor < 2.0:
        raise ValueError("Reduction factor has to be *at least* 2.0 to abide by the Nyquist Limit")
    if start_freq_factor == 2.0:
        warnings.warn("buffer_factor is at Nyquist Limit. Good performace is not gauranteed.")
    
    ### Using a factor of 2.0 sampling freq.
    # for n in range(16): # goes up to 2**16 = 65.536 KHz
    #     if 2.**n > f_ISCO * red_factor:
    #         if red_factor==2.0 and 2.**n - f_ISCO*red_factor < 50.0:
    #             raise ValueError("Sampling freq. too close to Nyquist Frequency")
    #         start_freq = 2.**n
    #         break
    
    
    # Set the starting frequency based on a buffer_factor
    start_freq = f_ISCO * start_freq_factor
    # Get the bin start and end idx along with the required sampling rate
    # Two bin addition are already done above, so we start range at 2
    detailed_bins = [[bins[n], bins[n+1], start_freq/fs_reduction_factor**(n-2)] for n in range(2, len(bins[:-1]))]
    # Adding bad bins
    detailed_bins = bad_bin + hqual_bin + detailed_bins
    # Manipulate detailed bins to account for reversed perspective of sample
    man = lambda b: int(max_signal_length*sample_rate) - b
    detailed_bins = [[man(b[1]), man(b[0]), b[2]] for b in detailed_bins]
    # Sort the bins based on start idx so its easier to concatenate later on
    detailed_bins = sorted(detailed_bins, key=itemgetter(0))
    
    """ Plotting the sampling frequency response wrt bins """
    check_t = t[::-1] # Only these offset values can be used with clip_f(f_check())
    check_f = check_f[::-1] * 2.0
    for sbin, ebin, fs in detailed_bins:
        if ebin == len(t):
            ebin = ebin -1
        # Check whether the freq. difference between the nyquist limit and the sampling freq.
        # is above a certain small threshold. If not, it may cause decimation artifacts.
        tlim_check = check_t[ebin]
        
        if tlim_check > 0.0 + merger_leeway:
            flim_check = clip_f(f_check(tlim_check))
            if fs - flim_check*2.0 < 20.0 and False:
                raise ValueError("The sampling frequency and the Nyquist Limit are too close to each other!")
    
    # Return contains (bin_start_idx, bin_end_idx, sample_rate_required)
    return np.array(detailed_bins)


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
    return 1.1 * lalsim.SimIMRPhenomDChirpTime(m1*1.989e+30, m2*1.989e+30, s1z, s2z, fl)


def get_tf_evolution_before_tc(prior_low_mass, signal_low_freq_cutoff, sample_rate):
    # Get npoints from tau
    npoints = get_imr_chirp_time(prior_low_mass, prior_low_mass, 
                                 0.99, 0.99, 
                                 signal_low_freq_cutoff) * sample_rate
    # Get tf of given waveform
    t, f = pnutils.get_inspiral_tf(tc=0.0, 
                        mass1=prior_low_mass, mass2=prior_low_mass, 
                        spin1=0.99, spin2=0.99, 
                        f_low=signal_low_freq_cutoff, 
                        n_points=int(npoints), 
                        pn_2order=7, 
                        approximant='IMRPhenomD')
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
    t, f = get_tf_evolution_before_tc(prior_low_mass, 
                                      signal_low_freq_cutoff, 
                                      sample_rate)
    time_at_decim_start_freq = get_time_at_freq(t, f, search_freq=decimation_start_freq)
    light_travel_time = Detector('H1').light_travel_time_to_detector(Detector('V1')) * 1.1
    pre_fudge_factor = (light_travel_time + time_at_decim_start_freq) * 1.1 # just in case
    # print('Pre fudge duration = {} s'.format(pre_fudge_factor))

    """ MR Sampling params """
    bins = {}
    # Noise block after ringdown
    bins['noise'] = []
    # Block for unchanged sampling rate
    bins['unchanged'] = []
    # Get block start freqs
    if split_with_times:
        block_times = np.linspace(-get_time_at_freq(t, f, decimation_start_freq), min(t), num_blocks)
        block_freqs = np.array([get_freq_at_time(t, f, -search_t) for search_t in block_times])[::-1]
        block_freqs = block_freqs // 1 * 1
    if split_with_freqs:
        block_freqs = np.linspace(signal_low_freq_cutoff, decimation_start_freq, num_blocks, dtype=int)
        block_freqs = block_freqs // 10 * 10

    if len(override_freqs) != 0:
        block_freqs = override_freqs
    
    ## Get start and stop of all blocks
    ends = []
    # 2048 Hz sampling rate bin (unchanged sampling rate)
    start_unchanged = int((tc_inject_lower - pre_fudge_factor) * sample_rate)
    len_unchanged = int((pre_fudge_factor + (tc_inject_upper - tc_inject_lower) + post_fudge_factor) * sample_rate)
    end_unchanged = start_unchanged + len_unchanged
    bins['unchanged'].append(start_unchanged)
    bins['unchanged'].append(end_unchanged)
    bins['unchanged'].append(int(sample_rate))
    # Ends will contain end idxs of all other blocks
    ends.append(start_unchanged)
    # Iterate through all other blocks and get start, end times
    for n, bfq in enumerate(block_freqs[-2::-1]):
        bname = 'block_{}'.format(n)
        bins[bname] = []
        # Get start and end times
        injstart = tc_inject_lower - light_travel_time
        start = int((injstart - get_time_at_freq(t, f, bfq)) * sample_rate)
        bins[bname].append(start if bfq != signal_low_freq_cutoff else 0)
        bins[bname].append(ends[-1])
        block_fs = (block_freqs[-(n+1)] * 2.) + gap_bw_nyquist_and_fs
        block_fs = int(block_fs) if block_fs >= lowest_allowed_fs else lowest_allowed_fs
        bins[bname].append(block_fs)
        # Add the start idx of this block as end idx for next block in iter
        ends.append(start)

    # Add noise pad after ringdown as lowest fs
    bins['noise'].append(end_unchanged)
    bins['noise'].append(int(signal_length * sample_rate))
    bins['noise'].append(lowest_allowed_fs)

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
            decimation_factor = int(round(data_cfg.sample_rate/new_sample_rate))
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
                # Is the above deprication an April's Fools joke? Absolutely not.
                
                # Prime-factorisation
                factors = prime_factors(decimation_factor)
                factors = np.array(factors)
                if len(factors) == 1:
                    raise ValueError("The decimation factor is prime and > 13. There are no factors.")
                if len(factors[factors>13]) > 0:
                    raise ValueError("One or more prime factors > 13. Edit buffer_factor to try get rid of this.")
                    
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
            num_samples_decimated = int(num_samples_original/decimation_factor)
            
            ## Convert the bin idxs to decimated idxs
            # Normalise the bin idxs
            start_idx_norm = start_idx/num_samples_original
            end_idx_norm = end_idx/num_samples_original
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
            chunk = signal[int(start_idx):int(end_idx)]
        
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
        multirate_signal = multirate_signal[lcorrupted_len:-1*rcorrupted_len]
    else:
        multirate_signal = multirate_signal

    if check:
        return None, None
    else:
        return multirate_signal
