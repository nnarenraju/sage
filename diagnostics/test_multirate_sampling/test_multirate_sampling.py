#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Wed Mar 30 05:58:43 2022

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
import h5py
import warnings
import numpy as np
from scipy.signal import decimate
from operator import itemgetter

# Plotting
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 22})


def figure(title=""):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(2, 2, figsize=(30.0, 20.0))
    fig.suptitle(title, fontsize=28, y=0.95)
    return axs

def _plot(ax, x=None, y=None, c=None, xlabel="", ylabel="", label="", ls='solid'):
    ax.plot(x, y, c=c, label=label, linewidth=3.0, ls=ls)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label!="":
        ax.legend()

def read(data_path):
    """ Read sample and return necessary training params """
    with h5py.File(data_path, "r") as gfile:
        ## Reading all data parameters
        # Detectors
        dets = list(gfile.keys())
        # Groups within detectors (times as dict)
        detector_group_1 = gfile[dets[0]]
        detector_group_2 = gfile[dets[1]]
        # Times as list
        times_1 = list(detector_group_1.keys())
        times_2 = list(detector_group_2.keys())
        # Noise data within each detector
        data_1 = detector_group_1[times_1[0]]
        data_2 = detector_group_2[times_2[0]]
        # Stack the signals together
        signals = np.stack([data_1, data_2], axis=0)
        
        # Get the target variable from HDF5 attribute
        attrs = dict(gfile.attrs)
        label_saved = attrs['label']
        sample_rate = attrs['sample_rate']
        
        # if the sample is pure signal
        if np.allclose(label_saved, np.array([1., 0.])):
            m1 = attrs['mass_1']
            m2 = attrs['mass_2']
            signal_low_freq_cutoff = attrs['signal_low_freq_cutoff']
        
    # Return necessary params
    if np.allclose(label_saved, np.array([1., 0.])): # pure signal
        return (signals, m1, m2, signal_low_freq_cutoff, sample_rate)
    else:
        raise ValueError("MLMDC1 dataset: sample label is not one of (1., 0.), (0., 1.)")

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


"""
Multi-rate Sampling Theory and Description:
    
    1. Lower cutoff frequency for signals is 20 Hz
    2. Upper cutoff frequency for signals is f_ISCO
    3. Length of longest signal is 20 s
    4. At t = 20s, the frequency is f_ISCO
    5. We also have the relationship, f_new = f_old * (dt_merg_old/dt_merg_new)**(3./8.)
    6. At t = 0.0s, f = 20 Hz
    7. At t = 20.0s, f = f_ISCO (for worst case masses)
    8. Subs. to pt5. -> 20.0 = f_ISCO_worst * [ (0.0/20.0)**(3./8.) + 0.090909 * A(dt_merg_old) ]
    9. Second term should activate if dt_merg_old is zero, and deactivate if dt_merg_old > 0
    10. 'A' is a unit delta function at dt_merg_old = 0.0
    11. Worst case scenario the signal 'tc' will be present at t=18.0s

"""

def get_sampling_rate_bins(low_mass=10.0, max_signal_length=20.0, signal_low_freq_cutoff=20.0,
                           sample_rate=2048., tc_inject_lower=18.0, tc_inject_upper=18.2,
                           ringdown_leeway=0.4, merger_leeway=0.05, start_freq_factor=2.5,
                           fs_reduction_factor=1.9, fbin_reduction_factor=2.0):
    # Signal low freq cutoff is taken to happen at max_signal_length for worst case
    ## Approximate value for f_ISCO with BBH system both with lowest mass in priors
    # f_ISCO = 4400./(low_mass+low_mass) # Hz
    ## Closer value of f_ISCO
    f_ISCO = f_schwarzchild_isco(low_mass + low_mass)
    # Get necessary constants
    C = signal_low_freq_cutoff/f_ISCO # used to obtain upper fval when freq>=f_ISCO
    # Function to obtain frequencies wrt time
    f_check = lambda t_: signal_low_freq_cutoff/((t_/max_signal_length)**(3./8.) + C * delta(t_))
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
    save_f = []
    for n, t_ in enumerate(t):
        # The last bin end val will be the last value in signal
        if f_edge < signal_low_freq_cutoff:
            break
        # Adding a tc_upper and ringdown leeway
        leeway = tc_inject_upper - tc_inject_lower + ringdown_leeway
        if (t_ < 0.0 and t_ >= -1*leeway) or (t_ > 0.0 and t_ < merger_leeway):
            f = f_hqual
        elif t_ > 0 and t_ > merger_leeway:
            f = clip_f(f_check(t_))
        else:
            f = f_bad # bad value check
        
        # Save output response for plotting
        save_f.append(f)
            
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
            # get the time when frequency of the inspiral reduces by a factor of 2.0
            bins.append(n)
            f_edge = f_edge/fbin_reduction_factor
            
    bins.append(len(t))
    save_f = np.array(save_f)
    
    # Add the bins and sampling frequency for the pure noise chunks
    bad_bin = [[bins[0], bins[1], 64.0]] # using low sampling freq for bad bin
    hqual_bin = [[bins[1], bins[2], sample_rate]] # using highest sampling freq for ringdown+merger pahse
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
    plot_t = t[::-1] # Only these offset values can be used with clip_f(f_check())
    save_f = save_f[::-1] * 2.0
    plot_fs = []
    for sbin, ebin, fs in detailed_bins:
        if ebin == len(t):
            ebin = ebin -1
        # Check whether the freq. difference between the nyquist limit and the sampling freq.
        # is above a certain small threshold. If not, it may cause decimation artifacts.
        tlim_check = plot_t[ebin]
        
        if tlim_check > 0.0 + merger_leeway:
            flim_check = clip_f(f_check(tlim_check))
            print("f-dist b/w sampling freq and Nyquist limit = {} at {}".format(fs-flim_check*2.0, max_signal_length-(tlim_check+offset)))
            if fs - flim_check*2.0 < 20.0 and False:
                raise ValueError("The sampling frequency and the Nyquist Limit are too close to each other!")
        
        plot_fs.extend([[plot_t[sbin], fs], [plot_t[ebin], fs]])
    
    plot_fs = np.array(plot_fs)
    plt.figure(figsize=(18.0, 12.0))
    plt.plot(max_signal_length-(plot_fs[:,0]+offset), plot_fs[:,1], linewidth=3.0, linestyle="dashed", label="Sampling frequency")
    plt.plot(plot_t+offset, save_f[::-1], linewidth=3.0, label="Nyquist Frequency")
    # plt.xscale('log')
    plt.yscale('log')
    # plt.xlim(9.25, 9.45)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Return contains (bin_start_idx, bin_end_idx, sample_rate_required)
    return np.array(detailed_bins)
    
    
def multirate_sampling(path, tc_inject_lower, tc_inject_upper):
    # Downsample the data into required sampling rates and slice intervals
    # These intervals are stitched together to for a sample with MRsampling
    # Reading the signal
    signals, m1, m2, signal_low_freq_cutoff, sample_rate = read(path)
    signals = [signal[int(2.5*2048.):int(len(signal)-2.5*2048.)] for signal in signals]
    # Get the sampling rates and their bins idx
    dbins = get_sampling_rate_bins(signal_low_freq_cutoff=signal_low_freq_cutoff, 
                                   sample_rate=sample_rate)
    
    multirate_chunks = []
    new_sample_rates = []
    # Now downsample the signals from both detectors based on dbins
    for start_idx, end_idx, new_sample_rate in dbins:
        new_sample_rates.append(new_sample_rate)
        if new_sample_rate != sample_rate:
            # Calculate decimation factor
            decimation_factor = int(round(sample_rate/new_sample_rate))
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
                tmp_signals = signals[:]
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
                    
                # Decimate the signal 'n' times by a decimation factor of 2.0
                for factor in factors:
                    tmp_signals = [decimate(signal, factor) for signal in tmp_signals]
                # Store the final decimated signal
                decimated_signals = tmp_signals
                
            else:
                decimated_signals = [decimate(signal, decimation_factor) for signal in signals]
            
            # Now slice the appropriate parts of the decimated signals using bin idx
            # Note than the bin idx was made using the original sampling rate
            num_samples_original = len(signals[0])
            num_samples_decimated = int(num_samples_original/decimation_factor)
            
            ## Convert the bin idxs to decimated idxs
            # Normalise the bin idxs
            start_idx_norm = start_idx/num_samples_original
            end_idx_norm = end_idx/num_samples_original
            # Using the normalised bins idxs, get the decimated idxs
            sidx_dec = int(start_idx_norm * num_samples_decimated)
            eidx_dec = int(end_idx_norm * num_samples_decimated)
            
            # Slice the decimated signals using the start and end decimated idx
            chunk = [signal[sidx_dec:eidx_dec] for signal in decimated_signals]
        else:
            # No decimation done, original sample rate is used
            chunk = [signal[int(start_idx):int(end_idx)] for signal in signals]
        
        # Append the decimated chunk together
        multirate_chunks.append(np.stack(chunk, axis=0))
    
    # Now properly concatenate all the decimated chunks together using numpy
    multirate_signals = np.column_stack(tuple(multirate_chunks))
    # Display diagnostics
    print("\nDetailed bins for the given configuration:")
    for i, j, k in dbins:
        print("Range = [{}, {}] -> {} Hz".format(i, j, k))
    print("\nOriginal length of the signals (in num) = {}".format(len(signals[0])))
    print("Multirate length of the signals (in num) = {}".format(len(multirate_signals[0])))
    print("Factor reduction = {}".format(len(signals[0])/len(multirate_signals[0])))
    print("Equivalent length of signal in original sampling freq = {}".format(len(multirate_signals[0])/sample_rate))
    return signals, multirate_signals, multirate_chunks, new_sample_rates


def run():
    path = "/Users/nnarenraju/Desktop/dataset_5e4_20s_D1_Batch_44/foreground/foreground_1.hdf"
    # Get the normal and multirate version of the input signals
    signals, multirate_signals, chunks, srs = multirate_sampling(path, tc_inject_lower=18.0, tc_inject_upper=18.2)
    
    ax = figure(title="Comparing fixed sample rate signals & their multirate counterparts")
    
    """ Pure Signals and Multi-rate Pure Signals """
    assert len(list(set([len(signal) for signal in signals]))) == 1
    assert len(list(set([len(signal) for signal in multirate_signals]))) == 1
    
    time_axis = np.arange(len(signals[0]))
    _plot(ax[0][0], time_axis, signals[0], c='r', xlabel="Sample num", ylabel="Strain", label="Pure Signal H1")
    _plot(ax[0][1], time_axis, signals[1], c='b', xlabel="Sample num", ylabel="Strain", label="Pure Signal L1")
    
    time_axis_chunks = []
    svalue = 0
    for chunk in chunks:
        evalue = len(chunk[0]) + svalue
        taxis = np.arange(svalue, evalue)
        time_axis_chunks.append(taxis)
        svalue = evalue
        
    colors = ['r', 'b', 'm', 'g', 'orange', 'k', 'cyan']
    
    for n, chunk in enumerate(chunks):
        _plot(ax[1][0], time_axis_chunks[n], chunk[0], c=colors[n], xlabel="Sample num", ylabel="Strain", label=np.around(srs[n],3))
        _plot(ax[1][1], time_axis_chunks[n], chunk[1], c=colors[n], xlabel="Sample num", ylabel="Strain", label=np.around(srs[n],3))
    
    # plt.savefig("check_multirate_sampling.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    run()





