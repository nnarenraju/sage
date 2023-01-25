#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Jan 17 14:59:18 2023

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
import os
import glob
import h5py
import numpy as np
import multiprocessing as mp

# PyCBC modules
import pycbc.waveform, pycbc.detector
from pycbc.filter.matchedfilter import sigmasq
from pycbc.types import TimeSeries, FrequencySeries

# Prettification
from tqdm import tqdm


def optimise_fmin(h_pol, signal_length, signal_low_freq_cutoff, sample_rate, waveform_kwargs):
    # Use self.waveform_kwargs to calculate the fmin for given params
    # Such that the length of the signal is atleast 20s by the time it reaches fmin
    current_start_time = -1*h_pol.get_sample_times()[0]
    req_start_time = signal_length - h_pol.get_sample_times()[-1]
    fmin = signal_low_freq_cutoff*(current_start_time/req_start_time)**(3./8.)
    
    while True:
        # fmin_new is the fmin required for the current params to produce 20.0s signal
        waveform_kwargs['f_lower'] = fmin
        h_plus, h_cross = pycbc.waveform.get_td_waveform(**waveform_kwargs)
        # Sanity check to verify the new signal length
        new_signal_length = len(h_plus)/sample_rate
        if new_signal_length > signal_length:
            break
        else:
            fmin = fmin - 3.0
        
    # Return new signal
    return h_plus, h_cross


def get_injection_snr(args):
    ## Generate the full waveform
    injection_values, data_cfg = args
    # LAL Detector Objects (used in project_wave)
    # Detector objects (these are lal objects and may present problems when parallelising)
    # Create the detectors (TODO: generalise this!!!)
    detectors_abbr = ('H1', 'L1')
    dets = []
    for det_abbr in detectors_abbr:
        dets.append(pycbc.detector.Detector(det_abbr))
    
    sample_rate = data_cfg.sample_rate
    signal_low_freq_cutoff = data_cfg.signal_low_freq_cutoff
    signal_approximant = data_cfg.signal_approximant
    reference_freq = data_cfg.reference_freq
    signal_length = data_cfg.signal_length
    whiten_padding = data_cfg.whiten_padding
    error_padding_in_s = data_cfg.error_padding_in_s
    error_padding_in_num = data_cfg.error_padding_in_num
    sample_length_in_num = data_cfg.sample_length_in_num
    
    # Generate source parameters
    waveform_kwargs = {'delta_t': 1./sample_rate}
    waveform_kwargs['f_lower'] = signal_low_freq_cutoff
    waveform_kwargs['approximant'] = signal_approximant
    waveform_kwargs['f_ref'] = reference_freq
    # Update waveform kwargs using injection values
    waveform_kwargs.update(injection_values)
    
    h_plus, h_cross = pycbc.waveform.get_td_waveform(**waveform_kwargs)
    # If the signal is smaller than 20s, we change fmin such that it is atleast 20s
    if -1*h_plus.get_sample_times()[0] + h_plus.get_sample_times()[-1] < signal_length:
        # Pass h_plus or h_cross
        h_plus, h_cross = optimise_fmin(h_plus, signal_length, signal_low_freq_cutoff, sample_rate, waveform_kwargs)
    
    if -1*h_plus.get_sample_times()[0] + h_plus.get_sample_times()[-1] > signal_length:
        new_end = h_plus.get_sample_times()[-1]
        new_start = -1*(signal_length - new_end)
        h_plus = h_plus.time_slice(start=new_start, end=new_end)
        h_cross = h_cross.time_slice(start=new_start, end=new_end)
    
    # Sanity check for signal lengths
    if len(h_plus)/sample_rate != signal_length:
        act = signal_length*sample_rate
        obs = len(h_plus)
        raise ValueError('Signal length ({}) is not as expected ({})!'.format(obs, act))
    
    # # Properly time and project the waveform (What there is)
    start_time = injection_values['injection_time'] + h_plus.get_sample_times()[0]
    end_time = injection_values['injection_time'] + h_plus.get_sample_times()[-1]
    
    # Calculate the number of zeros to append or prepend (What we need)
    # Whitening padding will be corrupt and removed in whiten transformation
    start_samp = injection_values['tc'] + (whiten_padding/2.0)
    start_interval = injection_values['injection_time'] - start_samp
    # subtract delta value for length error (0.001 if needed)
    end_padding = whiten_padding/2.0
    post_merger = signal_length - injection_values['tc']
    end_interval = injection_values['injection_time'] + post_merger + end_padding
    
    # Calculate the difference (if any) between two time sets
    diff_start = start_time - start_interval
    diff_end = end_interval - end_time
    # Convert num seconds to num samples
    diff_end_num = int(diff_end * sample_rate)
    diff_start_num = int(diff_start * sample_rate)
    
    expected_length = ((end_interval-start_interval) + error_padding_in_s*2.0) * sample_rate
    observed_length = len(h_plus) + (diff_start_num + diff_end_num + error_padding_in_num*2.0)
    diff_length = expected_length - observed_length
    if diff_length != 0:
        diff_end_num += diff_length
    
    # If any positive difference exists, add padding on that side
    # Pad h_plus and h_cross with zeros on both end for slicing
    if diff_end > 0.0:
        # Append zeros if we need samples after signal ends
        h_plus.append_zeros(int(diff_end_num + error_padding_in_num))
        h_cross.append_zeros(int(diff_end_num + error_padding_in_num))
    
    if diff_start > 0.0:
        # Prepend zeros if we need samples before signal begins
        # prepend_zeros arg must be an integer
        h_plus.prepend_zeros(int(diff_start_num + error_padding_in_num))
        h_cross.prepend_zeros(int(diff_start_num + error_padding_in_num))
    
    assert len(h_plus) == sample_length_in_num + error_padding_in_num*2.0
    assert len(h_cross) == sample_length_in_num + error_padding_in_num*2.0
    
    # Setting the start_time, sets epoch and end_time as well within the TS
    # Set the start time of h_plus and h_plus after accounting for prepended zeros
    h_plus.start_time = start_interval - error_padding_in_s
    h_cross.start_time = start_interval - error_padding_in_s
    
    # Calculate htilde from the above polarisation data
    declination, right_ascension = injection_values['dec'], injection_values['ra']
    # Using PyCBC project_wave to get h_t from h_plus and h_cross
    # Setting the start_time is important! (too late, too early errors are because of this)
    h_plus = TimeSeries(h_plus, delta_t=1./sample_rate)
    h_cross = TimeSeries(h_cross, delta_t=1./sample_rate)
    
    # Use project_wave and random realisation of polarisation angle, ra, dec to obtain augmented signal
    pol_angle = injection_values['polarization']
    strains = [det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle, method='constant') for det in dets]
    time_interval = (start_interval, end_interval)
    # Put both strains together
    pure_sample = [strain.time_slice(*time_interval, mode='nearest') for strain in strains]
    
    # Calculate the SNR of the given pure sample with the appropriate PSD
    # NOTE: PSD realisation is given as optional within the sigmasq pycbc module
    PSDs = {}
    data_loc = os.path.join(data_cfg.parent_dir, data_cfg.data_dir)
    psd_files = glob.glob(os.path.join(data_loc, "psds/*"))
    for psd_file in psd_files:
        with h5py.File(psd_file, 'r') as fp:
            data = np.array(fp['data'])
            delta_f = fp.attrs['delta_f']
            name = fp.attrs['name']
            
        psd_data = FrequencySeries(data, delta_f=delta_f)
        # Store PSD data into lookup dict
        PSDs[name] = psd_data
    
    if data_cfg.dataset == 1:
        psds_data = [PSDs['aLIGOZeroDetHighPower']]*2
    else:
        psds_data = [PSDs['median_det1'], PSDs['median_det2']]
    
    # Calculation of SNR
    snr = np.sqrt(sum([sigmasq(strain, psd=psd, low_frequency_cutoff=data_cfg.noise_low_freq_cutoff) 
                       for strain, psd in zip(pure_sample, psds_data)]))
    
    return snr


def get_snrs(injection_file, data_cfg, dataset_dir):
    # Calculate the SNRs of all testing dataset injections
    # Current dataset only has to deal with an ~96,000 signal subset
    injparams = {}
    with h5py.File(injection_file, 'r') as fp:
        params = list(fp.keys())
        for param in params:
            injparams[param] = fp[param][()]
    
    injlen = len(injparams['tc'])
    # Add injection times into injparams
    injparams['injection_time'] = injparams['tc']
    injparams['tc'] = np.random.uniform(18.0, 18.2, injlen)

    snrs = []
    # Multiprocessing SNR calculation
    print("Starting MP based SNR Calculation")
    
    # Create kwargs for input to the signal generation code
    injection_values = lambda n: {param:value[n] for param, value in injparams.items()}
    with mp.Pool(processes=64) as pool:
        with tqdm(total=injlen) as pbar:
            pbar.set_description("MP-SNR Calculation")
            for snr in pool.imap_unordered(get_injection_snr, [(injection_values(n), data_cfg) for n in range(injlen)]):
                snrs.append(snr)
                pbar.update()

    # Update injparams with the SNR values
    injparams['snr'] = snrs
   
    # Save all SNRs within the dataset directory as a .hdf file
    with h5py.File(os.path.join(dataset_dir, "snr.hdf"), 'a') as ds:
        ds.create_dataset('snr', data=injparams['snr'])
    
    return injparams['snr']
