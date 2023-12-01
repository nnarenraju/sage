# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sat Mar 26 22:39:13 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
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
import csv
import numpy as np

# PyCBC
from pycbc.psd import interpolate
from pycbc.types import TimeSeries
from pycbc.filter.matchedfilter import sigmasq


def calculate_network_snr(strains, psds, noise_low_freq_cutoff):
    # SNR Calculation
    return np.sqrt(sum([sigmasq(strain, psd=psd, low_frequency_cutoff=noise_low_freq_cutoff)
                        for strain, psd in zip(strains, psds)]))

def get_network_snr(signals, psds_data, params, save_dir, debug):
    # pure_sample, params, data_loc
    
    """ TimeSeries Signals """
    # Convert signals to TimeSeries object
    signals = [TimeSeries(signal, delta_t=1./params['sample_rate']) for signal in signals]
    
    """ Change delta_f of PSD to align with signals """
    # Calculating delta_f of signal and providing that to the PSD interpolation method
    # if delta_f is not equal between PSD and signal, it raises an exception
    assert len(list(set([len(signal) for signal in signals]))) == 1
    sample_length_in_s = len(signals[0])/params['sample_rate']
    delta_f = 1./sample_length_in_s
    # Interpolate the PSD to the required delta_f
    psd_data = [interpolate(psd, delta_f) for psd in psds_data]
    
    """ Calculation of SNR """
    network_snr = calculate_network_snr(signals, psd_data, params['noise_low_freq_cutoff']) 
    
    return network_snr
