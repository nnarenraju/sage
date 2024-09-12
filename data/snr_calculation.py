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

# Packages
import numpy as np

# PyCBC
from pycbc.types import TimeSeries
from pycbc.filter.matchedfilter import sigmasq


def get_network_snr(signals, psds_data, params, save_dir, debug):
    # Convert signals to TimeSeries object
    signals = [TimeSeries(signal, delta_t=1./params['sample_rate']) for signal in signals]
    assert len(list(set([len(signal) for signal in signals]))) == 1
    # Calculation of SNR
    network_snr = np.sqrt(sum([sigmasq(strain, low_frequency_cutoff=20.0) for strain in signals]))
    return network_snr
