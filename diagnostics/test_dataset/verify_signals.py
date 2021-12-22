# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Dec 17 15:34:00 2021

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

# IN-BUILT
import os
import h5py
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 22})

def _figure(name):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(3, 2, figsize=(20.0, 24.0))
    fig.suptitle(f"{name}", fontsize=20, y=0.95)
    return axs

def _plot(ax, x, y1, c=None, label=None):
    ax.plot(x, y1, c=c, linewidth=3.0, ls='solid', label=label)
    ax.grid(True, which='both')
    ax.set_xlabel("GPS Time [s]")
    ax.set_ylabel("Strain")

def _get_data(path, sample_rate):
    # Read data from HDF5 files
    with h5py.File(path, "r") as sigfile:
        ## Reading all data parameters
        # Detectors
        dets = list(sigfile.keys())
        # Groups within detectors (times as dict)
        detector_group_1 = sigfile[dets[0]]
        detector_group_2 = sigfile[dets[1]]
        # Times as list
        times_1 = list(detector_group_1.keys())
        times_2 = list(detector_group_2.keys())
        # Data within each detector
        data_1 = np.array(detector_group_1[times_1[0]])
        data_2 = np.array(detector_group_2[times_2[0]])
        # Common time axis for all data
        start_time = times_1[0]
        end_time = start_time + (len(data_1)//sigfile.attrs['sample_rate'])
        time_axis = np.linspace(start_time, end_time, len(data_1), dtype=np.int32)
        return (data_1, data_2, time_axis, dets)

def verify(dirs, check):
    """
    

    Parameters
    ----------
    data_dir : dict
        Signal, background, foreground and parent directory paths
    check : dict
        contains - ndata, sample_rate

    Returns
    -------
    None.

    """
    
    # Get the signal, background and foreground filenames
    signals = glob.glob(dirs['signal'] + "/signal_*")
    backgrounds = glob.glob(dirs['background'] + "/background_*")
    foregrounds = glob.glob(dirs['foreground'] + "/foreground_*")
    injections = dirs['parent'] + "/injections.hdf"
    
    # Read the injections file and obtain 'tc'
    with h5py.File(injections, "r") as foo:
        tc = np.array(foo['tc'])
    
    # Choose a random 'n' number of datasets to visualise 
    idxs = random.sample(range(len(foregrounds)), check['ndata'])
    # Create plots of those random idxs in a separate verify directory
    for idx in idxs:
        # Set figure
        ax = _figure(f"Strain in H1 & L1, Time of coalescence = {tc[idx]}")
        # Read data from signal, background, foreground and injections
        sr = check['sample_rate']
        signal_1, signal_2, time_signal, dets_signal = _get_data(signals[idx], sample_rate=sr)
        noise_1, noise_2, time_noise, dets_noise = _get_data(backgrounds[idx], sample_rate=sr)
        combined_1, combined_2, time_combined, dets_comb = _get_data(foregrounds[idx], sample_rate=sr)
            
        # Artificial signal data
        _plot(ax[0][0], time_signal, signal_1, c="r", label=f"{dets_signal[0]} signal")
        _plot(ax[0][1], time_signal, signal_2, c="b", label=f"{dets_signal[1]} signal")
        
        # Noise data
        _plot(ax[1][0], time_noise, noise_1, c="k", label=f"{dets_noise[0]} noise")
        _plot(ax[1][1], time_noise, noise_2, c="k", label=f"{dets_noise[1]} noise")
        
        # Overplot combined and signal
        _plot(ax[2][0], time_combined, combined_1, c="k", label=f"{dets_comb[0]} foreground")
        _plot(ax[2][1], time_combined, combined_2, c="k", label=f"{dets_comb[1]} foreground")
        _plot(ax[2][0], time_signal, signal_1, c="r", label=f"{dets_signal[0]} signal")
        _plot(ax[2][1], time_signal, signal_2, c="b", label=f"{dets_signal[1]} signal")
        
        # Close and save plot
        save_path = dirs['parent'] + f"/verification/signals/verify_signal_{idx}.png"
        plt.savefig(save_path)
        plt.close()
