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
import re
import h5py
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 22})

def _figure(name):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    fig, axs = plt.subplots(3, 2, figsize=(22.0, 26.0))
    fig.suptitle(f"{name}", fontsize=26, y=0.95)
    return axs

def _plot(ax, x, y1, c=None, label=None, signal=False):
    # ax.plot(x, y1, c=c, linewidth=3.0, ls='solid', label=label)
    ax.scatter(x, y1, c=c, label=label, marker='x')
    ax.grid(True, which='both')
    if not signal:
        ax.set_xlabel("GPS Time [s]")
    else:
        ax.set_xlabel("Time [s]")
    
    ax.legend()
    ax.set_ylabel("Strain")

def _get_data(path, signal=False):
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
        attrs = detector_group_1[times_1[0]].attrs
        # Common time axis for all data
        start_time = int(times_1[0])
        sample_rate = 1.0/attrs['delta_t']
        if not signal:
            end_time_1 = start_time + (len(data_1)//sample_rate)
            end_time_2 = start_time + (len(data_2)//sample_rate)
            time_axis_1 = np.linspace(start_time, end_time_1, len(data_1), dtype=np.float64)
            time_axis_2 = np.linspace(start_time, end_time_2, len(data_2), dtype=np.float64)
        else:
            end_time_1 = 0.0 + (len(data_1)//sample_rate)
            end_time_2 = 0.0 + (len(data_2)//sample_rate)
            time_axis_1 = np.linspace(0.0, end_time_1, len(data_1), dtype=np.float64)
            time_axis_2 = np.linspace(0.0, end_time_2, len(data_2), dtype=np.float64)
        
        return (data_1, data_2, time_axis_1, time_axis_2, dets)

def verify(dirs, check):
    """
    

    Parameters
    ----------
    data_dir : dict
        Signal, background, foreground and parent directory paths
    check : dict
        contains - ndata

    Returns
    -------
    None.

    """
    
    # Get the signal, background and foreground filenames
    signals = glob.glob(dirs['signal'] + "/signal_*")
    signals.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    backgrounds = sorted(glob.glob(dirs['background'] + "/background_*"))
    backgrounds.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    foregrounds = sorted(glob.glob(dirs['foreground'] + "/foreground_*"))
    foregrounds.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    injections = dirs['parent'] + "/injections.hdf"
    
    # Read the injections file and obtain 'tc'
    with h5py.File(injections, "r") as foo:
        tc = np.sort(np.array(foo['tc']))
    
    # Choose a random 'n' number of datasets to visualise 
    idxs = random.sample(range(len(foregrounds)), check['ndata'])
    # Create plots of those random idxs in a separate verify directory
    for idx in idxs:
        # Set figure
        ax = _figure(f"Strain in H1 & L1, Time of coalescence = {tc[idx]}")
        # Read data from signal, background, foreground and injections
        signal_1, signal_2, time_signal_1, time_signal_2, dets_signal = _get_data(signals[idx], signal=True)
        noise_1, noise_2, time_noise_1, time_noise_2, dets_noise = _get_data(backgrounds[idx])
        combined_1, combined_2, time_combined_1, time_combined_2, dets_comb = _get_data(foregrounds[idx])
            
        # Artificial signal data
        _plot(ax[0][0], time_signal_1, signal_1, c="r", label=f"{dets_signal[0]} signal", signal=True)
        _plot(ax[0][1], time_signal_2, signal_2, c="b", label=f"{dets_signal[1]} signal", signal=True)
        
        # Noise data
        _plot(ax[1][0], time_noise_1, noise_1, c="k", label=f"{dets_noise[0]} noise")
        _plot(ax[1][1], time_noise_2, noise_2, c="k", label=f"{dets_noise[1]} noise")
        
        # Overplot combined and signal
        _plot(ax[2][0], time_combined_1, combined_1, c="k", label=f"{dets_comb[0]} foreground")
        _plot(ax[2][1], time_combined_2, combined_2, c="k", label=f"{dets_comb[1]} foreground")
        # _plot(ax[2][0], time_signal_1, signal_1, c="r", label=f"{dets_signal[0]} signal")
        # _plot(ax[2][1], time_signal_2, signal_2, c="b", label=f"{dets_signal[1]} signal")
        
        # Close and save plot
        save_path = dirs['parent'] + f"/verification/signals/verify_signal_{idx}.png"
        plt.savefig(save_path)
        plt.close()
