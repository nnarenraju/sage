# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Mon Mar 14 22:32:51 2022

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
import numpy as np

# Plotting
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 22})

def figure(title=""):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(4, 2, figsize=(26.0, 30.0))
    fig.suptitle(title, fontsize=28, y=0.92)
    return axs

def _plot(ax, x=None, y=None, c=None, xlabel="", ylabel="", label="", ls='solid'):
    ax.plot(x, y, c=c, label=label, linewidth=3.0, ls=ls)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label!="":
        ax.legend()
    

def plot_unit(pure_signal, pure_noise, noisy_signal, trans_pure_signal, trans_noisy_signal,
              mass_1, mass_2, network_snr, sample_rate, save_path, data_dir, idx):
    
    # Plotting a unit from the MLMDC1 datasets object just before training
    mass_1 = np.around(mass_1, 3)
    mass_2 = np.around(mass_2, 3)
    network_snr = np.around(network_snr, 3)
    ax = figure(title="{}: m1={}, m2={}, snr={}".format(data_dir, mass_1, mass_2, network_snr))
    
    """ Pure Signal """
    # Plotting each sample variety
    assert len(list(set([len(signal) for signal in pure_signal]))) == 1
    end_time = len(pure_signal[0])/sample_rate
    time_axis = np.linspace(0.0, end_time, len(pure_signal[0]))
    _plot(ax[0][0], time_axis, pure_signal[0], c='r', xlabel="Time [s]", ylabel="Strain", label="Pure Signal H1")
    _plot(ax[0][1], time_axis, pure_signal[1], c='b', xlabel="Time [s]", ylabel="Strain", label="Pure Signal L1")

    
    """ Pure Noise """
    assert len(list(set([len(signal) for signal in pure_noise]))) == 1
    end_time = len(pure_noise[0])/sample_rate
    time_axis = np.linspace(0.0, end_time, len(pure_noise[0]))
    _plot(ax[1][0], time_axis, pure_noise[0], c='k', xlabel="Time [s]", ylabel="Strain", label="Pure Noise H1")
    _plot(ax[1][1], time_axis, pure_noise[1], c='k', xlabel="Time [s]", ylabel="Strain", label="Pure Noise L1")
    
    
    """ Noisy Signal and Pure Signal """
    assert len(list(set([len(signal) for signal in noisy_signal]))) == 1
    end_time = len(noisy_signal[0])/sample_rate
    time_axis = np.linspace(0.0, end_time, len(noisy_signal[0]))
    _plot(ax[2][0], time_axis, noisy_signal[0], c='k', xlabel="Time [s]", ylabel="Strain", label="Noisy Signal H1")
    _plot(ax[2][1], time_axis, noisy_signal[1], c='k', xlabel="Time [s]", ylabel="Strain", label="Noisy Signal L1")
    
    end_time = len(pure_signal[0])/sample_rate
    time_axis = np.linspace(0.0, end_time, len(pure_signal[0]))
    _plot(ax[2][0], time_axis, pure_signal[0], c='r', xlabel="Time [s]", ylabel="Strain")
    _plot(ax[2][1], time_axis, pure_signal[1], c='b', xlabel="Time [s]", ylabel="Strain")
    
    
    """ Transformed Noisy Signal and Transformed Pure Signal """
    assert len(list(set([len(signal) for signal in trans_noisy_signal]))) == 1
    end_time = len(trans_noisy_signal[0])/sample_rate
    time_axis = np.linspace(0.0, end_time, len(trans_noisy_signal[0]))
    _plot(ax[3][0], time_axis, trans_noisy_signal[0], c='k', xlabel="Time [s]", ylabel="Strain", label="Transformed Noisy Signal H1")
    _plot(ax[3][1], time_axis, trans_noisy_signal[1], c='k', xlabel="Time [s]", ylabel="Strain", label="Transformed Noisy Signal L1")
    
    assert len(list(set([len(signal) for signal in trans_pure_signal]))) == 1
    end_time = len(trans_pure_signal[0])/sample_rate
    time_axis = np.linspace(0.0, end_time, len(trans_pure_signal[0]))
    _plot(ax[3][0], time_axis, trans_pure_signal[0], c='r', xlabel="Time [s]", ylabel="Strain")
    _plot(ax[3][1], time_axis, trans_pure_signal[1], c='b', xlabel="Time [s]", ylabel="Strain")    

    
    # Saving plots
    save_dir = os.path.join(save_path, "SAMPLES")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    save_path = os.path.join(save_dir, "sample_idx_{}.png".format(idx))
    plt.savefig(save_path)
    plt.close()
