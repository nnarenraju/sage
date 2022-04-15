# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Dec  7 16:08:38 2021

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
import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 18})

"""
file_path = "output.hdf"

with h5py.File(file_path, "r") as foo:
    # Attributes of file (stat, time, var)
    time = np.array(foo['time'])
"""

def figure():
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(16.0, 14.0))
    fig.suptitle("Loss Curves (different scenarios)", fontsize=28, y=0.95)
    return axs

def _plot(ax, x, y1, y2, xlabel="Epochs", ylabel="BCE Loss", ls='solid', label="", 
          once=False, c=None, accuracy=None):
    ax.plot(x, y1, ls=ls, c=c, linewidth=3.0, label=label)
    if accuracy is not None:
        ax.plot(x, accuracy, ls='dashdot', c='k', linewidth=3.0, label="Max Accuracy")
    if once:
        ax.plot(x, y2, ls='dashed', c='red', linewidth=3.0, label="Validation Loss")
    else:
        ax.plot(x, y2, ls='dashed', c=c, linewidth=3.0)
        
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
        
if __name__ == "__main__":
    
    loss_control = np.loadtxt("losses.csv", delimiter=",")
    loss_testing_d1 = np.loadtxt("losses_1e4_2s_dataset1.txt")
    loss_dist = np.loadtxt("losses_1e4_training_2e3_validation_chirp_distance_130L_350U.txt")
    loss_xphm = np.loadtxt("losses_1e4_training_2e3_validation_IMRPhenomXPHM.txt")
    loss_psd = np.loadtxt("losses_1e4_training_2e3_validation_testing_PSD.txt")
    loss_large = np.loadtxt("losses_1e5_training_2e4_validation_IMRPhenomD.txt")
    loss_lowmass = np.loadtxt("losses_1e4_training_2e3_validation_7Msun_lower.txt")
    loss_close_d1 = np.loadtxt("losses_close_to_dataset1.txt")
    loss_closest_d1 = np.loadtxt("losses_dataset1_realistic_march3.txt")
    loss_closest_d1_rmModeArray = np.loadtxt("losses_closest_plotting_experiment.txt")
    
    num_epochs = 200
    epochs = loss_closest_d1_rmModeArray[:,0][:num_epochs]
    
    # Plotting
    ax = figure()
    
    ## All loss curves
    
    """
    # Control
    train_loss = loss_control[:,1][:num_epochs]
    val_loss = loss_control[:,2][:num_epochs]
    _plot(ax, epochs, train_loss, val_loss, label="Example IMRPhenomD", once=True, c='r')
    """
    
    # Testing dataset 1
    train_loss = loss_testing_d1[:,1][:100]
    val_loss = loss_testing_d1[:,2][:100]
    _plot(ax, epochs[:100], train_loss, val_loss, label="Testing dataset 1", c='orange')
    
    
    """
    # Testing dataset 1 - close
    train_loss = loss_close_d1[:,1][:num_epochs]
    val_loss = loss_close_d1[:,2][:num_epochs]
    _plot(ax, epochs, train_loss, val_loss, label="Explicit PyCBC close to dataset 1", c='m')
    """
    
    """
    # Testing dataset 1 - closest March 3rd
    train_loss = loss_closest_d1[:,1][:num_epochs]
    val_loss = loss_closest_d1[:,2][:num_epochs]
    _plot(ax, epochs, train_loss, val_loss, label="ePyCBC closest to dataset 1", c='k')
    """
    
    # Testing dataset 1 - closest March 3rd rmModeArray
    train_loss = loss_closest_d1_rmModeArray[:,1][:num_epochs]
    val_loss = loss_closest_d1_rmModeArray[:,2][:num_epochs]
    train_accuracy = loss_closest_d1_rmModeArray[:,3][:num_epochs]
    _plot(ax, epochs, train_loss, val_loss, label="ePyCBC closest to dataset 1", c='red', once=True)
    
    
    
    """
    # Distance
    train_loss = loss_dist[:,1][:num_epochs]
    val_loss = loss_dist[:,2][:num_epochs]
    _plot(ax, epochs, train_loss, val_loss, label="chirp distance prior [130, 150]", c='k')
    """
    
    """
    # IMRPhenomXPHM
    train_loss = loss_xphm[:,1][:num_epochs]
    val_loss = loss_xphm[:,2][:num_epochs]
    _plot(ax, epochs, train_loss, val_loss, label="Example IMRPhenomXPHM", c='green')
    
    # Testing data PSD for dataset 1
    train_loss = loss_psd[:,1][:num_epochs]
    val_loss = loss_psd[:,2][:num_epochs]
    # _plot(ax, epochs, train_loss, val_loss, label="Testing data PSD: dataset 1 on Control", c='green', once=True)
    
    # Larger dataset control
    train_loss = loss_large[:,1][:num_epochs]
    val_loss = loss_large[:,2][:num_epochs]
    _plot(ax, epochs, train_loss, val_loss, label="Example IMRPhenomD: 1e5 train", c='b')
    """
    
    """
    # Lower mass prior
    train_loss = loss_lowmass[:,1][:num_epochs]
    val_loss = loss_lowmass[:,2][:num_epochs]
    _plot(ax, epochs, train_loss, val_loss, label="Low mass prior = 7Msun (old:10Msun)", c='c')
    """
    
    
    
    
