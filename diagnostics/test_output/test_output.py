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

file_path = "output.hdf"

with h5py.File(file_path, "r") as foo:
    # Attributes of file (stat, time, var)
    time = np.array(foo['time'])

def figure():
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(12.0, 12.0))
    fig.suptitle("Loss Curve (1e4 training, 1e3 validation)", fontsize=28, y=0.95)
    return axs

def _plot(ax, x, y1, y2, xlabel="Epochs", ylabel="Losses", ls='solid'):
    ax.plot(x, y1, ls=ls, c='k', linewidth=3.0, label="Training Loss")
    ax.plot(x, y2, ls=ls, c='red', linewidth=3.0, label="Validation Loss")
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
        
if __name__ == "__main__":
    
    losses = pd.read_csv("losses.csv")
    epochs = losses['epoch']
    train_loss = losses['loss']
    val_loss = losses['val_loss']
    # Plotting
    ax = figure()
    _plot(ax, epochs, train_loss, val_loss)
