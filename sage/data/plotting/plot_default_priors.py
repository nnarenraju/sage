# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Mar 10 18:14:45 2022

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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 22})

def _figure(title=""):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(16.0, 14.0))
    fig.suptitle(title, fontsize=28, y=0.92)
    return axs

def _plot(ax, y=None, xlabel="", label=""):
    
    ax.hist(y, bins=100, label=label)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.legend()


def plot_priors(data_dir):
    
    # Read the priors from CSV data
    data_path = os.path.join(data_dir, 'injections/injections.csv')
    dataset_name = os.path.normpath(data_dir).split(os.path.sep)[-1]
    priors = pd.read_csv(data_path)
    names = list(priors.columns.values)
    
    # Plotting each prior distribution histogram
    for name in names:
        data = priors[name]
        try:
            _ = max(data)
        except:
            data = np.array([foo[0] for foo in data])
        
        ax = _figure("{}: {} prior histogram".format(dataset_name, name))
        _plot(ax, data, label="min={}, max={}, median={}".format(np.around(np.min(data),3), np.around(np.max(data),3), np.around(np.median(data),3)), xlabel=name)
        save_path = data_path = os.path.join(data_dir, 'injections/priors_{}.png'.format(name))
        plt.savefig(save_path)
        plt.close()
