# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Wed Mar 23 13:35:51 2022

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
import h5py
import numpy as np

import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 22})

def figure(title=""):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(16.0, 14.0))
    fig.suptitle(title, fontsize=28, y=0.92)
    return axs

def _plot(ax, x=None, y=None, xlabel="Epochs", ylabel="BCE Loss", ls='solid', 
          label="", c=None, scatter=False, hist=False, yscale='linear'):
    
    if scatter:
        ax.scatter(x, y, c=c, label=label)
    elif hist:
        ax.hist(y, bins=100, label=label, alpha=0.8)
    else:
        ax.plot(x, y, ls=ls, c=c, linewidth=3.0, label=label)
    
    ax.set_yscale(yscale)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

def get_data(file_path):
    # Get information about output
    with h5py.File(file_path, "r") as foo:
        # Attributes of file
        # Fields
        print("Fields in output: {}".format(list(foo.keys())))
        raise
        fields = list(foo.keys())
        stat = np.array(foo[fields[0]])
        time = np.array(foo[fields[1]])
        var = np.array(foo[fields[2]])
        
        print(set(stat))
        raise

def get_injections(path):
    # Get information about added injections
    with h5py.File(path, "r") as foo:
        # Attributes of file
        print("\n================ START ================\n")
        for attr in list(foo.attrs.keys()):
            print(f"{attr}: {foo.attrs[attr]}")
        
        print("\nGroups = {}".format(list(foo.keys())))
        lendata = len(np.array(foo['tc']))
        
        print(np.array(foo['tc']))
        for field in list(foo.keys()):
            print(f"{field} = {foo[field][0]}")
        print("\nTotal number of injections present = {}".format(lendata))
        print("\nDifferences between times of coalescence in injections (first 5):")
        sorted_tc = np.sort(foo['tc'])
        
        start_time = 1238205077
        end_time = start_time + 32000
        tcs = sorted_tc[sorted_tc<end_time]
        print(len(tcs))
        raise
        
        diff = np.sort(sorted_tc[1:] - sorted_tc[:-1])
        for n, tdiff in enumerate(diff[:10]):
            print(f"tc diff b/w signal {n+2} and {n+1} is {tdiff}")
        
        print("\n================ EOF ================\n\n")

if __name__ == "__main__":
    
    directory = "./"
    foreground_path = directory + "output_32000.hdf"
    injections_path = directory + "injections.hdf"
    eval_path = directory + "eval-output.hdf"
    # get_data(foreground_path)
    # get_injections(injections_path)
    get_data(eval_path)
