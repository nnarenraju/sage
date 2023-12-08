#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Mar 21 15:03:57 2023

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
import pickle
import h5py
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

# Prettification
from tqdm import tqdm


def figure(title=""):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(16.0, 14.0))
    fig.suptitle(title, fontsize=18, y=0.92)
    return axs, fig


def _plot(ax, x=None, y=None, xlabel="x-axis", ylabel="y-axis", ls='solid', 
          label="NULL", c=None, yscale='linear', xscale='linear', histogram=False):
    
    # Plotting type
    if histogram:
        ax.hist(y, bins=100, label=label, alpha=0.8)
    else:
        ax.plot(x, y, ls=ls, c=c, linewidth=3.0, label=label)
    
    # Plotting params
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    

def find_injection_times(fgfiles, injfile, padding_start=30, padding_end=30):
    # Determine injections which are contained in the file
    duration = 0
    times = []
    with h5py.File(fgfiles, 'r') as fp:
        det = list(fp.keys())[0]
        
        for key in fp[det].keys():
            ds = fp[f'{det}/{key}']
            start = ds.attrs['start_time']
            end = start + len(ds) * ds.attrs['delta_t']
            duration += end - start
            start += padding_start
            end -= padding_end
            if end > start:
                times.append([start, end])
    
    with h5py.File(injfile, 'r') as fp:
        injtimes = fp['tc'][()]
    
    # All injection times that are present within a given start and end 
    # are marked with True and rest of the injtimes are marked with False
    ret = np.full((len(times), len(injtimes)), False)
    for i, (start, end) in enumerate(times):
        ret[i] = np.logical_and(start <= injtimes, injtimes <= end)
    
    # np.any will compress the 2D array above and give a 1D array which has
    # True for all injection times that are present within the valid regions
    # and False for all injection times that are not present in the valid regions
    return duration, np.any(ret, axis=0)


def find_nclosest_index(trigger_times, injection_times, frange=[-0.3, +0.3]):
    # Find nclosest triggers given an injection time
    if len(trigger_times) == 0:
        raise ValueError('Cannot find closest index for empty trigger times array.')
    
    # Sort array before proceeding
    array = trigger_times.copy()
    array.sort()
    
    # Find nclosest indices for the given value in array
    tidxs = [np.argwhere((array >= inj + frange[0]) & (array <= inj + frange[1])) for inj in tqdm(injection_times)]
    return tidxs


def read_network_output(fgpath, bgpath):
    with h5py.File(fgpath, 'r') as fp:
        fg_events = [np.vstack([fp['time'], fp['stat'], fp['var']])]
        print(len(fp['time']))
    fg_events = np.concatenate(fg_events, axis=-1)
    sidxs = fg_events[0].argsort()
    fg_events = fg_events.T[sidxs].T
    
    # Read background events
    with h5py.File(bgpath, 'r') as fp:
        bg_events = [np.vstack([fp['time'], fp['stat'], fp['var']])]
    bg_events = np.concatenate(bg_events, axis=-1)
    sidxs = bg_events[0].argsort()
    bg_events = bg_events.T[sidxs].T

    return fg_events, bg_events


def ROC(injections_file, foreground_file, teams):
    """ Calculate the ROC Curve and AUC by calculating TPR and FPR """
    # Find all valid injection times for the given testing dataset
    duration, valid_idxs = find_injection_times(foreground_file, injections_file)
    # Read the injections file and get the values of injection params
    print("Reading the injections file for testing dataset")
    injparams = {}
    with h5py.File(injections_file, "r") as foo:
        # Attributes of file
        print("Injections Metadata:")
        for attr in list(foo.attrs.keys()):
            print(f"{attr}: {foo.attrs[attr]}")
        # Get all parameters
        params = list(foo.keys())
        # Get the data for for all injection params
        for param in params:
            data = foo[param][()]
            injparams[param] = data[valid_idxs]
    
    # Plotting routine
    ax, _ = figure(title="ROC Curve for Testing Dataset")

    for tname in teams.keys():
        print("\nComputing ROC Curve for {}".format(tname))
        fg_events = teams[tname]['fg_events']
        bg_events = teams[tname]['bg_events']

        ## Outputs and Labels for FG and BG
        outputs = []
        labels = []
        
        """ Foreground file analysis """
        print("\nAnalysing the foreground data")
        # Read the foreground output and get all the trigger times found
        fg_trigger_times = fg_events[0]
        fg_trigger_stats = fg_events[1]

        # Given all the valid injection times present in the injections file
        # get idx of all triggers that are within a bound of the given injection time
        print("Getting closest trigger indices for each valid injection time")
        fg_bound_triggers_idx = find_nclosest_index(trigger_times=fg_trigger_times, injection_times=injparams['tc'])
        
        print("Maximising over each set of triggers and obtaining the output and labels for ROC curve")
        ## Get outputs and labels to get ROC curve
        pbar = tqdm(fg_bound_triggers_idx)
        for btidx in pbar:
            pbar.set_description("Maximising FG triggers")
            if len(btidx) > 0:
                # Maximise on these triggers to get the event
                outputs.append(np.max(fg_trigger_stats[btidx]))
            else:
                # If there are no triggers given, then we assign 0.0 stat for event
                outputs.append(-np.inf)
            # Append labels as 1.0 since only GW events are looked at
            labels.append(1.0)
        
        """ Background file analysis """
        print("\nAnalysing the background data")
        # Read the foreground output and get all the trigger times found
        bg_trigger_times = bg_events[0]
        bg_trigger_stats = bg_events[1]
        
        # Given all the valid injection times present in the injections file
        # get idx of all triggers that are within a bound of the given injection time
        print("Getting closest trigger indices for each valid injection time")
        bg_bound_triggers_idx = find_nclosest_index(trigger_times=bg_trigger_times, injection_times=injparams['tc'])
        
        print("Maximising over each set of triggers and obtaining the output and labels for ROC curve")
        ## Get outputs and labels to get ROC curve
        pbar = tqdm(bg_bound_triggers_idx)
        for btidx in pbar:
            pbar.set_description("Maximising BG triggers")
            if len(btidx) > 0:
                # Maximise on these triggers to get the event
                outputs.append(np.max(bg_trigger_stats[btidx]))
            else:
                # If there are no triggers given, then we assign 0.0 stat for event
                outputs.append(-np.inf)
            # Append labels as 0.0 since only noise data is looked at
            labels.append(0.0)

        """ Calculating the ROC Curve """
        outputs = np.array(outputs)
        labels = np.array(labels)
        # Convert the outputs from raw outputs to prediction probabilites using sigmoid
        sigmoid = lambda x: 1./(1. + np.exp(-1*x))
        outputs = sigmoid(outputs)
        # Calculate the ROC curve
        fpr, tpr, threshold = metrics.roc_curve(labels, outputs)
        roc_auc = metrics.auc(fpr, tpr)
    
        # Log ROC Curve
        if tname == "PyCBC":
            colour = 'red'
        elif tname == "ORChiD":
            colour = "blue"

        _plot(ax, fpr, tpr, label="{}, AUC = {}".format(tname, np.around(roc_auc, 4)), c=colour, 
              ylabel="True Positive Rate", xlabel="False Positive Rate", 
              yscale='linear', xscale='linear')

    # Other plotting stuff
    _plot(ax, [0, 1], [0, 1], label="Random Classifier", c='k', 
          ylabel="True Positive Rate", xlabel="False Positive Rate", 
          ls="dashed", yscale='linear', xscale='linear')
    
    save_path = os.path.join("./testset_roc_curve.png")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    
    ## Parent directory
    parent_dir = "/local/scratch/igr/nnarenraju/testing_month_D3_seeded"
    
    ## Foreground and background dataset outputs from pipeline
    teams = {'PyCBC': {'fg_events': None, 'bg_events': None}, 
             'ORChiD': {'fg_events': None, 'bg_events': None}}

    # Paths
    fgpath = ["/home/nnarenraju/Research/results/PyCBC/ds3/fg.hdf", os.path.join(parent_dir, "ref/testing_foutput.hdf")]
    bgpath = ["/home/nnarenraju/Research/results/PyCBC/ds3/bg.hdf", os.path.join(parent_dir, "ref/testing_boutput.hdf")]
    # Get data from the outputs files
    for fgp, bgp, tname in zip(fgpath, bgpath, teams.keys()):
        fg_events, bg_events = read_network_output(fgp, bgp)
        teams[tname]['bg_events'] = bg_events
        teams[tname]['fg_events'] = fg_events
    
    ## Injections file and foreground file data
    injfile = os.path.join(parent_dir, "injections.hdf")
    fgfile = os.path.join(parent_dir, "foreground.hdf")
    
    ## Get the ROC curve for the testing dataset
    ROC(injfile, fgfile, teams)
    
    print("\nFIN")
