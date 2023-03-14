#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = A program to calculate the false-alarm rate as well as the sensitive
                  distance from a search algorithm. (Part of the MLGWSC-1)

Created on Mon Aug  8 10:22:49 2022

__author__      = nnarenraju
__copyright__   = Copyright 2022, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation:
    
    ./evaluate.py \
    --injection-file injections.hdf \
    --foreground-events <path to output of algorithm on foreground data> \
    --foreground-files foreground.hdf \
    --background-events <path to output of algorithm on background data> \
    --output-file eval-output.hdf \
    --output-dir ./ \
    --verbose

    python3 evaluate.py --injection-file /local/scratch/igr/nnarenraju/testing_64000_D3_seeded/injections.hdf \
    --foreground-events /local/scratch/igr/nnarenraju/testing_64000_D3_seeded/testing_foutput.hdf \
    --foreground-files /local/scratch/igr/nnarenraju/testing_64000_D3_seeded/foreground.hdf \
    --background-events /local/scratch/igr/nnarenraju/testing_64000_D3_seeded/testing_boutput.hdf \
    --far-scaling-factor 64000 --dataset 3 --output-file ./TESTING_64000_D3/evaluation.hdf \
    --output-dir ./TESTING_64000_D3 \
    --team1 ORChiD --team2 PyCBC \
    --verbose

The options mean the following

    --injection-file specifies the injections that were used to create the foreground data. 
      It corresponds to the output of generate_data.py --output-injection-file or the path 
      given to generate_data.py --injection-file.
      
    --foreground-events specifies the output of the search algorithm that was obtained using 
      the foreground file returned by generate_data.py --output-foreground-file. For details 
      on the structure of these files please refer to this page. Multiple paths may be provided 
      if the input data was split into multiple parts.
      
    --foreground-files specifies the foreground data that was used as input to the algorithm. 
      This file is only used to determine which injections were actually contained in the 
      foreground data and how much data was analyzed. It has to be the file created by 
      generate_data.py --output-foreground-file. Multiple paths may be provided if the input 
      data was split into multiple parts.
      
    --background-events specifies the output of the search algorithm that was obtained using 
      the background file returned by generate_data.py --output-background-file. For details 
      on the structure of these files please refer to this page. Multiple paths may be provided 
      if the input data was split into multiple parts.
      
    --output-file specifies where the analysis output should be stored.
    --verbose tells the script to print status updates.
    --force exists to allow the code to overwrite existing files.


"""

# Modules
import os
import h5py
import logging
import argparse
import itertools
import numpy as np
from pathlib import Path

# LOCAL
from data_configs import Default as data_cfg
from utils.get_testdata_snr import get_snrs

# Plotting
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 18})

# Prettification
from tqdm import tqdm


def find_injection_times(fgfiles, injfile, padding_start=0, padding_end=0):
    """
    Determine injections which are contained in the file.
    
    Arguments
    ---------
    fgfiles : list of str
        Paths to the files containing the foreground data (noise +
        injections).
    injfile : str
        Path to the file containing information on the injections in the
        foreground files.
    padding_start : {float, 0}
        The amount of time (in seconds) at the start of each segment
        where no injections are present.
    padding_end : {float, 0}
        The amount of time (in seconds) at the end of each segment
        where no injections are present.
    
    Returns
    -------
    duration:
        A float representing the total duration (in seconds) of all
        foreground files.
    bool-indices:
        A 1D array containing bools that specify which injections are
        contained in the provided foreground files.
    """
    duration = 0
    times = []
    for fpath in fgfiles:
        with h5py.File(fpath, 'r') as fp:
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
    
    ret = np.full((len(times), len(injtimes)), False)
    for i, (start, end) in enumerate(times):
        ret[i] = np.logical_and(start <= injtimes, injtimes <= end)
    
    return duration, np.any(ret, axis=0)


def find_closest_index(array, value, assume_sorted=False):
    """
    Find the index of the closest element in the array for the given
    value(s).
    
    Arguments
    ---------
    array : np.array
        1D numpy array.
    value : number or np.array
        The value(s) of which the closest array element should be found.
    assume_sorted : {bool, False}
        Assume that the array is sorted. May improve evaluation speed.
    
    Returns
    -------
    indices:
        Array of indices. The length is determined by the length of
        value. Each index specifies the element in array that is closest
        to the value at the same position.
    """
    if len(array) == 0:
        raise ValueError('Cannot find closest index for empty input array.')
    if not assume_sorted:
        array = array.copy()
        array.sort()
    ridxs = np.searchsorted(array, value, side='right')
    lidxs = np.maximum(ridxs - 1, 0)
    comp = np.fabs(array[lidxs] - value) < \
           np.fabs(array[np.minimum(ridxs, len(array) - 1)] - value)  # noqa: E127, E501
    lisbetter = np.logical_or((ridxs == len(array)), comp)
    ridxs[lisbetter] -= 1
    return ridxs


def mchirp(mass1, mass2):
    return (mass1 * mass2) ** (3. / 5.) / (mass1 + mass2) ** (1. / 5.)


def figure(title="", size_x=16.0, size_y=14.0):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(size_x, size_y))
    fig.suptitle(title, fontsize=28, y=0.92)
    return axs, fig


def _plot(ax, x=None, y=None, xlabel="x-axis", ylabel="y-axis", ls='solid', 
          label="", c=None, yscale='linear', xscale='linear', histogram=False,
          scatter=False, save_file=""):
    
    # Plotting type
    if histogram:
        ax.hist(y, bins=100, label=label, alpha=0.8)
    elif scatter:
        ax.scatter(x, y, marker='.', s=100.0)
    else:
        ax.plot(x, y, ls=ls, c=c, linewidth=3.0, label=label)
    
    # Plotting params
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label != "" or label != None:
        ax.legend()

    
    if save_file != "":
        plt.savefig(save_file)
        plt.close()


def param_vs_param(output_dir, injparams, found_injections):
    """ Plotting param vs param plots similar to MLGWSC-1 paper """
    
    vs_dir = os.path.join(output_dir, 'PARAM_VS_PARAM')
    if not os.path.exists(vs_dir):
        os.makedirs(vs_dir, exist_ok=False)
    
    # Plotting params
    plot_mchirp = injparams['mchirp'][found_injections[0].astype(int)]
    plot_distance = injparams['distance'][found_injections[0].astype(int)]
    plot_q = injparams['q'][found_injections[0].astype(int)]
    plot_dchirp = injparams['chirp_distance'][found_injections[0].astype(int)]
    
    # Signal duration
    lf = 20.0 # Hz
    G = 6.67e-11
    c = 3.0e8
    plot_signal_duration = 5. * (8.*np.pi*lf)**(-8./3.) * (plot_mchirp*1.989e30*G/c**3.)**(-5./3.)
    
    ## Other related plots
    ax, _ = figure('mchirp vs distance', 12.0, 12.0)
    spath = os.path.join(vs_dir, 'mchirp_vs_distance.png')
    _plot(ax, plot_mchirp, plot_distance, 'Chirp Mass', 'Distance', scatter=True, save_file=spath)
    
    ax, _ = figure('mchirp vs q', 12.0, 12.0)
    spath = os.path.join(vs_dir, 'mchirp_vs_q.png')
    _plot(ax, plot_mchirp, plot_q, 'Chirp Mass', 'Mass Ratio (m1/m2)', scatter=True, save_file=spath)
    
    ax, _ = figure('mchirp vs dchirp', 12.0, 12.0)
    spath = os.path.join(vs_dir, 'mchirp_vs_dchirp.png')
    _plot(ax, plot_mchirp, plot_dchirp, 'Chirp Mass', 'Chirp Distance', scatter=True, save_file=spath)
    
    ax, _ = figure('q vs dchirp', 12.0, 12.0)
    spath = os.path.join(vs_dir, 'q_vs_dchirp.png')
    _plot(ax, plot_q, plot_dchirp, 'Mass Ratio (m1/m2)', 'Chirp Distance', scatter=True, save_file=spath)
    
    # Signal duration plots
    ax, _ = figure('Tau_0 vs q', 12.0, 12.0)
    spath = os.path.join(vs_dir, 'tau0_vs_q.png')
    _plot(ax, plot_signal_duration, plot_q, 'Signal Duration [s]', 'Mass Ratio (m1/m2)', scatter=True, save_file=spath)
    
    ax, _ = figure('Tau_0 vs mchirp', 12.0, 12.0)
    spath = os.path.join(vs_dir, 'tau0_vs_mchirp.png')
    _plot(ax, plot_signal_duration, plot_mchirp, 'Signal Duration [s]', 'Chirp Mass', scatter=True, save_file=spath)
 
    ax, _ = figure('Tau_0 vs dchirp', 12.0, 12.0)
    spath = os.path.join(vs_dir, 'tau0_vs_dchirp.png')
    _plot(ax, plot_signal_duration, plot_dchirp, 'Signal Duration [s]', 'Chirp Distance', scatter=True, save_file=spath)
    
    ax, _ = figure('Tau_0 vs distance', 12.0, 12.0)
    spath = os.path.join(vs_dir, 'tau0_vs_distance.png')
    _plot(ax, plot_signal_duration, plot_distance, 'Signal Duration [s]', 'Distance', scatter=True, save_file=spath)


def found_param_plots(noise_stats, output_dir, injparams, found_injections):
    ### Get the thresholds for different false alarm rates
    # TODO: Add PyCBC's results overlayed on top
    # For a month-long testing dataset these should give FAR per month, per week and per day
    far_thresholds = noise_stats[::-1][[0, 3, 29, 99, 999]]
    thresh_names = ['1-per-month', '1-per-week', '1-per-day', '100-per-month', '1000-per-month']
    # How many signals are present above given threshold?
    far_found_idx = {thresh_names[n]: found_injections[0][found_injections[1] > thresh] for n, thresh in enumerate(far_thresholds)}
    
    ## Plotting the comparison plots (injections and found histogram) for all params 
    # cmap = cm.get_cmap('RdYlBu_r', 10)
    save_dir = os.path.join(output_dir, 'FOUND_INJECTIONS')
    for param in injparams.keys():
        param_dir = os.path.join(save_dir, '{}'.format(param))
        if not os.path.exists(param_dir):
            os.makedirs(param_dir, exist_ok=False)
            
        all_param = injparams[param]
        for key, value in far_found_idx.items():
            found_param = all_param[value.astype(int)]
            # Plotting the overlap histograms of all and found data
            plt.figure(figsize=(12.0, 12.0))
            plt.title('Injected vs Found (FAR = {}) - {}'.format(key, param))
            plt.hist(all_param, bins=100, label='{}-all'.format(param), alpha=0.8)
            plt.hist(found_param, bins=100, label='{}-found'.format(param), alpha=0.8)
            plt.grid(True, which='both')
            plt.xlabel('{}'.format(param))
            plt.ylabel('Number of Occurences')
            plt.legend()
            plt.savefig(os.path.join(param_dir, '{}-compare_FAR_{}.png'.format(param, key)))
            plt.close()
            

def network_output(found_injections, noise_stats, output_dir, team_name):
    # Plotting the noise and signals stats for found samples
    plt.figure(figsize=(12.0, 12.0))
    lower_threshold = 0.9999
    foo = found_injections[1][found_injections[1] > lower_threshold]
    plt.hist(foo, label='found_injections', bins=100, alpha=0.8)
    noise_stats = noise_stats[noise_stats > lower_threshold]
    plt.hist(noise_stats, label='noise', bins=100, alpha=0.8)
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'network_output_{}.png'.format(team_name)))
    plt.close()


def parameter_learning(injparams, noise_stats, found_injections, output_dir):
    ## Parameter learning
    learning_dir = os.path.join(output_dir, 'LEARNING')
    if not os.path.exists(learning_dir):
        os.makedirs(learning_dir, exist_ok=False)
    # Making the parameter learning plots
    source_params = {key: injparams[key][found_injections[0].astype(int)] for key in injparams.keys()}
    lf = 20.0 # Hz
    G = 6.67e-11
    c = 3.0e8
    source_params['signal_duration'] = 5. * (8.*np.pi*lf)**(-8./3.) * (source_params['mchirp']*1.989e30*G/c**3.)**(-5./3.)
    predicted_outputs = found_injections[1]
    save_name='raw_value'
    
    # Define FAR thresholds
    far_thresholds = noise_stats[::-1][[0, 3, 29, 99, 999]]
    thresh_names = ['1 per month', '1 per week', '1 per day', '100 per month', '1000 per month']
    for key in source_params.keys():
        # Sort the source_params for the particular key alongside the predicted outputs
        assert len(source_params[key]) == len(predicted_outputs)
        # Plotting the above data for the given parameter
        ax, fig = figure(title="Learning {}".format(key))
        _plot(ax, x=source_params[key], y=predicted_outputs, xlabel=key, ylabel=save_name, 
                  label=key, yscale='linear', xscale='linear', scatter=True)
        # Plotting FAR thresholds
        min_x = min(source_params[key])
        max_x = max(source_params[key])
        ax.set_xlim(min_x, max_x)
        for fthresh, nthresh in zip(far_thresholds, thresh_names):
            ax.plot([min_x, max_x], [fthresh, fthresh], label=nthresh, linewidth=2.0)
        # Saving the plot in export_dir
        save_path = os.path.join(learning_dir, 'learning_{}_{}.png'.format(save_name, key))
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def read_data(args, idxs):
    # Read injection parameters
    logging.info(f'Reading injections from {args.injection_file}')
    
    injparams = {}
    with h5py.File(args.injection_file, 'r') as fp:
        params = list(fp.keys())
        for param in params:
            data = fp[param][()]
            injparams[param] = data[idxs]
            
        use_chirp_distance = 'chirp_distance' in params
    
    team_1 = {'name': args.team1}
    team_2 = {'name': args.team2}
    other_results = "/home/nnarenraju/Research/results"
    other_teams = os.listdir(other_results)
    if args.team1 in other_teams:
        team_1['fgpath'] = [os.path.join(other_results, "{}/ds{}/fg.hdf".format(team_1['name'], args.dataset))]
        team_1['bgpath'] = [os.path.join(other_results, "{}/ds{}/bg.hdf".format(team_1['name'], args.dataset))]
    elif args.team1 == "ORChiD":
        team_1['fgpath'] = args.foreground_events
        team_1['bgpath'] = args.background_events
        
    if args.team2 in other_teams:
        team_2['fgpath'] = [os.path.join(other_results, "{}/ds{}/fg.hdf".format(team_2['name'], args.dataset))]
        team_2['bgpath'] = [os.path.join(other_results, "{}/ds{}/bg.hdf".format(team_2['name'], args.dataset))]
    elif args.team2 == "ORChiD":
        team_2['fgpath'] = args.foreground_events
        team_2['bgpath'] = args.background_events
    
    
    for nteam in [1, 2]:
        team = locals()["team_{}".format(nteam)]
        # Read foreground events
        logging.info(f'Reading foreground events from {team["fgpath"]}')
        fg_events = []
        for fpath in team['fgpath']:
            with h5py.File(fpath, 'r') as fp:
                fg_events.append(np.vstack([fp['time'], fp['stat'], fp['var']]))
        team['fgevents'] = np.concatenate(fg_events, axis=-1)
        
        # Read background events
        logging.info(f'Reading background events from {team["bgpath"]}')
        bg_events = []
        for fpath in team['bgpath']:
            with h5py.File(fpath, 'r') as fp:
                bg_events.append(np.vstack([fp['time'], fp['stat'], fp['var']]))
        team['bgevents'] = np.concatenate(bg_events, axis=-1)
    
    return team_1, team_2, injparams, use_chirp_distance


def compare_plot_1(team_1, team_2, save_dir):
    # Plot 1 (Histogram of all injections with found injections of both pipelines)
    os.makedirs(save_dir, exist_ok=False)
    params = team_1['params']
    ncols = 3
    nrows = len(params)//ncols + int(len(params)%ncols or False)
    
    thresh_names = ['1-per-month', '1-per-week', '1-per-day', '100-per-month', '1000-per-month']
    # How many signals are present above given threshold?
    for n, thresh in enumerate(team_1["far_thresholds"]):
        team_1[thresh_names[n]] = team_1['found_idx'][team_1["found_stats"] > thresh]
        team_2[thresh_names[n]] = team_2['found_idx'][team_2["found_stats"] > thresh]
    
    for thresh_name in thresh_names:
        # Subplotting
        fig, ax = plt.subplots(nrows, ncols, figsize=(8.0*ncols, 6.0*nrows))
        # Histogram kwargs
        kwargs = dict(histtype='stepfilled', alpha=0.5, density=True, bins=40)
        
        pidxs = list(itertools.product(range(nrows), range(ncols)))
        num_fin = 0
        for param, (i, j)  in zip(params, pidxs):
            ax[i][j].hist(team_1[param][team_1[thresh_name]], label=team_1['name'], color='blue', **kwargs)
            ax[i][j].hist(team_2[param][team_2[thresh_name]], label=team_2['name'], color='red', **kwargs)
            ax[i][j].set_title(param)
            ax[i][j].grid(True)
            ax[i][j].legend()
            num_fin+=1
        
        for i, j in pidxs[num_fin:]:
            ax[i][j].set_visible(False)
        
        plt.tight_layout()
        save_name = "compare_histogram_{}_and_{}-{}.png".format(team_1['name'], team_2['name'], thresh_name)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
    

def compare_plot_2(team_1, team_2, save_dir):
    # Plot 2 (Scatter plot of param vs param (unique finds from both teams are coloured))
    # Calculate signal duration
    lf = 20.0 # Hz
    G = 6.67e-11
    c = 3.0e8
    duration = lambda mchirp: 5. * (8.*np.pi*lf)**(-8./3.) * (mchirp*1.989e30*G/c**3.)**(-5./3.)
    team_1['duration'] = duration(team_1['mchirp'])
    team_2['duration'] = duration(team_2['mchirp'])
    
    ## VS Plots
    os.makedirs(save_dir, exist_ok=False)
    params = ['duration', 'mchirp', 'distance', 'q', 'chirp_distance', 'snr']
    ncols = 3
    plots = list(itertools.combinations(params, 2))
    nsubplots = len(plots)
    nrows = nsubplots//ncols + int(nsubplots%ncols or False)
    
    thresh_names = ['1-per-month', '1-per-week', '1-per-day', '100-per-month', '1000-per-month']
    # How many signals are present above given threshold?
    for n, thresh in enumerate(team_1["far_thresholds"]):
        team_1[thresh_names[n]] = team_1['found_idx'][team_1["found_stats"] > thresh]
        team_2[thresh_names[n]] = team_2['found_idx'][team_2["found_stats"] > thresh]
    
    for thresh_name in thresh_names:
        # Subplotting
        fig, ax = plt.subplots(nrows, ncols, figsize=(8.0*ncols, 5.0*nrows))
        kwargs = {}
        
        pidxs = list(itertools.product(range(nrows), range(ncols)))
        num_fin = 0
        for (param_1, param_2), (i, j)  in zip(plots, pidxs):
            # Team 1
            x = team_1[param_1][team_1[thresh_name]]
            y = team_1[param_2][team_1[thresh_name]]
            team1_set = set(zip(x, y)) 
            # Sanity check: What if two values are the same?
            assert len(list(team1_set)) == len(x)
            # Team 2
            x = team_2[param_1][team_2[thresh_name]]
            y = team_2[param_2][team_2[thresh_name]]
            team2_set = set(zip(x, y)) # TODO: What if two values are the same?
            assert len(list(team2_set)) == len(x)
            # Plots: A-B, B-A, A&B
            unique_team1 = np.array(list(team1_set - team2_set))
            unique_team2 = np.array(list(team2_set - team1_set))
            found_both = np.array(list(team1_set.intersection(team2_set)))
            # Scatter plotting
            kwargs.update({'color': 'blue', 's': 100.0, 'label': 'Unique {}'.format(team_1['name']), 'alpha': 0.7})
            if len(unique_team1) != 0:
                ax[i][j].scatter(unique_team1[:,0], unique_team1[:,1], **kwargs)
            kwargs.update({'color': 'red', 's': 100.0, 'label': 'Unique {}'.format(team_2['name']), 'alpha': 0.7})
            ax[i][j].scatter(unique_team2[:,0], unique_team2[:,1], **kwargs)
            kwargs.update({'color': 'darkgrey', 's': 30.0, 'label': 'Found by Both', 'alpha': 0.3})
            if len(found_both) != 0:
                ax[i][j].scatter(found_both[:,0], found_both[:,1], **kwargs)
            
            ax[i][j].set_xlabel(param_1)
            ax[i][j].set_ylabel(param_2)
            ax[i][j].grid(True)
            num_fin+=1
        
        for i, j in pidxs[num_fin:]:
            ax[i][j].set_visible(False)
        
        fig.suptitle('{} = Blue, {} = Red, Found by Both = Grey'.format(team_1['name'], team_2['name']))
        save_name = "param_vs_param_{}_and_{}-{}.png".format(team_1['name'], team_2['name'], thresh_name)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        plt.close()
    

def compare_plot_3(team_1, team_2, save_dir):
    # Plot 3 (Colour strip plot quantifying np.log10(Nnn/Nmf) found in each bin)
    os.makedirs(save_dir, exist_ok=False)
    params = team_1['params'] + ['duration']
    ncols = 3
    nrows = len(params)//ncols + int(len(params)%ncols or False)
    
    # Calculate signal duration
    lf = 20.0 # Hz
    G = 6.67e-11
    c = 3.0e8
    duration = lambda mchirp: 5. * (8.*np.pi*lf)**(-8./3.) * (mchirp*1.989e30*G/c**3.)**(-5./3.)
    team_1['duration'] = duration(team_1['mchirp'])
    team_2['duration'] = duration(team_2['mchirp'])
    
    thresh_names = ['1-per-month', '1-per-week', '1-per-day', '100-per-month', '1000-per-month']
    # How many signals are present above given threshold?
    for n, thresh in enumerate(team_1["far_thresholds"]):
        team_1[thresh_names[n]] = team_1['found_idx'][team_1["found_stats"] > thresh]
        team_2[thresh_names[n]] = team_2['found_idx'][team_2["found_stats"] > thresh]
    
    for thresh_name in thresh_names:
        # Subplotting
        fig, ax = plt.subplots(nrows, ncols, figsize=(5.0*ncols, 3.0*nrows))
        
        pidxs = list(itertools.product(range(nrows), range(ncols)))
        num_fin = 0
        for param, (i, j)  in zip(params, pidxs):
            team1_set = set(team_1[param][team_1[thresh_name]])
            team2_set = set(team_2[param][team_2[thresh_name]])
            # TODO: What if two values are the same?
            unique_team1 = np.array(list(team1_set - team2_set))
            unique_team2 = np.array(list(team2_set - team1_set))
            # Binning the two arrays before caluclating the ratio
            bins = np.linspace(min(team_1[param]), max(team_1[param]), 40, dtype=int, endpoint=True)
            team1_counts, _ = np.histogram(unique_team1, bins=bins)
            team2_counts, _ = np.histogram(unique_team2, bins=bins)
            # Calculate the ratio using the counts obtained
            # Sanity check
            team1_counts = team1_counts.astype(np.float32)
            team2_counts = team2_counts.astype(np.float32)
            team1_counts += 1e-3
            team2_counts += 1e-3
            
            ratio = team1_counts/team2_counts
            
            # Making the color strip plot
            height = 25
            divnorm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)
            kwargs = dict(cmap='seismic', norm=divnorm)
            axes = ax[i][j]
            cstr = ax[i][j].imshow(np.repeat(ratio, height).reshape(-1, height).T, **kwargs)
            ax[i][j].set_title(param)
            ax[i][j].set_yticks([])
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(cstr, cax=cax, orientation='vertical')
            num_fin+=1
        
        for i, j in pidxs[num_fin:]:
            ax[i][j].set_visible(False)
        
        fig.suptitle('Ratio = N_unique_{}/N_unique_{}'.format(team_1['name'], team_2['name']))
        save_name = "colour_strip_{}_and_{}-{}.png".format(team_1['name'], team_2['name'], thresh_name)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        plt.close()


def compare_plot_4(team_1, team_2, save_dir):
    # Plot 4 (Efficiency curves made for each of the two groups)
    os.makedirs(save_dir, exist_ok=False)
    
    thresh_names = ['1-per-month', '1-per-week', '1-per-day', '100-per-month', '1000-per-month']
    # How many signals are present above given threshold?
    for n, thresh in enumerate(team_1["far_thresholds"]):
        team_1[thresh_names[n]] = team_1['found_idx'][team_1["found_stats"] > thresh]
        team_2[thresh_names[n]] = team_2['found_idx'][team_2["found_stats"] > thresh]
    
    # (0, (3, 1, 1, 1, 1, 1)) is densely dashdotdotted in parameterised form
    # Refer: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 1, 1, 1, 1))]
    for n, thresh_name in enumerate(thresh_names):
        plt.figure(figsize=(12.0, 9.0))
        # Plotting the efficiency curve for each FAR threshold
        team1_data = team_1['snr'][team_1[thresh_name]]
        team2_data = team_2['snr'][team_2[thresh_name]]
        # Binning the two arrays before caluclating the TAP (True Alarm Probability)
        bins = np.linspace(min(team_1['snr']), max(team_1['snr']), 20, dtype=int, endpoint=True)
        all_counts, _ = np.histogram(team_1['snr'], bins=bins)
        team1_counts, _ = np.histogram(team1_data, bins=bins)
        team2_counts, _ = np.histogram(team2_data, bins=bins)
        # Plotting
        xbins = (bins[1:] + bins[:-1])/2.
        kwargs = dict(marker='o', markersize=12, fillstyle='none', linestyle=linestyles[n])
        plt.plot(xbins, team1_counts/all_counts, markerfacecolor='blue', color='blue', 
                 label="{}, {}".format(team_1['name'], thresh_name), **kwargs)
        plt.plot(xbins, team2_counts/all_counts, markerfacecolor='red', color='red', 
                 label="{}, {}".format(team_2['name'], thresh_name), **kwargs)
        
        plt.grid(True, which='both')
        plt.xlabel("Optimal SNR")
        plt.ylabel("True Alarm Probability")
        plt.title('Efficiency Curves ({} and {})'.format(team_1['name'], team_2['name']))
        save_name = "efficiency_curves_{}_and_{}-{}.png".format(team_1['name'], team_2['name'], thresh_name)
        save_path = os.path.join(save_dir, save_name)
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def compare_groups(team_1, team_2, output_dir):
    """
    Comparing the found injections by different (any 2) groups
    In this module we make 3 plots for comparison:
        1. Histogram of all injections with found injections of both pipelines
        2. Scatter plot of param vs param (unique finds from both teams are coloured)
        3. Colour strip plot quantifying np.log10(Nnn/Nmf) found in each bin
        4. Efficiency curves made for each of the two groups
        
    Each of these plots for all params are made for different FAR thresholds
    """
    
    save_dir = os.path.join(output_dir, "FOUND_AND_MISSED")
    os.makedirs(save_dir, exist_ok=False)
    
    compare_plot_1(team_1, team_2, os.path.join(save_dir, "histogram"))
    compare_plot_2(team_1, team_2, os.path.join(save_dir, "param_vs_param"))
    compare_plot_3(team_1, team_2, os.path.join(save_dir, "uniqueness_color_strips"))
    compare_plot_4(team_1, team_2, os.path.join(save_dir, "efficiency_curves"))
    

def get_stats(args, idxs, duration=None, output_dir=None, snrs=None):
    """
    Calculate the false-alarm rate and sensitivity of a search
    algorithm.
    
    Arguments
    ---------
    fgevents : np.array
        A numpy array with three rows. The first row has to contain the
        times returned by the search algorithm where it believes to have
        found a true signal. The second row contains a ranking statistic
        like quantity for each time. The third row contains the maxmimum
        distance to an injection for the given event to be counted as
        true positive. The values have to be determined on the
        foreground data, i.e. noise plus additive signals.
    bgevents : np.array
        A numpy array with three rows. The first row has to contain the
        times returned by the search algorithm where it believes to have
        found a true signal. The second row contains a ranking statistic
        like quantity for each time. The third row contains the maxmimum
        distance to an injection for the given event to be counted as
        true positive. The values have to be determined on the
        background data, i.e. pure noise.
    injparams : dict
        A dictionary containing at least two entries with keys `tc` and
        `distance`. Both entries have to be numpy arrays of the same
        length. The entry `tc` contains the times at which injections
        were made in the foreground. The entry `distance` contains the
        according luminosity distances of these injections.
    duration : {None or float, None}
        The duration of the analyzed background. If None the injections
        are used to infer the duration.
    
    Returns
    -------
    dict:
        Returns a dictionary, where each key-value pair specifies some
        statistic. The most important are the keys `far` and
        `sensitive-distance`.
    """
    
    # Get data from fg and bg events file
    team_1, team_2, injparams, chirp_distance = read_data(args, idxs)
    # Add SNRs into the injparams (this will automatically include it wihtin most plots)
    injparams['snr'] = snrs
    
    # Return data tmp var
    ret = {}
    
    ## COMMON ##
    # Get injection params
    injtimes = injparams['tc']
    dist = injparams['distance']
    
    # Get chirp mass from the source masses
    if chirp_distance:
        massc = mchirp(injparams['mass1'], injparams['mass2'])
    # Set duration if nothing is passed
    if duration is None:
        duration = injtimes.max() - injtimes.min()
        
    
    for nteam in [1, 2]:
        team = locals()["team_{}".format(nteam)]
        
        logging.info('Sorting foreground event times')
        sidxs = team["fgevents"][0].argsort()
        fgevents = team["fgevents"].T[sidxs].T
        
        logging.info('Finding injection times closest to event times')
        idxs = find_closest_index(injtimes, fgevents[0])
        diff = np.abs(injtimes[idxs] - fgevents[0])
        
        # If the difference between the injection time and trigger is within tc variance
        # The trigger is identified as an event (there may be duplicate triggers)
        logging.info('Finding true- and false-positives')
        tpbidxs = diff <= fgevents[2]
        tpidxs = np.arange(len(fgevents[0]))[tpbidxs]
        fpbidxs = diff > fgevents[2]
        fpidxs = np.arange(len(fgevents[0]))[fpbidxs]
        
        tpevents = fgevents.T[tpidxs].T
        fpevents = fgevents.T[fpidxs].T
        
        ## Update the returns dictionary
        if team['name'] == "ORChiD":
            ret['fg-events'] = fgevents
            ret['found-indices'] = np.arange(len(injtimes))[idxs]
            ret['missed-indices'] = np.setdiff1d(np.arange(len(injtimes)), ret['found-indices'])
            ret['true-positive-event-indices'] = tpidxs
            ret['false-positive-event-indices'] = fpidxs
            ret['sorting-indices'] = sidxs
            ret['true-positive-diffs'] = diff[tpidxs]
            ret['false-positive-diffs'] = diff[fpidxs]
            ret['true-positives'] = tpevents
            ret['false-positives'] = fpevents
        
        # Calculate foreground FAR
        logging.info('Calculating foreground FAR')
        noise_stats_fg = fpevents[1].copy()
        noise_stats_fg.sort()
        fgfar = len(noise_stats_fg) - np.arange(len(noise_stats_fg)) - 1
        fgfar = fgfar / duration
        if team['name'] == "ORChiD":
            ret['fg-far'] = fgfar
        
        # Calculate background FAR
        logging.info('Calculating background FAR')
        noise_stats = team["bgevents"][1].copy()
        noise_stats.sort()
        far = len(noise_stats) - np.arange(len(noise_stats)) - 1
        far = far / duration
        if team['name'] == "ORChiD":
            ret['far'] = far
        
        # Find best true-positive for each injection
        found_injections = []
        tmpsidxs = idxs.argsort()
        sorted_idxs = idxs[tmpsidxs]
        iidxs = np.full(len(idxs), False)
        for i in tqdm(range(len(injtimes)), ascii=True, desc='Determining found injections'):
            L = np.searchsorted(sorted_idxs, i, side='left')
            if L >= len(idxs) or sorted_idxs[L] != i:
                continue
            R = np.searchsorted(sorted_idxs, i, side='right')
            # All indices that point to the same injection
            iidxs[tmpsidxs[L:R]] = True
            # Indices of the true-positives that belong to the same injection
            eidxs = np.logical_and(iidxs[tmpsidxs[L:R]],
                                   tpbidxs[tmpsidxs[L:R]])
            if eidxs.any():
                found_injections.append([i, np.max(fgevents[1][tmpsidxs[L:R]][eidxs])])
            iidxs[tmpsidxs[L:R]] = False
    
        # Number of injections found within given testing data
        found_injections = np.array(found_injections).T
        print('Number of found injections = {}'.format(len(found_injections[0])))
        
        # Calculate sensitivity
        # CARE! THIS APPLIES ONLY WHEN THE DISTRIBUTION IS CHOSEN CORRECTLY
        logging.info('Calculating sensitivity')
        sidxs = found_injections[1].argsort() # Sort found injections
        found_injections = found_injections.T[sidxs].T
        
        if chirp_distance:
            found_mchirp_total = massc[found_injections[0].astype(int)]
            mchirp_max = massc.max()
            # print('found_mchirp_total is the chirp mass of all found injections')
            # print('max = {}, min = {}, mean={}, median = {}'.format(max(found_mchirp_total), min(found_mchirp_total), np.mean(found_mchirp_total), np.median(found_mchirp_total)))
            if team['name'] == "ORChiD":
                # Histogram of found injections vs all injections in 1-month testing dataset
                found_param_plots(noise_stats, output_dir, injparams, found_injections)
                # Plotting all param vs param
                param_vs_param(output_dir, injparams, found_injections)
            
        max_distance = dist.max()
        # print('Maximum distance given by injections = {}'.format(max_distance))
        vtot = (4. / 3.) * np.pi * max_distance**3.
        Ninj = len(dist)
        print('Total number of injections = {}'.format(Ninj))
        
        # Params to calculate the sensitive volume
        if chirp_distance:
            mc_norm = mchirp_max ** (5. / 2.) * len(massc)
        else:
            mc_norm = Ninj
            
        prefactor = vtot / mc_norm
        nfound = len(found_injections[1]) - np.searchsorted(found_injections[1],
                                                            noise_stats,
                                                            side='right')
        
        
        if chirp_distance:
            # Get found chirp-mass indices for given threshold
            fidxs = np.searchsorted(found_injections[1], noise_stats, side='right')
            # Plotting the network output
            network_output(found_injections, noise_stats, output_dir, team['name'])
            if team['name'] == "ORChiD":
                # Parameter learning
                parameter_learning(injparams, noise_stats, found_injections, output_dir)
            
            found_mchirp_total = np.flip(found_mchirp_total)
            
            # Calculate sum(found_mchirp ** (5/2))
            # with found_mchirp = found_mchirp_total[i:]
            # and i looped over fidxs
            # Code below is a vectorized form of that
            cumsum = np.flip(np.cumsum(found_mchirp_total ** (5./2.)))
            cumsum = np.concatenate([cumsum, np.zeros(1)])
            mc_sum = cumsum[fidxs]
            Ninj = np.sum((mchirp_max / massc) ** (5. / 2.))
            
            cumsumsq = np.flip(np.cumsum(found_mchirp_total ** 5))
            cumsumsq = np.concatenate([cumsumsq, np.zeros(1)])
            sample_variance_prefactor = cumsumsq[fidxs]
            sample_variance = sample_variance_prefactor / Ninj\
                              - (mc_sum / Ninj) ** 2  # noqa: E127
        else:
            mc_sum = nfound
            sample_variance = nfound / Ninj - (nfound / Ninj) ** 2
            
        vol = prefactor * mc_sum
        vol_err = prefactor * (Ninj * sample_variance) ** 0.5
        rad = (3 * vol / (4 * np.pi))**(1. / 3.)
        print('Radius or sensitive distance as calculated from the volume obtained ({})'.format(team['name']))
        print('min rad = {}, max rad = {}'.format(min(rad), max(rad)))
        
        if team['name'] == "ORChiD":
            ret['sensitive-volume'] = vol
            ret['sensitive-distance'] = rad
            ret['sensitive-volume-error'] = vol_err
            ret['sensitive-fraction'] = nfound / Ninj
        
        if team['name'] == "PyCBC":
            ret['sensitive-distance-pycbc'] = rad
            ret['far-pycbc'] = far
        
        # Update plotting params for each group
        team['found_idx'] = found_injections[0].astype(int)
        team['found_stats'] = found_injections[1]
        # Add all found injparams to to plotting dict
        team['params'] = list(injparams.keys())
        team.update(injparams)
        # The values given are indices and have to be 1 less than the number of FA per month req.
        team['far_thresholds'] = noise_stats[::-1][[0, 3, 29, 99, 999]]
        team['sens_dist'] = rad
        team['sens_frac'] = nfound / Ninj
    
    ## Save Data to analyse found injections and make plots comparing PyCBC and our pipeline
    compare_groups(team_1, team_2, output_dir)
        
    return ret


def main(raw_args=None, cfg_far_scaling_factor=None, dataset=None):
    
    parser = argparse.ArgumentParser(description='Testing phase evaluator')
    
    parser.add_argument('--injection-file', type=str, required=True,
                        help=("Path to the file containing information "
                              "on the injections. (The file returned by"
                              "`generate_data.py --output-injection-file`"))
    parser.add_argument('--foreground-events', type=str, nargs='+',
                        required=True,
                        help=("Path to the file containing the events "
                              "returned by the search on the foreground "
                              "data set as returned by "
                              "`generate_data.py --output-foreground-file`."))
    parser.add_argument('--foreground-files', type=str, nargs='+',
                        required=True,
                        help=("Path to the file containing the analyzed "
                              "foreground data output by"
                              "`generate_data.py --output-foreground-file`."))
    parser.add_argument('--background-events', type=str, nargs='+',
                        required=True,
                        help=("Path to the file containing the events "
                              "returned by the search on the background"
                              "data set as returned by "
                              "`generate_data.py --output-background-file`."))
    parser.add_argument("--far-scaling-factor", help="Rescale FAR when making sensitivity plot",
                        type=float, required=False, default=-1.0)
    parser.add_argument("--dataset", help="dataset type",
                        type=int, required=False, default=-1)
    parser.add_argument('--output-file', type=str, required=True,
                        help=("Path at which to store the output HDF5 "
                              "file. (Path must end in `.hdf`)"))
    parser.add_argument('--output-dir', type=str, required=True,
                        help=("Path at which to store the output png "
                              "files. (Path must exist within export_dir)"))
    
    # Teams
    parser.add_argument('--team1', type=str, required=False,
                        default="ORChiD",
                        help=("Team 1 to be compared using evalution plots"))
    parser.add_argument('--team2', type=str, required=False,
                        default="PyCBC",
                        help=("Team 2 to be compared using evalution plots"))
    
    
    parser.add_argument('--verbose', action='store_true',
                        help="Print update messages.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args(raw_args)
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    # Sanity check arguments here
    if os.path.splitext(args.output_file)[1] != '.hdf':
        raise ValueError('The output file must have the extension `.hdf`.')
    
    if os.path.isfile(args.output_file) and not args.force:
        raise IOError(f'The file {args.output_file} already exists. '
                      'Set the flag `force` to overwrite it.')
    
    if args.far_scaling_factor == -1 and cfg_far_scaling_factor == None:
        raise ValueError('FAR scaling factor not provided. Use the --far-scaling-factor argument when running.')
    elif cfg_far_scaling_factor == None:
         far_scaling_factor = args.far_scaling_factor
    elif cfg_far_scaling_factor != None:
        far_scaling_factor = cfg_far_scaling_factor

    if args.dataset == -1 and dataset == None:
        raise ValueError('Dataset type not provided. Use the --dataset argument when running.')
    elif dataset == None:
        dataset = args.dataset
    elif dataset != None:
        dataset = dataset
    
    # Caluclate the SNR for each injection in the testing dataset (if not present already)
    dataset_dir = Path(args.injection_file).parent.absolute()
    snrs_path = os.path.join(dataset_dir, "snr.hdf")
    if os.path.exists(snrs_path):
        with h5py.File(snrs_path, 'r') as fp:
            snrs = fp['snr'][()]
    else:
        snrs = get_snrs(args.injection_file, data_cfg, dataset_dir)
    
    # Find indices contained in foreground
    print("\nRunning Testing Phase Evaluator")
    logging.info('Finding injections contained in data')
    padding_start, padding_end = 30, 30
    dur, idxs = find_injection_times(args.foreground_files,
                                     args.injection_file,
                                     padding_start=padding_start,
                                     padding_end=padding_end)
    if np.sum(idxs) == 0:
        msg = 'The foreground data contains no injections! '
        msg += 'Probably a too small section of data was generated. '
        msg += 'Please make sure to generate at least {} seconds of data. '
        msg += 'Otherwise a sensitive distance cannot be calculated.'
        msg = msg.format(padding_start + padding_end + 24)
        raise RuntimeError(msg)
    
    print('Duration calculated by find_injection_times = {}'.format(dur))
    stats = get_stats(args, idxs, duration=dur, output_dir=args.output_dir, snrs=snrs)
    
    # Store results
    logging.info(f'Writing output to {args.output_file}')
    mode = 'w' if args.force else 'x'
    with h5py.File(args.output_file, mode) as fp:
        for key, val in stats.items():
            fp.create_dataset(key, data=np.array(val))
    
    # Create the sensitivity vs FAR/month plot from the output evaluation obtained
    assert dur == far_scaling_factor, 'FAR scaling factor discrepancy! Check duration.'
    with h5py.File(args.output_file, 'r') as fp:
        far = fp['far'][()]
        sens = fp['sensitive-distance'][()]
        sidxs = far.argsort()
        far = far[sidxs][1:] * far_scaling_factor
        sens = sens[sidxs][1:]
        
        far_pycbc = fp['far-pycbc'][()]
        sidxs_pycbc = far_pycbc.argsort()
        far_pycbc_chk = far_pycbc[sidxs_pycbc] * far_scaling_factor
        sens_pycbc_check = fp['sensitive-distance-pycbc'][()]
        sens_pycbc_check = sens_pycbc_check[sidxs_pycbc]

        print(len(far_pycbc), len(sens_pycbc_check))
    
    # Month FAR factor
    month = 30.0 * 24.0 * 60.0 * 60.0
    
    plt.figure(figsize=(18.0, 12.0))
    plt.title('Sensitivity Measure for Dataset {}'.format(dataset))
    plt.plot(far*(month/dur), sens, color='m', linewidth=3.0, label='nnarenraju')

    with h5py.File('/home/nnarenraju/Research/results/PyCBC/ds{}/eval.hdf'.format(dataset)) as fp:
        sens_pycbc = np.array(fp['sensitive-distance'])
        far_pycbc = np.array(fp['far'])
    plt.plot(far_pycbc*month, sens_pycbc, color='orange', linewidth=2.5, linestyle='dashed', label='PyCBC')

    with h5py.File('/home/nnarenraju/Research/results/TPI_FSU_Jena/ds{}/eval.hdf'.format(dataset)) as fp:
        sens_fsu = np.array(fp['sensitive-distance'])
        far_fsu = np.array(fp['far'])
    plt.plot(far_fsu*month, sens_fsu, color='red', linewidth=2.5, linestyle='dashed', label='TPI FSU Jena')

    with h5py.File('/home/nnarenraju/Research/results/Virgo-AUTh/ds{}/eval.hdf'.format(dataset)) as fp:
        sens_virgo = np.array(fp['sensitive-distance'])
        far_virgo = np.array(fp['far'])
    plt.plot(far_virgo*month, sens_virgo, color='blueviolet', linewidth=2.5, linestyle='dashed', label='Virgo-AUTh')

    with h5py.File('/home/nnarenraju/Research/results/CNN-Coinc/ds{}/eval.hdf'.format(dataset)) as fp:
        sens_cnn = np.array(fp['sensitive-distance'])
        far_cnn = np.array(fp['far'])
    plt.plot(far_cnn*month, sens_cnn, color='green', linewidth=2.5, linestyle='dashed', label='CNN-Coinc')

    with h5py.File('/home/nnarenraju/Research/results/MFCNN/ds{}/eval.hdf'.format(dataset)) as fp:
        sens_mfcnn = np.array(fp['sensitive-distance'])
        far_mfcnn = np.array(fp['far'])
    plt.plot(far_mfcnn*month, sens_mfcnn, color='blue', linewidth=2.5, linestyle='dashed', label='MFCNN')
    
    plt.grid(True, which='both')
    plt.xlim(1000, 1)
    plt.ylim(0, 3500)
    plt.xscale('log')
    plt.xlabel('False Alarm Rate (FAR) per month')
    plt.ylabel('Sensitive Distance [MPc]')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'sensitivity_all_teams.png'))
    plt.close()
    
    plt.figure(figsize=(18.0, 12.0))
    plt.title('Sensitivity Measure for Dataset {}'.format(dataset))
    plt.plot(far*(month/dur), sens, color='m', linewidth=3.0, label='nnarenraju')

    with h5py.File('/home/nnarenraju/Research/results/{}/ds{}/eval.hdf'.format(args.team2, dataset)) as fp:
        sens_team2 = np.array(fp['sensitive-distance'])
        far_team2 = np.array(fp['far'])
        plt.plot(far_team2*month, sens_team2, color='orange', linewidth=2.5, linestyle='dashed', label='{}'.format(args.team2))
    
    plt.grid(True, which='both')
    plt.xlim(1000, 1)
    plt.ylim(1500, 2100)
    plt.xscale('log')
    plt.xlabel('False Alarm Rate (FAR) per month')
    plt.ylabel('Sensitive Distance [MPc]')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'sensitivity_compare_teams.png'))
    plt.close()


if __name__ == "__main__":
    main()
