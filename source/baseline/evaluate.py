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

import os
import h5py
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 18})

from matplotlib import cm
from tqdm import tqdm

# PyCBC handling
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.detector


def find_injection_times(fgfiles, injfile, padding_start=0, padding_end=0):
    """Determine injections which are contained in the file.
    
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
    """Find the index of the closest element in the array for the given
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


def get_stats(fgevents, bgevents, injparams, duration=None,
              chirp_distance=False, output_dir=None):
    """Calculate the false-alarm rate and sensitivity of a search
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
    ret = {}
    injtimes = injparams['tc']
    dist = injparams['distance']
    if chirp_distance:
        massc = mchirp(injparams['mass1'], injparams['mass2'])
    if duration is None:
        duration = injtimes.max() - injtimes.min()
    logging.info('Sorting foreground event times')
    sidxs = fgevents[0].argsort()
    fgevents = fgevents.T[sidxs].T
    
    logging.info('Finding injection times closest to event times')
    idxs = find_closest_index(injtimes, fgevents[0])
    diff = np.abs(injtimes[idxs] - fgevents[0])
    print('Difference between injection times and event times in foreground')
    print('max = {}, min = {}, mean = {}, median = {}'.format(max(diff), min(diff), np.mean(diff), np.median(diff)))
    
    logging.info('Finding true- and false-positives')
    tpbidxs = diff <= fgevents[2]
    tpidxs = np.arange(len(fgevents[0]))[tpbidxs]
    fpbidxs = diff > fgevents[2]
    fpidxs = np.arange(len(fgevents[0]))[fpbidxs]
    
    tpevents = fgevents.T[tpidxs].T
    fpevents = fgevents.T[fpidxs].T
    
    ret['fg-events'] = fgevents
    ret['found-indices'] = np.arange(len(injtimes))[idxs]
    ret['missed-indices'] = np.setdiff1d(np.arange(len(injtimes)),
                                         ret['found-indices'])
    ret['true-positive-event-indices'] = tpidxs
    ret['false-positive-event-indices'] = fpidxs
    ret['sorting-indices'] = sidxs
    ret['true-positive-diffs'] = diff[tpidxs]
    ret['false-positive-diffs'] = diff[fpidxs]
    ret['true-positives'] = tpevents
    ret['false-positives'] = fpevents
    
    print('true positive diffs')
    print('max tp diff = {}, min tp diff = {}, mean tp diff = {}'.format(max(diff[tpidxs]), min(diff[tpidxs]), np.mean(diff[tpidxs])))
    
    print('false positive diffs')
    print('max fp diff = {}, min fp diff = {}, mean fp diff = {}'.format(max(diff[fpidxs]), min(diff[fpidxs]), np.mean(diff[fpidxs])))
    
    # Calculate foreground FAR
    logging.info('Calculating foreground FAR')
    noise_stats = fpevents[1].copy()
    noise_stats.sort()
    print('ranking statistics for false positive events')
    print('max = {}, min = {}, mean = {}, median = {}'.format(max(noise_stats), min(noise_stats), np.mean(noise_stats), np.median(noise_stats)))
    fgfar = len(noise_stats) - np.arange(len(noise_stats)) - 1
    fgfar = fgfar / duration
    ret['fg-far'] = fgfar
    
    # Calculate background FAR
    logging.info('Calculating background FAR')
    noise_stats = bgevents[1].copy()
    noise_stats.sort()
    print('ranking statistics for true positive events')
    print('max = {}, min = {}, mean = {}, median = {}'.format(max(noise_stats), min(noise_stats), np.mean(noise_stats), np.median(noise_stats)))
    far = len(noise_stats) - np.arange(len(noise_stats)) - 1
    far = far / duration
    ret['far'] = far
    
    # Find best true-positive for each injection
    verbose = logging.root.level is logging.INFO
    found_injections = []
    tmpsidxs = idxs.argsort()
    sorted_idxs = idxs[tmpsidxs]
    iidxs = np.full(len(idxs), False)
    for i in tqdm(range(len(injtimes)), ascii=True, disable=not verbose,
                  desc='Determining found injections'):
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
            found_injections.append([i,
                                    np.max(fgevents[1][tmpsidxs[L:R]][eidxs])])
        iidxs[tmpsidxs[L:R]] = False
    
    found_injections = np.array(found_injections).T
    print('Number of found injections = {}'.format(len(found_injections[0])))
    
    # Calculate sensitivity
    # CARE! THIS APPLIES ONLY WHEN THE DISTRIBUTION IS CHOSEN CORRECTLY
    logging.info('Calculating sensitivity')
    sidxs = found_injections[1].argsort()
    found_injections = found_injections.T[sidxs].T  # Sort found injections
    if chirp_distance:
        found_mchirp_total = massc[found_injections[0].astype(int)]
        print('found_mchirp_total is the chirp mass of all found injections')
        print('max = {}, min = {}, mean={}, median = {}'.format(max(found_mchirp_total), min(found_mchirp_total), np.mean(found_mchirp_total), np.median(found_mchirp_total)))
        
        mchirp_max = massc.max()
        
        ## Plotting the comparison plots (injections and found histogram) for all params 
        # cmap = cm.get_cmap('RdYlBu_r', 10)
        for param in injparams.keys():
            all_param = injparams[param]
            found_param = all_param[found_injections[0].astype(int)]
            # Plotting the overlap histograms of all and found data
            plt.figure(figsize=(12.0, 12.0))
            plt.title('Comparing testing data with found signals - {}'.format(param))
            plt.hist(all_param, bins=100, label='{}-all'.format(param), alpha=0.8)
            plt.hist(found_param, bins=100, label='{}-found'.format(param), alpha=0.8)
            plt.grid(True, which='both')
            plt.xlabel('{}'.format(param))
            plt.ylabel('Number of Occurences')
            plt.savefig(os.path.join(output_dir, '{}-compare.png'.format(param)))
            plt.close()
        
        ## Other related plots
        plt.figure(figsize=(12.0, 12.0))
        plt.title('mchirp vs distance')
        param_1 = injparams['mchirp'][found_injections[0].astype(int)]
        param_2 = injparams['distance'][found_injections[0].astype(int)]
        plt.scatter(param_1, param_2, marker='.', s=100.0)
        plt.grid(True, which='both')
        plt.xlabel('Chirp Mass')
        plt.ylabel('Distance')
        plt.savefig(os.path.join(output_dir, 'mchirp_vs_distance.png'))
        plt.close()
        
        plt.figure(figsize=(9.0, 9.0))
        plt.title('mchirp vs q')
        param_1 = injparams['mchirp'][found_injections[0].astype(int)]
        param_2 = injparams['q'][found_injections[0].astype(int)]
        plt.scatter(param_1, param_2, marker='.', s=100.0)
        plt.grid(True, which='both')
        plt.xlabel('Chirp Mass')
        plt.ylabel('Mass Ratio (m1/m2)')
        plt.savefig(os.path.join(output_dir, 'mchirp_vs_q.png'))
        plt.close()
        
        
    max_distance = dist.max()
    vtot = (4. / 3.) * np.pi * max_distance**3.
    Ninj = len(dist)
    print('Total number of injections = {}'.format(Ninj))
    
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
    print('Volumes found')
    print('min err = {}, max err = {}, mean err = {}'.format(min(vol), max(vol), np.mean(vol)))
    
    vol_err = prefactor * (Ninj * sample_variance) ** 0.5
    print('Volume error')
    print('min err = {}, max err = {}, mean err = {}'.format(min(vol_err), max(vol_err), np.mean(vol_err)))
    
    rad = (3 * vol / (4 * np.pi))**(1. / 3.)
    print('Radius or sensitive distance as calculated from the volume obtained')
    print('min rad = {}, max rad = {}, mean = {}'.format(min(rad), max(rad), np.mean(rad)))
    
    ret['sensitive-volume'] = vol
    ret['sensitive-distance'] = rad
    ret['sensitive-volume-error'] = vol_err
    ret['sensitive-fraction'] = nfound / Ninj
        
    return ret


def optimise_fmin(h_pol, signal_length, signal_low_freq_cutoff, sample_rate, waveform_kwargs):
    # Use self.waveform_kwargs to calculate the fmin for given params
    # Such that the length of the signal is atleast 20s by the time it reaches fmin
    current_start_time = -1*h_pol.get_sample_times()[0]
    req_start_time = signal_length - h_pol.get_sample_times()[-1]
    fmin = signal_low_freq_cutoff*(current_start_time/req_start_time)**(3./8.)
    
    while True:
        # fmin_new is the fmin required for the current params to produce 20.0s signal
        waveform_kwargs['f_lower'] = fmin
        h_plus, h_cross = pycbc.waveform.get_td_waveform(**waveform_kwargs)
        # Sanity check to verify the new signal length
        new_signal_length = len(h_plus)/sample_rate
        if new_signal_length > signal_length:
            break
        else:
            fmin = fmin - 3.0
        
    # Return new signal
    return h_plus, h_cross


def main(raw_args):
    
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
    parser.add_argument('--output-file', type=str, required=True,
                        help=("Path at which to store the output HDF5 "
                              "file. (Path must end in `.hdf`)"))
    parser.add_argument('--output-dir', type=str, required=True,
                        help=("Path at which to store the output png "
                              "files. (Path must exist within export_dir)"))
    
    
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
    
    # Read injection parameters
    logging.info(f'Reading injections from {args.injection_file}')
    
    injparams = {}
    with h5py.File(args.injection_file, 'r') as fp:
        params = list(fp.keys())
        for param in params:
            injparams[param] = fp[param][()][idxs]
        use_chirp_distance = 'chirp_distance' in params
    
    ### Get SNRs of all the injections
    """ Injection """
    
    """
    # Generate source parameters
    waveform_kwargs = {'delta_t': 1./sample_rate}
    waveform_kwargs['f_lower'] = signal_low_freq_cutoff
    waveform_kwargs['approximant'] = signal_approximant
    waveform_kwargs['f_ref'] = reference_freq
    
    for idx in range(len(injparams['tc'])):
        # Get the waveform kwargs from injparams
        waveform_kwargs = {param:injparams[param][idx] for param in injparams.keys()}
        
        # Generate the full waveform
        h_plus, h_cross = pycbc.waveform.get_td_waveform(**waveform_kwargs)
        # If the signal is smaller than 20s, we change fmin such that it is atleast 20s
        if -1*h_plus.get_sample_times()[0] + h_plus.get_sample_times()[-1] < self.signal_length:
            # Pass h_plus or h_cross
            h_plus, h_cross = self.optimise_fmin(h_plus)
        
        if -1*h_plus.get_sample_times()[0] + h_plus.get_sample_times()[-1] > self.signal_length:
            new_end = h_plus.get_sample_times()[-1]
            new_start = -1*(self.signal_length - new_end)
            h_plus = h_plus.time_slice(start=new_start, end=new_end)
            h_cross = h_cross.time_slice(start=new_start, end=new_end)
    
    # Sanity check for signal lengths
    # if len(h_plus)/self.sample_rate != self.signal_length:
    #     act = self.signal_length*self.sample_rate
    #     obs = len(h_plus)
    #     raise ValueError('Signal length ({}) is not as expected ({})!'.format(obs, act))
    
    # # Properly time and project the waveform (What there is)
    start_time = prior_values['injection_time'] + h_plus.get_sample_times()[0]
    end_time = prior_values['injection_time'] + h_plus.get_sample_times()[-1]
    
    # Calculate the number of zeros to append or prepend (What we need)
    # Whitening padding will be corrupt and removed in whiten transformation
    start_samp = prior_values['tc'] + (self.whiten_padding/2.0)
    start_interval = prior_values['injection_time'] - start_samp
    # subtract delta value for length error (0.001 if needed)
    end_padding = self.whiten_padding/2.0
    post_merger = self.signal_length - prior_values['tc']
    end_interval = prior_values['injection_time'] + post_merger + end_padding
    
    # Calculate the difference (if any) between two time sets
    diff_start = start_time - start_interval
    diff_end = end_interval - end_time
    # Convert num seconds to num samples
    diff_end_num = int(diff_end * self.sample_rate)
    diff_start_num = int(diff_start * self.sample_rate)
    
    expected_length = ((end_interval-start_interval) + self.error_padding_in_s*2.0) * self.sample_rate
    observed_length = len(h_plus) + (diff_start_num + diff_end_num + self.error_padding_in_num*2.0)
    diff_length = expected_length - observed_length
    if diff_length != 0:
        diff_end_num += diff_length
    
    # If any positive difference exists, add padding on that side
    # Pad h_plus and h_cross with zeros on both end for slicing
    if diff_end > 0.0:
        # Append zeros if we need samples after signal ends
        h_plus.append_zeros(int(diff_end_num + self.error_padding_in_num))
        h_cross.append_zeros(int(diff_end_num + self.error_padding_in_num))
    
    if diff_start > 0.0:
        # Prepend zeros if we need samples before signal begins
        # prepend_zeros arg must be an integer
        h_plus.prepend_zeros(int(diff_start_num + self.error_padding_in_num))
        h_cross.prepend_zeros(int(diff_start_num + self.error_padding_in_num))
    
    assert len(h_plus) == self.sample_length_in_num + self.error_padding_in_num*2.0
    assert len(h_cross) == self.sample_length_in_num + self.error_padding_in_num*2.0
    
    # Setting the start_time, sets epoch and end_time as well within the TS
    # Set the start time of h_plus and h_plus after accounting for prepended zeros
    h_plus.start_time = start_interval - self.error_padding_in_s
    h_cross.start_time = start_interval - self.error_padding_in_s
    """
    
    
    
    
    # Read foreground events
    logging.info(f'Reading foreground events from {args.foreground_events}')
    fg_events = []
    for fpath in args.foreground_events:
        with h5py.File(fpath, 'r') as fp:
            fg_events.append(np.vstack([fp['time'],
                                        fp['stat'],
                                        fp['var']]))
    fg_events = np.concatenate(fg_events, axis=-1)
    
    # Read background events
    logging.info(f'Reading background events from {args.background_events}')
    bg_events = []
    for fpath in args.background_events:
        with h5py.File(fpath, 'r') as fp:
            bg_events.append(np.vstack([fp['time'],
                                        fp['stat'],
                                        fp['var']]))
    bg_events = np.concatenate(bg_events, axis=-1)
    
    stats = get_stats(fg_events, bg_events, injparams,
                      duration=dur,
                      chirp_distance=use_chirp_distance,
                      output_dir=args.output_dir)
    
    # Store results
    logging.info(f'Writing output to {args.output_file}')
    mode = 'w' if args.force else 'x'
    with h5py.File(args.output_file, mode) as fp:
        for key, val in stats.items():
            fp.create_dataset(key, data=np.array(val))
    return


if __name__ == "__main__":
    main()
