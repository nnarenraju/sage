# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Mar 25 00:43:21 2022

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
import gc
import h5py
import glob
import math
import time
import random
import logging
import itertools
import tracemalloc
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# Plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# PyCBC handling
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.detector
from pycbc.distributions.utils import draw_samples_from_config

# LOCAL
from data.mlmdc_noise_generator import NoiseGenerator



class Normalise:
    """
    Normalise the parameter using prior ranges
    
        For example, norm_tc = (tc - min_val)/(max_val - min_val)
        The values of max_val and min_val are provided
        to the class. self.get_norm can be called during
        data generation to get normalised values of tc, if needed.
    
    """
    
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def norm(self, val):
        # Return lambda to use for normalisation
        return (val - self.min_val)/(self.max_val - self.min_val)


class GenerateData:
    
    # Slots magic for parameters input from data_configs.py
    __slots__ = ['dataset', 'parent_dir', 'data_dir', 'seed', 'export_dir', 'dirs',
                 'make_dataset', 'make_module', 'priors', 'chunk_size', 'nchunk', 'save_psds',
                 'psds', 'skylocation_dist', 'np_gen', 'psd_names', 'inj_path', 'error_padding_in_s',
                 'psd_len', 'delta_f', 'noise_low_freq_cutoff', 'idx_offset', 'max_nsamp_noise',
                 'label_wave', 'label_noise', 'num_waveforms', 'num_noises', 'max_nsamp_signal',
                 'iterable', 'filter_duration', 'sample_rate', 'signal_low_freq_cutoff',
                 'signal_approximant', 'reference_freq', 'detectors_abbr', 'noise_names',
                 'save_injection_priors', 'gc_collect_frequency', 'hdf5_tree', 'tmp',
                 'num_queues_datasave', 'num_cores_datagen', 'fieldnames', 'waveform_names',
                 'num_sample_save', 'signal_length', 'whiten_padding', 'groups', 'error_padding_in_num',
                 'sample_length_in_s', 'sample_length_in_num', 'waveform_kwargs', 'noise_generator',
                 'prior_low_mass', 'prior_high_mass', 'prior_low_chirp_dist', 'prior_high_chirp_dist',
                 'tc_inject_lower', 'tc_inject_upper', 'noise_high_freq_cutoff',
                 'max_signal_length', 'ringdown_leeway', 'merger_leeway', 'start_freq_factor',
                 'fs_reduction_factor', 'fbin_reduction_factor', 'dbins', 'check_seeds',
                 'norm_tc', 'norm_dist', 'norm_dchirp', 'norm_mchirp', 'norm_q']
    
    
    def __init__(self, **kwargs):
        ## Get slots magic attributes via input dict (use **kwargs)
        # This should set all slots given above
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        """ Save dir """
        self.export_dir = ""
        self.dirs = {}
        self.inj_path = ""
        
        """ Sanity check with gc_collect """
        # MP methods do not mix well with gc.collect
        # Always set this to -1. *DO NOT* change this.
        self.gc_collect_frequency = -1
        
        # Create the detectors
        self.detectors_abbr = ('H1', 'L1')
    
        ### Create labels
        self.label_wave = 1.0
        self.label_noise = 0.0
        
        self.iterable = None
        self.nchunk = -1
    
        """ Generating noise function """
        
        ### Create the power spectral densities of the respective detectors
        if self.dataset == 1:
            psd_fun = pycbc.psd.analytical.aLIGOZeroDetHighPower
            self.psd_names = ['aLIGOZeroDetHighPower', 'aLIGOZeroDetHighPower']
            self.psds = [psd_fun(self.psd_len, self.delta_f, self.noise_low_freq_cutoff)
                         for _ in range(len(self.detectors_abbr))]
        else:
            # Here, we should pick two PSDs randomly and read the files
            # PSDs are obtained from the noise generator
            self.psds = []
            self.psd_names = []
        
        # Fixed parameter options for iteration
        if self.dataset == 1:
            self.noise_generator = pycbc.noise.gaussian.frequency_noise_from_psd
        else:
            self.noise_generator = NoiseGenerator(self.dataset,
                                                  seed=self.seed,
                                                  delta_f=self.delta_f,
                                                  sample_rate=self.sample_rate,
                                                  low_frequency_cutoff=self.noise_low_freq_cutoff,
                                                  detectors=self.detectors_abbr)
        
        """ Generating signals """
        # Generate source parameters
        self.waveform_kwargs = {'delta_t': 1./self.sample_rate}
        self.waveform_kwargs['f_lower'] = self.signal_low_freq_cutoff
        self.waveform_kwargs['approximant'] = self.signal_approximant
        self.waveform_kwargs['f_ref'] = self.reference_freq
        
        # Prior samples obtained from dataset ini file
        self.priors = None
        self.idx_offset = None
        self.fieldnames = None
        # Normalising values for certain outputs
        self.norm_tc = None
        self.norm_dist = None
        self.norm_dchirp = None
        self.norm_mchirp = None
        self.norm_q = None
        
        ## Save tmp tree structure for HDF5 file
        self.hdf5_tree = []
        self.groups = ['2048']
        self.tmp = None
        
        ## Names
        # All datasets that need to be created from the sample dict
        self.waveform_names = ['h_plus', 'h_cross', 'start_time', 'interval_lower', 'interval_upper',
                          'norm_tc', 'norm_dist', 'norm_mchirp', 'norm_dchirp', 'norm_q',
                          'mass1', 'mass2', 'distance', 'mchirp', 'label']
        
        self.noise_names = ['noise_1', 'noise_2', 'label']
        
        # Create datasets for each field that needs to be saved
        self.max_nsamp_signal = int(self.chunk_size[0]/self.num_queues_datasave)
        self.max_nsamp_noise = int(self.chunk_size[1]/self.num_queues_datasave)
    
    
    def __str__(self):
        return 'MP based Batch Data Generation (avg rate: 117.5 per second on PC)'
    
    
    def make_data_dir(self):
        """ Creating required dirs """
        # Make directory structure for data storage
        self.dirs['parent'] = os.path.join(self.parent_dir, self.data_dir)
        self.dirs['injections'] = os.path.join(self.dirs['parent'], "injections")
        self.dirs['dataset'] = os.path.join(self.dirs['parent'], "dataset")
        
        if not os.path.exists(self.dirs['parent']):
            os.makedirs(self.dirs['parent'], exist_ok=False)
            os.makedirs(self.dirs['dataset'], exist_ok=False)
            
            if self.save_injection_priors:
                os.makedirs(self.dirs['injections'], exist_ok=False)
        else:
            raise IOError("make_default_dataset: Dataset dir already exists!")
    
    
    def save_PSD(self):
        # Saving the PSD file used for dataset/sample
        if self.dataset == 1:
            # Both detectors have the same PSD for dataset 1
            psd_save_dir = os.path.join(self.dirs['parent'], "psds")
            if not os.path.exists(psd_save_dir):
                os.makedirs(psd_save_dir, exist_ok=False)
            save_path = os.path.join(psd_save_dir, "psd-aLIGOZeroDetHighPower.hdf")
            psd_file_path = os.path.abspath(save_path)
            # Remove and rewrite if the PSD file already exists
            if os.path.exists(psd_file_path):
                os.remove(psd_file_path)
            # Write PSD in HDF5 format
            with h5py.File(psd_file_path, 'a') as fp:
                data = self.psds[0].numpy()
                key = 'data'
                fp.create_dataset(key, data=data, compression='gzip', 
                                  compression_opts=9, shuffle=True)
                # Adding all relevant attributes
                fp.attrs['delta_f'] = self.delta_f
                fp.attrs['name'] = 'aLIGOZeroDetHighPower'  
    
    
    def distance_from_chirp_distance(self, chirp_distance, mchirp, ref_mass=1.4):

        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the luminosity distance given a chirp distance and chirp mass.
        return chirp_distance * (2.**(-1./5) * ref_mass / mchirp)**(-5./6)
    
    
    def get_priors(self):
        ## Generate source prior parameters
        
        # A path to the .ini file.
        ini_parent = '.'
        CONFIG_PATH = "{}/ds{}.ini".format(ini_parent, self.dataset)
        random_seed = self.seed
        
        # Draw num_waveforms number of samples from ds.ini file
        priors = draw_samples_from_config(path=CONFIG_PATH,
                                          num=self.num_waveforms,
                                          seed=random_seed)
        
        ## Get normalisation params for certain output values
        
        # Normalise time of coalescence
        self.norm_tc = Normalise(min_val=self.tc_inject_lower, max_val=self.tc_inject_upper)
        
        # Normalise chirp mass
        ml = self.prior_low_mass
        mu = self.prior_high_mass
        # m2 will always be slightly lower than m1, but (m, m) will give limit
        # that the mchirp will never reach but tends to as num_samples tends to inf.
        # Range for mchirp can be written as --> (min_mchirp, max_mchirp)
        min_mchirp = (ml*ml / (ml+ml)**2.)**(3./5) * (ml + ml)
        max_mchirp = (mu*mu / (mu+mu)**2.)**(3./5) * (mu + mu)
        # Get normalised mchirp
        self.norm_mchirp = Normalise(min_val=min_mchirp, max_val=max_mchirp)
        
        # Get distance ranges from chirp distance priors
        # mchirp present in numerator of self.distance_from_chirp_distance.
        # Thus, min_mchirp for dist_lower and max_mchirp for dist_upper
        dist_lower = self.distance_from_chirp_distance(self.prior_low_chirp_dist, min_mchirp)
        dist_upper = self.distance_from_chirp_distance(self.prior_high_chirp_dist, max_mchirp)
        # Sanity check with priors
        assert dist_lower <= np.min(priors['distance'])
        assert dist_upper >= np.max(priors['distance'])
        # get normlised distance class
        self.norm_dist = Normalise(min_val=dist_lower, max_val=dist_upper)
        
        # Normalise chirp distance
        self.norm_dchirp = Normalise(min_val=self.prior_low_chirp_dist, max_val=self.prior_high_chirp_dist)
        
        # Normalise mass ratio
        # m2 is always less than m1, but as an approx. we keep min ratio as m/m=1
        # max ratio will just be (mu, ml) --> mu/ml
        # The range can be written as --> (min_val, max_val]
        self.norm_q = Normalise(min_val=1.0, max_val=mu/ml)
        
        ## End normalisation ##
        
        """
        ## Using the samples variable to get priors
        # Print all fieldnames stored within samples variable
        print(self.priors.fieldnames)
        # Print a certain parameter, for example 'mass1'.
        print(self.priors[0]['mass1'])
        """
        
        """ Write priors """
        self.fieldnames = priors.fieldnames
        if self.save_injection_priors:
            self.inj_path = os.path.join(self.dirs['injections'], 'injections.hdf')
            if not os.path.exists(self.inj_path):
                # Writing injections.hdf
                with h5py.File(self.inj_path, 'a') as fp:
                    # create a dataset for batch save
                    fp.create_dataset('data', data=priors, 
                                      compression='gzip', compression_opts=9, 
                                      shuffle=True)
        
    
    def optimise_fmin(self, h_pol):
        # Use self.waveform_kwargs to calculate the fmin for given params
        # Such that the length of the signal is atleast 20s by the time it reaches fmin
        current_start_time = -1*h_pol.get_sample_times()[0]
        req_start_time = self.signal_length - h_pol.get_sample_times()[-1]
        fmin = self.signal_low_freq_cutoff*(current_start_time/req_start_time)**(3./8.)
        
        while True:
            # fmin_new is the fmin required for the current params to produce 20.0s signal
            self.waveform_kwargs['f_lower'] = fmin
            h_plus, h_cross = pycbc.waveform.get_td_waveform(**self.waveform_kwargs)
            # Sanity check to verify the new signal length
            new_signal_length = len(h_plus)/self.sample_rate
            if new_signal_length > self.signal_length:
                break
            else:
                fmin = fmin - 3.0
            
        # Return new signal
        return h_plus, h_cross
    
    
    def worker(self, idx, queues=None, qidx=None, save_idx=None):
        
        """ Set the random seed for this iteration here """
        # Deprecation: This seed is not longer used to create priors. 
        # We can however, use this for selecting the queue to deposit data.
        np.random.seed(int(idx+1))
        
        """ Obtain sample """
        is_waveform = idx < self.num_waveforms
        sample = {}
        
        if not is_waveform:
            ## Generate noise
            if self.dataset == 1:
                maxlen = round(self.sample_length_in_num)
                noise = [self.noise_generator(psd).to_timeseries()[:maxlen]
                          for psd in self.psds]
                
                assert len(noise[0]) == maxlen
                assert len(noise[1]) == maxlen
            else:
                noise, self.psds = self.noise_generator(0.0, self.sample_length_in_s, None)
                noise = [noise[det].numpy() for det in self.detectors_abbr]
            
            # Saving noise params for storage
            sample['noise_1'] = noise[0]
            sample['noise_2'] = noise[1]
            sample['label'] = self.label_noise
            sample['psd1'] = self.psd_names[0]
            sample['psd2'] = self.psd_names[1]
        
        else:
            ## Generate signal
            """ Get input params from injections.hdf """
            # Iterate through injection parmas using idx and set params to waveform_kwargs
            # prior values as created by get_priors is a np.record object
            prior_values = self.priors[idx-self.idx_offset]
            # Convert np.record object to dict and append to waveform_kwargs dict
            self.waveform_kwargs.update(dict(zip(prior_values.dtype.names, prior_values)))
            
            """ Injection """
            # Generate the full waveform
            h_plus, h_cross = pycbc.waveform.get_td_waveform(**self.waveform_kwargs)
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
            
            
            # h_plus and h_cross is stored for augmentation purposes
            sample['h_plus'] = h_plus
            sample['h_cross'] = h_cross
            sample['label'] = self.label_wave
            # These have to be stored as lists for HDF5 dataset
            sample['start_time'] = float(h_plus.start_time)
            sample['interval_lower'] = start_interval
            sample['interval_upper'] = end_interval
            # Add normalised values of output params
            sample['norm_tc'] = self.norm_tc.norm(prior_values['tc'])
            sample['norm_dist'] = self.norm_dist.norm(prior_values['distance'])
            sample['norm_mchirp'] = self.norm_mchirp.norm(prior_values['mchirp'])
            sample['norm_dchirp'] = self.norm_dchirp.norm(prior_values['chirp_distance'])
            sample['norm_q'] = self.norm_q.norm(prior_values['q'])
            # waveform_kwargs will contain all prior values and additional attrs.
            sample.update(self.waveform_kwargs)
            ## norm SNR will be added to attributes after converting h_pols into h_t
        
        
        # Pass data onto Queue to be stored in a file
        data = {}
        data['sample'] = sample
        data['is_waveform'] = is_waveform
        data['idx'] = idx
        data['save_idx'] = save_idx
        data['kill'] = False
        
        # Give all relevant save data to Queue
        qidx = np.random.randint(low=0, high=len(queues))
        queues[qidx].put(data)
        
        """ Clean up RAM (occasionally) """
        # DO NOT DO THIS WITH MP Dataset Generation (slows down rates by a LOT)
        if self.gc_collect_frequency != -1:
            raise ValueError('MP based dataset generation is not efficient with gc.collect()')
    
    
    def _make_dataset(self, grp, key, data, maxshape):
        grp.create_dataset(key, data=data, 
                           compression='gzip', compression_opts=9, 
                           shuffle=True,
                           maxshape=maxshape)
    
    
    def _require_dataset(self, grp, key, maxshape, dtype):
        grp.require_dataset(key, shape=maxshape, dtype=dtype)
        
    
    def _add_fields(self, names, sample, max_nsamp, _group, maxlen):
        # Writing all datasets for signals
        for name in names:
            data = sample[name]
            if isinstance(data, pycbc.types.timeseries.TimeSeries):
                maxshape = (max_nsamp, None)
                curr_len = len(data)
            elif isinstance(data, np.ndarray):
                maxshape = (max_nsamp, None)
                curr_len = len(data)
            elif isinstance(data, float) or isinstance(data, str):
                maxshape = (max_nsamp,)
                curr_len = 1
                if isinstance(data, str):
                    data = data.encode("ascii")
            
            # Add an extra dimension the sample for appendable storage
            data = np.array([data])
            
            self._make_dataset(_group, name, data, maxshape)
            
            if curr_len > maxlen[name]:
                maxlen[name] = curr_len
    
    
    def _req_fields(self, names, max_nsamp, _group, _type):
        # Writing all datasets for signals
        signal_names = ['h_plus', 'h_cross']
        noise_names = ['noise_1', 'noise_2']
        nonsample_names = [foo for foo in names if foo not in signal_names and foo not in noise_names]
        dtype = np.float64
        sgroup = _group.create_group(_type)
        
        for name in names:
            if name in signal_names:
                maxshape = (max_nsamp, self.sample_length_in_num + self.error_padding_in_num*2.0)
            elif name in noise_names:
                maxshape = (max_nsamp, self.sample_length_in_num)
            elif name in nonsample_names:
                maxshape = (max_nsamp,)
                if name in ['psd1', 'psd2']:
                    # WARNING: Not implemented! Cannot raise exception within MP.
                    dtype = np.bytes_
            
            self._require_dataset(sgroup, name, maxshape, dtype)
    
    
    def listener(self, queue, qidx, tmp=None):
        """
        Write given sample data to a HDF file
        We also use this to append to priors
        
        ** WARNING!!! **: If files are not being written, there should be an error in here.
        This error will not be raised in MP and has not yet been handled properly.
        
        Vague check for this type of error:
            1. Print the queue.get() within the infinite loop.
            2. If this prints a singular output, then there is something wrong within listener.
            3. Then print a bunch of Lorem Ipsum here and there
            4. Print statements after the error occcurence will not be displayed
            5. Narrow down the error location and shoot your shot! (Sorry :P)
            
        """
        
        ## Continuously check for data to the Queue
        
        # Open and persist a HDF5 file for chunk storage
        # This file will be used by n-queues to write in async, the data
        # Keep an HDF5 file open for chunk storage (this does not take up extra RAM when appending)
        # Check test_hdf5_append under diagnistics in the repo for experimentation and proof
        # Keep an open HDF5 file until chunk finishes generating
        qname = "dataset_chunk_{}_qidx_{}.hdf".format(self.nchunk, qidx)
        store = os.path.join(self.dirs['dataset'], qname)
        chunk_file = h5py.File(store, 'a')
        
        # Intiate groups and datasets with maxshape within chunk hdf5 file
        # Create groups for each of the sampling rates
        groups = [chunk_file.create_group(grp) for grp in self.groups]
        
        # Create all necessary fields as datasets
        _ = [self._req_fields(self.waveform_names, self.max_nsamp_signal, grp, 'signal') for grp in groups]
        _ = [self._req_fields(self.noise_names, self.max_nsamp_noise, grp, 'noise') for grp in groups]
        
        while True:
            
            data = queue.get()
            
            """ The following is run ONLY if we get something from Queue """
            
            # if processes have ended, kill
            if data['kill']:
                break
            
            ## Segregate data from Queue
            # sample is a dict of information about sample (refer worker)
            sample = data['sample']
            idx = data['idx']
            save_idx = data['save_idx']
            is_waveform = data['is_waveform']
            
            """ Write sample """
            ## Save a chunk of samples in .hdf format
            # Create dataset objects for each storage array
            # 1. Sample saved at full 2048. Hz sampling rate with key as '2048/signal/<h_pol>'
            # 2. Get required MR sampling rates from dbins and create dataset for each sampling rate
            # 3. Add all necessary attributes exactly once
            
            # Sampling rate groups
            # TODO: We need to decimate the samples for each sample rate group
            for group in groups:
                    
                if is_waveform:
                    _group = group['signal']
                    names = self.waveform_names
                elif not is_waveform:
                    _group = group['noise']
                    names = self.noise_names
                
                for name in names:
                    data = sample[name]
                    # Add an extra dimension the sample for appendable storage
                    _group[name][save_idx] = data
        
        
        ## Add all necessary one time attributes (hdf5 file still open)
        # Dataset params
        
        chunk_file.attrs['dataset'] = self.dataset
        chunk_file.attrs['seed'] = self.seed
        # Sample params
        chunk_file.attrs['num_waveforms'] = self.num_waveforms
        chunk_file.attrs['num_noises'] = self.num_noises
        chunk_file.attrs['dataset_ratio_SNnum'] = self.num_waveforms/self.num_noises
        
        chunk_file.attrs['sample_rate'] = self.sample_rate
        chunk_file.attrs['signal_length'] = self.signal_length
        chunk_file.attrs['sample_length_in_s'] = self.sample_length_in_s
        chunk_file.attrs['sample_length_in_num'] = self.sample_length_in_num
        chunk_file.attrs['signal_approximant'] = self.signal_approximant
        # Prior values
        chunk_file.attrs['prior_low_mass'] = self.prior_low_mass
        chunk_file.attrs['prior_high_mass'] = self.prior_high_mass
        chunk_file.attrs['prior_low_chirp_dist'] = self.prior_low_chirp_dist
        chunk_file.attrs['prior_high_chirp_dist'] = self.prior_high_chirp_dist
        chunk_file.attrs['tc_inject_lower'] = self.tc_inject_lower
        chunk_file.attrs['tc_inject_upper'] = self.tc_inject_upper
        # Injection params (required)
        chunk_file.attrs['sample_rate'] = self.sample_rate
        chunk_file.attrs['noise_low_freq_cutoff'] = self.noise_low_freq_cutoff
        # PSD params
        chunk_file.attrs['delta_f'] = self.delta_f
        chunk_file.attrs['psd_len'] = self.psd_len
        
        # Close the chunk storage file
        chunk_file.close()
        
    
    def generate_dataset(self):
        """
        Create dataset using the explicit PyCBC method
        Dataset is made as close as possible to the testing dataset types
        This code is much faster to execute and easier to read
        """
        
        """ Handling number of cores for task """
        # Sanity check
        assert self.num_queues_datasave + self.num_cores_datagen <= mp.cpu_count()
        
        # Must use Manager queue here, or will not work
        manager = mp.Manager()
        queues = [manager.Queue() for _ in range(self.num_queues_datasave)]
            
        # Empty jobs every iteration
        jobs = []
        
        # Initialise pool
        pool = mp.Pool(int(self.num_cores_datagen+self.num_queues_datasave))
        
        # Put listener to work first (this will wait for data in MP and write to Queue)
        # NOTE: Cannot pass open files to listener
        watchers = [pool.apply_async(self.listener, (queue, n)) for n, queue in enumerate(queues)]
        
        # Create save qidx for each idx based on number of queues
        noi_offset = min(self.iterable[1])
        sig_offset = min(self.iterable[0])
        united_iterable = np.concatenate(self.iterable)
        idx_iterable = np.concatenate((self.iterable[0]-sig_offset, self.iterable[1]-noi_offset))
        
        assert sum(self.chunk_size) % self.num_queues_datasave == 0
        save_qidxs = list(range(len(queues))) * int(sum(self.chunk_size)/self.num_queues_datasave)
        sidxs1 = np.array([[n]*len(queues) for n in range(int(self.chunk_size[0]/self.num_queues_datasave))]).flatten()
        sidxs2 = np.array([[n]*len(queues) for n in range(int(self.chunk_size[1]/self.num_queues_datasave))]).flatten()
        save_idxs = np.concatenate((sidxs1, sidxs2))
        assert len(united_iterable) == len(save_qidxs)
        assert len(united_iterable) == len(save_idxs)
        
        for idx, save_qidx, save_idx in zip(united_iterable, save_qidxs, save_idxs):
            job = pool.apply_async(self.worker, (idx, queues, save_qidx, save_idx))
            jobs.append(job)

        # Collect results from the workers through the pool result queue
        tracemalloc.start() # memory tracer
        pbar = tqdm(jobs)
        # pbar = jobs
        for job in pbar:
            mem = tracemalloc.get_traced_memory()
            curr_mem = np.round(mem[0]/(1024*1024), 3)
            max_mem = np.round(mem[1]/(1024*1024), 3)
            pbar.set_description('MPB DataGen: RAM curr={} MB, peak={} MB'.format(curr_mem, max_mem))
            job.get()
        
        # Kill memory tracers
        tracemalloc.stop()
        # Kill the listener when all jobs are complete
        _ = [queue.put({'kill': True}) for queue in queues]
        # End the pool processes
        pool.close()
        pool.join()
        
        
        """ Make consolidated chunk dataset """
        # Combine all pool chunk files into one dataset
        chunkfile = "dataset_chunk_{}.hdf".format(self.nchunk)
        chunkpath = os.path.join(self.dirs['dataset'], chunkfile)
        chunk_dataset = h5py.File(chunkpath, 'w')
        
        # Create core groups for sample rates
        groups = [chunk_dataset.create_group(grp) for grp in self.groups]
        # Create sub-groups and datasets
        _ = [self._req_fields(self.waveform_names, self.chunk_size[0], grp, 'signal') for grp in groups]
        _ = [self._req_fields(self.noise_names, self.chunk_size[1], grp, 'noise') for grp in groups]
        
        seek = 0
        traverse_size = 0
        add_attrs = True
        for qidx in range(len(queues)):
            filename = "dataset_chunk_{}_qidx_{}.hdf".format(self.nchunk, qidx)
            filepath = os.path.join(self.dirs['dataset'], filename)
            with h5py.File(filepath, 'r') as fp:
                # fp.keys() --> different sample rate groups
                # fp[sample_rate_grp].keys() --> signal and noise groups
                sr_grps = list(fp.keys())
                sn_grps = list(fp[sr_grps[0]].keys())
                # Get all group tree
                gtree = [i+'/'+j for i,j in itertools.product(sr_grps, sn_grps)]
                
                for grp in gtree:
                    for obj in fp[grp].keys():
                        data = np.array(fp[grp][obj])
                        traverse_size = data.shape[0]
                        # Add the data to proper chunk
                        chunk_dataset[grp][obj][seek:seek+traverse_size] = data[:]
                
                # Add attrs
                attrs_save = dict(fp.attrs)
                if add_attrs:
                    for key, value in attrs_save.items():
                        chunk_dataset.attrs[key] = value
                    add_attrs = False
                
                seek += traverse_size
            
            # Delete all unnecessary qidx files
            os.remove(filepath)
        
        chunk_dataset.close()
        

    def _get_recursive_abspath_(self, directory):
        # Get abspath of all files within directory
        paths = np.array([])
        for dirpath, _, filenames in os.walk(os.path.abspath(directory)):
            for filename in filenames:
                 paths = np.append(paths, os.path.join(dirpath, filename))
        return paths
    
    
    def plot_priors(self, save_dir):
        """
        All Params created by ini file:
        
        ('mass1', 'mass2', 'ra', 'dec', 'inclination', 'coa_phase', 'polarization', 'chirp_distance', 
        'spin1_a', 'spin1_azimuthal', 'spin1_polar', 'spin2_a', 'spin2_azimuthal', 'spin2_polar', 
        'tc', 'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z', 'mchirp', 'q', 'distance')
        
        Plot any and all that is required as subplots.
        Deprecated: plotting each prior separately and storing vals as .npy files
        """
        
        # Data and params
        with h5py.File(self.inj_path, "r") as fp:
            # Get and return the batch data
            samples = np.array([list(foo) for foo in fp['data'][:]])
            
        n_bins = 100
        
        ignore = []
        fields = [foo for foo in self.fieldnames if foo not in ignore]
        num_columns = 3
        num_rows = math.ceil(len(fields)/num_columns)
        fig, ax = plt.subplots(num_rows, num_columns, figsize=(6.0*num_columns, 3.0*num_rows))
        pidxs = list(itertools.product(range(num_rows), range(num_columns)))
        
        num_fin = 0
        for n, (field, (i, j))  in enumerate(zip(fields, pidxs)):
            ax[i][j].hist(samples[:,n], bins=n_bins, density=True)
            ax[i][j].set_title(field)
            ax[i][j].grid()
            num_fin+=1
        
        for i, j in pidxs[num_fin:]:
            ax[i][j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('{}/priors_{}.png'.format(save_dir, self.data_dir))
        plt.close()
    
    
    def get_keys(self, name, obj):
        # Get key tree structure from HDF5 file
        """
        # Use the following if we have attribites to access
        for key, val in obj.attrs.items():
            print("{}: {}".format(key, val))
        """
        self.hdf5_tree.append(name)
        
    
    def make_training_lookup(self):
        # Make a lookup table with (id, path, target) for each training data
        # Should have an equal ratio, shuffled, of both classes (signal and noise)
        paths = []
        ids = []
        targets = []
        # Save the dataset paths alongside the target and ids as hdf5
        self.dirs['lookup'] = os.path.join(self.dirs['parent'], "training.hdf")
        # All other params for parameter estimation are stored within the sample files
        chunks = os.path.join(self.dirs['dataset'], "*.hdf")
        for cidx, chunk_file in enumerate(glob.glob(chunks)):
            ## Each of these chunk files has an equal number of signals and noise
            ids.append(cidx)
            paths.append(chunk_file)
            targets.append(-1)
        
        ## Create lookup using zip
        # Explicitly check for length inconsistancies. zip doesn't raise error.
        assert len(ids) == len(paths)
        assert len(paths) == len(targets)
        lookup = list(zip(ids, paths, targets))
        # Shuffle the column stack (signal and noise are not shuffled)
        # NumPy: "Multi-dimensional arrays are only shuffled along the first axis"
        random.shuffle(lookup)
        # Separate out the tuples for ids, paths and targets
        ids, paths, targets = zip(*lookup)
        
        # Write required fields as datasets in HDF5 training.hdf file
        with h5py.File(self.dirs['lookup'], 'a') as ds:
            """
            Shuffle Filter for HDF5:
                Block-oriented compressors like GZIP or LZF work better when presented with 
                runs of similar values. Enabling the shuffle filter rearranges the bytes in 
                the chunk and may improve compression ratio. No significant speed penalty, 
                lossless.
            """
            ds.create_dataset('id', data=ids, compression='gzip', compression_opts=9, shuffle=True)
            ds.create_dataset('path', data=paths, compression='gzip', compression_opts=9, shuffle=True)
            ds.create_dataset('target', data=targets, compression='gzip', compression_opts=9, shuffle=True)
        
        print("make_training_lookup: Lookup table created successfully!")
    
    
    def make_elink_lookup(self):
        # Make a lookup table with (id, path, target) for each training data
        # Should have an equal ratio, shuffled, of both classes (signal and noise)
        paths = []
        ids = []
        targets = []
        # Save the dataset paths alongside the target and ids as hdf5
        self.dirs['lookup'] = os.path.join(self.dirs['parent'], "extlinks.hdf")
        # Save this data in the hdf5 format as training.hdf
        main = h5py.File(self.dirs['lookup'], 'a', libver='latest')
        add_attrs = True
        idx = 0
        
        # All other params for parameter estimation are stored within the sample files
        chunks = os.path.join(self.dirs['dataset'], "*.hdf")
        for nfile, chunk_file in enumerate(glob.glob(chunks)):
            ## Each of these chunk files has an equal number of signals and noise
            ## Read the file, get all data and add data as ExternalLink to new training.hdf file
            # Read HDF5 tree and add tree to ExternalLink
            with h5py.File(chunk_file, 'r') as h5f:
                # fp.keys() --> different sample rate groups
                # fp[sample_rate_grp].keys() --> signal and noise groups
                sr_grps = list(h5f.keys())
                sn_grps = list(h5f[sr_grps[0]].keys())
                # Get all group tree
                gtree = [i+'/'+j for i,j in itertools.product(sr_grps, sn_grps)]
                
                for grp in gtree:
                    # Get sample length of each dataset
                    _branch = f"{nfile:03}" + '/' + grp
                    if bool(re.search('signal', grp)):
                        mode = 1
                    elif bool(re.search('noise', grp)):
                        mode = 0
                    
                    for obj in h5f[grp].keys():
                        # Add dataset as External Link
                        # Branch Format: 007 / 2048 / signal / h_plus
                        curr_branch = grp + '/' + obj
                        extl_branch = _branch + '/' + obj
                        main[extl_branch] = h5py.ExternalLink(chunk_file, curr_branch)
                    
                    # Maxshape of given np.ndarray
                    shape = np.array(h5f[grp][list(h5f[grp].keys())[0]]).shape[0]
                    # Update idxs
                    branch_ids = np.arange(idx, idx+shape)
                    ids.extend(branch_ids)
                    idx = idx+shape
                    # Update paths to dataset objects
                    branch_paths = itertools.product([_branch], np.arange(shape))
                    paths.extend([foo + '/' + str(bar) for foo, bar in branch_paths])
                    # Update target variable for each sample
                    targets.extend(np.full(shape, mode))
                        
                # Add attributes from chunk files once to main ExternalLink File
                if add_attrs:
                    attrs_save = dict(h5f.attrs)
                    for key, value in attrs_save.items():
                        main.attrs[key] = value
                    add_attrs = False
                    
                """
                ## Adding ExternalLink using visititems() method
                # Clean self.hdf5_tree
                self.hdf5_tree = []
                # Get all sampling rate datasets within file
                groups = list(h5f.keys())
                # Obtain tree structure under groups
                h5f.visititems(self.get_keys)
                # Remove the group directory from the tree structure
                self.hdf5_tree = [foo for foo in self.hdf5_tree if foo not in groups]
                # Remove the sample idx directory from the tree structure
                sample_idxs = list(h5f[groups[0]])
                rm_sample_dirs = [foo+'/'+bar for foo, bar in itertools.product(groups, sample_idxs)]
                self.hdf5_tree = [foo for foo in self.hdf5_tree if foo not in rm_sample_dirs]
                
                # Add attributes from chunk files once to main ExternalLink File
                attrs_save = dict(h5f.attrs)
                if not len(attrs_save.keys() & attrs_main.keys()):
                    for key, value in attrs_save.items():
                        main.attrs[key] = value
                """
        
        # Close file explicitly, or use with instead
        main.close()
        
        # Sanity check
        """ 
        with h5py.File(self.dirs['lookup'], 'a', libver='latest') as fp:
            start = time.time()
            print(np.array(fp['004/2048/signal/h_plus'][2]))
            fin = time.time() - start
            print(fin)
        raise
        """
        
        # visititems does not show h_plus, h_cross or noise datasets for some reason
        # If we add an extra /lorem to the linked dataset name, this shows up
        # However, this is not a phantom name. It will actually exist. 
        # Probably an issue with visititems method.
        """
        with h5py.File(self.dirs['lookup'], 'r') as h5f:
            self.hdf5_tree = []
            h5f.visititems(self.get_keys)
            for i in self.hdf5_tree:
                print(i)
            
            print(np.array(h5f['2048/1500/noise_1']))
            
        """
        
        ## Create lookup using zip
        # Explicitly check for length inconsistancies. zip doesn't raise error.
        assert len(ids) == len(paths)
        assert len(paths) == len(targets)
        lookup = list(zip(ids, paths, targets))
        # Shuffle the column stack (signal and noise are not shuffled)
        random.shuffle(lookup)
        # Separate out the tuples for ids, paths and targets
        ids, paths, targets = zip(*lookup)
        
        # Write required fields as datasets in HDF5 training.hdf file
        with h5py.File(self.dirs['lookup'], 'a') as ds:
            """
            Shuffle Filter for HDF5:
                Block-oriented compressors like GZIP or LZF work better when presented with 
                runs of similar values. Enabling the shuffle filter rearranges the bytes in 
                the chunk and may improve compression ratio. No significant speed penalty, 
                lossless.
            """
            ds.create_dataset('id', data=ids, compression='gzip', compression_opts=9, shuffle=True)
            ds.create_dataset('path', data=paths, compression='gzip', compression_opts=9, shuffle=True)
            ds.create_dataset('target', data=targets, compression='gzip', compression_opts=9, shuffle=True)
        
        print("make_elink_lookup: ExternalLink lookup table created successfully!")


def make(slots_magic_params, export_dir):
    """

    Parameters
    ----------
    slots_magic_params : TYPE
        DESCRIPTION.
    export_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    gd = GenerateData(**slots_magic_params)
    
    # Create storage directory sub-structure
    gd.make_data_dir()
    # Setting up the export directory
    gd.export_dir = export_dir
    # Save PSDs required for dataset
    gd.save_PSD()
    # Get priors for entire dataset
    gd.get_priors()
    
    ## Generate the dataset
    # Split the iterable into nchunks
    waveform_iterable = np.arange(gd.num_waveforms)
    noise_iterable = np.arange(gd.num_waveforms, gd.num_waveforms+gd.num_noises)
    # Number of chunks to split into
    nchunks_waveform = int(gd.num_waveforms/gd.chunk_size[0])
    nchunks_noise = int(gd.num_noises/gd.chunk_size[1])
    # Split iterables into chunks for waveform and noise separately
    waveform_iterables = np.array_split(waveform_iterable, nchunks_waveform)
    noise_iterables = np.array_split(noise_iterable, nchunks_noise)
    # WARNING: following code made for equal number of noise and waveforms only
    global_iterables = list(zip(waveform_iterables, noise_iterables))
    
    for nchunk, chunk in enumerate(global_iterables):
        
        start = time.time()
        # Generate chunk 'n' of dataset
        gd.iterable = chunk
        # Get prior values of chosen waveform idxs
        with h5py.File(gd.inj_path, "r") as fp:
            # Get and return the batch data
            gd.idx_offset = np.min(chunk[0])
            gd.priors = np.array(fp['data'][chunk[0]])
        
        gd.nchunk = nchunk
        gd.generate_dataset()
        
        # Delete all raw files once trainable dataset has been created
        # shutil.rmtree(os.path.join(data_dir, 'foreground'))
        
        # Freeing memory explicitly
        gd.priors = None
        gd.iterable = None
        gc.collect()
        
        # Display time taken for chunk generation
        finish = time.time() - start
        print("Time taken for chunk data generation = {} minutes".format(finish/60.))
    
    # Make dataset lookup table
    gd.make_training_lookup()
    # Make elink dataset lookup table
    gd.make_elink_lookup()
    # Making the prior distribution plots
    gd.plot_priors(gd.dirs['parent'])
