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
import csv
import h5py
import math
import time
import shutil
import logging
import warnings
import os, os.path
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# PyCBC handling
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.detector
from pycbc.distributions.utils import draw_samples_from_config

# LOCAL
from data.mlmdc_noise_generator import NoiseGenerator
from data.plot_default_priors import plot_priors



class GenerateData:
    
    # Slots magic for parameters input from data_configs.py
    __slots__ = ['dataset', 'parent_dir', 'data_dir', 'seed', 'export_dir', 'dirs',
                 'make_dataset', 'make_module', 'priors',
                 'psds', 'skylocation_dist', 'np_gen',
                 'psd_len', 'delta_f', 'noise_low_freq_cutoff',
                 'label_wave', 'label_noise', 'num_waveforms', 'num_noises',
                 'iterable', 'filter_duration', 'sample_rate', 'signal_low_freq_cutoff',
                 'signal_approximant', 'reference_freq', 'detectors_abbr',
                 'save_injection_priors', 'gc_collect_frequency', 
                 'sample_save_frequency', 'signal_length', 'whiten_padding',
                 'sample_length_in_s', 'sample_length_in_num', 'waveform_kwargs',
                 'psd_file_path_det1', 'psd_file_path_det2', 'noise_generator',
                 'prior_low_mass', 'prior_high_mass', 'prior_low_chirp_dist', 'prior_high_chirp_dist',
                 'tc_inject_lower', 'tc_inject_upper', 'noise_high_freq_cutoff',
                 'max_signal_length', 'ringdown_leeway', 'merger_leeway', 'start_freq_factor',
                 'fs_reduction_factor', 'fbin_reduction_factor', 'dbins', 'check_seeds',
                 'norm_tc', 'norm_dist', 'norm_dchirp', 'norm_mchirp', 'chirp_masses']
    
    def __init__(self, **kwargs):
        ## Get slots magic attributes via input dict (use **kwargs)
        # This should set all slots given above
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        """ Save dir """
        self.export_dir = ""
        self.dirs = {}
        
        """ Sanity check with gc_collect """
        # MP methods do not mix well with gc.collect
        # Always set this to -1. *DO NOT* change this.
        self.gc_collect_frequency = -1
        
        # Create the detectors
        self.detectors_abbr = ('H1', 'L1')
    
        ### Create labels
        self.label_wave = np.array([1., 0.])
        self.label_noise = np.array([0., 1.])
        
        self.iterable = np.arange(self.num_waveforms + self.num_noises)
    
        """ Generating noise function """
        ### Saving the PSDs
        self.psd_file_path_det1 = ""
        self.psd_file_path_det2 = ""
        
        ### Create the power spectral densities of the respective detectors
        if self.dataset == 1:
            psd_fun = pycbc.psd.analytical.aLIGOZeroDetHighPower
            self.psds = [psd_fun(self.psd_len, self.delta_f, self.noise_low_freq_cutoff) 
                         for _ in range(len(self.detectors_abbr))]
        else:
            # Here, we should pick two PSDs randomly and read the files
            # PSDs are obtained from the noise generator
            pass
        
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
        # Normalising values for certain outputs
        self.norm_tc = None
        self.norm_dist = None
        self.norm_dchirp = None
        self.norm_mchirp = None
        self.chirp_masses = None
        
    
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
            self.psd_file_path_det1 = os.path.abspath(save_path)
            self.psd_file_path_det2 = self.psd_file_path_det1
            # Remove and rewrite if the PSD file already exists
            if os.path.exists(self.psd_file_path_det1):
                os.remove(self.psd_file_path_det1)
            # Convert psds to pandas dataframe
            df = pd.DataFrame(data=self.psds[0].numpy(), columns=['psd_data'])
            # Save as hdf5 file with compression
            df.to_hdf(self.psd_file_path_det1, "data", complib="blosc:lz4", complevel=9, mode='a')
            # Adding all relevant attributes
            with h5py.File(self.psd_file_path_det1, 'a') as fp:
                fp.attrs['delta_f'] = self.delta_f
    
    
    def _get_norm_lambda_(self, min_val, max_val):
        # Return lambda to use for normalisation
        return lambda val: (val - min_val)/(max_val - min_val)
    
            
    def get_priors(self):
        ## Generate source prior parameters
        # A path to the .ini file.
        ini_parent = '.'
        CONFIG_PATH = "{}/ds{}.ini".format(ini_parent, self.dataset)
        random_seed = np.random.randint(low=0, high=2**32-1)
        
        # Draw num_waveforms number of samples from ds.ini file
        self.priors = draw_samples_from_config(path=CONFIG_PATH,
                                               num=self.num_waveforms, 
                                               seed=random_seed)
        
        # Get normalisation params for certain output values
        # self.norm_tc = self._get_norm_lambda_(np.min(self.priors['mass1']), 
        #                                       np.max(self.priors['mass1']))
        # self.norm_dist = self._get_norm_lambda_(np.min(self.priors['distance']), 
        #                                         np.max(self.priors['distance']))
        # self.norm_dchirp = self._get_norm_lambda_(np.min(self.priors['chirp_distance']), 
        #                                           np.max(self.priors['chirp_distance']))
        # Calculate chirp mass and get normalisation values
        m1 = self.priors['mass1']
        m2 = self.priors['mass2']
        # TODO: Add chirp_masses to numpy.record (self.priors) if possible. Cleaner.
        self.chirp_masses = (m1*m2 / (m1+m2)**2.)**(3./5) * (m1 + m2)
        # self.norm_mchirp = self._get_norm_lambda_(np.min(self.chirp_masses), 
        #                                           np.max(self.chirp_masses))
        
        """
        ## Using the samples variable to get priors
        # Print all fieldnames stored within samples variable
        print(self.priors.fieldnames)
        # Print a certain parameter, for example 'mass1'.
        print(self.priors[0]['mass1'])
        """
        
        """ Write priors """
        if self.save_injection_priors:
            # Save injection priors into injection.hdf
            inj_path = os.path.join(self.dirs['injections'], 'injections.hdf')
            # Writing injections.hdf
            with h5py.File(inj_path, 'a') as fp:
                # create a dataset for batch save
                fp.create_dataset('data', data=self.priors, 
                                  compression='gzip', compression_opts=9, 
                                  shuffle=True)
        
        
    def worker(self, idx, queues=None):
        
        """ Set the random seed for this iteration here """
        # Check whether all the processes use the same random seed
        # TODO: if we call all samples at the same time, idx alone can be used as seed
        # However, in some cases there is a memory leak that causes problems. Handle.
        seed = int(os.getpid() + time.time() + (idx+1) + np.random.randint(2**0, 2**31))
        np.random.seed(seed)
        
        """ Obtain sample """
        is_waveform = idx < self.num_waveforms
        sample = {}
        
        if not is_waveform:
            ## Generate noise
            if self.dataset == 1:
                noise = [self.noise_generator(psd).to_timeseries()[:self.sample_length_in_num] 
                          for psd in self.psds]
            else:
                noise, self.psds = self.noise_generator(0.0, self.sample_length_in_s, None)
                noise = [noise[det].numpy() for det in self.detectors_abbr]
            
            # Saving noise params for storage
            sample['noise'] = noise
            sample['label'] = self.label_noise
            sample['psd_1'] = ''
            sample['psd_2'] = ''
        
        else:
            ## Generate signal
            """ Get input params from injections.hdf """
            # Iterate through injection parmas using idx and set params to waveform_kwargs
            # prior values as created by get_priors is a np.record object
            prior_values = self.priors[idx]
            # Convert np.record object to dict and append to waveform_kwargs dict
            self.waveform_kwargs.update(dict(zip(prior_values.dtype.names, prior_values)))
            
            """ Injection """
            # Generate the full waveform
            h_plus, h_cross = pycbc.waveform.get_td_waveform(**self.waveform_kwargs)
            # Properly time and project the waveform
            start_time = prior_values['injection_time'] + h_plus.get_sample_times()[0]
            h_plus_end_time = h_plus.get_sample_times()[-1]
            # Pad h_plus and h_cross with zeros on both end for slicing
            # Prepend zeros if we need samples before signal begins
            h_plus.prepend_zeros(self.sample_length_in_num)
            h_cross.prepend_zeros(self.sample_length_in_num)
            # Set the start time of h_plus and h_plus after accounting for prepended zeros
            h_plus.start_time = start_time - self.sample_length_in_s
            h_cross.start_time = start_time - self.sample_length_in_s
            # Append zeros if we need samples after signal ends
            h_plus.append_zeros(self.sample_length_in_num)
            h_cross.append_zeros(self.sample_length_in_num)
            # Get time interval used on strains after project_wave is done
            time_placement = prior_values['tc'] + (self.whiten_padding/2.0)
            start_interval = prior_values['injection_time'] - time_placement
            # subtract delta value for length error
            end_padding = self.whiten_padding - 0.001
            end_interval = prior_values['injection_time']+(self.signal_length-time_placement)+end_padding
            # h_plus and h_cross is stored for augmentation purposes
            sample['h_plus'] = h_plus
            sample['h_cross'] = h_cross
            sample['start_time'] = h_plus.start_time
            sample['end_time'] = h_plus_end_time
            sample['label'] = self.label_wave
            sample['time_interval'] = (start_interval, end_interval)
            # waveform_kwargs will contain all prior values
            sample['inj_params'] = self.waveform_kwargs
            # Add normalised values of output params
            # sample['normalised_tc'] = self.norm_tc(prior_values['tc'])
            # sample['normalised_distance'] = self.norm_dist(prior_values['distance'])
            # sample['normalised_chirp_mass'] = self.norm_mchirp(self.chirp_masses[idx])
            # sample['normalised_chirp_distance'] = self.norm_dchirp(prior_values['chirp_distance'])
        
        
        # Pass data onto Queue to be stored in a file
        data = {}
        data['sample'] = sample
        data['seed'] = seed
        data['is_waveform'] = is_waveform
        data['idx'] = idx
        data['kill'] = False
        
        # Give all relevant save data to Queue
        qidx = np.random.randint(low=0, high=len(queues))
        queues[qidx].put(data)
        
        """ Clean up RAM (occasionally) """
        # DO NOT DO THIS WITH MP Dataset Generation (slows down rates by a LOT)
        if self.gc_collect_frequency != -1:
            raise ValueError('MP based dataset generation does not work well with gc.collect()')
    
    
    def listener(self, queue, qidx):
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
        store = os.path.join(self.dirs['dataset'], "dataset_chunk_{}.hdf".format(qidx))
        chunk_file = h5py.File(store, 'a')
        
        
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
            # seed = data['seed']
            is_waveform = data['is_waveform']
            
            """ Write sample """
            ## Save a chunk of samples in .hdf format
            # Create dataset objects for each storage array
            # 1. sample saved at full 2048. Hz sampling rate with key as '<det>/idx'
            # 2. 
            
            if is_waveform:
                key = 'h_plus/{}'.format(str(idx))
                chunk_file.create_dataset(key, data=sample['h_plus'], compression='gzip', 
                                          compression_opts=9, shuffle=True)
                key = 'h_cross/{}'.format(str(idx))
                chunk_file.create_dataset(key, data=sample['h_cross'], compression='gzip', 
                                          compression_opts=9, shuffle=True)
            
        # Close the chunk storage file
        chunk_file.close()
            
    
    def generate_dataset(self):
        """
        Create dataset using the explicit PyCBC method
        Dataset is made as close as possible to the testing dataset types
        This code is much faster to execute and easier to read
        """
        
        ## Get priors
        self.get_priors()
        
        """ Handling number of cores for task """
        num_queues_datasave = 3
        num_cores_datagen = 7
        # Sanity check
        assert num_queues_datasave + num_cores_datagen <= mp.cpu_count()
        
        # Must use Manager queue here, or will not work
        manager = mp.Manager()
        queues = [manager.Queue() for _ in range(num_queues_datasave)]
            
        # Empty jobs every iteration
        jobs = []
        
        # Initialise pool
        pool = mp.Pool(int(num_cores_datagen+num_queues_datasave))
        
        # Put listener to work first (this will wait for data in MP and write to Queue)
        watchers = [pool.apply_async(self.listener, (queue, n)) for n, queue in enumerate(queues)]
        
        for idx in self.iterable:
            job = pool.apply_async(self.worker, (idx, queues,))
            jobs.append(job)

        # Collect results from the workers through the pool result queue
        pbar = tqdm(jobs)
        for job in pbar:
            pbar.set_description('Generate Data-Chunk')
            job.get()

        # Kill the listener when all jobs are complete
        _ = [queue.put({'kill': True}) for queue in queues]
        # End the pool processes
        pool.close()
        pool.join()
            

    def _get_recursive_abspath_(self, directory):
        # Get abspath of all files within directory
        paths = np.array([])
        for dirpath, _, filenames in os.walk(os.path.abspath(directory)):
            for filename in filenames:
                 paths = np.append(paths, os.path.join(dirpath, filename))
        return paths

    def make_training_lookup(self):
        # Make a lookup table with (id, path, target) for each training data
        # Should have an equal ratio, shuffled, of both classes (signal and noise)
        # Save this data in the hdf5 format
        ### NOTE: target, tc, normalised_tc is stored within the .hdf sample file
        ### So, you need only save paths. Other params not required.
        noise_path = self.dirs['background']
        signal_path = self.dirs['foreground']
        # Get all absolute paths of files within fg and bg
        noise_abspaths = self._get_recursive_abspath_(noise_path)
        signal_abspaths = self._get_recursive_abspath_(signal_path)
        all_abspaths = np.concatenate((noise_abspaths, signal_abspaths))
        # Get the ids for each data sample
        dataset_length = len(all_abspaths)
        ids = np.linspace(0, dataset_length, dataset_length, dtype=np.int32)
        # Get the target/label value for each data sample
        # These labels *ONLY* specify whether the given sample is signal or not
        targets = [0]*len(noise_abspaths) + [1]*len(signal_abspaths)
        # Column stack (ids, path, target) for the entire dataset
        lookup = np.column_stack((ids, all_abspaths, targets))
        # Shuffle the column stack ('tc' is in ascending order, signal and noise are not shuffled)
        # NumPy: "Multi-dimensional arrays are only shuffled along the first axis"
        np.random.shuffle(lookup)
        # Convert to pandas dataframe
        df = pd.DataFrame(data=lookup, columns=["id", "path", "target"])
        # Save the dataset paths alongside the target and ids as hdf5
        self.dirs['lookup'] = os.path.join(self.export_dir, "training.hdf")
        df.to_hdf(self.dirs['lookup'], "lookup", complib="blosc:lz4", complevel=9, mode='a')
        
        with h5py.File(self.dirs['lookup'], 'a') as ds:
            ds.attrs['dataset'] = self.dataset
            ds.attrs['seed'] = self.seed
            ds.attrs['sampling_rate'] = self.sample_rate
            ds.attrs['noise_num'] = len(noise_abspaths)
            ds.attrs['signal_num'] = len(signal_abspaths)
            ds.attrs['dataset_ratio'] = len(signal_abspaths)/len(noise_abspaths)
        
        # Create a copy of training.hdf to the dataset directory
        # This file is used later if using this dataset with make_dataset=False
        shutil.copy(self.dirs['lookup'], self.dirs['parent'])
        
        print("make_training_lookup: Lookup table created successfully!")


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
    # Generate the dataset
    gd.generate_dataset()
    # Make dataset lookup table
    # gd.make_training_lookup()
    # Making the prior distribution plots
    # plot_priors(gd.dirs['parent'])
