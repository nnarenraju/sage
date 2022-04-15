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
import gc
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
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector
from pycbc import distributions

# LOCAL
from data.mlmdc_noise_generator import NoiseGenerator
from data.plot_default_priors import plot_priors


""" Work-around for using unpicklable function within MP """
# Using global variables to "remove" un-picklable func from MP processes
global project_wave
# lal.Detector objects cannot be pickled (funny little things)
detectors_abbr = ('H1', 'L1')
detectors = []
for det_abbr in detectors_abbr:
    detectors.append(pycbc.detector.Detector(det_abbr))
# Faux-project_wave function calling det.project_wave
project_wave = lambda hp, hc, pol, ra, dec: [det.project_wave(hp, hc, ra, dec, pol) for det in detectors]


class GenerateData:
    
    # Slots magic for parameters input from data_configs.py
    __slots__ = ['dataset', 'parent_dir', 'data_dir', 'seed', 'export_dir', 'dirs',
                 'make_dataset', 'make_module',
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
                 'fs_reduction_factor', 'fbin_reduction_factor', 'dbins', 'check_seeds']
    
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
        # Always set this to -1. Do not change this.
        self.gc_collect_frequency = -1
        
        """ Initialise dataset params """
        # Create the detectors
        self.detectors_abbr = ('H1', 'L1')
        
        """ Seed repetition checker """
        self.check_seeds = []
        
        ### Create the power spectral densities of the respective detectors
        if self.dataset == 1:
            psd_fun = pycbc.psd.analytical.aLIGOZeroDetHighPower
            self.psds = [psd_fun(self.psd_len, self.delta_f, self.noise_low_freq_cutoff) 
                         for _ in range(len(self.detectors_abbr))]
        else:
            # Here, we should pick two PSDs randomly and read the files
            raise NotImplementedError("PSDs not implemented for D2-D4")
        
        ### Saving the PSDs
        self.psd_file_path_det1 = ""
        self.psd_file_path_det2 = ""
        
        ### Initialize the random distributions
        self.skylocation_dist = pycbc.distributions.sky_location.UniformSky()
        # Reseed for each process if using MP or just use pycbc.distributions
        self.np_gen = np.random.default_rng()
    
        ### Create labels
        self.label_wave = np.array([1., 0.])
        self.label_noise = np.array([0., 1.])
    
        ### Generate data
        logging.info(("Generating dataset with %i injections and %i pure "
                    "noise samples") % (self.num_waveforms, self.num_noises))
        
        self.iterable = np.arange(self.num_waveforms + self.num_noises)
        
    
        """ Generating noise function """
        # Fixed parameter options for iteration
        if self.dataset == 1:
            self.noise_generator = pycbc.noise.gaussian.frequency_noise_from_psd
        else:
            raise NotImplementedError("PSD function for datasets 2 and 3 have not been implemented")
            self.noise_generator = NoiseGenerator(self.dataset,
                                                  seed=self.seed,
                                                  filter_duration=self.filter_duration,
                                                  sample_rate=self.sample_rate,
                                                  low_frequency_cutoff=self.noise_low_freq_cutoff,
                                                  detectors=self.detectors_abbr)
        
        # Generate source parameters
        self.waveform_kwargs = {'delta_t': 1./self.sample_rate, 'f_lower': self.signal_low_freq_cutoff}
        self.waveform_kwargs['approximant'] = self.signal_approximant
        self.waveform_kwargs['f_ref'] = self.reference_freq
    
    
    def make_data_dir(self):
        """ Creating required dirs """
        # Make directory structure for data storage
        self.dirs['parent'] = os.path.join(self.parent_dir, self.data_dir)
        self.dirs['injections'] = os.path.join(self.dirs['parent'], "injections")
        self.dirs['background'] = os.path.join(self.dirs['parent'], "background")
        self.dirs['foreground'] = os.path.join(self.dirs['parent'], "foreground")
        
        if not os.path.exists(self.dirs['parent']):
            os.makedirs(self.dirs['parent'], exist_ok=False)
            os.makedirs(self.dirs['background'], exist_ok=False)
            os.makedirs(self.dirs['foreground'], exist_ok=False)
            
            if self.save_injection_priors:
                os.makedirs(self.dirs['injections'], exist_ok=False)
        else:
            raise IOError("make_default_dataset: Dataset dir already exists!")
    
    def store_ts(self, path, det, ts, det_num=None, force=False):
        """
        Utility function to save a time series.
        
        Arguments
        ---------
        path : str or None
            The path at which to store the time series. If None the function
            will return immediately and not save anything.
        det : str
            The detector of the time series that should be saved.
        ts : TimeSeries
            The time series to save.
        force : {bool, False}
            Overwrite existing files.
        """
        
        if path is None:
            return
        
        if not isinstance(ts, np.ndarray):
            # Saves time series in path with HDF append mode
            group = '{}/{}'.format(det, int(ts.start_time))
            ts.save(path, group=group)
        
        else:
            # There are problems if ts is not time series but np.ndarray
            # HDF5 was measured to have the fastest IO (r->46ms, w->172ms)
            # NPY read/write was not tested. Github says HDF5 is faster.
            with h5py.File(path, 'a') as fp:
                # create a dataset for batch save
                key = '{}/{}'.format(det, str(det_num))
                fp.create_dataset(key, data=ts,
                                  compression='gzip',
                                  compression_opts=9, shuffle=True)
    
    
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
        
        
    def worker(self, i, queue):
        
        """ Set the random seed for this iteration here """
        # Check whether all the processes use the same random seed
        seed = int(os.getpid() + time.time() + (i+1) + np.random.randint(2**0, 2**31))
        np.random.seed(seed)
        
        is_waveform = i < self.num_waveforms
        
        """ Generate noise """
        if not is_waveform:
            if self.dataset == 1:
                noise = [self.noise_generator(psd).to_timeseries()[:self.sample_length_in_num] 
                          for psd in self.psds]
                
                """
                To convert to numpy before storing:
                noise = [self.noise_generator(psd).to_timeseries().numpy()[:self.sample_length_in_num] 
                         for psd in self.psds]
                noise = np.stack(noise, axis=0)
                """
                
            else:
                raise NotImplementedError("make_default_dataset: noise gen not implemented for D2,D3")
                

        """ Generate signal """
        # If in the first part of the dataset, generate waveform
        if is_waveform:
            
            ## Generate source parameters
            """ Masses """
            mass_gen = distributions.Uniform(mass=(self.prior_low_mass, self.prior_high_mass))
            masses = mass_gen.rvs(size=2)
            masses = [float(masses[0][0]), float(masses[1][0])]
            m1, m2 = max(masses), min(masses)
            self.waveform_kwargs['mass1'] = m1
            self.waveform_kwargs['mass2'] = m2
            
            """ Distance and Chirp Distance """
            # Adding distance measure
            mchirp = (m1*m2 / (m1+m2)**2.)**(3./5) * (m1 + m2)
            dist_gen = distributions.power_law.UniformRadius
            chirp_distance_dist = dist_gen(distance=(self.prior_low_chirp_dist, self.prior_high_chirp_dist))
            chirp_distance = chirp_distance_dist.rvs(size=1)
            chirp_distance = float(chirp_distance[0][0])
            # Storing priors
            distance = chirp_distance * (2.**(-1./5) * 1.4 / mchirp)**(-5./6)
            self.waveform_kwargs['distance'] = distance
            
            """ Coalescence Phase, Inclination, ra, dec and Polarisation Angle """
            uniform_angle_distr = distributions.angular.UniformAngle(uniform_angle=(0., 2.0*np.pi))
            uniform_angles = uniform_angle_distr.rvs(size=2)
            sin_angle_distr = distributions.angular.SinAngle(sin_angle=(0., np.pi))
            sin_angles = sin_angle_distr.rvs(size=1)

            self.waveform_kwargs['coa_phase'] = uniform_angles[0][0]
            self.waveform_kwargs['inclination'] = sin_angles[0][0]
            declination, right_ascension = self.skylocation_dist.rvs()[0]
            pol_angle = uniform_angles[1][0]
            
            """ Injection """            
            # Take the injection time randomly in the LIGO O3a era
            inj_gen = distributions.Uniform(injection=(1238166018, 1253977218))
            injection_time = np.int64(inj_gen.rvs(size=1)[0][0])
            # Generate the full waveform
            waveform = pycbc.waveform.get_td_waveform(**self.waveform_kwargs)
            h_plus, h_cross = waveform
            # Properly time and project the waveform
            start_time = injection_time + h_plus.get_sample_times()[0]
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
            
            global project_wave
            strains = project_wave(h_plus, h_cross, pol_angle, right_ascension, declination)
            
            # Place merger randomly within the window between lower and upper bound
            tc_gen = distributions.Uniform(tc=(self.tc_inject_lower, self.tc_inject_upper))
            place_tc = np.float64(tc_gen.rvs(size=1)[0][0])
            time_placement = place_tc + (self.whiten_padding/2.0)
            time_interval = injection_time-time_placement
            time_interval = (time_interval, injection_time+(self.signal_length-time_placement) + 
                              self.whiten_padding-0.001) # subtract for length error
            # Checking whether end time of time interval contains the end of simulated ringdown
            if time_interval[1] - injection_time <= h_plus_end_time:
                raise ValueError("generate_dataset: end time of slice is less than ringdown end time!")
            
            normalised_tc = (place_tc-self.tc_inject_lower)/(self.tc_inject_upper-self.tc_inject_lower)
            
            # Get time slices strains
            strains = [strain.time_slice(*time_interval) for strain in strains]
            
            # Sanity check for sample_length
            for strain in strains:
                to_append = self.sample_length_in_num - len(strain)
                if to_append>0:
                    strain.append_zeros(to_append)
                if len(strain) != self.sample_length_in_num:
                    raise ValueError("Sample length greater than expected!")
            
            
            """ Saving the injection parameters """
            if self.save_injection_priors:
                # Storing priors
                _save_ = {}
                _save_['chirp_distance'] = chirp_distance
                _save_['coa_phase'] = uniform_angles[0][0]
                _save_['dec'] = declination
                _save_['ra'] = right_ascension
                _save_['distance'] = distance
                _save_['inclination'] = sin_angles[0][0]
                _save_['mass_1'] = m1
                _save_['mass_2'] = m2
                _save_['mchirp'] = mchirp
                _save_['polarisation'] = pol_angle
                _save_['tc'] = injection_time
                _save_['normalised_tc'] = normalised_tc
            
            # Noise + Signal (injection) will *NOT* be performed for the sake of augmentation
            # Combine noise and signal when initialising the data
            # Do this simply by: sample = noise + signal (after converting to numpy)
            # Use the following to convert to numpy:
            # sample = np.stack([strain.numpy() for strain in strains], axis=0)
            # Update (Apr 10, 2022): h_plus and h_cross are being stored instead of h_t
            # This is in favour of augmenting on polarisation angle, ra, dec and distance
            # Update (Apr 12, 2022): We went back to h_t and project_wave is parallelised
            sample = strains
            
        # if in the second part of the dataset, merely use pure noise as the full sample
        else:
            sample = noise
        
        """ Save sample (signal/noise) with label, tc and snr && PSD """
        if is_waveform:
            label = self.label_wave
            # Path to save sample (pure signal)
            store = os.path.join(self.dirs['foreground'], "foreground_{}.hdf".format(i))
        else:
            label = self.label_noise
            # Path to save sample (pure noise)
            store = os.path.join(self.dirs['background'], "background_{}.hdf".format(i))
        
        # Pass data onto Queue to be stored in a file
        data = {}
        data['sample'] = sample
        data['store_path'] = store
        data['label'] = label
        data['is_waveform'] = is_waveform
        data['kill'] = False
        data['seed'] = seed
        
        # Prior attrs
        if self.save_injection_priors and is_waveform:
            data['prior_data'] = _save_
        
        # Data specific attrs
        if is_waveform:
            data['m1'] = m1
            data['m2'] = m2
            data['distance'] = distance
            data['time_interval'] = time_interval
            data['injection_time'] = injection_time
            data['normalised_tc'] = normalised_tc
        
        # Give all relevant save data to Queue
        queue.put(data)
        
        """ Clean up RAM (occasionally) """
        # DO NOT DO THIS WITH MP Dataset Generation (slows down rates by a LOT)
        if self.gc_collect_frequency != -1:
            if i % self.gc_collect_frequency == 0 or i == 2.0*self.num_waveforms-1:
                gc.collect()
    
    
    def listener(self, queue):
        """
        Write given sample data to a HDF file
        We also use this to append to priors
        
        ** WARNING!!! **: If files are not being written, there should be an error in here.
        This error will not be raised and has not yet been handled properly.
        
        Vague check for this type of error:
            1. Print the queue.get() within the infinite loop.
            2. If this prints a singular output, then there is something wrong within listener.
            3. Then print a bunch of Lorem Ipsum here and there
            4. Print statements after the error occcurence will not be displayed
            5. Narrow down the error location and shoot your shot! (Sorry :P)
            
        """
        
        # Continuously check for data to the Queue
        
        while True:
            data = queue.get()
            """ The following is run ONLY if we get something from Queue """
            
            # if processes have ended, kill
            if data['kill']:
                break
            
            # Segregate data from Queue
            sample = data['sample']
            label = data['label']
            store = data['store_path']
            is_waveform = data['is_waveform']
            seed = data['seed']
            if seed in self.check_seeds:
                warnings.warn("Random seed seen more than once! This sample will not be saved.")
                continue
            
            self.check_seeds.append(seed)
            
            
            """ Write priors """
            if self.save_injection_priors and is_waveform:
                prior_data = data['prior_data']
                # CSV write injections
                # Read this CSV using pandas (optimal)
                inj_path = os.path.join(self.dirs['injections'], 'injections.csv')
                
                # Write the field names right after creating file
                # this ensures that it is not written randomly based on MP processes
                write_field_names = not os.path.exists(inj_path)
                
                with open(inj_path, 'a', newline='') as fp:
                    writer = csv.writer(fp)
                    # Writing the fields into injections.csv
                    if write_field_names:
                        writer.writerow(list(prior_data.keys()))
                    writer.writerow(list(prior_data.values()))
            
            
            """ Write sample """
            # Save each sample as .hdf with appropriate attrs
            for n, (detector, time_series) in enumerate(zip(self.detectors_abbr, sample)):
                self.store_ts(store, detector, time_series, det_num=n)
            
            # Adding all relevant attributes
            with h5py.File(store, 'a') as fp:
                fp.attrs['unique_dataset_id'] = "NotImplementedError"
                fp.attrs['dataset'] = self.dataset
                fp.attrs['seed'] = self.seed
                fp.attrs['sample_rate'] = self.sample_rate
                fp.attrs['sample_length_in_s'] = self.sample_length_in_s
                fp.attrs['detectors'] = self.detectors_abbr
                # Common training parameter
                fp.attrs['label'] = label
                
                if is_waveform:
                    fp.attrs['mass_1'] = data['m1']
                    fp.attrs['mass_2'] = data['m2']
                    fp.attrs['distance'] = data['distance']
                    fp.attrs['time_interval'] = data['time_interval']
                    fp.attrs['signal_low_freq_cutoff'] = self.signal_low_freq_cutoff
                    # Training parameters
                    fp.attrs['tc'] = data['injection_time']
                    fp.attrs['normalised_tc'] = data['normalised_tc']
                else:
                    fp.attrs['psd_file_path_det1'] = self.psd_file_path_det1
                    fp.attrs['psd_file_path_det2'] = self.psd_file_path_det2
                    fp.attrs['noise_low_freq_cutoff'] = self.noise_low_freq_cutoff
                    # Training parameters
                    # Use a single linear layer trained only on 'tc'
                    # This layer should be concatenated to the larger network once trained
                    # This should have a ReLU output. 'tc' can be constant or uniform random
                    # but always negative. Which one do we chose?
                    """
                    constant negative - not much to learn, but the boundary between proper 'tc' and
                                        invalid 'tc' is ambiguous
                    uniform random negative - model might learn superfluous parameter, but a clear 
                                        boundary can be set between proper 'tc' and invalid 'tc'
                    Nothing - we give the noise case no 'tc' and this will not be used in the loss
                              at all. (We use this!)
                    """
                    fp.attrs['tc'] = -1.0
                    fp.attrs['normalised_tc'] = -1.0
    
    
    def generate_dataset(self):
        """
        Create dataset using the explicit PyCBC method
        Dataset is made as close as possible to the testing dataset types
        This code is much faster to execute and easier to read
        """

        # Run workers to generate samples
        # Split the iterable into several small iterables
        file_size = 0.7 # MB
        limit_RAM = 1000. # MB
        # Number of files in each split
        num_files = limit_RAM/file_size
        # Number of splits to iterable
        num_splits = int(math.ceil(len(self.iterable)/num_files))
        
        print("\nBatching the dataset (if needed) to limit virtual memory usage")
        iterables = np.array_split(self.iterable, num_splits)
        
        # Must use Manager queue here, or will not work
        manager = mp.Manager()
        queue = manager.Queue()
        
        for niter, iterable in enumerate(iterables):
            
            # Empty jobs every iteration
            jobs = []
            
            # Initialise pool
            pool = mp.Pool(int(8))
            # Put listener to work first (this will wait for data in MP and write to Queue)
            watcher = pool.apply_async(self.listener, (queue,))
            
            for i in iterable:
                job = pool.apply_async(self.worker, (i, queue))
                jobs.append(job)
    
            # Collect results from the workers through the pool result queue
            pbar = tqdm(jobs)
            for job in pbar:
                pbar.set_description("Running GenerateData (Batch {}/{})".format(niter+1, num_splits))
                job.get()

            # Kill the listener when all jobs are complete
            queue.put({'kill': True})
            pool.close()
            pool.join()
            
            # Sleeping for a bit (helps keep the performance up)
            time.sleep(5)
            

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
    gd.make_training_lookup()
    # Making the prior distribution plots
    plot_priors(gd.dirs['parent'])
