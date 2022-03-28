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
import shutil
import logging
import os, os.path
import numpy as np
import pandas as pd

# PyCBC handling
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector
from pycbc import distributions

# LOCAL
from data.mlmdc_noise_generator import NoiseGenerator
from data.plot_default_priors import plot_priors

class GenerateData:
    
    # Slots magic for parameters input from data_configs.py
    __slots__ = ['dataset', 'parent_dir', 'data_dir', 'seed', 'export_dir', 'dirs',
                 'make_dataset', 'make_module',
                 'detectors', 'psds', 'skylocation_dist', 'np_gen',
                 'psd_len', 'delta_f', 'noise_low_freq_cutoff',
                 'label_wave', 'label_noise', 'num_waveforms', 'num_noises',
                 'iterable', 'filter_duration', 'sample_rate', 'signal_low_freq_cutoff',
                 'signal_approximant', 'reference_freq', 'detectors_abbr',
                 'save_injection_priors', 'gc_collect_frequency', 
                 'sample_save_frequency', 'signal_length', 'whiten_padding',
                 'sample_length_in_s', 'sample_length_in_num', 'waveform_kwargs',
                 'psd_file_path_det1', 'psd_file_path_det2', 'noise_generator',
                 'prior_low_mass', 'prior_high_mass', 'prior_low_chirp_dist', 'prior_high_chirp_dist',
                 'tc_inject_lower', 'tc_inject_upper', 'noise_high_freq_cutoff']
    
    def __init__(self, **kwargs):
        ## Get slots magic attributes via input dict (use **kwargs)
        # This should set all slots given above
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        """ Save dir """
        self.export_dir = ""
        self.dirs = {}
        
        """ Initialise dataset params """
        # Create the detectors
        self.detectors_abbr = ('H1', 'L1')
        self.detectors = []
        for det_abbr in self.detectors_abbr:
            self.detectors.append(pycbc.detector.Detector(det_abbr))
        
        ### Create the power spectral densities of the respective detectors
        if self.dataset == 1:
            psd_fun = pycbc.psd.analytical.aLIGOZeroDetHighPower
            self.psds = [psd_fun(self.psd_len, self.delta_f, self.noise_low_freq_cutoff) 
                         for _ in range(len(self.detectors))]
        
        ### Saving the PSDs
        self.psd_file_path_det1 = ""
        self.psd_file_path_det2 = ""
        
        ### Initialize the random distributions
        self.skylocation_dist = pycbc.distributions.sky_location.UniformSky()
        self.np_gen = np.random.default_rng()
    
        ### Create labels
        self.label_wave = np.array([1., 0.])
        self.label_noise = np.array([0., 1.])
    
        ### Generate data
        logging.info(("Generating dataset with %i injections and %i pure "
                    "noise samples") % (self.num_waveforms, self.num_noises))
        
        self.iterable = range(self.num_waveforms + self.num_noises)
        
    
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
    
    def store_ts(self, path, det, ts, force=False):
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
        
        # Saves time series in path with HDF append mode
        group = '{}/{}'.format(det, int(ts.start_time))
        ts.save(path, group=group)
    
    def generate_dataset(self):
        """ 
        Create dataset using the explicit PyCBC method
        Dataset is made as close as possible to the testing dataset types
        This code is much faster to execute and easier to read
        """
        
        psd_save_flag = True
        for i in self.iterable:
            
            print("Num sample = {}".format(i))
    
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
                    noise_generate_duration = 8. # seconds
                    noise_slice_duration = 1. # seconds
                    whitening_padding = 0.25 # seconds (two-sided)
        
                    noise_slice_start = 2048.*4.
                    noise_slice_end   = noise_slice_start + 2048.*noise_slice_duration + 2048.*whitening_padding
                    noise_length = noise_slice_end - noise_slice_start
                    noise, psds = self.noise_generator(0, noise_generate_duration, generate_duration=10)
                    noise = [foo.numpy()[noise_slice_start: noise_slice_end] for foo in noise.values()]
                    noise = np.stack(noise, axis=0)
                
                    set_delta_f = lambda psd_series: pycbc.psd.estimate.interpolate(psd_series, delta_f=1./duration)
                    psds = [pycbc.types.FrequencySeries(psd, delta_f=1./128.) for psd in psds]
                    duration = noise_length*(1./2048.)
                    psds = [set_delta_f(psd) for psd in psds]
            
            
                # Saving the PSD file used for dataset/sample
                if self.dataset == 1 and psd_save_flag:
                    psd_save_flag = False
                    # Both detectors have the same PSD for dataset 1
                    if not os.path.exists("data/psds"):
                        os.makedirs("data/psds", exist_ok=False)
                    self.psd_file_path_det1 = os.path.abspath("data/psds/psd-aLIGOZeroDetHighPower.hdf")
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
                
                if self.dataset == 2 or self.dataset == 3:
                    raise NotImplementedError("PSDs for dataset type 2 and 3 not implemented")
                    
    
            """ Generate signal """
            # If in the first part of the dataset, generate waveform
            if is_waveform:
                
                ## Generate source parameters
                """ Masses """
                masses = self.np_gen.uniform(self.prior_low_mass, self.prior_high_mass, 2)
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
                uniform_angles = uniform_angle_distr.rvs(size=3)
                sin_angle_distr = distributions.angular.SinAngle(sin_angle=(0., np.pi))
                sin_angles = sin_angle_distr.rvs(size=1)
    
                self.waveform_kwargs['coa_phase'] = uniform_angles[0][0]
                self.waveform_kwargs['inclination'] = sin_angles[0][0]
                declination, right_ascension = self.skylocation_dist.rvs()[0]
                pol_angle = uniform_angles[1][0]
                
                """ Injection """            
                # Take the injection time randomly in the LIGO O3a era
                injection_time = self.np_gen.uniform(1238166018, 1253977218)
                # Generate the full waveform
                waveform = pycbc.waveform.get_td_waveform(**self.waveform_kwargs)
                h_plus, h_cross = waveform
                # Properly time and project the waveform
                start_time = injection_time + h_plus.get_sample_times()[0]
                h_plus.start_time = start_time
                h_cross.start_time = start_time
                h_plus.append_zeros(self.sample_length_in_num)
                h_cross.append_zeros(self.sample_length_in_num)
                strains = [det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle) for det in self.detectors]
                # Place merger randomly within the window between lower and upper bound
                place_tc = self.np_gen.uniform(self.tc_inject_lower, self.tc_inject_upper)
                time_placement = place_tc + (self.whiten_padding/2.0)
                time_interval = injection_time-time_placement
                time_interval = (time_interval, injection_time+(self.signal_length-time_placement) + 
                                 self.whiten_padding-0.001) # subtract for length error
                strains = [strain.time_slice(*time_interval) for strain in strains]
                normalised_tc = (place_tc-self.tc_inject_lower)/(self.tc_inject_upper-self.tc_inject_lower)
                
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
                    
                    # CSV write injections
                    # Read this CSV using pandas (optimal)
                    inj_path = os.path.join(self.dirs['injections'], 'injections.csv')
                    with open(inj_path, 'a', newline='') as fp:
                        writer = csv.writer(fp)
                        # Writing the fields into injections.csv
                        if i == 0:
                            writer.writerow(list(_save_.keys()))
                        writer.writerow(list(_save_.values()))
                
                # Noise + Signal (injection) will *NOT* be performed for the sake of augmentation
                # Combine noise and signal when initialising the data
                # Do this simply by: sample = noise + signal (after converting to numpy)
                # Use the following to convert to numpy:
                # sample = np.stack([strain.numpy() for strain in strains], axis=0)
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
            
            # Save each sample as .hdf with appropriate attrs
            # Store the time_series using PyCBC method
            for detector, time_series in zip(self.detectors_abbr, sample):
                self.store_ts(store, detector, time_series)
            
            # Adding all relevant attributes
            with h5py.File(store, 'a') as fp:
                fp.attrs['unique_dataset_id'] = "unknown"
                fp.attrs['dataset'] = self.dataset
                fp.attrs['seed'] = self.seed
                fp.attrs['sample_rate'] = self.sample_rate
                fp.attrs['sample_length_in_s'] = self.sample_length_in_s
                fp.attrs['detectors'] = self.detectors_abbr
                # Common training parameter
                fp.attrs['label'] = label
                
                if is_waveform:
                    fp.attrs['mass_1'] = m1
                    fp.attrs['mass_2'] = m2
                    fp.attrs['signal_low_freq_cutoff'] = self.signal_low_freq_cutoff
                    # Training parameters
                    fp.attrs['tc'] = injection_time
                    fp.attrs['normalised_tc'] = normalised_tc
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
                    
                    Final Decision = uniform negative 'tc'. Experimentation required.
                    """
                    fp.attrs['tc'] = -1.0
                    fp.attrs['normalised_tc'] = -1.0
                    
            
            """ Clean up RAM (occasionally) """
            if i % self.gc_collect_frequency == 0 or i == 2.0*self.num_waveforms-1:
                gc.collect()

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
    # Generate the dataset
    gd.generate_dataset()
    # Make dataset lookup table
    gd.make_training_lookup()
    # Making the prior distribution plots
    plot_priors(gd.dirs['parent'])
