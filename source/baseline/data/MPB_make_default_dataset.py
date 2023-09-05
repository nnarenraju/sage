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
import requests
import itertools
import tracemalloc
import numpy as np
import configparser
import multiprocessing as mp

# Prettification
from tqdm import tqdm

# Plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

# PyCBC handling
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.detector

from pycbc.psd import interpolate, welch
from pycbc.distributions.power_law import UniformRadius
from pycbc.distributions.utils import draw_samples_from_config
from pycbc.types import load_frequencyseries, complex_same_precision_as, FrequencySeries

# LOCAL
from data.testdata_slicer import Slicer
from data.prior_modifications import *
from data.mlmdc_noise_generator import NoiseGenerator
from data.real_noise_datagen import RealNoiseGenerator

# Addressing HDF5 error with file locking (used to address PSD file read error)
# Issue (October 1st, 2022): File locking takes place with MP even with this option
# Fix (October 5th, 2022): Reading all PSD files and storing data as global var (so as to not mess up MP)
# Deprecation (October 17th, 2022): This option is no longer being used within MPB datagen.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def psd_to_asd(psd, start_time, end_time,
               sample_rate=2048.,
               low_frequency_cutoff=15.0,
               filter_duration=128):
    
    psd = psd.copy()

    flen = int(sample_rate / psd.delta_f) // 2 + 1
    oldlen = len(psd)
    psd.resize(flen)

    # Want to avoid zeroes in PSD.
    max_val = psd.max()
    for i in range(len(psd)):
        if i >= (oldlen-1):
            psd.data[i] = psd[oldlen - 2]
        if psd[i] == 0:
            psd.data[i] = max_val

    fil_len = int(filter_duration * sample_rate)
    wn_dur = int(end_time - start_time) + 2 * filter_duration
    if psd.delta_f >= 1. / (2.*filter_duration):
        # If the PSD is short enough, this method is less memory intensive than
        # resizing and then calling inverse_spectrum_truncation
        psd = pycbc.psd.interpolate(psd, 1.0 / (2. * filter_duration))
        # inverse_spectrum_truncation truncates the inverted PSD. To truncate
        # the non-inverted PSD we give it the inverted PSD to truncate and then
        # invert the output.
        psd = 1. / pycbc.psd.inverse_spectrum_truncation(
                                1./psd,
                                fil_len,
                                low_frequency_cutoff=low_frequency_cutoff,
                                trunc_method='hann')
        psd = psd.astype(complex_same_precision_as(psd))
        # Zero-pad the time-domain PSD to desired length. Zeroes must be added
        # in the middle, so some rolling between a resize is used.
        psd = psd.to_timeseries()
        psd.roll(fil_len)
        psd.resize(int(wn_dur * sample_rate))
        psd.roll(-fil_len)
        # As time series is still mirrored the complex frequency components are
        # 0. But convert to real by using abs as in inverse_spectrum_truncate
        psd = psd.to_frequencyseries()

    kmin = int(low_frequency_cutoff / psd.delta_f)
    psd[:kmin].clear()
    asd = (psd.squared_norm())**0.25
    return asd


def get_complex_asds():
    # Save complex PSDs or ASDs as global variables
    # Since this is not within the multiprocessing block, it will not be counted towards shared RAM
    psd_options = {'H1': [f'./data/psds/H1/psd-{i}.hdf' for i in range(20)],
                   'L1': [f'./data/psds/L1/psd-{i}.hdf' for i in range(20)]}
    # Iterate through all PSD files for detector and compute the median PSD
    complex_asds = {'H1': [], 'L1': []}
    detectors_abbr = ['H1', 'L1']
    for i, det in enumerate(detectors_abbr):
        # Read all detector PSDs as frequency series with appropriate delta_f
        for psd_det in psd_options[det]:
            psd = load_frequencyseries(psd_det)
            psd = interpolate(psd, 1.0/25.0)
            # Convert PSD's to ASD's for colouring the white noise
            foo = psd_to_asd(psd, 0.0, 25.0,
                             sample_rate=2048.,
                             low_frequency_cutoff=15.0,
                             filter_duration=25.0)
            complex_asds[det].append(foo)
    
    return complex_asds


# Get complex asds
global asds
asds = get_complex_asds()



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
                 'norm_tc', 'norm_dist', 'norm_dchirp', 'norm_mchirp', 'norm_q', 'norm_invq',
                 'complex_asds', 'psd_est_segment_length', 'psd_est_segment_stride', 'blackout_max_ratio',
                 'globtmp', 'network_sample_length', '_decimated_bins', 'corrupted_len',
                 'mixed_noise', 'mix_ratio', 'use_d3_psds_for_d4', 'modification']
    
    
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
        elif self.dataset in [2, 3, 4]:
            # Here, we should pick two PSDs randomly and read the files
            # PSDs are obtained from the noise generator
            self.psds = []
            self.psd_names = ['median_det1', 'median_det2']
        
        # Fixed parameter options for iteration
        if self.dataset == 1:
            self.noise_generator = pycbc.noise.gaussian.frequency_noise_from_psd
        elif self.dataset in [2, 3, 4]:
            self.noise_generator = None
        
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
        self.norm_invq = None
        
        ## Save tmp tree structure for HDF5 file
        self.hdf5_tree = []
        self.groups = ['2048']
        self.tmp = None
        
        ## Names
        # All possible params to add into the chunk files
        # ('mass1', 'mass2', 'ra', 'dec', 'inclination', 'coa_phase', 'polarization', 'chirp_distance', 
        # 'spin1_a', 'spin1_azimuthal', 'spin1_polar', 'spin2_a', 'spin2_azimuthal', 'spin2_polar', 
        # 'tc', 'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z', 'mchirp', 'q', 'distance')
        # invq is not a part of priors but is included explicitly into sample var in worker
        # All datasets that need to be created from the sample dict
        self.waveform_names = ['h_plus', 'h_cross', 'start_time', 'interval_lower', 'interval_upper',
                               'norm_tc', 'norm_dist', 'norm_mchirp', 'norm_dchirp', 'norm_q', 'norm_invq',
                               'mass1', 'mass2', 'distance', 'mchirp', 'tc', 'chirp_distance', 'q', 'invq', 
                               'label']
        
        self.noise_names = ['noise_1', 'noise_2', 'label']
        
        # Create datasets for each field that needs to be saved
        self.max_nsamp_signal = int(self.chunk_size[0]/self.num_queues_datasave)
        self.max_nsamp_noise = int(self.chunk_size[1]/self.num_queues_datasave)

        # Global tmp
        self.globtmp = None
    
    
    def __str__(self):
        return 'MP based Batch Data Generation'
    
    
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
        
    
    def _save_(self, save_name, psd_data, psd_name):
        ## Store the PSD in HDF5 format
        psd_save_dir = os.path.join(self.dirs['parent'], "psds")
        if not os.path.exists(psd_save_dir):
            os.makedirs(psd_save_dir, exist_ok=False)
        save_path = os.path.join(psd_save_dir, save_name)
        psd_file_path = os.path.abspath(save_path)
        # Remove and rewrite if the PSD file already exists
        if os.path.exists(psd_file_path):
            os.remove(psd_file_path)
        # Write PSD in HDF5 format
        with h5py.File(psd_file_path, 'a') as fp:
            data = psd_data
            key = 'data'
            fp.create_dataset(key, data=data, compression='gzip', 
                              compression_opts=9, shuffle=True)
            # Adding all relevant attributes
            fp.attrs['delta_f'] = self.delta_f
            fp.attrs['name'] = psd_name
    

    def _MP_PSD_D4_(self, idx, slicer, sample_rate, ndet):
        # Get slicer data for an hour
        O3a_real_noise_split, _ = slicer.__getitem__(idx)
        delta_t = 1.0/sample_rate
        O3a_real_noise_split = pycbc.types.TimeSeries(O3a_real_noise_split[ndet], delta_t=delta_t)
        # PSD estimation using the welch's method
        seg_len = int(self.psd_est_segment_length / delta_t)
        seg_stride = int(seg_len * self.psd_est_segment_stride)
        estimated_psd = welch(O3a_real_noise_split, seg_len=seg_len, seg_stride=seg_stride)
        estimated_psd = interpolate(estimated_psd, self.delta_f)
        return estimated_psd
    
    
    def save_PSD(self):
        # Saving the PSD file used for dataset/sample
        if self.dataset == 1:
            save_name = "psd-aLIGOZeroDetHighPower.hdf"
            psd_data = self.psds[0].numpy()
            psd_name = 'aLIGOZeroDetHighPower'
            self._save_(save_name, psd_data, psd_name)
            
        elif self.dataset in [2, 3] or self.use_d3_psds_for_d4:
            ## Save the median PSD from the PSD files provided
            psd_options = {'H1': [f'./data/psds/H1/psd-{i}.hdf' for i in range(20)],
                           'L1': [f'./data/psds/L1/psd-{i}.hdf' for i in range(20)]}
            # Frequencies in the PSD
            f = np.linspace(self.noise_low_freq_cutoff, self.noise_high_freq_cutoff, self.psd_len)
            
            # Iterate through all PSD files for detector and compute the median PSD
            for det in self.detectors_abbr:
                # Plotting
                plt.figure(figsize=(32.0, 12.0))
                # Read all detector PSDs as frequency series with appropriate delta_f
                det_psds = []
                sample_frequencies = None
                for psd_det in psd_options[det]:
                    psd = load_frequencyseries(psd_det)
                    psd = interpolate(psd, self.delta_f)
                    sample_frequencies = psd.sample_frequencies
                    psd = psd.numpy()
                    det_psds.append(psd)
                    plt.plot(f, psd, linewidth=2.0, color='red')
                    
                # Compute the median of all PSDs along axis=0
                median_psd = np.median(det_psds, axis=0)
                max_psd = np.maximum.reduce(det_psds)

                # Remove large variation regions from the median PSD when compared to max PSD
                ratio_psd = max_psd / median_psd
                # Black out all regions where the ratio is greater than threshold
                blackout_idxs = np.argwhere(ratio_psd > self.blackout_max_ratio).flatten()
                # EXPERIMENTAL: Remove low frequency components from notching
                # max_lf_idx = max(np.argwhere(sample_frequencies < 100.0).flatten())
                # blackout_idxs = blackout_idxs[blackout_idxs >= max_lf_idx]
                # Ratio is freq bins blacked out
                del_ratio = len(blackout_idxs)/len(ratio_psd)
                print('Percentage of deleted bins while blacking out (D2,D3) = {}%'.format(del_ratio*100))
                # Create blacked PSD
                median_psd[blackout_idxs] = 999_999_999_999.0 # sqrt is about 1e6
                blacked_psd = median_psd
                
                ### Plotting
                plt.title('PSDs and their median for detector {} (removed = {}%)'.format(det, round(del_ratio*100.0, 3)))
                plt.yscale('log')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('PSD Magnitude')
                plt.grid(True)
                plt.plot(f, median_psd, linewidth=3.0, label='{} median PSD'.format(det), color='k')
                for idx in blackout_idxs:
                    plt.plot([f[idx], f[idx]], [1e-49, 1e-38], linewidth=1.0, color='gray')
                plt.ylim(1e-49, 1e-38)
                plt.legend()
                save_png = os.path.join(self.dirs['parent'], 'median_PSDs_{}_full.png'.format(det))
                plt.savefig(save_png)
                plt.close()
                
                # Save the PSD in HDF5 format
                save_name = 'psd-median-{}.hdf'.format(det)
                psd_name = 'median_det{}'.format(1 if det=='H1' else 2)
                self._save_(save_name, blacked_psd, psd_name)
                
        elif self.dataset == 4 and not self.use_d3_psds_for_d4:
            ## Estimation of PSD for dataset 4
            # Calculate the PSD for each day of the 81 days of real O3a noise data
            # It is necessary that we use only 51 days of this noise that we DO NOT use for testing
            noise_dir = os.path.join(self.parent_dir, 'O3a_real_noise')
            training_real_noise_file = os.path.join(noise_dir, 'training_real_noise.hdf')
            # Get the required attributes from real noise file
            with h5py.File(training_real_noise_file, 'r') as fp:
                assert fp.attrs['dataset'] == 4, "Wrong dataset accessed for D4 PSD calculation!"
                assert fp.attrs['mode'] == 'training', "Data Leakage: File accessed is not in training mode!"
                sample_rate = fp.attrs['sample_rate']
                training_duration = fp.attrs['duration']
                
            step_size = 3600.0 # seconds (1 hour)
            slice_length = 3600.0 # seconds (1 hour)
            assert step_size <= training_duration, "Step size chosen is larger than total O3a real noise training dataset!"
            # Initialise Slicer object and create noise generator
            # In the following, peak_offset is just an arbitrary value that works. We don't need it here.
            # If slice_length == step_size, there is no overlap between sample
            with h5py.File(training_real_noise_file, 'r') as infile:
                kwargs = dict(infile=infile, 
                              step_size=step_size, 
                              peak_offset=18.0,
                              whiten_padding=0.0,
                              slice_length=int(slice_length*self.sample_rate))
            
                # The slicer object can take an index and return the required training data sample
                slicer = Slicer(**kwargs)
            
                # Use this to iterate over slicer using sequential idx
                slicer_length = len(slicer)
            
                # Frequencies in the PSD
                f = np.linspace(self.noise_low_freq_cutoff, self.noise_high_freq_cutoff, self.psd_len)
                
                for ndet, det in enumerate(self.detectors_abbr):
                    # Plotting
                    fig = plt.figure(figsize=(32.0, 12.0))
                    ax = fig.add_subplot(111)
            
                    # The median of all the PSDs alongside a blacking out of bad frequencies
                    all_psds = []

                    print("Estimating PSD for real O3a noise using training dataset")
                    pbar = tqdm(range(slicer_length))
                    worst_psd = [0.0]
                    for idx in pbar:
                        pbar.set_description("D4 PSD Estimation")
                        estimated_psd = self._MP_PSD_D4_(idx, slicer, sample_rate, ndet)
                        # Save the estimated PSDs for D4
                        if self.mixed_noise:
                            d4_psds = os.path.join(noise_dir, 'd4_psds')
                            if not os.path.exists(d4_psds):
                                os.makedirs(d4_psds)
                            save_name = "real_psd_{}_{}.hdf".format(det, idx)
                            save_path = os.path.join(d4_psds, save_name)
                            psd_file_path = os.path.abspath(save_path)
                            # Remove and rewrite if the PSD file already exists
                            if os.path.exists(psd_file_path):
                                os.remove(psd_file_path)
                            # Write PSD in HDF5 format
                            with h5py.File(psd_file_path, 'a') as fp:
                                data = estimated_psd
                                key = 'data'
                                fp.create_dataset(key, data=data, compression='gzip',
                                                  compression_opts=9, shuffle=True)
                                # Adding all relevant attributes
                                fp.attrs['delta_f'] = self.delta_f
                        
                        all_psds.append(estimated_psd)
                        ax.plot(f, estimated_psd, linewidth=1.0, alpha=0.3)
                        if max(estimated_psd) > max(worst_psd):
                            worst_psd = estimated_psd
            
                    # Compute the median of all PSDs along axis=0
                    median_psds = np.median(all_psds, axis=0)
                    max_psds = np.maximum.reduce(all_psds)
            
                    # Blacking out of unwanted bands in the frequency spectrum
                    # Remove large variation regions from the median PSD when compared to max PSD
                    ratio_psds = max_psds / median_psds
                    # Black out all regions where the ratio is greater than threshold
                    blackout_idxss = np.argwhere(ratio_psds > self.blackout_max_ratio).flatten()
                    # Ratio is freq bins blacked out
                    del_ratios = len(blackout_idxss)/len(ratio_psds)
                    print('Percentage of deleted bins while blacking out (D4 - detector {}) = {}%'.format(det, del_ratios*100))

                    # Create blacked PSD
                    blacked_psds = np.copy(median_psds)
                    blacked_psds[blackout_idxss] = 999_999_999_999.0 # sqrt is about 1e6
            
                    ### Plotting
                    plt.title('PSDs and their median for detector {} (removed = {}%)'.format(det, round(del_ratios*100.0, 3)))
                    plt.yscale('log')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('PSD Magnitude')
                    plt.grid(True)
                    plt.plot(f, median_psds, linewidth=2.0, label='{} median PSD'.format(det), color='k')
                    plt.ylim(1e-49, 1e-38)
                    plt.legend()
                    save_png = os.path.join(self.dirs['parent'], 'median_PSDs_{}_full.png'.format(det))
                    plt.savefig(save_png)
                    plt.close()
            
                    # Save the PSD in HDF5 format
                    save_name = 'psd-worst-{}.hdf'.format(det)
                    psd_name = 'median_det{}'.format(ndet+1)
                    self._save_(save_name, worst_psd, psd_name)
                    
                    # Save the PSD in HDF5 format
                    save_name = 'psd-median-{}.hdf'.format(det)
                    psd_name = 'median_det{}'.format(ndet+1)
                    self._save_(save_name, blacked_psds, psd_name)
            
            
    def distance_from_chirp_distance(self, chirp_distance, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the luminosity distance given a chirp distance and chirp mass.
        return chirp_distance * (2.**(-1./5) * ref_mass / mchirp)**(-5./6)
    
    
    def get_priors(self):
        ## Generate source prior parameters
        
        # A path to the .ini file.
        ini_parent = './data/ini_files'
        CONFIG_PATH = "{}/ds{}.ini".format(ini_parent, self.dataset)
        random_seed = self.seed

        ## Sanity check
        # data_config and ini_file both have definitions of tc range
        # There may be a discrepancy between the two
        # NOTE: Change this such that we only need to define it in one spot
        config_reader = configparser.ConfigParser()
        config_reader.read(CONFIG_PATH)
        assert float(config_reader['prior-tc']['min-tc']) == self.tc_inject_lower, 'min-tc in ini file {} different from tc_inject_lower {} in data_config'.format(config_reader['prior-tc']['min-tc'], self.tc_inject_lower)
        assert float(config_reader['prior-tc']['max-tc']) == self.tc_inject_upper, 'max-tc in ini file {} different from tc_inject_upper {} in data_config'.format(config_reader['prior-tc']['max-tc'], self.tc_inject_upper)
        
        # Draw num_waveforms number of samples from ds.ini file
        priors = draw_samples_from_config(path=CONFIG_PATH,
                                          num=self.num_waveforms,
                                          seed=random_seed)
        
        ## MODIFICATIONS ##
        if self.modification != None and isinstance(self.modification, str):
            # Check for known modifications
            if self.modification in ['uniform_signal_duration', 'uniform_chirp_mass']:
                _mass1, _mass2 = get_uniform_masses(self.prior_low_mass, self.prior_high_mass, self.num_waveforms)
                # Masses used for mass ratio need not be used later as mass1 and mass2
                # We calculate them again after getting chirp mass
                q = q_from_uniform_mass1_mass2(_mass1, _mass2)

            if self.modification == 'uniform_signal_duration':
                # Get chirp mass from uniform signal duration
                tau_lower, tau_upper = get_tau_priors(self.prior_low_mass, self.prior_high_mass, self.signal_low_freq_cutoff)
                mchirp = mchirp_from_uniform_signal_duration(tau_lower, tau_upper, self.num_waveforms, self.signal_low_freq_cutoff)

            elif self.modification == 'uniform_chirp_mass':
                # Get uniform chirp mass
                mchirp = get_uniform_mchirp(self.prior_low_mass, self.prior_high_mass, self.num_waveforms)

            else:
                raise ValueError("get_priors: Unknown modification added to data_cfg.modification!")

            if self.modification in ['uniform_signal_duration', 'uniform_chirp_mass']:
                # Using mchirp and q, get mass1 and mass2
                mass1, mass2 = mass1_mass2_from_mchirp_q(mchirp, q)
                # Get chirp distances using the power law distribution
                chirp_distance_distr = UniformRadius(distance=(self.prior_low_chirp_dist, self.prior_high_chirp_dist))
                dchirp = np.asarray([chirp_distance_distr.rvs()[0][0] for _ in range(self.num_waveforms)])
                # Get distance from chirp distance and chirp mass
                distance = self.distance_from_chirp_distance(dchirp, mchirp)
                ## Update Priors
                priors['q'] = q
                priors['mchirp'] = mchirp
                priors['mass1'] = mass1
                priors['mass2'] = mass2
                priors['chirp_distance'] = dchirp
                priors['distance'] = distance
                
        elif self.modification != None and not isinstance(self.modification, str):
            raise ValueError("get_priors: Unknown data type used in data_cfg.modification!")

        ## Get normalisation params for certain output values ##
        
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
        
        # Normalise mass ratio (m1/m2 is mass ratio 'q')
        # This is as per PyCBC definition
        # m2 is always less than m1, and as an approx. we keep min ratio as m/m=1.0
        # max ratio will just be (mu/ml) --> max ratio = 50/7 ~ 7 
        # The range can be written as --> (min_val, max_val]
        self.norm_q = Normalise(min_val=1.0, max_val=mu/ml)
        
        # Define inv_q as well (m2/m1)
        self.norm_invq = Normalise(min_val=0.0, max_val=1.0)
        
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
       

    def download_data(self, path, resume=True):
        """
        Download noise data from the central server.
        
        Arguments
        ---------
        path : {str or None, None}
            Path at which to store the file. Must end in `.hdf`. If set to
            None a default path will be used.
        resume : {bool, True}
            Resume the file download if it was interrupted.
            
        """
        
        # Sanity check
        if os.path.exists(path):
            print('O3a real noise file already exists! Your WiFi thanks you.')
            return
        else:
            # Issue a warning before downloading the large file (~96 GB)
            print('\nWARNING: About to download ~96 GB of real O3a noise.')
            print("WARNING (contd.): Hide yo wives and get yo knives, a thicc boi is comin to town.")
        
        assert os.path.splitext(path)[1] == '.hdf'
        url = 'https://www.atlas.aei.uni-hannover.de/work/marlin.schaefer/MDC/real_noise_file.hdf'
        header = {}
        resume_size = 0
        if os.path.isfile(path) and resume:
            mode = 'ab'
            resume_size = os.path.getsize(path)
            header['Range'] = f'bytes={resume_size}-'
        else:
            mode = 'wb'
        with open(path, mode) as fp:
            response = requests.get(url, stream=True, headers=header)
            total_size = response.headers.get('content-length')

            if total_size is None:
                print("No file length found")
                fp.write(response.content)
            else:
                total_size = int(total_size)
                desc = f"Downloading real_noise_file.hdf to {path}"
                print(desc)
                with tqdm.tqdm(total=int(total_size),
                               unit='B',
                               unit_scale=True,
                               dynamic_ncols=True,
                               desc="Progress: ",
                               initial=resume_size) as progbar:
                    for data in response.iter_content(chunk_size=4000):
                        fp.write(data)
                        progbar.update(4000)
        
    
    def optimise_fmin(self, h_pol):
        # Use self.waveform_kwargs to calculate the fmin for given params
        # Such that the length of the sample is atleast 20s by the time it reaches fmin
        # This DOES NOT mean we produce signals that are exactly 20s long
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
        seed = int(idx+1)
        np.random.seed(seed)
        
        """ Obtain sample """
        is_waveform = idx < self.num_waveforms
        sample = {}
        
        if not is_waveform:
            ## Generate noise
            maxlen = round(self.sample_length_in_num)
            if self.dataset == 1:
                noise = [self.noise_generator(psd).to_timeseries()[:maxlen]
                          for psd in self.psds]
            
            elif self.dataset in [2, 3]:
                global asds
                self.noise_generator = NoiseGenerator(self.dataset,
                                                      seed=int(idx+1),
                                                      delta_f=self.delta_f,
                                                      sample_rate=self.sample_rate,
                                                      low_frequency_cutoff=self.noise_low_freq_cutoff,
                                                      detectors=self.detectors_abbr,
                                                      asds=asds)
                
                noise = self.noise_generator(0.0, self.sample_length_in_s, self.sample_length_in_s)
                noise = [noise[det].numpy() for det in self.detectors_abbr]
            
            elif self.dataset == 4:
                # We don't have to colour the noise ourselves, so they would not fill up RAM memory
                # idx values will now offset to [0, 0.8*num_noises] for training
                # idx values will now offset to [0.8*num_noises, num_noises] for validation
                index = int(idx - self.num_waveforms)
                
                # Example notes for mixed_noise case
                # O3a noise: Slicer has items with ids [0-250_000]
                # Artificial noise: Seeds must be unique
                # Training: [0-200_000] O3a noise, [200_000-400_000] artificial noise
                # Validation: [0-50_000] O3a noise (globtmp=200_000), [50_000-100_000] artificial noise
                if index < 0.5*self.num_noises or not self.mixed_noise:
                    index = int(index + self.globtmp)
                    noise, _ = noigen_slicer.__getitem__(index)
                else:
                    # Make sure the seeds are unique
                    self.noise_generator = NoiseGenerator(self.dataset,
                                                          seed=int(index+1+1000_000),
                                                          delta_f=self.delta_f,
                                                          sample_rate=self.sample_rate,
                                                          low_frequency_cutoff=self.noise_low_freq_cutoff,
                                                          detectors=self.detectors_abbr,
                                                          asds=d4_asds)
                    
                    noise = self.noise_generator(0.0, self.sample_length_in_s, self.sample_length_in_s)
                    noise = [noise[det].numpy() for det in self.detectors_abbr]

            assert len(noise[0]) == self.sample_length_in_num, "Sample Length Error: Expected={}, Observed={}".format(self.sample_length_in_num, len(noise[0]))
            assert len(noise[1]) == self.sample_length_in_num, "Sample Length Error: Expected={}, Observed={}".format(self.sample_length_in_num, len(noise[1]))
            
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

            elif diff_start < 0.0:
                h_plus = h_plus.crop(left=-1*((diff_start_num + self.error_padding_in_num)/2048.), right=0.0)
                h_cross = h_cross.crop(left=-1*((diff_start_num + self.error_padding_in_num)/2048.), right=0.0)
            
            assert len(h_plus) == self.sample_length_in_num + self.error_padding_in_num*2.0, 'Expected length = {}, actual length = {}'.format(self.sample_length_in_num + self.error_padding_in_num*2.0, len(h_plus))
            assert len(h_cross) == self.sample_length_in_num + self.error_padding_in_num*2.0, 'Expected length = {}, actual length = {}'.format(self.sample_length_in_num + self.error_padding_in_num*2.0, len(h_cross))
            
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
            sample['norm_invq'] = 1./sample['norm_q']
            # Add invq as well, since it is not a part of priors
            sample['invq'] = 1.0/prior_values['q']
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
            # idx = data['idx'] --> Use if needed
            sample = data['sample']
            save_idx = data['save_idx']
            is_waveform = data['is_waveform']
            
            """ Write sample """
            ## Save a chunk of samples in .hdf format
            # Create dataset objects for each storage array
            # 1. Sample saved at full 2048. Hz sampling rate with key as '2048/signal/<h_pol>'
            # 2. Get required MR sampling rates from dbins and create dataset for each sampling rate
            # 3. Add all necessary attributes exactly once
            
            # Sampling rate groups
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
                    # Check whether save_idx receives the same value 
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
        united_iterable = np.concatenate(self.iterable)
        
        ## Deprecated on Jun 15th, 2022, using random assignment of qidx instead
        assert sum(self.chunk_size) % self.num_queues_datasave == 0
        # Creating a list of repeating qidx (like [0,1,2,3,0,1,2,3, ...])
        save_qidxs = list(range(len(queues))) * int(sum(self.chunk_size)/self.num_queues_datasave)
        assert len(united_iterable) == len(save_qidxs)
        
        # Save idx (like [0,0,0,0,1,1,1,1,2,2,2,2, ..., 0,0,0,0,1,1,1,1,2,2,2,2, ...]) for signal and noise
        # TODO: Fix this method for multiple queues
        sidxs1 = np.array([[n]*len(queues) for n in range(int(self.chunk_size[0]/self.num_queues_datasave))]).flatten()
        sidxs2 = np.array([[n]*len(queues) for n in range(int(self.chunk_size[1]/self.num_queues_datasave))]).flatten()
        save_idxs = np.concatenate((sidxs1, sidxs2))
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
        """
        samples = []
        for injpath in self.inj_path:
            with h5py.File(injpath, "r") as fp:
                # Get and return the batch data
                tmp = np.array([list(foo) for foo in fp['data'][:]])
                samples.append(tmp)
        """

        with h5py.File(self.inj_path, "r") as fp:
            samples = np.array([list(foo) for foo in fp['data'][:]])

        
        # samples = np.row_stack((samples[0], samples[1]))

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
    
    
    def make_elink_lookup(self):
        # Make a lookup table with (id, path, target) for each training data
        # Should have an equal ratio, shuffled, of both classes (signal and noise)
        # Save the dataset paths alongside the target and ids as hdf5
        if self.dataset == 4:
            dsetdirs = ['dataset', 'validation_dataset']
        else:
            dsetdirs = ['dataset']
        
        # Other params to be saved in extlink file
        other_params = ['mchirp', 'chirp_distance']
        # Save params
        all_data = {'id': [], 'path': [], 'target': [], 'type': []}
        # Save other params into all_data
        oparams_dict = {foo: [] for foo in other_params}
        all_data.update(oparams_dict)
        
        self.dirs['lookup'] = os.path.join(self.dirs['parent'], 'extlinks.hdf')
        # Save this data in the hdf5 format as training.hdf
        main = h5py.File(self.dirs['lookup'], 'a', libver='latest')
        
        idx = 0
        # Iterate through the available dataset dirs
        ref_nfile = 0
        for dsetdir in dsetdirs:
            # tmp var
            tmp_var = {foo:[] for foo in all_data.keys()}
            # Add attributes if present
            add_attrs = True
            
            # All other params for parameter estimation are stored within the sample files
            self.dirs['dataset'] = os.path.join(self.dirs['parent'], dsetdir)
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
                        if dsetdir == 'dataset':
                            ref_nfile = nfile
                        elif dsetdir == 'validation_dataset':
                            nfile += ref_nfile + 1
                        
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
                        # Other params of each sample
                        for oparam in other_params:
                            if oparam in h5f[grp].keys():
                                oparam_ = np.array(h5f[grp][oparam])
                            else:
                                oparam_ = np.full(shape, -1)
                            # Update oparam
                            tmp_var[oparam].extend(oparam_)

                        # Update idxs
                        branch_ids = np.arange(idx, idx+shape)
                        tmp_var['id'].extend(branch_ids)
                        idx = idx+shape
                        # Update paths to dataset objects
                        branch_paths = itertools.product([_branch], np.arange(shape))
                        tmp_var['path'].extend([foo + '/' + str(bar) for foo, bar in branch_paths])
                        # Update target variable for each sample
                        tmp_var['target'].extend(np.full(shape, mode))
                            
                    # Add attributes from chunk files once to main ExternalLink File
                    if add_attrs:
                        attrs_save = dict(h5f.attrs)
                        for key, value in attrs_save.items():
                            main.attrs[key] = value
                        add_attrs = False
        
            # Sanity check
            """ 
            with h5py.File(self.dirs['lookup'], 'a', libver='latest') as fp:
                start = time.time()
                print(np.array(fp['004/2048/signal/h_plus'][2]))
                fin = time.time() - start
                print(fin)
            raise
            """
            
            # NOTE: On using visititems() method in h5py
            # visititems does not show h_plus, h_cross or noise datasets for some reason
            # If we add an extra /lorem to the linked dataset name, this bug shows up
            # Probably an issue with visititems method.
            
            ## Create lookup using zip
            # Explicitly check for length inconsistancies. zip doesn't raise error.
            tmp_var_lens = [len(tmp_var[foo]) for foo in tmp_var.keys() if foo != 'type']
            assert len(list(set(tmp_var_lens))) == 1, 'var:tmp_var fields in extlinks have inconsistent column lengths!'
            
            tmp_var.pop('type') # This is taken care of separately
            lookup = list(zip(*tmp_var.values()))
            # Shuffle the column stack (signal and noise are not shuffled)
            random.shuffle(lookup)
            # Separate out the tuples for ids, paths and targets
            for foo, key in zip(zip(*lookup), tmp_var.keys()):
                all_data[key].extend(foo)
            
            # Append type var to all_data
            if dsetdir == 'dataset':
                dstype = ['training']*len(all_data['id'])
            elif dsetdir == 'validation_dataset':
                # Only used in D4
                dstype = ['validation']*len(all_data['id'])
            all_data['type'].extend(dstype)
        
        # Close file explicitly, or use with instead
        main.close()
        
        # Write required fields as datasets in HDF5 training.hdf file
        with h5py.File(self.dirs['lookup'], 'a') as ds:
            """
            Shuffle Filter for HDF5:
                Block-oriented compressors like GZIP or LZF work better when presented with 
                runs of similar values. Enabling the shuffle filter rearranges the bytes in 
                the chunk and may improve compression ratio. No significant speed penalty, 
                lossless.
            """
            for _param in all_data.keys():
                ds.create_dataset(_param, data=all_data[_param], compression='gzip', compression_opts=9, shuffle=True)
            
        print("make_elink_lookup: ExternalLink lookup table created successfully!")


def _gen_(gd, mode=None, default_nums=None):
    ## Generate the dataset
    # Module assumes a 0.8 to 0.2 split between training and validation
    # Currently this is not configurable in data_cfg
    # mode is set only for D4
    print("Running _gen_ in {} mode for D4".format(mode))
    if mode == 'training' or mode == None:
        start = 0
        if mode == 'training':
            gd.num_waveforms = int(0.8*default_nums[0])
            gd.num_noises = int(0.8*default_nums[1])
            gd.globtmp = 0
    elif mode == 'validation':
        start = int(0.8*default_nums[0])
        gd.num_waveforms = int(0.2*default_nums[0])
        gd.num_noises = int(0.2*default_nums[1])
        mix_ratio = gd.mix_ratio if gd.mixed_noise else 1.0
        # Add a small leeway of 1000 samples so that PSDs are a little different
        # between the last training sample and first validation sample
        gd.globtmp = int(mix_ratio*0.8*default_nums[1] + 1000)
        gd.dirs['dataset'] = os.path.join(gd.dirs['parent'], "validation_dataset")
        if not os.path.exists(gd.dirs['dataset']):
            os.makedirs(gd.dirs['dataset'], exist_ok=False)
        
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
    assert len(waveform_iterables) == len(noise_iterables)
    global_iterables = list(zip(waveform_iterables, noise_iterables))
    
    for nchunk, chunk in enumerate(global_iterables):
        
        start_time = time.time()
        # Generate chunk 'n' of dataset
        gd.iterable = chunk
        # Get prior values of chosen waveform idxs
        with h5py.File(gd.inj_path, "r") as fp:
            # Get and return the batch data for signals alone
            gd.idx_offset = np.min(chunk[0])
            gd.priors = np.array(fp['data'][start:][chunk[0]])
        
        gd.nchunk = nchunk
        gd.generate_dataset()
        
        # Delete all raw files once trainable dataset has been created
        # shutil.rmtree(os.path.join(data_dir, 'foreground'))
        
        # Freeing memory explicitly
        gd.priors = None
        gd.iterable = None
        gc.collect()
        
        # Display time taken for chunk generation
        finish = time.time() - start_time
        print("Time taken for chunk data generation = {} minutes".format(finish/60.))


def get_D4_noise(gd):
    
    # Get the O3a real noise data (if using dataset 4)
    noise_dir = os.path.join(gd.parent_dir, 'O3a_real_noise')
    noise_file = os.path.join(noise_dir, 'O3a_real_noise.hdf')

    """
    # Download real noise (if not already present, checked inside download_data)
    gd.download_data(noise_file)
    # Sometimes the above command claims to be completed but
    # the file may still not be written to disk completely
    # Sanity check
    while True:
        try:
            with h5py.File(noise_file, 'r') as fp:
                _ = fp.attrs
            # If it works, we can leave the loop
            break
        except:
            # If not, we resume download
            gd.download_data(noise_file, resume=True)
    """

    # If using dataset 4, split the O3a real noise into training and testing segments
    # Testing segments should consist of 30 days. Whatever is left will be used for training.
    # During testing phase, call the testing noise .hdf file that we create here
    store_output = {'training': os.path.join(noise_dir, 'training_real_noise.hdf'),
                    'testing': os.path.join(noise_dir, 'testing_real_noise.hdf')}
    # noise_seed = {'training': gd.seed,
    #               'testing': 2_514_409_456}
    noise_seed = {'training': gd.seed,
                  'testing': 25}
    # We produce 1 month of data (30 days) using the seed provided in the MLGWSC-1 paper (for testing)
    # NOTE: This seed is only used for OverlapSegment class to get noise data for two detectors
    # This should not affect the training in any manner.
    # Just in case: I might not understand the OverlapSegment fully. Will choose a different seed for training.
    # TODO: Add all options below within the data_config class
    real_noise_kwargs = dict(real_noise_path = noise_file,
                             start_offset = 0.0,
                             duration = 2592000.0,
                             slide_buffer = 240,
                             min_segment_duration = 7200,
                             detectors = ['H1', 'L1'],
                             store_output = store_output,
                             seed = noise_seed,
                             sample_length_in_num = gd.sample_length_in_num,
                             num_noises = gd.num_noises)
    
    # Run the RealNoiseGenerator to retrieve training and testing dataset
    _ = RealNoiseGenerator(**real_noise_kwargs)
    infile = h5py.File(store_output['training'], 'r')
    # Calculate the step size required to obtain the number of noise samples
    # We add an extra 1000 samples as buffer, we forgo these samples so training and validation
    # does not have any overlap between them.
    # We get 0.5*num_noises as real noise from the data. The rest of the noise samples will be
    # Gaussian noise coloured using D4 PSDs.
    mix_ratio = gd.mix_ratio if gd.mixed_noise else 1.0
    # Add a leeway of 2000 samples so we can have a gap between training and validation samples
    step_size = (infile.attrs['duration'] - gd.sample_length_in_num)/(mix_ratio*gd.num_noises + 2000)
    # step_size = 0.1

    # The peak_offset option here is a dummy variable, we don't use the output times from slicer
    kwargs = dict(infile=infile,
                  step_size=step_size,
                  peak_offset=18.1,
                  whiten_padding=5.0,
                  slice_length=int(gd.signal_length*gd.sample_rate))

    print("Creating a slicer object using real O3a noise for training dataset")
    # The slicer object can take an index and return the required training data sample
    slicer = Slicer(**kwargs)
    # Sanity check the length of slicer
    print("Number of samples in D4 slicer = {}".format(len(slicer)))
    assert len(slicer) >= mix_ratio*gd.num_noises, "Insufficient number ({}/{}) of samples in slicer object!".format(len(slicer), mix_ratio*gd.num_noises)
    global noigen_slicer
    noigen_slicer = slicer
    
    return infile


def get_D4_psds(gd):
    print("Gathering the estimated PSDs for D4 into a global variable")
    noise_dir = os.path.join(gd.parent_dir, 'O3a_real_noise')
    d4_psds = os.path.join(noise_dir, 'd4_psds')
    # Get required real PSD files
    h1_psd_files = glob.glob(os.path.join(d4_psds, "real_psd_H1_*.hdf"))
    h1_psd_files.sort()
    l1_psd_files = glob.glob(os.path.join(d4_psds, "real_psd_L1_*.hdf"))
    l1_psd_files.sort()
    psd_options = {'H1': h1_psd_files, 'L1':l1_psd_files}
    
    complex_asds = {'H1': [], 'L1': []}
    detectors_abbr = ['H1', 'L1']
    for i, det in enumerate(detectors_abbr):
        # Read all detector PSDs as frequency series with appropriate delta_f
        for psd_det in psd_options[det]:
            with h5py.File(psd_det, 'r') as fp:
                data = np.array(fp['data'])
                delta_f = fp.attrs['delta_f']
            psd = FrequencySeries(data, delta_f=delta_f)
            # Convert PSD's to ASD's for colouring the white noise
            # TODO: Generalise this method
            foo = psd_to_asd(psd, 0.0, 15.0,
                             sample_rate=2048.,
                             low_frequency_cutoff=15.0,
                             filter_duration=15.0)
            complex_asds[det].append(foo)

    global d4_asds
    d4_asds = complex_asds
    

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
    
    if gd.dataset == 4:
        infile = get_D4_noise(gd)
    
    # Save PSDs required for dataset
    gd.save_PSD()
    
    if gd.dataset == 4 and gd.mixed_noise:
        get_D4_psds(gd)
    
    # Get priors for entire dataset
    gd.get_priors()
    
    if gd.dataset in [1, 2, 3]:
        _gen_(gd)
    elif gd.dataset == 4:
        nums = [gd.num_waveforms, gd.num_noises]
        for mode in ['training', 'validation']:
            # We take care of the splits here for D4
            _gen_(gd, mode=mode, default_nums=nums)
    
    # Make elink dataset lookup table
    gd.make_elink_lookup()
    
    # gd.inj_path = [os.path.join(gd.dirs['injections'], 'injections_1.hdf'), os.path.join(gd.dirs['injections'], 'injections_2.hdf')]
    gd.inj_path = os.path.join(gd.dirs['injections'], 'injections.hdf')
    # Making the prior distribution plots
    gd.plot_priors(gd.dirs['parent'])

    # Closing any unnecessary files
    if gd.dataset == 4:
        infile.close()
