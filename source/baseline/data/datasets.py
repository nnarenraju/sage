# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Mon Dec  6 11:07:42 2021

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
import h5py
import glob
import torch
import random
import numpy as np

from torch.utils.data import Dataset
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


# LOCAL
from data.psd_loader import load_psds
from data.prior_modifications import *
from data.distributions import get_distributions
from data.normalisation import get_normalisations
from data.snr_calculation import get_network_snr
from data.plot_dataloader_unit import plot_unit
from data.multirate_sampling import get_sampling_rate_bins

# PyCBC
import pycbc
from pycbc import distributions
from pycbc.types import TimeSeries
from pycbc.types import FrequencySeries
from pycbc.psd import welch, interpolate
from pycbc.distributions.utils import draw_samples_from_config

# Datatype for storage
tensor_dtype=torch.float32


""" Utility Classes """

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


""" Dataset Objects """

class MLMDC1(Dataset):
    
    def __init__(self, data_paths, targets, transforms=None, target_transforms=None,
                 signal_only_transforms=None, noise_only_transforms=None, 
                 training=False, testing=False, store_device='cpu', train_device='cpu', 
                 cfg=None, data_cfg=None):
        
        super().__init__()
        self.data_paths = data_paths
        self.targets = targets
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.signal_only_transforms = signal_only_transforms
        self.noise_only_transforms = noise_only_transforms
        self.store_device = store_device
        self.train_device = train_device
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.data_loc = os.path.join(self.data_cfg.parent_dir, self.data_cfg.data_dir)
        self.epoch = -1
        self.plot_on_first_batch = self.cfg.plot_on_first_batch
        self.nplot_on_first_batch = 0
        self.plot_batch_fnames = []
        
        self.training = training
        
        # Set CUDA device for pin_memory if needed
        if bool(re.search('cuda', self.cfg.store_device)):
            setattr(self, 'foo', torch.cuda.set_device(self.cfg.store_device))
        
        
        """ PSD Handling (used in whitening) """
        # Store the PSD files here in RAM. This reduces the overhead when whitening.
        # Read all psds in the data_dir and store then as FrequencySeries
        self.psds_data = load_psds(self.data_loc, self.data_cfg)
            
        """ Multi-rate Sampling """
        # Get the sampling rates and their bins idx
        self.data_cfg.dbins = get_sampling_rate_bins(self.data_cfg)
        
        """ LAL Detector Objects (used in project_wave - AugPolSky) """
        # Detector objects (these are lal objects and may present problems when parallelising)
        # Create the detectors (TODO: generalise this!!!)
        self.detectors_abbr = ('H1', 'L1')
        self.detectors = [pycbc.detector.Detector(det_abbr) for det_abbr in self.detectors_abbr]
        
        """ PyCBC Distributions (used in AugPolSky and AugDistance) """
        ## Distribution objects for augmentation
        self.distrs = get_distributions(self.data_cfg)

        """ Normalising the augmented params (if needed) """
        ## Normnalisation dict
        self.norm, self.limits = get_normalisations(self.cfg, self.data_cfg)

        """ Weighting the custom BCE loss using the input prior distributions of parameters """
        # injnames = ['mass1', 'mass2', 'ra', 'dec', 'inclination', 'coa_phase', 'polarization',
        #            'chirp_distance', 'spin1_a', 'spin1_azimuthal', 'spin1_polar', 'spin2_a',
        #            'spin2_azimuthal', 'spin2_polar', 'tc', 'tcsamp', 'spin1x', 'spin1y', 'spin1z',
        #            'spin2x', 'spin2y', 'spin2z', 'mchirp', 'q', 'distance']
        
        """
        injections_file = os.path.join(self.data_loc, "injections/injections_1.hdf")
        with h5py.File(injections_file, "r") as foo:
            # Attributes of file        
            injections = np.asarray(foo['data'])
            injections = np.asarray([list(foo) for foo in injections])
        """
        # Get the number of GW signals within each bin of injections
        # We do this for all params, to be used within custom losses
        self.weighted_bce_data = {}
        #for param in self.cfg.weighted_bce_loss_params:
        #    self.weighted_bce_data[param] = np.histogram(injections, bins=64)

        """ Data Save Params (for plotting sample just before training) """
        if self.data_cfg.num_sample_save == None:
            self.num_sample_save = 100
        else:
            self.num_sample_save = data_cfg.num_sample_save
        
        """ Random noise realisation """
        self.noise_idx = np.argwhere(self.targets == 0).flatten()
        self.noise_norm_idx = np.arange(len(self.noise_idx))
        self.noise_paths = self.data_paths[self.noise_idx]
        
        """ Keep ExternalLink Lookup table open till end of run """
        lookup = os.path.join(cfg.export_dir, 'extlinks.hdf')
        self.extmain = h5py.File(lookup, 'r', libver='latest')
        self.sample_rate = self.extmain.attrs['sample_rate']
        self.noise_low_freq_cutoff = self.extmain.attrs['noise_low_freq_cutoff']

        """ Dataset 1 PSD """
        psd_fun = pycbc.psd.analytical.aLIGOZeroDetHighPower
        self.unet_psds = [psd_fun(self.data_cfg.psd_len, self.data_cfg.delta_f, self.data_cfg.noise_low_freq_cutoff) for _ in range(2)]
        self.noise_generator = pycbc.noise.gaussian.frequency_noise_from_psd
        
        """ numpy random """
        self.np_gen = np.random.default_rng()
        
        ## DEBUG
        self.debug = cfg.debug
        if self.debug:
            self.debug_dir = os.path.join(cfg.export_dir, 'DEBUG')
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir, exist_ok=False)
        else:
            self.debug_dir = ''

        ## SPECIAL
        self.special = {}
        self.special['distrs'] = self.distrs
        self.special['norm'] = self.norm
        self.special['cfg'] = self.cfg
        self.special['data_cfg'] = self.data_cfg
        self.special['dets'] = self.detectors
        self.special['psds'] = self.psds_data
        self.special['default_keys'] = self.special.keys()
        self.special['training'] = self.training
        self.special['limits'] = self.limits

        # Set epoch priors (if required)
        self.priors = None

        ## Ignore Params
        self.ignore_params = {'start_time', 'interval_lower', 'interval_upper', 
                              'sample_rate', 'noise_low_freq_cutoff', 'declination', 
                              'right_ascension', 'polarisation_angle'}


    def __len__(self):
        return len(self.data_paths)

    
    def _read_(self, data_path):
        
        # Store all params within chunk file
        params = {}
        targets = {}
        
        # Get data from ExternalLink'ed lookup file
        HDF5_Dataset, didx = os.path.split(data_path)
        # Dataset Index should be an integer
        didx = int(didx)
        # Check whether data is signal or noise with target
        target = 1 if bool(re.search('signal', HDF5_Dataset)) else 0
        targets['gw'] = target
        # Access group
        group = self.extmain[HDF5_Dataset]
        
        if not target:
            ## Read noise data
            noise_1 = np.array(group['noise_1'][didx])
            noise_2 = np.array(group['noise_2'][didx])
            sample = np.stack([noise_1, noise_2], axis=0)
            # Dummy noise params
            #targets['norm_dchirp'] = -1
            #targets['norm_dist'] = -1
            targets['norm_mchirp'] = -1
            #targets['norm_q'] = -1
            #targets['norm_invq'] = -1
            targets['norm_tc'] = -1
            # Dummy params
            params['mass1'] = -1
            params['mass2'] = -1
            params['distance'] = -1
            params['mchirp'] = -1
            params['dchirp'] = -1
            params['tc'] = -1
            #params['chirp_distance'] = -1
            #params['q'] = -1
            #params['invq'] = -1
            params['network_snr'] = -1
        else:
            ## Read signal data
            h_plus = np.array(group['h_plus'][didx])
            h_cross = np.array(group['h_cross'][didx])
            sample = np.stack([h_plus, h_cross], axis=0)
            # Signal params
            params['start_time'] = group['start_time'][didx]
            params['interval_lower'] = group['interval_lower'][didx]
            params['interval_upper'] = group['interval_upper'][didx]
            params['mass1'] = group['mass1'][didx]
            params['mass2'] = group['mass2'][didx]
            params['distance'] = group['distance'][didx]
            params['mchirp'] = group['mchirp'][didx]
            params['tc'] = group['tc'][didx]
            #params['chirp_distance'] = group['chirp_distance'][didx]
            #params['q'] = group['q'][didx]
            #params['invq'] = group['invq'][didx]
            # Target params
            # targets['norm_dchirp'] = group['norm_dchirp'][didx]
            # targets['norm_dist'] = group['norm_dist'][didx]
            targets['norm_mchirp'] = group['norm_mchirp'][didx]
            #targets['norm_q'] = group['norm_q'][didx]
            #targets['norm_invq'] = group['norm_invq'][didx]
            targets['norm_tc'] = group['norm_tc'][didx]
        
        # Generic params
        params['sample_rate'] = self.sample_rate
        params['noise_low_freq_cutoff'] = self.noise_low_freq_cutoff
        
        return (sample, targets, params)
    
    
    def _dist_from_dchirp(self, chirp_distance, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the luminosity distance given a chirp distance and chirp mass.
        return chirp_distance * (2.**(-1./5) * ref_mass / mchirp)**(-5./6)
    
    
    def _dchirp_from_dist(self, dist, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the chirp distance given the luminosity distance and chirp mass.
        return dist * (2.**(-1./5) * ref_mass / mchirp)**(5./6)
    
    
    def _augmentation_(self, sample, target, params, mode=None):
        """ Signal and Noise only Augmentation """
        if mode == None:
            raise ValueError('Augmentation mode not chosen!')
        
        ## Noise and Signal Augmentation
        if target and self.signal_only_transforms and mode=='signal':
            self.special['epoch'] = self.epoch.value
            sample, params, _special = self.signal_only_transforms(sample, params, self.special, self.debug_dir)
            self.special.update(_special)
        
        elif not target and self.noise_only_transforms and mode=='noise':
            sample = self.noise_only_transforms(sample, self.debug_dir)
        
        return sample, params
    
    
    def _noise_realisation_(self, sample, targets, params):
        """ Finding random noise realisation for signal """
        # Random noise realisation is the only procedure available
        # Fixed noise realisation was deprecated
        
        if self.cfg.add_random_noise_realisation and targets['gw']:
            # Pick a random noise realisation to add to the signal
            random_noise_idx = random.choice(self.noise_idx)
            random_noise_data_path = self.data_paths[random_noise_idx]
            
            # Read the noise data
            pure_noise, targets_noise, params_noise = self._read_(random_noise_data_path)
            target_noise = targets_noise['gw']
            # BugFix (Feb 2023): Data Leakage - Artifacts introduced in noise augmentation will not be present
            # in noise added to signals without augmenting this noise as well. 
            # Bug fixes to noise augmentation completed. Artifacts should now not be introduced to noise.
            # Solution: Whitening has to be done before cyclic shift of noise. 
            # transformed_noise = self._transforms_(pure_noise, key='stage1')
            pure_noise, _ = self._augmentation_(pure_noise, target_noise, params_noise, mode='noise')
            
            """ Adding noise to signals """
            if isinstance(pure_noise, np.ndarray) and isinstance(sample, np.ndarray): 
                noisy_signal = sample + pure_noise
            else:
                raise TypeError('pure_signal or pure_noise is not an np.ndarray!')
            
        elif not self.cfg.add_random_noise_realisation and targets['gw']:
            # Fixed noise realisation to add to the signal
            raise DeprecationWarning('Fixed noise realisation feature deprecated on June 8th, 2022')
            
        else:
            # If the sample is pure noise
            noisy_signal = sample
            pure_noise = sample
        
        return (noisy_signal, pure_noise)
    
    
    def _transforms_(self, sample, key=None):
        """ Transforms """
        # Apply transforms to signal and target (if any)
        if self.transforms:
            sample_transforms = self.transforms(sample, self.special, key=key)
        else:
            sample_transforms = {'sample': sample}
        
        return sample_transforms
    
    
    def _plotting_(self, idx, pure_sample, pure_noise, noisy_sample, trans_noisy_sample, params):
        """ Plotting idx data (if flag is set to True) """
        # Input parameters
        if self.transforms:
            trans_pure_signal = self.transforms(pure_sample, self.special, key='stage1')
            update_transforms = self.transforms(trans_pure_signal['sample'], self.special, key='stage2')
            trans_pure_signal.update(update_transforms)
        else:
            trans_pure_signal = None
        
        save_path = self.cfg.export_dir
        data_dir = os.path.normpath(self.data_loc).split(os.path.sep)[-1]
        # Plotting unit data
        plot_unit(pure_sample, pure_noise, noisy_sample, trans_pure_signal, trans_noisy_sample, 
                  params['mass1'], params['mass2'], params['network_snr'], params['sample_rate'],
                  save_path, data_dir, idx)
    
    
    def mfsnr_calculation(self, sample, stilde):
        # Handling the noise sample
        stilde = [TimeSeries(noi, delta_t=1./self.sample_rate) for noi in stilde]
        # Handing the signal sample
        det1 = TimeSeries(sample[0], delta_t=1./self.sample_rate)
        #det1.resize(len(stilde[0]))
        det2 = TimeSeries(sample[1], delta_t=1./self.sample_rate)
        #det2.resize(len(stilde[1]))
        ### Estimate the PSD
        # We'll choose 4 seconds PSD samples that are overlapped 50 %
        delta_t = 1.0/2048.
        seg_len = int(4. / delta_t)
        seg_stride = int(seg_len / 2.)
        estimated_psd1 = welch(stilde[0], seg_len=seg_len, seg_stride=seg_stride)
        estimated_psd1 = interpolate(estimated_psd1, self.data_cfg.delta_f)
        estimated_psd2 = welch(stilde[1], seg_len=seg_len, seg_stride=seg_stride)
        estimated_psd2 = interpolate(estimated_psd2, self.data_cfg.delta_f)
        # SNR calculation
        mfsnr_H1 = pycbc.filter.matched_filter(det1, stilde[0], psd=estimated_psd1, low_frequency_cutoff=self.data_cfg.noise_low_freq_cutoff)
        mfsnr_H1 = mfsnr_H1[512: len(mfsnr_H1)-512]
        snrts_H1 = max(abs(mfsnr_H1))
        mfsnr_L1 = pycbc.filter.matched_filter(det2, stilde[1], psd=estimated_psd2, low_frequency_cutoff=self.data_cfg.noise_low_freq_cutoff)
        mfsnr_L1 = mfsnr_L1[512: len(mfsnr_L1)-512]
        snrts_L1 = max(abs(mfsnr_L1))
        matched_filter_snr = np.sqrt(sum([snrts_H1, snrts_L1]))
        return matched_filter_snr
    
    
    def adfuller_test(self, sample):
        # Test for non-stationarity of data (D4 noise in particular)
        # Augmented Dickey Fuller Test for non-stationarity
        # Link: https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test
        # Null Hypothesis (H0): If failed to be rejected, it suggests the time series 
        # has a unit root, meaning it is non-stationary. It has some time dependent structure.
        # If p-value > 0.05 we fail to reject the null hypothesis (H0), 
        # the data has a unit root and is non-stationary.
        # If p-value <= 0.05, reject the null hypothesis (H0), the data does not have 
        # a unit root and is stationary.
        for detdata, det in zip(sample, self.detectors_abbr):
            test_results_det = adfuller(detdata)
            p_value = test_results_det[1]
            if p_value > 0.05:
                adf_test_statistic = test_results_det[0]
                print("ADF Test Statistic = {}".format(adf_test_statistic))
                # Critical thresholds at 1%, 5% and 10% confidence intervals
                print("Critical Values:")
                for key, value in test_results_det[4].items():
                    print('\t%s: %.3f' % (key, value))
                    
                raise ValueError("FAILED: ADF Test for non-stationary - \
                                 p-value ({}) greater than 0.05 and NULL hypothesis \
                                 H0 cannot be rejected. Sample is non-stationary.".format(p_value))
                
    
    def target_handling(self, targets, params):
        # Gather up all parameters required for training and validation
        # NOTE: We can't use structured arrays here as PyTorch does not support it yet.
        #       Dictionaries are slow but it does the job.
        """ Targets """
        ## Storing targets as dictionary
        all_targets = {}
        # targets contain all other norm values for PE
        all_targets.update(targets)
        # Update parameter labels if augmentation changed them
        # aug_labels must have the same keys as targets dict
        target_update = {foo: self.special[foo] for foo in self.special.keys() if foo not in self.special['default_keys']}
        # Exception
        if 'norm_snr' not in target_update.keys():
            target_update['norm_snr'] = -1

        # Update targets dict
        all_targets.update(target_update)

        # Add the weighted BCE loss terms into all_targets
        # all_targets['weighted_bce_data'] = self.weighted_bce_data
        
        """ Source Parameters """
        ## Add sample params to all_targets variable
        source_params = {}
        # Distance and dchirp could have been alterred when rescaling SNR
        source_params.update(params)

        ## Exceptions
        for foo in self.ignore_params:
            if foo in source_params.keys():
                source_params.pop(foo)

        if 'dchirp' not in params.keys():
            source_params['dchirp'] = self._dchirp_from_dist(params['distance'], params['mchirp'])
            
        if targets['gw']:
            # Calculating the duration of the given signal
            lf = self.data_cfg.signal_low_freq_cutoff
            G = 6.67e-11
            c = 3.0e8
            source_params['signal_duration'] = 5. * (8.*np.pi*lf)**(-8./3.) * (params['mchirp']*1.989e30*G/c**3.)**(-5./3.)
        else:
            source_params['signal_duration'] = -1
        
        return all_targets, source_params
    
    
    def squash_bugs(self, idx, pure_sample, pure_noise, noisy_sample, sample_transforms, targets, params):
        check_dir = os.path.join(self.cfg.export_dir, 'SAMPLES')
        if os.path.isdir(check_dir):
            check_path = os.path.join(check_dir, '*.png')
            num_created = len(glob.glob(check_path))
        else:
            num_created = 0
            
        if targets['gw'] and num_created < self.cfg.num_sample_save:
            self._plotting_(idx, pure_sample, pure_noise, noisy_sample, sample_transforms, params)
        
        ## Test for non-stationarity of sample
        self.adfuller_test(sample_transforms['sample'])
    

    def get_priors_for_epoch(self, random_seed):
        ## Generate source prior parameters
        # A path to the .ini file.
        ini_parent = './data/ini_files'
        CONFIG_PATH = '{}/ds{}.ini'.format(ini_parent, self.data_cfg.dataset)
        
        # Draw num_waveforms number of samples from ds.ini file
        priors = draw_samples_from_config(path=CONFIG_PATH,
                                          num=self.data_cfg.num_waveforms,
                                          seed=random_seed)
        
        ## MODIFICATIONS ##
        if self.data_cfg.modification != None and isinstance(self.data_cfg.modification, str):
            # Check for known modifications
            if self.data_cfg.modification in ['uniform_signal_duration', 'uniform_chirp_mass']:
                _mass1, _mass2 = get_uniform_masses(self.data_cfg.prior_low_mass, self.data_cfg.prior_high_mass, self.data_cfg.num_waveforms)
                # Masses used for mass ratio need not be used later as mass1 and mass2
                # We calculate them again after getting chirp mass
                q = q_from_uniform_mass1_mass2(_mass1, _mass2)

            if self.data_cfg.modification == 'uniform_signal_duration':
                # Get chirp mass from uniform signal duration
                tau_lower, tau_upper = get_tau_priors(self.data_cfg.prior_low_mass, self.data_cfg.prior_high_mass, self.data_cfg.signal_low_freq_cutoff)
                mchirp = mchirp_from_uniform_signal_duration(tau_lower, tau_upper, self.data_cfg.num_waveforms, self.data_cfg.signal_low_freq_cutoff)

            elif self.data_cfg.modification == 'uniform_chirp_mass':
                # Get uniform chirp mass
                mchirp = get_uniform_mchirp(self.data_cfg.prior_low_mass, self.data_cfg.prior_high_mass, self.data_cfg.num_waveforms)

            else:
                raise ValueError("get_priors: Unknown modification ({}) added to data_cfg.modification!".format(self.data_cfg.modification))

            if self.data_cfg.modification in ['uniform_signal_duration', 'uniform_chirp_mass']:
                # Using mchirp and q, get mass1 and mass2
                mass1, mass2 = mass1_mass2_from_mchirp_q(mchirp, q)
                # Get chirp distances using the power law distribution
                dchirp = np.asarray([self.distrs['dchirp'].rvs()[0][0] for _ in range(self.data_cfg.num_waveforms)])
                # Get distance from chirp distance and chirp mass
                distance = self._dist_from_dchirp(dchirp, mchirp)
                ## Update Priors
                priors['q'] = q
                priors['mchirp'] = mchirp
                priors['mass1'] = mass1
                priors['mass2'] = mass2
                priors['chirp_distance'] = dchirp
                priors['distance'] = distance
                
        elif self.data_cfg.modification != None and not isinstance(self.data_cfg.modification, str):
            raise ValueError("get_priors: Unknown data type used in data_cfg.modification!")
        
        self.priors = iter(priors)


    def __getitem__(self, idx):
        
        # Setting the unique seed for given sample
        np.random.seed(idx+1)
        
        # Get data paths for external link
        data_path = self.data_paths[idx]
        # Get data from ExternalLink'ed lookup file
        HDF5_Dataset, didx = os.path.split(data_path)
        # If we are generating signals on the fly, we do not need to access data via extlinks.hdf
        gencondition = 'GenerateNewSignal' in [foo.__class__.__name__ for foo in self.cfg.transforms['signal'].transforms] and self.training
        if gencondition and (1 if bool(re.search('signal', HDF5_Dataset)) else 0):
            # Dummy sample to handle to transforms wrapper
            sample = np.empty([2, 2])
            # Generate signal params from given prior and convert to dict (to be used as waveform_kwargs)
            # Run this once per epoch for the required number of waveforms
            params = next(self.priors)
            params = dict(zip(params.dtype.names, params))
            params['sample_rate'] = self.sample_rate
            params['noise_low_freq_cutoff'] = self.noise_low_freq_cutoff
            params['dchirp'] = params['chirp_distance']
            # Create target for signal params
            targets = {}
            targets['gw'] = 1
            targets['norm_tc'] = self.norm['tc'].norm(params['tc'])
            targets['norm_mchirp'] = self.norm['mchirp'].norm(params['mchirp'])
        else:
            ## Read the sample(s)
            sample, targets, params = self._read_(data_path)
        
        ## Signal Augmentation
        pure_sample, params = self._augmentation_(sample, targets['gw'], params, mode='signal')
        keys = list(params.keys())[:]
        for key in keys:
            if key not in ['mass1', 'mass2', 'distance', 'mchirp', 'tc', 'dchirp', 'network_snr']:
                params.pop(key, None)

        ## Add noise realisation to the signals
        noisy_sample, pure_noise = self._noise_realisation_(pure_sample, targets, params)
        
        ## Noise Augmentation
        if self.training:
            noisy_sample, params = self._augmentation_(noisy_sample, targets['gw'], params, mode='noise')

        ## Transformation Stage 1 (HighPass and Whitening)
        # These transformations are possible for pure noise before augmentation
        # NOTE: Cyclic shifting and phase augmentation cannot be performed before whitening
        #       The discontinuities in the sample will cause issues.
        # sample_transforms['sample'] will always provide the output of the last performed transform
        sample_transforms = self._transforms_(noisy_sample, key='stage1')

        ## Transformation Stage 2 (Multirate Sampling)
        mrsampling = self._transforms_(sample_transforms['sample'], key='stage2')
        # With the update 'sample' should point to mrsampled data
        # but still contain all other transformations in the correct key.
        sample_transforms.update(mrsampling)
        
        ## Creating a UNet target using pure signal and white Gaussian noise
        """
        noise = [self.noise_generator(psd).to_timeseries()[:pure_sample.shape[-1]] for psd in self.unet_psds]
        if targets['gw']:
            unet_signal = pure_sample + noise
            unet_transforms = self._transforms_(unet_signal, key='stage1')
            unet_transforms = self._transforms_(unet_transforms['sample'], key='stage2')
            targets['whitened'] = unet_transforms['sample'][:, :3680]
        else:
            unet_transforms = self._transforms_(noise, key='stage1')
            unet_transforms = self._transforms_(unet_transforms['sample'], key='stage2')
            targets['whitened'] = unet_transforms['sample'][:, :3680]
        """

        ## Targets and Parameters Handling
        all_targets, source_params = self.target_handling(targets, params)
        
        ## DEBUG Modules
        if self.debug:
            args = (idx,)
            # Pure samples
            args += (pure_sample, pure_noise)
            # Transformed samples
            args += (noisy_sample, sample_transforms)
            # Targets and source parameters
            args += (targets, params)
            self.squash_bugs(*args)
        
        # Reducing memory footprint
        # This can only be performed after transforms and augmentation
        _sample = np.array(sample_transforms['sample'], dtype=np.float32)
        sample = _sample[:].copy()
        
        # Tensorification
        # Convert signal/target to Tensor objects
        sample = torch.from_numpy(sample)
        # First batch processes
        if self.plot_on_first_batch:
            # NOTE: Structured arrays are faster but PyTorch does not support them.
            #       Dictionaries are slower but they do the job.
            # Convert all samples into float16 for the sake of saving memory
            # Also, use .copy() for some arrays as PyTorch does not support data with negative stride
            nsamp = (pure_sample/max(abs(pure_sample[0]))).astype(np.float16).copy()
            hpass = (sample_transforms['HighPass']/max(abs(sample_transforms['HighPass'][0]))).astype(np.float16).copy()
            white = (sample_transforms['Whiten']).astype(np.float16).copy()
            # msamp = (sample_transforms['MultirateSampling']).astype(np.float16).copy()
            msamp = (sample_transforms['Whiten']).astype(np.float16).copy()
            plot_batch = [nsamp, hpass, white, msamp]
            self.plot_batch_fnames = ['PureSample', 'HighPass', 'Whiten', 'MRSampling']
            self.nplot_on_first_batch = 4
        else:
            plot_batch = [None] * self.nplot_on_first_batch

        return (sample, all_targets, source_params, plot_batch)





""" Other Loaders """

class Simple(Dataset):
    """
    Simple read-and-load-type dataset object
    WARNING!!!: This custom dataset *cannot* be used in configs as primary dataset object
    
    """
    
    def __init__(self, samples, targets, store_device='cpu', train_device='cpu'):
        
        super().__init__()
        self.samples = samples
        self.targets = targets
        self.store_device = store_device
        self.train_device = train_device
        assert len(self.samples) == len(self.targets)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        """ Tensorification and Device Compatibility """
        # Convert signal/target to Tensor objects and set device and data_type
        global tensor_dtype
        sample = self.samples[idx].to(dtype=tensor_dtype, device=self.train_device)
        target = self.targets[idx].to(dtype=tensor_dtype, device=self.train_device)
        
        return (sample, target)
