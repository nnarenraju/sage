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
import time
import glob
import torch
import shutil
import random
import numpy as np
from itertools import cycle
from torch.utils.data import Dataset

# LOCAL
from data.snr_calculation import get_network_snr
from data.plot_dataloader_unit import plot_unit
from data.multirate_sampling import get_sampling_rate_bins

# PyCBC
import pycbc
from pycbc import distributions
from pycbc.types import FrequencySeries

# Datatype for storage
tensor_dtype=torch.float32

""" Dataset Objects """

class MLMDC1_IterBatch(Dataset):
    """
    Procedure for data storage:
        1. Output h_plus and h_cross for each signal, output noise
        2. Use the h_plus and h_cross to realise a unique signal wrt polarisation, ra, dec
        3. Augment this signal wrt distance
        4. If noise, Augment the noise (time shifting)
        5. Choose a random realisation of noise from the background dir and add to signal
        6. Apply Bandpass filter
        7. Apply Whitening
        8. Apply Multirate sampling
    
    Here, the random realisation of noise added to the signal and augmentation of signal/noise
    happens differently every epoch. This ensures essentially an infinite amount of data. In this
    setup, each epoch sees the same prior distribution.
    If we were to save trainable data using this procedure, every epoch will see the same realisation
    of signal, noise and augmented values. 
    
    Questions:
        1. Is it possible to apply (6, 7, 8) to h_plus and h_cross. If so, we can store transformed
           h_plus and h_cross in the trainable dataset.
        2. This trainable dataset can be used with project_wave to obtain a unique signal.
        3. It can then be augmented by distance.
        4. Noise will not be affected by applying (6, 7, 8) beforehand.
        5. The augmented h_t can then be added to a random realisation of noise
        
    However, I believe that project_wave cannot be applied to a signal where multi-rate sampling
    has already been performed.
    
    To Check:
        1. What WallClock overhead does each transformation procedure take? (FIN)
        2. How to make these transformation as fast as possible. Use C/C++ based libraries. (FIN)
        3. Is it possible to use num_workers > 0, if using project_wave as a part of transforms (FIN)
    
    """
    
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
        self.training = training
        self.testing = testing
        self.store_device = store_device
        self.train_device = train_device
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.data_loc = os.path.join(self.data_cfg.parent_dir, self.data_cfg.data_dir)
        
        
        """ PSD Handling (used in whitening) """
        # Store the PSD files here in RAM. This reduces the overhead when whitening.
        # Read all psds in the data_dir and store then as FrequencySeries
        self.PSDs = {}
        psd_files = glob.glob(os.path.join(self.data_loc, "psds/*"))
        for psd_file in psd_files:
            with h5py.File(psd_file, 'r') as fp:
                data = np.array(fp['data'])
                delta_f = fp.attrs['delta_f']
                name = fp.attrs['name']
                
            psd_data = FrequencySeries(data, delta_f=delta_f)
            # Store PSD data into lookup dict
            self.PSDs[name] = psd_data
        
        if self.data_cfg.dataset == 1:
            self.psds_data = [self.PSDs['aLIGOZeroDetHighPower']]*2
            
        
        """ Multi-rate Sampling """
        # Get the sampling rates and their bins idx
        data_cfg.dbins = get_sampling_rate_bins(data_cfg)
        
        
        """ LAL Detector Objects (used in project_wave - AugPolSky) """
        # Detector objects (these are lal objects and may present problems when parallelising)
        # Create the detectors (TODO: generalise this!!!)
        detectors_abbr = ('H1', 'L1')
        self.detectors = []
        for det_abbr in detectors_abbr:
            self.detectors.append(pycbc.detector.Detector(det_abbr))
        
        
        """ PyCBC Distributions (used in AugPolSky and AugDistance) """
        ## Distribution objects for augmentation
        # Used for obtaining random polarisation angle
        self.uniform_angle_distr = distributions.angular.UniformAngle(uniform_angle=(0., 2.0*np.pi))
        # Used for obtaining random ra and dec
        self.skylocation_distr = distributions.sky_location.UniformSky()
        # Used for obtaining random mass
        self.mass_distr = distributions.Uniform(mass=(data_cfg.prior_low_mass, data_cfg.prior_high_mass))
        # Used for obtaining random chirp distance
        dist_gen = distributions.power_law.UniformRadius
        self.chirp_distance_distr = dist_gen(distance=(data_cfg.prior_low_chirp_dist, 
                                                       data_cfg.prior_high_chirp_dist))
        # Distributions object
        self.distrs = {'pol': self.uniform_angle_distr, 'sky': self.skylocation_distr,
                       'mass': self.mass_distr, 'dchirp': self.chirp_distance_distr}
        
        
        """ Data Save Params (for plotting sample just before training) """
        # Saving frequency with idx plotting
        # TODO: Add compatibility for using cfg.splitter with K-folds
        if self.data_cfg.num_sample_save == None:
            self.num_sample_save = int(len(self.data_paths)/100.0)
        else:
            self.num_sample_save = data_cfg.num_sample_save
        # Initialise save counter
        self.save_counter = 0
        
        
        """ Sequential noise realisation """
        seq_noise = random.sample(list(self.data_paths), len(self.data_paths))
        self.seq_noise = cycle(seq_noise)
        
        
        """ Save Times """
        self.record_times = {}
        self.signal_aug_times = {}
        self.noise_aug_times = {}
        self.transform_times = {}
        
        
        """ Training or Testing? """
        if training:
            assert testing == False
        if testing:
            assert training == False
        if not training and not testing:
            raise ValueError("Neither training or testing phase chosen for dataset class. Bruh?")


    def __len__(self):
        return len(self.data_paths)

    
    def _read_(self, data_path, noise_realisation=False):
        
        """
        Read sample and return necessary training params to DataLoader.
        
        Attribute list present in the main group (example):
            
            'dataset': 1
            'dataset_ratio_SNnum': 1.0
            'delta_f': 0.04
            'noise_low_freq_cutoff': 15.0
            'num_noises': 1000
            'num_waveforms': 1000
            'prior_high_chirp_dist': 350.0
            'prior_high_mass': 50.0
            'prior_low_chirp_dist': 130.0
            'prior_low_mass': 10.0
            'psd_len': 25601
            'sample_rate': 2048.0
            'seed': 42
            'signal_approximant': 'IMRPhenomXPHM'
            'signal_length': 20.0
            'tc_inject_lower': 18.0
            'tc_inject_upper': 18.2
        
        Dataset list present in the sample group:
            
            'h_plus': np.ndarray of np.ndarrays
            'h_cross': np.ndarray of np.ndarrays
            'start_time': 1253708517.0813572
            'interval_lower': 1253708537.2702007
            'interval_upper': 1253708562.2692008
            'label': 1.0
            'mass1': 31.188553235692467
            'mass2': 28.00319443048299
            'distance': 3312.94430482992342
            'mchirp': based on m1 and m2
            'norm_dchirp': 0.5020209536920043
            'norm_dist': 0.3832194260438881
            'norm_mchirp': 0.48861046775800465
            'norm_q': 0.02843745927912822
            'norm_tc': 0.555782761582899
            
            'noise_1': np.ndarray of np.ndarrays
            'noise_2': np.ndarray of np.ndarrays
            'label': 0.0
            
        """
        
        # The data_path here refers to key within the HDF5 file
        # Get dataset attributes
        # attrs = dict(self.linked_dataset.attrs)
        # Get sample attributes
        # sample_attrs = dict(self.linked_dataset[data_path].attrs)
        # Ouputs params
        # params = attrs | sample_attrs
        
        start = time.time()
        
        # Store all params within chunk file
        params = {}
        
        with h5py.File(data_path, 'r') as fp:
            sr_grps = list(fp.keys())
            for sr_grp in sr_grps:
                
                ## Noise data
                noise_grp = fp[sr_grp]['noise']
                # Read noise data
                noise_1 = np.array(noise_grp['noise_1'])[:,:self.data_cfg.sample_length_in_num]
                noise_2 = np.array(noise_grp['noise_2'])[:,:self.data_cfg.sample_length_in_num]
                noise = np.stack([noise_1, noise_2], axis=0)
                
                if noise_realisation:
                    return noise
                    
                ## Signal data
                signal_grp = fp[sr_grp]['signal']
                # Read all signals and related params from signal grp
                h_plus = np.array(signal_grp['h_plus'])
                h_cross = np.array(signal_grp['h_cross'])
                signal = np.stack([h_plus, h_cross], axis=0)
                # Params
                params['start_time'] = np.array(signal_grp['start_time'])
                params['interval_lower'] = np.array(signal_grp['interval_lower'])
                params['interval_upper'] = np.array(signal_grp['interval_upper'])
                params['mass1'] = np.array(signal_grp['mass1'])
                params['mass2'] = np.array(signal_grp['mass2'])
                params['distance'] = np.array(signal_grp['distance'])
                params['mchirp'] = np.array(signal_grp['mchirp'])
                params['norm_dchirp'] = np.array(signal_grp['norm_dchirp'])
                params['norm_dist'] = np.array(signal_grp['norm_dist'])
                params['norm_mchirp'] = np.array(signal_grp['norm_mchirp'])
                params['norm_q'] = np.array(signal_grp['norm_q'])
                params['norm_tc'] = np.array(signal_grp['norm_tc'])
                params['sample_rate'] = fp.attrs['sample_rate']
        
        self.record_times['Load Sample'] = time.time() - start
        
        return (signal, noise, params)
    
    
    def _augmentation_(self, pure_signal, pure_noise, params):
        """ Signal and Noise only Augmentation """
        ## Convert the signal from h_plus and h_cross to h_t
        # During this procedure randomise the value of polarisation angle, ra and dec
        # This should give us the strains required (project_wave might cause issues with MP)
        start = time.time()
        
        if self.signal_only_transforms:
            sstart = time.time()
            pure_signal, times = self.signal_only_transforms(pure_signal, self.detectors, self.distrs, **params)
            self.signal_aug_times = self.signal_aug_times | times
            self.signal_aug_times['Total Time'] = time.time() - sstart
        else:
            add = {foo.__class__.__name__:0.0 for foo in self.signal_only_transforms.transforms}
            self.signal_aug_times.update(add)
            self.signal_aug_times['Total Time'] = 0.0
        
        if self.noise_only_transforms:
            nstart = time.time()
            pure_noise, times = self.noise_only_transforms(pure_noise)
            self.noise_aug_times = self.noise_aug_times | times
            self.noise_aug_times['Total Time'] = time.time() - nstart
        else:
            add = {foo.__class__.__name__:0.0 for foo in self.noise_only_transforms.transforms}
            self.noise_aug_times.update(add)
            self.noise_aug_times['Total Time'] = 0.0
            
        self.record_times['Augmentation'] = time.time() - start
        
        return (pure_signal, pure_noise)
    
    
    def _noise_realisation_(self, pure_signal, params):
        """ Finding random noise realisation for signal """
        start = time.time()
        
        if self.cfg.add_random_noise_realisation:
            # Pick a random noise realisation to add to the signal
            random_noise_data_path = random.choice(self.data_paths)
            # Read the noise data
            _pure_noise = self._read_(random_noise_data_path, noise_realisation=True)
            
            """ Calculation of Network SNR (use pure signal, before adding noise realisation) """
            # network_snr = get_network_snr(pure_sample, self.psds_data, params, self.data_loc)
            
            """ Adding noise to signals """
            if isinstance(_pure_noise, np.ndarray) and isinstance(pure_signal, np.ndarray):
                noisy_signal = pure_signal + _pure_noise
            else:
                raise TypeError('pure_signal or pure_noise is not an np.ndarray!')
            
        
        else:
            # Pick a fixed noise realisation to add to the signal
            noise_data_path = next(self.seq_noise)
            # Read the noise data
            _pure_noise = self._read_(noise_data_path, noise_realisation=True)
            
            """ Calculation of Network SNR (use pure signal, before adding noise realisation) """
            # network_snr = get_network_snr(pure_sample, self.psds_data, params, self.data_loc)
            
            """ Adding noise to signals """
            noisy_signal = pure_signal + _pure_noise
        
        self.record_times['Noise Realisation'] = time.time() - start
        
        return noisy_signal
    
    
    def _transforms_(self, noisy_sample, target):
        """ Transforms """
        start = time.time()
    
        # Apply transforms to signal and target (if any)
        if self.transforms:
            sample, times = self.transforms(noisy_sample, self.psds_data, self.data_cfg)
            self.transform_times = self.transform_times | times
            self.transform_times['Total Time'] = time.time() - start
        else:
            sample = noisy_sample
            
        if self.target_transforms:
            target = self.target_transforms(target)
        
        self.record_times['Transforms'] = time.time() - start
        return (sample, target)
    
    
    def _plotting_(self, pure_sample, _noise, noisy_sample, sample, network_snr, idx, params):
        """ Plotting idx data (if flag is set to True) """
        start = time.time()
        
        if params['label'] and self.save_counter <= self.num_sample_save and False:
            # Input parameters
            pure_signal = pure_sample
            pure_noise = _noise
            noisy_signal = noisy_sample
            if self.transforms:
                trans_pure_signal, times = self.transforms(pure_signal, self.psds_data, self.data_cfg)
            else:
                trans_pure_signal = None
                
            trans_noisy_signal = sample
            save_path = self.data_loc
            data_dir = os.path.normpath(save_path).split(os.path.sep)[-1]
            # Plotting unit data
            plot_unit(pure_signal, pure_noise, noisy_signal, trans_pure_signal, trans_noisy_signal, 
                      params['mass1'], params['mass2'], network_snr, params['sample_rate'],
                      save_path, data_dir, idx)
            # Update save counter
            self.save_counter += 1
        
        self.record_times['Plotting'] = time.time() - start
    
    
    def __getitem__(self, idx):
        
        main_start = time.time()
        # Record time taken for all sections of __getitem__
        self.record_times = {}
        self.signal_aug_times = {}
        self.noise_aug_times = {}
        self.transform_times = {}
        
        data_path = self.data_paths[idx]
        
        ## Read the sample(s)
        pure_signal, pure_noise, params = self._read_(data_path)
        
        
        try:
            ## Signal and Noise Augmentation
            pure_signal, pure_noise = self._augmentation_(pure_signal, pure_noise, params)
            ## Add noise realisation to the signals
            noisy_signal = self._noise_realisation_(pure_signal, params)
            
            ## Target handling
            target = np.concatenate((np.full(pure_signal.shape[1], 1.0), np.full(pure_noise.shape[1], 0.0)))
            target = np.reshape(target, (target.shape[0], 1))
            # Concatenating the normalised_tc within the target variable
            # target = np.append(target, normalised_tc)
            ## Target should look like (1., 0., 0.567) for signal
            ## Target should look like (0., 1., -1.0) for noise
            
            ## Transforms
            input_data = np.concatenate((noisy_signal, pure_noise), axis=1)
            sample, target = self._transforms_(input_data, target)
            sample = np.array(list(zip(sample[0], sample[1])))
            
            ## Plotting
            # self._plotting_(pure_sample, _noise, noisy_sample, sample, network_snr, idx, params)
        
        except Exception as e:
            print('\n\n{}: {}'.format(e.__class__, e))
            shutil.rmtree(self.cfg.export_dir)
            print('datasets.py: Terminated due to raised exception.')
            exit(1)
        
        
        """ Shuffling """
        zipped = list(zip(sample, target))
        random.shuffle(zipped)
        sample, target = zip(*zipped)
        
        """ Reducing memory footprint """
        # This can only be performed after transforms and augmentation
        sample = np.array(sample, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        
        """ Tensorification """
        # Convert signal/target to Tensor objects
        sample = torch.from_numpy(sample)
        target = torch.from_numpy(target)        
        
        self.record_times['Other'] = (time.time() - main_start) - sum(self.record_times.values())
        
        # Return as tuple for immutability
        all_times = {'sections': self.record_times, 
                     'signal_aug': self.signal_aug_times, 
                     'noise_aug': self.noise_aug_times, 
                     'transforms': self.transform_times}
        
        return (sample, target, all_times)



class MLMDC1_IterSample(Dataset):
    
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
        
        # Set CUDA device for pin_memory if needed
        if bool(re.search('cuda', self.cfg.store_device)):
            setattr(self, 'foo', torch.cuda.set_device(self.cfg.store_device))
        
        
        """ PSD Handling (used in whitening) """
        # Store the PSD files here in RAM. This reduces the overhead when whitening.
        # Read all psds in the data_dir and store then as FrequencySeries
        self.PSDs = {}
        psd_files = glob.glob(os.path.join(self.data_loc, "psds/*"))
        for psd_file in psd_files:
            with h5py.File(psd_file, 'r') as fp:
                data = np.array(fp['data'])
                delta_f = fp.attrs['delta_f']
                name = fp.attrs['name']
                
            psd_data = FrequencySeries(data, delta_f=delta_f)
            # Store PSD data into lookup dict
            self.PSDs[name] = psd_data
        
        if self.data_cfg.dataset == 1:
            self.psds_data = [self.PSDs['aLIGOZeroDetHighPower']]*2
            
        
        """ Multi-rate Sampling """
        # Get the sampling rates and their bins idx
        data_cfg.dbins = get_sampling_rate_bins(data_cfg)
        
        
        """ LAL Detector Objects (used in project_wave - AugPolSky) """
        # Detector objects (these are lal objects and may present problems when parallelising)
        # Create the detectors (TODO: generalise this!!!)
        detectors_abbr = ('H1', 'L1')
        self.detectors = []
        for det_abbr in detectors_abbr:
            self.detectors.append(pycbc.detector.Detector(det_abbr))
        
        
        """ PyCBC Distributions (used in AugPolSky and AugDistance) """
        ## Distribution objects for augmentation
        # Used for obtaining random polarisation angle
        self.uniform_angle_distr = distributions.angular.UniformAngle(uniform_angle=(0., 2.0*np.pi))
        # Used for obtaining random ra and dec
        self.skylocation_distr = distributions.sky_location.UniformSky()
        # Used for obtaining random mass
        self.mass_distr = distributions.Uniform(mass=(data_cfg.prior_low_mass, data_cfg.prior_high_mass))
        # Used for obtaining random chirp distance
        dist_gen = distributions.power_law.UniformRadius
        self.chirp_distance_distr = dist_gen(distance=(data_cfg.prior_low_chirp_dist, 
                                                       data_cfg.prior_high_chirp_dist))
        # Distributions object
        self.distrs = {'pol': self.uniform_angle_distr, 'sky': self.skylocation_distr,
                       'mass': self.mass_distr, 'dchirp': self.chirp_distance_distr}
        
        
        """ Data Save Params (for plotting sample just before training) """
        # Saving frequency with idx plotting
        # TODO: Add compatibility for using cfg.splitter with K-folds
        if self.data_cfg.num_sample_save == None:
            self.num_sample_save = int(len(self.data_paths)/100.0)
        else:
            self.num_sample_save = data_cfg.num_sample_save
        
        
        """ Random noise realisation """
        self.noise_idx = np.argwhere(self.targets == 0).flatten()
        self.noise_paths = self.data_paths[self.noise_idx]
        
        
        """ Keep ExternalLink Lookup table open till end of run """
        lookup = os.path.join(cfg.export_dir, 'extlinks.hdf')
        self.extmain = h5py.File(lookup, 'r', libver='latest')
        self.sample_rate = self.extmain.attrs['sample_rate']
        self.noise_low_freq_cutoff = self.extmain.attrs['noise_low_freq_cutoff']
        
        self.debug = cfg.debug
        if self.debug:
            self.debug_dir = os.path.join(cfg.export_dir, 'DEBUG')
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir, exist_ok=False)
        else:
            self.debug_dir = ''
        

    def __len__(self):
        return len(self.data_paths)

    
    def _read_(self, data_path):
        
        # Store all params within chunk file
        params = {}
        
        # Get data from ExternalLink'ed lookup file
        HDF5_Dataset, didx = os.path.split(data_path)
        # Dataset Index should be an integer
        didx = int(didx)
        # Check whether data is signal or noise with target
        target = 1 if bool(re.search('signal', HDF5_Dataset)) else 0
        # Access group
        group = self.extmain[HDF5_Dataset]
        
        if not target:
            ## Read noise data
            noise_1 = np.array(group['noise_1'][didx])
            noise_2 = np.array(group['noise_2'][didx])
            sample = np.stack([noise_1, noise_2], axis=0)
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
            params['norm_dchirp'] = group['norm_dchirp'][didx]
            params['norm_dist'] = group['norm_dist'][didx]
            params['norm_mchirp'] = group['norm_mchirp'][didx]
            params['norm_q'] = group['norm_q'][didx]
            params['norm_tc'] = group['norm_tc'][didx]
        
        # Generic params
        params['sample_rate'] = self.sample_rate
        params['noise_low_freq_cutoff'] = self.noise_low_freq_cutoff
        
        return (sample, target, params)
    
    
    def _augmentation_(self, sample, target, params, debug):
        """ Signal and Noise only Augmentation """
        ## Convert the signal from h_plus and h_cross to h_t
        # During this procedure randomise the value of polarisation angle, ra and dec
        # This should give us the strains required (project_wave might cause issues with MP)
        if target and self.signal_only_transforms:
            sample, times = self.signal_only_transforms(sample, self.detectors, self.distrs, self.debug_dir, **params)
        elif not target and self.noise_only_transforms:
            sample, times = self.noise_only_transforms(sample, self.debug_dir)
        
        return sample
    
    
    def _noise_realisation_(self, sample, target, params):
        """ Finding random noise realisation for signal """
        # Random noise realisation is the only procedure available
        # Fixed noise realisation was deprecated
        
        if self.cfg.add_random_noise_realisation and target:
            # Pick a random noise realisation to add to the signal
            random_noise_data_path = random.choice(self.noise_paths)
            # Read the noise data
            pure_noise, _, _ = self._read_(random_noise_data_path)
            
            """ Calculation of Network SNR (use pure signal, before adding noise realisation) """
            if self.debug:
                network_snr = get_network_snr(sample, self.psds_data, params, self.cfg.export_dir)
            else:
                network_snr = -1
            
            """ Adding noise to signals """
            if isinstance(pure_noise, np.ndarray) and isinstance(sample, np.ndarray):
                noisy_signal = sample + pure_noise
            else:
                raise TypeError('pure_signal or pure_noise is not an np.ndarray!')
            
        elif not self.cfg.add_random_noise_realisation and target:
            # Fixed noise realisation to add to the signal
            raise DeprecationWarning('Fixed noise realisation feature deprecated on June 8th, 2022')
            
        else:
            # If the sample is pure noise
            noisy_signal = sample
            pure_noise = None
            network_snr = -1
        
        return (noisy_signal, pure_noise, network_snr)
    
    
    def _transforms_(self, noisy_sample, target):
        """ Transforms """
        # Apply transforms to signal and target (if any)
        if self.transforms:
            sample, _ = self.transforms(noisy_sample, self.psds_data, self.data_cfg)
        else:
            sample = noisy_sample
            
        if self.target_transforms:
            target = self.target_transforms(target)
        
        return (sample, target)
    
    
    def _plotting_(self, pure_sample, pure_noise, noisy_sample, trans_noisy_sample, network_snr, idx, params):
        """ Plotting idx data (if flag is set to True) """
        # Input parameters
        if self.transforms:
            trans_pure_signal, _ = self.transforms(pure_sample, self.psds_data, self.data_cfg)
        else:
            trans_pure_signal = None
        
        save_path = self.cfg.export_dir
        data_dir = os.path.normpath(self.data_loc).split(os.path.sep)[-1]
        # Plotting unit data
        plot_unit(pure_sample, pure_noise, noisy_sample, trans_pure_signal, trans_noisy_sample, 
                  params['mass1'], params['mass2'], network_snr, params['sample_rate'],
                  save_path, data_dir, idx)
    
    
    def __getitem__(self, idx):
        
        data_path = self.data_paths[idx]
        
        ## Read the sample(s)
        sample, target, params = self._read_(data_path)
        
        
        try:
            ## Signal and Noise Augmentation
            pure_sample = self._augmentation_(sample, target, params, self.debug)
            ## Add noise realisation to the signals
            noisy_sample, pure_noise, network_snr = self._noise_realisation_(pure_sample, target, params)
            
            ## Target handling
            target = np.array([self.targets[idx]])
            # Concatenating the normalised_tc within the target variable
            # target = np.append(target, normalised_tc)
            ## Target should look like (1., 0., 0.567) for signal
            ## Target should look like (0., 1., -1.0) for noise
            
            ## Transforms
            sample, target = self._transforms_(noisy_sample, target)
            
            ## Plotting
            if self.debug:
                check_dir = os.path.join(self.cfg.export_dir, 'SAMPLES')
                if os.path.isdir(check_dir):
                    check_path = os.path.join(check_dir, '*.png')
                    num_created = len(glob.glob(check_path))
                    
                if target[0] and num_created < self.cfg.num_sample_save:
                    self._plotting_(pure_sample, pure_noise, noisy_sample, sample, network_snr, idx, params)
        
        except Exception as e:
            print('\n\n{}: {}'.format(e.__class__, e))
            shutil.rmtree(self.cfg.export_dir)
            print('datasets.py: Terminated due to raised exception.')
            exit(1)
        
        """ Reducing memory footprint """
        # This can only be performed after transforms and augmentation
        sample = np.array(sample, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        
        """ Tensorification """
        # Convert signal/target to Tensor objects
        sample = torch.from_numpy(sample)
        target = torch.from_numpy(target)
        
        return (sample, target, [], network_snr)





""" Other Loaders """


# **DEPRECATED** #
class BatchLoader(Dataset):
    """
    Batch read-and-load-type dataset object
    Designed to be be used alongside save_trainable_dataset
    Each file should contain large batch (not cfg.batch_size, bigger) number of data samples
    
    DEPRECATION: This method has been incorporated into MLMDC1 and deprecated as of May 24th, 2022
    
    """
    
    def __init__(self, data_paths, targets, transforms=None, target_transforms=None,
                 signal_only_transforms=None, noise_only_transforms=None,
                 training=False, testing=False, store_device='cpu', train_device='cpu', 
                 cfg=None, data_cfg=None, **dataset_params):
        
        raise DeprecationWarning('BatchLoader has been deprecated and may result in errors.')
        
        super().__init__()
        # Unpacking kwargs
        for key, value in dataset_params.items():
            setattr(self, key, value.to_numpy())
        
        # Primary parameters
        self.data_paths = data_paths
        self.targets = targets
        self.train_device = train_device
        self.signal_only_transforms = signal_only_transforms
        self.noise_only_transforms = noise_only_transforms
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.data_loc = os.path.join(self.data_cfg.parent_dir, self.data_cfg.data_dir)
        
        # Used for obtaining random mass
        self.mass_distr = distributions.Uniform(mass=(data_cfg.prior_low_mass, data_cfg.prior_high_mass))
        # Used for obtaining random chirp distance
        dist_gen = distributions.power_law.UniformRadius
        self.chirp_distance_distr = dist_gen(distance=(data_cfg.prior_low_chirp_dist, 
                                                       data_cfg.prior_high_chirp_dist))
        # Distributions object
        self.distrs = {'dchirp': self.chirp_distance_distr}
        
        # Saving frequency with idx plotting
        # TODO: Add compatibility for using cfg.splitter with K-folds
        if self.data_cfg.sample_save_frequency == None:
            self.sample_save_frequency = int(len(self.data_paths)/100.0)
        else:
            self.sample_save_frequency = data_cfg.sample_save_frequency


    def __len__(self):
        return len(self.data_paths)

    def _read_(self, data_path):
        """ Read sample and return necessary training params """
        # Should contain an entire batch of data samples
        with h5py.File(data_path, "r") as fp:
            # Get and return the batch data
            return np.array(fp['data'][:])
    
    def _demystify_garbage_(self, garbage):
        # targets looks like absolute shite after coming out of the booty that is trainable.json
        # TODO: This is very inefficient. Fix me!!! Didn't mean the line, me :(
        # Hello darkness, my old friend. Evenings spent in the lab again.
        # These are easter eggs. I'm not actually sad. Or am i? No. Or am i?
        # Use this to clean up and obtain targets
        return np.array(list(garbage), dtype=np.float64)
    
    def __getitem__(self, idx):
        
        """ Read the sample """
        # check whether the sample is noise/signal for adding random noise realisation
        data_path = self.data_paths[idx]
        batch_signals = self._read_(data_path)
            
        """ Target """
        # Target for training or testing phase (obtained from trainable.json)
        batch_targets = self._demystify_garbage_(self.targets[idx])
        
        # Concatenating the normalised_tc within the target variable
        # This can be used when normalised_tc is also stored in trainable.hdf
        # normalised_tc = self.norm_tc
        # target = np.append(target, normalised_tc)
        ## Target should look like (1., 0., 0.567) for signal
        ## Target should look like (0., 1., -1.0) for noise
        
        
        """ Finding random noise realisation for signals """
        if self.cfg.add_random_noise_realisation:
            # Find the signals within the batch, and add secondary noise to it
            primary_signal_idx = np.argwhere(batch_targets[:,0] == 1.).flatten()
            # Pick a secondary batch to snatch noise data
            secondary_file_idx = random.choice(range(len(self.data_paths)))
            # Read secondary file for all data
            secondary_data = self._read_(self.data_paths[secondary_file_idx])
            # Find the indices where this file contains pure noise
            secondary_noise_idx = np.argwhere(np.array(self.targets[secondary_file_idx])[:,1] == 1.).flatten()
            # Sometimes batch may have lower number of samples due to non-unique seed error
            # NOTE: This procedure is not necessary with the new data generation code
            if len(secondary_noise_idx) < len(primary_signal_idx):
                primary_signal_idx = primary_signal_idx[:len(secondary_noise_idx)]
            elif len(secondary_noise_idx) > len(primary_signal_idx):
                secondary_noise_idx = secondary_noise_idx[:len(primary_signal_idx)]
            # Shuffle the secondary idx to add an extra layer of randomness
            np.random.shuffle(secondary_noise_idx)
            
            # Get the secondary noise data
            secondary_noise = secondary_data[secondary_noise_idx]
            # Get the primary signal data
            primary_signals = batch_signals[primary_signal_idx]
            # This assertion should not trigger if we use StratifiedKFold splitting method
            assert len(primary_signal_idx) == len(secondary_noise_idx)
            
            """ Distance Augmentation to the signals in our batch """
            # Get the required params alone for distance and mchirp
            distance = np.array(self.distance[idx])[primary_signal_idx]
            mchirp = np.array(self.mchirp[idx])[primary_signal_idx]
            # We do this before adding any noise to it
            primary_signals = self.signal_only_transforms(primary_signals, distrs=self.distrs, 
                                            **{'distance': distance, 'mchirp':mchirp})
            
            # Add the secondary noise to the primary signals
            """ Augmentation (cyclic_shift) to secondary noise """
            # Cyclic shift may not be possible if we use transformed signal as input
            batch_signals[primary_signal_idx] = secondary_noise + primary_signals
            # Now all noise should be untouched, and signals should have random noise added
            batch_samples = batch_signals
        
        """
        import matplotlib.pyplot as plt
        plt.plot(range(len(batch_samples[0][0])), batch_samples[0][0])
        plt.savefig("sample.png")
        print(primary_signals)
        plt.plot(range(len(primary_signals[0][0])), primary_signals[0][0])
        plt.savefig("signal.png")
        plt.plot(range(len(secondary_noise[0][0])), secondary_noise[0][0])
        plt.savefig("noise.png")
        plt.plot(range(len(batch_signals[primary_signal_idx][0][0])), batch_signals[primary_signal_idx][0][0])
        plt.savefig("noisy_signal.png")
        """
        
        """ Tensorification and Device Compatibility """
        # Convert signal/target to Tensor objects
        samples = torch.from_numpy(batch_samples)
        targets = torch.from_numpy(batch_targets)
        
        # Set the device and dtype
        global tensor_dtype
        samples = samples.to(dtype=tensor_dtype, device=self.train_device)
        targets = targets.to(dtype=tensor_dtype, device=self.train_device)
        
        # Return as tuple for immutability
        return (samples, targets)


class Simple(Dataset):
    """
    Simple read-and-load-type dataset object
    Designed to be be used alongside BatchLoader
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
