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
import h5py
import time
import glob
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# LOCAL
from data.snr_calculation import get_network_snr
from data.plot_dataloader_unit import plot_unit
from data.multirate_sampling import get_sampling_rate_bins

# PyCBC
import pycbc
from pycbc import distributions
from pycbc.types import load_frequencyseries, FrequencySeries

# Datatype for storage
tensor_dtype=torch.float32

""" Dataset Objects """

class MLMDC1(Dataset):
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
                 signal_only_transforms=None, training=False, testing=False, 
                 store_device='cpu', train_device='cpu', data_cfg=None):
        
        super().__init__()
        self.data_paths = data_paths
        self.targets = targets
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.signal_only_transforms = signal_only_transforms
        self.training = training
        self.testing = testing
        self.store_device = store_device
        self.train_device = train_device
        self.data_cfg = data_cfg
        self.data_loc = os.path.join(self.data_cfg.parent_dir, self.data_cfg.data_dir)
        
        # Store the PSD files here in RAM. This reduces the overhead when whitening.
        # Read all psds in the data_dir and store then as FrequencySeries
        self.PSDs = {}
        psd_files = glob.glob(os.path.join(self.data_loc, "psds/*"))
        for psd_file in psd_files:
            try:
                # This should load the PSD as a FrequencySeries object with delta_f assigned
                psd_data = load_frequencyseries(psd_file)
            except:
                data = pd.read_hdf(psd_file, 'data').to_numpy().flatten()
                psd_data = FrequencySeries(data, delta_f=data_cfg.delta_f)
            # Store PSD data into lookup dict
            self.PSDs[psd_file] = psd_data
        
        # Multi-rate sampling
        # Get the sampling rates and their bins idx
        data_cfg.dbins = get_sampling_rate_bins(data_cfg)
        
        # Detector objects (these are lal objects and may present problems when parallelising)
        # Create the detectors (TODO: generalise this!!!)
        detectors_abbr = ('H1', 'L1')
        self.detectors = []
        for det_abbr in detectors_abbr:
            self.detectors.append(pycbc.detector.Detector(det_abbr))
            
        ## Distribution objects for augmentation
        # Used for obtaining random polarisation angle
        self.uniform_angle_distr = distributions.angular.UniformAngle(uniform_angle=(0., 2.0*np.pi))
        # Used for obtaining random ra and dec
        self.skylocation_dist = distributions.sky_location.UniformSky()
        # Distributions object
        self.distrs = {'pol': self.uniform_angle_distr, 'sky': self.skylocation_dist}
            
        # Saving frequency with idx plotting
        # TODO: Add compatibility for using cfg.splitter with K-folds
        if self.data_cfg.sample_save_frequency == None:
            self.sample_save_frequency = int(len(self.data_paths)/100.0)
        else:
            self.sample_save_frequency = data_cfg.sample_save_frequency
        
        if training:
            assert testing == False
        if testing:
            assert training == False
        if not training and not testing:
            raise ValueError("Neither training or testing phase chosen for dataset class. Bruh?")


    def __len__(self):
        return len(self.data_paths)

    
    def _read_(self, data_path):
        """ Read sample and return necessary training params """
        with h5py.File(data_path, "r") as gfile:
            ## Reading all data parameters
            # Detectors
            dets = list(gfile.keys())
            # Groups within detectors (times as dict)
            detector_group_1 = gfile[dets[0]]
            detector_group_2 = gfile[dets[1]]
            # Times as list
            times_1 = list(detector_group_1.keys())
            times_2 = list(detector_group_2.keys())
            # Noise data within each detector
            data_1 = detector_group_1[times_1[0]]
            data_2 = detector_group_2[times_2[0]]
            # Stack the signals together
            signals = np.stack([data_1, data_2], axis=0)
            
            # Get the target variable from HDF5 attribute
            attrs = dict(gfile.attrs)
            label_saved = attrs['label']
            
            # if the sample is pure noise
            if np.allclose(label_saved, np.array([0., 1.])):
                sample_rate = attrs['sample_rate']
                # *Only* noise files have these attributes
                noise_low_freq_cutoff = attrs['noise_low_freq_cutoff']
                psd_1 = attrs['psd_file_path_det1']
                psd_2 = attrs['psd_file_path_det2']
            
            # if the sample is pure signal
            if np.allclose(label_saved, np.array([1., 0.])):
                m1 = attrs['mass_1']
                m2 = attrs['mass_2']
                # Use this to get the h_t from project_wave transformation
                time_interval = attrs['time_interval']
            
            # Use Normalised 'tc' such that it is always b/w 0 and 1
            # if place_tc = [0.5, 0.7]s in a 1.0 second sample
            # then normalised tc will be (place_tc-0.5)/(0.7-0.5)
            # tc is the GPS time in O3 era when injection is made
            tc = attrs['tc']
            normalised_tc = attrs['normalised_tc']
        
        # Return necessary params
        if np.allclose(label_saved, np.array([1., 0.])): # pure signal
            return ('signal', signals, label_saved, tc, normalised_tc, m1, m2, time_interval)
        elif np.allclose(label_saved, np.array([0., 1.])): # pure noise
            return ('noise', signals, label_saved, sample_rate, noise_low_freq_cutoff, 
                    psd_1, psd_2, tc, normalised_tc)
        else:
            raise ValueError("MLMDC1 dataset: sample label is not one of (1., 0.), (0., 1.)")
    
    
    def __getitem__(self, idx):
        
        data_path = self.data_paths[idx]
        
        """ Read the sample """
        # check whether the sample is noise/signal for adding random noise realisation
        data_params = self._read_(data_path)
        data_type = data_params[0]
        
        # Get sample params
        if data_type == 'signal':
            _, signals, label_saved, tc, normalised_tc, m1, m2, time_interval = data_params
        elif data_type == 'noise':
            _, noise, label_saved, sample_rate, noise_low_freq_cutoff, \
            psd_1, psd_2, tc, normalised_tc = data_params
            # Concat psds
            psds = [psd_1, psd_2]
        
        
        """ Convert the signal from h_plus and h_cross to h_t """
        # During this procedure randomise the value of polarisation angle, ra and dec
        # This should give us the strains required (project_wave might cause issues with MP)
        if data_type == 'signal' and self.signal_only_transforms:
            signals = self.signal_only_transforms(signals, self.detectors, time_interval, self.distrs)
        
        
        """ Finding *ONE* random noise realisation for signals """
        if data_type == 'signal':
            noise_dir = os.path.join(self.data_loc, "background")
            noise_files = glob.glob(os.path.join(noise_dir, "*.hdf"))
            # Pick a random noise realisation to add to the signal
            noise_data_path = random.choice(noise_files)
            # Read the noise data
            noise_data_params = self._read_(noise_data_path)
            # Sanity check
            if noise_data_params[0] != 'noise':
                raise ValueError("MLMDC1 dataset: random noise path did not return noise file!")
            
            # Extracting the noise params
            _, noise, _, sample_rate, noise_low_freq_cutoff, psd_1, psd_2, _, _ = noise_data_params
            # Concat PSD paths
            psds = [psd_1, psd_2]
        
            ############################################################################
            # For 'N' random noise realisations:
            #     If we need 'n' random noise realisation for each signal, then
            #     duplicate the signal data_paths in training.hdf 'n' times.
            #     Each of these duplicates will be assigned a random noise path and
            #     thus can be considered an entirely new signal.
            ############################################################################
        
            """ Calculation of Network SNR (use pure signal, before adding noise realisation) """
            network_snr = get_network_snr(signals, psds, sample_rate, noise_low_freq_cutoff, self.data_loc)
            
            """ Adding noise to signals """
            raw_sample = noise + signals
        
        else:
            raw_sample = noise
            
        
        """ Target """
        # Target for training or testing phase (obtained from training.hdf)
        # labels in training.hdf *ONLY* specify whether given sample is signal or not
        label_check = np.array([float(self.targets[idx]), 1.0-float(self.targets[idx])])
        # Sanity check for labels and storage
        if not np.allclose(label_saved, label_check):
            raise ValueError("MLMDC1 dataset: label_saved and label_check are not equal!")
        
        # Save label_saved into the target variable
        # Both label_saved and label_check should be a numpy array
        target = label_saved.astype(np.float64)
        # Concatenating the normalised_tc within the target variable
        # target = np.append(target, normalised_tc)
        ## Target should look like (1., 0., 0.567) for signal
        ## Target should look like (0., 1., -1.0) for noise
        
        
        """ Transforms """
        # Apply transforms to signal and target (if any)
        if self.transforms:
            psds_data = [self.PSDs[psd_name] for psd_name in psds]
            sample = self.transforms(raw_sample, psds_data, self.data_cfg)
        if self.target_transforms:
            target = self.target_transforms(target)
        
        
        """ Plotting idx data (if flag is set to True) """
        # TODO: This does not save when needed. Look into this!
        if data_type == 'signal' and idx % self.sample_save_frequency == 0 and idx!=0:
            # Input parameters
            pure_signal = signals
            pure_noise = noise
            noisy_signal = raw_sample
            if self.transforms:
                psds_data = [self.PSDs[psd_name] for psd_name in psds]
                trans_pure_signal = self.transforms(pure_signal, psds_data, self.data_cfg)
            else:
                trans_pure_signal = None
            trans_noisy_signal = sample
            save_path = self.data_loc
            data_dir = os.path.normpath(save_path).split(os.path.sep)[-1]
            # Plotting unit data
            plot_unit(pure_signal, pure_noise, noisy_signal, trans_pure_signal, trans_noisy_signal,
                      m1, m2, network_snr, sample_rate, save_path, data_dir, idx)
        
        
        """ Tensorification and Device Compatibility """
        # Convert signal/target to Tensor objects
        # Check: Converting to a tensor normalises a sample b/w the range [0., 1.]
        sample = torch.from_numpy(sample)
        target = torch.from_numpy(target)
        # Set the device and dtype
        global tensor_dtype
        sample = sample.to(dtype=tensor_dtype, device=self.train_device)
        target = target.to(dtype=tensor_dtype, device=self.train_device)
        
        # Return as tuple for immutability
        return (sample, target)


""" Other Loaders """

class BatchLoader(Dataset):
    """
    Batch read-and-load-type dataset object
    Designed to be be used alongside save_trainable_dataset
    Each file should contain cfg.batch_size number of data samples
    
    """
    
    def __init__(self, data_paths, targets, transforms=None, target_transforms=None,
                 training=False, testing=False, store_device='cpu', train_device='cpu',
                 data_cfg=None):
        
        super().__init__()
        self.data_paths = data_paths
        self.targets = targets
        self.transforms = None
        self.target_transforms = None
        self.training = training
        self.testing = testing
        self.store_device = store_device
        self.train_device = train_device
        self.data_cfg = data_cfg
        self.data_loc = os.path.join(self.data_cfg.parent_dir, self.data_cfg.data_dir)
        
        # Saving frequency with idx plotting
        # TODO: Add compatibility for using cfg.splitter with K-folds
        if self.data_cfg.sample_save_frequency == None:
            self.sample_save_frequency = int(len(self.data_paths)/100.0)
        else:
            self.sample_save_frequency = data_cfg.sample_save_frequency
        
        if training:
            assert testing == False
        if testing:
            assert training == False
        if not training and not testing:
            raise ValueError("Neither training or testing phase chosen for dataset class. Bruh?")


    def __len__(self):
        return len(self.data_paths)

    def _read_(self, data_path):
        """ Read sample and return necessary training params """
        # Should contain an entire batch of data samples
        with h5py.File(data_path, "r") as fp:
            # Get and return the batch data
            # When BatchLoader is True, batch_size is 1, therefore [0]
            return np.array(fp['data'][:])
    
    def __getitem__(self, idx):
        
        """ Read the sample """
        # check whether the sample is noise/signal for adding random noise realisation
        data_path = self.data_paths[idx]
        batch_samples = self._read_(data_path)
        
        ## TODO: We should add a random noise realisation to the signal here!!!
        ## Remove that method from the save trainable dataset method
        
        """ Target """
        # Target for training or testing phase (obtained from trainable.json)
        # TODO: This is very inefficient. Fix me!!!
        batch_targets = np.array(list(self.targets[idx]), dtype=np.float64)
        
        # Concatenating the normalised_tc within the target variable
        # This can be used when normalised_tc is also stored in trainable.hdf
        # target = np.append(target, normalised_tc)
        ## Target should look like (1., 0., 0.567) for signal
        ## Target should look like (0., 1., -1.0) for noise
        
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
