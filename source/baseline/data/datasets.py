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
import glob
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# LOCAL
from data.snr_calculation import get_network_snr
from data.plot_dataloader_unit import plot_unit

# Datatype for storage
tensor_dtype=torch.float32

""" Dataset Objects """

class MLMDC1(Dataset):
    """
    
    """
    
    def __init__(self, data_paths, targets, transforms=None, target_transforms=None,
                 training=False, testing=False, store_device='cpu', train_device='cpu',
                 data_cfg=None):
        
        super().__init__()
        self.data_paths = data_paths
        self.targets = targets
        self.transforms = transforms
        self.target_transforms = target_transforms
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
            
            # Use Normalised 'tc' such that it is always b/w 0 and 1
            # if place_tc = [0.5, 0.7]s in a 1.0 second sample
            # then normalised tc will be (place_tc-0.5)/(0.7-0.5)
            # tc is the GPS time in O3 era when injection is made
            tc = attrs['tc']
            normalised_tc = attrs['normalised_tc']
        
        # Return necessary params
        if np.allclose(label_saved, np.array([1., 0.])): # pure signal
            return ('signal', signals, label_saved, tc, normalised_tc, m1, m2)
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
            _, signals, label_saved, tc, normalised_tc, m1, m2 = data_params
        elif data_type == 'noise':
            _, noise, label_saved, sample_rate, noise_low_freq_cutoff, \
            psd_1, psd_2, tc, normalised_tc = data_params
            # Concat psds
            psds = [psd_1, psd_2]
        
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
            sample = self.transforms(raw_sample, psds, self.data_cfg)
        if self.target_transforms:
            target = self.target_transforms(target)
        
        
        """ Plotting idx data (if flag is set to True) """
        if data_type == 'signal' and idx % self.sample_save_frequency == 0:
            # Input parameters
            pure_signal = signals
            pure_noise = noise
            noisy_signal = raw_sample
            if self.transforms:
                trans_pure_signal = self.transforms(pure_signal, psds, self.data_cfg)
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
        sample = torch.from_numpy(sample)
        target = torch.from_numpy(target)
        # Set the device and dtype
        global tensor_dtype
        sample = sample.to(dtype=tensor_dtype, device=self.train_device)
        target = target.to(dtype=tensor_dtype, device=self.train_device)
        
        # Return as tuple for immutability
        return (sample, target)




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
            return np.array(fp['data'][:])[0]
    
    def __getitem__(self, idx):
        
        """ Read the sample """
        # check whether the sample is noise/signal for adding random noise realisation
        data_path = self.data_paths[idx]
        batch_samples = self._read_(data_path)
        
        """ Target """
        # Target for training or testing phase (obtained from trainable.json)
        # TODO: This is very inefficient. Fix me!!!
        batch_targets = np.array(list(self.targets), dtype=np.float64)
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
