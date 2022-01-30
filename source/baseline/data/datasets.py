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
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

# Datatype for storage
data_type=torch.float32

""" Dataset Objects """

class MLMDC1(Dataset):
    """
    
    """
    
    def __init__(self, data_paths, targets, transforms=None, target_transforms=None,
                 training=False, testing=False, store_device='cuda:0', train_device='cuda:0'):
        
        self.data_paths = data_paths
        self.targets = targets
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.training = training
        self.testing = testing
        self.store_device = store_device
        self.train_device = train_device
        
        if training:
            assert testing == False
        if testing:
            assert training == False
        if not training and not testing:
            raise ValueError("Neither training or testing phase chosen for dataset class.")

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
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
            # Convert both data segments into numpy arrays
            signal_1 = np.zeros(data_1.shape)
            signal_2 = np.zeros(data_2.shape)
            data_1.read_direct(signal_1)
            data_2.read_direct(signal_2)
            
            # Get the 'tc' attribute from the foreground file IF in training phase
            if self.training:
                attrs = dict(gfile.attrs)
                tc = attrs['tc']
                # Normalise 'tc' such that it is always b/w 0 and 1
                start_time = attrs['start_time']
                # NOTE: For training, we assume that global start time is always 0.0
                # Normalisation (Subtract start_time from 'tc' and normalise wrt duration)
                # NOTE: Also assuming a constant 20.0 second segment for all training data
                # Depending on llimit and ulimit for 'tc', the value should be ~[0.6, 0.9]
                tc = (tc - start_time) / 20.0
            
        signal = np.row_stack((signal_1, signal_2)).astype(np.float32)
        
        # Target for training or testing phase
        target = np.array(self.targets[idx]).astype(np.float32)
        if self.training:
            target = np.column_stack((target, tc))
        
        # Apply transforms to signal and target (if any)
        if self.transforms:
            signal = self.transforms(signal)
        if self.target_transforms:
            target = self.target_transforms(target)
        
        # Convert signal/target to Tensor objects
        signal = torch.from_numpy(signal)
        target = torch.from_numpy(target)
        # Set the device and dtype
        signal = signal.to(dtype=data_type, device=self.train_device)
        target = target.to(dtype=data_type, device=self.train_device)
        # Return as tuple for immutability
        return tuple(signal, target)
