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


""" Dataset Objects """

class MLMDC1(Dataset):
    """
    
    """
    
    def __init__(self, data_paths, targets, transforms=None, target_transforms=None):
        self.data_paths = data_paths
        self.targets = targets
        self.transforms = transforms
        self.target_transforms = target_transforms

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
            
        signal = np.row_stack((signal_1, signal_2)).astype(np.float32)
        target = self.targets[idx]
        # Apply transforms to signal and target (if any)
        if self.transforms:
            signal = self.transforms(signal)
        if self.target_transforms:
            target = self.target_transforms(target)
        
        # Convert signal/target to Tensor objects
        signal = torch.as_tensor(signal)
        target = torch.as_tensor(target)
        # Return as tuple for immutability
        return tuple(signal, target)
