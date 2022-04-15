#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Apr 14 15:02:10 2022

__author__      = nnarenraju
__copyright__   = Copyright 2022, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

import glob
import h5py
import numpy as np
import itertools
from tqdm import tqdm


def _read_(data_path):
    """ Read entire fold and return necessary trainable data """
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
            psd_1 = attrs['psd_file_path_det1']
            psd_2 = attrs['psd_file_path_det2']
        
        # if the sample is pure signal
        if np.allclose(label_saved, np.array([1., 0.])):
            m1 = attrs['mass_1']
            m2 = attrs['mass_2']
            mchirp = (m1*m2 / (m1+m2)**2.)**(3./5) * (m1 + m2)
            distance = attrs['distance']
        
        # Use Normalised 'tc' such that it is always b/w 0 and 1
        # if place_tc = [0.5, 0.7]s in a 1.0 second sample
        # then normalised tc will be (place_tc-0.5)/(0.7-0.5)
        normalised_tc = attrs['normalised_tc']
    
    # Return necessary params
    if np.allclose(label_saved, np.array([1., 0.])): # pure signal
        return ('signal', signals, label_saved, normalised_tc, mchirp, distance)
    elif np.allclose(label_saved, np.array([0., 1.])): # pure noise
        return ('noise', signals, label_saved, psd_1, psd_2, normalised_tc)
    else:
        raise ValueError("MLMDC1 dataset: sample label is not one of (1., 0.), (0., 1.)")



files = glob.glob("/Users/nnarenraju/Desktop/dataset_5e4_20s_D1_Batch_17/background/*.hdf")
all_noise = []
for file in files[:1000]:
    _, data, label, _, _, _ = _read_(file)
    all_noise.append(data)

all_perms = itertools.permutations(all_noise, 2)
length = len(list(all_perms))
pbar = tqdm(all_perms) # LOL doesn't work
for n, (a, b) in enumerate(pbar):
    pbar.set_description("Noise samples {}/{}".format(n, length))
    if np.allclose(a, b, atol=1e-45, rtol=1e-45):
        raise ValueError("Noise files are the same!")

print("\nSuccess! All noise files are unique.")
