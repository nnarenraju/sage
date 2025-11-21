#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : real_noise.py
Description     : Short description of the file

Created on 2025-11-06 15:00:16

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import os
import h5py
import math
import glob
import json
import pickle
import scipy
import itertools
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sys import getsizeof

import sys
import time
import multiprocessing as mp

from tqdm import tqdm
from scipy import signal

from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.types import TimeSeries as TS

from pycbc import DYN_RANGE_FAC


def downsample(strain, sample_rate=2048.):
    res = resample_to_delta_t(strain, 1./sample_rate)
    ret = highpass(res, low_freq_cutoff).astype(np.float32)
    ret = ret.time_slice(float(ret.start_time) + crop_after_decimation,
                         float(ret.end_time) - crop_after_decimation)
    return ret

def get_detector_data(args):
    n, left_boundary, right_boundary, detector = args
    success = False
    try:
        data = TimeSeries.fetch_open_data(detector, left_boundary, right_boundary, cache=1)
        success = True
    except:
        raise

    if success:
        data = TS(data.value, delta_t=data.dt.value)
        data = downsample(data).numpy()
        data = data * DYN_RANGE_FAC
        data = data.astype(np.float32)
        return (n, data)
    else:
        return (n, None)

# Fetch data in MP && save data chunk
def fetcher(GPS_boundaries, num_workers=4, det="", run="", parent_dir=""):
    # Make directory if not present
    savedir_path = os.path.join(parent_dir, "data_{}_{}".format(det, run))
    if not os.path.exists(savedir_path):
        os.makedirs(savedir_path, exist_ok=False)
    
    detector_data = []
    print('Fetching GWOSC data for detector {} ({}) using {} cores'.format(det, run, num_workers))
    # Download data at each GPS range
    if num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            with tqdm(total=len(GPS_boundaries)) as pbar:
                pbar.set_description("MP-DET_SCIENCE_DATA GWOSC")
                for out in pool.imap_unordered(get_detector_data, [(n, bound[0], bound[1], det) for n, bound in enumerate(GPS_boundaries)]):
                    n, data = out
                    if isinstance(data, np.ndarray):
                        with h5py.File(os.path.join(savedir_path, 'data_{}_{}_chunk_{}.hdf'.format(det, run, n)), 'a') as hf:
                            hf.create_dataset('data', data=data, compression="gzip", chunks=True)
                    pbar.update()
    else:
        with tqdm(total=len(GPS_boundaries)) as pbar:
            pbar.set_description("DET_SCIENCE_DATA GWOSC")
            for args in [(n, bound[0], bound[1], det) for n, bound in enumerate(GPS_boundaries)]:
                n, data = get_detector_data(args)
                if isinstance(data, np.ndarray):
                    with h5py.File(os.path.join(savedir_path, 'data_{}_{}_chunk_{}.hdf'.format(det, run, n)), 'a') as hf:
                        hf.create_dataset('data', data=data, compression="gzip", chunks=True)
                pbar.update()


if __name__ == '__main__':

    # PARAMS
    low_freq_cutoff = 15.0 # Hz
    minimum_segment_duration = 22.0 # seconds
    crop_after_decimation = 2.5 # seconds

    # DATA flag was included and none of the flags {CBC_CAT1, CBC_CAT2, CBC_HW_INJ, or BURST_HW_INJ} were included
    link_O3a_H1 = "https://gwosc.org/timeline/segments/O3a_4KHZ_R1/H1_DATA/1238166018/15811200/"
    link_O3a_L1 = "https://gwosc.org/timeline/segments/O3a_4KHZ_R1/L1_DATA/1238166018/15811200/"
    link_O3a_V1 = "https://gwosc.org/timeline/segments/O3a_4KHZ_R1/V1_DATA/1238166018/15811200/"
    link_O3b_H1 = "https://gwosc.org/timeline/segments/O3b_4KHZ_R1/H1_DATA/1256655618/12708000/"
    link_O3b_L1 = "https://gwosc.org/timeline/segments/O3b_4KHZ_R1/L1_DATA/1256655618/12708000/"
    link_O3b_V1 = "https://gwosc.org/timeline/segments/O3b_4KHZ_R1/V1_DATA/1256655618/12708000/"
    # Downlaod segments data
    urllib.request.urlretrieve(link_O3a_H1, "./data/H1_O3a_segments.txt")
    urllib.request.urlretrieve(link_O3a_L1, "./data/L1_O3a_segments.txt")
    urllib.request.urlretrieve(link_O3a_V1, "./data/V1_O3a_segments.txt")
    urllib.request.urlretrieve(link_O3b_H1, "./data/H1_O3b_segments.txt")
    urllib.request.urlretrieve(link_O3b_L1, "./data/L1_O3b_segments.txt")
    urllib.request.urlretrieve(link_O3b_V1, "./data/V1_O3b_segments.txt")

    # Data collected for O3a and O3b
    H1_O3a_segments = np.loadtxt('./data/H1_O3a_segments.txt')
    L1_O3a_segments = np.loadtxt('./data/L1_O3a_segments.txt')
    V1_O3a_segments = np.loadtxt('./data/V1_O3a_segments.txt')
    H1_O3b_segments = np.loadtxt('./data/H1_O3b_segments.txt')
    L1_O3b_segments = np.loadtxt('./data/L1_O3b_segments.txt')
    V1_O3b_segments = np.loadtxt('./data/V1_O3b_segments.txt')

    check_url = False

    # Check available valid data segments
    ranges = {'O3a': {'H1':list(zip(H1_O3a_segments[:,0], H1_O3a_segments[:,1], H1_O3a_segments[:,2])), 
                      'L1':list(zip(L1_O3a_segments[:,0], L1_O3a_segments[:,1], L1_O3a_segments[:,2])),
                      'V1':list(zip(V1_O3a_segments[:,0], V1_O3a_segments[:,1], V1_O3a_segments[:,2]))}, 
              'O3b': {'H1':list(zip(H1_O3b_segments[:,0], H1_O3b_segments[:,1], H1_O3b_segments[:,2])), 
                      'L1':list(zip(L1_O3b_segments[:,0], L1_O3b_segments[:,1], L1_O3b_segments[:,2])),
                      'V1':list(zip(V1_O3b_segments[:,0], V1_O3b_segments[:,1], V1_O3b_segments[:,2]))}
             }

    valid_det_bounds = {'O3a': {'H1':[], 'L1':[], 'V1':[]}, 'O3b': {'H1':[], 'L1':[], 'V1':[]}}

    for coord in itertools.product(['O3a', 'O3b'], ['H1', 'L1', 'V1']):
        total_valid_duration = 0
        available_valid_duration = 0
        run = coord[0] # O3a or O3b
        det = coord[1] # H1, L1 or V1
        print(det, run)

        for det_start, det_end, dur in tqdm(ranges[run][det]):
            available_valid_duration += dur
            if det_end - det_start < minimum_segment_duration:
                continue
            if check_url:
                try:
                    # if these can run, the whole data segment is available
                    # Getting one second of data on the edges of the data segment
                    TimeSeries.fetch_open_data(det, det_start, det_start+1, cache=1)
                    TimeSeries.fetch_open_data(det, det_end-1, det_end, cache=1)
                except:
                    raise
            # Getting all valid segments based on duration
            boundary = np.array([det_start, det_end])
            valid_det_bounds[run][det].append(boundary)
            total_valid_duration += det_end - det_start
        
        # Comparing available and total valid duration for the given run+det
        print('Total available duration in {} for {} = {}'.format(det, run, available_valid_duration))
        print('Total valid duration in {} for {} = {}\n'.format(det, run, total_valid_duration))
        time.sleep(1) # for some reason the print statements are omitted when using tqdm without this line. welp.

    # Get all data segments from H1 for O3a valid GPS times
    fetcher(valid_det_bounds['O3a']['H1'], num_workers=8, det='H1', run='O3a', parent_dir="/data/wiay/nnarenraju/O3_SGWC_DATA")

    # Get all data segments from H1 for O3a valid GPS times
    fetcher(valid_det_bounds['O3a']['L1'], num_workers=8, det='L1', run='O3a', parent_dir="/data/wiay/nnarenraju/O3_SGWC_DATA")

    # Get all data segments from H1 for O3a valid GPS times
    fetcher(valid_det_bounds['O3a']['V1'], num_workers=8, det='V1', run='O3a', parent_dir="/data/wiay/nnarenraju/O3_SGWC_DATA")

    # Get all data segments from H1 for O3a valid GPS times
    fetcher(valid_det_bounds['O3b']['H1'], num_workers=8, det='H1', run='O3b', parent_dir="/data/wiay/nnarenraju/O3_SGWC_DATA")

    # Get all data segments from H1 for O3a valid GPS times
    fetcher(valid_det_bounds['O3b']['L1'], num_workers=8, det='L1', run='O3b', parent_dir="/data/wiay/nnarenraju/O3_SGWC_DATA")

    # Get all data segments from H1 for O3a valid GPS times
    fetcher(valid_det_bounds['O3b']['V1'], num_workers=8, det='V1', run='O3b', parent_dir="/data/wiay/nnarenraju/O3_SGWC_DATA")
