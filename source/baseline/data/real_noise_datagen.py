#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = real_noise_datagen.py
Description     = Handling Dataset 4 real O3a noise for training and testing dataset

Created on Tue Jan 24 13:55:47 2023

__author__      = nnarenraju
__copyright__   = Copyright 2022, ORChiD
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = inProgress


Github Repository: NULL

Documentation:
    
    1. Get the start and end times of all required segments for testing and training datasets (FIN)
    2. Start time of training dataset should be end time of testing dataset + some buffer (FIN)
    3. Save a 30 day testing dataset in a separate HDF5 file (FIN)
    4. Save a (total_ndays - ndays_testing) as training dataset in a separate HDF5 file (FIN)
    *** Till here for O3a testing dataset ***
    5. Run the Slicer class with a given step size (based on req. overlap) on training dataset
    6. Each signal_length sample given by Slicer is saved as a noise sample for D4
    7. Assign unique IDs to each sample. Our default is 500_000 total noise samples
    8. Call the RealNoiseGenerator during the worker phase to access req. noise sample for given idx

"""

# Modules
import os
import csv
import h5py
import requests
import warnings
import numpy as np

# PyCBC and ligo
import ligo.segments

from pycbc import DYN_RANGE_FAC
from pycbc.types import TimeSeries
from segments import OverlapSegment

# LOCAL
from data.testdata_slicer import Slicer


class RealNoiseGenerator:
    """
    Part of the code has been taken and modified from the MLGWSC-1 Github repository
    This is to partly to ensure that the O3a real noise created for training is handled in a 
    similar manner to testing.
    
    """
    
    def __init__(self, **kwargs):
        # Get all required attributes for real noise generation
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        ## TESTING DATASET
        mode = 'testing'
        self.testing_segments = []
        self.testing_buffer = 8.0 # seconds
        self.get_real_noise(mode=mode)
        
        ## TRAINING DATASET
        mode = 'training'
        self.duration = 999_999_999_999 # impossibly large duration
        self.available_training_duration = 0.0
        self.training_buffer = 8.0 # seconds
        self.get_real_noise(mode=mode)
        
    
    def load_segments(self):
        tmp_dir = "./tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=False)
        path = os.path.join(tmp_dir, 'segments.csv')
        # Download data if it does not exist
        if not os.path.isfile(path):
            # TODO: At some point, this will not exist. Remove dependancy.
            url = 'https://www.atlas.aei.uni-hannover.de/work/marlin.schaefer/MDC/segments.csv'
            response = requests.get(url)
            with open(path, 'wb') as fp:
                fp.write(response.content)

        # Load data from CSV file
        segs = ligo.segments.segmentlist([])
        with open(path, 'r') as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                idx, start, end = row
                segs.append(ligo.segments.segment([int(start), int(end)]))

        return segs
    
    
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
    
        group = f'{det}/{int(ts.start_time)}'
        ts.save(path, group=group)


    def restrict_segments(self, mode=None):

        segments = self.load_segments()
        # Change the segments for training based on testing end time
        

        past_duration = 0
        ret = ligo.segments.segmentlist([])
        for seg in segments:
            # Check if enough data has been generated
            if past_duration - self.start_offset >= self.duration:
                continue
            start, end = seg
            segduration = end - start
            
            # Check if segment fulfills minimum duration requirements
            if self.min_segment_duration is not None and segduration - self.slide_buffer < self.min_segment_duration:
                continue
            
            # Check if segment does not cut into start_offset
            if past_duration + segduration < self.start_offset:
                past_duration += segduration - self.slide_buffer
                continue
            
            # Check if segment is only partially required to cover previous time
            if past_duration < self.start_offset:
                start += self.start_offset - past_duration
                segduration = end - start
                past_duration = self.start_offset
                # Check if remainder of segment fulfills minimum duration requirements
                if self.min_segment_duration is not None and segduration - self.slide_buffer < self.min_segment_duration:
                    continue

            # Check if entire segment is too long to be used completely
            if past_duration + segduration - self.slide_buffer > self.start_offset + self.duration:
                end -= past_duration + segduration - (self.start_offset + self.duration + self.slide_buffer)
                # Add some buffer to end. These segments will be used later to produce testing dataset.
                end += self.testing_buffer
                segduration = end - start
            
            if mode == 'testing':
                self.testing_segments.append([start, end])
            elif mode == 'training':
                if [start, end] in self.testing_segments:
                    continue
                if start == self.testing_segments[-1][0] and end != self.testing_segments[-1][1]:
                    available_buffer = end - self.testing_segments[-1][1]
                    if available_buffer > self.training_buffer:
                        start = self.testing_segments[-1][1] + self.training_buffer
                    else:
                        continue
                    
            ret.append(ligo.segments.segment([start, end]))
            past_duration += segduration - self.slide_buffer
        
        ret.coalesce()
        if past_duration < self.start_offset + self.duration:
            if mode == 'training':
                # All leftover data is returned, warning not required
                self.available_training_duration = past_duration - self.training_buffer
                return ret
            warnings.warn("Not enough segments to generate the entire requested duration.")
        
        # Reset duration to available training duration (only for training)
        # This will be written as an attribute which we can later access
        if mode == 'training':
            self.duration = self.available_training_duration
        
        return ret
    
    
    def get_real_noise(self, mode):
        # Load and restrict segments depending on training or testing mode
        raw_segments = self.load_segments()
        segments = self.restrict_segments()
        
        load_times = {}
        for seg in segments:
            for rawseg in raw_segments:
                if seg in rawseg:
                    load_times[seg] = rawseg
                    break;
            if seg not in load_times:
                raise RuntimeError
                
        if self.store_output[mode] is not None:
            # This seed depends on whether we are training or testing
            # Testing uses the seed given on MLGWSC-1 paper
            # Training using a different seed selected in the data_config file
            rs = np.random.RandomState(self.seed[mode])
        with h5py.File(self.real_noise_path, 'r') as fp:
            for seg in segments:
                start_time = load_times[seg][0]
                segdur = seg[1] - seg[0] - self.slide_buffer
                overlap_seg = OverlapSegment(duration=segdur)
                for det in self.detectors:
                    # Which key does the current segment belong to in real noise file
                    key = f'{det}/{start_time}'
                    # Get the start time attribute from the given segment
                    epoch = fp[key].attrs['start_time']
                    # Get the dt attribute from the given segment
                    dt = fp[key].attrs['delta_t']
                    # Get the required portion of given segment
                    sidx = int((seg[0] - epoch) / dt)
                    eidx = int((seg[1] - epoch) / dt)
                    ts = TimeSeries(fp[key][sidx:eidx],
                                    delta_t=dt,
                                    epoch=float(seg[0]))
                    ts = ts.astype(np.float64) / DYN_RANGE_FAC
                    overlap_seg.add_timeseries((det, ts))
                    
                tmpseed = rs.randint(0, int(1e6))
                data = overlap_seg.get(shift=True, seed=tmpseed)
                for det, ts in zip(overlap_seg.detectors, data):
                    self.store_ts(self.store_output[mode], det, ts)
        
        with h5py.File(self.store_output[mode], 'a') as fp:
            fp.attrs['dataset'] = 4
            fp.attrs['mode'] = mode
            fp.attrs['start_offset'] = self.start_offset
            fp.attrs['duration'] = self.duration
            fp.attrs['seed'] = self.seed[mode]
            fp.attrs['sample_rate'] = 2048.
            fp.attrs['low_frequency_cutoff'] = 15.
            fp.attrs['filter_duration'] = 128.
            fp.attrs['min_segment_duration'] = self.min_segment_duration
            fp.attrs['real_noise_path'] = self.real_noise_path if self.real_noise_path is not None else 'None'
            fp.attrs['slide_buffer'] = self.slide_buffer
            fp.attrs['detectors'] = self.detectors
    
    
    def get_real_noise_generator(self):
        ## After __init__ has completed creating training and testing datasets
        ## Call the Slicer function in the testing module and return this as noise generator
        ## Passing an index will be sufficient to obtain samples from Slicer
        # Calculate the step size required to obtain the number of noise samples
        step_size = (self.available_training_duration - self.sample_length_in_num)/self.num_noises
        # Initialise Slicer object and create noise generator
        # In the following, peak_offset is just an arbitrary value that works. We don't need it here.
        kwargs = dict(infile=self.store_output['training'], 
                      step_size=step_size, 
                      peak_offset=18.0,
                      slice_length=self.sample_length_in_num)
        
        # The slicer object can take an index and return the required training data sample
        slicer = Slicer(**kwargs)
        # Sanity check the length of slicer
        assert len(slicer) >= self.num_noises, "Insufficient number of samples in slicer object!"
        
        return slicer
        
