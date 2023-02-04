#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Wed Jan 25 15:53:13 2023

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

# Modules
import pycbc
import numpy as np


class Slicer(object):
    """
    Class that is used to slice and iterate over a single input data
    file.
    
    Arguments
    ---------
    infile : open file object
        The open HDF5 file from which the data should be read.
    step_size : {float, 0.1}
        The step size (in seconds) for slicing the data.
    peak_offset : {float, 18.1}
        The time (in seconds) from the start of each window where the
        peak is expected to be on average.
    slice_length : {int, 2048}
        The length of the output slice in samples.
    detectors : {None or list of datasets}
        The datasets that should be read from the infile. If set to None
        all datasets listed in the attribute 'detectors' will be read.
        
    """
    
    def __init__(self, infile, step_size, peak_offset, whiten_padding, slice_length, detectors=None,
                 transforms=None, psds_data=None, data_cfg=None):
        
        # Data params
        self.infile = infile
        
        # Slicing params
        self.step_size = step_size
        self.peak_offset = peak_offset
        self.slice_length = slice_length
        self.whiten_padding = whiten_padding
        self.sample_rate = 2048. # Hz
        
        # Detectors
        self.detectors = detectors
        if self.detectors is None:
            self.detectors = [self.infile[key] for key in list(self.infile.attrs['detectors'])]
        self.keys = sorted(list(self.detectors[0].keys()), key=lambda inp: int(inp))
        
        # MISC
        self.determine_nslices()
    
    
    def determine_nslices(self):
        self.n_slices = {}
        start = 0
        # Iterating over the detector keys
        for ds_key in self.keys:
            ds = self.detectors[0][ds_key] # eg. 32000 seconds
            dt = ds.attrs['delta_t'] # eg. 1./2048.
            index_step_size = int(self.step_size / dt) # eg. int(0.1 * 2048.) = 204
            # Number of steps taken -> eg. (32000 * 2048 - 40960 - 10240) // 204 = 321003 segments
            nsteps = int((len(ds) - self.slice_length - (self.whiten_padding * self.sample_rate)) // index_step_size)
            # Dictionary containing params of how to slice large segment
            # We can slice the data when needed using these params
            self.n_slices[ds_key] = {'start': start,
                                     'stop': start + nsteps,
                                     'len': nsteps}
            start += nsteps
    
    
    def __len__(self):
        # Length of the number of slices
        return sum([val['len'] for val in self.n_slices.values()])
    
    
    def _generate_access_indices(self, index):
        assert index.step is None or index.step == 1, 'Slice with step is not supported'
        ret = {}
        start = index.start
        stop = index.stop
        for key in self.keys:
            cstart = self.n_slices[key]['start']
            cstop = self.n_slices[key]['stop']
            if cstart <= start and start < cstop:
                ret[key] = slice(start, min(stop, cstop))
                start = ret[key].stop
        return ret
    
    
    def generate_data(self, key, index):
        # Ideally set dt = self.detectors[0][key].attrs['delta_t']
        # Due to numerical limitations this may be off by a single sample
        dt = 1. / 2048. # This definition limits the scope of this object
        index_step_size = int(self.step_size / dt)
        # Create start and end indices from slice dict
        sidx = (index.start - self.n_slices[key]['start']) * index_step_size
        eidx = (index.stop - self.n_slices[key]['start']) * index_step_size + self.slice_length + int(self.whiten_padding * self.sample_rate)
        # Slice raw data using above indices
        if not isinstance(sidx, int) or not isinstance(eidx, int):
            sidx = int(sidx)
            eidx = int(eidx)
        
        rawdata = [det[key][sidx:eidx] for det in self.detectors]
        # Get times offset by average peak 'tc' value
        times = (self.detectors[0][key].attrs['start_time'] + sidx * dt) + index_step_size * dt * np.arange(index.stop - index.start) + self.peak_offset
        
        # Get segment data
        data = np.zeros((index.stop - index.start, len(rawdata), self.slice_length+int(self.whiten_padding * self.sample_rate)))
        for detnum, rawdat in enumerate(rawdata):
            for i in range(index.stop - index.start):
                sidx = i * index_step_size
                eidx = sidx + self.slice_length + int(self.whiten_padding * self.sample_rate)
                ts = pycbc.types.TimeSeries(rawdat[sidx:eidx], delta_t=dt)
                data[i, detnum, :] = ts.numpy()
        
        return data, times
    
    
    def __getitem__(self, index):
        is_single = False
        if isinstance(index, int):
            is_single = True
            if index < 0:
                index = len(self) + index
            index = slice(index, index+1)
        
        access_slices = self._generate_access_indices(index)
        
        data = []
        times = []
        for key, idxs in access_slices.items():
            dat, t = self.generate_data(key, idxs)
            data.append(dat)
            times.append(t)
            
        data = np.concatenate(data)
        times = np.concatenate(times)
        
        if is_single:
            return data[0], times[0]
        else:
            return data, times
