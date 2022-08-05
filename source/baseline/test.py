#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Jul 22 16:39:59 2022

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

# BUILT-IN
import h5py
import torch
import pycbc
import numpy as np
from tqdm import tqdm

# Torch default datatype
dtype = torch.float32


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
    
    def __init__(self, infile, step_size, peak_offset, slice_length, detectors=None,
                 transforms=None, psds_data=None, data_cfg=None):
        
        # Data params
        self.infile = infile
        
        # Slicing params
        self.step_size = step_size
        self.peak_offset = peak_offset
        self.slice_length = slice_length
        
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
            ds = self.detectors[0][ds_key]
            dt = ds.attrs['delta_t']
            index_step_size = int(self.step_size / dt)
            # Number of steps taken
            nsteps = int((len(ds) - self.slice_length - 512) // index_step_size)
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
        sidx = (index.start - self.n_slices[key]['start']) * index_step_size
        eidx = (index.stop - self.n_slices[key]['start']) * index_step_size + self.slice_length + 512
        rawdata = [det[key][sidx:eidx] for det in self.detectors]
        times = (self.detectors[0][key].attrs['start_time'] + sidx * dt) + index_step_size * dt * np.arange(index.stop - index.start) + self.peak_offset
        
        data = np.zeros((index.stop - index.start, len(rawdata), self.slice_length))
        for detnum, rawdat in enumerate(rawdata):
            for i in range(index.stop - index.start):
                sidx = i * index_step_size
                eidx = sidx + self.slice_length + 512
                ts = pycbc.types.TimeSeries(rawdat[sidx:eidx], delta_t=dt)
                ts = ts.whiten(0.5, 0.25, low_frequency_cutoff=18.)
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


class TorchSlicer(Slicer, torch.utils.data.Dataset):
    
    def __init__(self, *args, **kwargs):
        torch.utils.data.Dataset.__init__(self)
        Slicer.__init__(self, *args, **kwargs)
        self.transforms = kwargs['transforms']
        self.psds_data = kwargs['psds_data']
        self.data_cfg = kwargs['data_cfg']

    def _transforms_(self, noisy_sample):
        # Apply transforms to signal and target (if any)
        print(self.transforms)
        print(self.psds_data)
        if self.transforms:
            sample = self.transforms(noisy_sample, self.psds_data, self.data_cfg)
        else:
            sample = noisy_sample
            raise ValueError('Transforms was not invoked.')
        
        return sample

    def __getitem__(self, index):
        next_slice, next_time = Slicer.__getitem__(self, index)
        # Convert all noisy samples using transformations
        print(next_slice)
        next_transformed_slice = self._transforms_(next_slice)
        print(next_transformed_slice)
        raise
        
        return torch.from_numpy(next_transformed_slice), torch.tensor(next_time)


def get_clusters(triggers, cluster_threshold=0.35):
    """ 
    Cluster a set of triggers into candidate detections.
    
    Arguments
    ---------
    triggers : list of triggers
        A list of triggers.  A trigger is a list of length two, where
        the first entry represents the trigger time and the second value
        represents the accompanying output value from the network.
    cluster_threshold : {float, 0.35}
        Cluster triggers together which are no more than this amount of
        time away from the boundaries of the corresponding cluster.
    
    Returns
    cluster_times :
        A numpy array containing the single times associated to each
        cluster.
    cluster_values :
        A numpy array containing the trigger values at the corresponing
        cluster_times.
    cluster_timevars :
        The timing certainty for each cluster. Injections must be within
        the given value for the cluster to be counted as true positive.
        
    """
    
    clusters = []
    for trigger in triggers:
        new_trigger_time = trigger[0]
        if len(clusters)==0:
            start_new_cluster = True
        else:
            last_cluster = clusters[-1]
            last_trigger_time = last_cluster[-1][0]
            start_new_cluster = (new_trigger_time - last_trigger_time)>cluster_threshold
        if start_new_cluster:
            clusters.append([trigger])
        else:
            last_cluster.append(trigger)

    print("Clustering has resulted in {} independent triggers. Centering triggers at their maxima.".format(len(clusters)))

    cluster_times = []
    cluster_values = []
    cluster_timevars = []

    # Determine maxima of clusters and the corresponding times and append them to the cluster_* lists
    for cluster in clusters:
        times = [trig[0] for trig in cluster]
        values = np.array([trig[1] for trig in cluster])
        max_index = np.argmax(values)
        cluster_times.append(times[max_index])
        cluster_values.append(values[max_index])
        cluster_timevars.append(0.2)

    cluster_times = np.array(cluster_times)
    cluster_values = np.array(cluster_values)
    cluster_timevars = np.array(cluster_timevars)

    return cluster_times, cluster_values, cluster_timevars


def get_triggers(Network, inputfile, step_size, trigger_threshold, 
                 slice_length, peak_offset,
                 data_cfg, transforms, psds_data,
                 device, verbose):
    """
    Use a network to generate a list of triggers, where the network
    outputs a value above a given threshold.
    
    Arguments
    ---------
    Network : network as returned by get_network
        The network to use during the evaluation.
    inputfile : str
        Path to the input data file.
    step_size : {float, 0.1}
        The step size (in seconds) to use for slicing the data.
    trigger_threshold : {float, 0.2}
        The value to use as a threshold on the network output to create
        triggers.
    device : {str, `cpu`}
        The device on which the calculations are carried out.
    verbose : {bool, False}
        Print update messages.
    
    Returns
    -------
    triggers:
        A list of of triggers. A trigger is a list of length two, where
        the first entry represents the trigger time and the second value
        represents the accompanying output value from the network.
        
    """
    
    # Move network into cuda device (if needed)
    Network.to(dtype=dtype, device=device)
    
    triggers = []
    # Read data from testing dataset and slice into overlapping segments
    with h5py.File(inputfile, 'r') as infile:
        slicer = TorchSlicer(infile, step_size=step_size, 
                             peak_offset=peak_offset, slice_length=slice_length,
                             transforms=transforms, psds_data=psds_data,
                             data_cfg=data_cfg)
        
        data_loader = torch.utils.data.DataLoader(slicer, batch_size=100, shuffle=False)
        ### Gradually apply network to all samples and if output exceeds the trigger threshold
        iterable = tqdm(data_loader, desc="Testing Dataset") if verbose else data_loader
        
        for slice_batch, slice_times in iterable:
            
            print(slice_batch)
            
            # Running evaluation procedure on testing dataset
            with torch.cuda.amp.autocast():
                # Gradient evaluation is not required for validation and testing
                # Make sure that we don't do a .backward() function anywhere inside this scope
                with torch.no_grad():
                    testing_output = Network(slice_batch.to(dtype=dtype, device=device))
                    # Get required output values from dictionary
                    pred_prob = testing_output['pred_prob']
                    # Get a boolean vector of output values greater than the trigger threshold
                    trigger_bools = torch.gt(pred_prob, trigger_threshold)
                    
                for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, pred_prob):
                    if trigger_bool.clone().cpu().item():
                        triggers.append([slice_time.clone().cpu().item(), output_value.clone().cpu().item()])
        
        print("A total of {} slices have exceeded the threshold of {}".format(len(triggers), trigger_threshold))
    
    return triggers


def run_test(Network, testfile, evalfile, psds_data, transforms, data_cfg,
             step_size=0.1, slice_length=51200, trigger_threshold=0.2, cluster_threshold=0.35, 
             peak_offset=18.1, device='cpu', verbose=False):
    """
    Run the inference module
    
    Arguments
    ---------
    Network : {ModelClass}
        Network defined in train.py with weights.pt applied
    testfile : {str}
        Input test dataset to check for triggers
    evalfile : (str)
        Output file containing tc, stat and var in HDF5 format. (Can be used alongside evaluate.py)
    step_size : {float}
        Step size (in seconds) used in Slicer class for testing dataset overlapped slice (approx value)
    slice_length : {int}
        Number of samples taken from testing dataset for one slice
    trigger_threshold : {float}
        The value to use as a threshold on the network output to create triggers.
    cluster_threshold : {float}
        Cluster triggers together which are no more than this amount of
        time away from the boundaries of the corresponding cluster
    peak_offset : {float, 18.1}
        The time (in seconds) from the start of each window where the
        peak is expected to be on average
    device : {str}
        Device to place data and network in when running inference module
    verbose : {bool}
        Toggle verbosity
    
    Returns
    -------
    None
    
    """
    
    # Run inference and get triggers from the testing dataset
    triggers = get_triggers(Network,
                            testfile,
                            step_size=step_size,
                            trigger_threshold=trigger_threshold,
                            peak_offset=peak_offset,
                            slice_length=slice_length,
                            data_cfg=data_cfg,
                            transforms=transforms,
                            psds_data=psds_data,
                            device=device,
                            verbose=verbose)
    
    # Cluster the triggers and obtain {tc, ranking statistic, variance on tc} as output
    time, stat, var = get_clusters(triggers, cluster_threshold)
    
    # Write the required output into HDF5 format file
    with h5py.File(evalfile, 'w') as outfile:
        # Save clustered values to the output file and close it
        print("Saving clustered triggers into {}".format(evalfile))
    
        outfile.create_dataset('time', data=time)
        outfile.create_dataset('stat', data=stat)
        outfile.create_dataset('var', data=var)
    
        print("Triggers saved in HDF5 format for evaluation")

