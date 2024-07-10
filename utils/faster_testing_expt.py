#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Nov 10 11:08:36 2022

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
import os
import glob
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm

# PyCBC
import pycbc
from pycbc.types import FrequencySeries

# LOCAL
from evaluate import main as evaluator
from data.prepare_data import DataModule as dat
from data.multirate_sampling import get_sampling_rate_bins

# Torch default datatype
dtype = torch.float32


class Slicer(object):
    """
    Class that is used to slice and iterate over a single input data
    file.
    
    Arguments
    ---------
    testdata : open file object
        The open HDF5 file from which the data should be read.
    step_size : {float, 0.1}
        The step size (in seconds) for slicing the data.
    peak_offset : {float, 18.1}
        The time (in seconds) from the start of each window where the
        peak is expected to be on average.
    slice_length : {int}
        The length of the output slice in samples.
    data_cfg : {None or dict of data config options}
        Configuration file used in data_config.py
        
    """
    
    def __init__(self, testdata, step_size, peak_offset, slice_length, data_cfg=None):
        
        ## Wording legend: (Dataset > Block > Segment > Sample > Data Point)
        
        # Data params
        self.testdata = testdata
        
        # Slicing params
        self.step_size = step_size
        self.peak_offset = peak_offset
        self.sample_length = slice_length
        self.data_cfg = data_cfg
        
        # Detector data
        self.detectors = [self.testdata[key] for key in list(self.testdata.attrs['detectors'])]
        
        ## !!! WARNING !!! (Deprecated on November 16th, 2022)
        # Using numpy stride tricks to obtain fast overlapping segments
        # Seen in scipy conference: https://mentat.za.net/numpy/numpy_advanced_slides/
        # Numpy Notes: **This function has to be used with extreme care**
        # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html
        # Using numpy.lib.stride_tricks.sliding_window_view is a safer alternative
        # When using stride lengths other than x.strides, we are relying on continguous memory layout
        self.as_strided = np.lib.stride_tricks.as_strided
        
        # Add whitening padding to the sample length when getting testing data
        self.segment_length = self.sample_length + int(self.data_cfg.whiten_padding * self.data_cfg.sample_rate)
        # Start time for testing dataset
        self.start_times = np.array([])
        # Testing dataset group lengths given in the HDF5 file
        # eg., If we request 64000 seconds of testing data, we obtain 'n' unequal segments
        # that total a length of 64000 seconds. Each segment must at least be 3600 seconds long.
        self.group_lengths = np.array([])
        # Save dataset delta_t (constant for entire dataset)
        self.delta_t = None
        self.sample_rate = None
        # Total number of data points present in the testing dataset
        # Add all segment lengths together
        self.total_data_points = 0
        # Length of the entire segmented testing dataset
        self.testdata_length_in_s = self.get_duration()
        # Length of the segmented testing dataset
        self.num_segments = len(self)
        
        # Local block (present as dataset object, so will not affect RAM too much)
        self.local_blocks = None
        self.local_block_idx = None
    
    def __len__(self):
        # Number of segments determined from step_size and total length in s
        # Determine the number of start times possible for given total length
        # Subtract segment length from total length when determining num start times
        max_accepted_lengths = self.group_lengths - self.segment_length
        # Add 1 to num segments to account for first segment with start time = 0
        num_segments = np.sum((max_accepted_lengths / self.step_size) + 1)
        return num_segments
    
    def get_duration(self, padding_start=30, padding_end=30):
        """
        Determine total duration of the testing data.
        
        Arguments
        ---------
        padding_start : {float, 0}
            The amount of time (in seconds) at the start of each segment
            where no injections are present.
        padding_end : {float, 0}
            The amount of time (in seconds) at the end of each segment
            where no injections are present.
        
        Returns
        -------
        duration:
            A float representing the total duration (in seconds) of all
            foreground files.
        """
        duration = 0
        # Get duration from open testing data file
        det = list(self.testdata.keys())[0]
        
        for key in self.testdata[det].keys():
            # Accessing block within testing dataset
            # Each block is at least 3600 seconds long
            ds = self.testdata[f'{det}/{key}']
            start = ds.attrs['start_time']
            # dataset length
            dlen = len(ds)
            # Setting dataset params
            if self.delta_t == None:
                self.delta_t = ds.attrs['delta_t']
                self.sample_rate = 1./self.delta_t
            np.append(self.start_times, start)
            np.append(self.group_lengths, dlen)
            self.total_data_points += dlen
            
            end = start + dlen * ds.attrs['delta_t']
            duration += end - start
            start += padding_start
            end -= padding_end

        return duration
    
    def get_segment_params(self, idx):
        ## Wording legend: (Dataset > Block > Segment > Sample > Data Point)
        # Get the block ID given a global idx
        # Following line gives the index of the first occurence of given condition
        # This results in the index of the block within with idx resides
        block_idx = np.nonzero(idx < np.cumsum(self.group_lengths))[0][0]
        # Get start and end times for segment (times depends on block under question)
        start_time = self.start_times[block_idx] + (self.step_size * (idx-self.group_lengths[block_idx-1]))
        # Uncomment end time if needed
        # end_time = start_time + self.segment_length
        # Get start and end global idx for each segment (local idx within the block)
        local_start_idx = (self.step_size * (idx-self.group_lengths[block_idx-1])) * self.sample_rate
        local_end_idx = local_start_idx + self.segment_length * self.sample_rate
        # Get expected time of coalescence for each segment based on peak offset
        ## Peak offset is the mean value is seconds where the GW is expected to be
        ## given a sample length in s. If tc prior is 18.0 and 18.2, offset is 18.1 s
        expected_tc = start_time + self.peak_offset
        
        # Create a numpy table with required dims for each idx
        return (block_idx, local_start_idx, local_end_idx, expected_tc, )
        
    def get_segment_data(self, idx):
        # Use the idx to get segment params, then obtain the segment data
        block_idx, local_start_idx, local_end_idx, expected_tc = self.get_segment_params(idx)
        # Obtain the lazy segment required for the given idx
        if self.local_block_idx == None or self.local_block_idx != block_idx:
            self.local_block_idx = block_idx
            self.local_blocks = [det[str(self.start_times[self.local_block_idx])] for det in self.detectors]
        
        # Get the required segment from local block
        current_segment = [np.array(block[local_start_idx:local_end_idx]) for block in self.local_blocks]
        # Return the current segment and the expected tc for the current segment
        return (current_segment, expected_tc, )


class TorchSlicer(Slicer, torch.utils.data.Dataset):
    
    def __init__(self, *args, **kwargs):
        torch.utils.data.Dataset.__init__(self)
        Slicer.__init__(self, *args, **kwargs)
        self.transforms = kwargs['transforms']
        self.psds_data = kwargs['psds_data']
        self.data_cfg = kwargs['data_cfg']

    def __getitem__(self, index):
        next_slice, next_time = Slicer.get_segment_data(self, index)
        # Convert all noisy samples using transformations
        exp_length = self.data_cfg.sample_length_in_num
        if len(next_slice[0]) != exp_length or len(next_slice[1]) != exp_length:
            raise ValueError('Length error in next_slice. Expected = {}, observed = {}'.format(self.data_cfg.sample_length_in_num, len(next_slice[0])))
            
        sample = self.transforms(next_slice, self.psds_data, self.data_cfg)
        return torch.from_numpy(sample), torch.tensor(next_time)


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
                 slice_length, peak_offset, cfg,
                 data_cfg, transforms, psds_data, batch_size,
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
        
        data_loader = torch.utils.data.DataLoader(slicer, batch_size=batch_size, shuffle=False, 
                                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, 
                                                  prefetch_factor=cfg.prefetch_factor, 
                                                  persistent_workers=cfg.persistent_workers)
        
        ### Gradually apply network to all samples and if output exceeds the trigger threshold
        iterable = tqdm(data_loader, desc="Testing Dataset") if verbose else data_loader
        
        for slice_batch, slice_times in iterable:
            
            # Running evaluation procedure on testing dataset
            with torch.cuda.amp.autocast():
                # Gradient evaluation is not required for validation and testing
                # Make sure that we don't do a .backward() function anywhere inside this scope
                with torch.no_grad():
                    testing_output = Network(slice_batch.to(dtype=dtype, device=device))
                    # Get required output values from dictionary
                    # Use raw values here as sigmoid tends to lose dynamic range
                    raw_values = testing_output['raw']
                    # Get a boolean vector of output values greater than the trigger threshold
                    trigger_bools = torch.gt(raw_values, trigger_threshold)
                    
                for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, raw_values):
                    if trigger_bool.clone().cpu().item():
                        triggers.append([slice_time.clone().cpu().item(), output_value.clone().cpu().item()])
        
        print("A total of {} slices have exceeded the threshold of {}".format(len(triggers), trigger_threshold))
        _triggers = np.array(triggers)
        print('raw values of output: max = {}, min = {}'.format(max(_triggers[:,1]), min(_triggers[:,1])))
    
    return triggers


def get_psd_data(data_cfg):
    """ PSD Handling (used in whitening) """
    # Store the PSD files here in RAM. This reduces the overhead when whitening.
    # Read all psds in the data_dir and store then as FrequencySeries
    PSDs = {}
    data_loc = os.path.join(data_cfg.parent_dir, data_cfg.data_dir)
    psd_files = glob.glob(os.path.join(data_loc, "psds/*"))
    for psd_file in psd_files:
        with h5py.File(psd_file, 'r') as fp:
            data = np.array(fp['data'])
            delta_f = fp.attrs['delta_f']
            name = fp.attrs['name']
            
        psd_data = FrequencySeries(data, delta_f=delta_f)
        # Store PSD data into lookup dict
        PSDs[name] = psd_data
    
    if data_cfg.dataset == 1:
        psds_data = [PSDs['aLIGOZeroDetHighPower']]*2
    else:
        psds_data = [PSDs['median_det1'], PSDs['median_det2']]
    
    return psds_data


def run_test(Network, testfile, evalfile, transforms, cfg, data_cfg,
             step_size=0.1, slice_length=40960, trigger_threshold=0.2, cluster_threshold=0.35, 
             batch_size=100, device='cpu', verbose=False):
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
    
    if not os.path.exists(cfg.export_dir):
        raise IOError('Export directory does not exist. Cannot write testing output files.')
    
    # Make a testing directory within the export_dir
    testing_dir = os.path.join(cfg.export_dir, 'TESTING')
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir, exist_ok=False)
        
    """ Multi-rate Sampling """
    # Get the sampling rates and their bins idx
    data_cfg.dbins = get_sampling_rate_bins(data_cfg)
    
    # Get the psd data for transformation methods
    psds_data = get_psd_data(data_cfg)
    # Average value in seconds where signal peak would be present
    peak_offset = (data_cfg.tc_inject_lower + data_cfg.tc_inject_upper) / 2.0
    # Account for the loss of corrupted data during whitening process in the peak offset value
    peak_offset += data_cfg.whiten_padding / 2.0
    
    # Run inference and get triggers from the testing dataset
    triggers = get_triggers(Network,
                            testfile,
                            step_size=step_size,
                            trigger_threshold=trigger_threshold,
                            peak_offset=peak_offset,
                            slice_length=slice_length,
                            cfg=cfg,
                            data_cfg=data_cfg,
                            transforms=transforms,
                            psds_data=psds_data,
                            batch_size=batch_size,
                            device=device,
                            verbose=verbose)
    
    print('Clustering the triggers to obtain events')
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
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default='Baseline',
                        help="Uses the pipeline architecture as described in configs.py")
    parser.add_argument("--data-config", type=str, default='Default',
                        help="Creates or uses a particular dataset as provided in data_configs.py")
    
    opts = parser.parse_args()
    
    """ Prepare Data """
    # Get model configuration
    cfg = dat.configure_pipeline(opts)
    # Get data creation/usage configuration
    data_cfg = dat.configure_dataset(opts)
    transforms = cfg.transforms['test']
    
    # Initialise Network with best weight found in export dir
    if not os.path.exists(cfg.export_dir):
        raise IOError('Export directory does not exist. Cannot write testing output files.')
    
    print('\nApplying best weights from the {} run to Network'.format(cfg.export_dir))
    check_dir = os.path.join(cfg.export_dir, 'BEST')
    # Sanity Check - check for early testing (BEST dir will not exist)
    if not os.path.exists(check_dir):
        check_dir = cfg.export_dir
    
    # Set the optimal weights to network
    weights_path = os.path.join(check_dir, cfg.weights_path)
    Network = cfg.model(**cfg.model_params)
    Network.load_state_dict(torch.load(weights_path))
    
    """ WARNING """
    # Causes a lot of trouble if not included before testing phase
    # Weights are essentially allowed to change during the testing phase
    # Since there are more noise samples than signals, this will skew the results significantly
    Network.eval()
    
    testfile = os.path.join(cfg.testing_dir, cfg.test_foreground_dataset)
    evalfile = os.path.join(cfg.testing_dir, cfg.test_foreground_output)
    print('\nInitiating the testing module for foreground data')
    run_test(Network, testfile, evalfile, transforms, cfg, data_cfg,
             step_size=cfg.step_size, slice_length=int(data_cfg.signal_length*data_cfg.sample_rate),
             trigger_threshold=cfg.trigger_threshold, cluster_threshold=cfg.cluster_threshold, 
             batch_size = cfg.batch_size, device=cfg.testing_device, verbose=cfg.verbose)
    
    testfile = os.path.join(cfg.testing_dir, cfg.test_background_dataset)
    evalfile = os.path.join(cfg.testing_dir, cfg.test_background_output)
    print('\nInitiating the testing module for background data')
    run_test(Network, testfile, evalfile, transforms, cfg, data_cfg,
             step_size=cfg.step_size, slice_length=int(data_cfg.signal_length*data_cfg.sample_rate),
             trigger_threshold=cfg.trigger_threshold, cluster_threshold=cfg.cluster_threshold, 
             batch_size = cfg.batch_size, device=cfg.testing_device, verbose=cfg.verbose)
    
    # Run the evaluator for the testing phase and add required files to TESTING dir in export_dir
    output_testing_dir = os.path.join(cfg.export_dir, 'TESTING')
    raw_args =  ['--injection-file', os.path.join(cfg.testing_dir, cfg.injection_file)]
    raw_args += ['--foreground-events', os.path.join(cfg.testing_dir, cfg.test_foreground_output)]
    raw_args += ['--foreground-files', os.path.join(cfg.testing_dir, cfg.test_foreground_dataset)]
    raw_args += ['--background-events', os.path.join(cfg.testing_dir, cfg.test_background_output)]
    out_eval = os.path.join(output_testing_dir, cfg.evaluation_output)
    raw_args += ['--output-file', out_eval]
    raw_args += ['--output-dir', output_testing_dir]
    raw_args += ['--verbose']
    
    # Running the evaluator to obtain output triggers (with clustering)
    evaluator(raw_args, far_scaling_factor=float(cfg.far_scaling_factor), dataset=data_cfg.dataset)
