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
import os
import glob
import h5py
import torch
import sklearn
import argparse
import numpy as np
from tqdm import tqdm

# PyCBC
import pycbc
from pycbc.types import FrequencySeries

# LOCAL
from evaluator import main as evaluator
from data.prepare_data import DataModule as dat
from data.multirate_sampling import get_sampling_rate_bins_type1, get_sampling_rate_bins_type2

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
        self.data_cfg = data_cfg
        
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
            nsteps = int((len(ds) - self.slice_length - (self.data_cfg.whiten_padding * self.data_cfg.sample_rate)) // index_step_size)
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
        eidx = (index.stop - self.n_slices[key]['start']) * index_step_size + self.slice_length + int(self.data_cfg.whiten_padding * self.data_cfg.sample_rate)
        # Slice raw data using above indices
        if not isinstance(sidx, int) or not isinstance(eidx, int):
            sidx = int(sidx)
            eidx = int(eidx)
        rawdata = [det[key][sidx:eidx] for det in self.detectors]
        # Get times offset by average peak 'tc' value
        times = (self.detectors[0][key].attrs['start_time'] + sidx * dt) + index_step_size * dt * np.arange(index.stop - index.start) + self.peak_offset
        
        # Get segment data
        data = np.zeros((index.stop - index.start, len(rawdata), self.slice_length+int(self.data_cfg.whiten_padding * self.data_cfg.sample_rate)))
        for detnum, rawdat in enumerate(rawdata):
            for i in range(index.stop - index.start):
                sidx = i * index_step_size
                eidx = sidx + self.slice_length + int(self.data_cfg.whiten_padding * self.data_cfg.sample_rate)
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


class TorchSlicer(Slicer, torch.utils.data.Dataset):
    
    def __init__(self, *args, **kwargs):
        torch.utils.data.Dataset.__init__(self)
        Slicer.__init__(self, *args, **kwargs)
        self.transforms = kwargs['transforms']
        self.psds_data = kwargs['psds_data']
        self.data_cfg = kwargs['data_cfg']

    def __getitem__(self, index):
        next_slice, next_time = Slicer.__getitem__(self, index)
        # Convert all noisy samples using transformations
        exp_length = self.data_cfg.sample_length_in_num
        if len(next_slice[0]) != exp_length or len(next_slice[1]) != exp_length:
            raise ValueError('Length error in next_slice. Expected = {}, observed = {}'.format(self.data_cfg.sample_length_in_num, len(next_slice[0])))

        special = {}
        special['data_cfg'] = self.data_cfg
        special['psds'] = self.psds_data
        sample_transforms = self.transforms(next_slice, special, key='stage1')
        sample_transforms = self.transforms(sample_transforms['sample'], special, key='stage2')
        sample = sample_transforms['sample'][:]
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
        cluster_timevars.append(0.3)

    cluster_times = np.array(cluster_times)
    cluster_values = np.array(cluster_values)
    cluster_timevars = np.array(cluster_timevars)
    
    """
    ## Experimental clustering algorithm
    print("WARNING: Using experimental clustering algorithm in test.py")
    
    # dbscan clustering
    from sklearn.cluster import DBSCAN
    # define the model
    model = DBSCAN(eps=1.438, min_samples=3)
    # fit model and predict clusters
    triggers = np.asarray(triggers)
    yhat = model.fit_predict(np.column_stack((triggers[:,0], triggers[:,1])))
    # retrieve unique clusters
    clusters = unique(yhat)
    print(clusters)
    raise
    """

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
        
        data_loader = torch.utils.data.DataLoader(slicer, batch_size=64, shuffle=False, 
                                                  num_workers=48, pin_memory=cfg.pin_memory, 
                                                  prefetch_factor=100, 
                                                  persistent_workers=cfg.persistent_workers)

        ### Gradually apply network to all samples and if output exceeds the trigger threshold
        iterable = tqdm(data_loader, desc="Testing Dataset") if verbose else data_loader
        max_trigger = torch.tensor(-999)
        
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

                max_trigger = torch.max(max_trigger, torch.max(raw_values))
                iterable.set_description("Max Trigger = {}".format(max_trigger.cpu().detach().item()))
                for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, raw_values):
                    if trigger_bool.clone().cpu().item():
                        triggers.append([slice_time.clone().cpu().item(), output_value.clone().cpu().item()])
                
        
        print("A total of {} slices have exceeded the threshold of {}".format(len(triggers), trigger_threshold))
        _triggers = np.array(triggers)
        if len(_triggers) == 0:
            raise ValueError("No triggers found when searching for events!")
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
    try:
        if data_cfg.srbins_type == 1:
            data_cfg.dbins = get_sampling_rate_bins_type1(data_cfg)
        elif data_cfg.srbins_type == 2:
            data_cfg.dbins = get_sampling_rate_bins_type2(data_cfg)
    except:
        data_cfg.dbins = get_sampling_rate_bins_type1(data_cfg)
    
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
    parser.add_argument("--no-test-background", action='store_true',
                        help="Option to test background file of testing dataset")
    parser.add_argument("--no-test-foreground", action='store_true',
                        help="Option to test foreground file of testing dataset")
    
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
    # Sanity Check - check for early testing (BEST dir does not exist yet)
    if not os.path.exists(check_dir):
        check_dir = cfg.export_dir
    
    # Set the optimal weights to network
    weights_path = os.path.join(check_dir, cfg.weights_path)
    Network = cfg.model(**cfg.model_params)
    ## Error (unsolved): CUDA out of memory when loading weights 
    ## Work-around: mapping weights to CPU before loading into GPU
    # Refer: https://discuss.pytorch.org/t/cuda-error-out-of-memory-when-load-models/38011/3
    checkpoint = torch.load(weights_path, map_location='cpu')
    Network.load_state_dict(checkpoint['model_state_dict'])

    # Try to use multiple GPUs using DataParallel
    # Network = torch.nn.DataParallel(Network)

    ### WARNING ###
    # Causes a lot of trouble if not included before testing phase
    # Weights are essentially allowed to change during the testing phase
    # Since there are more noise samples than signals, this will skew the results significantly
    Network.eval()

    if not opts.no_test_background:
        testfile = os.path.join(cfg.testing_dir, cfg.test_background_dataset)
        evalfile = os.path.join(cfg.testing_dir, cfg.test_background_output)
        print('\nInitiating the testing module for background data')
        run_test(Network, testfile, evalfile, transforms, cfg, data_cfg,
                step_size=cfg.step_size, slice_length=int(data_cfg.signal_length*data_cfg.sample_rate),
                trigger_threshold=cfg.trigger_threshold, cluster_threshold=cfg.cluster_threshold, 
                batch_size = cfg.batch_size, device=cfg.testing_device, verbose=cfg.verbose)
    
    if not opts.no_test_foreground:
        testfile = os.path.join(cfg.testing_dir, cfg.test_foreground_dataset)
        evalfile = os.path.join(cfg.testing_dir, cfg.test_foreground_output)
        print('\nInitiating the testing module for foreground data')
        run_test(Network, testfile, evalfile, transforms, cfg, data_cfg,
                step_size=cfg.step_size, slice_length=int(data_cfg.signal_length*data_cfg.sample_rate),
                trigger_threshold=cfg.trigger_threshold, cluster_threshold=cfg.cluster_threshold, 
                batch_size = cfg.batch_size, device=cfg.testing_device, verbose=cfg.verbose)
    
    if opts.no_test_background and opts.no_test_foreground:
        print('WARNING: Choosing to not test foreground or background file')
        print('Assuming that testing directory contains previous testing outputs')
    
    # Run the evaluator for the testing phase and add required files to TESTING dir in export_dir
    output_testing_dir = os.path.join(cfg.export_dir, 'TESTING')
    raw_args =  ['--injection-file', os.path.join(cfg.testing_dir, cfg.injection_file)]
    raw_args += ['--foreground-events', os.path.join(cfg.testing_dir, cfg.test_foreground_output)]
    #raw_args += ['--foreground-events', "/home/nnarenraju/Research/ORChiD/gw-detection-deep-learning/results_bad/fg.hdf"]
    raw_args += ['--foreground-files', os.path.join(cfg.testing_dir, cfg.test_foreground_dataset)]
    raw_args += ['--background-events', os.path.join(cfg.testing_dir, cfg.test_background_output)]
    #raw_args += ['--background-events', "/home/nnarenraju/Research/ORChiD/gw-detection-deep-learning/results_bad/bg.hdf"]
    out_eval = os.path.join(output_testing_dir, cfg.evaluation_output)
    raw_args += ['--output-file', out_eval]
    raw_args += ['--output-dir', output_testing_dir]
    raw_args += ['--verbose']
    
    # Running the evaluator to obtain output triggers (with clustering)
    evaluator(raw_args, cfg_far_scaling_factor=float(cfg.far_scaling_factor), dataset=data_cfg.dataset)
