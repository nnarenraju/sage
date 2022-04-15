#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Apr 12 14:38:08 2022

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
import h5py
import glob
import json
import math
import time
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# LOCAL
from data.multirate_sampling import get_sampling_rate_bins

# PyCBC
from pycbc.types import load_frequencyseries, FrequencySeries


class MP_Trainable:
    
    def __init__(self, cfg, data_cfg, fold_lim, nbatch):
        self.nbatch = nbatch
        self.nfold = -1
        self.fold_lim = fold_lim
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.data_paths = None
        self.targets = None
        self.transforms = self.cfg.transforms['train']
        self.target_transforms = None
        self.export_dir = self.cfg.export_dir
        self.data_loc = os.path.join(self.data_cfg.parent_dir, self.data_cfg.data_dir)
        self.store_dir = ""
        
        # Iterable used within MP methods
        self.iterable = None
        
        # Make a trainable dataset dir if not exist
        self._make_trainable_dir()
        
        # Store the PSD files here in RAM. This reduces the overhead when whitening.
        # Read all psds in the data_dir and store then as FrequencySeries
        self.PSDs = {}
        psd_files = glob.glob(os.path.join(self.data_loc, "psds/*"))
        for psd_file in psd_files:
            try:
                # This should load the PSD as a FrequencySeries object with delta_f assigned
                psd_data = load_frequencyseries(psd_file)
            except:
                data = pd.read_hdf(psd_file, 'data').to_numpy().flatten()
                psd_data = FrequencySeries(data, delta_f=self.data_cfg.delta_f)
            # Store PSD data into lookup dict
            self.PSDs[psd_file] = psd_data
        
        # Multi-rate sampling
        # Get the sampling rates and their bins idx
        self.data_cfg.dbins = get_sampling_rate_bins(self.data_cfg)
        
        # Save parameters
        self.save_samples = []
        self.save_targets = []
        self.save_abspaths = []
        # TODO: These two can be removed if dataset is all made at 1MPc
        # These are saved for distance augmentation
        self.save_distances = []
        self.save_mchirps = []
        self.save_norm_tcs = []


    def _read_(self, data_path):
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
    
    
    def _make_trainable_dir(self):
        # Path to store trainable dataset
        parent_dir = os.path.join(self.data_cfg.parent_dir, self.data_cfg.data_dir)
        self.store_dir = os.path.join(parent_dir, "trainable_batched_dataset")
        
        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir, exist_ok=False)            
            
    
    def _save_batch_to_hdf(self, batch_samples, store_path):
        """ Store Trainable Training/Validation Data """
        # Saving target and path (this will be a list of np arrays of labels from each batch)
        # HDF5 was measured to have the fastest IO (r->46ms, w->172ms)
        # NPY read/write was not tested. Github says HDF5 is faster.
        with h5py.File(store_path, 'a') as fp:
            # create a dataset for batch save
            fp.create_dataset("data", shape=batch_samples.shape, dtype=np.float32, 
                              data=batch_samples,
                              compression='gzip',
                              compression_opts=9, 
                              shuffle=True)
    
    
    def worker(self, idx, queue):
        
        data_path = self.data_paths[idx]
        
        """ Read the sample """
        # check whether the sample is noise/signal for adding random noise realisation
        data_params = self._read_(data_path)
        data_type = data_params[0]
        
        # Get sample params
        if data_type == 'signal':
            _, signals, label_saved, normalised_tc, mchirp, distance = data_params
            # This should be pure signal
            raw_sample = signals
        elif data_type == 'noise':
            _, noise, label_saved, psd_1, psd_2, normalised_tc = data_params
            distance = -1
            mchirp = -1
            # Concat psds
            psds = [psd_1, psd_2]
            # This should be pure noise
            raw_sample = noise
            
        """ Get PSDs """
        if self.data_cfg.dataset == 1:
            psds = [os.path.abspath(os.path.join(self.data_loc, "psds/psd-aLIGOZeroDetHighPower.hdf"))]*2
        else:
            raise NotImplementedError("PSDs for D2-3 not implemented yet")
        
        """ Target """
        # Save label_saved into the target variable
        target = label_saved.astype(np.float64)
        # Concatenating the normalised_tc within the target variable
        # target = np.append(target, normalised_tc)
        ## Target should look like (1., 0., 0.567) for signal
        ## Target should look like (0., 1., -1.0) for noise
        
        """ Transforms """
        # Apply transforms to signal and target (if any)
        if self.transforms:
            psds_data = [self.PSDs[psd_name] for psd_name in psds]
            sample = self.transforms(raw_sample, psds_data, self.data_cfg)
        if self.target_transforms:
            target = self.target_transforms(target)
        
        """ Reduce data size by using float32. Transforms are erroneous on float32 """
        # Convert signal/target to Tensor objects
        sample = sample.astype(np.float32)
        target = target.astype(np.float32)
        
        """ Save all data that needs to be passed """
        data = {}
        data['sample'] = sample
        data['target'] = target.tolist()
        data['norm_tc'] = normalised_tc
        data['mchirp'] = mchirp
        data['distance'] = distance
        data['kill'] = False
        
        # Give all relevant save data to Queue
        queue.put(data)
    
    
    def listener(self, queue):
        """
        Write given sample data to a HDF file
        We also use this to append to priors
        
        ** WARNING!!! **: If files are not being written, there should be an error in here.
        This error will not be raised and has not yet been handled properly.
        
        Vague check for this type of error:
            1. Print the queue.get() within the infinite loop.
            2. If this prints a singular output, then there is something wrong within listener.
            3. Then print a bunch of Lorem Ipsum here and there
            4. Print statements after the error occcurence will not be displayed
            5. Narrow down the error location and shoot your shot! (Sorry :P)
            
        """
        
        # Continuously check for data to the Queue
        
        while True:
            data = queue.get()
            """ The following is run ONLY if we get something from Queue """
            
            # if processes have ended, kill
            if data['kill']:
                break
            
            # Instead of appending data one by one save data within class (in RAM)
            # The batch size was chosen based on RAM capability, so this should be fine
            # The size was chosen based on the size of the transformed dataset, not the raw samples
            self.save_samples.append(data['sample'])
            self.save_targets.append(data['target'])
            self.save_norm_tcs.append(data['norm_tc'])
            self.save_mchirps.append(data['mchirp'])
            self.save_distances.append(data['distance'])
            
            # Save if limit EOI (end-of-iterable)
            if len(self.save_distances) == len(self.iterable):
                # Save path
                store_path = os.path.join(self.store_dir, "trainable_batch_{}.hdf".format(self.nbatch))
                # Pass all samples to save_batch and put into HDF5 file
                self._save_batch_to_hdf(np.array(self.save_samples), store_path)
                self.save_abspaths.append(store_path)
                
                if self.nfold == self.fold_lim:
                    self.make_lookup()
    
    
    def transform_dataset(self):
        """
        Create transformed dataset using the explicit PyCBC method
        This code is much faster to execute and easier to read
        """
        
        # TODO: Handle splitting of transformed dataset.
        # Data doesn't seem to persist in memory in listener.
        # self.save_data is empty once it exits the listener.
        iterables = [self.iterable]
        num_splits = 1
        
        # Must use Manager queue here, or will not work
        manager = mp.Manager()
        queue = manager.Queue()
        
        for niter, iterable in enumerate(iterables):
            
            # Empty jobs every iteration
            jobs = []
            
            # Initialise pool
            pool = mp.Pool(int(8))
            # Put listener to work first (this will wait for data in MP and write to Queue)
            watcher = pool.apply_async(self.listener, (queue,))
            
            for i in iterable:
                job = pool.apply_async(self.worker, (i, queue))
                jobs.append(job)
    
            # Collect results from the workers through the pool result queue
            pbar = tqdm(jobs)
            for job in pbar:
                pbar.set_description("Transforming Dataset (Batch {}/{})".format(niter+1, num_splits))
                job.get()

            # Kill the listener when all jobs are complete
            queue.put({'kill': True})
            pool.close()
            pool.join()
            
            # Sleeping for a bit (helps keep the performance up)
            time.sleep(5)
            
    
    def save_trainable_dataset(self):
        # Iterate through a set of idxs and get data
        self.iterable = range(len(self.targets))
        # Call MP based transform data function
        self.transform_dataset()
        
        
    def make_lookup(self):
        """ Save trainable.json for lookup """
        ## Creating trainable.json similar to training.hdf
        # un-JSONified version (np array is not JSON serializable)
        ids = np.arange(len(self.save_abspaths)).tolist()
        # Shuffling is not required as the DataLoader should have already shuffled it
        # Save the lookup table as a json file (this works better for batch saving)
        lookup = {'ids': ids, 'path': self.save_abspaths, 'target': self.save_targets,
                  'norm_tc': self.save_norm_tcs, 'distance': self.save_distances,
                  'mchirp': self.save_mchirps}
        
        # Save the dataset paths alongside the target and ids as .JSON
        lookup_trainable = os.path.join(self.data_loc, "trainable_{}.json".format(self.nbatch))
        with open(lookup_trainable, 'w') as fp:
            json.dump(lookup, fp)
        # Create a copy of trainable.json to the dataset directory
        shutil.copy(lookup_trainable, self.export_dir)
        print("manual.py: Trainable dataset has been created and stored!")
