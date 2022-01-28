# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Wed Dec 15 17:11:51 2021

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation:

From Pseudo-code:
    1. Creating a dataset using PyCBC using a fixed length of segment
    2. Modify generate_data.py testing set generation code to create training set
    3. Make sure the data is similar to dataset 3/4 in the challenge
    4. Save each segment in the HDF5 format
    5. Obtain the paths to each segment by using efficient IDs
    6. Get the targets/labels for each segment (signal/noise)
    7. Create a large HDF5 file containing the details (ID, path, target)

Steps:
    - Main steps
    1. Call make_segments.py using appropriate opts to create segments.csv (FIN)
    2. Call generate_data.py from ML-MDC1 using segments.csv to make training data (FIN)
    3. Change generate_data to store each segment as a separate HDF5 file (FIN)
    
    - Verification steps
    3. Verify whether all params and segments are as intended (FIN)
    4. Cross verify the prior distribution with the observed distribution (FIN)
    5. Visualise the signals, noise and signal+noise for by-eye verification (FIN)
    6. Check input to antenna pattern functions
    
    - Data save steps
    6. Create a training.hdf5 that handles ids, paths and target of training data (FIN)
    7. Store all the above data into a data-read directory for next step of the algorithm (FIN)
    
"""

# IN-BUILT
import os
import sys
import h5py
import uuid
import shutil
import warnings
import datetime
import numpy as np
import pandas as pd

# LOCAL
from data.make_segments import make_segments as mksegments
# Add MLMDC1 repo from GWastro
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../../external/ml-mock-data-challenge-1')
from generate_data import main as gendata

# Verification
from data.verify_segments import verify as segment_verification
from data.verify_priors import verify as prior_verification
from data.verify_signals import verify as signal_verification


class GenerateData:
    """
    
    """
    
    # Slots magic for parameters input from data_configs.py
    __slots__ = ['dataset', 'parent_dir', 'data_dir', 'seed', 'start_offset', 
                 'dataset_duration', 'verbose', 'force', 'segment_GPS_start_time',
                 'time_window_llimit', 'time_window_ulimit', 'segment_length',
                 'segment_gap', 'ninjections']

    
    def __init__(self, **kwargs):
        ## Get slots magic attributes via input dict (use **kwargs)
        # This should set all slots given above
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Export directory to store training.hdf
        self.export_dir = ""
        
        ## generate_data script parameters
        # Path to store output HDF5 files
        self.output_injection_file = "injections.hdf"
        self.output_foreground_file = "foreground.hdf"
        self.output_background_file = "background.hdf"
        self.output_signal_file = "signal.hdf"
        
        ## make_injections using pycbc_create_injections (uses self.seed)
        # params will be used to call above function via generate_data.py
        # pycbc_create_injections has been modified by nnarenraju (Dec 15th, 2021)
        self.output_segment_file = "segments.csv"

        # Distance b/w two adjacent 'tc'
        self.time_step = self.segment_length
        
        # Other options
        self.sample_rate = 2048.0 # Hz
        self.check_n_signals = 5 # used in verify_signals
        
        # Metadata and identification
        self.unique_dataset_id = uuid.uuid4().hex
        self.creation_time = str(datetime.datetime.now())
        
        ## Storage options
        # Store all required dirs as a dict
        self.dirs = {}
    
    def __str__(self):
        # All general dataset parameters
        out = "Generate Data Class:\n"
        out += f"Dataset type = {self.dataset}\n"
        out += f"Output injection file path = {self.output_injection_file}\n"
        out += f"Output foreground file path = {self.output_foreground_file} for each segment\n"
        out += f"Output background file path = {self.output_background_file} for each segment\n"
        out += f"Initialisation seed = {self.seed}\n"
        out += f"Start Offset = {self.start_offset}\n"
        out += f"Duration of dataset = {self.dataset}\n\n"
        # Segment parameters
        out += "Segment Details:\n"
        out += f"Output segment file = {self.output_segment_file}\n"
        out += f"Segment GPS start time = {self.segment_GPS_start_time}\n"
        out += f"Time step b/w adjacent 'tc' ~ {self.time_step}\n"
        out += f"Time window for tc placement (llimit) = {self.time_window_llimit}\n"
        out += f"Time window for tc placement (ulimit) = {self.time_window_ulimit}\n"
        out += f"Length of each segment = {self.segment_length}\n"
        out += f"Total number of injections present in dataset = {self.ninjections}\n"
        out += f"Gap duration between each segment = {self.segment_gap()}\n\n\n"
        # Identification 
        out += f"Unique dataset ID = {self.unique_dataset_id}\n"
        out += f"Time of dataset creation = {self.creation_time}"
    
    def _get_recursive_abspath_(self, directory):
        # Get abspath of all files within directory
        paths = np.array([])
        for dirpath, _, filenames in os.walk(os.path.abspath(directory)):
            for filename in filenames:
                 paths = np.append(paths, os.path.join(dirpath, filename))
        return paths
            
    def make_segments(self):
        # Make segments.csv to be used by generate_data script
        output_segments_file = os.path.join(self.dirs['parent'], self.output_segment_file)
        # Create an object for Segments class
        args =  (self.segment_GPS_start_time,)
        args += (self.segment_gap,)
        args += (self.segment_length,)
        args += (self.ninjections,)
        args += (output_segments_file,)
        args += (self.force,)
        
        # Make segments.csv (equal length assumed)
        mksegments(*args)

    def call_gendata(self):
        ## Main params to call generate_data script
        # Set paths to absolute location of dataset dir
        inj_path = os.path.join(self.dirs['parent'], self.output_injection_file)
        bg_path = os.path.join(self.dirs['background'], self.output_background_file)
        fg_path = os.path.join(self.dirs['foreground'], self.output_foreground_file)
        seg_path = os.path.join(self.dirs['parent'], self.output_segment_file)
        sig_path = os.path.join(self.dirs['signal'], self.output_signal_file)
        
        # Warning! for dataset 4 large noise data download
        if self.dataset == 4:
            warnings.warn("Dataset type 4 downloads an ~94GB noise file.")
            
        raw_args =  ['--data-set', str(self.dataset)]
        raw_args += ['--output-injection-file', inj_path]
        raw_args += ['--output-foreground-file', fg_path]
        raw_args += ['--output-background-file', bg_path]
        raw_args += ['--output-signal-file', sig_path]
        
        # sanity check for segments.csv
        if os.path.exists(seg_path):
            raw_args += ['--input-segments-file', seg_path]
        else:
            raise IOError("{} does not exist!".format(seg_path))
            
        # Segment parameters
        raw_args += ['--time-step', str(self.time_step)]
        raw_args += ['--time-window-llimit', str(self.time_window_llimit)]
        raw_args += ['--time-window-ulimit', str(self.time_window_ulimit)]
        raw_args += ['--segment-gap', str(self.segment_gap)]
        raw_args += ['--unique-dataset-id', str(self.unique_dataset_id)]
        
        # generic generate_data script params
        raw_args += ['--seed', str(self.seed)]
        raw_args += ['--start-offset', str(self.start_offset)]
        raw_args += ['--duration', str(self.dataset_duration)]
        
        # MISC params
        if self.verbose:
            raw_args += ['--verbose']
        if self.force:
            raw_args += ['--force']
        
        gendata(raw_args)
    
    def make_data_dir(self):
        # Make directory structure for data storage
        self.dirs['parent'] = os.path.join(self.parent_dir, self.data_dir)
        self.dirs['signal'] = os.path.join(self.dirs['parent'], "signals")
        self.dirs['background'] = os.path.join(self.dirs['parent'], "background")
        self.dirs['foreground'] = os.path.join(self.dirs['parent'], "foreground")
        # Parent, Signals, Background and Foreground
        for directory in list(self.dirs.values()):
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=False)
            else:
                # If True, it already exists
                return True
        # If False, all dirs created successfully
        return False
        
    def check_segments(self):
        # No output directory required for verify_segments
        # dataset_type, seed, uqid, sample_rate, segment_length, segments_path, gap
        check = {'dataset_type': self.dataset,
                 'seed': self.seed,
                 'uqid': self.unique_dataset_id,
                 'sample_rate': self.sample_rate,
                 'segment_length': self.segment_length,
                 'segments_path': os.path.join(self.dirs['parent'], self.output_segment_file),
                 'gap': self.segment_gap}
        
        # Verification
        segment_verification(dirs=self.dirs, check=check)
        if self.verbose:
            print("verify_segments: All verifications passed successfully!")
    
    def check_priors(self):
        # Output directory needs to be created
        # This should create verification directory and priors
        """
        In os.path.join, the second argument shouldn't start with a "/"
        Why not?:
            t = 'C:\\Users\\Admin'

            In [14]: os.path.join(t, "/verification/priors")
            Out[14]: 'C:/verification/priors'

            In [15]: os.path.join(t, "verification/priors")
            Out[15]: 'C:\\Users\\Admin\\verification/priors
        """
        
        save_path = os.path.join(self.dirs['parent'], "verification/priors")
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=False)
        # Injection file input
        injection_file = os.path.join(self.dirs['parent'], self.output_injection_file)
        # save_dir, tc_llimit, tc_ulimit, segment_length, gap
        check = {'save_dir': save_path,
                 'tc_llimit': self.time_window_llimit,
                 'tc_ulimit': self.time_window_ulimit,
                 'segment_length': self.segment_length,
                 'gap': self.segment_gap}
        
        # Verification
        prior_verification(injection_file, check)
        if self.verbose:
            print("verify_priors: Histograms created using injections file. Verify manually.")
    
    def check_signals(self):
        # Output directory for verify signals
        # Verification should already exist from check_priors call
        save_path = os.path.join(self.dirs['parent'], "verification/signals")
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=False)
        # ndata
        check = {'ndata': self.check_n_signals}
        # Verification
        signal_verification(self.dirs, check)
        if self.verbose:
            print(f"verify_signals: {self.check_n_signals} strain plots created. Check manually.")
    
    def make_training_lookup(self):
        # Make a lookup table with (id, path, target) for each training data
        # Should have an equal ratio, shuffled, of both classes (signal and noise)
        # Save this data in the hdf5 format
        noise_path = os.path.join(self.dirs['parent'], "background")
        noisy_signal_path = os.path.join(self.dirs['parent'], "foreground")
        # Get all absolute paths of files within fg and bg
        noise_abspaths = self._get_recursive_abspath_(noise_path)
        signal_abspaths = self._get_recursive_abspath_(noisy_signal_path)
        all_abspaths = np.concatenate((noise_abspaths, signal_abspaths))
        # Get the ids for each data sample
        dataset_length = len(all_abspaths)
        ids = np.linspace(0, dataset_length, dataset_length, dtype=np.int32)
        # Get the target/label value for each data sample
        targets = [0]*len(noise_abspaths) + [1]*len(signal_abspaths)
        # Column stack (ids, path, target) for the entire dataset
        lookup = np.column_stack((ids, all_abspaths, targets))
        # Shuffle the column stack ('tc' is in ascending order, signal and noise are not shuffled)
        # NumPy: "Multi-dimensional arrays are only shuffled along the first axis"
        np.random.shuffle(lookup)
        # Convert to pandas dataframe
        df = pd.DataFrame(data=lookup, columns=["id", "path", "target"])
        # Save the dataset paths alongside the target and ids as hdf5
        self.dirs['lookup'] = os.path.join(self.export_dir, "training.hdf")
        df.to_hdf(self.dirs['lookup'], "lookup", complib="blosc:lz4", complevel=9, mode='a')
        
        with h5py.File(self.dirs['lookup'], 'a') as ds:
            ds.attrs['unique_dataset_id'] = self.unique_dataset_id
            ds.attrs['creation_time'] = self.creation_time
            ds.attrs['dataset'] = self.dataset
            ds.attrs['seed'] = self.seed
            ds.attrs['segment_length'] = self.segment_length
            ds.attrs['tc_window_llimit'] = self.time_window_llimit
            ds.attrs['tc_window_ulimit'] = self.time_window_ulimit
            ds.attrs['sampling_rate'] = self.sample_rate
            ds.attrs['noise_num'] = len(noise_abspaths)
            ds.attrs['signal_num'] = len(signal_abspaths)
            ds.attrs['dataset_ratio'] = len(signal_abspaths)/len(noise_abspaths)
        
        # Create a copy of training.hdf to the dataset directory
        # This file is used later if using this dataset with make_dataset=False
        shutil.copy(self.dirs['lookup'], self.dirs['parent'])
        
        if self.verbose:
            print("make_training_lookup: Lookup table created successfully!")
    
    def check_dataset(self):
        # Verify whether all required training files are located in the desired dir
        # This is the last data preparation step before the ML pipeline takes over
        if not os.path.exists(self.dirs['lookup']):
            raise ValueError("training.hdf does not exist. ML-Pipeline will fail.")
        _, extension = os.path.splitext(self.dirs['lookup'])
        if extension != '.hdf':
            raise ValueError("Lookup table for training dataset should be .hdf file")
        # Successful
        if self.verbose:
            print("check_dataset: Lookup table has passed verification test.")
        
        
def make(slots_magic_params, export_dir):
    
    gd = GenerateData(**slots_magic_params)
    
    # Create storage directory sub-structure
    dataset_exists = gd.make_data_dir()
    # Setting up the export directory
    gd.export_dir = export_dir
    
    if not dataset_exists:
        gd.make_segments()
        gd.call_gendata()
        gd.check_segments()
        gd.check_priors()
        gd.check_signals()
    
        # Make dataset lookup table
        gd.make_training_lookup()
        # Verify location and extension of lookup for next step
        gd.check_dataset()
        
    else:
        raise ValueError("Dataset already exists. Overwriting causes unwanted behaviour.")
    
    """
    except Exception as e:
        with open(os.path.join(gd.dirs['parent'], "err.txt"), 'w') as file:
            file.write(str(e))
        if os.path.exists(gd.dirs['parent']):
            shutil.rmtree(gd.dirs['parent'])
    """
