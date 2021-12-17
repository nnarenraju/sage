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
    1. Call make_segments.py using appropriate opts to create segments.csv (FIN)
    2. Call generate_data.py from ML-MDC1 using segments.csv to make training data (FIN)
    3. Change generate_data to store each segment as a separate HDF5 file
    3. Verify whether all params and segments are as intended
    4. Cross verify the prior distribution with the observed distribution
    5. Cross verify the location of the GW signal and its 'tc'
    6. Visualise the signals, noise and signal+noise for by-eye verification
    7. After all sanity checks and verification, save dataset in the appropriate format
    8. Create a training.hdf5 that handles ids, paths and target of training data
    9. Store all the above data into a data-read directory for next step of the algorithm

"""

# IN-BUILT
import os
import sys
import h5py
import uuid
import warnings
import datetime

# LOCAL
from make_segments import make_segments as mksegments
# Add MLMDC1 repo from GWastro
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../external/ml-mock-data-challenge-1')
from generate_data import main as gendata


class GenerateData:
    """
    
    """
    
    def __init__(self):
        ## generate_data script parameters
        # Type of dataset (1, 2, 3 or 4)
        # Refer https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Data-Sets 
        self.dataset = 3
        # Path to store output HDF5 files
        self.output_injection_file = "injections.hdf"
        self.output_foreground_file = "foreground.hdf"
        self.output_background_file = "background.hdf"
        # Random seed provided to generate_data script
        # This will be unique and secret for the testing set
        self.seed = 42
        # Dataset params
        self.start_offset = 0
        self.dataset_duration = 2000
        # Other options
        self.verbose = True
        self.force = False
        
        ## make_injections using pycbc_create_injections (uses self.seed)
        # params will be used to call above function via generate_data.py
        # pycbc_create_injections has been modified by nnarenraju (Dec 15th, 2021)
        self.output_segment_file = ""
        # Start time of segments
        self.segment_GPS_start_time = 0.0
        # Distance b/w two adjacent 'tc'
        # For training set, set to a value larger than segment length
        self.time_step = 20
        # Time window within which to place the merger
        # 'tc' is located within this window
        self.time_window_llimit = None
        self.time_window_ulimit = None
        # Length of segment/duration (in seconds)
        self.segment_length = 20
        self.ninjections = 0
        # Gap b/w adjacent segments (if any)
        self.segment_gap = None
        
        # Metadata and identification
        self.id = uuid.uuid4().hex
        self.creation_time = datetime.datetime.now()
    
    def __str__(self):
        pass
    
    def make_segments(self):
        # Make segments.csv to be used by generate_data script
        # Create an object for Segments class
        args =  (self.segment_GPS_start_time,)
        args += (self.segment_gap,)
        args += (self.segment_length,)
        args += (self.ninjections,)
        args += (self.output_segment_file,)
        args += (self.force)
        
        # Make segments.csv (equal length assumed)
        mksegments(*args)

    def call_gendata(self):
        # Main params to call generate_data script
        # Warning! for dataset 4 large noise data download
        if self.dataset == 4:
            warnings.warn("Dataset type 4 downloads an ~94GB noise file.")
            
        raw_args =  ['--data-set', str(self.dataset)]
        raw_args += ['--output-injection-file', self.output_injection_file]
        raw_args += ['--output-foreground-file', self.output_foreground_file]
        raw_args += ['--output-background-file', self.output_background_file]
        
        # sanity check for segments.csv
        if os.path.exists(self.output_segment_file):
            raw_args += ['--input-segment-file', self.output_segment_file]
        else:
            raise IOError("{} does not exist!".format(self.output_segment_file))
            
        raw_args += ['--seed', str(self.seed)]
        raw_args += ['--start-offset', str(self.start_offset)]
        raw_args += ['--duration', str(self.dataset_duration)]
        
        # MISC params
        if self.verbose:
            raw_args += ['--verbose']
        if self.force:
            raw_args += ['--force']
        
        gendata(raw_args)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
