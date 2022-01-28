# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Jan 27 00:05:55 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

# IN-BUILT


# LOCAL



# WARNING: Removing any of the parameters present in default will result in errors.

""" DEFAULT """
class Default:
    
    """ Make or Use """
    # if True, a new dataset is created based on the options below
    # else, searches for existing dataset located at os.join(parent_dir, data_dir)
    make_dataset = True
    
    """ Location """
    # Dataset location directory
    # Data storage drive or /mnt absolute path
    parent_dir = ""
    # Dataset directory within parent_dir
    data_dir = "dataset_0"
    
    """ Basic dataset options """
    # These options are used by generate_data.py
    # Type of dataset (1, 2, 3 or 4)
    # Refer https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Data-Sets 
    dataset = 3
    # Random seed provided to generate_data script
    # This will be unique and secret for the testing set
    seed = 42
    # Dataset params
    start_offset = 0
    dataset_duration = 2000
    # Other options
    verbose = True
    force = True
    
    """ pycbc_create_injections options """
    ## make_injections using pycbc_create_injections (uses self.seed)
    # params will be used to call above function via generate_data.py
    # pycbc_create_injections has been modified by nnarenraju (Dec 15th, 2021)
    # Start time of segments
    segment_GPS_start_time = 0.0
    # Time window within which to place the merger
    # 'tc' is located within this window
    time_window_llimit = 14
    time_window_ulimit = 16
    # Length of segment/duration (in seconds)
    segment_length = 20
    # Gap b/w adjacent segments (if any)
    segment_gap = 1
    
    """ Other Options """
    # NOTE: ninjections here is *NOT* provided to pycbc_create_injections
    # Only used to create segments.csv via make_segments.py
    # If 100 injections with 20 second segments are requested,
    # the total duration in segments.csv will be ~2100.0 seconds including gap.
    # This value may be larger than self.duration which is provided to generate_data.py
    # If we request 200s, we obtain the subset of segments required to produce that 
    # from segments.csv. So, we obtain 10 segments, each with one signal.
    ninjections = 100
