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
    1. Call make_segments.py using appropriate opts to create segments.csv
    2. Call generate_data.py from ML-MDC1 using segments.csv to make training data
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
import numpy as np

# LOCAL
from make_segments import Segments
# Add MLMDC1 repo from GWastro
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../../external/mlmdc1')
from generate_data import main as gendata

def call_gendata():
    raw_args = ['--data-set', str(3)]
    raw_args += ['--output-injection-file', "injections.hdf"]
    raw_args += ['--output-foreground-file', "foreground.hdf"]
    raw_args += ['--output-background-file', "background.hdf"]
    raw_args += ['--seed', str(42)]
    raw_args += ['--start-offset', str(0)]
    raw_args += ['--duration', str(2000)]
    raw_args += ['--verbose']
    raw_args += ['--force']
    
    gendata(raw_args)


if __name__ == "__main__":
    
    call_gendata()

