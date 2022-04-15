#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Wed Apr 13 18:27:42 2022

__author__      = nnarenraju
__copyright__   = Copyright 2022, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation:
    
    This is a quick code to create a large dataset
    Will be changed later on to include argparse and other relevant options

"""

# BUILT-IN
import os
import gc
import time
import shutil

# LOCAL
from train import run_trainer


# Run the code and pass necessary argparse options
class AutoPilot:
    
    nbatch = -1
    # Configurations
    config = "KF_Trainable"
    data_config = "Default"
    # Testing
    inference = False
    # Training modes
    lightning = False
    manual = True
    # Get model summary on stdout
    summary = False
    # MISC
    debug = False
    silent = False


# Use this to iteratively create large datasets
# Delete raw_samples out of memory

nsamples = 2.0e6
batch_size = 50000
nbatches = int(nsamples/batch_size) + 1

for nbatch in range(44, 61):
    
    start = time.time()
    # Create an autopilot object and set batch params
    AP = AutoPilot()
    AP.nbatch = nbatch
    
    # Call train.py using the run function
    data_dir = run_trainer(autopilot=True, autotools=AP)
    # Delete all raw files once trainable dataset has been created
    shutil.rmtree(os.path.join(data_dir, 'foreground'))
    shutil.rmtree(os.path.join(data_dir, 'background'))
    # Delete the Batch_n export dir, this is no longer needed
    export_dir = os.path.join(os.path.split(data_dir)[0], "Batch_{}".format(nbatch))
    shutil.rmtree(export_dir)
    
    # Explicitly free up memory
    del AP; del data_dir; gc.collect()
    finish = time.time() - start
    print("Time taken for batch creation and transformation = {} minutes".format(finish/60.))
