#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Apr 14 12:37:59 2022

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
import json
import shutil
import argparse
import numpy as np
from collections import defaultdict
# Using defaultdict to concatenate dicts with same key (loaded json)
# Make dd with list as anything else might now be JSON hashable
dd = defaultdict(list)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch-dir", type=str, default='/Users/nnarenraju/Desktop',
                        help="Directory containing all trainable.json files and trainable data files")
    parser.add_argument("--save-dir", type=str, default='dataset_trainable_batched',
                        help="Dataset dir in the usual format containing all relevant data")
    parser.add_argument("--file_name", type=str, default='trainable.json',
                        help="Name of the final contatenated trainable.json file")
    parser.add_argument("--silent", action='store_true')
    
    opts = parser.parse_args()
    
    
    # Get all dirs required to make full dataset
    data_dirs = glob.glob(os.path.join(opts.batch_dir, "dataset_5e4_20s_D1_Batch_*"))
    
    trainable_lookup = []
    trainable_data = []
    for data_dir in data_dirs:
        # Search the batch dir and get all the trainable.json files
        trainable_lookup.extend(glob.glob(os.path.join(data_dir, "*.json")))
        # Look for trainable data within batch folder
        trainable_data.extend(glob.glob(os.path.join(data_dir, "trainable_batched_dataset/*.hdf")))
    
    # Get all JSON file data
    all_json_data = []
    for json_file in trainable_lookup:
        with open(json_file) as data_file:
            all_json_data.append(json.load(data_file))
    
    # Concatenate all JSON data together into one trainable.json
    for n, dic in enumerate(all_json_data):
        for key, value in dic.items():
            if key == "path":
                value = trainable_data[n]
            if key == "ids":
                value = n
            if key in ["target", "norm_tc", "distance", "mchirp"]:
                if len(value) == 1:
                    value = value[0]
            
            dd[key].append(value)
    
    # Make a dataset directory structure similar to train.py
    save_dir = os.path.join(opts.batch_dir, opts.save_dir)
    os.makedirs(save_dir, exist_ok=False)
    
    # Save trainable.json in the correct location
    save_path = os.path.join(save_dir, opts.file_name)
    with open(save_path, 'w', encoding='utf-8') as fp:
        json.dump(dd, fp)
        
    
    
    
    
    
    
    
            
    

