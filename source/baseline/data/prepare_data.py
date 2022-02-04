# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sat Nov 27 14:39:57 2021

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
import os
import shutil
import numpy as np
import pandas as pd
import torch.utils.data as D

# LOCAL
from configs import *
from data_configs import *
from data.make_dataset import make as make_dataset

class DataModule:
    """
    Module to handle dataset preparation.
    Not to be confused with PyTorch Lightning DataModule (L-DM).
    # Message written on 27/11/2021
    Due to lack of compatibility with cross-validation, L-DM was not used
    
    User-Defined Functions
    ----------------------
    configure_path - 
    get_summary - 
    get_metadata - 
    get_dataset_objects - 
    get_data_loader -
    
    """

    def configure_pipeline(opts):
        """ 
        Read pipeline configuration and create export directory
        
        Arguments
        ---------
        opts : object
            argparse object from config.py
        
        Returns
        -------
        cfg : object
            model object from config.py
            
        """
        
        # cfg contains the model class (see config.py)
        cfg = eval(opts.config)
        # Results directory (preferred: /mnt/nnarenraju)
        cfg.export_dir.mkdir(parents=True, exist_ok=True)
        return cfg
    
    def configure_dataset(opts):
        """ 
        Read dataset configuration
        
        Arguments
        ---------
        opts : object
            argparse object from data_config.py
        
        Returns
        -------
        cfg : object
            model object from data_config.py
            
        """
        
        # cfg contains the model class (see config.py)
        cfg = eval(opts.data_config)
        return cfg
    
    def get_summary(cfg, export_dir):
        """
        Creates/uses a training dataset in hdf format
        Consolidate the IDs, target and path into CSV files
        Done for training and testing data
    
        Parameters
        ----------
        cfg : class
            Class containing data_config information
            eg. cfg is Default class in data_configs.py
        
        export_dir : str
            Location of output directory for results and lightning
    
        Returns
        -------
        None.
    
        """
        # TODO: A testing dataset should also be created in a similar format
        
        # Getting the attributes of data_config class as dict
        dc_attrs = {key:value for key, value in cfg.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        # Training data
        if dc_attrs['make_dataset']:
            make_dataset(dc_attrs, export_dir)
        else:
            # Check if dataset already exists
            # If it does, check if training.hdf exists within it and save in export dir
            check_dir = os.path.join(dc_attrs['parent_dir'], dc_attrs['data_dir'])
            if os.path.isdir(check_dir):
                check_file = os.path.join(check_dir, "training.hdf")
                if os.path.isfile(check_file):
                    # Move the training.hdf to export_dir for pipeline
                    shutil.copy(check_file, export_dir)
        
        # Testing data
        # Under construction!
            
    def get_metadata(cfg):
        """
        Retrieve dataset metadata
        
        Parameters
        ----------
        cfg : object
            model object from config.py
        
        Returns
        -------
        train : pd dataframe
            training metadata from input dir
        test : pd dataframe
            testing metadata from input dir
        folds : K-fold split data idx
            train/test indices to split data in train/test sets    
            
        """
        
        # Using a dask Dataframe for larger CSV files
        train = pd.read_hdf(os.path.join(cfg.export_dir, "training.hdf"), 'lookup')
        # TODO: Do the same prodecure for testing dataset. Does not require splitting.
        # Under construction!
        # if debug, use a data subset
        if cfg.debug:
            train = train.iloc[:10000]
        ## Splitting
        if cfg.splitter is not None:
            # Function ensures equal ratio of all classes in each fold
            folds = list(cfg.splitter.split(train, train['target']))
        else:
            # Splitting training and validation in 80-20 sections
            N = len(train)
            idxs = np.arange(N)
            folds = [(idxs[:int(0.8*N)], idxs[int(0.8*N):])]
        
        return (train, folds)
    
    def get_dataset_objects(cfg, train_fold, valid_fold):
        """
        Initialise Training/Validation/Testing Dataset
        
        Options provided in dataset object:
            [1] data_paths, targets
            [2] transforms=None, target_transforms=None
            
            TODO: implement cache data option
            [3] cache=None
        
        Link to map-style dataset object:
            https://pytorch.org/docs/stable/data.html
        
        Parameters
        ----------
        cfg : object
            model object from config.py
        
        Returns
        -------
        train_dataset : map-style dataset object
        valid_dataset : map-style dataset object
        
        """
        # TODO: Dataset object needs to be manipulated to handle the hdf formats for foreground
        
        train_dataset = cfg.dataset(
                data_paths=train_fold['path'].values, targets=train_fold['target'].values,
                transforms=cfg.transforms['train'], target_transforms=cfg.transforms['target'],
                training = True, **cfg.dataset_params)
        
        valid_dataset = cfg.dataset(
                data_paths=valid_fold['path'].values, targets=valid_fold['target'].values,
                transforms=cfg.transforms['test'], target_transforms=cfg.transforms['target'],
                training=True, **cfg.dataset_params)
        
        return (train_dataset, valid_dataset)
    
    def get_dataloader(cfg, train_data, valid_data):
        """
        Create Pytorch DataLoader objects
        
        Signature of DataLoader:
        DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
                   batch_sampler=None, num_workers=0, collate_fn=None,
                   pin_memory=False, drop_last=False, timeout=0,
                   worker_init_fn=None, *, prefetch_factor=2,
                   persistent_workers=False)
    
        Parameters
        ----------
        train_data : map-style dataset object
        valid_data : map-style dataset object
    
        Returns
        -------
        train_loader : Pytorch DataLoader object
        valid_loader : Pytorch DataLoader object
        
        """
        
        train_loader = D.DataLoader(
            train_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=False)
        valid_loader = D.DataLoader(
            valid_data, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=False)
        
        return (train_loader, valid_loader)
