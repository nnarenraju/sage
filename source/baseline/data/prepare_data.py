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
import h5py
import shutil
import numpy as np
import pandas as pd
import torch.utils.data as D

# LOCAL
from configs import *
from data_configs import *


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
        # This creates a new instance, handle this in run.py
        cfg = eval(opts.config)
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
    
    
    def make_export_dir(cfg):
        # Results directory (preferred: /mnt/nnarenraju)
        cfg.export_dir.mkdir(parents=True, exist_ok=False)
    
    
    def get_summary(cfg, data_cfg, export_dir):
        """
        Creates/uses a training dataset in hdf format
        Consolidate the IDs, target and path into CSV files
        Done for training and testing data
    
        Parameters
        ----------
        data_cfg : class
            Class containing data_config information
            eg. cfg is Default class in data_configs.py
        
        export_dir : str
            Location of output directory for results and lightning
    
        Returns
        -------
        None.
    
        """
        
        # Getting the attributes of data_config class as dict
        dc_attrs = {key:value for key, value in data_cfg.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        # Set "make_dataset" to the appropriate method
        make_module = dc_attrs['make_module']
        
        # Training data
        if dc_attrs['make_dataset']:
            make_module(dc_attrs, export_dir)
        
        # Check if dataset already exists
        # If it does, check if training.hdf exists within it and save in export dir
        check_dir = os.path.join(dc_attrs['parent_dir'], dc_attrs['data_dir'])
        if os.path.isdir(check_dir):
            # Lookup table formats
            check_file = os.path.join(check_dir, "training.hdf")
            elink_file = os.path.join(check_dir, "extlinks.hdf")
            # Copy lookup tables to export_dir
            if os.path.isfile(check_file):
                # Move the training.hdf to export_dir for pipeline
                move_location = os.path.join(export_dir, 'training.hdf')
                shutil.copy(check_file, move_location)
            if os.path.isfile(elink_file):
                # Move the extlinks.hdf to export_dir for pipeline
                move_location = os.path.join(export_dir, 'extlinks.hdf')
                shutil.copy(elink_file, move_location)
        else:
            raise FileNotFoundError(f"prepare_data.get_summary: {check_dir} not found!")
        
        # Testing data
        # TODO: A testing dataset should also be created in a similar format
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
        
        # Set lookup table
        if cfg.megabatch:
            lookup_file = 'training.hdf'
        else:
            lookup_file = 'extlinks.hdf'
        
        lookup_table = os.path.join(cfg.export_dir, lookup_file)
        # Deprecated: Usage of self.get_metadata with make_default_dataset (non-MP)
        with h5py.File(lookup_table, 'a') as fp:
            ids = np.array(fp['id'][:])
            paths = np.array([foo.decode('utf-8') for foo in fp['path']])
            targets = np.array(fp['target'][:])
        
        # Chunk the entire lookup table for BatchLoader method used within MLMDC1 dataset object
        # The independant BatchLoader Method has been deprecated as of May 24th, 2022
        # mega_batch_size method deprecated on May 31st, 2022
        
        """
        Deprecated MegaBatch method
        ---------------------------
            
            if cfg.mega_batch_size != -1 and cfg.mega_batch_size != None:
                paths = np.array_split(paths, int(len(paths)/cfg.mega_batch_size))
                targets = np.array_split(targets, int(len(targets)/cfg.mega_batch_size))
                # ids are not split, they are converted to regular single int ids
                ids = np.arange(len(targets))
                
        """
        
        # Create a consolidated lookup date as a Pandas Dataframe
        lookup = list(zip(ids, paths, targets))
        train = pd.DataFrame(lookup, columns=['id', 'path', 'target'])
        
        # TODO: Do the same prodecure for testing dataset. Does not require splitting.
        # Under construction!
        # if debug, use a data subset
        if cfg.debug:
            train = train.iloc[:cfg.debug_size]
        ## Splitting
        if cfg.splitter is not None:
            # Function ensures equal ratio of all classes in each fold
            folds = list(cfg.splitter.split(train, train['target']))
        else:
            # Splitting training and validation in 80-20 sections
            # This essentially has all the data in 1 fold
            N = len(train)
            idxs = np.arange(N)
            folds = [(idxs[:int(0.8*N)], idxs[int(0.8*N):])]
        
        return (train, folds)
    
    
    def get_dataset_objects(cfg, data_cfg, train_fold, valid_fold):
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
        
        # transforms are not required if the cfg.dataset is made for trainable data
        # this is automagically taken care of within the BatchLoader datasets class
        train_dataset = cfg.dataset(
                data_paths=train_fold['path'].values, targets=train_fold['target'].values,
                transforms=cfg.transforms['train'], target_transforms=cfg.transforms['target'],
                signal_only_transforms=cfg.transforms['signal'], 
                noise_only_transforms=cfg.transforms['noise'],
                training = True, cfg=cfg, data_cfg=data_cfg, store_device=cfg.store_device,
                train_device=cfg.train_device, **cfg.dataset_params)
        
        valid_dataset = cfg.dataset(
                data_paths=valid_fold['path'].values, targets=valid_fold['target'].values,
                transforms=cfg.transforms['test'], target_transforms=cfg.transforms['target'],
                signal_only_transforms=cfg.transforms['signal'],
                noise_only_transforms=cfg.transforms['noise'],
                training=True, cfg=cfg, data_cfg=data_cfg, store_device=cfg.store_device,
                train_device=cfg.train_device, **cfg.dataset_params)
        
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
        
        # Check cfg and set batch_size to 1 if using MegaBatch
        if cfg.megabatch:
            batch_size = 1
            num_workers = 0
            pin_memory = False
            persistent_workers = False
        else:
            batch_size = cfg.batch_size
            num_workers = cfg.num_workers
            pin_memory = cfg.pin_memory
            persistent_workers = cfg.persistent_workers
        
        # Sometimes in MAC systems, setting num_workers > 0 causes a intraop warning to appear
        # This does not seem to produce any incorrect results. However it is worrying.
        # Temporary fix (suppresses the error, but does not fix the underlying issue)
        os.environ["OMP_NUM_THREADS"] = "1"
        
        train_loader = D.DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, 
            prefetch_factor=cfg.prefetch_factor, persistent_workers=persistent_workers)
        
        valid_loader = D.DataLoader(
            valid_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, 
            prefetch_factor=cfg.prefetch_factor, persistent_workers=persistent_workers)
        
        return (train_loader, valid_loader)
