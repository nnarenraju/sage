# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sun Nov 21 15:30:43 2021

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
import gc
import torch
import argparse
import numpy as np
import pytorch_lightning as pl

from torchsummary import summary
from datetime import datetime
from distutils.dir_util import copy_tree

# Warnings
import warnings
warnings.filterwarnings("ignore")

# LOCAL
from lightning import simple
from manual import train as manual_train
from data.prepare_data import DataModule as dat
from data.MP_save_trainable import MP_Trainable
from utils.plotter import debug_plotter, snr_plotter

# Tensorboard
from torch.utils.tensorboard import SummaryWriter


def run_trainer():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default='Baseline',
                        help="Uses the pipeline architecture as described in configs.py")
    parser.add_argument("--data-config", type=str, default='Default',
                        help="Creates or uses a particular dataset as provided in data_configs.py")
    parser.add_argument("--inference", action='store_true',
                        help="Running the inference module using trained model")
    parser.add_argument("--lightning", action='store_true',
                        help="Running the pipeline using PyTorch Lightning")
    parser.add_argument("--manual", action='store_true',
                        help="Running the pipeline using manual PyTorch")
    parser.add_argument("--summary", action='store_true',
                        help="Store model summary using pytorch_summary")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--silent", action='store_true')
    
    opts = parser.parse_args()
    
    """ Prepare Data """
    # Get model configuration
    cfg = dat.configure_pipeline(opts)
    # Get data creation/usage configuration
    data_cfg = dat.configure_dataset(opts)
    
    # Make export dir
    dat.make_export_dir(cfg)
    
    # Prepare input data for training and testing
    # TODO: Currently get_summary() does not handle testing dataset
    # This should create/use a dataset and save a copy of the lookup table
    dat.get_summary(cfg, data_cfg, cfg.export_dir)
    
    # Prepare dataset (read, split and return fold idx)
    # Folds are based on stratified-KFold method in Sklearn (preserves class ratio)
    # TODO: Test data is not split (Under Construction!)
    train, folds = dat.get_metadata(cfg)

    """ Training """
    # Folds are obtained only by splitting the training dataset
    # Saving trainable dataset for each fold (if option enabled)
    if cfg.save_trainable_dataset:
        save_obj = MP_Trainable(cfg, data_cfg, len(folds)-1, opts.nbatch)
        
    # Use folds for cross-validation
    for nfold, (train_idx, val_idx) in enumerate(folds):
        
        # TODO: Implement inference section
        if opts.inference:
            continue
        
        if cfg.splitter != None:
            print(f'\n========================= TRAINING FOLD {nfold} =========================\n')

        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]
        
        # Saving a trainable dataset
        """ Trainable data storage snippet """
        # If we have cfg.splitter set to None, entire data is split in 80-20 partitions
        # nfold will immediately be the last one, so we should store trainable.hdf for that one
        if cfg.save_trainable_dataset:
            # We do not discriminate between training and validation dataset
            save_obj.data_paths = np.append(train_fold['path'].values, val_fold['path'].values)
            save_obj.targets = np.append(train_fold['target'].values, val_fold['target'].values)
            save_obj.nfold = nfold
            # Run the trainable dataset iterator
            save_obj.save_trainable_dataset()
            # Return and leave if trainable data storage is complete
            print("manual.py: Trainable data storage complete. Train with simple DataLoader.")
            # Training and validation is not done when this option is on.
            continue
        
        # Get the dataset objects for training and validation
        train_data, val_data = dat.get_dataset_objects(cfg, data_cfg, train_fold, val_fold)
        
        # Get the Pytorch DataLoader objects of train and valid data
        train_loader, val_loader = dat.get_dataloader(cfg, train_data, val_data)
        
        # Initialise chosen model architecture (Backend + Frontend)
        # Equivalent to the "Network" variable in manual mode
        ModelClass = cfg.model(**cfg.model_params)
        
        # Load weights snapshot
        if cfg.pretrained and cfg.weights_path!='':
            if os.path.exists(cfg.weights_path):
                weights = torch.load(cfg.weights_path, cfg.store_device)
                ModelClass.load_state_dict(weights)
                del weights; gc.collect()
            else:
                raise ValueError("train.py: cfg.weights_path does not exist!")
        elif cfg.pretrained and cfg.weights_path=='':
            raise ValueError("CFG: pretrained==True, but no weights path provided!")
        
        # Model Summary (frontend + backend)
        if opts.summary:
            # Using TorchSummary to get # trainable params and general overview
            summary(ModelClass, (2, 3072), batch_size=cfg.batch_size)
            print("")
            # Using TensorBoard summary writer to create detailed graph of ModelClass
            tb = SummaryWriter()
            samples, labels = next(iter(train_loader))
            tb.add_graph(ModelClass, samples)
            tb.close()
        
        # Optimizer and Scheduler (Set to None if unused)
        if cfg.optimizer is not None:
            optimizer = cfg.optimizer(ModelClass.parameters(), **cfg.optimizer_params)
        else:
            optimizer = None
            
        if cfg.scheduler is not None:
            scheduler = cfg.scheduler(optimizer, **cfg.scheduler_params)
        else:
            scheduler = None
        
        # Loss function used
        loss_function = cfg.loss_function
        
        
        """ Training and Validation Methods """
        ## MANUAL
        if opts.manual:
            # Running the manual pipeline version using pure PyTorch
            # Initialise the trainer
            manual_train(cfg, data_cfg, ModelClass, optimizer, scheduler, loss_function, 
                         train_loader, val_loader, verbose=cfg.verbose)
        
        ## LIGHTNING
        if opts.lightning:
            # Get Lightning Classifier from lightning.py
            model = simple(ModelClass, optimizer, scheduler, loss_function)
            
            # Initialise the trainer
            trainer = pl.Trainer(max_steps=cfg.num_steps, max_epochs=cfg.num_epochs)
            
            """ Fit """
            trainer.fit(model, train_loader, val_loader)

        # Loading the network with the best weights path
        # This step is mandatory before the inference/testing module
        # Network.load_state_dict(torch.load(cfg.weights_path))
        
        # Debug method plotting
        debug_dir = os.path.join(cfg.export_dir, 'DEBUG')
        debug_plotter(debug_dir)
        
        # Plotting the SNR histogram
        snr_dir = os.path.join(cfg.export_dir, 'SNR')
        snr_plotter(snr_dir, cfg.num_epochs)
        
        # Move export dir for current run to online workspace
        file_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        www_dir = 'RUN-{}-dataset{}-model-{}-remark-{}'.format(file_time, data_cfg.dataset, cfg.model_params['model_name'], cfg.save_remarks)
        copy_tree(cfg.export_dir, os.path.join(cfg.online_workspace, www_dir))
        

if __name__ == "__main__":
    
    run_trainer()
    
    
