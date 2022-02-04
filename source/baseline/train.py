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
import argparse
import pytorch_lightning as pl
from torchsummary import summary

# Warnings
import warnings
warnings.filterwarnings("ignore")

# LOCAL
from lightning import simple
from manual import train as manual_train
from data.prepare_data import DataModule as dat
from architectures.frontend import get_sample_network


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default='Baseline',
                        help="Uses the pipeline architecture as described in configs.py")
    parser.add_argument("--data-config", type=str, default='Default',
                        help="Creates or uses a particular dataset as provided in data_configs.py")
    parser.add_argument("--inference", action='store_true',
                        help="Running the inference module using trained model")
    parser.add_argument("--manual", action='store_true',
                        help="Running the pipeline using manual pytorch instead of lightning")
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
    
    # Prepare input data for training and testing
    # TODO: Currently get_summary() does not handle testing dataset
    # This should create/use a dataset and save a copy of the lookup table
    dat.get_summary(data_cfg, cfg.export_dir)
    
    # Prepare dataset (read, split and return fold idx)
    # Folds are based on stratified-KFold method in Sklearn (preserves class ratio)
    # TODO: Test data is not split (Under Construction!)
    train, folds = dat.get_metadata(cfg)
    
    """ Training """
    # Folds are obtained only by splitting the training dataset
    # Use folds for cross-validation
    for fold, (train_idx, val_idx) in enumerate(folds):
        
        # TODO: Implement inference section
        if opts.inference:
            continue

        print(f'\n========================= TRAINING FOLD {fold} =========================\n')

        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]

        # Get the dataset objects for training and validation
        train_data, val_data = dat.get_dataset_objects(cfg, train_fold, val_fold)
        
        # Get the Pytorch DataLoader objects of train and valid data
        train_loader, val_loader = dat.get_dataloader(cfg, train_data, val_data)
        
        
        if not opts.manual:
            # Initialise chosen model architecture (Backend + Frontend)
            ModelClass = cfg.model(**cfg.model_params)
            
            # Model Summary (frontend + backend)
            if opts.summary:
                summary(ModelClass, (2, 40960), batch_size=cfg.batch_size)
                print("")
            
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
            
            # Get Lightning Classifier from lightning.py
            model = simple(ModelClass, optimizer, scheduler, loss_function)
            
            # Initialise trainer
            trainer = pl.Trainer(max_steps=cfg.num_steps, max_epochs=cfg.num_epochs)
            
            """ Fit """
            trainer.fit(model, train_loader, val_loader)
        
        else:
            # Running the manual pipeline version using pure PyTorch
            weights_path = "/home/nnarenraju/weights.pt"
            output_dir = "/home/nnarenraju"
            Network = get_sample_network()
            # Model Summary (frontend + backend)
            if opts.summary:
                summary(Network, (2, 40960), batch_size=cfg.batch_size)
                print("")
                
            Network = manual_train(Network, train_loader, val_loader, output_dir, weights_path,
                        batch_size=cfg.batch_size, learning_rate=5e-5,
                        epochs=cfg.num_epochs, clip_norm=100.0, verbose=True)
