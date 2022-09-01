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
import glob
import h5py
import torch
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 18})

from torchsummary import summary
from datetime import datetime
from distutils.dir_util import copy_tree

# Warnings
import warnings
warnings.filterwarnings("ignore")

# LOCAL
from test import run_test
from lightning import simple
from evaluate import main as evaluator
from manual import train as manual_train
from data.prepare_data import DataModule as dat
from utils.plotter import debug_plotter, snr_plotter, overlay_plotter

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
    
    opts = parser.parse_args()
    
    
    """ Prepare Data """
    # Get model configuration
    cfg = dat.configure_pipeline(opts)
    # Get data creation/usage configuration
    data_cfg = dat.configure_dataset(opts)
    
    # Make export dir
    dat.make_export_dir(cfg)
    
    # Prepare input data for training and testing
    # This should create/use a dataset and save a copy of the lookup table
    dat.get_summary(cfg, data_cfg, cfg.export_dir)
    
    # Prepare dataset (read, split and return fold idx)
    # Folds are based on stratified-KFold method in Sklearn (preserves class ratio)
    train, folds = dat.get_metadata(cfg)

    """ Training """
    # Folds are obtained only by splitting the training dataset
        
    # Use folds for cross-validation
    for nfold, (train_idx, val_idx) in enumerate(folds):
        
        if cfg.splitter != None:
            raise NotImplementedError('K-fold cross validation method under construction!')
            print(f'\n========================= TRAINING FOLD {nfold} =========================\n')

        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]
        
        # Get the dataset objects for training and validation
        train_data, val_data, tsamp, vsamp = dat.get_dataset_objects(cfg, data_cfg, train_fold, val_fold)
        
        # Get the Pytorch DataLoader objects of train and valid data
        train_loader, val_loader = dat.get_dataloader(cfg, train_data, val_data, tsamp, vsamp)
        
        # Initialise chosen model architecture (Backend + Frontend)
        # Equivalent to the "Network" variable in manual mode without weights
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
            Network = manual_train(cfg, data_cfg, ModelClass, optimizer, scheduler, loss_function, 
                                   train_loader, val_loader, verbose=cfg.verbose)
        
        ## LIGHTNING
        if opts.lightning:
            # Get Lightning Classifier from lightning.py
            raise NotImplementedError('Lightning module under construction!')
            model = simple(ModelClass, optimizer, scheduler, loss_function)
            
            # Initialise the trainer
            trainer = pl.Trainer(max_steps=cfg.num_steps, max_epochs=cfg.num_epochs)
            
            """ Fit """
            trainer.fit(model, train_loader, val_loader)
            
        
        """ Saving results and moving to online workspace """
        # Debug method plotting
        if cfg.debug:
            # Debug directory and plots
            debug_dir = os.path.join(cfg.export_dir, 'DEBUG')
            debug_plotter(debug_dir)
            # Plotting the SNR histogram
            snr_dir = os.path.join(cfg.export_dir, 'SNR')
            snr_plotter(snr_dir, cfg.num_epochs)
        
        # Move export dir for current run to online workspace
        file_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        if cfg.debug:
            run_type = 'DEBUG'
        else:
            run_type = 'RUN'
        www_dir = '{}-{}-D{}-{}-{}'.format(run_type, file_time, data_cfg.dataset, cfg.model_params['model_name'], cfg.save_remarks)
        copy_tree(cfg.export_dir, os.path.join(cfg.online_workspace, www_dir))
        
        ## Making an overlay plot of all runs in the online workspace
        # All runs that are in DEBUG mode are ignored for the overlay plots
        overview_paths = []
        roc_paths = []
        run_names = []
        roc_aucs = []
        flag_1 = False
        flag_2 = False
        flag_3 = False
        for run_dir in glob.glob(os.path.join(cfg.online_workspace, 'RUN-*')):
            # Get the loss, accuracy and ROC curve data from the best file (if present)
            overview_path = os.path.join(run_dir, 'losses.txt')
            if os.path.exists(overview_path):
                flag_1 = True
            roc_path = os.path.join(run_dir, 'BEST/roc_best.npy')
            if os.path.exists(roc_path):
                flag_2 = True
            roc_auc_path = os.path.join(run_dir, 'BEST/roc_auc_best.npy')
            if os.path.exists(roc_auc_path):
                flag_3 = True
            if flag_1 and flag_2 and flag_3:
                run_names.append(os.path.split(run_dir)[-1])
                overview_paths.append(overview_path)
                roc_paths.append(roc_path)
                roc_aucs.append(roc_auc_path)
            flag_1 = False
            flag_2 = False
            flag_3 = False
        
        save_dir = os.path.join(cfg.online_workspace, 'ALL_OVERLAY')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=False)
        if run_names != []:
            overlay_plotter(overview_paths, roc_paths, roc_aucs, save_dir, run_names)
        
        # Save a copy of the entire code used to run this config into the RUN/DEBUG directory
        # The GIT file size may be too large. Storing it each time within online_workspace may be overkill.
        # shutil.make_archive(os.path.join(cfg.online_workspace, www_dir), 'zip', src)
        
        
        """ TESTING """
        if opts.inference:
            # Running the testing phase for foreground data
            transforms = cfg.transforms['test']
            jobs = ['foreground', 'background']
            
            output_testing_dir = os.path.join(cfg.export_dir, 'TESTING')
            for job in jobs:
                # Get the required data based on testing job
                if job == 'foreground':
                    testfile = os.path.join(cfg.testing_dir, cfg.test_foreground_dataset)
                    evalfile = os.path.join(output_testing_dir, cfg.test_foreground_output)
                elif job == 'background':
                    testfile = os.path.join(cfg.testing_dir, cfg.test_background_dataset)
                    evalfile = os.path.join(output_testing_dir, cfg.test_background_output)
                    
                print('\nRunning the testing phase on {} data'.format(job))
                run_test(Network, testfile, evalfile, transforms, cfg, data_cfg,
                         step_size=cfg.step_size, slice_length=int(data_cfg.signal_length*data_cfg.sample_rate),
                         trigger_threshold=cfg.trigger_threshold, cluster_threshold=cfg.cluster_threshold, 
                         batch_size = cfg.batch_size,
                         device=cfg.testing_device, verbose=cfg.verbose)
        
        # Run the evaluator for the testing phase and add required files to TESTING dir in export_dir
        raw_args =  ['--injection-file', os.path.join(cfg.testing_dir, cfg.injection_file)]
        raw_args += ['--foreground-events', os.path.join(output_testing_dir, cfg.test_foreground_output)]
        raw_args += ['--foreground-files', os.path.join(cfg.testing_dir, cfg.test_foreground_dataset)]
        raw_args += ['--background-events', os.path.join(output_testing_dir, cfg.test_background_output)]
        raw_args += ['--far-scaling-factor', float(cfg.far_scaling_factor)]
        out_eval = os.path.join(output_testing_dir, cfg.evaluation_output)
        raw_args += ['--output-file', out_eval]
        raw_args += ['--output-dir', output_testing_dir]
        raw_args += ['--verbose']
        
        # Running the evaluator to obtain output triggers (with clustering)
        evaluator(raw_args)
        
        
        

if __name__ == "__main__":
    
    run_trainer()
    
    
