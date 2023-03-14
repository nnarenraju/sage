#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Mon Mar 13 20:48:18 2023

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

# Modules
import os
import glob
import datetime

from distutils.dir_util import copy_tree

# LOCAL
from utils.plotter import debug_plotter, snr_plotter, overlay_plotter


def save(cfg, data_cfg):
    
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
