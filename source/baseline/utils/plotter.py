#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Apr 21 19:41:42 2022

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

import os
import random
import numpy as np
import matplotlib.pyplot as plt


def _plotter(title, xlabel, ylabel, hists, labels, save_path):
    plt.figure(figsize=(9.0, 6.0))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for data, label in zip(hists, labels):
        plt.hist(data, bins=100, alpha=0.8, label=label)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def figure(title=""):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(16.0, 14.0))
    fig.suptitle(title, fontsize=28, y=0.92)
    return fig, axs


def _overplot(ax, x=None, y=None, xlabel="x-axis", ylabel="y-axis", ls='solid', 
          label="", c=None, yscale='linear', xscale='linear'):
    # Plotting type
    if label != "":
        ax.plot(x, y, ls=ls, c=c, linewidth=3.0, label=label)
    else:
        ax.plot(x, y, ls=ls, c=c, linewidth=3.0)
    # Plotting params
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def overlay_plotter(overview_filepaths, roc_paths, roc_aucs, save_dir, run_names):
    # Read diagnostic file
    fig_loss, ax_loss = figure(title="Loss Curves")
    fig_accr, ax_accr = figure(title="Accuracy Curves")
    fig_roc, ax_roc  = figure(title="ROC Curves")
    # colour map
    numc = len(overview_filepaths)
    cmap = ["#"+''.join([random.choice('ABCDEF0123456789') for _ in range(6)]) for _ in range(numc)]
    
    for n, overview_filepath in enumerate(overview_filepaths):
        data = np.loadtxt(overview_filepath)
        
        # All data fields
        epochs = data[:,0] + 1.0
        training_loss = data[:,1]
        validation_loss = data[:,2]
        training_accuracy = data[:,3]
        validation_accuracy = data[:,4]
        
        ## Loss Curves
        _overplot(ax_loss, epochs, training_loss, label=run_names[n], ylabel='Avg Loss', xlabel='Epochs', c=cmap[n])
        _overplot(ax_loss, epochs, validation_loss, ls='dashed', ylabel='Avg Loss', xlabel='Epochs', c=cmap[n])
        
        ## Accuracy Curves
        _overplot(ax_accr, epochs, training_accuracy, label=run_names[n], ylabel='Avg Accuracy', xlabel='Epochs', c=cmap[n])
        _overplot(ax_accr, epochs, validation_accuracy, ylabel='Avg Accuracy', xlabel='Epochs', ls='dashed', c=cmap[n])
    
    fig_loss.savefig(os.path.join(save_dir, 'overlay_loss.png'))
    fig_accr.savefig(os.path.join(save_dir, 'overlay_accuracy.png'))
    plt.close(fig_loss)
    plt.close(fig_accr)
    
    # Plotting the ROC overlay plot
    # colour map
    numc = len(roc_paths)
    cmap = ["#"+''.join([random.choice('ABCDEF0123456789') for _ in range(6)]) for _ in range(numc)]
    
    for n, roc_path in enumerate(roc_paths):
        fpr, tpr = np.load(roc_path)
        auc = np.load(roc_aucs[n])
        ## Loss Curves
        # Log ROC Curve
        _overplot(ax_roc, fpr, tpr, c=cmap[n], 
              ylabel="True Positive Rate", xlabel="False Positive Rate", 
              yscale='log', xscale='log', label=run_names[n]+'-AUC_{}'.format(np.around(auc, 3)))
    
    _overplot(ax_roc, [0, 1], [0, 1], label="Random Classifier", c='k', 
              ylabel="True Positive Rate", xlabel="False Positive Rate", 
              ls="dashed", yscale='log', xscale='log')
    
    fig_roc.savefig(os.path.join(save_dir, 'overlay_roc.png'))
    plt.close(fig_roc)
    

def snr_plotter(snr_dir, nepochs):
    snrs = np.loadtxt(snr_dir + '/snr_priors.csv')
    title = 'SNR Histogram over nepochs = {} with augmentation'.format(nepochs)
    xlabel = 'SNR'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(snr_dir, 'prior_SNR.png')
    _plotter(title, xlabel, ylabel, [snrs], ['Augmented'], save_path)


def debug_plotter(debug_dir):

    distance_new = np.loadtxt(debug_dir + '/save_augment_distance_new.txt')
    distance_old = np.loadtxt(debug_dir + '/save_augment_distance_old.txt')
    title = 'Comparing the old and new distance priors'
    xlabel = 'Distance'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'AUG_compare_distance.png')
    _plotter(title, xlabel, ylabel, [distance_old, distance_new], ['Original', 'Augmented'], save_path)
    
    pol = np.loadtxt(debug_dir + '/save_augment_pol.txt')
    title = 'Polarisation for 20 epochs of approx 1e4 signals each'
    xlabel = 'Polarisation Angle'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'AUG_pol_angle.png')
    _plotter(title, xlabel, ylabel, [pol], ['Pol Angle'], save_path)
    
    ra = np.loadtxt(debug_dir + '/save_augment_ra.txt')
    title = 'RA for 20 epochs of approx 1e4 signals each'
    xlabel = 'RA'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'AUG_ra.png')
    _plotter(title, xlabel, ylabel, [ra], ['ra'], save_path)
    
    dec = np.loadtxt(debug_dir + '/save_augment_dec.txt')
    title = 'DEC for 20 epochs of approx 1e4 signals each'
    xlabel = 'DEC'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'AUG_dec.png')
    _plotter(title, xlabel, ylabel, [dec], ['dec'], save_path)
    
    noise_slide_1 = np.loadtxt(debug_dir + '/save_augment_noise_shift_1.txt')
    title = 'Noise Slide 1 for 20 epochs of approx 1e4 signals each'
    xlabel = 'Noise Slide 1'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'AUG_noise_slide_1.png')
    _plotter(title, xlabel, ylabel, [noise_slide_1], ['noise_slide_1'], save_path)
    
    noise_slide_2 = np.loadtxt(debug_dir + '/save_augment_noise_shift_2.txt')
    title = 'Noise Slide 2 for 20 epochs of approx 1e4 signals each'
    xlabel = 'Noise Slide 2'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'AUG_noise_slide_2.png')
    _plotter(title, xlabel, ylabel, [noise_slide_2], ['noise_slide_2'], save_path)
    
    dchirp = np.loadtxt(debug_dir + '/save_augment_dchirp.txt')
    title = 'Chirp Distance for 20 epochs of approx 1e4 signals each'
    xlabel = 'Chirp Distance'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'AUG_dchirp.png')
    _plotter(title, xlabel, ylabel, [dchirp], ['dchirp'], save_path)
    
    noise_idx = np.loadtxt(debug_dir + '/save_augment_train_random_noise_idx.txt')
    title = 'Train Random noise idx for 20 epochs of approx 1e4 signals each'
    xlabel = 'Train Random noise idx'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'AUG_train_noise_idx.png')
    _plotter(title, xlabel, ylabel, [noise_idx], ['random_noise_idx'], save_path)
    
    noise_idx = np.loadtxt(debug_dir + '/save_augment_valid_random_noise_idx.txt')
    title = 'Valid Random noise idx for 20 epochs of approx 1e4 signals each'
    xlabel = 'Valid Random noise idx'
    ylabel = 'Number of Occurences'
    save_path = os.path.join(debug_dir, 'AUG_valid_noise_idx.png')
    _plotter(title, xlabel, ylabel, [noise_idx], ['random_noise_idx'], save_path)
