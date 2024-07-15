# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Feb  4 22:12:11 2022

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

# BUILT-IN
import os
import sys
import time
import glob
import torch
import errno
import random
import shutil
import operator
import itertools
import traceback
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from tqdm import tqdm
from scipy import signal
from collections import defaultdict
from distutils.dir_util import copy_tree
from contextlib import nullcontext

## RayTune for parameter tuning
from functools import partial
# RayTune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Turning off Torch debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
# Turning on cuDNN autotune
torch.backends.cudnn.benchmark = True
# Clear PyTorch Cache before init
torch.cuda.empty_cache()

import pygtc
from matplotlib import cm
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 18})

# LOCAL
from test import run_test
from evaluator import main as evaluator



def figure(title=""):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(16.0, 14.0))
    fig.suptitle(title, fontsize=28, y=0.92)
    return axs, fig


def _plot(ax, x=None, y=None, xlabel="x-axis", ylabel="y-axis", ls='solid', 
          label="NULL", c=None, yscale='linear', xscale='linear', histogram=False):
    
    # Plotting type
    if histogram:
        ax.hist(y, bins=100, label=label, alpha=0.8)
    else:
        ax.plot(x, y, ls=ls, c=c, linewidth=3.0, label=label)
    
    # Plotting params
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, which='both')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def display_outputs(training_output, training_labels):
    # Printing the training output and labels together
    # Depending on the batch_size, this output can be large
    for out, lab in zip(training_output, training_labels):
        print("output = {}, label = {}".format(out.detach().numpy(), lab.detach().numpy()))


def calculate_accuracy(output, labels, threshold = 0.5):
    # Calculate accuracy using training output and traning labels
    correct = 0
    apply_thresh = lambda x: round(x - threshold + 0.5)
    for toutput, tlabel in zip(output, labels):
        output_check = apply_thresh(float(toutput))
        labels_check = apply_thresh(float(tlabel))
        if output_check == labels_check:
            correct+=1

    accuracy = correct/len(output)
    return accuracy


def roc_curve(nep, output, labels, export_dir):
    # ROC Curve save plot
    save_dir = os.path.join(export_dir, 'ROC')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Calculating ROC
    fpr, tpr, threshold = metrics.roc_curve(labels, output)
    roc_auc = metrics.auc(fpr, tpr)
    
    # Plotting routine
    ax, _ = figure(title="ROC Curve at Epoch = {}".format(nep))
    
    # Log ROC Curve
    _plot(ax, fpr, tpr, label='AUC = %0.5f' % roc_auc, c='red', 
          ylabel="True Positive Rate", xlabel="False Positive Rate", 
          yscale='log', xscale='log')
    _plot(ax, [0, 1], [0, 1], label="Random Classifier", c='blue', 
          ylabel="True Positive Rate", xlabel="False Positive Rate", 
          ls="dashed", yscale='log', xscale='log')
    
    plt.legend()
    save_path = os.path.join(save_dir, "roc_curve_{}.png".format(nep))
    plt.savefig(save_path)
    plt.close()
    
    return (roc_auc, fpr, tpr)
    

def prediction_raw(nep, output, labels, export_dir):
    # Save (append) the above list of lists to CSV file
    save_dir = os.path.join(export_dir, "PRED_RAW")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Get raw values from output  
    foo = np.ma.masked_where(labels == 1.0, labels)
    noise_mask = ~foo.mask
    signal_mask = foo.mask
    raw_tn = output[noise_mask]
    raw_tp = output[signal_mask]
    
    # Plotting routine
    fig, ax = plt.subplots(1, 2, figsize=(12.0*2, 8.0*1))
    plt.suptitle('Raw Output at Epoch = {}'.format(nep))
    # Log pred probs
    _plot(ax[0], y=raw_tp, label="Signals",
          ylabel="log10 Number of Occurences", xlabel="Raw Output Values", 
          yscale='log', histogram=True)
    _plot(ax[0], y=raw_tn, label="Noise",
          ylabel="log10 Number of Occurences", xlabel="Raw Output Values", 
          yscale='log', histogram=True)
    
    # FAR counts
    sorted_noise_stats = np.sort(raw_tn)
    count_curve = []
    # Goes from biggest noise stat to smallest
    for thresh in sorted_noise_stats[::-1]:
        count_curve.append([thresh, len(raw_tp[raw_tp>thresh])/len(raw_tp)])
    # convert to numpy
    count_curve = np.array(count_curve)

    _plot(ax[1], x=count_curve[:,0], y=count_curve[:,1], xlabel="Noise Stat Threshold", 
          ylabel="Frac Signals Detected above Threshold", ls='solid', c='red', yscale='linear', 
          xscale='linear', histogram=False, label='1FAR/x = {}/{}'.format(int(count_curve[:,1][0]*len(raw_tp)), len(raw_tp)))
    
    plt.legend()
    # Add secondary axis and limits
    convert = lambda frac: frac * len(raw_tp)
    inverse = lambda length: length / len(raw_tp)
    secay = ax[1].secondary_yaxis('right', functions=(convert, inverse))
    secay.set_ylabel("Num Signals Detected above Threshold")
    ax[1].set_xlim(min(count_curve[:,0]), max(count_curve[:,0]))

    save_path = os.path.join(save_dir, "log_raw_output_{}.png".format(nep))
    plt.savefig(save_path)
    plt.close()

    return count_curve[:,1][0]*len(raw_tp)


def prediction_probability(nep, output, labels, export_dir):
    # Save (append) the above list of lists to CSV file
    save_dir = os.path.join(export_dir, "PRED_PROB")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Get pred probs from output
    mx = np.ma.masked_array(output, mask=labels)
    # For labels == signal, true positive
    pred_prob_tp = mx[mx.mask == True].data
    # For labels == noise, true negative
    pred_prob_tn = mx[mx.mask == False].data
    
    # Diffference between noise and signal stats
    boundary_diff = np.around(max(pred_prob_tp) - max(pred_prob_tn), 8)
    
    # Plotting routine
    ax, _ = figure(title="Pred prob output at Epoch = {}".format(nep))
    # Log pred probs
    _plot(ax, y=pred_prob_tp, label="Signals", c='red', 
          ylabel="log10 Number of Occurences", xlabel="Prediction Probabilities (Sigmoid)", 
          yscale='log', histogram=True)
    _plot(ax, y=pred_prob_tn, label="Noise", c='blue', 
          ylabel="log10 Number of Occurences", xlabel="Prediction Probabilities (Sigmoid)", 
          yscale='log', histogram=True)
    
    plt.legend()
    save_path = os.path.join(save_dir, "log_pred_prob_output_{}.png".format(nep))
    plt.savefig(save_path)
    plt.close()


def efficiency_curves(nep, source_params, predicted_outputs, labels, save_name='unknown_output', export_dir=''):
    # Save directory
    save_dir = os.path.join(export_dir, "EFFICIENCY/epoch_{}".format(nep))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Get pred probs from output
    mx = np.ma.masked_array(predicted_outputs, mask=labels)
    # For labels == signal, true positive
    data_tp = mx[mx.mask == True].data
    
    # Create overlapping bins for the source_params and get the average value of predicted outputs
    bin_width = 500 # samples
    overlap = 10 # samples
    for key in source_params.keys():
        # Sort the source_params for the particular key alongside the predicted outputs
        source_data = np.ma.masked_array(source_params[key])
        source_data = source_data[mx.mask == True].data
        assert len(source_data) == len(data_tp)
        zipped = zip(source_data, data_tp)

        zsorted = np.array(sorted(zipped, key=lambda x: x[0]))
        # Get the plotting data by averaging output in overlapping bins
        overlap_range = range(0, len(zsorted), overlap)
        plot = []
        for idx in overlap_range:
            bdata = np.where((zsorted[:,0]>zsorted[:,1][idx]) & (zsorted[:,0]<zsorted[:,1][idx+bin_width]))[0]
            plot.append(np.median(bdata), bin_width/len(bdata))
        plot = np.array(plot)
        # Plotting the above data for the given parameter
        ax, fig = figure(title="Efficiency Curve for {}".format(key))
        _plot(ax, x=plot[:,0], y=plot[:,1], xlabel=key, ylabel=save_name, ls='solid', 
              label=key, c='k', yscale='linear', xscale='linear', histogram=False)
        plt.legend()
        # Saving the plot in export_dir
        save_path = os.path.join(save_dir, 'efficiency_{}_{}_{}.png'.format(save_name, key, nep))
        plt.savefig(save_path)
        plt.close()
    

def learning_parameter_prior(nep, source_params, predicted_outputs, labels, save_name='unknown_output', export_dir=''):
    # Save directory
    save_dir = os.path.join(export_dir, "LEARN_PARAMS/epoch_{}".format(nep))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Get pred probs from output
    mx = np.ma.masked_array(predicted_outputs, mask=labels)
    # For labels == signal, true positive
    data_tp = mx[mx.mask == True].data
        
    # Create overlapping bins for the source_params and get the average value of predicted outputs
    bin_width = 500 # samples
    overlap = 10 # samples
    for key in source_params.keys():
        # Sort the source_params for the particular key alongside the predicted outputs
        source_data = np.ma.masked_array(source_params[key])
        source_data = source_data[mx.mask == True].data
        assert len(source_data) == len(data_tp)
        zipped = zip(source_data, data_tp)

        zsorted = np.array(sorted(zipped, key=lambda x: x[0]))
        # Get the plotting data by averaging output in overlapping bins
        overlap_range = range(0, len(zsorted), overlap)
        plot = [[zsorted[:,0][idx], np.mean(zsorted[:,1][idx:idx+bin_width])] for idx in overlap_range]
        plot = np.array(plot)
        # Plotting the above data for the given parameter
        ax, fig = figure(title="Learning {}".format(key))
        _plot(ax, x=plot[:,0], y=plot[:,1], xlabel=key, ylabel=save_name, ls='solid', 
              label=key, c='blue', yscale='linear', xscale='linear', histogram=False)
        # Plotting the percentiles
        plt.fill_between(plot[:,0], plot[:,1]-np.percentile(plot[:,1], 95),
                         plot[:,1]+np.percentile(plot[:,1], 95), color='red',
                         alpha = 0.3, label="95th percentile")
        plt.fill_between(plot[:,0], plot[:,1]-np.percentile(plot[:,1], 50),
                         plot[:,1]+np.percentile(plot[:,1], 50), color='green',
                         alpha = 0.3, label="50th percentile")
        plt.fill_between(plot[:,0], plot[:,1]-np.percentile(plot[:,1], 5), 
                         plot[:,1]+np.percentile(plot[:,1], 5), color='blue', 
                         alpha = 0.3, label="5th percentile")
        plt.legend()
        plt.xlim(min(plot[:,0]), max(plot[:,0]))
        # Saving the plot in export_dir
        save_path = os.path.join(save_dir, 'learning_{}_{}_{}.png'.format(save_name, key, nep))
        plt.savefig(save_path)
        plt.close()


def diagonal_compare(nep, outputs, labels, network_snrs, export_dir):
    # Save (append) the above list of lists to CSV file
    save_dir = os.path.join(export_dir, "DIAGONAL/epoch_{}".format(nep))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Mask function
    mask_function = lambda foo: 1 if foo>=0.0 else 0
    mask = [mask_function(foo) for foo in network_snrs]
    mx0 = np.ma.masked_array(network_snrs, mask=mask)
    
    # Colormap
    cmap = cm.get_cmap('RdYlBu_r', 10)
    
    for param in outputs.keys():
        if param == 'gw':
            continue
        # Plotting routine
        ax, fig = figure(title="Diagonal Plot of {} at Epoch = {}".format(param, nep))
        ax_snr_gt8, fig_gt8 = figure(title="Diagonal Plot of {} (SNR>8) at Epoch = {}".format(param, nep))
        # Plotting the observed value vs actual value scatter
        mx1 = np.ma.masked_array(outputs[param], mask=mask)
        mx2 = np.ma.masked_array(labels[param], mask=mask)
        # For labels == signal, true positive
        plot_output = mx1[mx1.mask == True].data
        plot_labels = mx2[mx2.mask == True].data
        plot_snrs = mx0[mx0.mask == True].data
        # Order the data
        data = np.stack((plot_output, plot_labels, plot_snrs), axis=1)
        data = data[data[:,2].argsort()]
        plot_output, plot_labels, plot_snrs = np.hsplit(data, data.shape[1])
        # Get rows where SNR > 8
        data_gt8 = data[data[:,2]>8.0]
        plot_output_gt8, plot_labels_gt8, plot_snrs_gt8 = np.hsplit(data_gt8, data_gt8.shape[1])
        # Plotting
        foo = ax.scatter(plot_output, plot_labels, marker='.', s=200.0, c=plot_snrs, cmap=cmap)
        bar = ax_snr_gt8.scatter(plot_output_gt8, plot_labels_gt8, marker='.', s=200.0, c=plot_snrs_gt8, cmap=cmap)
        # Colorbar
        cbar = fig.colorbar(foo)
        cbar.set_label('Network SNR', rotation=270, labelpad=40)
        cbar_snr_gt8 = fig_gt8.colorbar(bar)
        cbar_snr_gt8.set_label('Network SNR', rotation=270, labelpad=40)
        
        # Plotting params
        ax.grid(True, which='both')
        ax.set_xlabel('Observed Value [{}]'.format(param))
        ax.set_ylabel('Actual Value [{}]'.format(param))
        # Plotting the diagonal dashed line for reference
        if param != 'norm_dist' and param != 'norm_dchirp':
            _plot(ax, [0, 1], [0, 1], label="Best Classifier", c='k', 
                  ylabel='Actual Value [{}]'.format(param), 
                  xlabel='Observed Value [{}]'.format(param), ls="dashed")
        
        # Plotting params
        ax_snr_gt8.grid(True, which='both')
        ax_snr_gt8.set_xlabel('Observed Value [{}]'.format(param))
        ax_snr_gt8.set_ylabel('Actual Value [{}]'.format(param))
        # Plotting the diagonal dashed line for reference
        if param != 'norm_dist' and param != 'norm_dchirp':
            _plot(ax_snr_gt8, [0, 1], [0, 1], label="Best Classifier", c='k', 
                  ylabel='Actual Value [{}]'.format(param), 
                  xlabel='Observed Value [{}]'.format(param), ls="dashed")
        
        plt.legend()
        # Saving the plots
        save_path = os.path.join(save_dir, "diagonal_{}_{}.png".format(param, nep))
        save_path_gt8 = os.path.join(save_dir, "diagonal_snr_gt8_{}_{}.png".format(param, nep))
        fig.savefig(save_path)
        fig_gt8.savefig(save_path_gt8)
        plt.close(fig)
        plt.close(fig_gt8)


def plot_network_io(cfg, export_dir, plot_batch, training_output, training_labels, epoch, mode, extremes_io=False, snrs=None):
    # Plotting the network io of any layer used (provide non MR sample as input)
    dir_name = 'EXTREMES_IO' if extremes_io else 'NETWORK_IO'
    parent_dir = os.path.join(export_dir, dir_name)
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=False)
    # Training and validation dir
    save_dir = os.path.join(parent_dir, '{}/epoch_{}'.format(mode, epoch))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=False)

    # Outputs to be plotted
    normed = training_output['normed']
    network_input = training_output['input']
    raws = training_output['raw']
    pred_probs = training_output['pred_prob']
    cnn_output = training_output['cnn_output']

    for n, data in enumerate(zip(training_labels, raws, pred_probs, snrs, normed, network_input, cnn_output)):
        ## Analysing normed output of the pipeline
        # Convert all tensors to numpy arrays
        label, raw, pred_prob, snr, norm, netinpt, cnnopt = [foo.cpu().detach().numpy() for foo in data]
        # Checking for the effect of normalisation layers used
        nrows = 4 + len(plot_batch)
        pblen = len(plot_batch)
        fig, ax = plt.subplots(nrows, 2, figsize=(12.0*2, 8.0*nrows))
        raw = np.around(raw, 3)
        pred_prob = np.around(pred_prob, 3)
        
        plt.suptitle('raw={}, pred_prob={}, label={}, snr={}'.format(raw, pred_prob, label, snr))
        ndet = 2
        pbdat = [plot_batch[i][n] for i in range(pblen)]
        for i in range(ndet):
            # Input data without MR sampling
            for npb, pbi in enumerate(pbdat):
                ax[npb][i].plot(pbi[i], label='Transform {}'.format(npb))
                ax[npb][i].grid()
                ax[npb][i].legend()
            ## All plots after plot_batch
            # Spectrogram of input sample
            f, t, Sxx = signal.spectrogram(pbdat[2][i], 2048.)
            ax[pblen][i].pcolormesh(t, f, Sxx, shading='gouraud')
            ax[pblen][i].set_ylabel('Frequency [Hz]')
            ax[pblen][i].set_xlabel('Time [sec]')
            ax[pblen][i].grid()
            # Network input (after MR sampling)
            ax[pblen+1][i].plot(netinpt[i], label='Network input')
            ax[pblen+1][i].grid()   
            ax[pblen+1][i].legend() 
            # Normed output of the sample
            ax[pblen+2][i].plot(norm[i], label='Normed output')
            ax[pblen+2][i].grid()
            ax[pblen+2][i].legend()
            # CNN output
            ax[pblen+3][i].imshow(cnnopt[i])
            ax[pblen+3][i].grid()
        
        name = "noise" if label == 0.0 else "signal"
        savefile_name = 'extremes_io' if extremes_io else 'network_io'
        save_path = os.path.join(save_dir, '{}_{}_{}.png'.format(savefile_name, name, n))
        fig.subplots_adjust(top=0.95)
        plt.savefig(save_path)
        plt.close()


def outputbin_param_distribution(cfg, export_dir, network_output, labels, sample_params, epoch):
    ## Distribution of bin of network output, compared to entire distribution
    parent_dir = os.path.join(export_dir, 'OUTBIN_PARAM_DISTR')
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=False)
    save_dir = os.path.join(parent_dir, 'epoch_{}'.format(epoch))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=False)

    ## Split the network output distribution for signals into n bins
    foo = np.ma.masked_where(labels == 1.0, labels)
    noise_mask = ~foo.mask
    signal_mask = foo.mask
    noise_stats = network_output[noise_mask]
    signal_stats = network_output[signal_mask]
    # NOTE: Assuming that the distributions have overlap and min_signal < max_noise
    # NOTE: Expects raw values from network, not sigmoid
    # These are custom bin edges chosen for experimentation
    bin_ranges = [(min(signal_stats), 0.0), (min(signal_stats), max(noise_stats)), 
                 (0.0, max(noise_stats)), (max(noise_stats), max(signal_stats)),
                 (max(noise_stats), np.median(signal_stats[signal_stats>max(noise_stats)])),
                 (np.median(signal_stats[signal_stats>max(noise_stats)]), max(signal_stats)),
                 (0.0, max(signal_stats))]
    bin_names = ['(min signal stat to 0.0)', '(min signal stat to max noise stat)',
                 '(0.0 to max noise stat)', '(max noise stat to max signal stat)',
                 '(max noise stat to median signal stat above worst noise)',
                 '(median signal stat above worst noise to max signal stat)',
                 '(0.0 to max signal stat)']
    ## Using the indices of each bin, get the sample_params for each bin
    # Histogram kwargs
    kwargs = dict(histtype='stepfilled', alpha=0.5, bins=100)
    for n, bin_range in enumerate(bin_ranges):
        outdir = os.path.join(save_dir, 'bin_range_{}'.format(n))
        os.makedirs(outdir, exist_ok=False)
        idxs = np.argwhere((signal_stats > bin_range[0]) & (signal_stats < bin_range[1])).flatten()
        # Histogram details
        params = sample_params.keys()
        ncols = 3
        nrows = len(params)//ncols + int(len(params)%ncols or False)
        # Plotting
        fig, ax = plt.subplots(nrows, ncols, figsize=(8.0*ncols, 6.0*nrows))
        pidxs = list(itertools.product(range(nrows), range(ncols)))
        num_fin = 0
        for (param, distr), (i, j) in zip(sample_params.items(), pidxs):
            ## Plot the distribution of each param for given bin
            masked_distr = distr[signal_mask]
            ax[i][j].hist(masked_distr[idxs], label='binned', color='blue', **kwargs)
            ax[i][j].hist(masked_distr, label='all', color='red', **kwargs)
            ax[i][j].set_title(param)
            ax[i][j].grid(True)
            ax[i][j].legend()
            num_fin+=1
            
        for i, j in pidxs[num_fin:]:
            ax[i][j].set_visible(False)
        
        plt.tight_layout()
        save_name = "outbin_distr.png"
        save_path = os.path.join(outdir, save_name)
        plt.savefig(save_path)
        plt.close()

        ## Save a plot of the network output with the bin range highlighted
        ax, _ = figure(title="{}".format(bin_names[n]))
        # Signal raw
        _plot(ax, y=signal_stats, label="Signals",
              ylabel="log10 Number of Occurences", xlabel="Network Raw Output Values", 
              yscale='log', histogram=True)
        # Shade the required region
        plt.axvline(x=bin_range[0], color='k', ls=':', lw=3, label='lower bin edge = {}'.format(np.round(bin_range[0], 2)))
        plt.axvline(x=bin_range[1], color='k', ls='--', lw=3, label='upper bin edge = {}'.format(np.round(bin_range[1], 2)))
        # Save
        plt.legend()
        save_path = os.path.join(outdir, "binned_network_output.png")
        plt.savefig(save_path)
        plt.close()
        
        """
        ## Save the corner plot version of the above plot for each bin range
        chainLabels = ["Binned"]
        samples_binned = []
        samples_all = []
        names = []
        flag = 0
        for (param, distr) in sample_params.items():
            names.append(param)
            masked_distr = distr[signal_mask]
            samples_binned.append(masked_distr[idxs])
            samples_all.append(masked_distr)
            if len(masked_distr[idxs]) == 0:
                flag = 1
        
        if flag:
            continue
        # Make the corner plot and save
        samples_binned = np.stack(samples_binned, axis=0).T
        samples_all = np.stack(samples_all, axis=0).T
        names = np.array(names)

        save_path = os.path.join(outdir, "outbin_corner_plot.pdf")
        GTC = pygtc.plotGTC(chains=[samples_binned],
                            paramNames=names,
                            chainLabels=chainLabels,
                            figureSize='MNRAS_page',
                            plotName=save_path)
        """


def paramfrac_detected_above_thresh(cfg, export_dir, network_output, labels, sample_params, epoch):
    ## Fraction detected in bin of network param above a given noise threshold
    parent_dir = os.path.join(export_dir, 'PARAMFRAC_ABOVE_THRESH')
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=False)
    save_dir = os.path.join(parent_dir, 'epoch_{}'.format(epoch))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=False)

    ## Get noise and signal masks 
    foo = np.ma.masked_where(labels == 1.0, labels)
    noise_mask = ~foo.mask
    signal_mask = foo.mask
    noise_stats = network_output[noise_mask]
    sorted_noise_stats = np.sort(noise_stats)
    signal_stats = network_output[signal_mask]
    ## Make n bins of parameter space for each parameter
    # We need to ignore all values of -1
    for param, distr in sample_params.items():
        # Dir handling
        out_dir = os.path.join(save_dir, param)
        os.makedirs(out_dir, exist_ok=False)
        # Plotting routine
        fig, ax = plt.subplots(1, 2, figsize=(12.0*2, 8.0*1))
        plt.suptitle('Epoch {}: Fraction of binned {} above noise thresholds'.format(epoch, param))
        # Binned parameter distribution
        _plot(ax[0], y=distr[signal_mask], label="Signals",
              ylabel="Number of Occurences", xlabel="{} (epoch {})".format(param, epoch), 
              yscale='linear', histogram=True)

        # Get signal distr (remove -1's from noise)
        masked_distr = distr[signal_mask]
        # nbins is arbitrary
        count, bin_edges = np.histogram(masked_distr, bins=4)
        # Shade based on bin edges
        fill_count, _ = np.histogram(masked_distr, bins=100)
        border = 8
        y_min, y_max = 0.0, max(fill_count) + border
        # Random colours we like, woooo
        # Shoutout to ORChiD. woooo
        # ...Why am I like this?
        colours = ['gold', 'forestgreen', 'orchid', 'royalblue', 'orangered', 'gray']

        for n in range(len(bin_edges)-1):
            bin_range = (bin_edges[n], bin_edges[n+1])
            idxs = np.argwhere((masked_distr > bin_range[0]) & (masked_distr < bin_range[1])).flatten()
            ## Mapping to masked and binned network output
            binmasked_stats = signal_stats[idxs]
            binmasked_distr = masked_distr[idxs]
            # Sanity check
            if len(binmasked_stats) == 0:
                continue
            ## Find local fraction of signals above a given noise threshold
            count_curve = []
            # Goes from biggest noise stat to smallest
            for thresh in sorted_noise_stats[::-1]:
                count_curve.append([thresh, len(binmasked_stats[binmasked_stats>thresh])/len(binmasked_stats)])
            # convert to numpy
            count_curve = np.array(count_curve)
            # Shading the edges
            low = bin_range[0]
            up = bin_range[1]
            ax[0].fill([low, low, up, up], [y_min, y_max, y_max, y_min], color = colours[n], alpha = 0.3)
            ax[0].set_ylim(y_min, y_max)
            # Making the count plot
            _plot(ax[1], x=count_curve[:,0], y=count_curve[:,1], xlabel="Noise Stat Threshold", 
                  ylabel="Frac Signals Detected above Threshold", ls='solid', c=colours[n], yscale='linear', 
                  xscale='linear', histogram=False, label="{} to {}, 1FAR/x = {}/{}".format(np.round(bin_range[0], 2), np.round(bin_range[1], 2), 
                  int(count_curve[:,1][0]*len(binmasked_distr)), len(binmasked_distr)))

            # limits
            ax[1].set_xlim(min(count_curve[:,0]), max(count_curve[:,0]))

        # Saving plots
        plt.legend()
        save_path = os.path.join(out_dir, "paramfrac_{}.png".format(param))
        plt.savefig(save_path)
        plt.close()


def loss_and_accuracy_curves(cfg, filepath, export_dir, best_epoch=-1):
    # Read diagnostic file
    # option engine='python' required if sep > 1 character in length
    data = pd.read_csv(filepath, sep="    ", engine='python')
    if len(data['epoch']) == 1:
        return
    
    # All data fields
    epochs = data['epoch'] + 1.0
    training_loss = data['tot_tloss']
    validation_loss = data['tot_vloss']
    training_accuracy = data['train_acc']
    validation_accuracy = data['valid_acc']
    
    ## Loss Curves
    ax, _ = figure(title="Loss Curves")
    _plot(ax, epochs, training_loss, label="Training Loss", c='red')
    _plot(ax, epochs, validation_loss, label="Validation Loss", c='red', ls='dashed')
    if best_epoch != -1:
        ax.scatter(epochs[best_epoch], training_loss[best_epoch], marker='*', s=200.0, c='k')
        ax.scatter(epochs[best_epoch], validation_loss[best_epoch], marker='*', s=200.0, c='k')
    plt.legend()
    save_path = os.path.join(export_dir, "loss_curves.png")
    plt.savefig(save_path)
    plt.close()
    
    ## Parameter Estimation Loss Curves
    tsearch_names = [foo+'_tloss' for foo in cfg.parameter_estimation+('gw', )]
    vsearch_names = [foo+'_vloss' for foo in cfg.parameter_estimation+('gw', )]
    ax, _ = figure(title="PE Loss Curves")
    # Colour map
    numc = len(tsearch_names)
    cmap = ["#"+''.join([random.choice('ABCDEF0123456789') for _ in range(6)]) for _ in range(numc)]
    
    for n, (tsearch_name, vsearch_name) in enumerate(zip(tsearch_names, vsearch_names)):
        _plot(ax, epochs, data[tsearch_name], label=tsearch_name, c=cmap[n])
        _plot(ax, epochs, data[vsearch_name], label=vsearch_name, c=cmap[n], ls='dashed')
        # Marking the best epoch    
        if best_epoch != -1:
            ax.scatter(epochs[best_epoch], data[tsearch_name][best_epoch], marker='*', s=300.0, c='k')
            ax.scatter(epochs[best_epoch], data[vsearch_name][best_epoch], marker='*', s=300.0, c='k')
    
    plt.legend()
    save_path = os.path.join(export_dir, "pe_loss_curves.png")
    plt.savefig(save_path)
    plt.close()
    
    ## Accuracy Curves
    # Figure define
    ax, _ = figure(title="Accuracy Curves")
    _plot(ax, epochs, training_accuracy, label="Avg Training Accuracy", c='red', ylabel="Avg Accuracy")
    _plot(ax, epochs, validation_accuracy, label="Avg Validation Accuracy", c='blue', ylabel="Avg Accuracy")
    if best_epoch != -1:
        ax.scatter(epochs[best_epoch], training_accuracy[best_epoch], marker='*', s=300.0, c='k')
        ax.scatter(epochs[best_epoch], validation_accuracy[best_epoch], marker='*', s=300.0, c='k')
    plt.legend()
    save_path = os.path.join(export_dir, "accuracy_curves.png")
    plt.savefig(save_path)
    plt.close()


def batchshuffle_noise(training_labels, training_samples, nbatch, nep): 
    # Shuffle the samples between detectors (ONLY FOR PURE NOISE)
    # This procedure should **NOT** be done for signal samples
    noise_idx = np.argwhere(training_labels['gw'].numpy() == 0).flatten()
    if len(noise_idx) > 0:
        # assert len(noise_idx) > 0, "No noise samples found in this batch! Looks a bit sus."
        det1 = np.copy(noise_idx)
        det2 = np.copy(noise_idx)
        seed1 = int((nep+1)*1e6 + nbatch + 1)
        np.random.seed(seed1)
        np.random.shuffle(det1)
        seed2 = int((nep+1)*1e6 + nbatch)
        np.random.seed(seed2)
        np.random.shuffle(det2)
        assert seed1 != seed2, "Noise idx have not been shuffled properly. Seeds are the same for detectors!"
        shuffler = {foo: (d1, d2) for foo, d1, d2 in zip(noise_idx, det1, det2)}
        # Shuffle the noise samples within a given detector
        shuffled_training_samples = []
        for bidx in range(training_samples.shape[0]):
            if bidx in noise_idx:
                d1, d2 = shuffler[bidx]
                shuffled_training_samples.append([training_samples[d1][0].numpy(), training_samples[d2][1].numpy()])
            else:
                shuffled_training_samples.append([training_samples[bidx][0].numpy(), training_samples[bidx][1].numpy()])
                    
        # Convert the entire batch back to tensors
        shuffled_training_samples = np.array(shuffled_training_samples)
        shuffled_training_samples = torch.from_numpy(shuffled_training_samples)
        return shuffled_training_samples
    else:
        return training_samples


def consolidated_display(train_time, total_time):
    # Display wall clock time statistics
    print('\n----------------------------------------------------------------------')
    print('Total time taken to train epoch = {}s'.format(np.around(np.sum(train_time), 3)))
    print('Average time taken to train per batch = {}s'.format(np.around(np.mean(train_time), 3)))
    # Calculate MP load time
    load_time = total_time-np.sum(train_time)
    print('Total time taken to load (read & preprocess) epoch = {}s'.format(np.around(load_time, 3)))
    print('Total time taken = {}s'.format(np.around(total_time, 3)))
    print('----------------------------------------------------------------------')


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def training_phase(nep, cfg, data_cfg, Network, optimizer, scheduler, scheduler_step, 
                   loss_function, training_samples, training_labels, source_params,
                   optional, plot_batch, export_dir):
    # Optimizer step on a single batch of training data
    # Set zero_grad to apply None values instead of zeroes
    optimizer.zero_grad(set_to_none = True)
    
    with torch.cuda.amp.autocast() if cfg.do_AMP else nullcontext():
        # Obtain training output from network
        training_output = Network(training_samples)
        
        # Plotting cnn_output in debug mode
        if optional['network_io'] and cfg.model.__name__ in cfg.permitted_models:
            plot_network_io(cfg, export_dir, plot_batch, training_output, training_labels['gw'], nep, 'training', False, source_params['network_snr'])
            optional['network_io'] = False

        # Loss calculation
        all_losses = loss_function(training_output, training_labels, source_params, cfg.parameter_estimation)
        # Backward propogation using loss_function
        training_loss = all_losses['total_loss']
        training_loss.backward()
    
    # Accuracy calculation
    accuracy = calculate_accuracy(training_output['pred_prob'], training_labels['gw'], 0.5)
    # Clip gradients to make convergence somewhat easier
    torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=cfg.clip_norm)
    # Make the actual optimizer step and save the batch loss
    optimizer.step()
    # Scheduler step
    if scheduler and scheduler.__class__.__name__ not in ['ReduceLROnPlateau']:
        scheduler.step(scheduler_step)
    
    return (all_losses, accuracy)


def validation_phase(nep, cfg, data_cfg, Network, loss_function, validation_samples, validation_labels,
                     source_params, optional, plot_batch, export_dir):
    # Evaluation of a single validation batch
    with torch.cuda.amp.autocast() if cfg.do_AMP else nullcontext():
        # Gradient evaluation is not required for validation and testing
        # Make sure that we don't do a .backward() function anywhere inside this scope
        with torch.no_grad():
            validation_output = Network(validation_samples)
            # Plotting the normed outputs
            if optional['network_io'] and cfg.model.__name__ in cfg.permitted_models:
                plot_network_io(cfg, export_dir, plot_batch, validation_output, validation_labels['gw'], nep, 'validation', False, source_params['network_snr'])
                optional['network_io'] = False
            # Calculate validation loss with params if required
            all_losses = loss_function(validation_output, validation_labels, source_params, cfg.parameter_estimation)
    
    # Accuracy calculation
    accuracy = calculate_accuracy(validation_output['pred_prob'], validation_labels['gw'], 0.5)
    # Returning quantities if saving data
    return (all_losses, accuracy, validation_output)


def train(cfg, data_cfg, td, vd, Network, optimizer, scheduler, loss_function, trainDL, validDL, auxDL, nepoch, cflag, checkpoint, verbose=False):
    
    """
    Train a network on given data.
    
    Arguments
    ---------
    cfg : config object
        configuration options for the pipeline
    Network : network as returned by get_network
        The network to train.
    optimizer : object
        Optimizer object intialised with relevant params
    scheduler : object
        Scheduler object intialised with relevant params
    loss_function : object
        Loss function from torch or custom loss function
    trainDL : DataLoader object
        Training dataset dataloader with initialised batch_size
    validDL : DataLoader object
        Validation dataset dataloader with initialised batch_size
    verbose : {bool, False}
        Print update messages.
    
    Returns
    -------
    network
    
    """
    
    # Set up sanity check for raytune
    if cfg.rtune_optimise:
        run_id = time.strftime("%d%m%Y_%H-%M-%S")
        export_dir = os.path.join(cfg.export_dir, "run_{}".format(run_id))
    else:
        export_dir = cfg.export_dir

    # Save the run config
    # Move the config file to the BEST directory
    # shutil.copy("./configs.py", os.path.join(export_dir, 'configs.py'))
    # shutil.copy("./data_configs.py", os.path.join(export_dir, 'data_configs.py'))
    
    try:
        """ Training and Validation """
        ### Initialise global (over all epochs) params
        # Update if using checkpoint
        best_loss = checkpoint['loss'] if cfg.resume_from_checkpoint else 1.e10 # impossibly bad value
        best_accuracy = 0.0 # bad value
        best_roc_auc = 0.0 # bad value
        best_low_far_nsignals = 0 # bad value
        # Best params for weights
        best_params = {'loss': best_loss, 'accuracy': best_accuracy, 'roc_auc': best_roc_auc, 'low_far_nsignals': best_low_far_nsignals}

        # Other
        best_epoch = checkpoint['epoch'] if cfg.resume_from_checkpoint else 0
        best_roc_data = None
        overfitting_check = 0
        
        # Optional things to do during training 
        optional = {}
        
        # Setting the filepath to write epoch info
        loss_filepath = os.path.join(export_dir, 'losses.txt')
        
        start_epoch = int(checkpoint['epoch'])+1 if cfg.resume_from_checkpoint else 0
        for nep in range(start_epoch, cfg.num_epochs):
            
            # Update epoch number in dataset object for training and validation
            nepoch.value = nep

            # Update first batch plotting tasks toggler
            # TODO: Use this for validation phase as well
            if cfg.plot_on_first_batch:
                td.plot_on_first_batch = True
            
            # Prettification string
            epoch_string = "\n" + "="*65 + " Epoch {} ".format(nep) + "="*65
            print(epoch_string)
            
            # Training epoch
            Network.train()
            
            # Necessary save and update params
            training_running_loss = dict(zip(cfg.parameter_estimation, [0.0]*len(cfg.parameter_estimation)))
            training_running_loss.update({'total_loss': 0.0, 'gw': 0.0})
            training_batches = 0
            # Store accuracy params
            acc_train = []
            acc_valid = []
            
            
            """
            PHASE 1 - Training 
                [1] Do gradient clipping. Set value in cfg.
            """
            print("\nTraining Phase Initiated")
            pbar = tqdm(trainDL, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}', position=0, leave=True)
            
            # Total Number of batches
            num_train_batches = len(trainDL)
            # Recording the time taken for training
            start = time.time()
            train_times = []
            
            # Optional things to do during training
            optional['network_io'] = cfg.network_io
            # Sanity check
            if cfg.network_io:
                assert cfg.network_io and td.plot_on_first_batch, "NETWORK_IO option does not work without plot_on_first_batch (training)"
                assert cfg.network_io and vd.plot_on_first_batch, "NETWORK_IO option does not work without plot_on_first_batch (validation)"
            
            for nbatch, (training_samples, training_labels, source_params, plot_batch) in enumerate(pbar):
                
                # BatchShuffle Noise samples for an extra dimension of augmentation
                if cfg.batchshuffle_noise:
                    training_samples = batchshuffle_noise(training_labels, training_samples, nbatch, nep)
                
                # Update params
                if scheduler:
                    scheduler_step = nep + nbatch / num_train_batches
                else:
                    scheduler_step = None
                
                # Record time taken for training
                start_train = time.time()
                
                ## Class balance assertions
                ## NOTE: This may be too stringent for small batch sizes
                ##       Also, the variance in dataset balance might help training. 
                # batch_labels = training_labels['gw'].numpy()
                # check_balance = len(batch_labels[batch_labels == 1])/len(batch_labels)
                # assert check_balance >= 0.30 and check_balance <= 0.70
                
                """ Tensorification and Device Compatibility """
                ## Performing this here rather than in the Dataset object
                ## will reduce the overhead of having to move each sample to CUDA 
                ## rather than moving a batch of data.
                # Set the device and dtype
                training_samples = training_samples.to(dtype=torch.float32, device=cfg.train_device)
                for key, value in training_labels.items():
                    training_labels[key] = value.to(dtype=torch.float32, device=cfg.train_device)
                
                """ Call Training Phase """
                # Run training phase and get loss and accuracy
                training_loss, accuracy = training_phase(nep, cfg, data_cfg, Network, 
                                                         optimizer, 
                                                         scheduler, scheduler_step,
                                                         loss_function, 
                                                         training_samples, training_labels,
                                                         source_params, optional,
                                                         plot_batch, export_dir)
                
                ## Display stuff
                loss = np.around(training_loss['total_loss'].clone().cpu().item(), 8)
                # Update pbar with loss, acc
                pbar.set_description("Epoch {}, batch {} - loss = {}".format(nep, nbatch, loss))
                # Stop all first batch processes
                td.plot_on_first_batch = False
                # Update losses and accuracy
                training_batches = nbatch
                acc_train.append(accuracy)
                for key in training_running_loss.keys():
                    training_running_loss[key] += training_loss[key].clone().cpu().item()
                # Record time taken to load data (calculate avg time later)
                train_times.append(time.time() - start_train)
            
            
            ## Time taken to train data
            # Total time taken for training phase
            total_time = time.time() - start
            
            consolidated_display(train_times, total_time)
            
            
    
            """
            PHASE 2 - Validation
                [1] Save confusion matrix elements and prediction probabilties
                [2] Save the ROC save data
            """            
            print("\nValidation Phase Initiated")
            # Evaluation on the validation dataset
            Network.eval()
            with torch.no_grad():

                # Optional things to do during training
                optional['network_io'] = cfg.network_io
                optional['extremes_io'] = cfg.extremes_io

                # Extremes IO
                if optional['extremes_io']:
                    best_epoch_samples = {'signal': {'stat': torch.tensor(-999_999.0)}, 'noise': {'stat': torch.tensor(999_999.0)}}
                    worst_epoch_samples = {'signal': {'stat': torch.tensor(999_999.0)}, 'noise': {'stat': torch.tensor(-999_999.0)}}
                
                # Update validation loss dict
                validation_running_loss = dict(zip(cfg.parameter_estimation, [0.0]*len(cfg.parameter_estimation)))
                validation_running_loss.update({'total_loss': 0.0, 'gw': 0.0})
                validation_batches = 0
                
                # Other params for plotting and logging
                # Creating a defaultdict of lists (logging outputs)
                dd = defaultdict(list)
                epoch_labels = {foo: np.array([]) for foo in cfg.parameter_estimation + ('gw', )}
                epoch_outputs = {foo: np.array([]) for foo in cfg.parameter_estimation + ('gw', )}
                sample_params = None
                raw_output = np.array([])
                
                pbar = tqdm(validDL, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}', position=0, leave=True)
                for nbatch, (validation_samples, validation_labels, source_params, plot_batch) in enumerate(pbar):
                    
                    """ Set the device and dtype """
                    validation_samples = validation_samples.to(dtype=torch.float32, device=cfg.train_device)
                    for key, value in validation_labels.items():
                        validation_labels[key] = value.to(dtype=torch.float32, device=cfg.train_device)
                        
                    """ Call Validation Phase """
                    # Run training phase and get loss and accuracy
                    validation_loss, accuracy, voutput = validation_phase(nep, cfg, data_cfg, Network, 
                                                                          loss_function,
                                                                          validation_samples, 
                                                                          validation_labels, source_params,
                                                                          optional, plot_batch, export_dir)

                    # Display stuff
                    loss = np.around(validation_loss['total_loss'].clone().cpu().item(), 8)
                    pbar.set_description("Epoch {}, batch {} - loss = {}".format(nep, validation_batches, loss))
                    
                    # Update losses and accuracy
                    validation_batches = nbatch
                    acc_valid.append(accuracy)
                    for key in validation_running_loss.keys():
                        validation_running_loss[key] += validation_loss[key].clone().cpu().item()
                    
                    ## Update extremes
                    if optional['extremes_io']:
                        enum = validation_labels['gw'].new_tensor(np.arange(len(validation_labels['gw'])))
                        noise_mask = torch.eq(validation_labels['gw'], 0.0)
                        signal_mask = torch.eq(validation_labels['gw'], 1.0)
                        noise_stats = torch.masked_select(voutput['raw'], noise_mask)
                        signal_stats = torch.masked_select(voutput['raw'], signal_mask)
                        ## Signal stats
                        if len(signal_stats)!=0:
                            # Save signal stats
                            if torch.max(signal_stats) > best_epoch_samples['signal']['stat']:
                                maxarg = int(torch.masked_select(enum, signal_mask)[torch.argmax(signal_stats)].item())
                                best_epoch_samples['signal']['stat'] = torch.max(signal_stats)
                                best_epoch_samples['signal']['plot_batch'] = [foo[maxarg:maxarg+1] for foo in plot_batch]
                                best_epoch_samples['signal']['vout_slice'] = slice(maxarg, maxarg+1)
                                best_epoch_samples['signal']['vlabels'] = validation_labels['gw'][maxarg:maxarg+1]
                                best_epoch_samples['signal']['snr'] = source_params['network_snr'][maxarg:maxarg+1]
                            
                            if torch.min(signal_stats) < worst_epoch_samples['signal']['stat']:
                                minarg = int(torch.masked_select(enum, signal_mask)[torch.argmin(signal_stats)].item())
                                worst_epoch_samples['signal']['stat'] = torch.min(signal_stats)
                                worst_epoch_samples['signal']['plot_batch'] = [foo[minarg:minarg+1] for foo in plot_batch]
                                worst_epoch_samples['signal']['vout_slice'] = slice(minarg, minarg+1)
                                worst_epoch_samples['signal']['vlabels'] = validation_labels['gw'][minarg:minarg+1]
                                worst_epoch_samples['signal']['snr'] = source_params['network_snr'][minarg:minarg+1]

                        ## Noise stats
                        # Possible error: RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0. 
                        # Specify the reduction dim with the 'dim' argument.
                        # This error means that the argument to min or max is empty.
                        if len(noise_stats)!=0:
                            if torch.min(noise_stats) < best_epoch_samples['noise']['stat']: 
                                minarg = int(torch.masked_select(enum, noise_mask)[torch.argmin(noise_stats)].item())
                                best_epoch_samples['noise']['stat'] = torch.min(noise_stats)
                                best_epoch_samples['noise']['plot_batch'] = [foo[minarg:minarg+1] for foo in plot_batch]
                                best_epoch_samples['noise']['vout_slice'] = slice(minarg, minarg+1)
                                best_epoch_samples['noise']['vlabels'] = validation_labels['gw'][minarg:minarg+1]
                                best_epoch_samples['noise']['snr'] = source_params['network_snr'][minarg:minarg+1]

                            if torch.max(noise_stats) > worst_epoch_samples['noise']['stat']: 
                                maxarg = int(torch.masked_select(enum, noise_mask)[torch.argmax(noise_stats)].item())
                                worst_epoch_samples['noise']['stat'] = torch.max(noise_stats)
                                worst_epoch_samples['noise']['plot_batch'] = [foo[maxarg:maxarg+1] for foo in plot_batch]
                                worst_epoch_samples['noise']['vout_slice'] = slice(maxarg, maxarg+1)
                                worst_epoch_samples['noise']['vlabels'] = validation_labels['gw'][maxarg:maxarg+1]
                                worst_epoch_samples['noise']['snr'] = source_params['network_snr'][maxarg:maxarg+1]


                    # Params for storing labels and outputs
                    if nep % cfg.validation_plot_freq == 0:
                        
                        # Move labels and outputs from cuda to cpu
                        # Detach to remove gradient information from tensors (is this required?)
                        raw_output = np.concatenate([raw_output, voutput['raw'].cpu()])
                        epoch_labels['gw'] = np.concatenate([epoch_labels['gw'], validation_labels['gw'].cpu()])
                        epoch_outputs['gw'] = np.concatenate([epoch_outputs['gw'], voutput['pred_prob'].cpu()])
                        
                        # Update source params
                        # Since the keys of source_params are not deterministic
                        for key in source_params.keys():
                            dd[key].append(source_params[key].cpu())
                        
                        # Add parameter estimation actual and observed values
                        for param in cfg.parameter_estimation:
                            epoch_labels[param] = np.concatenate([epoch_labels[param], validation_labels[param].cpu()])
                            epoch_outputs[param] = np.concatenate([epoch_outputs[param], voutput[param].cpu()])
                
                # Convert source params into dictionary of np.ndarray
                sample_params = dict((key, np.concatenate(tuple(val))) for key, val in dd.items()) 


                if nep % cfg.validation_plot_freq == 0:

                    """ ROC Curve save data """
                    roc_auc, fpr, tpr = roc_curve(nep, epoch_outputs['gw'], epoch_labels['gw'], export_dir)
                    
                    """ Confusion Matrix and FPR,TPR,FNR,TNR Evolution """
                    # Confusion matrix and rate evolution plots have been deprecated as of June 10, 2022
                    # Confusion matrix and rate evolution plots have been reinstated as of May 09, 2023

                    """ Prediction Probabilitiy OR Raw Value histograms  """
                    low_far_nsignals = prediction_raw(nep, raw_output, epoch_labels['gw'], export_dir)
                    prediction_probability(nep, epoch_outputs['gw'], epoch_labels['gw'], export_dir)
                    
                    """ Source parameter vs prediction probabilities OR raw values """
                    learning_parameter_prior(nep, sample_params, raw_output, epoch_labels['gw'], 'raw_value', export_dir)
                    learning_parameter_prior(nep, sample_params, epoch_outputs['gw'], epoch_labels['gw'], 'pred_prob', export_dir)
                    
                    """ Value VS Value Plots for PE """
                    diagonal_compare(nep, epoch_outputs, epoch_labels, sample_params['network_snr'], export_dir)

                    """ Extremes IO """
                    if optional['extremes_io']:
                        tmp = {'best_signal': {}, 'best_noise': {}, 'worst_signal': {}, 'worst_noise': {}}
                        # Used to store voutput dictionary at specific idx
                        for key, value in voutput.items():
                            tmp['best_signal'][key] = value[best_epoch_samples['signal']['vout_slice']]
                            tmp['best_noise'][key] = value[best_epoch_samples['noise']['vout_slice']]
                            tmp['worst_signal'][key] = value[worst_epoch_samples['signal']['vout_slice']]
                            tmp['worst_noise'][key] = value[worst_epoch_samples['noise']['vout_slice']]
                        
                        # BEST
                        plot_network_io(cfg, export_dir, best_epoch_samples['signal']['plot_batch'], tmp['best_signal'], 
                                        best_epoch_samples['signal']['vlabels'], nep, 'best_signal', extremes_io=True, 
                                        snrs=best_epoch_samples['signal']['snr'])
                        plot_network_io(cfg, export_dir, best_epoch_samples['noise']['plot_batch'], tmp['best_noise'], 
                                        best_epoch_samples['noise']['vlabels'], nep, 'best_noise', extremes_io=True,
                                        snrs=best_epoch_samples['noise']['snr'])
                        # WORST
                        plot_network_io(cfg, export_dir, worst_epoch_samples['signal']['plot_batch'], tmp['worst_signal'], 
                                        worst_epoch_samples['signal']['vlabels'], nep, 'worst_signal', extremes_io=True,
                                        snrs=worst_epoch_samples['signal']['snr'])
                        plot_network_io(cfg, export_dir, worst_epoch_samples['noise']['plot_batch'], tmp['worst_noise'], 
                                        worst_epoch_samples['noise']['vlabels'], nep, 'worst_noise', extremes_io=True,
                                        snrs=worst_epoch_samples['noise']['snr'])
                    
                    """ Param distribution of network raw output bins """
                    outputbin_param_distribution(cfg, export_dir, raw_output, epoch_labels['gw'], sample_params, nep)
                    
                    """ Param Fraction Plot """
                    paramfrac_detected_above_thresh(cfg, export_dir, raw_output, epoch_labels['gw'], sample_params, nep)
                    
                    """ Efficiency Curves """
                    # efficiency_curves(nep, sample_params, epoch_outputs['gw'], epoch_labels['gw'], save_name='efficiency_validation', export_dir=export_dir)

            
            ### Auxiliary validation set checks
            print('\nAUX validation phase')
            with torch.no_grad():
                
                aux_val_running_loss = {'aux_0': 0.0, 'aux_1': 0.0, 'aux_2': 0.0, 'aux_3': 0.0}
                for cf in range(4):
                    # Change AUX label in dataloader
                    cflag.value = cf

                    # Update validation loss dict
                    aux_val_batches = 0

                    print('AUX validation dataset {}'.format(cf))
                    pbar = tqdm(auxDL, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}', position=0, leave=True)
                    for nbatch, (aux_samples, aux_labels, source_params, plot_batch) in enumerate(pbar):
                        
                        """ Set the device and dtype """
                        aux_samples = aux_samples.to(dtype=torch.float32, device=cfg.train_device)
                        for key, value in aux_labels.items():
                            aux_labels[key] = value.to(dtype=torch.float32, device=cfg.train_device)
                            
                        """ Call Validation Phase """
                        # Run training phase and get loss and accuracy
                        aux_loss, accuracy, _ = validation_phase(nep, cfg, data_cfg, Network, 
                                                                 loss_function,
                                                                 aux_samples, 
                                                                 aux_labels, source_params,
                                                                 optional, plot_batch, export_dir)

                        # Display stuff
                        loss = np.around(aux_loss['total_loss'].clone().cpu().item(), 8)
                        pbar.set_description("AUX Epoch {}, batch {} - loss = {}".format(nep, aux_val_batches, loss))
                        
                        # Update losses and accuracy
                        aux_val_batches = nbatch
                        aux_val_running_loss['aux_{}'.format(cf)] += aux_loss['gw'].clone().cpu().item()
                
                cflag.value = -1



            """
            PHASE 3 - TESTING
                [1] Run the network using the epoch weights on a small testing dataset
                [2] We can make an AUC for the sensitive distance curve and plot them
            """
            # Testing the network with the current epoch weights
            if cfg.epoch_testing:
                print("\nEpoch-Testing Phase Initiated")
                # Make epoch testing dir
                epoch_testing_dir = os.path.join(cfg.export_dir, 'EPOCH_TESTING')
                if not os.path.exists(epoch_testing_dir):
                    os.makedirs(epoch_testing_dir, exist_ok=False)
                    
                # Running the testing phase for foreground data
                transforms = cfg.transforms['test']
                jobs = ['foreground', 'background']
                
                output_testing_dir = os.path.join(epoch_testing_dir, 'epoch_testing_{}'.format(nep))
                for job in jobs:
                    # Get the required data based on testing job
                    if job == 'foreground':
                        testfile = os.path.join(cfg.epoch_testing_dir, cfg.test_foreground_dataset)
                        evalfile = os.path.join(output_testing_dir, cfg.test_foreground_output)
                    elif job == 'background':
                        testfile = os.path.join(cfg.epoch_testing_dir, cfg.test_background_dataset)
                        evalfile = os.path.join(output_testing_dir, cfg.test_background_output)
                        
                    print('\nRunning epoch testing on {} data'.format(job))
                    run_test(Network, testfile, evalfile, transforms, cfg, data_cfg,
                             step_size=cfg.step_size, slice_length=int(data_cfg.signal_length*data_cfg.sample_rate),
                             trigger_threshold=cfg.trigger_threshold, cluster_threshold=cfg.cluster_threshold, 
                             batch_size = cfg.batch_size,
                             device=cfg.testing_device, verbose=cfg.verbose)
            
                # Run the evaluator for the testing phase and add required files to TESTING dir in export_dir
                raw_args =  ['--injection-file', os.path.join(cfg.epoch_testing_dir, cfg.injection_file)]
                raw_args += ['--foreground-events', os.path.join(output_testing_dir, cfg.test_foreground_output)]
                raw_args += ['--foreground-files', os.path.join(cfg.epoch_testing_dir, cfg.test_foreground_dataset)]
                raw_args += ['--background-events', os.path.join(output_testing_dir, cfg.test_background_output)]
                out_eval = os.path.join(output_testing_dir, cfg.evaluation_output)
                raw_args += ['--output-file', out_eval]
                raw_args += ['--output-dir', output_testing_dir]
                raw_args += ['--verbose']
                
                # Running the evaluator to obtain output triggers (with clustering)
                evaluator(raw_args, cfg_far_scaling_factor=float(cfg.far_scaling_factor), dataset=data_cfg.dataset)
            
            
            """
            PHASE 4 - Save
                [1] Save losses, accuracy and confusion matrix elements
                [2] Save the best model weights path if global loss is reduced
                [3] Reload the new weights once all phases are complete
            """
            # Print information on the training and validation loss in the current epoch and 
            # save current network state
            epoch_validation_loss = {}
            epoch_training_loss = {}
            for key in training_running_loss.keys():
                _key = "tot" if key == "total_loss" else key
                epoch_validation_loss[_key] = np.around(validation_running_loss[key]/validation_batches, 8)
                epoch_training_loss[_key] = np.around(training_running_loss[key]/training_batches, 8)
            
            avg_acc_valid = np.around(sum(acc_valid)/len(acc_valid), 8)
            avg_acc_train = np.around(sum(acc_train)/len(acc_train), 8)
            roc_auc = np.around(roc_auc, 8)
            # Auxilliary validation losses
            aux_0 = np.around(aux_val_running_loss['aux_0']/aux_val_batches, 8)
            aux_1 = np.around(aux_val_running_loss['aux_1']/aux_val_batches, 8)
            aux_2 = np.around(aux_val_running_loss['aux_2']/aux_val_batches, 8)
            aux_3 = np.around(aux_val_running_loss['aux_3']/aux_val_batches, 8)
            
            # Save args
            args = (nep, )
            args += tuple(np.around(foo, 8) for foo in epoch_training_loss.values())
            args += tuple(np.around(foo, 8) for foo in epoch_validation_loss.values())
            args += (avg_acc_train, avg_acc_valid, roc_auc, aux_0, aux_1, aux_2, aux_3)
            output_string = '{}    ' * len(args)
            output_string = output_string.format(*args)
            
            # Add the field names to file header
            if not os.path.exists(loss_filepath):
                add_fieldnames = True
            else:
                add_fieldnames = False
            
            with open(loss_filepath, 'a') as outfile:
                # Add optional fieldnames
                if add_fieldnames:
                    field_names = "{}    " * len(args)
                    epoch_field = ('epoch', )
                    tloss_fields = tuple(epoch_training_loss.keys())
                    tloss_fields = tuple(foo+'_tloss' for foo in tloss_fields)
                    vloss_fields = tuple(epoch_validation_loss.keys())
                    vloss_fields = tuple(foo+'_vloss' for foo in vloss_fields)
                    other_fields = ('train_acc', 'valid_acc', 'roc_auc', 'aux_0', 'aux_1', 'aux_2', 'aux_3')
                    all_fields = epoch_field + tloss_fields + vloss_fields + other_fields
                    field_names = field_names.format(*all_fields)
                    outfile.write(field_names + '\n')
                # Save output string in losses.txt
                outfile.write(output_string + '\n')
            
            loss_and_accuracy_curves(cfg, loss_filepath, export_dir)
            
            """ Save Checkpoint """
            # Create Checkpoint
            if cfg.save_checkpoint and nep%cfg.checkpoint_freq==0:
                save_dir = os.path.join(export_dir, 'CHECKPOINTS')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=False)
                checkpoint_save_path = os.path.join(save_dir, 'checkpoint_epoch_{}.pt'.format(nep))
                torch.save({
                            'epoch': nep,
                            'model_state_dict': Network.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': epoch_validation_loss['gw'],
                            }, checkpoint_save_path)
            
            """ Save the best weights (if global loss reduces) """
            split_weights_path = os.path.splitext(cfg.weights_path)
            weights_root_name = 'weights'
            weights_file_ext = split_weights_path[1]
            # NOTE: Best loss is considered to be when the GW classification loss is lowest
            epoch_params = {'loss': epoch_validation_loss['gw'], 'accuracy': avg_acc_valid, 'roc_auc': roc_auc, 'low_far_nsignals': low_far_nsignals}
            param_operator = {'loss': operator.lt, 'accuracy': operator.gt, 'roc_auc': operator.gt, 'low_far_nsignals': operator.gt}
            
            for wtype in cfg.weight_types:
                if param_operator[wtype](epoch_params[wtype], best_params[wtype]) and wtype in cfg.weight_types:
                    weights_path = '{}_{}{}'.format(weights_root_name, wtype, weights_file_ext)
                    weights_save_path = os.path.join(export_dir, weights_path)
                    torch.save({
                        'epoch': nep,
                        'model_state_dict': Network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metric': epoch_params[wtype],
                        }, weights_save_path)
                    if cfg.save_best_option == wtype:
                        best_epoch = nep

            # If we want to save the weights at a particular epoch
            if nep in cfg.save_epoch_weight:
                weights_path = '{}_{}_{}{}'.format(weights_root_name, wtype, nep, weights_file_ext)
                weights_save_path = os.path.join(export_dir, weights_path)
                torch.save(Network.state_dict(), weights_save_path)

            # Update best params irrespective of chosen weight types
            if epoch_params['loss'] < best_params['loss']:
                best_params['loss'] = epoch_params['loss']
            if epoch_params['accuracy'] > best_params['accuracy']:
                best_params['accuracy'] = epoch_params['accuracy']
            if epoch_params['roc_auc'] > best_params['roc_auc']:
                best_params['roc_auc'] = epoch_params['roc_auc']
            if epoch_params['low_far_nsignals'] > best_params['low_far_nsignals']:
                best_params['low_far_nsignals'] = epoch_params['low_far_nsignals']

            """ Update Scheduler """
            # ReduceLRonPlateau requires scheduler to be updated with epoch validation loss
            if scheduler.__class__.__name__ in ['ReduceLROnPlateau']:
                scheduler.step(epoch_params['loss'])

            """ RayTune Logging """
            if cfg.rtune_optimise: 
                with tune.checkpoint_dir(nep) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((Network.state_dict(), optimizer.state_dict()), path)
            
                tune.report(loss=epoch_validation_loss['gw'], accuracy=avg_acc_valid)
            
            
            """ Epoch Display """
            print("\nBest Validation Loss (wrt all past epochs) = {}".format(best_params['loss']))
            
            print('\n----------------------------------------------------------------------')
            print("Average losses in Validation Phase:")
            print("Total Loss = {}".format(epoch_validation_loss['tot']))
            print("1. GW Loss = {}".format(epoch_validation_loss['gw']))
            for n, param in enumerate(cfg.parameter_estimation):
                print("{}. {} Loss = {}".format(n+2, param, epoch_validation_loss[param]))
            
            print('----+----+----+----+----+----+----+----+----')
            print("Average losses in Training Phase:")
            print("Total Loss = {}".format(epoch_training_loss['tot']))
            print("1. GW Loss = {}".format(epoch_training_loss['gw']))
            for n, param in enumerate(cfg.parameter_estimation):
                print("{}. {} Loss = {}".format(n+2, param, epoch_training_loss[param]))
            print('----------------------------------------------------------------------')
            
            print("\nAverage Validation Accuracy = {}".format(avg_acc_valid))
            print("Average Training Accuracy = {}".format(avg_acc_train))
            print("ROC Area Under the Curve (ROC-AUC) = {}".format(roc_auc))

            """ Stop run if stop folder made in export directory """
            stop_dir = os.path.join(export_dir, 'stop')
            if os.path.exists(stop_dir):
                print('\nTerminating pipeline: Logs and outputs will be moved to the export dir.')
                os.remove(stop_dir)
                break
    
    
    except KeyboardInterrupt:
        print(traceback.format_exc())
        print('manual.py: Terminated due to user controlled KeyboardInterrupt.')
    
        
    print("\n==========================================================================\n")
    print("Training Complete!")
    print("Best validation loss = {}".format(best_params['loss']))
    print("Best validation accuracy = {}".format(best_params['accuracy']))
    print("Best ROC AUC = {}".format(best_params['roc_auc']))
    print()
    
    try:
        # Saving best epoch results
        best_dir = os.path.join(export_dir, 'BEST')
        if not os.path.isdir(best_dir):
            os.makedirs(best_dir, exist_ok=False)
        
        # Move premade plots
        roc_dir = 'ROC'
        roc_file = "roc_curve_{}.png".format(best_epoch)
        roc_path = os.path.join(export_dir, os.path.join(roc_dir, roc_file))
        shutil.copy(roc_path, os.path.join(best_dir, roc_file))
        
        # Best prediction probabilities
        pred_dir = 'PRED_RAW'
        pred_file = "log_raw_output_{}.png".format(best_epoch)
        pred_path = os.path.join(export_dir, os.path.join(pred_dir, pred_file))
        shutil.copy(pred_path, os.path.join(best_dir, pred_file))
        
        # Best prediction probabilities
        pred_dir = 'PRED_PROB'
        pred_file = "log_pred_prob_output_{}.png".format(best_epoch)
        pred_path = os.path.join(export_dir, os.path.join(pred_dir, pred_file))
        shutil.copy(pred_path, os.path.join(best_dir, pred_file))
        
        permitted_models = ['KappaModel', 'KappaModelPE']
        if cfg.debug and cfg.model.__name__ in permitted_models:
            # Move best CNN features
            src_best_features = os.path.join(export_dir, 'CNN_OUTPUT/epoch_{}'.format(best_epoch))
            dst_best_features = os.path.join(best_dir, 'CNN_features_epoch_{}'.format(best_epoch))
            copy_tree(src_best_features, dst_best_features)
        
        # Move best diagonal plots
        src_best_diagonals = os.path.join(export_dir, 'DIAGONAL/epoch_{}'.format(best_epoch))
        dst_best_diagonals = os.path.join(best_dir, 'diagonal_epoch_{}'.format(best_epoch))
        copy_tree(src_best_diagonals, dst_best_diagonals)
        # Move network_io and learn params
        if cfg.network_io:
            src_best_networkio = os.path.join(export_dir, 'NETWORK_IO/training/epoch_{}'.format(best_epoch))
            dst_best_networkio = os.path.join(best_dir, 'network_io/training/epoch_{}'.format(best_epoch))
            copy_tree(src_best_networkio, dst_best_networkio)
            src_best_networkio = os.path.join(export_dir, 'NETWORK_IO/validation/epoch_{}'.format(best_epoch))
            dst_best_networkio = os.path.join(best_dir, 'network_io/validation/epoch_{}'.format(best_epoch))
            copy_tree(src_best_networkio, dst_best_networkio)
        # Learn params
        src_best_learn = os.path.join(export_dir, 'LEARN_PARAMS/epoch_{}'.format(best_epoch))
        dst_best_learn = os.path.join(best_dir, 'learn_params_epoch_{}'.format(best_epoch))
        copy_tree(src_best_learn, dst_best_learn)
        
        # Remake loss curve and accuracy curve with best epoch marked
        loss_and_accuracy_curves(cfg, loss_filepath, best_dir, best_epoch=best_epoch)
        
        # Move weights to BEST dir
        weights_save_paths = glob.glob(os.path.join(export_dir, '*.pt'))
        for wpath in weights_save_paths:
            if os.path.exists(wpath):
                foo = os.path.split(wpath)
                shutil.move(wpath, os.path.join(best_dir, foo[1]))
            else:
                pass
                
        # Return the trained network with the best possible weights
        # This step is mandatory before the inference/testing module
        split_weights_path = os.path.splitext(cfg.weights_path)
        weights_root_name = 'weights'
        weights_file_ext = split_weights_path[1]
        weights_path = '{}_{}{}'.format(weights_root_name, cfg.save_best_option, weights_file_ext)
        if os.path.exists(weights_path):
            Network.load_state_dict(torch.load(os.path.join(best_dir, weights_path)))
        else:
            Network = None

    except Exception as e:
        foo = traceback.format_exception(*sys.exc_info())
        print("Received a {} error while saving data in BEST directory.".format(e))
    
    return Network
