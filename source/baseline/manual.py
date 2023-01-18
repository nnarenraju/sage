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
import time
import glob
import torch
import random
import shutil
import traceback
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from tqdm import tqdm
from distutils.dir_util import copy_tree

# Turning off Torch debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
# Turning on cuDNN autotune
torch.backends.cudnn.benchmark = True
# Clear PyTorch Cache before init
torch.cuda.empty_cache()

from matplotlib import cm
import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 18})


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
    ax.legend()


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
    
    save_path = os.path.join(save_dir, "roc_curve_{}.png".format(nep))
    plt.savefig(save_path)
    plt.close()
    
    return (roc_auc, fpr, tpr)
    

def prediction_raw(nep, output, labels, export_dir):
    # Save (append) the above list of lists to CSV file
    save_dir = os.path.join(export_dir, "PRED_RAW")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Get pred probs from output
    mx = np.ma.masked_array(output, mask=labels)
    # For labels == signal, true positive
    raw_tp = mx[mx.mask == True].data
    # For labels == noise, true negative
    raw_tn = mx[mx.mask == False].data
    
    # Diffference between noise and signal stats
    boundary_diff = np.around(max(raw_tp) - max(raw_tn), 3)
    
    # Plotting routine
    ax, _ = figure(title="Raw output at Epoch = {}".format(nep))
    # Log pred probs
    _plot(ax, y=raw_tp, label="Signals", c='red', 
          ylabel="log10 Number of Occurences", xlabel="Raw Output Values", 
          yscale='log', histogram=True)
    _plot(ax, y=raw_tn, label="Noise", c='blue', 
          ylabel="log10 Number of Occurences", xlabel="Raw Output Values", 
          yscale='log', histogram=True)
    
    save_path = os.path.join(save_dir, "log_raw_output_{}.png".format(nep))
    plt.savefig(save_path)
    plt.close()


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
    boundary_diff = np.around(max(pred_prob_tp) - max(pred_prob_tn), 3)
    
    # Plotting routine
    ax, _ = figure(title="Pred prob output at Epoch = {}".format(nep))
    # Log pred probs
    _plot(ax, y=pred_prob_tp, label="Signals", c='red', 
          ylabel="log10 Number of Occurences", xlabel="Prediction Probabilities (Sigmoid)", 
          yscale='log', histogram=True)
    _plot(ax, y=pred_prob_tn, label="Noise", c='blue', 
          ylabel="log10 Number of Occurences", xlabel="Prediction Probabilities (Sigmoid)", 
          yscale='log', histogram=True)
    
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
    bin_width = 100 # samples
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
        plt.fill_between(plot[:,0], plot[:,1]-np.percentile(plot[:,1], 5), 
                         plot[:,1]+np.percentile(plot[:,1], 5), color='blue', 
                         alpha = 0.3, label="5th percentile")
        plt.fill_between(plot[:,0], plot[:,1]-np.percentile(plot[:,1], 50), 
                         plot[:,1]+np.percentile(plot[:,1], 50), color='green', 
                         alpha = 0.3, label="50th percentile")
        plt.fill_between(plot[:,0], plot[:,1]-np.percentile(plot[:,1], 95), 
                         plot[:,1]+np.percentile(plot[:,1], 95), color='red', 
                         alpha = 0.3, label="95th percentile")
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
        
        # Saving the plots
        save_path = os.path.join(save_dir, "diagonal_{}_{}.png".format(param, nep))
        save_path_gt8 = os.path.join(save_dir, "diagonal_snr_gt8_{}_{}.png".format(param, nep))
        fig.savefig(save_path)
        fig_gt8.savefig(save_path_gt8)
        plt.close(fig)
        plt.close(fig_gt8)


def plot_cnn_output(cfg, training_output, training_labels, network_snr, epoch):
    # Plotting the frontend CNN features
    save_dir = os.path.join(cfg.export_dir, 'CNN_OUTPUT/epoch_{}'.format(epoch))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    outputs = training_output['pred_prob']
    features = training_output['cnn_output']
    for n, (output, label, feature, snr) in enumerate(zip(outputs, training_labels, features, network_snr)):
        # Plotting CNN frontend output feature
        det_features = feature.cpu().detach().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(6.0*2, 6.0*1))
        output = np.around(output.cpu().detach().numpy(), 3)
        label = label.cpu().detach().numpy()
        snr = np.around(snr.cpu().detach().numpy(), 3)
        plt.suptitle('DEBUG CNN feature: output={}, label={}, network_snr={}'.format(output, label, snr))
        ax[0].imshow(det_features[0])
        ax[0].grid()
        ax[1].imshow(det_features[1])
        ax[1].grid()
        if snr != -1:
            save_path = os.path.join(save_dir, 'debug_cnn_feature_SNR-{}_{}.png'.format(snr, n))
        else:
            save_path = os.path.join(save_dir, 'debug_cnn_feature_noise_{}.png'.format(n))
            
        plt.tight_layout()
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
    save_path = os.path.join(export_dir, "accuracy_curves.png")
    plt.savefig(save_path)
    plt.close()


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

def training_phase(cfg, Network, optimizer, scheduler, loss_function, training_samples, training_labels, params):
    # Optimizer step on a single batch of training data
    # Set zero_grad to apply None values instead of zeroes
    optimizer.zero_grad(set_to_none = True)
    
    with torch.cuda.amp.autocast():
        # Obtain training output from network
        training_output = Network(training_samples)
        # Get necessary output params from dict output
        pred_prob = training_output['pred_prob']
        
        # Plotting cnn_output in debug mode
        permitted_models = ['KappaModel', 'KappaModelPE']
        if cfg.debug and params['cnn_output'] and cfg.model.__name__ in permitted_models:
            plot_cnn_output(cfg, training_output, training_labels['gw'], params['network_snr'], params['epoch'])
            params['cnn_output'] = False
           
        # Loss calculation
        all_losses = loss_function(training_output, training_labels, cfg.parameter_estimation)
        # Backward propogation using loss_function
        training_loss = all_losses['total_loss']
        training_loss.backward()
    
    # Accuracy calculation
    accuracy = calculate_accuracy(pred_prob, training_labels['gw'], cfg.accuracy_thresh)
    # Clip gradients to make convergence somewhat easier
    torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=cfg.clip_norm)
    # Make the actual optimizer step and save the batch loss
    optimizer.step()
    # Scheduler step
    if scheduler:
        scheduler.step(params['scheduler_step'])
    
    return (all_losses, accuracy)


def validation_phase(cfg, Network, loss_function, validation_samples, validation_labels):
    # Evaluation of a single validation batch
    with torch.cuda.amp.autocast():
        # Gradient evaluation is not required for validation and testing
        # Make sure that we don't do a .backward() function anywhere inside this scope
        with torch.no_grad():
            validation_output = Network(validation_samples)
            # Calculate validation loss with params if required
            all_losses = loss_function(validation_output, validation_labels, cfg.parameter_estimation)
    
    # Accuracy calculation
    accuracy = calculate_accuracy(validation_output['pred_prob'], validation_labels['gw'], cfg.accuracy_thresh)
    # Returning quantities if saving data
    return (all_losses, accuracy, validation_output)


def train(cfg, data_cfg, Network, optimizer, scheduler, loss_function, trainDL, validDL, verbose=False):
    
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
    
    try:
        """ Training and Validation """
        ### Initialise global (over all epochs) params
        best_loss = 1.e10 # impossibly bad value
        best_accuracy = 0.0 # bad value
        best_roc_auc = 0.0 # bad value
        lowest_max_noise_stat = 1.0 # bad value
        lowest_min_noise_stat = 1.0 # bad value
        highest_max_signal_stat = 0.0 # bad value
        highest_min_signal_stat = 0.0 # bad value
        best_noise_stats = -1 # impossible value
        best_signal_stats = -1 # impossible value
        best_stat_compromise = -1 # impossible value
        best_overlap_area = 1.e10 # impossibly bad value
        best_signal_area = 1.e10 # impossibly bad value
        best_noise_area = 1.e10 # impossibly bad value
        best_diff_distance = 0.0 # bad value
        
        # Other
        best_epoch = 0
        best_roc_data = None
        overfitting_check = 0
        params = {}
        
        loss_filepath = os.path.join(cfg.export_dir, cfg.output_loss_file)
        
        for nep in range(cfg.num_epochs):
            
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
            pbar = tqdm(trainDL, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
            
            # Total Number of batches
            num_train_batches = len(trainDL)
            # Recording the time taken for training
            start = time.time()
            train_times = []
            
            if cfg.debug:
                params['cnn_output'] = True
                params['epoch'] = nep
            
            
            for nstep, (training_samples, training_labels, source_params) in enumerate(pbar):
                
                # Update params
                params['scheduler_step'] = nep + nstep / num_train_batches
                params['network_snr'] = source_params['snr']
                
                # Record time taken for training
                start_train = time.time()
                
                ## Class balance assertions
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
                training_loss, accuracy = training_phase(cfg, Network, optimizer, scheduler,
                                                         loss_function, 
                                                         training_samples, training_labels,
                                                         params)
                
                # Display stuff
                loss = np.around(training_loss['total_loss'].clone().cpu().item(), 4)
                pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, training_batches, 
                                                                                       loss,
                                                                                       accuracy))
                # Updating similar things (same same but different, but still same)
                training_batches += 1
                # Update losses and accuracy
                for key in training_running_loss.keys():
                    training_running_loss[key] += training_loss[key].clone().cpu().item()
                acc_train.append(accuracy)
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
                
                validation_running_loss = dict(zip(cfg.parameter_estimation, [0.0]*len(cfg.parameter_estimation)))
                validation_running_loss.update({'total_loss': 0.0, 'gw': 0.0})
                validation_batches = 0
                
                epoch_labels = {foo: [] for foo in cfg.parameter_estimation + ('gw', )}
                epoch_outputs = {foo: [] for foo in cfg.parameter_estimation + ('gw', )}
                sample_params = {}
                raw_output = {'raw': []}
                validation_snrs = []
                
                pbar = tqdm(validDL, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
                for nstep, (validation_samples, validation_labels, source_params) in enumerate(pbar):
                    
                    ## Class balance assertions
                    # batch_labels = validation_labels['gw'].numpy()
                    # check_balance = len(batch_labels[batch_labels == 1])/len(batch_labels)
                    # assert check_balance >= 0.30 and check_balance <= 0.70
                    
                    """ Set the device and dtype """
                    validation_samples = validation_samples.to(dtype=torch.float32, device=cfg.train_device)
                    for key, value in validation_labels.items():
                        validation_labels[key] = value.to(dtype=torch.float32, device=cfg.train_device)
                        
                    """ Call Validation Phase """
                    # Run training phase and get loss and accuracy
                    validation_loss, accuracy, voutput = validation_phase(cfg, Network, 
                                                                        loss_function, 
                                                                        validation_samples, 
                                                                        validation_labels)
                    
                    # Display stuff
                    loss = np.around(validation_loss['total_loss'].clone().cpu().item(), 4)
                    pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, validation_batches, 
                                                                                           loss,
                                                                                           accuracy))
                    # Update losses and accuracy
                    validation_batches += 1
                    for key in validation_running_loss.keys():
                        validation_running_loss[key] += validation_loss[key].clone().cpu().item()
                    acc_valid.append(accuracy)
                    
                    # Params for storing labels and outputs
                    if nep % cfg.save_freq == 0:
                        # Save SNRs for the validation phase (used in diagonal plots)
                        validation_snrs.append(source_params['snr'].cpu())
                        # Move labels from cuda to cpu
                        epoch_labels['gw'].append(validation_labels['gw'].cpu())
                        epoch_outputs['gw'].append(voutput['pred_prob'].cpu().detach().numpy())
                        raw_output['raw'].append(voutput['raw'].cpu().detach().numpy())
                        if nstep == 0:
                            for key in source_params.keys():  
                                sample_params[key] = source_params[key].cpu().detach().numpy()
                        else:
                            for key in source_params.keys():
                                sample_params[key] = np.append(sample_params[key], source_params[key].cpu().detach().numpy())
                                
                        # Add parameter estimation actual and observed values
                        for param in cfg.parameter_estimation:
                            epoch_labels[param].append(validation_labels[param].cpu())
                            epoch_outputs[param].append(voutput[param].cpu().detach().numpy())
                            
                
                if nep % cfg.save_freq == 0:
                    # Concatenate all np arrays together
                    validation_snrs = np.concatenate(tuple(validation_snrs))
                    raw_output['raw'] = np.concatenate(tuple(raw_output['raw']))
                    for param in epoch_labels.keys():
                        epoch_labels[param] = np.concatenate(tuple(epoch_labels[param]))
                        epoch_outputs[param] = np.concatenate(tuple(epoch_outputs[param]))
                    
                    """ ROC Curve save data """
                    roc_auc, fpr, tpr = roc_curve(nep, epoch_outputs['gw'], epoch_labels['gw'], cfg.export_dir)
                    
                    """ Prediction Probabilitiy OR Raw value histograms  """
                    # Confusion matrix has been deprecated as of June 10, 2022
                    # apply_thresh = lambda x: round(x - cfg.accuracy_thresh + 0.5)
                    prediction_raw(nep, raw_output['raw'], epoch_labels['gw'], cfg.export_dir)
                    prediction_probability(nep, epoch_outputs['gw'], epoch_labels['gw'], cfg.export_dir)
                    
                    """ Source parameter vs prediction probabilities OR raw values """
                    learning_parameter_prior(nep, sample_params, raw_output['raw'], epoch_labels['gw'], 'raw_value', cfg.export_dir)
                    learning_parameter_prior(nep, sample_params, epoch_outputs['gw'], epoch_labels['gw'], 'pred_prob', cfg.export_dir)
                    
                    """ Value VS Value Plots for PE """
                    diagonal_compare(nep, epoch_outputs, epoch_labels, validation_snrs, cfg.export_dir)
                    
    
            
            """
            PHASE 3 - Save
                [1] Save losses, accuracy and confusion matrix elements
                [2] Save the best model weights path if global loss is reduced
                [3] Reload the new weights once all phases are complete
            """
            # Print information on the training and validation loss in the current epoch and save current network state
            epoch_validation_loss = {}
            epoch_training_loss = {}
            for key in training_running_loss.keys():
                if key == 'total_loss':
                    epoch_validation_loss['tot'] = np.around(validation_running_loss[key]/validation_batches, 4)
                    epoch_training_loss['tot'] = np.around(training_running_loss[key]/training_batches, 4)
                else:
                    epoch_validation_loss[key] = np.around(validation_running_loss[key]/validation_batches, 4)
                    epoch_training_loss[key] = np.around(training_running_loss[key]/training_batches, 4)
            
            avg_acc_valid = np.around(sum(acc_valid)/len(acc_valid), 4)
            avg_acc_train = np.around(sum(acc_train)/len(acc_train), 4)
            roc_auc = np.around(roc_auc, 4)
            
            # Save args
            args = (nep, )
            args += tuple(np.around(foo, 4) for foo in epoch_training_loss.values())
            args += tuple(np.around(foo, 4) for foo in epoch_validation_loss.values())
            args += (avg_acc_train, avg_acc_valid, roc_auc, )
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
                    other_fields = ('train_acc', 'valid_acc', 'roc_auc', )
                    all_fields = epoch_field + tloss_fields + vloss_fields + other_fields
                    field_names = field_names.format(*all_fields)
                    outfile.write(field_names + '\n')
                # Save output string in losses.txt
                outfile.write(output_string + '\n')
            
            loss_and_accuracy_curves(cfg, loss_filepath, cfg.export_dir)
            
            
            """ Save the best weights (if global loss reduces) """
            split_weights_path = os.path.splitext(cfg.weights_path)
            weights_root_name = 'weights'
            weights_file_ext = split_weights_path[1]
            
            if epoch_validation_loss['tot'] < best_loss and 'loss' in cfg.weight_types:
                weights_path = '{}_loss{}'.format(weights_root_name, weights_file_ext)
                weights_save_path = os.path.join(cfg.export_dir, weights_path)
                torch.save(Network.state_dict(), weights_save_path)
                best_loss = epoch_validation_loss['tot']
                if cfg.save_best_option == 'loss':
                    best_epoch = nep
            
            if avg_acc_valid > best_accuracy and 'accuracy' in cfg.weight_types:
                weights_path = '{}_accuracy{}'.format(weights_root_name, weights_file_ext)
                weights_save_path = os.path.join(cfg.export_dir, weights_path)
                torch.save(Network.state_dict(), weights_save_path)
                best_accuracy = avg_acc_valid
                if cfg.save_best_option == 'accuracy':
                    best_epoch = nep
            
            if roc_auc > best_roc_auc and 'roc_auc' in cfg.weight_types:
                weights_path = '{}_roc_auc{}'.format(weights_root_name, weights_file_ext)
                weights_save_path = os.path.join(cfg.export_dir, weights_path)
                torch.save(Network.state_dict(), weights_save_path)
                best_roc_auc = roc_auc
                best_roc_data = [fpr, tpr, roc_auc]
                if cfg.save_best_option == 'roc_auc':
                    best_epoch = nep
            
            # Get raw values from output
            mx1 = np.ma.masked_array(raw_output['raw'], mask=epoch_labels['gw'])
            # For labels == signal, true positive
            raw_tp = mx1[mx1.mask == True].data
            # For labels == noise, true negative
            raw_tn = mx1[mx1.mask == False].data
            
            # Get pred probs for output
            # Pred probs == signal
            mx2 = np.ma.masked_array(epoch_outputs['gw'], mask=epoch_labels['gw'])
            pred_prob_tp = mx2[mx2.mask == True].data
            pred_prob_tn = mx2[mx2.mask == False].data
            
            # Histogram data for noise and signals
            Hn = np.histogram(raw_tn, bins=50, density=True)
            Hs = np.histogram(raw_tp, bins=50, density=True)
            
            if nep == 0:
                lowest_max_noise_stat = max(raw_tn)
                lowest_min_noise_stat = min(raw_tn)
                highest_max_signal_stat = max(raw_tp)
                highest_min_signal_stat = min(raw_tp)
                best_noise_stats = min(pred_prob_tn) + max(pred_prob_tn)
                best_signal_stats = min(pred_prob_tp) + max(pred_prob_tp)
                best_stat_compromise = (best_noise_stats - 0.0)**2. + (best_signal_stats - 2.0)**2.
                best_overlap_area = np.sum(np.minimum(Hn[0], Hs[0])) * (Hn[1][1:]-Hn[1][:-1])[0]
                noise_argmax = np.argmax(Hn[1])
                best_signal_area = np.sum(Hs[0][Hs[1][1:] > Hn[1][noise_argmax]]) * (Hn[1][1:]-Hn[1][:-1])[0]
                # Get histogram values of noise above thresh value of noise
                thresh_dist = 0.7 * (np.max(Hn[1]) - np.min(Hn[1]))
                thresh = np.min(Hn[1]) + thresh_dist
                best_noise_area = np.sum(Hn[0][Hn[1][1:] > thresh]) * (Hn[1][1:]-Hn[1][:-1])[0]
                best_diff_distance = np.max(raw_tp) - np.max(raw_tn)

                
            else:
                if max(raw_tn) < lowest_max_noise_stat and 'lmax_noise_stat' in cfg.weight_types:
                    weights_path = '{}_lmax_noise_stat{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    lowest_max_noise_stat = np.max(raw_tn)
                    if cfg.save_best_option == 'lmax_noise_stat':
                        best_epoch = nep
                
                if min(raw_tn) < lowest_min_noise_stat and 'lmin_noise_stat' in cfg.weight_types:
                    weights_path = '{}_lmin_noise_stat{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    lowest_min_noise_stat = np.min(raw_tn)
                    if cfg.save_best_option == 'lmin_noise_stat':
                        best_epoch = nep
                
                if max(raw_tp) > highest_max_signal_stat and 'hmax_signal_stat' in cfg.weight_types:
                    weights_path = '{}_hmax_signal_stat{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    highest_max_signal_stat = np.max(raw_tp)
                    if cfg.save_best_option == 'hmax_signal_stat':
                        best_epoch = nep
                
                if min(raw_tp) > highest_min_signal_stat and 'hmin_signal_stat' in cfg.weight_types:
                    weights_path = '{}_hmin_signal_stat{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    highest_min_signal_stat = np.min(raw_tp)
                    if cfg.save_best_option == 'hmin_signal_stat':
                        best_epoch = nep
                
                if min(pred_prob_tn) + max(pred_prob_tn) < best_noise_stats and 'best_noise_stat' in cfg.weight_types:
                    weights_path = '{}_best_noise_stat{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    best_noise_stats = np.min(pred_prob_tn) + np.max(pred_prob_tn)
                    if cfg.save_best_option == 'best_noise_stat':
                        best_epoch = nep
                
                if min(pred_prob_tp) + max(pred_prob_tp) > best_signal_stats and 'best_signal_stat' in cfg.weight_types:
                    weights_path = '{}_best_signal_stat{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    best_signal_stats = min(pred_prob_tp) + max(pred_prob_tp)
                    if cfg.save_best_option == 'best_signal_stat':
                        best_epoch = nep
                
                if (best_noise_stats - 0.0)**2. + (best_signal_stats - 2.0)**2. < best_stat_compromise and 'best_stat_compromise' in cfg.weight_types:
                    weights_path = '{}_best_stat_compromise{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    best_stat_compromise = (best_noise_stats - 0.0)**2. + (best_signal_stats - 2.0)**2.
                    if cfg.save_best_option == 'best_stat_compromise':
                        best_epoch = nep
                
                ### Overlap area between signal and noise stats on normalised histograms
                # Calculate normalised area of overlap [0, 1] +/- epsilon
                overlap_area = np.sum(np.minimum(Hn[0], Hs[0])) * (Hn[1][1:]-Hn[1][:-1])[0]
                if overlap_area > best_overlap_area and 'best_overlap_area' in cfg.weight_types:
                    weights_path = '{}_best_overlap_area.{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    best_overlap_area = overlap_area
                    if cfg.save_best_option == 'best_overlap_area':
                        best_epoch = nep
            
                ### Area of normalised histogram to the right of max noise stat
                # Get histogram values of signal above max value of noise
                noise_argmax = np.argmax(Hn[1])
                fp_signal_area = np.sum(Hs[0][Hs[1][1:] > Hn[1][noise_argmax]]) * (Hn[1][1:]-Hn[1][:-1])[0]
                if fp_signal_area > best_signal_area and 'best_signal_area' in cfg.weight_types:
                    weights_path = '{}_best_signal_area{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    best_signal_area = fp_signal_area
                    if cfg.save_best_option == 'best_signal_area':
                        best_epoch = nep
                
                ### Area of normalised noise histogram above threshold
                # Get histogram values of noise above thresh value of noise
                thresh_dist = 0.7 * (np.max(Hn[1]) - np.min(Hn[1]))
                thresh = np.min(Hn[1]) + thresh_dist
                fp_noise_area = np.sum(Hn[0][Hn[1][1:] > thresh]) * (Hn[1][1:]-Hn[1][:-1])[0]
                if fp_noise_area > best_noise_area and 'best_noise_area' in cfg.weight_types:
                    weights_path = '{}_best_noise_area{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    best_signal_area = fp_noise_area
                    if cfg.save_best_option == 'best_noise_area':
                        best_epoch = nep
            
                ### Distance between maximum raw values of noise and signal stats
                diff_distance = np.max(raw_tp) - np.max(raw_tn)
                if diff_distance > best_diff_distance and 'best_diff_distance' in cfg.weight_types:
                    weights_path = '{}_best_diff_distance{}'.format(weights_root_name, weights_file_ext)
                    weights_save_path = os.path.join(cfg.export_dir, weights_path)
                    torch.save(Network.state_dict(), weights_save_path)
                    best_diff_distance = diff_distance
                    if cfg.save_best_option == 'best_diff_distance':
                        best_epoch = nep
                
            
            """ Epoch Display """
            print("\nBest Validation Loss (wrt all past epochs) = {}".format(best_loss))
            
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
            
            
            """ Early Stopping """
            if epoch_validation_loss['tot'] > 1.1*epoch_training_loss['tot'] and cfg.early_stopping:
                overfitting_check += 1
                if overfitting_check > 3:
                    print("\nThe current model may be overfitting! Terminating.")
                    break
    
    
    except KeyboardInterrupt:
        print(traceback.format_exc())
        print('manual.py: Terminated due to user controlled KeyboardInterrupt.')
    
        
    print("\n==========================================================================\n")
    print("Training Complete!")
    print("Best validation loss = {}".format(best_loss))
    print("Best validation accuracy = {}".format(best_accuracy))
    
    # Saving best epoch results
    best_dir = os.path.join(cfg.export_dir, 'BEST')
    if not os.path.isdir(best_dir):
        os.makedirs(best_dir, exist_ok=False)
    
    # Move premade plots
    roc_dir = 'ROC'
    roc_file = "roc_curve_{}.png".format(best_epoch)
    roc_path = os.path.join(cfg.export_dir, os.path.join(roc_dir, roc_file))
    shutil.copy(roc_path, os.path.join(best_dir, roc_file))
    # Save best ROC curve raw data
    np.save(os.path.join(best_dir, 'roc_best.npy'), np.stack(best_roc_data[:2], axis=0))
    np.save(os.path.join(best_dir, 'roc_auc_best.npy'), np.array(best_roc_data[2]))
    
    # Best prediction probabilities
    pred_dir = 'PRED_RAW'
    pred_file = "log_raw_output_{}.png".format(best_epoch)
    pred_path = os.path.join(cfg.export_dir, os.path.join(pred_dir, pred_file))
    shutil.copy(pred_path, os.path.join(best_dir, pred_file))
    
    # Best prediction probabilities
    pred_dir = 'PRED_PROB'
    pred_file = "log_pred_prob_output_{}.png".format(best_epoch)
    pred_path = os.path.join(cfg.export_dir, os.path.join(pred_dir, pred_file))
    shutil.copy(pred_path, os.path.join(best_dir, pred_file))
    
    # Move weights to BEST dir
    weights_save_paths = glob.glob(os.path.join(cfg.export_dir, '*.pt'))
    for wpath in weights_save_paths:
        foo = os.path.split(wpath)
        shutil.move(wpath, os.path.join(best_dir, foo[1]))
    
    permitted_models = ['KappaModel', 'KappaModelPE']
    if cfg.debug and cfg.model.__name__ in permitted_models:
        # Move best CNN features
        src_best_features = os.path.join(cfg.export_dir, 'CNN_OUTPUT/epoch_{}'.format(best_epoch))
        dst_best_features = os.path.join(best_dir, 'CNN_features_epoch_{}'.format(best_epoch))
        copy_tree(src_best_features, dst_best_features)
    # Move best diagonal plots
    src_best_diagonals = os.path.join(cfg.export_dir, 'DIAGONAL/epoch_{}'.format(best_epoch))
    dst_best_diagonals = os.path.join(best_dir, 'diagonal_epoch_{}'.format(best_epoch))
    copy_tree(src_best_diagonals, dst_best_diagonals)
    
    # Remake loss curve and accuracy curve with best epoch marked
    loss_and_accuracy_curves(cfg, loss_filepath, best_dir, best_epoch=best_epoch)
    
    # Return the trained network with the best possible weights
    # This step is mandatory before the inference/testing module
    weights_path = '{}_{}{}'.format(weights_root_name, cfg.save_best_option, weights_file_ext)
    Network.load_state_dict(torch.load(os.path.join(best_dir, weights_path)))
    
    return Network
