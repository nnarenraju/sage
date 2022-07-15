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

import matplotlib.pyplot as plt
# Font and plot parameters
plt.rcParams.update({'font.size': 22})


def figure(title=""):
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    fig, axs = plt.subplots(1, figsize=(16.0, 14.0))
    fig.suptitle(title, fontsize=28, y=0.92)
    return axs


def _plot(ax, x=None, y=None, xlabel="x-axis", ylabel="y-axis", ls='solid', 
          label="", c=None, yscale='linear', xscale='linear', histogram=False):
    
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
    ax = figure(title="ROC Curve at Epoch = {}".format(nep))
    
    # Log ROC Curve
    _plot(ax, fpr, tpr, label='AUC = %0.2f' % roc_auc, c='red', 
          ylabel="True Positive Rate", xlabel="False Positive Rate", 
          yscale='log', xscale='log')
    _plot(ax, [0, 1], [0, 1], label="Random Classifier", c='blue', 
          ylabel="True Positive Rate", xlabel="False Positive Rate", 
          ls="dashed", yscale='log', xscale='log')
    
    save_path = os.path.join(save_dir, "roc_curve_{}.png".format(nep))
    plt.savefig(save_path)
    plt.close()
    
    return (roc_auc, fpr, tpr)
    

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
    
    # Plotting routine
    ax = figure(title="Prediction Probabilities at Epoch = {}".format(nep))
    # Log pred probs
    _plot(ax, y=pred_prob_tp, label="Signals", c='red', 
          ylabel="log10 Number of Occurences", xlabel="Predicted Probabilities", 
          yscale='log', histogram=True)
    _plot(ax, y=pred_prob_tn, label="Noise", c='blue', 
          ylabel="log10 Number of Occurences", xlabel="Predicted Probabilities", 
          yscale='log', histogram=True)
    
    save_path = os.path.join(save_dir, "log_pred_prob_{}.png".format(nep))
    plt.savefig(save_path)
    plt.close()


def diagonal_compare(nep, outputs, labels, network_snrs, export_dir):
    # Save (append) the above list of lists to CSV file
    save_dir = os.path.join(export_dir, "DIAGONAL/epoch_{}".format(nep))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Mask function
    mask_function = lambda foo: True if foo>=0.0 else False
    print(network_snrs)
    mask = [mask_function(foo) for foo in network_snrs]
    mx0 = np.ma.masked_array(network_snrs, mask=mask)
    
    for param in outputs.keys():
        # Plotting routine
        ax = figure(title="Diagonal Plot of {} at Epoch = {}".format(param, nep))
        # Plotting the observed value vs actual value scatter
        mx1 = np.ma.masked_array(outputs[param], mask=mask)
        mx2 = np.ma.masked_array(labels[param], mask=mask)
        # For labels == signal, true positive
        plot_output = mx1[mx1.mask == True].data
        plot_labels = mx2[mx2.mask == True].data
        plot_snrs = mx0[mx0.mask == True].data
        # Plotting
        ax.scatter(plot_output, plot_labels, marker='.', s=200.0, c=plot_snrs)
        # Plotting params
        ax.grid(True, which='both')
        ax.set_xlabel('Observed Value [{}]'.format(param))
        ax.set_ylabel('Actual Value [{}]'.format(param))
        # Plotting the diagonal dashed line for reference
        _plot(ax, [0, 1], [0, 1], label="Best Classifier", c='k', 
              ylabel='Actual Value [{}]'.format(param), 
              xlabel='Observed Value [{}]'.format(param), ls="dashed")
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "diagonal_{}_{}.png".format(param, nep))
        plt.savefig(save_path)
        plt.close()


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
    ax = figure(title="Loss Curves")
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
    ax = figure(title="PE Loss Curves")
    # Colour map
    numc = len(cfg.parameter_estimation)
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
    ax = figure(title="Accuracy Curves")
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
        if cfg.debug and params['cnn_output']:
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
        best_epoch = 0
        best_roc_data = None
        overfitting_check = 0
        params = {}
        
        loss_filepath = os.path.join(cfg.export_dir, cfg.output_loss_file)
        
        for nep in range(cfg.num_epochs):
            
            print("\n====================== Epoch {} ======================".format(nep))
            
            # Training epoch
            Network.train()
            
            # Necessary save and update params
            training_running_loss = dict(zip(cfg.parameter_estimation, [0.0]*len(cfg.parameter_estimation)))
            training_running_loss.update({'total_loss': 0.0})
            training_batches = 0
            # Store accuracy params
            acc_train = []
            acc_valid = []
            
            
            """
            PHASE 1 - Training 
                [1] Do gradient clipping. Set value in cfg.
            """
            print("\nTraining Phase Initiated")
            pbar = tqdm(trainDL)
            
            # Total Number of batches
            num_train_batches = len(trainDL)
            # Recording the time taken for training
            start = time.time()
            train_times = []
            
            if cfg.debug:
                params['cnn_output'] = True
                params['epoch'] = nep
            
            
            for nstep, (training_samples, training_labels) in enumerate(pbar):
                
                # Update params
                params['scheduler_step'] = nep + nstep / num_train_batches
                params['network_snr'] = training_labels['snr']
                
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
                pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, training_batches, training_loss['total_loss'], accuracy))
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
                validation_running_loss.update({'total_loss': 0.0})
                validation_batches = 0
                
                epoch_labels = {foo: [] for foo in cfg.parameter_estimation + ('gw', )}
                epoch_outputs = {foo: [] for foo in cfg.parameter_estimation + ('gw', )}
                validation_snrs = []
                
                pbar = tqdm(validDL)
                for validation_samples, validation_labels in pbar:
                    
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
                    pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, validation_batches, validation_loss['total_loss'], accuracy))
                    # Update losses and accuracy
                    validation_batches += 1
                    for key in validation_running_loss.keys():
                        validation_running_loss[key] += validation_loss[key].clone().cpu().item()
                    acc_valid.append(accuracy)
                    
                    # Params for storing labels and outputs
                    if nep % cfg.save_freq == 0:
                        # Save SNRs for the validation phase (used in diagonal plots)
                        validation_snrs.append(validation_labels['snr'].cpu())
                        # Move labels from cuda to cpu
                        epoch_labels['gw'].append(validation_labels['gw'].cpu())
                        epoch_outputs['gw'].append(voutput['pred_prob'].cpu().detach().numpy())
                        # Add parameter estimation actual and observed values
                        for param in cfg.parameter_estimation:
                            epoch_labels[param].append(validation_labels[param].cpu())
                            epoch_outputs[param].append(voutput[param].cpu().detach().numpy())
                            
                
                if nep % cfg.save_freq == 0:
                    # Concatenate all np arrays together
                    validation_snrs = np.concatenate(tuple(validation_snrs))
                    for param in epoch_labels.keys():
                        epoch_labels[param] = np.concatenate(tuple(epoch_labels[param]))
                        epoch_outputs[param] = np.concatenate(tuple(epoch_outputs[param]))
                    
                    """ ROC Curve save data """
                    roc_auc, fpr, tpr = roc_curve(nep, epoch_outputs['gw'], epoch_labels['gw'], cfg.export_dir)
                    
                    """ Calculating Pred Probs """
                    # Confusion matrix has been deprecated as of June 10, 2022
                    # apply_thresh = lambda x: round(x - cfg.accuracy_thresh + 0.5)
                    prediction_probability(nep, epoch_outputs['gw'], epoch_labels['gw'], cfg.export_dir)
                    
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
                    epoch_validation_loss['tot'] = validation_running_loss[key]/validation_batches
                    epoch_training_loss['tot'] = training_running_loss[key]/training_batches
                else:
                    epoch_validation_loss[key] = validation_running_loss[key]/validation_batches
                    epoch_training_loss[key] = training_running_loss[key]/training_batches
                
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
            if epoch_validation_loss['tot'] < best_loss:
                weights_save_path = os.path.join(cfg.export_dir, cfg.weights_path)
                torch.save(Network.state_dict(), weights_save_path)
                best_loss = epoch_validation_loss['tot']
                best_epoch = nep
                best_roc_data = [fpr, tpr, roc_auc]
            
            if avg_acc_valid > best_accuracy:
                best_accuracy = avg_acc_valid
            
            
            """ Epoch Display """
            print("\nBest Validation Loss (wrt all past epochs) = {}".format(best_loss))
            
            print("\n-- Average losses in Validation Phase --")
            print("Total Loss = {}".format(epoch_validation_loss['tot']))
            print("  GW Loss = {}".format(epoch_validation_loss['gw']))
            for param in cfg.parameter_estimation:
                print("  {} Loss = {}".format(param, epoch_validation_loss[param]))
            
            print("-- Average losses in Training Phase --")
            print("Total Loss = {}".format(epoch_training_loss['tot']))
            print("  GW Loss = {}".format(epoch_training_loss['gw']))
            for param in cfg.parameter_estimation:
                print("  {} Loss = {}".format(param, epoch_training_loss[param]))
            
            print("\nAverage Validation Accuracy = {}".format(avg_acc_valid))
            print("Average Training Accuracy = {}".format(avg_acc_train))
            print("\n")
            
            
            """ Early Stopping """
            if epoch_validation_loss['tot'] > 1.1*epoch_training_loss['tot'] and cfg.early_stopping:
                overfitting_check += 1
                if overfitting_check > 3:
                    print("\nThe current model may be overfitting! Terminating.")
                    break
    
    
    except KeyboardInterrupt:
        print(traceback.format_exc())
        print('manual.py: Terminated due to user controlled KeyboardInterrupt.')
    
        
    print("\n================================================================\n")
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
    pred_dir = 'PRED_PROB'
    pred_file = "log_pred_prob_{}.png".format(best_epoch)
    pred_path = os.path.join(cfg.export_dir, os.path.join(pred_dir, pred_file))
    shutil.copy(pred_path, os.path.join(best_dir, pred_file))
    
    # Move best weights
    shutil.move(weights_save_path, os.path.join(best_dir, cfg.weights_path))
    # Move best CNN features
    src_best_features = os.path.join(cfg.export_dir, 'CNN_OUTPUT/epoch_{}'.format(best_epoch))
    dst_best_features = os.path.join(best_dir, 'CNN_features_epoch_{}'.format(best_epoch))
    copy_tree(src_best_features, dst_best_features)
    
    # Remake loss curve and accuracy curve with best epoch marked
    loss_and_accuracy_curves(cfg, loss_filepath, best_dir, best_epoch=best_epoch)
    
    print('\nFIN')
    
    
    
