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
import numpy as np
from tqdm import tqdm
import torch.utils.data as D
import sklearn.metrics as metrics

# LOCAL
from data.datasets import Simple
from utils.record_times import record

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
          label="", c=None, yscale='linear', histogram=False):
    
    # Plotting type
    if histogram:
        ax.hist(y, bins=100, label=label, alpha=0.8)
    else:
        ax.plot(x, y, ls=ls, c=c, linewidth=3.0, label=label)
    
    # Plotting params
    ax.set_xscale(yscale)
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
        output_check = apply_thresh(float(toutput[0]))
        labels_check = apply_thresh(float(tlabel[0]))
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
          yscale='log')
    _plot(ax, [0, 1], [0, 1], label="Random Classifier", c='blue', 
          ylabel="True Positive Rate", xlabel="False Positive Rate", 
          ls="dashed", yscale='log')
    
    save_path = os.path.join(save_dir, "roc_curve_{}.png".format(nep))
    plt.savefig(save_path)
    plt.close()
    
    return roc_auc
    

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
          hist=True, yscale='log', histogram=True)
    _plot(ax, y=pred_prob_tn, label="Noise", c='blue', 
          ylabel="log10 Number of Occurences", xlabel="Predicted Probabilities", 
          hist=True, yscale='log', histogram=True)
    
    save_path = os.path.join(save_dir, "log_pred_prob_{}.png".format(nep))
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
        if 'tc' in list(training_output.keys()):
            tc = training_output['tc']
        
        ## TODO: Loss and accuracy need to be redefined to include 'tc'
        # Loss calculation
        training_loss = loss_function(training_output, training_labels)
        # Backward propogation using loss_function
        training_loss.backward()
    
    # Accuracy calculation
    accuracy = calculate_accuracy(pred_prob, training_labels, cfg.accuracy_thresh)
    # Clip gradients to make convergence somewhat easier
    torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=cfg.clip_norm)
    # Make the actual optimizer step and save the batch loss
    optimizer.step()
    # Scheduler step
    scheduler.step(params['scheduler_step'])
    
    return (training_loss, accuracy)


def validation_phase(cfg, Network, loss_function, validation_samples, validation_labels):
    # Evaluation of a single validation batch
    with torch.cuda.amp.autocast():
        # Gradient evaluation is not required for validation and testing
        # Make sure that we don't do a .backward() function anywhere inside this scope
        with torch.no_grad():
            validation_output = Network(validation_samples)
            # Get necessary output params from dict output
            pred_prob = validation_output['pred_prob']
            if 'tc' in list(validation_output.keys()):
                tc = validation_output['tc']
            
            ## TODO: loss and accuracy must be redefined to include 'tc'
            validation_loss = loss_function(validation_output, validation_labels)
    
    # Accuracy calculation
    accuracy = calculate_accuracy(pred_prob, validation_labels, cfg.accuracy_thresh)
    # Returning pred_prob if saving data
    return (validation_loss, accuracy, pred_prob)


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
    
    
    """ Training and Validation """
    with open(os.path.join(cfg.export_dir, cfg.output_loss_file), 'w') as outfile:

        ### Initialise global (over all epochs) params
        best_loss = 1.e10 # impossibly bad value
        best_accuracy = 0.0 # bad value
        overfitting_check = 0
        
        for nep, epoch in enumerate(range(cfg.num_epochs)):
            
            print("\n=========== Epoch {} ===========".format(nep))
            
            # Training epoch
            Network.train()
            
            # Necessary save and update params
            training_running_loss = 0.
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
            
            if cfg.num_workers == 0:
                # Store loading time split
                section_times = []
                signal_aug_times = []
                noise_aug_times = []
                transfrom_times = []
            
            
            for nstep, (training_samples, training_labels, all_times) in enumerate(pbar):
                
                # Update params
                params = {}
                params['scheduler_step'] = nep + nstep / num_train_batches
                
                if cfg.num_workers == 0:
                    section_times.append(all_times['sections'])
                    transfrom_times.append(all_times['transforms'])
                    signal_aug_times.append(all_times['signal_aug'])
                    noise_aug_times.append(all_times['noise_aug'])
                
                # Record time taken for training
                start_train = time.time()
                
                
                """ Tensorification and Device Compatibility """
                ## Performing this here rather than in the Dataset object
                ## will reduce the overhead of having to move each sample to CUDA 
                ## rather than moving a batch of data.
                # Set the device and dtype
                training_samples = training_samples.to(dtype=torch.float32, device=cfg.train_device)
                training_labels = training_labels.to(dtype=torch.float32, device=cfg.train_device)
                
                
                batch_training_loss = 0.
                accuracies = []
                # Get all mini-folds and run training phase for each batch
                # Here each batch is cfg.batch_size. Each mini-fold contains multiple batches
                if cfg.megabatch:
                    for batch_samples, batch_labels in zip(training_samples, training_labels):
                        # Convert the training_sample into a Simple dataset object
                        # We take the first element since we give 1 batch to the BatchLoader
                        batch_train_dataset = Simple(batch_samples, batch_labels, 
                                               store_device=cfg.store_device, 
                                               train_device=cfg.train_device)
                        # Pass Simple dataset into a dataloader with cfg.batch_size
                        batch_train_loader = D.DataLoader(
                            batch_train_dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                            prefetch_factor=cfg.prefetch_factor, persistent_workers=cfg.persistent_workers)
                        # Now iterate through this dataset and run training phase for each batch
                        for samples, labels in batch_train_loader:
                            # Run training phase and get loss and accuracy
                            tloss, accuracy = training_phase(cfg, Network, optimizer, scheduler,
                                                             loss_function, samples, labels,
                                                             params)
                            
                            # Display stuff
                            pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, training_batches, tloss, accuracy))
                            # Updating things (Do not judge this comment. It was a Friday evening.)
                            batch_training_loss += tloss.clone().cpu().item()
                            accuracies.append(accuracy)
                            training_batches += 1
                        
                else:
                    # Run training phase and get loss and accuracy
                    training_loss, accuracy = training_phase(cfg, Network, optimizer, scheduler,
                                                             loss_function, 
                                                             training_samples, training_labels,
                                                             params)
                    
                    # Display stuff
                    pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, training_batches, training_loss, accuracy))
                    # Updating similar things (same same but different, but still same)
                    batch_training_loss += training_loss.clone().cpu().item()
                    accuracies.append(accuracy)
                    training_batches += 1
                
                # Update losses and accuracy
                training_running_loss += batch_training_loss
                acc_train.extend(accuracies)
                
                # Record time taken to load data (calculate avg time later)
                train_times.append(time.time() - start_train)
            
            ## Time taken to train data
            # Total time taken for training phase
            total_time = time.time() - start
            
            if cfg.num_workers == 0:
                # Plotting
                plot_times = {}
                plot_times['section'] = section_times
                plot_times['signal_aug'] = signal_aug_times
                plot_times['noise_aug'] = noise_aug_times
                plot_times['transforms'] = transfrom_times
                plot_times['train'] = train_times
                plot_times['load'] = None
            
                record(plot_times, total_time, cfg)
            
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
                
                validation_running_loss = 0.
                validation_batches = 0
                
                pbar = tqdm(validDL)
                for validation_samples, validation_labels, _ in pbar:
                    
                    # Set the device and dtype
                    validation_samples = validation_samples.to(dtype=torch.float32, device=cfg.train_device)
                    validation_labels = validation_labels.to(dtype=torch.float32, device=cfg.train_device)
                    
                    batch_validation_loss = 0.
                    accuracies = []
                    pred_prob = []
                    # Get all mini-folds and run training phase for each batch
                    # Here each batch is cfg.batch_size. Each mini-fold contains multiple batches
                    if cfg.megabatch:
                        for batch_samples, batch_labels in zip(validation_samples, validation_labels):
                            # Convert the training_sample into a Simple dataset object
                            batch_valid_dataset = Simple(batch_samples, batch_labels, 
                                                   store_device=cfg.store_device, 
                                                   train_device=cfg.train_device)
                            # Pass Simple dataset into a dataloader with cfg.batch_size
                            batch_valid_loader = D.DataLoader(
                                batch_valid_dataset, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                                prefetch_factor=cfg.prefetch_factor, persistent_workers=cfg.persistent_workers)
                            # Now iterate through this dataset and run training phase for each batch
                            for samples, labels in batch_valid_loader:
                                # Run training phase and get loss and accuracy
                                vloss, accuracy, preds = validation_phase(cfg, Network, 
                                                                          loss_function, 
                                                                          validation_samples, validation_labels)
                                
                                # Display stuff
                                pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, validation_batches, vloss, accuracy))
                                batch_validation_loss += vloss.clone().cpu().item()
                                # Updating things but now its validation
                                accuracies.append(accuracy)
                                pred_prob.append(preds.cpu().detach().numpy())
                                validation_batches += 1
                            
                    else:
                        # Run training phase and get loss and accuracy
                        validation_loss, accuracy, preds = validation_phase(cfg, Network, 
                                                                            loss_function, 
                                                                            validation_samples, validation_labels)
                        
                        # Display stuff
                        pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, validation_batches, validation_loss, accuracy))
                        # Updating
                        batch_validation_loss += validation_loss.clone().cpu().item()
                        accuracies.append(accuracy)
                        pred_prob.append(preds.cpu().detach().numpy())
                        validation_batches += 1
                    
                    pred_prob = np.row_stack(pred_prob)
                    # Update losses and accuracy
                    validation_running_loss += batch_validation_loss
                    acc_valid.extend(accuracies)

                    if nep % cfg.save_freq == 0:
                        # Move labels from cuda to cpu
                        labels = validation_labels.cpu()[:,0]
                        outputs = pred_prob[:,0]
                        
                        """ ROC Curve save data """
                        roc_auc = roc_curve(nep, outputs, labels, cfg.export_dir)

                        """ Calculating Pred Probs """
                        # Confusion matrix has been deprecated as of June 10, 2022
                        # apply_thresh = lambda x: round(x - cfg.accuracy_thresh + 0.5)
                        prediction_probability(nep, outputs, labels, cfg.export_dir)


            """
            PHASE 3 - Save
                [1] Save losses, accuracy and confusion matrix elements
                [2] Save the best model weights path if global loss is reduced
                [3] Reload the new weights once all phases are complete
            """
            # Print information on the training and validation loss in the current epoch and save current network state
            epoch_validation_loss = validation_running_loss/validation_batches
            epoch_training_loss = training_running_loss/training_batches
            avg_acc_valid = sum(acc_valid)/len(acc_valid)
            avg_acc_train = sum(acc_train)/len(acc_train)
            
            # Save output string in losses.txt
            output_string = '%04i    %f    %f    %f    %f    %f' % \
                            (epoch, epoch_training_loss, epoch_validation_loss,
                             avg_acc_train, avg_acc_valid, roc_auc)
                            
            outfile.write(output_string + '\n')
            
            """ Save the best weights (if global loss reduces) """
            if epoch_validation_loss < best_loss:
                weights_save_path = os.path.join(cfg.export_dir, cfg.weights_path)
                torch.save(Network.state_dict(), weights_save_path)
                best_loss = epoch_validation_loss
            
            if avg_acc_valid > best_accuracy:
                best_accuracy = avg_acc_valid
            
            
            """ Epoch Display """
            print("\nBest Validation Loss (wrt all past epochs) = {}".format(best_loss))
            print("\nEpoch Validation Loss = {}".format(epoch_validation_loss))
            print("Epoch Training Loss = {}".format(epoch_training_loss))
            print("Average Validation Accuracy = {}".format(avg_acc_valid))
            print("Average Training Accuracy = {}".format(avg_acc_train))
            if epoch_validation_loss > 1.1*epoch_training_loss and cfg.early_stopping:
                overfitting_check += 1
                if overfitting_check > 3:
                    print("\nThe current model may be overfitting! Terminating.")
                    break
        
        print("\n================================================================\n")
        print("Training Complete!")
        print("Best validation loss = {}".format(best_loss))
        print("Best validation accuracy = {}".format(best_accuracy))
        print('\n\n')
