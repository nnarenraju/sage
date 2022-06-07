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
import csv
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data as D

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


def save_data(rowsdata, file_path):
    # Append row data to the given CSV file path
    with open(file_path, 'a', newline='') as fp:
        writer = csv.writer(fp)
        for rowdata in rowsdata:
            writer.writerow(rowdata)


def roc_save_data(nep, output, labels, export_dir):
    # ROC Curve save data
    # Convert output to saveable format
    voutput = [list(foo) for foo in output]
    vlabels = [list(foo) for foo in labels]
    # Save (append) the above list of lists to CSV file
    save_dir = os.path.join(export_dir, "data_roc_curve")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Save file paths
    save_vout = os.path.join(save_dir, "output_save_epoch_{}.csv".format(nep))
    save_vlab = os.path.join(save_dir, "labels_save_epoch_{}.csv".format(nep))
    # Save voutputs and vlabels in the above CSV files
    save_data(voutput, save_vout)
    save_data(vlabels, save_vlab)
    

def prediction_probability_save_data(nep, vlabels, voutput_0, export_dir):
    # Save (append) the above list of lists to CSV file
    save_dir = os.path.join(export_dir, "data_pred_prob")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    
    # Save file paths
    save_tp = os.path.join(save_dir, "pred_prob_tp_epoch_{}.csv".format(nep))
    save_tn = os.path.join(save_dir, "pred_prob_tn_epoch_{}.csv".format(nep))
    
    data = voutput_0

    # Input is a signal (pred_prob_tp)
    if vlabels[0] == 1:
        save_data([[data]], save_tp)
    # Input is noise (pred_prob_tn)
    if vlabels[1] == 1:
        save_data([[data]], save_tn)
    

def training_phase(cfg, Network, optimizer, loss_function, training_samples, training_labels):
    # Optimizer step on a single batch of training data
    optimizer.zero_grad()
    # Obtain training output from network
    training_output = Network(training_samples)
    # Get necessary output params from dict output
    pred_prob = training_output['pred_prob']
    if 'tc' in list(training_output.keys()):
        tc = training_output['tc']
    
    ## TODO: Loss and accuracy need to be redefined to include 'tc'
    # Loss calculation
    training_loss = loss_function(pred_prob, training_labels)
    # Accuracy calculation
    accuracy = calculate_accuracy(pred_prob, training_labels, cfg.accuracy_thresh)
    # Backward propogation using loss_function
    training_loss.backward()
    # Clip gradients to make convergence somewhat easier
    torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=cfg.clip_norm)
    # Make the actual optimizer step and save the batch loss
    optimizer.step()
    
    return (training_loss, accuracy)


def validation_phase(cfg, nep, Network, optimizer, loss_function, validation_samples, validation_labels):
    # Evaluation of a single validation batch
    validation_output = Network(validation_samples)
    # Get necessary output params from dict output
    pred_prob = validation_output['pred_prob']
    if 'tc' in list(validation_output.keys()):
        tc = validation_output['tc']
    
    ## TODO: loss and accuracy must be redefined to include 'tc'
    validation_loss = loss_function(pred_prob, validation_labels)
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
        overfitting_check = 0
        
        for nep, epoch in enumerate(range(cfg.num_epochs)):
            
            print("\n=========== Epoch {} ===========".format(nep))
            
            # Training epoch
            Network.train()
            
            # Necessary save and update params
            training_running_loss = 0.
            training_batches = 0
            # Confusion matrix elements
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            # Store accuracy params
            acc_train = []
            acc_valid = []
            
            """
            PHASE 1 - Training
                [1] Do gradient clipping. Set value in cfg.
            """
            print("\nTraining Phase Initiated")
            pbar = tqdm(trainDL)
            
            # Recording the time taken for training
            start = time.time()
            load_times = [start]
            train_times = []
            # Store loading time split
            section_times = []
            signal_aug_times = []
            noise_aug_times = []
            transfrom_times = []
            
            
            for training_samples, training_labels, all_times in pbar:
                
                # Record time taken to load data (calculate avg time later)
                load_times.append(time.time())
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
                if cfg.dataset.__name__ == "BatchLoader" or cfg.megabatch:
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
                            tloss, accuracy = training_phase(cfg, Network, optimizer, loss_function, 
                                                             samples, labels)
                            
                            # Display stuff
                            pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, training_batches, tloss, accuracy))
                            # Updating things (Do not judge this comment. It was a Friday evening.)
                            batch_training_loss += tloss.clone().cpu().item()
                            accuracies.append(accuracy)
                            training_batches += 1
                        
                else:
                    # Run training phase and get loss and accuracy
                    training_loss, accuracy = training_phase(cfg, Network, optimizer, loss_function, 
                                                             training_samples, training_labels)
                    
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
                end_train = time.time() - start_train
                train_times.append(end_train)
            
            ## Time taken to train data
            # Total time taken for training phase
            total_time = time.time() - start
            # Plotting
            plot_times = {}
            plot_times['section'] = section_times
            plot_times['signal_aug'] = signal_aug_times
            plot_times['noise_aug'] = noise_aug_times
            plot_times['transforms'] = transfrom_times
            plot_times['train'] = train_times
            plot_times['load'] = load_times
            
            print(np.average(train_times))
            print(plot_times)
            record(plot_times, total_time, cfg)
            raise

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
                    
                    batch_validation_loss = 0.
                    accuracies = []
                    pred_prob = []
                    # Get all mini-folds and run training phase for each batch
                    # Here each batch is cfg.batch_size. Each mini-fold contains multiple batches
                    if cfg.dataset.__name__ == "BatchLoader" or cfg.megabatch:
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
                                vloss, accuracy, preds = validation_phase(cfg, nep, Network, optimizer, loss_function, 
                                                                          samples, labels)
                                
                                # Display stuff
                                pbar.set_description("Epoch {}, batch {} - loss = {}, acc = {}".format(nep, validation_batches, vloss, accuracy))
                                batch_validation_loss += vloss.clone().cpu().item()
                                # Updating things but now its validation
                                accuracies.append(accuracy)
                                pred_prob.append(preds.cpu().detach().numpy())
                                validation_batches += 1
                            
                    else:
                        # Run training phase and get loss and accuracy
                        validation_loss, accuracy, preds = validation_phase(cfg, nep, Network, optimizer, loss_function, 
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

                    """ ROC Curve save data """
                    if nep % cfg.save_freq == 0 and nep!=0:
                        roc_save_data(nep, pred_prob, validation_labels, cfg.export_dir)

                    """ Calculating confusion matrix and Pred Probs """
                    apply_thresh = lambda x: round(x - cfg.accuracy_thresh + 0.5)
                    for voutput, vlabel in zip(pred_prob, validation_labels):
                        # Get labels based on threshold
                        vlabel = vlabel.cpu().detach().numpy()
                        coutput = apply_thresh(float(voutput[0]))
                        clabel = apply_thresh(float(vlabel[0]))
                        
                        """ Prediction Probabilties """
                        # Storing predicted probabilities
                        if nep % cfg.save_freq == 0 and nep!=0:
                            prediction_probability_save_data(nep, vlabel, voutput[0], cfg.export_dir)
                            
                        """ Confusion Matrix """
                        # True
                        if coutput == clabel:
                            # Positive
                            if coutput == 1:
                                tp += 1
                            # Negative
                            elif coutput == 0:
                                tn += 1
                        # False
                        elif coutput != clabel:
                            # Positive
                            if coutput == 1:
                                fp += 1
                            # Negative
                            elif coutput == 0:
                                fn += 1


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
            output_string = '%04i    %f    %f    %f    %f    %f    %f    %f    %f' % \
                            (epoch, epoch_training_loss, epoch_validation_loss,
                             avg_acc_train, avg_acc_valid, tp, tn, fp, fn)
                            
            outfile.write(output_string + '\n')
            
            """ Save the best weights (if global loss reduces) """
            if epoch_validation_loss < best_loss:
                weights_save_path = os.path.join(cfg.export_dir, cfg.weights_path)
                torch.save(Network.state_dict(), weights_save_path)
                best_loss = epoch_validation_loss
            
            """ Epoch Display """
            print("Best Validation Loss = {}".format(best_loss))
            print("\nEpoch Validation Loss = {}".format(epoch_validation_loss))
            print("Epoch Training Loss = {}".format(epoch_training_loss))
            print("Average Validation Accuracy = {}".format(avg_acc_valid))
            print("Average Training Accuracy = {}".format(avg_acc_train))
            if epoch_validation_loss > 1.1*epoch_training_loss and cfg.early_stopping:
                overfitting_check += 1
                if overfitting_check > 3:
                    print("\nThe current model may be overfitting! Terminating.")
                    break
        
        print("\n================================================================")
        print("Training complete with best validation loss = {}".format(best_loss))
