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
import h5py
import json
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.utils.data as D

# LOCAL
from data.datasets import Simple


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
        output_check = [apply_thresh(float(toutput[0])), apply_thresh(float(toutput[1]))]
        labels_check = [apply_thresh(float(tlabel[0])), apply_thresh(float(tlabel[1]))]
        if output_check[0] == labels_check[0] and output_check[1] == labels_check[1]:
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
    voutput = [list(foo) for foo in output.detach().numpy()]
    vlabels = [list(foo) for foo in labels.detach().numpy()]
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
    
    data = voutput_0.cpu().detach().numpy().tolist()

    # Input is a signal (pred_prob_tp)
    if vlabels[0] == 1:
        save_data([[data]], save_tp)
    # Input is noise (pred_prob_tn)
    if vlabels[1] == 1:
        save_data([[data]], save_tn)


def save_batch_to_hdf(dataloader, store_dir, id_offset=0):
    # Iterate through the DataLoader and save all samples in HDF5 format
    # We also have to save target for creating trainable.hdf
    targets = []
    all_abspaths = []
    """ Store Trainable Training/Validation Data """
    pbar = tqdm(dataloader)
    for n, (samples, labels) in enumerate(pbar):
        # We need to detach from cuda and use .cpu() to access host memory
        # example: if batch_size if 100, labels and samples will have have dims (100, len_data)
        # if slice is provided it splits the batch or gets element of batch
        samples = samples.cpu().detach().numpy()[:]
        labels = labels.cpu().detach().numpy()[:]
        # Saving target and path (this will be a list of np arrays of labels from each batch)
        targets.append(labels.tolist())
        # Iterate throught the trainDL and store all trainable training data in HDF5 format
        store_path = os.path.join(store_dir, "trainable_{}.hdf".format(n+id_offset))
        pbar.set_description("Processing nsample {} into {}".format(n, "trainable_{}.hdf".format(n+id_offset)))
        # Store path (one path for each batch)
        all_abspaths.append(os.path.abspath(store_path))
        # HDF5 was measured to have the fastest IO (r->46ms, w->172ms)
        # NPY read/write was not tested. Github says HDF5 is faster.
        with h5py.File(store_path, 'a') as fp:
            # create a dataset for batch save
            dst = fp.create_dataset("data", shape=samples.shape, dtype=np.float64, 
                                    data=samples,
                                    compression='gzip',
                                    compression_opts=9, 
                                    shuffle=True)
                
            # fill the <batch_size> number of slots with sample data
            for nsample, sample in enumerate(samples):
                dst[nsample] = sample
        
    return targets, all_abspaths, n+1


def save_trainable_dataset(cfg, data_cfg, trainDL, validDL):
    # Saving trainable dataset after transforms
    # Path to store trainable dataset
    parent_dir = os.path.join(data_cfg.parent_dir, data_cfg.data_dir)
    store_dir = os.path.join(parent_dir, "trainable_batched_dataset")
    
    if not os.path.exists(store_dir):
        os.makedirs(store_dir, exist_ok=False)
    
    """ Store trainable data for training and validation """
    print("Running pipeline using trainable option. Saving transformed trainable dataset.")
    # Training and validation will the saved in the same place and split (no need to distinguish)
    targets_train, train_abspaths, offset = save_batch_to_hdf(trainDL, store_dir)
    targets_valid, valid_abspaths, _ = save_batch_to_hdf(validDL, store_dir, offset)
    
    """ Save trainable.json for lookup """
    targets = targets_train + targets_valid
    all_abspaths = train_abspaths + valid_abspaths
    ## Creating trainable.json similar to training.hdf
    ids = np.arange(len(targets)).tolist() # un-JSONified version (np array is not JSON serializable)
    # Shuffling is not required as the DataLoader should have already shuffled it
    # Save the lookup table as a json file (this works better for batch saving)
    lookup = {'ids': ids, 'path': all_abspaths, 'target': targets, 'batch_size': cfg.batch_size}
    # Save the dataset paths alongside the target and ids as .JSON
    lookup_trainable = os.path.join(cfg.export_dir, "trainable.json")
    with open(lookup_trainable, 'w') as fp:
        json.dump(lookup, fp)
    # Create a copy of trainable.json to the dataset directory
    shutil.copy(lookup_trainable, data_cfg.parent_dir)
    print("manual.py: Trainable dataset has been created and stored!")
    

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
    

def train(cfg, data_cfg, Network, optimizer, scheduler, loss_function, trainDL, validDL, 
          verbose=False, trainable_dataset_save_mode=False):
    
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
    
    """ Trainable data storage snippet """
    # Create the necessary directories
    if trainable_dataset_save_mode:
        save_trainable_dataset(cfg, data_cfg, trainDL, validDL)
        # Return and leave if trainable data storage is complete
        print("manual.py: Trainable data storage complete. Train with simple DataLoader.")
        # Training and validation is not done when this option is on.
        return
        
    
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
            for training_samples, training_labels in pbar:
                
                batch_training_loss = 0.
                accuracies = []
                # Get all mini-folds and run training phase for each batch
                # Here each batch is cfg.batch_size. Each mini-fold contains multiple batches
                if cfg.dataset.__name__ == "BatchLoader":
                    # Convert the training_sample into a Simple dataset object
                    batch_train_dataset = Simple(training_samples, training_labels, 
                                           store_device=cfg.store_device, 
                                           train_device=cfg.train_device)
                    # Pass Simple dataset into a dataloader with cfg.batch_size
                    batch_train_loader = D.DataLoader(
                        batch_train_dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)
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
                for validation_samples, validation_labels in pbar:
                    
                    batch_validation_loss = 0.
                    accuracies = []
                    pred_prob = []
                    # Get all mini-folds and run training phase for each batch
                    # Here each batch is cfg.batch_size. Each mini-fold contains multiple batches
                    if cfg.dataset.__name__ == "BatchLoader":
                        # Convert the training_sample into a Simple dataset object
                        batch_valid_dataset = Simple(validation_samples, validation_labels, 
                                               store_device=cfg.store_device, 
                                               train_device=cfg.train_device)
                        # Pass Simple dataset into a dataloader with cfg.batch_size
                        batch_valid_loader = D.DataLoader(
                            batch_valid_dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)
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
                            pred_prob.append(preds)
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
                        pred_prob.append(preds)
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
                        coutput = [apply_thresh(float(voutput[0])), apply_thresh(float(voutput[1]))]
                        clabel = [apply_thresh(float(vlabel[0])), apply_thresh(float(vlabel[1]))]
                        
                        """ Prediction Probabilties """
                        # Storing predicted probabilities
                        if nep % cfg.save_freq == 0 and nep!=0:
                            prediction_probability_save_data(nep, vlabel, voutput[0], cfg.export_dir)
                            
                        """ Confusion Matrix """
                        # True
                        if coutput == clabel:
                            # Positive
                            if coutput[0] == 1:
                                tp += 1
                            # Negative
                            elif coutput[0] == 0:
                                tn += 1
                        # False
                        elif coutput != clabel and coutput[0] != coutput[1]:
                            # Positive
                            if coutput[0] == 1:
                                fp += 1
                            # Negative
                            elif coutput[0] == 0:
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
                weights_save_path = os.path.join(cfg.export_dir, cfg.model_params['weights_path'])
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
