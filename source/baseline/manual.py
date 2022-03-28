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
import torch
from tqdm import tqdm


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
    with open('name', 'a', newline='') as fp:
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
    
    # Input is a signal (pred_prob_tp)
    if vlabels[0] == 1:
        save_data(voutput_0.cpu().detach().numpy().tolist(), save_tp)
    # Input is noise (pred_prob_tn)
    if vlabels[1] == 1:
        save_data(voutput_0.cpu().detach().numpy().tolist(), save_tn)


def train(cfg, Network, optimizer, scheduler, loss_function, trainDL, validDL, verbose=False):
    
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
    
    with open(os.path.join(cfg.export_dir, cfg.output_loss_file), 'w') as outfile:

        ### Training loop
        best_loss = 1.e10 # impossibly bad value

        for nep, epoch in enumerate(range(cfg.num_epochs)):
            
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
            print("\n\nTraining Phase Initiated")
            for training_samples, training_labels in trainDL:
                # Optimizer step on a single batch of training data
                optimizer.zero_grad()
                # Obtain training output from network
                training_output = Network(training_samples)
                # Loss calculation
                training_loss = loss_function(training_output, training_labels)
                # Accuracy calculation
                accuracy = calculate_accuracy(training_output, training_labels, cfg.accuracy_thresh)
                # Backward propogation using loss_function
                training_loss.backward()
                # Clip gradients to make convergence somewhat easier
                torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=cfg.clip_norm)
                # Make the actual optimizer step and save the batch loss
                optimizer.step()
                training_running_loss += training_loss.clone().cpu().item()
                training_batches += 1
                
                print("\nEpoch {}: Training Loss = {}".format(nep, training_loss))
                print("Epoch {}: Training Accuracy = {}".format(nep, accuracy))
                acc_train.append(accuracy)

            
            """
            PHASE 2 - Validation
                [1] Save confusion matrix elements and prediction probabilties
                [2] Save the ROC save data
            """            
            print("Validation Phase Initiated")
            # Evaluation on the validation dataset
            Network.eval()
            with torch.no_grad():
                
                validation_running_loss = 0.
                validation_batches = 0

                for validation_samples, validation_labels in validDL:
                    # Evaluation of a single validation batch
                    validation_output = Network(validation_samples)
                    validation_loss = loss_function(validation_output, validation_labels)
                    # Accuracy calculation
                    accuracy = calculate_accuracy(validation_output, validation_labels, cfg.accuracy_thresh)
                    # Update running loss and batches
                    validation_running_loss += validation_loss.clone().cpu().item()
                    validation_batches += 1
                    
                    print("\nEpoch {}: Validation Loss = {}".format(nep, validation_loss))
                    print("Epoch {}: Validation Accuracy = {}".format(nep, accuracy))
                    acc_valid.append(accuracy)

                    """ ROC Curve save data """
                    if nep % cfg.save_freq == 0:
                        roc_save_data(nep, validation_output, validation_labels, cfg.export_dir)

                    """ Calculating confusion matrix and Pred Probs """
                    apply_thresh = lambda x: round(x - cfg.accuracy_thresh + 0.5)
                    for voutput, vlabel in zip(validation_output, validation_labels):
                        # Get labels based on threshold
                        coutput = [apply_thresh(float(voutput[0])), apply_thresh(float(voutput[1]))]
                        clabel = [apply_thresh(float(vlabel[0])), apply_thresh(float(vlabel[1]))]
                        
                        """ Prediction Probabilties """
                        # Storing predicted probabilities
                        if nep % cfg.save_freq == 0:
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
            validation_loss = validation_running_loss/validation_batches
            training_loss = training_running_loss/training_batches
            avg_acc_valid = sum(acc_valid)/len(acc_valid)
            avg_acc_train = sum(acc_train)/len(acc_train)
            
            # Save output string in losses.txt
            output_string = '%04i    %f    %f    %f    %f    %f    %f    %f    %f' % \
                            (epoch, training_loss, validation_loss,
                             avg_acc_train, avg_acc_valid, tp, tn, fp, fn)
                            
            outfile.write(output_string + '\n')
            
            """ Save the best weights (if global loss reduces) """
            if validation_loss < best_loss:
                torch.save(Network.state_dict(), cfg.model_params['weights_path'])
                best_loss = validation_loss


        print("Training complete with best validation loss = {}".format(best_loss))
