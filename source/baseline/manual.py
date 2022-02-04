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
import tqdm
import torch

# LOCAL
from losses.custom_loss_functions import reg_BCELoss


def train(Network, training_dataloader, validation_dataloader, output_training,
          weights_path, store_device='cuda:0', train_device='cuda:0',
          batch_size=32, learning_rate=5e-5, epochs=100, clip_norm=100,
          verbose=False):
    
    """Train a network on given data.
    
    Arguments
    ---------
    Network : network as returned by get_network
        The network to train.
    training_dataloader : object
        The dataloader to use for training.
    validation_dataloader : object
        The dataloader to use for validation.
    output_training : str
        Path to a directory in which the loss history and the best
        network weights will be stored.
    weights_path: str
        Path where the trained network weights will be stored.
    store_device : {str, 'cpu'/'cuda:0'}
        The device on which the data sets should be stored.
    train_device : {str, 'cpu'/'cuda:0'}
        The device on which the network should be trained.
    batch_size : {int, 32}
        The mini-batch size used for training the network.
    learning_rate : {float, 5e-5}
        The learning rate to use with the optimizer.
    epochs : {int, 100}
        The number of full passes over the training data.
    clip_norm : {float, 100}
        The value at which to clip the gradient to prevent exploding
        gradients.
    verbose : {bool, False}
        Print update messages.
    
    Returns
    -------
    network
    """
    ### Set up data loaders as a PyTorch convenience
    print("Setting up datasets and data loaders.")
    # Get data loader
    TrainDL = training_dataloader
    ValidDL = validation_dataloader

    ### Initialize loss function, optimizer and output file
    print("Initializing loss function, optimizer and output file.")
    loss = reg_BCELoss(dim=2)
    opt = torch.optim.Adam(Network.parameters(), lr=learning_rate)
    with open(os.path.join(output_training, 'losses.txt'), 'w') as outfile:

        ### Training loop
        best_loss = 1.0e10 # impossibly bad value
        iterable1 = range(1, epochs+1)
        iterable1 = tqdm(iterable1, desc="Optimizing network") if verbose else iterable1
        for epoch in iterable1:
            # Training epoch
            Network.train()
            training_running_loss = 0.0
            training_batches = 0
            iterable2 = TrainDL
            iterable2 = tqdm(iterable2, desc="Iterating over training dataset", leave=False) if verbose else iterable2
            for training_samples, training_labels in iterable2:
                # Optimizer step on a single batch of training data
                opt.zero_grad()
                training_output = Network(training_samples)
                training_loss = loss(training_output, training_labels)
                training_loss.backward()
                # Clip gradients to make convergence somewhat easier
                torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=clip_norm)
                # Make the actual optimizer step and save the batch loss
                opt.step()
                training_running_loss += training_loss.clone().cpu().item()
                training_batches += 1
            
            # Evaluation on the validation dataset
            Network.eval()
            with torch.no_grad():
                validation_running_loss = 0.
                validation_batches = 0
                iterable2 = ValidDL
                iterable2 = tqdm(iterable2, desc="Computing validation loss", leave=False) if verbose else iterable2
                for validation_samples, validation_labels in iterable2:
                    # Evaluation of a single validation batch
                    validation_output = Network(validation_samples)
                    validation_loss = loss(validation_output, validation_labels)
                    validation_running_loss += validation_loss.clone().cpu().item()
                    validation_batches += 1
            # Print information on the training and validation loss in the current epoch and save current network state
            validation_loss = validation_running_loss/validation_batches
            output_string = '%04i    %f    %f' % (epoch, training_running_loss/training_batches, validation_loss)
            outfile.write(output_string + '\n')
            # Save 
            if validation_loss<best_loss:
                torch.save(Network.state_dict(), weights_path)
                best_loss = validation_loss

        print("Training complete with best validation loss = {}".format(best_loss))
    
    Network.load_state_dict(torch.load(weights_path))
    return Network
