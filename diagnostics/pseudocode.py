# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = pseudocode.py
Description     = Pseudocode for the MLMDC1 pipeline

Created on Sat Dec 11 10:03:59 2021

__author__      = nnarenraju
__copyright__   = Copyright 2021, ML-MDC1
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = inProgress


Github Repository: NULL


Timeline:
    
    November (No deadline)
    ----------------------
    0. Learning and reading
    1. Data preparation
    
    December (deadline 31st, 23:59 PM BST)
    --------------------------------------
    1. Continuing Data Preparation
    2. Data Augmentation
    3. Multi-rate Sampling
    4. Use Dataset 4 (most realistic)
    5. Creating a ranking statistic
    6. Initialisation and configuration file
    7. Training and Validation on simplest network
    
    January (deadline 31st, 23:59 PM BST)
    -------------------------------------
    1. Data visualisation for all input and output
    2. Create a large dataset for training
    3. Experiment on different architectures and models
    4. Hyper-parameter tuning
    5. Efficient parallelisation and speed
    6. Read all relevant rival papers and optimise
    
    February (deadline 28th, 23:59 PM BST)
    --------------------------------------
    1. Create testing/inference module
    2. Test the best model from January with 1 month data
    3. Optimise the model and data based on testing/inference
    
        By the end of the month, we should have a model that works reasonably well
        on any given 1 month testing data (in accuracy and runtime)
    
    March (deadline 15th, 23:59 PM BST)
    -----------------------------------
    1. Most optimal model possible in the given timescale
    2. Cleaning-up of all code for submission
    3. Think of the best way to submit the code/model to AEI
    
    March (deadline 25th, 23:59 PM BST)
    -----------------------------------
    1. Final code/model should be ready for full submission
    2. Documentation (if time permits)
    
    March (deadline 31st, 23:59 PM BST)
    -----------------------------------
    Submission.
    
    
    

Pseudo-code:
    
    Data preparation
    ----------------
    
    1a. Creating a dataset using PyCBC using a fixed length of segment
        i. Modify generate_data.py testing set generation code to create training set
    1b. Make sure the dataset is similar to dataset 3/4 in the challenge
    1c. Save each segment in the HDF5 format
    2. Obtain the paths to each segment by using efficient IDs
    3. Get the targets/labels for each segment (signal/noise)
    4. Create a large HDF5 file containing the details (ID, path, target)
    5. Efficiently split the dataset into training and validation sets
    6. Reading and accessing the items in a given dataset should be fast
    7. Does C++ help with this stage? HDF5 handling is much faster in C++
    8. If [7] is true, does it have any overhead when called from Python
    9. Can [7] be implemented without calling from Python
    
    Data Augmentation
    -----------------
    1.
    2.
    3.
    
    Multi-rate Sampling of the input data
    -------------------------------------
    1.
    2.
    3.
    
    Convert pipeline to use Dataset 4 instead of Dataset 3
    ------------------------------------------------------
    1.
    2.
    3.
    
    Create a Ranking Statistic for the Prediction Stage
    ---------------------------------------------------
    1.
    2.
    3.
    
    Initialisation & Configuration File
    -----------------------------------
    
    1. Current configuration file is a Class (objects are slow)
    2. Using the number of calls to config, use the most efficient config format
    3. Add details to config format to experiment with every aspect of the code
    4. Use a Config and Param file like AREPO
    5. Implement a source, exec, input, output format at appropriate locations
    
    Reading and manipulating the data
    ---------------------------------
    
    1. Use optimum HDF5 handling to read in the given data
    2. Data needs to be converted to numpy or Tensor for further usage
    3. Apply appropriate transforms when data is present in efficient format
    4. Optimise batches and parallel handling
    
    Training and Validation
    -----------------------
    
    1. Optimise Pytorch-Lightning as much as possible for fast computation
    2. Read PyTorch documentation to use fast formats for all Tensors
    3. Logging and verbosity should be implemented alongside time profiling
    4. Saving checkpoints, early-stopping, snapshots and other callbacks
    5. Output format for TensorBoard at optimum times to avoid memory overhead
    6. Check for overfitting and optimise
    
    Model Architectures
    -------------------
    
    1. Efficiently test many model architectures from timm
    2. Write our own architecture using PyTorch (use tutorials)
    3. What model is the fastest and which model is the most efficient?
    4. Save all model architectures in an appropriate format
    5. Hyper-parameter tuning for Pytorch (do this for each architecture)
    
    Testing Data Preparation
    ------------------------
    
    1. Efficiently split the 1 month data using "tc" or other formats
    2. The data can be used for training/validation purposes as well
    3. Each split should be stored in the same format as mentioned in Data Prep section
    4. Concatenate testing section into Config and Param
    5. Concatenate testing data prep into all other data handling modules
    
    
    
"""
    
    
