#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Wed May  4 12:47:03 2022

__author__      = nnarenraju
__copyright__   = Copyright 2022, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation:
    
    import matplotlib.pyplot as plt
    # Schedulers
    scheduler_1 = ExponentialGrowth(num_epochs=20, start_value=0.0001, end_value=0.2)
    scheduler_2 = ExponentialGrowth(num_epochs=20, start_value=0.2, end_value=0.9)
    # Apply functions
    prepend_zeroes = PrependZeros(num_epochs=5)
    append_ones = AppendOnes(num_epochs=10)
    append_val_1 = AppendVals(num_epochs=20, value=0.2)
    append_val_2 = AppendVals(num_epochs=20, value=0.9)
    # Sequential application of all sections
    seq = Sequential(scheduler_1, 
                     prepend_zeroes, 
                     append_val_1,
                     scheduler_2,
                     append_val_2,
                     append_ones)

    # Run sequential
    out = seq()
    plt.plot(out)

"""

# Modules
import numpy as np

class Sequential:
    """
    Apply all classes sequentially
    
    Parameters
    ----------
    apply_classes : list
        All classes to be applied sequentially provided as args
    
    Return
    ------
    out : np.ndarray
        Array containing loss scheduler values after applying sequence
    
    """
    
    def __init__(self, *apply_classes):
        self.apply_classes = list(apply_classes)
        
    def __call__(self):
        out = self.apply_classes[0]()
        for _class in self.apply_classes[1:]:
            out = _class(out)
        return out


class AppendOnes:
    """
    Append Ones into np.ndarray loss scheduler output
    
    Parameters
    ----------
    num_epochs : int
        Number of epochs to append ones
    
    Return
    ------
    prefix : np.ndarray
        Loss scheduler values with ones appended to it
    
    """
    
    
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        
    def __call__(self, scheduler_vals):
        # Append ones to the above prefix variables
        return np.append(scheduler_vals, [1.0]*self.num_epochs)


class AppendVals:
    """
    Append n-Vals into np.ndarray loss scheduler output
    
    Parameters
    ----------
    num_epochs : int
        Number of epochs to append given value
    value : float
        value to be appended
    
    Return
    ------
    prefix : np.ndarray
        Loss scheduler values with value appended to it
    
    """
    
    
    def __init__(self, num_epochs, value):
        self.num_epochs = num_epochs
        self.value = value
        
    def __call__(self, scheduler_vals):
        # Append ones to the above prefix variables
        return np.append(scheduler_vals, [self.value]*self.num_epochs)
    

class PrependZeros:
    """
    Prepend zeros into np.ndarray loss scheduler output
    
    Parameters
    ----------
    num_epochs : int
        Number of epochs to prepend zeroes
    
    Return
    ------
    prefix : np.ndarray
        Loss scheduler values with zeroes prepending to it
    
    """
    
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        
    def __call__(self, scheduler_vals):
        # Prepend zeroes
        prefix = np.insert(scheduler_vals, 0, [0.0]*self.num_epochs)
        return prefix
    

class ExponentialGrowth:
    """
    Exponential growth function
    Range = [start_value, end_value]
    
    Parameters
    ----------
    num_epochs : int
        Traverse the [start_value, end_value] linear space within the given number of epochs
    start_value : float
        Starting value for the exponential growth function
        Should be less than end_value
    end_value : float
        Starting value for the exponential growth function
    
    Return
    ------
    _ : np.ndarray
        Array containing loss scheduler values
        
    """
    
    def __init__(self, num_epochs=5, start_value=0.01, end_value=1.0):
        assert start_value < end_value
        self.num_epochs = num_epochs
        self.start_value = start_value
        self.end_value = end_value
        
        # Estimated growth rate for the given number of epochs and start_value
        self.growth_rate = (self.end_value/self.start_value)**(1./(self.num_epochs-1)) - 1.0
        # Loss prefix term grows using exponential growth function
        self.exp_growth = lambda epoch: self.start_value * (1.0 + self.growth_rate)**epoch
        
    def __call__(self, scheduler_vals=None):
        # Return np array of exp growth values
        if not isinstance(scheduler_vals, np.ndarray):
            return self.exp_growth(np.arange(self.num_epochs))
        else:
            return np.append(scheduler_vals, self.exp_growth(np.arange(self.num_epochs)))
