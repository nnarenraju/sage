# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Nov 26 21:24:31 2021

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

# Future imports
from __future__ import annotations

# CNN-1D dependancies
import torch
import torch.nn as nn
import pytorch_lightning as pl

from collections.abc import Callable

class CNN_1D(pl.LightningModule):
    """
    3-Channel 1D CNN Model
    
    Calculating the output shape:
        output_dim = ((W-K+2.0*P)/S)+1
        W - Input volume
        K - Kernel size
        P - Padding
        S - Stride
        (Round-down if the result is not an integer)
    
    Example for G2NET dataset:
        2 second time-series sampled at 2048 Hz
        W = 4096 samples
        K = 5 kernel size
        P = (5 - 1)//2 = 2
        S = 4
        output_dim = ((4096-5+2.0*2)/4.0)+1 = 1024.75 = 1024 (round-down)
        output_shape = (3 channels, 128 out_channels, 1024 output_dim)
                     = (3, 128, 1024) for each batch
    
    Stride Selection for down-sampling:
        0.1 seconds = (1/10) * 2048 = 204.8 samples
        To keep loss to a minium, set stride = 2 (max.)
        Keep stride = 1 to preserve input shape
        If padding == "same", no striding is allowed
    
    Link to Weight-Initialisation Techniques:
    https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    TL;DR:
        [1] Use Xavier and normalised Xavier for Sigmoid and Tanh activation
        [2] Use Kaiming Normal (aka. He Weight Initialisation) for ReLU activation
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    num_channels : tuple
        Number of channels in each hidden layer
    kernel_size : int
        Kernel size for all layers that use it (set to odd num by convention)
    stride : int
        Stride for layers that require it
    force_out_size : int | tuple
        Force output size of CNN using AdaptiveAvgPool2d
    check_out_size : bool
        Display output size once for sanity check
    conv : Callable
        CNN main layer. Set to Conv1d. Changing this may result in errors.
    disable_AMP : bool
        Enable/disable Automatic Mixed Precision
    weight_init : bool
        Initialise weights for Conv1d and BatchNorm layers
    
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 num_channels: tuple = (32, 64, 128), 
                 kernel_size: int = 3, 
                 stride: int = 2,
                 force_out_size: int | tuple = None,
                 check_out_size: bool = False,
                 conv: Callable = nn.Conv1d,
                 disable_AMP: bool = False, 
                 weight_init: bool = True):

        super().__init__()
        
        # Set variables
        self.out_channels = out_channels
        self.num_layers = len(num_channels)
        self.force_out_size = force_out_size
        self.check_out_size = check_out_size
        self.disable_AMP = disable_AMP
        # Create module list object for backend
        self.backend = nn.ModuleList()
        
        
        """" Create the Model Architecture """
        for channel_num in range(self.out_channels):
            # For each channel the tmp_block init should be the same
            """ Input Layer (CNN-1D) """
            tmp_block = [
                conv(
                   in_channels, 
                   num_channels[0],
                   kernel_size=kernel_size,
                   padding='same')
                ]    
            
            """ All hidden layers and output layer """
            # Limit is set to num_layers - 1 as tmp_block is included
            # Last layer will have output channels = num_channels[-1]
            for layer_num in range(self.num_layers-1):
                # Append layers together
                tmp_block = tmp_block + [
                    nn.BatchNorm1d(num_channels[layer_num]),
                    nn.SiLU(inplace=True),
                    nn.MaxPool1d(kernel_size=3, stride=4),
                    conv(
                        num_channels[layer_num], 
                        num_channels[layer_num+1],
                        kernel_size=kernel_size,
                        padding='same')
                ]
                
            self.backend.append(nn.Sequential(*tmp_block))
        
        """ Force output size """
        if self.force_out_size is not None:
            if isinstance(self.force_out_size, int):
                # Use this to preserve number of one dimension
                self.pool = nn.AdaptiveAvgPool2d((None, self.force_out_size))
            else:
                # force_out_size is a tuple
                self.pool = nn.AdaptiveAvgPool2d(self.force_out_size)
            
        """ Weight-initialisation """
        if weight_init:
            # self.modules is a part of super
            for module in self.modules():
                if isinstance(module, nn.Conv1d):
                    # Since we have ReLU activation, use He init
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm1d):
                    # Initialise with mean = 1, variance = 0
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)


    def forward(self, x):
        out = []
        """ Enable/disable Automatic Mixed Precision """
        # (if enabled) np.float16 and np.float32 is used for modules that need it
        # (disabled) computation may be slower
        if self.disable_AMP: 
            with torch.cuda.amp.autocast(enabled=False):
                for channel_num in range(self.out_channels):
                    out.append(self.backend[channel_num](x))
        else:
            for channel_num in range(self.out_channels):
                out.append(self.backend[channel_num](x))
        
        """ Stack output of all channels together """
        out = torch.stack(out, dim=1)
        
        """ Force output size """
        if self.force_out_size is not None:
            out = self.pool(out)
            
        # Checking size of output
        if self.check_out_size:
            print("\nOutput shape from frontend = {}".format(out.shape))
            self.check_out_size = False
        
        return out
