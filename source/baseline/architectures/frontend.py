# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Nov 21 17:19:53 2021

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

import timm
import pytorch_lightning as pl

class AlphaModel(pl.LightningModule):
    """
    Alpha-type Model Architecture
    
    Description - consists of a 3-channel trainable backend and 3-channel timm frontend
    
    Parameters
    ----------
    model_name  = 'resnet34' : string
        Timm model name for Frontend
    pretrained  = False : Bool
        Pretrained option for Timm models
    num_classes = 1 : int
        Number of classes to Timm Model
    timm_params = {} : dict
        Parameters passed to Timm Model
        Link: https://fastai.github.io/timmdocs/training#The-training-script-in-20-steps
        Look at "The Training Script in 20 Steps" for more details.
    backend = None : class
        Backend of Alpha-type Model architecture
    backend_params = {} : dict
        Parameters provided to trainable backend
        
    """

    def __init__(self, 
                 model_name='resnet34', 
                 pretrained=False, 
                 num_classes=1,
                 timm_params={},
                 backend=None,
                 backend_params={}):
        
        super().__init__()
        
        # Initialise Backend Model
        self.backend = backend(**backend_params)
        # Initialise Frontend Model
        self.frontend = timm.create_model(model_name,
                                          pretrained=pretrained, 
                                          num_classes=num_classes,
                                          **timm_params)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Trainable backend
        backend = self.backend(x)
        # Append backend to frontend
        out = self.frontend(backend)
        return out


class BetaModel(pl.LightningModule):
    """
    Beta-type Model Architecture
    
    Description - consists of a 3-channel timm frontend (no backend)
    
    Parameters
    ----------
    model_name  = 'resnet34' : string
        Timm model name for Frontend
    pretrained  = False : Bool
        Pretrained option for Timm models (keep this False)
    num_classes = 1 : int
        Number of classes to Timm Model
    timm_params = {} : dict
        Parameters passed to Timm Model
        Link: https://fastai.github.io/timmdocs/training#The-training-script-in-20-steps
        Look at "The Training Script in 20 Steps" for more details.
        
    """

    def __init__(self, 
                 model_name='resnet34', 
                 pretrained=False, 
                 num_classes=1,
                 timm_params={}):
        
        super().__init__()
        
        # Initialise Frontend Model
        self.frontend = timm.create_model(model_name,
                                          pretrained=pretrained, 
                                          num_classes=num_classes,
                                          **timm_params)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Timm frontend (no backend)
        out = self.frontend(x)
        return out
