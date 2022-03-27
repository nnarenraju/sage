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

# IN-BUILT
import os
import timm
import torch
import pytorch_lightning as pl

# Datatype for storage
data_type=torch.float32

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


class GammaModel(pl.LightningModule):
    """
    Gamma-type Model Architecture
    
    Description - consists of a 2-channel simple NN frontend (no backend)
    
    Parameters
    ----------
    model_name  = 'simple' : string
        Simple NN model name for Frontend. Save model with this name as attribute.
    pretrained  = False : Bool
        Pretrained option for saved models
        If True, weights are stored under the model_name in saved_models dir
        If model name already exists, throws an error (safety)
    in_channels = 2 : int
        Number of input channels (number of detectors)
    out_channels = 2 : int
        Number of output channels (signal, noise)
    store_device = 'cpu' : str
        Storage device for network (NOTE: make sure data is also stored in the same device)
    weights_path = '' : str
        Absolute path to the weights.pt file. Used when pretrained == True
        
    """

    def __init__(self, 
                 model_name='simple', 
                 pretrained=False,
                 in_channels: int = 2,
                 out_channels: int = 2,
                 store_device='cpu',
                 weights_path: str = ''):
        
        super().__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.store_device = store_device
        self.weights_path = weights_path
        
        # Initialise Frontend Model
        # Add the following line as last layer if softmax is needed
        # torch.nn.Softmax(dim=1) --> (signal, noise)
        self.frontend = torch.nn.Sequential(                # Shapes
                torch.nn.BatchNorm1d(self.in_channels),     #  2x2048
                torch.nn.Conv1d(2, 4, 64),                  #  4x1985
                torch.nn.ELU(),                             #  4x1985
                torch.nn.Conv1d(4, 4, 32),                  #  4x1954
                torch.nn.MaxPool1d(4),                      #  4x 489
                torch.nn.ELU(),                             #  4x 489
                torch.nn.Conv1d(4, 8, 32),                  #  8x 458
                torch.nn.ELU(),                             #  8x 458
                torch.nn.Conv1d(8, 8, 16),                  #  8x 443
                torch.nn.MaxPool1d(3),                      #  8x 147
                torch.nn.ELU(),                             #  8x 147
                torch.nn.Conv1d(8, 16, 16),                 # 16x 132
                torch.nn.ELU(),                             # 16x 132
                torch.nn.Conv1d(16, 16, 16),                # 16x 117
                torch.nn.MaxPool1d(4),                      # 16x  29
                torch.nn.ELU(),                             # 16x  29
                torch.nn.Flatten(),                         #     464
                torch.nn.Linear(464, 32),                   #      32
                torch.nn.Dropout(p=0.5),                    #      32
                torch.nn.ELU(),                             #      32
                torch.nn.Linear(32, 16),                    #      16
                torch.nn.Dropout(p=0.5),                    #      16
                torch.nn.ELU(),                             #      16
                torch.nn.Linear(16, self.out_channels),     #       2
                torch.nn.Softmax(dim=1)                     #       2
        )
        
        # Pretrained and saved weights
        if pretrained and self.weights_path != '':
            if os.path.exists(self.weights_path):
                self.frontend.load_state_dict(torch.load(self.weights_path))
            else:
                raise ValueError("GammaModel: weights_path does not exist!")
        
        # Convert network into given dtype and store in proper device
        self.frontend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Simple NN frontend (no backend)
        out = self.frontend(x)
        return out




""" FOR MANUAL USAGE ONLY, *DO NOT* USE WITH PYTORCH LIGHTNING """

### Set data type to be used
dtype = torch.float32

def get_sample_network(path=None, device='cuda:0'):
    """Return an instance of a network.
    
    Arguments
    ---------
    path : {None or str, None}
        Path to the network (weights) that should be loaded. If None
        a new network will be initialized.
    device : {str, 'cpu'}
        The device on which the network is located.
    
    Returns
    -------
    network
    """

    frontend = torch.nn.Sequential(                     #  Shapes
            torch.nn.BatchNorm1d(2),                    #  2x40960
            torch.nn.Conv1d(2, 4, 64, 2),               #  4x20449
            torch.nn.ReLU(),                            #  4x20449
            torch.nn.Conv1d(4, 4, 32, 2),               #  4x10209
            torch.nn.MaxPool1d(4),                      #  4x 2552
            torch.nn.ReLU(),                            #  4x 2552
            torch.nn.Conv1d(4, 8, 32, 2),               #  8x 1261
            torch.nn.ReLU(),                            #  8x 1261
            torch.nn.Conv1d(8, 8, 16, 2),               #  8x 623
            torch.nn.MaxPool1d(3),                      #  8x 207
            torch.nn.ReLU(),                            #  8x 207
            torch.nn.Conv1d(8, 16, 16),                 # 16x 192
            torch.nn.ReLU(),                            # 16x 192
            torch.nn.Conv1d(16, 16, 16),                # 16x 177
            torch.nn.MaxPool1d(4),                      # 16x  44
            torch.nn.ReLU(),                            # 16x  44
            torch.nn.Flatten(),                         #     704
            torch.nn.Linear(704, 32),                   #      32
            torch.nn.Dropout(p=0.5),                    #      32
            torch.nn.ReLU(),                            #      32
            torch.nn.Linear(32, 16),                    #      16
            torch.nn.Dropout(p=0.5),                    #      16
            torch.nn.ReLU(),                            #      16
            torch.nn.Linear(16, 1),                     #       1
            torch.nn.Softmax(dim=1)                     #       1
    )
    
    if path is not None:
        frontend.load_state_dict(torch.load(path))
    frontend.to(dtype=dtype, device=device)
    return frontend
