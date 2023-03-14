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

Documentation

1, 128 filters, size 64, maxpool 16 (128, 8000/16)
Or use a fixed t-f representation


"""

# IN-BUILT
import timm
import torch
import numpy as np
from torch import nn
from scipy import signal

# Importing architecture snippets from zoo
from architectures.zoo.dain import DAIN_Layer
from architectures.zoo.kaggle import ConvBlock, _initialize_weights, ConvBlock_Apr21

# Datatype for storage
data_type=torch.float32


class AlphaModel(torch.nn.Module):
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
                 model_name='fixed_backend_trainable_frontend', 
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


class BetaModel(torch.nn.Module):
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
                 model_name='lonely_timm',
                 num_classes=1,
                 timm_params={}):
        
        super().__init__()
        
        # Initialise Frontend Model
        self.frontend = timm.create_model(**timm_params)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Timm frontend (no backend)
        out = self.frontend(x)
        return out


class GammaModel(torch.nn.Module):
    """
    Gamma-type Model Architecture
    
    Description - consists of a 2-channel simple NN frontend (no backend)
    
    Parameters
    ----------
    model_name  = 'simple' : string
        Simple NN model name for Frontend. Save model with this name as attribute.
    in_channels = 2 : int
        Number of input channels (number of detectors)
    out_channels = 2 : int
        Number of output channels (signal, noise)
    store_device = 'cpu' : str
        Storage device for network (NOTE: make sure data is also stored in the same device)
        
    """

    def __init__(self, 
                 model_name='simple', 
                 in_channels: int = 2,
                 out_channels: int = 2,
                 flatten_size: int = 1088,
                 store_device: str = 'cpu'):
        
        super().__init__()
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flatten_size = flatten_size
        self.store_device = store_device
        
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
                torch.nn.Flatten(),                         #      xx
                torch.nn.Linear(self.flatten_size, 32),     #      32
                torch.nn.Dropout(p=0.5),                    #      32
                torch.nn.ELU(),                             #      32
                torch.nn.Linear(32, 16),                    #      16
                torch.nn.Dropout(p=0.5),                    #      16
                torch.nn.ELU(),                             #      16
                torch.nn.Linear(16, self.out_channels),     #       2/1
        )
        
        # Convert network into given dtype and store in proper device
        self.frontend.to(dtype=data_type, device=self.store_device)
        self.sigmoid = torch.nn.Sigmoid()
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Simple NN frontend (no backend)
        raw = self.frontend(x)
        pred_prob = self.sigmoid(raw)
        return {'pred_prob': pred_prob, 'raw': raw}


class GammaModelPE(torch.nn.Module):
    """
    Gamma-type Model PE Architecture
    
    Description - consists of a 2-channel simple NN frontend (no backend)
    
    Parameters
    ----------
    model_name  = 'simple' : string
        Simple NN model name for Frontend. Save model with this name as attribute.
    in_channels = 2 : int
        Number of input channels (number of detectors)
    out_channels = 2 : int
        Number of output channels (signal, noise)
    store_device = 'cpu' : str
        Storage device for network (NOTE: make sure data is also stored in the same device)
        
    """

    def __init__(self, 
                 model_name='simple', 
                 in_channels: int = 2,
                 out_channels: int = 2,
                 store_device: str = 'cpu'):
        
        super().__init__()
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.store_device = store_device
        
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
                torch.nn.Linear(1088, 32),                  #      32 - 1088 for 3712 sample len
                torch.nn.Dropout(p=0.5),                    #      32
                torch.nn.ELU(),                             #      32
                torch.nn.Linear(32, 16),                    #      16
                torch.nn.Dropout(p=0.5),                    #      16
                torch.nn.ELU()                              #      16
        )
        
        # Mod layers
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # PE layers
        self.signal_to_noise = nn.Linear(16, self.out_channels)
        self.coalescence_time = nn.Linear(16, 1)
        self.distance = nn.Linear(16, 1)
        self.chirp_mass = nn.Linear(16, 1)
        
        # Convert network into given dtype and store in proper device
        self.signal_to_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.frontend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Simple NN frontend (no backend)
        x = self.frontend(x)
        # Outputs
        pred_prob = self.softmax(self.signal_to_noise(x))
        tc = self.sigmoid(self.coalescence_time(x))
        distance = self.sigmoid(self.distance(x))
        mchirp = self.sigmoid(self.chirp_mass(x))
        # Return all outputs as dict
        return {'pred_prob': pred_prob, 'tc': tc, 'distance': distance, 'mchirp': mchirp}


class KappaModel(torch.nn.Module):
    """
    Kappa-type Model Architecture
    
    Description - consists of a 2-channel ConvBlock backend and a Timm model frontend
                  this Model-type can be used to test the Kaggle architectures
    
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
    
    Notes on necessary layers :-
    Double-check the definitions before usage
    
    nn.AdaptiveAvgPool1d(8):
        1. For an input.shape (1, 4, 64) --> output.shape (1, 4, 8)
        2. Expects an input of 2-3 dimesions
    nn.AdaptiveAvgPool2d((5, 5)):
        1. For an input.shape (1, 4, 64, 8) --> output.shape (1, 4, 5, 5)
        2. Can accept an input shape of 4-dimensions
    Essentially, the first two dims are untouched as these are (batch_size and num_channels)
    
    nn.Softmax(dim=1):
        1. Input shape is preserved in the output
        2. Output values should all add up to 1.0
    
    nn.Dropout(0.25):
        1. p=0.25 is the probability of an element to be zeroed
        2. Output is of the same shape as the input
        
    """

    def __init__(self, 
                 model_name='trainable_backend_and_frontend', 
                 filter_size: int = 16,
                 kernel_size: int = 32,
                 timm_params: dict = {'model_name': 'resnet34', 'pretrained': True, 'in_chans': 2, 'drop_rate': 0.25},
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.timm_params = timm_params
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        
        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size)
        _initialize_weights(self)
        
        """ Frontend """
        # resnet34 --> 21 Mil. trainable params trainable frontend
        self.frontend = timm.create_model(**timm_params)
        
        # reset_classifier edits the number of outputs that Timm produces
        # The following is set if we need a two-class output from Timm
        # This can be set to a larger value and connected to a linear layer
        self.frontend.reset_classifier(1)
        
        """ Mods """
        ## Penultimate and output layers
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(self.frontend.num_features)
        self.batchnorm = nn.BatchNorm1d(2)
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(0.25)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.ReLU = nn.ReLU()
        
        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.backend = {'det1': self._det1, 'det2': self._det2}
        self.frontend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Batch Normalisation
        x = self.batchnorm(x)
        # Conv Backend
        cnn_output = torch.cat([self.backend['det1'](x[:, 0:1]), self.backend['det2'](x[:, 1:2])], dim=1)
        # Timm Frontend
        x = self.frontend(cnn_output)
        ## Manipulate encoder output to get params
        # Global Pool
        # x = self.flatten(self.avg_pool_1d(x))
        # In the Kaggle architecture a dropout is added at this point
        # I see no reason to include at this stage. But we can experiment.
        ## Output necessary params
        # Use sigmoid here if not using BCEWithLogitsLoss
        pred_prob = self.sigmoid(x)
        # Return ouptut params (pred_prob)
        return {'raw': x, 'pred_prob': pred_prob, 'cnn_output': cnn_output}


class KappaModelPE(torch.nn.Module):
    """
    Kappa-type Model PE Architecture
    
    Description - consists of a 2-channel ConvBlock backend and a Timm model frontend
                  this Model-type can be used to test the Kaggle architectures
    
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
                 model_name='trainable_backend_and_frontend', 
                 filter_size: int = 32,
                 kernel_size: int = 64,
                 timm_params: dict = {'model_name': 'resnet34', 'pretrained': True, 'in_chans': 2, 'drop_rate': 0.25},
                 norm_layer: str = 'layernorm',
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.timm_params = timm_params
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.norm_layer = norm_layer
        
        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size)
        _initialize_weights(self)
        
        """ Frontend """
        # resnet34 --> 21 Mil. trainable params trainable frontend
        self.frontend = timm.create_model(**timm_params)
        
        """ Mods """
        ## Penultimate and output layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        # Primary outputs
        self.signal_or_noise = nn.Linear(self.frontend.num_features, 1)
        self.coalescence_time = nn.Linear(self.frontend.num_features, 1)
        self.chirp_distance = nn.Linear(self.frontend.num_features, 1)
        self.chirp_mass = nn.Linear(self.frontend.num_features, 1)
        self.distance = nn.Linear(self.frontend.num_features, 1)
        self.mass_ratio = nn.Linear(self.frontend.num_features, 1)
        self.inv_mass_ratio = nn.Linear(self.frontend.num_features, 1)
        self.snr = nn.Linear(self.frontend.num_features, 1)
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, 4830])
        self.layernorm_cnn = nn.LayerNorm([2, 128, 150])
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(self.frontend.num_features)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        
        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
        # Mod layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.chirp_distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.mass_ratio.to(dtype=data_type, device=self.store_device)
        self.inv_mass_ratio.to(dtype=data_type, device=self.store_device)
        self.snr.to(dtype=data_type, device=self.store_device)
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.backend = {'det1': self._det1, 'det2': self._det2}
        self.frontend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == 'batchnorm':
            normed = self.batchnorm(x)
        elif self.norm_layer == 'dain':
            normed, gate = self.dain(x)
        elif self.norm_layer == 'layernorm':
            normed = self.layernorm(x)
        
        # Data quality flag and veto of bad segments
        # [(0, 3342), (3342, 4485), (4485, 4705), (4705, 5729), (5729, 5838)]
        # ref = normed[:, :, 3697: -128]

        """
        segments = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 4697), (-100, -1)]
        for n, (sidx, eidx) in enumerate(segments):
            # Each chunk should be of a different sampling rate
            # Alpha should vary as [0.0, 1.0]
            # We don't use the hqual chunk as alpha = 1.0 in that chunk always
            chunk = normed[:, :, sidx: eidx]
            cmin = ref.min(dim=2).values.reshape(32, 2, 1)
            cmax = ref.max(dim=2).values.reshape(32, 2, 1)
            chunk = (chunk - cmin) / (cmax - cmin)
            mean = torch.mean(chunk, dim=2)
            mean = torch.reshape(mean, (32, 2, 1))
            diffs = chunk - mean # (32, 2, 1000)
            var = torch.mean(torch.pow(diffs, 2.0), dim=2)
            var = torch.reshape(var, (32, 2, 1))
            std = torch.pow(var, 0.5)
            zscores = diffs / std
            kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=2) - 3.0
            # veto = self.glitch_veto_simple[n](chunk)
            # veto = self.glitch_veto_kurtosis(kurtosis)
            for n, kurt in enumerate(kurtoses):
                if abs(kurt[0] - kurt[1]) > 1.0:
                    normed[n, :, sidx: eidx] = torch.tanh(normed[n, :, sidx: eidx])

            # alpha = torch.reshape(veto['pred_prob'], (32, 2, 1))
            # This is meant to downplay any chunks that impede with classification
            # normed[:, :, sidx: eidx] *= alpha
        """

        # Applying a tanh function to the normed input
        # cmin = ref.min(dim=2).values.reshape(32, 2, 1)
        # cmax = ref.max(dim=2).values.reshape(32, 2, 1)
        # normed = (normed - cmin) / (cmax - cmin)
        # normed = torch.tanh(normed)
        # normed = self.layernorm(normed)
        
        # Conv Backend
        cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)
        # Apply LayerNorm to CNN output before passing to ResNet
        cnn_output = self.layernorm_cnn(cnn_output)
        
        """
        # TODO: Experimental cwt input
        # Get the CWT of the input sample and attach it to the cnn_output before resnet section
        # cnn_output is (32, 2, 128, 182)
        widths = np.arange(1, cnn_output[0][0].shape[0]+1)
        
        cwt_output = []
        for n, sample in enumerate(x):
            det1, det2 = sample
            det1 = det1.detach().cpu().numpy()
            det2 = det2.detach().cpu().numpy()
            cwt_H1 = torch.from_numpy(signal.cwt(det1, signal.ricker, widths))
            cwt_L1 = torch.from_numpy(signal.cwt(det2, signal.ricker, widths))
            cwt_H1 = cwt_H1.to(torch.device('cuda:1'), dtype=torch.float32)
            cwt_L1 = cwt_L1.to(torch.device('cuda:1'), dtype=torch.float32)
            cwt_output_tmp = torch.stack((cwt_H1, cwt_L1))
            cwt_output.append(cwt_output_tmp)

        cwt_output = torch.stack(cwt_output)
        cwt_output.to(torch.device('cuda:1'), dtype=torch.float32)
        cwt_cnn_output = torch.cat([cnn_output, cwt_output], dim=3)
        """ 
        
        # Timm Frontend
        out = self.frontend(cnn_output) # (batch_size, 1000) by default
        ## Manipulate encoder output to get params
        # Global Pool
        out = self.flatten_d1(self.avg_pool_1d(out))
        # In the Kaggle architecture a dropout is added at this point
        # I see no reason to include at this stage. But we can experiment.
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(out))
        pred_prob = self.sigmoid(raw)
        # Parameter Estimation
        tc = self.flatten_d0(self.sigmoid(self.coalescence_time(out)))
        dchirp = self.flatten_d0(self.sigmoid(self.chirp_distance(out)))
        mchirp = self.flatten_d0(self.sigmoid(self.chirp_mass(out)))
        dist = self.flatten_d0(self.sigmoid(self.distance(out)))
        q = self.flatten_d0(self.sigmoid(self.mass_ratio(out)))
        invq = self.flatten_d0(self.sigmoid(self.inv_mass_ratio(out)))
        raw_snr = self.flatten_d0(self.snr(out))
        snr = self.sigmoid(raw_snr)
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'cnn_output': cnn_output,
                'norm_tc': tc, 'norm_dchirp': dchirp, 'norm_mchirp': mchirp,
                'norm_dist': dist, 'norm_q': q, 'norm_invq': invq, 'norm_snr': snr,
                'raw_snr': raw_snr, 'input': x, 'normed': normed}


class KappaModelSimplified(torch.nn.Module):
    """
    Kappa-type Model Simplified Architecture
    
    Description - Same as Kappa Model but uses a simpler ConvBlock with modified 
                  stride values that takes advatage of the MR sampling output size.
                  This model outputs only prediction probabilities with a direct
                  output from resnet-34 using reset_classifier(2)
        
    """

    def __init__(self, 
                 model_name='trainable_backend_and_frontend', 
                 filter_size: int = 32,
                 kernel_size: int = 64,
                 timm_params: dict = {'model_name': 'resnet34', 'pretrained': True, 'in_chans': 2, 'drop_rate': 0.25},
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.timm_params = timm_params
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        
        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock_Apr21(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock_Apr21(self.filter_size, self.kernel_size)
        _initialize_weights(self)
        
        """ Frontend """
        # resnet34 --> 21 Mil. trainable params trainable frontend
        self.frontend = timm.create_model(**timm_params)
        
        # reset_classifier edits the number of outputs that Timm produces
        # The following is set if we need a two-class output from Timm
        # This can be set to a larger value and connected to a linear layer
        self.frontend.reset_classifier(2)
        
        """ Mods """
        ## Penultimate and output layers
        # Manipulation layers
        self.softmax = torch.nn.Softmax(dim=1)
        
        ## Convert network into given dtype and store in proper device
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.backend = {'det1': self._det1, 'det2': self._det2}
        self.frontend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Conv Backend
        x = torch.cat([self.backend['det1'](x[:, 0:1]), self.backend['det2'](x[:, 1:2])], dim=1)
        # Timm Frontend
        x = self.frontend(x)
        ## Output necessary params
        pred_prob = self.softmax(x)
        # Return ouptut params pred_prob
        return {'pred_prob': pred_prob}


class DeltaModelPE(torch.nn.Module):
    """
    Delta-type Model PE Architecture
    
    Description - consists of a 2-channel Timm model frontend
    
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
                 model_name='trainable', 
                 timm_params: dict = {'model_name': 'resnet34', 'pretrained': True, 'in_chans': 2, 'drop_rate': 0.25},
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.timm_params = timm_params
        
        """ Frontend """
        # resnet34 --> 21 Mil. trainable params trainable frontend
        self.frontend = timm.create_model(**timm_params)
        
        """ Mods """
        # Primary outputs
        self.signal_or_noise = nn.Linear(self.frontend.num_features, 1)
        self.coalescence_time = nn.Linear(self.frontend.num_features, 1)
        self.chirp_distance = nn.Linear(self.frontend.num_features, 1)
        self.chirp_mass = nn.Linear(self.frontend.num_features, 1)
        self.distance = nn.Linear(self.frontend.num_features, 1)
        self.mass_ratio = nn.Linear(self.frontend.num_features, 1)
        self.inv_mass_ratio = nn.Linear(self.frontend.num_features, 1)
        self.snr = nn.Linear(self.frontend.num_features, 1)
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.batchnorm2d = nn.BatchNorm2d(2)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(self.frontend.num_features)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        
        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.batchnorm2d.to(dtype=data_type, device=self.store_device)
        # Mod layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.chirp_distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.mass_ratio.to(dtype=data_type, device=self.store_device)
        self.inv_mass_ratio.to(dtype=data_type, device=self.store_device)
        self.snr.to(dtype=data_type, device=self.store_device)
        # Main layers
        self.frontend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        x = self.batchnorm2d(x)
        # x = torch.reshape(x, (32, 2, 1, 5838))
        
        # Timm Frontend
        x = self.frontend(x) # (100, 1000) by default
        ## Manipulate encoder output to get params
        # Global Pool
        x = self.flatten_d1(self.avg_pool_1d(x))
        # In the Kaggle architecture a dropout is added at this point
        # I see no reason to include at this stage. But we can experiment.
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(x))
        pred_prob = self.sigmoid(raw)
        # Parameter Estimation
        tc = self.flatten_d0(self.sigmoid(self.coalescence_time(x)))
        dchirp = self.flatten_d0(self.sigmoid(self.chirp_distance(x)))
        mchirp = self.flatten_d0(self.sigmoid(self.chirp_mass(x)))
        dist = self.flatten_d0(self.sigmoid(self.distance(x)))
        q = self.flatten_d0(self.sigmoid(self.mass_ratio(x)))
        invq = self.flatten_d0(self.sigmoid(self.inv_mass_ratio(x)))
        raw_snr = self.flatten_d0(self.snr(x))
        snr = self.sigmoid(raw_snr)
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'norm_tc': tc, 'norm_dchirp': dchirp, 
                'norm_mchirp': mchirp, 'norm_dist': dist, 'norm_q': q, 'norm_invq': invq, 
                'norm_snr': snr, 'raw_snr': raw_snr}
