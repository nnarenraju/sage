# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename         =  foobar.py
Description      =  Lorem ipsum dolor sit amet

Created on 18/07/2024 at 10:36:15

__author__       =  Narenraju Nagarajan
__copyright__    =  Copyright 2024, ProjectName
__credits__      =  nnarenraju
__license__      =  MIT Licence
__version__      =  0.0.1
__maintainer__   =  nnarenraju
__affiliation__  =  University of Glasgow
__email__        =  nnarenraju@gmail.com
__status__       =  ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

# PACKAGES
import timm
import torch
from torch import nn

# Importing architecture snippets from zoo
from architectures.zoo.dain import DAIN_Layer
from architectures.zoo.resnet_cbam import resnet50_cbam, resnet152_cbam, resnet34_cbam
from architectures.zoo.res2net_v1b import res2net101_v1b_26w_4s, res2net50_v1b_26w_4s, res2net152_v1b_26w_4s
from architectures.zoo.osnet1d import osnet_ain_custom as osnet1d
from architectures.frontend import ConvBlock, _initialize_weights

# Datatype for storage
data_type=torch.float32



## Models without point parameter estimation ##

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



## Models with point parameter estimation ##

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



class KappaModel_Res2Net(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with Res2Net Block
    
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
                 resnet_size: int = 50,
                 norm_layer: str = 'instancenorm',
                 _input_length: int = 6111,
                 _decimated_bins = None,
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.norm_layer = norm_layer
        self._decimated_bins = _decimated_bins
        
        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size)
        _initialize_weights(self)
        
        """ Frontend """
        # resnet50 --> Based on Res2Net blocks
        # Pretrained model is for 3-channels. We use 2 channels.
        if resnet_size == 50:
            self.backend = res2net50_v1b_26w_4s(pretrained=False, num_classes=512)
        elif resnet_size == 101:
            self.backend = res2net101_v1b_26w_4s(pretrained=False, num_classes=512)
        elif resnet_size == 152:
            self.backend = res2net152_v1b_26w_4s(pretrained=False, num_classes=512)
        
        """ Mods """
        # Manipulation layers
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.groupnorm  = nn.GroupNorm(num_groups=2, num_channels=2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        # self.layernorm_cnn = nn.LayerNorm([2, 128, int(_input_length/32.)])
        self.layernorm_cnn = nn.LayerNorm([2, 128, 169])
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

        # Primary outputs
        self.signal_or_noise = nn.Linear(512, 1)
        self.coalescence_time = nn.Linear(512, 2)
        self.chirp_distance = nn.Linear(512, 2)
        self.chirp_mass = nn.Linear(512, 2)
        self.distance = nn.Linear(512, 2)
        self.mass_ratio = nn.Linear(512, 2)
        self.inv_mass_ratio = nn.Linear(512, 2)
        self.snr = nn.Linear(512, 2)
        # Mod layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.chirp_distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.mass_ratio.to(dtype=data_type, device=self.store_device)
        self.inv_mass_ratio.to(dtype=data_type, device=self.store_device)
        self.snr.to(dtype=data_type, device=self.store_device)
        
        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        self.groupnorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.frontend = {'det1': self._det1, 'det2': self._det2}
        self.backend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == 'batchnorm':
            normed = self.batchnorm(x)
        elif self.norm_layer == 'dain':
            normed, gate = self.dain(x)
        elif self.norm_layer == 'layernorm':
            normed = self.layernorm(x)
        elif self.norm_layer == 'instancenorm':
            normed = self.instancenorm(x)
        elif self.norm_layer == 'groupnorm':
            normed = self.groupnorm(x)
        
        # Conv Backend
        cnn_output = torch.cat([self.frontend['det1'](normed[:, 0:1]), self.frontend['det2'](normed[:, 1:2])], dim=1)
         
        # Timm Frontend
        out = self.backend(cnn_output).squeeze() # (batch_size, 512)
        ## Output necessary params
        raw = self.signal_or_noise(out).squeeze()
        pred_prob = self.sigmoid(raw)

        ## Parameter Estimation
        # Time of Coalescence
        tc_ = self.coalescence_time(out)
        tc = self.flatten_d0(tc_[:,0])
        norm_tc = self.sigmoid(tc)
        tc_var = self.Tanh(self.flatten_d0(tc_[:,1]))
        # Chirp Distance
        dchirp_ = self.chirp_distance(out)
        dchirp = self.flatten_d0(dchirp_[:,0])
        norm_dchirp = self.sigmoid(dchirp)
        dchirp_var = self.flatten_d0(dchirp_[:,1])
        # Chirp Mass
        mchirp_ = self.chirp_mass(out)
        mchirp = self.flatten_d0(mchirp_[:,0])
        norm_mchirp = self.sigmoid(mchirp)
        mchirp_var = self.Tanh(self.flatten_d0(mchirp_[:,1]))
        # Distance
        dist_ = self.distance(out)
        dist = self.flatten_d0(dist_[:,0])
        norm_dist = self.sigmoid(dist)
        dist_var = self.flatten_d0(dist_[:,1])
        # Mass Ratio
        q_ = self.mass_ratio(out)
        q = self.flatten_d0(q_[:,0])
        norm_q = self.sigmoid(q)
        q_var = self.flatten_d0(q_[:,1])
        # Inverse Mass Ratio
        invq_ = self.inv_mass_ratio(out)
        invq = self.flatten_d0(invq_[:,0])
        norm_invq = self.sigmoid(invq)
        invq_var = self.flatten_d0(invq_[:,1])
        # SNR
        snr_ = self.snr(out)
        snr = self.flatten_d0(snr_[:,0])
        norm_snr = self.sigmoid(snr)
        snr_var = self.flatten_d0(snr_[:,1])
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'cnn_output': cnn_output,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr,
                'norm_tc_var': tc_var, 'norm_mchirp_var': mchirp_var, 'norm_snr_var': snr_var,
                'norm_q_var': q_var, 'norm_invq_var': invq_var, 'norm_dist_var': dist_var,
                'norm_dchirp_var': dchirp_var, 'input': x, 'normed': normed}



class KappaModel_ResNet_CBAM(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet and CBAM
    
    Description - consists of a 2-channel ConvBlock frontend and a Timm model backend
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
                 resnet_size: int = 50,
                 norm_layer: str = 'instancenorm',
                 upsample_factor: float = 1.0,
                 _input_length: int = 4254, # 4859
                 _decimated_bins = None,
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.norm_layer = norm_layer
        self.upsample_factor = upsample_factor
        self._decimated_bins = _decimated_bins
        
        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size)
        _initialize_weights(self)
        
        """ Frontend """
        # resnet50 --> Based on Res2Net blocks
        # Pretrained model is for 3-channels. We use 2 channels.
        if resnet_size == 50:
            self.backend = resnet50_cbam(pretrained=False)
        elif resnet_size == 152:
            self.backend = resnet152_cbam(pretrained=False)
        
        """ Mods """
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        #self.layernorm_cnn = nn.LayerNorm([2, 128, int(_input_length/32.)])
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=self.upsample_factor, mode='bicubic')
        
        ## Convert network into given dtype and store in proper device 
        # Primary outputs
        self.signal_or_noise = nn.Linear(512, 1)
        self.coalescence_time = nn.Linear(512, 2)
        self.chirp_distance = nn.Linear(512, 2)
        self.chirp_mass = nn.Linear(512, 2)
        self.distance = nn.Linear(512, 2)
        self.mass_ratio = nn.Linear(512, 2)
        self.inv_mass_ratio = nn.Linear(512, 2)
        self.snr = nn.Linear(512, 2)
        # Mod layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.chirp_distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.mass_ratio.to(dtype=data_type, device=self.store_device)
        self.inv_mass_ratio.to(dtype=data_type, device=self.store_device)
        self.snr.to(dtype=data_type, device=self.store_device)

        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        #self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.frontend = {'det1': self._det1, 'det2': self._det2}
        self.backend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == 'batchnorm':
            normed = self.batchnorm(x)
        elif self.norm_layer == 'dain':
            normed, gate = self.dain(x)
        elif self.norm_layer == 'layernorm':
            normed = self.layernorm(x)
        elif self.norm_layer == 'instancenorm':
            normed = self.instancenorm(x)
        
        # Upsampling of input normed data
        if self.upsample_factor != 1.0:
            normed = torch.cat([self.upsample(normed[:, 0:1]), self.upsample(normed[:, 1:2])], dim=1)

        # Conv Backend
        cnn_output = torch.cat([self.frontend['det1'](normed[:, 0:1]), self.frontend['det2'](normed[:, 1:2])], dim=1)

        # Timm Frontend
        out = self.backend(cnn_output) # (batch_size, 512)
        out = self.flatten_d1(self.avg_pool_1d(out))
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(out))
        pred_prob = self.sigmoid(raw)

        ## Parameter Estimation
        # Time of Coalescence
        tc_ = self.coalescence_time(out)
        tc = self.flatten_d0(tc_[:,0])
        norm_tc = self.sigmoid(tc)
        tc_var = self.Tanh(self.flatten_d0(tc_[:,1]))
        # Chirp Distance
        dchirp_ = self.chirp_distance(out)
        dchirp = self.flatten_d0(dchirp_[:,0])
        norm_dchirp = self.sigmoid(dchirp)
        dchirp_var = self.flatten_d0(dchirp_[:,1])
        # Chirp Mass
        mchirp_ = self.chirp_mass(out)
        mchirp = self.flatten_d0(mchirp_[:,0])
        norm_mchirp = self.sigmoid(mchirp)
        mchirp_var = self.Tanh(self.flatten_d0(mchirp_[:,1]))
        # Distance
        dist_ = self.distance(out)
        dist = self.flatten_d0(dist_[:,0])
        norm_dist = self.sigmoid(dist)
        dist_var = self.flatten_d0(dist_[:,1])
        # Mass Ratio
        q_ = self.mass_ratio(out)
        q = self.flatten_d0(q_[:,0])
        norm_q = self.sigmoid(q)
        q_var = self.flatten_d0(q_[:,1])
        # Inverse Mass Ratio
        invq_ = self.inv_mass_ratio(out)
        invq = self.flatten_d0(invq_[:,0])
        norm_invq = self.sigmoid(invq)
        invq_var = self.flatten_d0(invq_[:,1])
        # SNR
        snr_ = self.snr(out)
        snr = self.flatten_d0(snr_[:,0])
        norm_snr = self.sigmoid(snr)
        snr_var = self.flatten_d0(snr_[:,1])
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'cnn_output': cnn_output,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr,
                'norm_tc_sigma': tc_var, 'norm_mchirp_sigma': mchirp_var, 'norm_snr_sigma': snr_var,
                'norm_q_sigma': q_var, 'norm_invq_sigma': invq_var, 'norm_dist_sigma': dist_var,
                'norm_dchirp_sigma': dchirp_var, 'input': x, 'normed': normed}



class Dummy_ResNet_CBAM(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet and CBAM
    
    Description - consists of a 2-channel ConvBlock frontend and a Timm model backend
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
                 model_name='ResNet_CBAM', 
                 filter_size: int = 32,
                 kernel_size: int = 64,
                 resnet_size: int = 50,
                 parameter_estimation: tuple = (),
                 norm_layer: str = 'instancenorm',
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.norm_layer = norm_layer
        self.parameter_estimation = parameter_estimation
        
        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size)
        _initialize_weights(self)
        
        """ Frontend """
        # resnet50 --> Based on Res2Net blocks
        # Pretrained model is for 3-channels. We use 2 channels.
        if resnet_size == 50:
            self.backend = resnet50_cbam(pretrained=False)
        elif resnet_size == 152:
            self.backend = resnet152_cbam(pretrained=False)
        
        """ Mods """
        # Normalisation layers
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        # Shape manipulation
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        # Value transformation
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        # Regularisation layers
        self.dropout = nn.Dropout(0.25)
        
        ## Convert network into given dtype and store in proper device 
        # Primary outputs
        self.signal_or_noise = nn.Linear(512, 1)
        self.coalescence_time = nn.Linear(512, 1)
        self.chirp_distance = nn.Linear(512, 1)
        self.chirp_mass = nn.Linear(512, 1)
        self.distance = nn.Linear(512, 1)
        self.mass_ratio = nn.Linear(512, 1)
        self.inv_mass_ratio = nn.Linear(512, 1)
        self.snr = nn.Linear(512, 1)
        # Mod layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.chirp_distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.mass_ratio.to(dtype=data_type, device=self.store_device)
        self.inv_mass_ratio.to(dtype=data_type, device=self.store_device)
        self.snr.to(dtype=data_type, device=self.store_device)

        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        #self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.frontend = {'det1': self._det1, 'det2': self._det2}
        self.backend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == 'batchnorm':
            normed = self.batchnorm(x)
        elif self.norm_layer == 'dain':
            normed, gate = self.dain(x)
        elif self.norm_layer == 'layernorm':
            normed = self.layernorm(x)
        elif self.norm_layer == 'instancenorm':
            normed = self.instancenorm(x)

        # 1D CNN Frontend
        cnn_output = torch.cat([self.frontend['det1'](normed[:, 0:1]), self.frontend['det2'](normed[:, 1:2])], dim=1)

        # ResNet CBAM Backend
        out = self.backend(cnn_output) # (batch_size, embedding_size)
        out = self.flatten_d1(self.avg_pool_1d(out))
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(out))
        pred_prob = self.sigmoid(raw)

        ## Parameter Estimation
        # Time of Coalescence
        tc = self.flatten_d0(self.coalescence_time(out))
        norm_tc = self.sigmoid(tc)
        # Chirp Distance
        dchirp = self.flatten_d0(self.chirp_distance(out))
        norm_dchirp = self.sigmoid(dchirp)
        # Chirp Mass
        mchirp = self.flatten_d0(self.chirp_mass(out))
        norm_mchirp = self.sigmoid(mchirp)
        # Distance
        dist = self.flatten_d0(self.distance(out))
        norm_dist = self.sigmoid(dist)
        # Mass Ratio
        q = self.flatten_d0(self.mass_ratio(out))
        norm_q = self.sigmoid(q)
        # Inverse Mass Ratio
        invq = self.flatten_d0(self.inv_mass_ratio(out))
        norm_invq = self.sigmoid(invq)
        # SNR
        snr = self.flatten_d0(self.snr(out))
        norm_snr = self.sigmoid(snr)
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'cnn_output': cnn_output,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr,
                'input': x, 'normed': normed}



class KappaModel_ResNet_small(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet50
    
    Description - consists of a 2-channel ConvBlock frontend and a Timm model backend
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
                 model_name='smallnet',
                 timm_params: dict = {'model_name': 'resnet50', 'pretrained': False, 'in_chans': 2, 'drop_rate': 0.25},
                 norm_layer: str = 'instancenorm',
                 _input_length: int = 4254,
                 _decimated_bins = None,
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.norm_layer = norm_layer
        
        """ Frontend """
        # Pretrained model is for 3-channels. We use 2 channels.
        self.backend = timm.create_model(**timm_params)
        
        # reset_classifier edits the number of outputs that Timm produces
        # The following is set if we need a two-class output from Timm
        # This can be set to a larger value and connected to a linear layer
        self.backend.reset_classifier(1)
        
        """ Mods """
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=self.upsample_factor, mode='bicubic')
        
        ## Convert network into given dtype and store in proper device 
        # Primary outputs
        self.signal_or_noise = nn.Linear(512, 1)
        self.coalescence_time = nn.Linear(512, 2)
        self.chirp_distance = nn.Linear(512, 2)
        self.chirp_mass = nn.Linear(512, 2)
        self.distance = nn.Linear(512, 2)
        self.mass_ratio = nn.Linear(512, 2)
        self.inv_mass_ratio = nn.Linear(512, 2)
        self.snr = nn.Linear(512, 2)
        # Mod layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.chirp_distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.mass_ratio.to(dtype=data_type, device=self.store_device)
        self.inv_mass_ratio.to(dtype=data_type, device=self.store_device)
        self.snr.to(dtype=data_type, device=self.store_device)

        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        #self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.frontend = {'det1': self._det1, 'det2': self._det2}
        self.backend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == 'batchnorm':
            normed = self.batchnorm(x)
        elif self.norm_layer == 'instancenorm':
            normed = self.instancenorm(x)
        
        # Upsampling of input normed data
        if self.upsample_factor != 1.0:
            normed = torch.cat([self.upsample(normed[:, 0:1]), self.upsample(normed[:, 1:2])], dim=1)

        # Conv Backend
        cnn_output = torch.cat([self.frontend['det1'](normed[:, 0:1]), self.frontend['det2'](normed[:, 1:2])], dim=1)

        # Timm Frontend
        out = self.backend(cnn_output) # (batch_size, 512)
        out = self.flatten_d1(self.avg_pool_1d(out))
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(out))
        pred_prob = self.sigmoid(raw)

        ## Parameter Estimation
        # Time of Coalescence
        tc_ = self.coalescence_time(out)
        tc = self.flatten_d0(tc_[:,0])
        norm_tc = self.sigmoid(tc)
        tc_var = self.Tanh(self.flatten_d0(tc_[:,1]))
        # Chirp Distance
        dchirp_ = self.chirp_distance(out)
        dchirp = self.flatten_d0(dchirp_[:,0])
        norm_dchirp = self.sigmoid(dchirp)
        dchirp_var = self.flatten_d0(dchirp_[:,1])
        # Chirp Mass
        mchirp_ = self.chirp_mass(out)
        mchirp = self.flatten_d0(mchirp_[:,0])
        norm_mchirp = self.sigmoid(mchirp)
        mchirp_var = self.Tanh(self.flatten_d0(mchirp_[:,1]))
        # Distance
        dist_ = self.distance(out)
        dist = self.flatten_d0(dist_[:,0])
        norm_dist = self.sigmoid(dist)
        dist_var = self.flatten_d0(dist_[:,1])
        # Mass Ratio
        q_ = self.mass_ratio(out)
        q = self.flatten_d0(q_[:,0])
        norm_q = self.sigmoid(q)
        q_var = self.flatten_d0(q_[:,1])
        # Inverse Mass Ratio
        invq_ = self.inv_mass_ratio(out)
        invq = self.flatten_d0(invq_[:,0])
        norm_invq = self.sigmoid(invq)
        invq_var = self.flatten_d0(invq_[:,1])
        # SNR
        snr_ = self.snr(out)
        snr = self.flatten_d0(snr_[:,0])
        norm_snr = self.sigmoid(snr)
        snr_var = self.flatten_d0(snr_[:,1])
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'cnn_output': cnn_output,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr,
                'norm_tc_sigma': tc_var, 'norm_mchirp_sigma': mchirp_var, 'norm_snr_sigma': snr_var,
                'norm_q_sigma': q_var, 'norm_invq_sigma': invq_var, 'norm_dist_sigma': dist_var,
                'norm_dchirp_sigma': dchirp_var, 'input': x, 'normed': normed}



class SigmaModel(torch.nn.Module):
    """
    Sigma-type Model PE Architecture with ResNet and CBAM
    Sigma = sum of all hard work.
    
    Description - consists of two separate OSnet frontend for feature extraction
                  and a ResNet CBAM as backend for classification.
    
    Parameters
    ----------
    model_name  = 'simple' : string
        Model name for architecture.
    
    store_device = 'cpu' : str
        Storage device for network (NOTE: make sure data is also stored in the same device)
        
    """

    def __init__(self, 
                 model_name='sigmanet',
                 channels=[16, 32, 64, 128],
                 kernel_sizes=[
                    [[3,3,3,3,3], [3,3,3,3,3]], [[3,3,3,3,3], [3,3,3,3,3]],
                    [[3,3,3,3,3], [3,3,3,3,3]]
                 ], 
                 strides=[2,2,8,4],
                 stacking=True,
                 initial_dim_reduction=False,
                 channel_gate_reduction=8,
                 resnet_size: int = 50,
                 norm_layer: str = 'instancenorm',
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.norm_layer = norm_layer
        
        """ Frontend """
        self._det1 = osnet1d(channels=channels,
                             kernel_sizes=kernel_sizes, 
                             strides=strides, 
                             stacking=stacking, 
                             initial_dim_reduction=initial_dim_reduction,
                             channel_gate_reduction=channel_gate_reduction)

        self._det2 = osnet1d(channels=channels,
                             kernel_sizes=kernel_sizes, 
                             strides=strides, 
                             stacking=stacking, 
                             initial_dim_reduction=initial_dim_reduction,
                             channel_gate_reduction=channel_gate_reduction)
        
        """ Backend """
        # resnet50 --> Based on ResNetCBAM blocks
        # Pretrained model is for 3-channels. We use 2 channels.
        if resnet_size == 50:
            self.backend = resnet50_cbam(pretrained=False)
        elif resnet_size == 152:
            self.backend = resnet152_cbam(pretrained=False)
        
        """ Mods """
        ## Manipulation layers
        # Normalisation
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        # Flattening
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        # Others
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.sigmoid = torch.nn.Sigmoid()
        
        ## Convert network into given dtype and store in proper device 
        # Primary outputs
        self.signal_or_noise = nn.Linear(512, 1)
        self.coalescence_time = nn.Linear(512, 2)
        self.chirp_distance = nn.Linear(512, 2)
        self.chirp_mass = nn.Linear(512, 2)
        self.distance = nn.Linear(512, 2)
        self.mass_ratio = nn.Linear(512, 2)
        self.inv_mass_ratio = nn.Linear(512, 2)
        self.snr = nn.Linear(512, 2)
        # Mod layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.coalescence_time.to(dtype=data_type, device=self.store_device)
        self.chirp_distance.to(dtype=data_type, device=self.store_device)
        self.chirp_mass.to(dtype=data_type, device=self.store_device)
        self.distance.to(dtype=data_type, device=self.store_device)
        self.mass_ratio.to(dtype=data_type, device=self.store_device)
        self.inv_mass_ratio.to(dtype=data_type, device=self.store_device)
        self.snr.to(dtype=data_type, device=self.store_device)

        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.frontend = {'det1': self._det1, 'det2': self._det2}
        self.backend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == 'batchnorm':
            normed = self.batchnorm(x)
        elif self.norm_layer == 'dain':
            normed, gate = self.dain(x)
        elif self.norm_layer == 'instancenorm':
            normed = self.instancenorm(x)

        # Conv Backend (batch_size, 128, 128) for sample length = 4096
        osnet_output = torch.cat([self.frontend['det1'](normed[:, 0:1]), self.frontend['det2'](normed[:, 1:2])], dim=1)

        # Timm Frontend
        out = self.backend(osnet_output) # (batch_size, 512)
        out = self.flatten_d1(self.avg_pool_1d(out))
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(out))
        pred_prob = self.sigmoid(raw)

        ## Parameter Estimation
        # Time of Coalescence
        tc_ = self.coalescence_time(out)
        tc = self.flatten_d0(tc_[:,0])
        norm_tc = self.sigmoid(tc)
        tc_var = self.flatten_d0(tc_[:,1])
        # Chirp Distance
        dchirp_ = self.chirp_distance(out)
        dchirp = self.flatten_d0(dchirp_[:,0])
        norm_dchirp = self.sigmoid(dchirp)
        dchirp_var = self.flatten_d0(dchirp_[:,1])
        # Chirp Mass
        mchirp_ = self.chirp_mass(out)
        mchirp = self.flatten_d0(mchirp_[:,0])
        norm_mchirp = self.sigmoid(mchirp)
        mchirp_var = self.flatten_d0(mchirp_[:,1])
        # Distance
        dist_ = self.distance(out)
        dist = self.flatten_d0(dist_[:,0])
        norm_dist = self.sigmoid(dist)
        dist_var = self.flatten_d0(dist_[:,1])
        # Mass Ratio
        q_ = self.mass_ratio(out)
        q = self.flatten_d0(q_[:,0])
        norm_q = self.sigmoid(q)
        q_var = self.flatten_d0(q_[:,1])
        # Inverse Mass Ratio
        invq_ = self.inv_mass_ratio(out)
        invq = self.flatten_d0(invq_[:,0])
        norm_invq = self.sigmoid(invq)
        invq_var = self.flatten_d0(invq_[:,1])
        # SNR
        snr_ = self.snr(out)
        snr = self.flatten_d0(snr_[:,0])
        norm_snr = self.sigmoid(snr)
        snr_var = self.flatten_d0(snr_[:,1])
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'cnn_output': osnet_output,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr,
                'norm_tc_sigma': tc_var, 'norm_mchirp_sigma': mchirp_var, 'norm_snr_sigma': snr_var,
                'norm_q_sigma': q_var, 'norm_invq_sigma': invq_var, 'norm_dist_sigma': dist_var,
                'norm_dchirp_sigma': dchirp_var, 'input': x, 'normed': normed}
