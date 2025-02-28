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
import json
import torch
from torch import nn
from datetime import date

# Importing architecture snippets from zoo
from architectures.zoo.dain import DAIN_Layer
from architectures.zoo.resnet_cbam import resnet50_cbam, resnet152_cbam, resnet34_cbam
from architectures.zoo.res2net_v1b import res2net101_v1b_26w_4s, res2net50_v1b_26w_4s, res2net152_v1b_26w_4s
from architectures.zoo.osnet1d import osnet_ain_custom as osnet1d
from architectures.zoo.kaggle import ConvBlock, _initialize_weights
from architectures.frontend import MSFeatureExtractor, MultiScaleBlock
from architectures.zoo.resnet1d import resnet50, resnet101, resnet152

# Code Review
from utils.decorators import unreviewed_model
from utils.review import set_review_date

# Datatype for storage
data_type=torch.float32



@unreviewed_model
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



@unreviewed_model
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



@unreviewed_model
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



class Rigatoni_MS_ResNetCBAM(torch.nn.Module):
    """
    Rigatoni-type model with multi-scale feature extractor && ResNet-CBAM
    
    Description - Consists of a MSFeatureExtractor frontend for each detector and a 
                  ResNet-CBAM model backend. Capable of PE point estimate 
                  regularisation.
    
    Parameters
    ----------
    
        
    """

    def __init__(self, 
                 model_name: str = 'Rigatoni_MS_ResNetCBAM',
                 scales: list = [1, 2, 4, 0.5, 0.25],
                 blocks: list = [
                        [MultiScaleBlock, MultiScaleBlock], 
                        [MultiScaleBlock, MultiScaleBlock], 
                        [MultiScaleBlock, MultiScaleBlock]
                    ],
                 out_channels: list = [[32, 32], [64, 64], [128, 128]],
                 base_kernel_sizes: list = [
                        [64, 64 // 2 + 1], 
                        [64 // 2 + 1, 64 // 4 + 1], 
                        [64 // 4 + 1, 64 // 4 + 1]
                    ], 
                 compression_factor: list = [8, 4, 0],
                 in_channels: int = 1,
                 resnet_size: int = 50,
                 parameter_estimation: tuple = (),
                 norm_layer: str = 'instancenorm',
                 store_device: str = 'cpu',
                 review: bool = False,
                 **kwargs):
        
        super().__init__()
        
        # Saving last review date
        if review:
            last_review_date = date.today()
            model_name = self.__class__.__name__
            parent_name = "models"
            set_review_date(parent_name, model_name, last_review_date)

        self.model_name = model_name
        self.norm_layer = norm_layer
        self.parameter_estimation = parameter_estimation
        self.store_device = store_device
        
        """ Backend """
        # Initialisation of weights and biases performed upon call
        self._det1 = MSFeatureExtractor(scales, blocks, out_channels, base_kernel_sizes, 
                                        compression_factor, in_channels)
        self._det2 = MSFeatureExtractor(scales, blocks, out_channels, base_kernel_sizes, 
                                        compression_factor, in_channels)
        
        """ Frontend """
        # Pretrained model is for 3-channels. We use 2 channels.
        # When training  on HLV, we can use pretrained model.
        if resnet_size == 50:
            self.backend = resnet50_cbam(pretrained=False)
        elif resnet_size == 152:
            self.backend = resnet152_cbam(pretrained=False)
        
        """ Mods """
        # Normalisation layers
        self.batchnorm = nn.BatchNorm1d(2)
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

        # 1D CNN Frontend
        cnn_output = torch.cat([self.frontend['det1'](normed[:, 0:1]), 
                                self.frontend['det2'](normed[:, 1:2])], dim=1)

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



@unreviewed_model
class Rigatoni_MS_ResNetCBAM_legacy(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet and CBAM
    
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
                 upsample_factor: float = 1.0,
                 parameter_estimation = ('norm_tc', 'norm_mchirp', ),
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
            self.frontend = resnet50_cbam(pretrained=False)
        elif resnet_size == 152:
            self.frontend = resnet152_cbam(pretrained=False)
        
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
        elif self.norm_layer == 'instancenorm':
            normed = self.instancenorm(x)
        
        # Upsampling of input normed data
        if self.upsample_factor != 1.0:
            normed = torch.cat([self.upsample(normed[:, 0:1]), self.upsample(normed[:, 1:2])], dim=1)

        # Conv Backend
        cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)

        # Timm Frontend
        out = self.frontend(cnn_output) # (batch_size, 512)
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


@unreviewed_model
class Rigatoni_MS_ResNetCBAM_legacy_minimal(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet and CBAM
    
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
                 upsample_factor: float = 1.0,
                 parameter_estimation = ('norm_tc', 'norm_mchirp', ),
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
            self.frontend = resnet50_cbam(pretrained=False)
        elif resnet_size == 152:
            self.frontend = resnet152_cbam(pretrained=False)
        
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

        # Manipulation layers
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
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.backend = {'det1': self._det1, 'det2': self._det2}
        self.frontend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        normed = self.instancenorm(x)
        # Conv Backend
        self.backend['det1'] = self.backend['det1'].to(device=x.device)
        self.backend['det2'] = self.backend['det2'].to(device=x.device)
        cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)
        # CNN frontend
        out = self.frontend(cnn_output) # (batch_size, 512)
        out = self.flatten_d1(self.avg_pool_1d(out))
        ## Output necessary params
        raw = self.flatten_d0(self.signal_or_noise(out))
        pred_prob = self.sigmoid(raw)

        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob}


@unreviewed_model
class Rigatoni_MS_ResNetCBAM_legacy_singledet(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet and CBAM
    
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
                 upsample_factor: float = 1.0,
                 parameter_estimation = ('norm_tc', 'norm_mchirp', ),
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
            self.frontend_1 = resnet50_cbam(pretrained=False, in_channels=1)
            self.frontend_2 = resnet50_cbam(pretrained=False, in_channels=1)
        elif resnet_size == 152:
            raise NotImplementedError('Use ResNet50 instead!')
        
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
        self.signal_or_noise_1 = nn.Linear(512, 1)
        self.signal_or_noise_2 = nn.Linear(512, 1)
        self.coalescence_time = nn.Linear(1024, 2)
        self.chirp_distance = nn.Linear(1024, 2)
        self.chirp_mass = nn.Linear(1024, 2)
        self.distance = nn.Linear(1024, 2)
        self.mass_ratio = nn.Linear(1024, 2)
        self.inv_mass_ratio = nn.Linear(1024, 2)
        self.snr = nn.Linear(1024, 2)
        self.combine_ranking = nn.Linear(1024, 1)
        # Mod layers
        self.signal_or_noise_1.to(dtype=data_type, device=self.store_device)
        self.signal_or_noise_2.to(dtype=data_type, device=self.store_device)
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
        self.backend = {'det1': self._det1, 'det2': self._det2}
        self.frontend_1.to(dtype=data_type, device=self.store_device)
        self.frontend_2.to(dtype=data_type, device=self.store_device)
    
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

        # Conv Backend
        cnn_output_1 = self.backend['det1'](normed[:, 0:1])
        cnn_output_2 = self.backend['det2'](normed[:, 1:2])

        # Timm Frontend
        out_1 = self.frontend_1(cnn_output_1) # (batch_size, 512)
        out_1 = self.flatten_d1(self.avg_pool_1d(out_1))
        out_2 = self.frontend_2(cnn_output_2) # (batch_size, 512)
        out_2 = self.flatten_d1(self.avg_pool_1d(out_2))
        out = torch.cat([out_1, out_2])

        raw_1 = self.flatten_d0(self.signal_or_noise_1(out_1))
        raw_2 = self.flatten_d0(self.signal_or_noise_2(out_2))

        ## Output necessary params
        raw = self.flatten_d0(self.combine_ranking(out))
        pred_prob = self.sigmoid(raw)
        all_raw_out = torch.cat([raw, raw_1, raw_2])

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
        return {'raw': all_raw_out, 'pred_prob': pred_prob, 'cnn_output': cnn_output,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr,
                'norm_tc_sigma': tc_var, 'norm_mchirp_sigma': mchirp_var, 'norm_snr_sigma': snr_var,
                'norm_q_sigma': q_var, 'norm_invq_sigma': invq_var, 'norm_dist_sigma': dist_var,
                'norm_dchirp_sigma': dchirp_var, 'input': x, 'normed': normed}
    


@unreviewed_model
class KappaModel_ResNet1D(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet1D
    
    Description - consists of a ResNet1D backend
        
    """

    def __init__(self, 
                 model_name='resnet1d',
                 resnet_size: int = 50,
                 norm_layer: str = 'instancenorm',
                 parameter_estimation = ('norm_tc', 'norm_mchirp', ),
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.norm_layer = norm_layer
        
        """ 1D ResNet """
        if resnet_size == 50:
            self.resnet = resnet50(num_classes=512)
        elif resnet_size == 101:
            self.resnet = resnet101(num_classes=512)
        elif resnet_size == 152:
            self.resnet = resnet152(num_classes=512)
        
        """ Mods """
        # Manipulation layers
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
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self.resnet.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        if self.norm_layer == 'batchnorm':
            normed = self.batchnorm(x)
        elif self.norm_layer == 'instancenorm':
            normed = self.instancenorm(x)

        # Resnet1D
        out = self.resnet(normed) # (batch_size, 512)
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
        return {'raw': raw, 'pred_prob': pred_prob, 'input': x,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr}



@unreviewed_model
class SigmaModel(torch.nn.Module):
    """
    Sigma-type Model PE Architecture with ResNet and CBAM
    
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


@unreviewed_model
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
        # Primary outputs
        self.signal_or_noise = nn.Linear(self.frontend.num_features, 1)
        self.coalescence_time = nn.Linear(self.frontend.num_features, 2)
        self.chirp_distance = nn.Linear(self.frontend.num_features, 1)
        self.chirp_mass = nn.Linear(self.frontend.num_features, 2)
        self.distance = nn.Linear(self.frontend.num_features, 1)
        self.mass_ratio = nn.Linear(self.frontend.num_features, 1)
        self.inv_mass_ratio = nn.Linear(self.frontend.num_features, 1)
        self.snr = nn.Linear(self.frontend.num_features, 2)
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, 5830])
        self.layernorm_cnn = nn.LayerNorm([2, 128, 182])
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