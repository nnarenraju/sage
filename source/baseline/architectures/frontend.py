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


"""

# PACKAGES
import timm
import torch
import numpy as np
from torch import nn
from scipy import signal

# Diffusers from HuggingFace
# from diffusers.models.unet_1d import UNet1DModel as UNet1d

# Importing architecture snippets from zoo
from architectures.zoo.FCN import FCN
from architectures.zoo.dain import DAIN_Layer
from architectures.zoo.wavenet_type1 import WaveNetModel as wavenet_type1
from architectures.zoo.wavenet_type2 import WaveNet as wavenet_type2
from architectures.zoo.resnet1d_type1 import ResNet1d
from architectures.zoo.resnet_cbam import resnet50_cbam, resnet152_cbam, resnet34_cbam
from architectures.zoo.res2net import res2net50, res2net101_26w_4s
from architectures.zoo.res2net_v1b import res2net101_v1b_26w_4s, res2net50_v1b_26w_4s, res2net152_v1b_26w_4s
from architectures.zoo.res2net1d import res2net50 as res2net50_1d
from architectures.zoo.res2net1d import res2net101_26w_4s as res2net101_1d
from architectures.zoo.aresnet2d import ResNet38 as AResNet38
from architectures.zoo.virgo_unedited import ResNet54Double as virgonet
from architectures.zoo.residual_attention_network import ResidualAttentionModel_56 as AResNet56
from architectures.zoo.kaggle import ConvBlock, _initialize_weights, ConvBlock_Apr21
from architectures.zoo.inception_time import inceptiontime_def as InceptionTime_DEF
from architectures.zoo.osnet1d import osnet_ain_custom as osnet1d

# Datatype for storage
data_type=torch.float32


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



class MuModelPE(torch.nn.Module):
    """
    Mu-type Model PE Architecture
    
    Description - consists of a 2-channel ConvBlock backend and a Timm model frontend
                  this Model-type can be used to test the Kaggle architectures. Made for 
                  usage with metric losses.
    
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
                 _input_length: int = 0,
                 _decimated_bins = None,
                 glitch_veto: bool = False,
                 maxnorm: bool = False,
                 clipped: bool = False,
                 norm_cnn_output: bool = False,
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.timm_params = timm_params
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.norm_layer = norm_layer
        # Optional
        self.glitch_veto = glitch_veto
        self.maxnorm = maxnorm
        self.clipped = clipped
        self.norm_cnn_output = norm_cnn_output
        self._decimated_bins = _decimated_bins
        
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
        self.chirp_distance = nn.Linear(self.frontend.num_features, 2)
        self.chirp_mass = nn.Linear(self.frontend.num_features, 2)
        self.distance = nn.Linear(self.frontend.num_features, 2)
        self.mass_ratio = nn.Linear(self.frontend.num_features, 2)
        self.inv_mass_ratio = nn.Linear(self.frontend.num_features, 2)
        self.snr = nn.Linear(self.frontend.num_features, 2)
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        self.layernorm_cnn = nn.LayerNorm([2, 128, int(_input_length/32.)])
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
    
    # x.shape: (batch size, type, wave channel, length of wave)
    def forward(self, x):
        # batch_size, type, channel, signal_length = s.shape
        # Here sample type will be in the order (positive, negative, anchor) if
        # we use TripletLoss OR (positive, negative) for MarginRankingLoss.
        data = []
        positive = x[:, 0, :, :]
        negative = x[:, 1, :, :]
        if self.triplet:
            anchor = x[:, 2, :, :]
            data = [positive, negative, anchor]
        else:
            data = [positive, negative]
        
        for datum in data:
            if self.norm_layer == 'batchnorm':
                normed = self.batchnorm(x)
            elif self.norm_layer == 'dain':
                normed, gate = self.dain(x)
            elif self.norm_layer == 'layernorm':
                normed = self.layernorm(x)
        
        # Data quality flag and veto of bad segments
        # example: [(0, 3342), (3342, 4485), (4485, 4705), (4705, 5729), (5729, 5838)]
        # example[-2] will be where the merger is present for default tc prior
        # TODO: Generalise this for all choices of 'tc' priors
        merger_bin = self._decimated_bins[-2]
        ref = normed[:, :, merger_bin[0]: merger_bin[1]]

        if self.glitch_veto:
            segments = self._decimated_bins
            for n, (sidx, eidx) in enumerate(segments):
                # Each chunk should be of a different sampling rate
                # Alpha should vary as [0.0, 1.0]
                # We don't use the hqual chunk as alpha = 1.0 in that chunk always
                chunk = normed[:, :, sidx: eidx]
                cmin = ref.min(dim=2).values.reshape(x.shape[0], 2, 1)
                cmax = ref.max(dim=2).values.reshape(x.shape[0], 2, 1)
                chunk = (chunk - cmin) / (cmax - cmin)
                mean = torch.mean(chunk, dim=2)
                mean = torch.reshape(mean, (x.shape[0], 2, 1))
                diffs = chunk - mean # (batch_size, 2, 1000)
                var = torch.mean(torch.pow(diffs, 2.0), dim=2)
                var = torch.reshape(var, (x.shape[0], 2, 1))
                std = torch.pow(var, 0.5)
                zscores = diffs / std
                kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=2) - 3.0
                for n, kurt in enumerate(kurtoses):
                    if abs(kurt[0] - kurt[1]) > 1.0:
                        normed[n, :, sidx: eidx] = torch.tanh(normed[n, :, sidx: eidx])

        # Applying a tanh function to the normed input
        if self.maxnorm:
            cabs = torch.abs(ref)
            cmax = cabs.max(dim=2).values.reshape(x.shape[0], 2, 1)
            normed = normed/cmax

        if self.clipped:
            normed = torch.clamp(normed, **clipped_params)
        
        # Conv Backend
        cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)

        # Apply LayerNorm to CNN output before passing to ResNet
        if self.norm_cnn_output:
            cnn_output = self.layernorm_cnn(cnn_output)

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
        ## Parameter Estimation
        # Time of Coalescence
        tc_ = self.coalescence_time(out)
        tc = self.flatten_d0(tc_[:,0])
        norm_tc = self.sigmoid(tc)
        tc_var = self.sigmoid(self.flatten_d0(tc_[:,1]))
        # Chirp Distance
        dchirp_ = self.chirp_distance(out)
        dchirp = self.flatten_d0(dchirp_[:,0])
        norm_dchirp = self.sigmoid(dchirp)
        dchirp_var = self.sigmoid(self.flatten_d0(dchirp_[:,1]))
        # Chirp Mass
        mchirp_ = self.chirp_mass(out)
        mchirp = self.flatten_d0(mchirp_[:,0])
        norm_mchirp = self.sigmoid(mchirp)
        mchirp_var = self.sigmoid(self.flatten_d0(mchirp_[:,1]))
        # Distance
        dist_ = self.distance(out)
        dist = self.flatten_d0(dist_[:,0])
        norm_dist = self.sigmoid(dist)
        dist_var = self.sigmoid(self.flatten_d0(dist_[:,1]))
        # Mass Ratio
        q_ = self.mass_ratio(out)
        q = self.flatten_d0(q_[:,0])
        norm_q = self.sigmoid(q)
        q_var = self.sigmoid(self.flatten_d0(q_[:,1]))
        # Inverse Mass Ratio
        invq_ = self.inv_mass_ratio(out)
        invq = self.flatten_d0(invq_[:,0])
        norm_invq = self.sigmoid(invq)
        invq_var = self.sigmoid(self.flatten_d0(invq_[:,1]))
        # SNR
        snr_ = self.snr(out)
        snr = self.flatten_d0(snr_[:,0])
        norm_snr = self.sigmoid(snr)
        snr_var = self.sigmoid(self.flatten_d0(snr_[:,1]))
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'cnn_output': cnn_output,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr,
                'norm_tc_var': tc_var, 'norm_mchirp_var': mchirp_var, 'norm_snr_var': snr_var,
                'norm_q_var': q_var, 'norm_invq_var': invq_var, 'norm_dist_var': dist_var,
                'norm_dchirp_var': dchirp_var, 'input': x, 'normed': normed, 'vector': vector}



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
                 norm_layer: str = 'instancenorm',
                 _input_length: int = 3226,
                 _decimated_bins = None,
                 glitch_veto: bool = False,
                 maxnorm: bool = False,
                 clipped: bool = False,
                 norm_cnn_output: bool = False,
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.timm_params = timm_params
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.norm_layer = norm_layer
        # Optional
        self.glitch_veto = glitch_veto
        self.maxnorm = maxnorm
        self.clipped = clipped
        self.norm_cnn_output = norm_cnn_output
        self._decimated_bins = _decimated_bins
        
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
        self.chirp_distance = nn.Linear(self.frontend.num_features, 2)
        self.chirp_mass = nn.Linear(self.frontend.num_features, 2)
        self.distance = nn.Linear(self.frontend.num_features, 2)
        self.mass_ratio = nn.Linear(self.frontend.num_features, 2)
        self.inv_mass_ratio = nn.Linear(self.frontend.num_features, 2)
        self.snr = nn.Linear(self.frontend.num_features, 2)
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        self.layernorm_cnn = nn.LayerNorm([2, 128, int(_input_length/32.)])
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(self.frontend.num_features)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        
        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
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
        elif self.norm_layer == 'instancenorm':
            normed = self.instancenorm(x)
        
        # Data quality flag and veto of bad segments
        # example: [(0, 3342), (3342, 4485), (4485, 4705), (4705, 5729), (5729, 5838)]
        # example[-2] will be where the merger is present for default tc prior
        # TODO: Generalise this for all choices of 'tc' priors
        if self._decimated_bins != None:
            merger_bin = self._decimated_bins[-2]
            ref = normed[:, :, merger_bin[0]: merger_bin[1]]

        if self.glitch_veto:
            segments = self._decimated_bins
            for n, (sidx, eidx) in enumerate(segments):
                # Each chunk should be of a different sampling rate
                # Alpha should vary as [0.0, 1.0]
                # We don't use the hqual chunk as alpha = 1.0 in that chunk always
                chunk = normed[:, :, sidx: eidx]
                cmin = ref.min(dim=2).values.reshape(x.shape[0], 2, 1)
                cmax = ref.max(dim=2).values.reshape(x.shape[0], 2, 1)
                chunk = (chunk - cmin) / (cmax - cmin)
                mean = torch.mean(chunk, dim=2)
                mean = torch.reshape(mean, (x.shape[0], 2, 1))
                diffs = chunk - mean # (batch_size, 2, 1000)
                var = torch.mean(torch.pow(diffs, 2.0), dim=2)
                var = torch.reshape(var, (x.shape[0], 2, 1))
                std = torch.pow(var, 0.5)
                zscores = diffs / std
                kurtoses = torch.mean(torch.pow(zscores, 4.0), dim=2) - 3.0
                for n, kurt in enumerate(kurtoses):
                    if abs(kurt[0] - kurt[1]) > 1.0:
                        if kurt[0] > kurt[1]:
                            normed[n, 0, sidx: eidx] = torch.tanh(normed[n, 0, sidx: eidx])
                        else:
                            normed[n, 1, sidx: eidx] = torch.tanh(normed[n, 1, sidx: eidx])

        # Applying a tanh function to the normed input
        if self.maxnorm:
            cabs = torch.abs(ref)
            cmax = cabs.max(dim=2).values.reshape(x.shape[0], 2, 1)
            normed = normed/cmax

        if self.clipped:
            # normed = torch.clamp(normed, **clipped_params)
            pass
        
        # Conv Backend
        cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)

        # Apply LayerNorm to CNN output before passing to ResNet
        if self.norm_cnn_output:
            cnn_output = self.layernorm_cnn(cnn_output)

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



class Rho2Model1D(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with Res2Net Block 1D
    
    Description - 1D Res2Net model
    
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
                 model_name='res2net1d', 
                 norm_layer: str = 'instancenorm',
                 _input_length: int = 3897,
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.norm_layer = norm_layer
        
        """ Mods """
        # Manipulation layers
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.sigmoid = torch.nn.Sigmoid()
        
        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self.res2net = res2net50_1d().to(dtype=data_type, device=self.store_device)

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
        
        ## Running 1D Res2Net
        raw = self.res2net(normed).squeeze()
        ## Output necessary params
        pred_prob = self.sigmoid(raw)
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob,
                'input': x, 'normed': normed, 'norm_tc': raw, 'norm_mchirp': raw}



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
            self.frontend = res2net50_v1b_26w_4s(pretrained=False, num_classes=512)
        elif resnet_size == 101:
            self.frontend = res2net101_v1b_26w_4s(pretrained=False, num_classes=512)
        elif resnet_size == 152:
            self.frontend = res2net152_v1b_26w_4s(pretrained=False, num_classes=512)
        
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
        elif self.norm_layer == 'groupnorm':
            normed = self.groupnorm(x)
        
        # Conv Backend
        cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)
         
        # Timm Frontend
        out = self.frontend(cnn_output).squeeze() # (batch_size, 512)
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



class KappaModel_ResNet_small(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet50
    
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
        self.frontend = timm.create_model(**timm_params)
        
        # reset_classifier edits the number of outputs that Timm produces
        # The following is set if we need a two-class output from Timm
        # This can be set to a larger value and connected to a linear layer
        self.frontend.reset_classifier(1)
        
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
        self.backend = {'det1': self._det1, 'det2': self._det2}
        self.frontend.to(dtype=data_type, device=self.store_device)
    
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



class PiModel_ResNet_CBAM(torch.nn.Module):
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
        self._det1 = ConvBlock(self.filter_size, self.kernel_size, in_channels=2)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size, in_channels=2)
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

        # Conv Backend
        # Condition the data using PSD
        det1_input = None
        det2_input = None
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



class KappaModel_ResNet(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with ResNet
    
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
                 timm_params: dict = {'model_name': 'resnet50', 'pretrained': True, 'in_chans': 2, 'drop_rate': 0.25},
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
        self.frontend = timm.create_model(**timm_params)
        
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



class OmegaModel_ResNet_CBAM(torch.nn.Module):
    """
    Omega-type Model PE Architecture with ResNet and CBAM
    
    Description - consists of a 2-channel WaveNet backend and a ResNet CBAM frontend
    
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
                 model_name='Wavenet_CBAM', 
                 resnet_size: int = 50,
                 norm_layer: str = 'instancenorm',
                 _input_length: int = 4254,
                 _decimated_bins = None,
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.norm_layer = norm_layer
        self._decimated_bins = _decimated_bins
        
        """ Frontend """
        self.frontend = wavenet_type2(output_size = 224)
        
        """ Backend """
        # resnet50 --> Based on Res2Net blocks
        # Pretrained model is for 3-channels. We use 2 channels.
        if resnet_size == 50:
            self.backend = resnet50_cbam(pretrained=False, in_channels=2)
        elif resnet_size == 152:
            self.backend = resnet152_cbam(pretrained=False, in_channels=2)
        elif resnet_size == 34:
            self.backend = resnet34_cbam(pretrained=False, in_channels=2)
        
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
        self.backend.to(dtype=data_type, device=self.store_device)
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

        # Conv Backend
        # cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)
        feature_extract = self.frontend(normed)

        # Timm Frontend
        out = self.backend(feature_extract) # (batch_size, 512)
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
        return {'raw': raw, 'pred_prob': pred_prob,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr,
                'norm_tc_sigma': tc_var, 'norm_mchirp_sigma': mchirp_var, 'norm_snr_sigma': snr_var,
                'norm_q_sigma': q_var, 'norm_invq_sigma': invq_var, 'norm_dist_sigma': dist_var,
                'norm_dchirp_sigma': dchirp_var, 'input': x, 'normed': normed}
        


class IotaModelPE(torch.nn.Module):
    def __init__(self, 
                 model_name='InceptionTime', 
                 norm_layer: str = 'instancenorm',
                 _input_length: int = 4664,
                 _decimated_bins = None,
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.norm_layer = norm_layer
        
        """ Frontend """
        # InceptionTime module
        self.frontend = InceptionTime_DEF()
        
        """ Mods """
        # Manipulation layers
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
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
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main Layers
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

        # InceptionTime 1D
        out = self.frontend(normed)
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
        return {'raw': raw, 'pred_prob': pred_prob,
                'norm_tc': norm_tc, 'norm_dchirp': norm_dchirp, 'norm_mchirp': norm_mchirp,
                'norm_dist': norm_dist, 'norm_q': norm_q, 'norm_invq': norm_invq, 'norm_snr': norm_snr,
                'tc': tc, 'dchirp': dchirp, 'mchirp': mchirp, 'dist': dist, 'q': q, 'invq': invq, 'snr': snr,
                'norm_tc_sigma': tc_var, 'norm_mchirp_sigma': mchirp_var, 'norm_snr_sigma': snr_var,
                'norm_q_sigma': q_var, 'norm_invq_sigma': invq_var, 'norm_dist_sigma': dist_var,
                'norm_dchirp_sigma': dchirp_var, 'input': x, 'normed': normed}



class KappaModel_Res2Net_branched(torch.nn.Module):
    """
    Kappa-type Model PE Architecture with Res2Net Block
    
    Description - consists of a 2-channel ConvBlock backend and a Timm model frontend
                  this Model-type can be used to test the Kaggle architectures.
                  Has different branches for different signal durations. Combines together
                  in backend.
    
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
                 norm_layer: str = 'instancenorm',
                 _input_length: int = 6239,
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
        # self.frontend = res2net50(pretrained=False)
        self.frontend = res2net50_v1b_26w_4s(pretrained=False, num_classes=250)
        
        """ Mods """
        # Manipulation layers
        self.signal_or_noise = nn.Linear(1000, 1)
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        self.layernorm_cnn = nn.LayerNorm([2, 128, int(_input_length/32.)])
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        
        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.signal_or_noise.to(dtype=data_type, device=self.store_device)
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
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
        
        branch_out = []
        for _ in range(4):
            # Conv Frontend
            cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)
            # Res2Net Backend
            tmp = self.frontend(cnn_output)
            branch_out.append(tmp)

        branch_out = torch.cat(branch_out, dim=0)
        out = self.signal_or_noise(branch_out)

        ## Manipulate encoder output to get params
        # In the Kaggle architecture a dropout is added at this point
        # I see no reason to include at this stage. But we can experiment.
        ## Output necessary params
        raw = self.flatten_d0(out)
        pred_prob = self.sigmoid(raw)
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob,
                'input': x, 'normed': normed, 'norm_tc': raw, 'norm_mchirp': raw}



class UpsilonModel_Res2Net(torch.nn.Module):
    """
    Upsilon-type Model Architecture with Res2Net Block
    
    Description - consists of a 2-channel 1D UNet for whitening followed by
                  a KappaModel-type multiscale-CNN frontend and a Res2Net backend.
                  1D UNet should serve as a pre-whitening network before classification.
    
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
                 norm_layer: str = 'instancenorm',
                 _input_length: int = 3681,
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
        
        """ Whitening """
        self.whitening = UNet1d(in_channels=18)

        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size)
        _initialize_weights(self)
        
        """ Frontend """
        # resnet50 --> Based on Res2Net blocks
        # Pretrained model is for 3-channels. We use 2 channels.
        # self.frontend = res2net50(pretrained=False)
        self.frontend = res2net50_v1b_26w_4s(pretrained=False)
        
        """ Mods """
        # Manipulation layers
        self.ts_reshape = nn.AdaptiveAvgPool1d(6200)
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        self.layernorm_cnn = nn.LayerNorm([2, 128, int(_input_length/32.)])
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        
        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
        self.instancenorm.to(dtype=data_type, device=self.store_device)
        # Main layers
        self.whitening.to(dtype=data_type, device=self.store_device)
        self._det1.to(dtype=data_type, device=self.store_device)
        self._det2.to(dtype=data_type, device=self.store_device)
        self.backend = {'det1': self._det1, 'det2': self._det2}
        self.frontend.to(dtype=data_type, device=self.store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        # Whitening using UNet1D
        whitened = self.whitening(x, 1).sample

        # Normalisation
        if self.norm_layer == 'batchnorm':
            normed = self.batchnorm(whitened)
        elif self.norm_layer == 'dain':
            normed, gate = self.dain(whitened)
        elif self.norm_layer == 'layernorm':
            normed = self.layernorm(whitened)
        elif self.norm_layer == 'instancenorm':
            normed = self.instancenorm(whitened)

        # Conv Backend
        cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)

        # Timm Frontend
        out = self.frontend(cnn_output) # (batch_size, 1000) by default
        ## Manipulate encoder output to get params
        # In the Kaggle architecture a dropout is added at this point
        # I see no reason to include at this stage. But we can experiment.
        ## Output necessary params
        raw = self.flatten_d0(out)
        pred_prob = self.sigmoid(raw)
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'whitened': whitened,
                'input': x, 'normed': normed, 'norm_tc': raw, 'norm_mchirp': raw}



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



class AlphaModel(torch.nn.Module):
    """
    Alpha-type Model Architecture
    
    Description - consists of a 2-channel 2D ResNet with Attention (Stand-Alone Self Attention)
                  and a 2-channel frontend used in KaggleNet (KappaModelPE). Since the time dimension
                  is preserved in this frontend model, it is fine to use a non-attention frontend.
    
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
                 model_name='aresnet2d', 
                 filter_size: int = 32,
                 kernel_size: int = 64,
                 norm_layer: str = 'layernorm',
                 _input_length: int = 3226,
                 _decimated_bins = None,
                 store_device: str = 'cpu',
                 **kwargs):
        
        super().__init__()
        
        self.model_name = model_name
        self.store_device = store_device
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.norm_layer = norm_layer
        # Optional
        self._decimated_bins = _decimated_bins
        
        """ Backend """
        # filters_start=16, kernel_start=32 --> 1.3 Mil. trainable params backend
        # filters_start=32, kernel_start=64 --> 9.6 Mil. trainable params backend
        self._det1 = ConvBlock(self.filter_size, self.kernel_size)
        self._det2 = ConvBlock(self.filter_size, self.kernel_size)
        _initialize_weights(self)
        
        """ Frontend """
        # aresnet38 --> SASA Paper
        self.frontend = AResNet56()
        
        """ Mods """
        # Manipulation layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d((224, 224))
        self.batchnorm = nn.BatchNorm1d(2)
        self.dain = DAIN_Layer(mode='full', input_dim=2)
        self.layernorm = nn.LayerNorm([2, _input_length])
        self.layernorm_cnn = nn.LayerNorm([2, 128, int(_input_length/32.)])
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.flatten_d1 = nn.Flatten(start_dim=1)
        self.flatten_d0 = nn.Flatten(start_dim=0)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        
        ## Convert network into given dtype and store in proper device
        # Manipulation layers
        self.batchnorm.to(dtype=data_type, device=self.store_device)
        self.dain.to(dtype=data_type, device=self.store_device)
        self.layernorm.to(dtype=data_type, device=self.store_device)
        self.layernorm_cnn.to(dtype=data_type, device=self.store_device)
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
        
        # Conv Backend
        cnn_output = torch.cat([self.backend['det1'](normed[:, 0:1]), self.backend['det2'](normed[:, 1:2])], dim=1)

        # Timm Frontend
        cnn_output = self.avg_pool_2d(cnn_output)
        raw = self.frontend(cnn_output).squeeze()
        pred_prob = self.sigmoid(raw)
        
        # Return ouptut params (pred_prob, raw, cnn_output, pe_params)
        return {'raw': raw, 'pred_prob': pred_prob, 'cnn_output': cnn_output,
                'input': x, 'normed': x, 'norm_tc': raw, 'norm_mchirp': raw}



class ZetaModel(torch.nn.Module):
    """
    Zeta-type Model Architecture
    
    Description - consists of a 2-channel 1D ResNet (Virgo-AuTH)
    
    Parameters
    ----------
    num_classes = 1 : int
        Number of classes to ResNet model
        
    """

    def __init__(self, store_device, _input_length: int = 2048, _decimated_bins = None, **kwargs):
       
        super().__init__()
        
        # Initialise Virgo ResNet Model
        self.classifier = virgonet(store_device)
        self.instancenorm = nn.InstanceNorm1d(2, affine=True)
        self.instancenorm.to(dtype=torch.float32, device=store_device)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        x = self.instancenorm(x)
        out = self.classifier.forward(x).squeeze(dim=1)
        return {'raw': out, 'pred_prob': out,
                'input': x, 'normed': x, 'norm_tc': out, 'norm_mchirp': out}



class RhoModel(torch.nn.Module):
    """
    Rho-type Model Architecture
    
    Description - consists of a 2-channel 1D ResNet (Ref: https://github.com/hsd1503/resnet1d)
    
    Parameters
    ----------
    num_classes = 1 : int
        Number of classes to ResNet model
        
    """

    def __init__(self, store_device, _input_length: int = 3226, _decimated_bins = None, **kwargs):
       
        super().__init__()
        
        # Initialise Custom 1D ResNet Model
        self.classifier = ResNet1d(store_device)
        self.layernorm = nn.LayerNorm([2, _input_length])
        self.layernorm.to(dtype=torch.float32, device='cuda:1')
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        x = self.layernorm(x)
        out = self.classifier.forward(x)
        return {'raw': out, 'pred_prob': out,
                'input': x, 'normed': x, 'norm_tc': out, 'norm_mchirp': out}



class PhiModel(torch.nn.Module):
    """
    Phi-type Model Architecture
    
    Description - consists of a 2-channel 1D FCN
    
    Parameters
    ----------
    num_classes = 1 : int
        Number of classes to FCN model
        
    """

    def __init__(self, store_device, _input_length: int = 3226, _decimated_bins = None, **kwargs):
        super().__init__()
        # Initialise Virgo ResNet Model
        self.classifier = FCN(store_device, _input_length)
    
    # x.shape: (batch size, wave channel, length of wave)
    def forward(self, x):
        # batch_size, channel, signal_length = s.shape
        out = self.classifier.forward(x)
        return {'raw': out, 'pred_prob': out,
                'input': x, 'normed': x, 
                'norm_tc': out, 'norm_mchirp': out}



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
