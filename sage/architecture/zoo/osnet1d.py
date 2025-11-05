# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename         =  foobar.py
Description      =  Lorem ipsum dolor sit amet

Created on 19/02/2024 at 15:56:53

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

Documentation:

Modified OSnet architecture for 1D time series analysis
Original Github Link: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet_ain.py#L309
Link to Paper: https://arxiv.org/abs/1905.00953

Major Modifications && Notes:

1. Converting OSnet from 2D to 1D (needs to extract features from time series)
2. Throwing away layers that don't contribute to our purpose
3. Extracting features using large kernels (Link: https://openaccess.thecvf.com/content_cvpr_2017/papers/Peng_Large_Kernel_Matters_CVPR_2017_paper.pdf)
4. Making the network fully convolutional (no fully connected layers) and removing global pooling layers (both localisation and classification is better)
5. Using a more stable activation function (getting rid of ReLU -> probably Leaky ReLU, SiLU or GELU)
6. Input has been manufactured to be exactly 4096 samples in length. Set strides to get it down to 128 across one dimension.
7. 20 Hz (signal low freq cutoff) at 220 Hz sampling freq will require at least kernel size = 11 to capture one full cycle.
8. AveragePool or MaxPool or both (currently this in OSnet)? Maybe even MinPool for funsies.
   Thinking about it more: MaxPool gets bright features from darker images. MinPool should do the opposite.
   What if MinPool actually helps in gathering low-SNR features? Maybe try it for realsies.

NOTES:
1. The first two values in strides var is used by initial dim reduction layers
2. The last two values in strides var is used by maxpool after bottlenecks
3. If using initial dim reduction, all four strides are done. Make sure to set them to proper values.

Example usage:

[1] Custom kernels and feature extraction without dim reduction

# All Custom Kernels
kernel_sizes = []
tmp = []
for i in np.arange(1, 7)[::-1]:
    tmp.append([(2**i)*n + 1 for n in range(1, 6)])
    if i%2!=0:
        kernel_sizes.append(tmp)
        tmp = []

# Without initial dim reduction, num channels starts with channels[1]
network = osnet_ain_custom(
            channels=[16, 64, 96, 128],
            kernel_sizes=kernel_sizes,
            strides=[2,2,8,4],
            stacking=False,
            initial_dim_reduction=False
        )

input = torch.randn(1, 1, 4096)
output = network(input)
print(output.shape)

[2] Custom Kernels and feature extraction with dim reduction

# Kernel sizes on modified OSnet (type 1odd)
# We could re-use the kernel sizes to make up six bottleneck layers
kernel_sizes = []
kernel_sizes.append([[17, 33, 65, 129, 257], [9, 17, 33, 65, 129]])
kernel_sizes.append([[9, 17, 33, 65, 129], [3, 5, 9, 17, 33]])
kernel_sizes.append([[3, 5, 9, 17, 33], [3, 5, 9, 17, 33]])

network = osnet_ain_custom(
            channels=[16, 32, 64, 128],
            kernel_sizes=kernel_sizes,
            strides=[4,2,2,2],
            stacking=False,
            initial_dim_reduction=True
        )

input = torch.randn(64, 1, 4096)
output = network(input)
print(output.shape)

[3] Custom kernels similar to KaggleNet

# Kernel sizes on modified OSnet (type 1)
# We could re-use the kernel sizes to make up six bottleneck layers
kernel_sizes = []
kernel_sizes.append([[16, 32, 64, 128, 256], [8, 16, 32, 64, 128]])
kernel_sizes.append([[8, 16, 32, 64, 128], [2, 4, 8, 16, 32]])
kernel_sizes.append([[2, 4, 8, 16, 32], [2, 4, 8, 16, 32]])

# Without initial dim reduction, num channels starts with channels[1]
# We can reduce these number for a smaller network
# ChannelGate has a reduction factor which is 16 by default. Has issues with setting channels=32.
network = osnet_ain_custom(
            channels=[16, 32, 64, 128],
            kernel_sizes=kernel_sizes,
            strides=[2,2,8,4],
            stacking=False,
            initial_dim_reduction=False
        )

input = torch.randn(64, 1, 4096)
output = network(input)
print(output.shape)

"""

# Packages
import numpy as np
import warnings
import math

# Machine learning
import torch
from torch import nn
from torch.nn import functional as F

############################# BASIC LAYERS ################################
# MODIFIED
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm1d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU() # Using the swish function instead of ReLU

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.silu(x)

# MODIFIED
class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.silu(x)

# MODIFIED
class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride, 
            padding=0, 
            bias=False
        )
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

# MODIFIED
class LightConvNxN(nn.Module):
    """Lightweight NxN convolution.

    1x1 (linear) + dw NxN (nonlinear).
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(LightConvNxN, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        # Before applying kernel size, use padding=same if required
        padding = 'same' if kernel_size != 3 else 1
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.silu(x)

# MODIFIED
class LightConvStream(nn.Module):
    """Lightweight convolution stream. Stacking conv layers for larger receptive field."""

    def __init__(self, in_channels, out_channels, depth, kernel_size):
        super(LightConvStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(
            depth
        )
        layers = []
        layers += [LightConvNxN(in_channels, out_channels, kernel_size)]
        for i in range(depth - 1):
            layers += [LightConvNxN(out_channels, out_channels, kernel_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


########################## BUILDING BLOCKS FOR OMNI-SCALE NETWORK ################################
# MODIFIED
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        # Global average pooling here is only used to get channel-wise weights
        # Each scale learnt is given an individual weight
        # The vector of weights is then multiplied with the different scales to get weighted scales
        # This does not get rid of localisation features in any way
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.fc2 = nn.Conv1d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU()
        elif gate_activation == 'silu':
            self.gate_activation = nn.SiLU()
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        # Each scale should now be weighted
        return input * x

# MODIFIED
class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, kernel_sizes, channel_gate_reduction=16, reduction=4, T=5, stacking=False, **kwargs):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        # Stacking conv layers for increased receptive field and low cost
        # We can either use non-stacked large kernels or stacked small kernels
        # k=3 with T=5 gives an effective receptive field of (2*5+1=11)
        self.conv2 = nn.ModuleList()
        if stacking:
            # Iterating through different scales
            for n, t in enumerate(range(1, T + 1)):
                self.conv2 += [LightConvStream(mid_channels, mid_channels, t, kernel_sizes[n])]
        else:
            # Using larger kernel sizes without stacking
            for kernel_size in kernel_sizes:
                self.conv2 += [LightConvNxN(mid_channels, mid_channels, kernel_size)]
        
        self.gate = ChannelGate(mid_channels, reduction=channel_gate_reduction)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.silu(out)

# MODIFIED
class OSBlockINin(nn.Module):
    """Omni-scale feature learning block with instance normalization."""

    def __init__(self, in_channels, out_channels, kernel_sizes, channel_gate_reduction=16, reduction=4, T=5, stacking=False, **kwargs):
        super(OSBlockINin, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        # Stacking conv layers for increased receptive field and low cost
        # We can either use non-stacked large kernels or stacked small kernels
        # k=3 with T=5 gives an effective receptive field of (2*5+1=11)
        self.conv2 = nn.ModuleList()
        if stacking:
            # Iterating through different scales
            for n, t in enumerate(range(1, T + 1)):
                self.conv2 += [LightConvStream(mid_channels, mid_channels, t, kernel_sizes[n])]
        else:
            # Using larger kernel sizes without stacking
            for kernel_size in kernel_sizes:
                self.conv2 += [LightConvStream(mid_channels, mid_channels, 1, kernel_size)]
        
        self.gate = ChannelGate(mid_channels, reduction=channel_gate_reduction)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm1d(out_channels, affine=True)
    
    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        x3 = self.IN(x3) # IN inside residual
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)

######################################## NETWORK ARCHITECTURE ########################################
# MODIFIED
class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        blocks,
        layers,
        channels,
        kernel_sizes,
        strides,
        channel_gate_reduction,
        conv1_IN=False,
        in_channels=1,
        stacking=True,
        initial_dim_reduction = False
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1

        # options
        self.initial_dim_reduction = initial_dim_reduction

        # convolutional backbone
        self.conv1 = ConvLayer(
            in_channels, channels[0], 7, stride=strides[0], padding=3, IN=conv1_IN
        )
        self.maxpool = nn.MaxPool1d(3, stride=strides[1], padding=1)

        ## OSnet bottlenecks and dimensionality reduction
        # conv2 = bottleneck x2
        if not self.initial_dim_reduction:
            channels_init = in_channels
        else:
            channels_init = channels[0]
        self.conv2 = self._make_layer(
            blocks[0], layers[0], kernel_sizes[0], channels_init, channels[1], stacking, channel_gate_reduction
        )
        # pool2 = 1x1 conv + 2x2 avg pool + !!!stride 2!!!
        # Length of the array reduced by x2
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]), nn.AvgPool1d(strides[2], stride=strides[2])
        )

        # conv3 = bottleneck x2
        self.conv3 = self._make_layer(
            blocks[1], layers[1], kernel_sizes[1], channels[1], channels[2], stacking, channel_gate_reduction
        )
        # pool3 = 1x1 conv + 2x2 avg pool + !!!stride 2!!!
        # Length of the array reduced by x4
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]), nn.AvgPool1d(strides[3], stride=strides[3])
        )

        # conv4 = bottleneck x2
        self.conv4 = self._make_layer(
            blocks[2], layers[2], kernel_sizes[2], channels[2], channels[3], stacking, channel_gate_reduction
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        
        self._init_params()

    def _make_layer(self, blocks, layer, kernel_sizes, in_channels, out_channels, stacking, channel_gate_reduction):
        # I'm guessing layer variable is not used here because it's always (2,2,2)
        layers = []
        layers += [blocks[0](in_channels, out_channels, kernel_sizes=kernel_sizes[0], channel_gate_reduction=channel_gate_reduction, stacking=stacking)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels, kernel_sizes=kernel_sizes[i], channel_gate_reduction=channel_gate_reduction, stacking=stacking)]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.InstanceNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        if self.initial_dim_reduction:
            x = self.conv1(x)
            x = self.maxpool(x)
        # Rest of OSnet
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x).unsqueeze(1)
        return x

    def forward(self, x):
        # Feature maps is all we need from frontend
        return self.featuremaps(x)

########################################## OSNET INITIALISATION #########################################
def osnet_ain_custom(channels, kernel_sizes, strides, stacking=True, initial_dim_reduction=False,
                     channel_gate_reduction=16):
    model = OSNet(
        blocks=[
            [OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock]
        ],
        layers=[2, 2, 2],
        channels=channels,
        conv1_IN=True,
        kernel_sizes=kernel_sizes,
        strides=strides,
        channel_gate_reduction=channel_gate_reduction,
        stacking=stacking,
        initial_dim_reduction = initial_dim_reduction
    )
    return model