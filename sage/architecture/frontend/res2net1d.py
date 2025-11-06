#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = res2net1d.py
Description     = 1D version of the Res2Net Block
                  (Refer: https://arxiv.org/abs/1904.01169)

Created on Fri Jul 28 10:00:50 2023

__author__      = nnarenraju
__copyright__   = Copyright 2023, ORChiD
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = inProgress


Github Repository: NULL

Documentation: NULL

"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ['Res2Net', 'res2net50']


model_urls = {
    'res2net50_26w_4s': '',
    'res2net50_48w_2s': '',
    'res2net50_14w_8s': '',
    'res2net50_26w_6s': '',
    'res2net50_26w_8s': '',
    'res2net101_26w_4s': '',
}


class Bottle2neck(nn.Module):
    
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, 
                 basewidth=26, scale = 4, stype='normal'):
        """ 
        Constructor
        
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            basewidth: basic width of conv3x3
            scale: number of scale.
            stype: {'normal': normal set, 'stage': first block of a new stage}
        
        """
        
        super(Bottle2neck, self).__init__()
        
        width = int(math.floor(out_channels * (basewidth/64.0)))
        wscale = width * scale
        
        """ STEP 1 """
        # Initial 1X1 convolution of the Res2Net block
        self.conv1 = nn.Conv1d(in_channels, wscale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(wscale)
        
        """ STEP 2 """
        # Number of scales to analyse (dynamic range)
        self.nums = 1 if scale == 1 else scale - 1
        
        # If first block of new stage, apply average pooling 
        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=3, stride = stride, padding=1)
        
        # Define 3X3 convolutions for each scale
        convs2 = []
        bns2 = []
        for _ in range(self.nums):
            convs2.append(nn.Conv1d(width, width, kernel_size=3, stride=stride, 
                                   padding=1, bias=False))
            bns2.append(nn.BatchNorm1d(width))
        
        # Convert layers into modules
        self.convs2 = nn.ModuleList(convs2)
        self.bns2 = nn.ModuleList(bns2)
        
        """ STEP 3 """
        # Final 1X1 convolution of the Res2Net block
        self.conv3 = nn.Conv1d(wscale, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        ## Opts
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        # ResNet residual
        residual = x
        
        ## STEP 1: 1X1 convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        ## STEP 2: Learning different scales
        # Splitting the output of 1X1 convolution into chunks
        spx = torch.split(out, self.width, 1)
        # Each chunk is then passed through a convolution
        # The output of previous chunks is added to the next chunk
        for i in range(self.nums):
            # Add previous input or not
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Compute (convolution, norm, activation) block
            sp = self.convs2[i](sp)
            sp = self.relu(self.bns2[i](sp))
            # concatenate outputs together
            out = sp if i == 0 else torch.cat((out, sp), 1)
        
        # Concatenate outputs from different scales before merging
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])),1)
        
        ## STEP 3: 1X1 convolution
        # Merging different scales using 1X1 convolution
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Downsampling (if required)
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Add residuals back to output (Put the Res in ResNet)
        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, basewidth=26, scale=4, num_classes=1):
        super(Res2Net, self).__init__()
        # Basic params
        self.in_channels = 64
        self.basewidth = basewidth
        self.scale = scale
        
        # Initial stage of Res2Net
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Deeper layers of Res2Net
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Final embedding layer/ output layer
        # If using (batch_size, 1000) embedding, we can perform PE layer 
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        # Downsample (if necessary)
        # This is the number of input channels from initial stage
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        
        # Apply Res2Net block for downsampling
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, 
                            downsample=downsample, stype='stage', 
                            basewidth = self.basewidth, scale=self.scale))
        
        # in_channels is fine already. If not, downsampling should fix it.
        self.in_channels = out_channels * block.expansion
        
        # Apply main Res2Net blocks
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 
                                basewidth = self.basewidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Generic forward method for all Res2Net depths
        x = self.conv1(x)    # [2, 2, 2048]
        x = self.bn1(x)      # [2, 64, 1024]
        x = self.relu(x)     # [2, 64, 1024]
        x = self.maxpool(x)  # [2, 64, 512]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50(**kwargs):
    """
    Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], basewidth = 26, scale = 4, 
                    **kwargs)
    return model

def res2net50_26w_4s(**kwargs):
    """
    Constructs a Res2Net-50_26w_4s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], basewidth = 26, scale = 4, 
                    **kwargs)
    return model

def res2net101_26w_4s(**kwargs):
    """
    Constructs a Res2Net-50_26w_4s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], basewidth = 26, scale = 4, 
                    **kwargs)
    return model

def res2net50_26w_6s(**kwargs):
    """
    Constructs a Res2Net-50_26w_4s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], basewidth = 26, scale = 6, 
                    **kwargs)
    return model

def res2net50_26w_8s(**kwargs):
    """
    Constructs a Res2Net-50_26w_4s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], basewidth = 26, scale = 8, 
                    **kwargs)
    return model

def res2net50_48w_2s(**kwargs):
    """
    Constructs a Res2Net-50_48w_2s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], basewidth = 48, scale = 2, 
                    **kwargs)
    return model

def res2net50_14w_8s(**kwargs):
    """
    Constructs a Res2Net-50_14w_8s model.
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], basewidth = 14, scale = 8, 
                    **kwargs)
    return model


if __name__ == '__main__':
    time_series = torch.rand(2, 2, 3226)
    model = res2net101_26w_4s()
    model = model
    print(model(time_series).size())
