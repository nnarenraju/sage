# Packages
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 in_channels=1,
                 classes=128,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.in_channels = in_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.compress = nn.AdaptiveAvgPool2d((128, 128))

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.in_channels,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=1,
                                  bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)
        
        self.receptive_field = receptive_field
    
    def dilate(self, x, dilation, init_dilation=1, pad_start=True):
        [n, c, l] = x.size()
        dilation_factor = dilation / init_dilation
        if dilation_factor == 1:
            return x

        # zero padding for reshaping
        new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
        if new_l != l:
            l = new_l
            diff = new_l - x.shape[2]
            x = torch.nn.functional.pad(x, (diff, 0), mode='constant', value=0)

        l_old = int(round(l / dilation_factor))
        n_old = int(round(n * dilation_factor))
        l = math.ceil(l * init_dilation / dilation)
        n = math.ceil(n * dilation / init_dilation)

        # reshape according to dilation
        x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
        x = x.view(c, l, n)
        x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

        return x

    def wavenet(self, input, dilation_func):

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, i)

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = F.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = F.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                 s = self.dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, (self.kernel_size - 1):]

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = self.dilate(input, dilation, init_dilation)
        return x
    
    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def forward(self, input):
        x = self.wavenet(input, dilation_func=self.wavenet_dilate)
        x = x[:].unsqueeze(0)
        x = self.compress(x).squeeze(0)
        # print('wavenet output = n:{}, c:{}, l:{}'.format(n, c, l))
        # print(self.parameter_count())
        x = x[:].unsqueeze(1)
        return x