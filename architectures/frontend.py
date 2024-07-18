# Modules
import math
import torch
import torch.nn as nn

from torch import conv1d
from typing import Optional
from torch.nn import MaxPool1d, BatchNorm1d
from torch.nn.functional import pad


class Conv1dSame(nn.Conv1d):
    """ Tensorflow like 'SAME' convolution wrapper """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1,
        padding: int = 0, 
        dilation: int = 1, 
        groups: int = 1, 
        bias: bool = False,
    ):
        args = (in_channels, out_channels, kernel_size, stride, 0, dilation, 
                groups, bias)
        super(Conv1dSame, self).__init__(*args)
    
    def get_same_padding(x: int, k: int, s: int, d: int):
        # Calculate asymmetric TensorFlow-like 'SAME' padding
        return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

    def pad_same(x, k: int, s: int, d: int = 1, value: float = 0):
        # Dynamically pad input x with 'SAME' padding for conv
        iw = x.size()[-1]
        pad_w = get_same_padding(iw, k, s, d)
        if pad_w > 0:
            x = pad(x, [pad_w // 2, pad_w - pad_w // 2], value=value)
        return x

    def conv1d_same(
        x, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor] = None, 
        stride: int = 1,
        padding: int = 0, 
        dilation: int = 1, 
        groups: int = 1,
    ):
        x = pad_same(x, weight.shape[-1], stride, dilation)
        return conv1d(x, weight, bias, stride, padding, dilation, groups)

    def forward(self, x):
        return conv1d_same(x, self.weight, self.bias, self.stride[0], 
                           self.padding[0], self.dilation[0], self.groups)


class ConvLayer(nn.Module):
    """ Convolution layer (conv + bn + relu) """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1
    ):
        super(ConvLayer, self).__init__()
        self.conv = Conv1dSame(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride,
            padding, 
            groups=groups, 
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU() # Using the swish function instead of ReLU

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.silu(x)


class Conv1x1(nn.Module):
    """ 1x1 convolution + bn + relu """

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = Conv1dSame(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride,
            padding=0, 
            groups=groups, 
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.silu(x)


class MultiScaleBlock(nn.Module):
    """ Multi-scale feature learning block """

    def __init__(
        self, 
        scales: list,
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int = 1, 
        padding: int = 0, 
        dilation: int = 1, 
        groups: int = 1, 
        bias: bool = False,
    ):
        super().__init__()
        # Multi-scale convolutions
        convs = []
        for scale in scales:
            ksize = math.floor(kernel_size * scale)
            convs.append(ConvLayer(in_channels, out_channels, ksize))
        self.convs = nn.ModuleList(convs)
        # Blurring 1x1 convolution
        self.conv1x1 = Conv1x1(out_channels * 5 + in_channels, out_channels)

    def forward(self, x):
        ms = [conv(x) for conv in self.convs]
        x = torch.cat(ms+[x], dim=1)
        x = self.conv1x1(x)
        return x


class MSFeatureExtractor(nn.Module):
    """ 
    Multi-scale frontend feature extractor 
    
    Example MSFeatureExtractor:
    Default parameters describe the following
      
    
                     BLOCK 1
            MSBlock(in=1, out=32, kernel=64)
        MSBlock(in=32, out=32, kernel=64//2+1)
                MaxPool(stride=8)
                        |
                        V
                     BLOCK 2
        MSBlock(in=32, out=64, kernel=64//2+1)
        MSBlock(in=64, out=64, kernel=64//4+1)
                MaxPool(stride=4)
                        |
                        V
                     BLOCK 3
        MSBlock(in=64, out=128, kernel=64//4+1)
        MSBlock(in=128, out=128, kernel=64//4+1)
                        |
                        V
                    2D OUTPUT

    """

    def __init__(
        self,
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
        in_channels: int = 1
    ):
        super().__init__()
        # Feature Extractor Blocks
        block_metadata = (blocks, out_channels, 
                          base_kernel_sizes, compression_factor)

        all_modules = []
        in_chans = in_channels
        # Iterating through feature extractor blocks
        for (block, out_chans, base_kernels, comp) in zip(*block_metadata):
            # Iterating through each multi-scale block within
            for msb, oc, bk in zip(block, out_chans, base_kernels):
                all_modules.append(msb(scales, in_chans, oc, bk))
                # Out channels of current block is in channels of next block
                # Pooling does not change num channels
                in_chans = oc
            # Adding compression blocks if required
            if not comp:
                all_modules.append(MaxPool1d(kernel_size=comp, stride=comp))

        # All modules can be put into one Sequential module
        self.feature_extractor = nn.Sequential(*all_modules)
        # Initialise the weights for all layers
        self._init_params()
    
    def _init_params(self):
        # Initialise weights and biases for layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.feature_extractor(x).unsqueeze(1)
        return x