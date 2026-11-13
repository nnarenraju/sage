#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D ResNet backend with CBAM channel and spatial attention in every
residual block.

Filename        : resnet2d_cbam.py
Description     : Short description of the file

Created on 2026-03-06 13:29:06

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ..antialias import BlurPool2d


__all__ = [
    "ResNet",
    "resnet18_cbam",
    "resnet34_cbam",
    "resnet50_cbam",
    "resnet101_cbam",
    "resnet152_cbam",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding 1 and no bias.

    Parameters
    ----------
    in_planes, out_planes : int
        Input and output channels.
    stride : int
        Convolution stride.

    Returns
    -------
    nn.Conv2d
        The convolution.
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ChannelAttention(nn.Module):
    """
    CBAM channel attention weights.

    Average- and max-pooled channel descriptors pass through a shared
    two-layer 1x1-convolution MLP; their sum goes through a sigmoid.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    ratio : int
        Unused; the MLP bottleneck is ``in_planes // 16``.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Channel weights of ``x``.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, C, H, W)``
            Input features.

        Returns
        -------
        torch.Tensor, shape ``(B, C, 1, 1)``
            Sigmoid weights per channel.
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    CBAM spatial attention weights.

    The channel-wise mean and max pass through one convolution and a
    sigmoid.

    Parameters
    ----------
    kernel_size : int
        Kernel size of the convolution; odd keeps the spatial size.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Spatial weights of ``x``.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, C, H, W)``
            Input features.

        Returns
        -------
        torch.Tensor, shape ``(B, 1, H, W)``
            Sigmoid weights per position.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    """
    Basic residual block with CBAM attention, for ResNet-18/34.

    Conv-BN-ReLU, Conv-BN, channel then spatial attention, dropout,
    residual add, ReLU. Output channels equal ``planes``.

    Parameters
    ----------
    inplanes : int
        Number of input channels.
    planes : int
        Number of output channels.
    stride : int
        Stride of the first convolution.
    downsample : nn.Module or None
        Projection shortcut when the shapes differ.
    dropout : float
        Channel dropout rate before the residual add.
    use_blurpool : bool
        Unused; keeps the block signature uniform.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0,
                 use_blurpool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.drop = nn.Dropout2d(dropout)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Residual block output.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, inplanes, H, W)``
            Input features.

        Returns
        -------
        torch.Tensor, shape ``(B, planes, H / stride, W / stride)``
            Output features.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.drop(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block with CBAM attention, for ResNet-50+.

    1x1 Conv-BN-ReLU, 3x3 Conv-BN-ReLU (strided or blur-pooled),
    1x1 Conv-BN, channel then spatial attention, dropout, residual add,
    ReLU.

    Parameters
    ----------
    inplanes : int
        Number of input channels.
    planes : int
        Bottleneck width; output channels are ``4 * planes``.
    stride : int
        Downsampling factor of the block.
    downsample : nn.Module or None
        Projection shortcut when the shapes differ.
    dropout : float
        Channel dropout rate before the residual add.
    use_blurpool : bool
        Downsample with a BlurPool after a stride-1 3x3 convolution
        instead of a strided 3x3 convolution.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0,
                 use_blurpool=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # With use_blurpool the stride moves from conv2 to a BlurPool
        # after it
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=(1 if use_blurpool else stride), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.blur2 = (BlurPool2d(planes, stride=stride)
                      if (use_blurpool and stride != 1) else nn.Identity())
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
        self.drop = nn.Dropout2d(dropout)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Residual block output.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, inplanes, H, W)``
            Input features.

        Returns
        -------
        torch.Tensor, shape ``(B, 4 * planes, H / stride, W / stride)``
            Output features.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.blur2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.drop(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    2D ResNet with CBAM attention in every residual block.

    The torchvision ResNet with ``in_channels`` input channels; global
    average pooling and a linear layer give ``num_classes`` features.

    Parameters
    ----------
    block : type
        :class:`BasicBlock` (ResNet-18/34) or :class:`Bottleneck`
        (ResNet-50/101/152).
    layers : list of int
        Number of blocks per stage, e.g. ``[3, 4, 6, 3]`` for ResNet-50.
    num_classes : int
        Output feature dimension.
    in_channels : int
        Number of input channels.
    dropout : float
        Channel dropout rate in every residual block.
    use_blurpool : bool
        Anti-aliased (BlurPool) downsampling in the stem and the
        bottleneck blocks.
    use_resnet_cd : bool
        ResNet-C deep stem and ResNet-D average-pool shortcuts.
    zero_init_residual : bool
        Zero the last BatchNorm scale of every residual block.
    """

    def __init__(self, block, layers, num_classes=512, in_channels=2, dropout=0.0,
                 use_blurpool=True, use_resnet_cd=True, zero_init_residual=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # With use_resnet_cd, the ResNet-C deep stem: three 3x3
        # convolutions for one 7x7
        if use_resnet_cd:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Stem downsample: stride-1 max-pool then BlurPool, or a
        # stride-2 max-pool
        if use_blurpool:
            self.maxpool = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                BlurPool2d(64, stride=2),
            )
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        _lk = dict(dropout=dropout, use_blurpool=use_blurpool,
                   use_resnet_cd=use_resnet_cd)
        self.layer1 = self._make_layer(block, 64, layers[0], **_lk)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **_lk)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **_lk)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **_lk)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero the last BN scale of each block so it starts as the
        # identity; overrides the ones-init above
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def _make_layer(self, block, planes, blocks, stride=1, dropout=0.0,
                    use_blurpool=True, use_resnet_cd=True):
        """
        One ResNet stage: a stack of residual blocks.

        Parameters
        ----------
        block : type
            Block class to instantiate.
        planes : int
            Channel width of the stage.
        blocks : int
            Number of residual blocks.
        stride : int
            Downsampling factor of the first block.
        dropout : float
            Channel dropout rate of every block.
        use_blurpool : bool
            Passed to every block.
        use_resnet_cd : bool
            ResNet-D shortcut: average pool then a stride-1 1x1
            convolution, instead of a strided 1x1 convolution.

        Returns
        -------
        nn.Sequential
            The stage.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_resnet_cd:
                # ResNet-D shortcut
                downsample = nn.Sequential(
                    (nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
                     if stride != 1 else nn.Identity()),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                # Strided 1x1 projection shortcut
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            dropout=dropout, use_blurpool=use_blurpool))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=dropout,
                                use_blurpool=use_blurpool))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Backend features.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, in_channels, H, W)``
            Input feature maps.

        Returns
        -------
        torch.Tensor, shape ``(B, num_classes)``
            Features.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_cbam(pretrained=False, **kwargs):
    """
    ResNet-18 with CBAM attention.

    Parameters
    ----------
    pretrained : bool
        Load the torchvision ImageNet weights into the model.
    **kwargs
        Passed to :class:`ResNet`.

    Returns
    -------
    ResNet
        The backend.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet18"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """
    ResNet-34 with CBAM attention.

    Parameters
    ----------
    pretrained : bool
        Load the torchvision ImageNet weights into the model.
    **kwargs
        Passed to :class:`ResNet`.

    Returns
    -------
    ResNet
        The backend.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet34"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """
    ResNet-50 with CBAM attention.

    Parameters
    ----------
    pretrained : bool
        Load the torchvision ImageNet weights into the model.
    **kwargs
        Passed to :class:`ResNet`.

    Returns
    -------
    ResNet
        The backend.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet50"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """
    ResNet-101 with CBAM attention.

    Parameters
    ----------
    pretrained : bool
        Load the torchvision ImageNet weights into the model.
    **kwargs
        Passed to :class:`ResNet`.

    Returns
    -------
    ResNet
        The backend.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet101"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """
    ResNet-152 with CBAM attention.

    Parameters
    ----------
    pretrained : bool
        Load the torchvision ImageNet weights into the model.
    **kwargs
        Passed to :class:`ResNet`.

    Returns
    -------
    ResNet
        The backend.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet152"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model
