#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

# Packages
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
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ChannelAttention(nn.Module):
    """
    CBAM channel attention gate.

    Squeezes spatial dimensions to a single scalar per channel using both
    average and max pooling, then routes through a shared two-layer MLP and
    sums the results before sigmoid gating.  Reduces channel redundancy by
    re-weighting each feature map proportionally to its global importance.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    ratio : int
        Reduction ratio for the bottleneck MLP (default 16).
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
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    CBAM spatial attention gate.

    Collapses the channel dimension into a 2-channel descriptor (channel-wise
    average and max), then applies a single convolution to produce a spatial
    weight map.  Highlights informative spatial locations while suppressing
    background.

    Parameters
    ----------
    kernel_size : int
        Kernel size of the spatial convolution (default 7; must be odd).
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    """
    ResNet basic residual block with CBAM attention (for ResNet-18/34).

    Structure: Conv-BN-ReLU → Conv-BN → channel attention → spatial attention
    → residual add → ReLU.  Expansion factor is 1 (output channels == planes).

    Parameters
    ----------
    inplanes : int
        Number of input channels.
    planes : int
        Number of output channels (= expansion * planes).
    stride : int
        Stride for the first convolution (default 1).
    downsample : nn.Module or None
        Optional 1×1 projection shortcut when dimensions mismatch.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0,
                 use_blurpool=True):
        # use_blurpool accepted for a uniform block API but unused here: BasicBlock
        # (resnet18/34) is outside the anti-aliasing recipe and stays old-style.
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        # Spatial (channel) dropout on the block output before the residual add.
        # p=0 is a no-op, so default leaves the block unchanged.
        self.drop = nn.Dropout2d(dropout)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
    ResNet bottleneck residual block with CBAM attention (for ResNet-50/101/152).

    Structure: 1×1 Conv-BN-ReLU → 3×3 Conv-BN-ReLU → 1×1 Conv-BN →
    channel attention → spatial attention → residual add → ReLU.
    Expansion factor is 4 (output channels = 4 * planes).

    Parameters
    ----------
    inplanes : int
        Number of input channels.
    planes : int
        Bottleneck width; output channels = 4 * planes.
    stride : int
        Stride applied to the 3×3 convolution (default 1).
    downsample : nn.Module or None
        Optional 1×1 projection shortcut when dimensions mismatch.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0,
                 use_blurpool=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Anti-aliased downsampling (use_blurpool): the stride moves OUT of conv2
        # into a BlurPool applied after it (Zhang, ICML 2019). Otherwise the
        # stride stays in conv2 with an identity blur (pre-df55e89 / o3b_dummy_1).
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
        # Spatial (channel) dropout on the block output before the residual add.
        # p=0 is a no-op, so default leaves the block unchanged.
        self.drop = nn.Dropout2d(dropout)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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
    2D ResNet backbone with CBAM attention in every residual block.

    Adapted from the standard torchvision ResNet to:
    1. Accept multi-channel gravitational-wave inputs (default ``in_channels=2``
       for H1/L1 detector pair).
    2. Insert :class:`ChannelAttention` and :class:`SpatialAttention` gates
       inside every :class:`BasicBlock` and :class:`Bottleneck`.
    3. Output a flat feature vector of size ``num_classes`` (default 512)
       via global average pooling + a linear projection.

    The model is used as the **backend** of the Sage detection network,
    receiving 2D feature maps produced by the per-detector 1D CNN frontend.

    Parameters
    ----------
    block : type
        Residual block class — :class:`BasicBlock` (ResNet-18/34) or
        :class:`Bottleneck` (ResNet-50/101/152).
    layers : list[int]
        Number of blocks per stage, e.g. ``[3, 4, 6, 3]`` for ResNet-50.
    num_classes : int
        Output feature dimension (default 512).
    in_channels : int
        Number of input channels (default 2 for dual-detector).
    """

    def __init__(self, block, layers, num_classes=512, in_channels=2, dropout=0.0,
                 use_blurpool=True, use_resnet_cd=True, zero_init_residual=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # STEM. use_resnet_cd -> ResNet-C deep stem: three stacked 3x3 convs
        # replace one 7x7 (He et al., "Bag of Tricks", CVPR 2019); same 2x
        # downsample, stronger early features at ~equal FLOPs. Else the original
        # single 7x7 stride-2 conv (pre-df55e89 / o3b_dummy_1).
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
        # STEM downsample. use_blurpool -> anti-aliased (dense max stride-1 then
        # BlurPool, Zhang ICML 2019); else the original naive stride-2 max-pool.
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

        # Zero-gamma residual init (He et al., "Bag of Tricks", CVPR 2019): zero
        # the LAST BN scale in each residual block so the block starts as identity
        # (residual branch == 0). Eases optimisation of the deep stack, no
        # inference cost. Must run AFTER the ones-init above (which it overrides).
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
        """Build one ResNet stage as a sequential stack of residual blocks.

        Parameters
        ----------
        block : type
            Block class to instantiate.
        planes : int
            Target channel width for this stage.
        blocks : int
            Number of residual blocks in this stage.
        stride : int
            Stride for the first block's convolution (halves spatial resolution
            when stride=2).

        Returns
        -------
        nn.Sequential
            The stacked residual stage.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_resnet_cd:
                # ResNet-D downsample: average-pool (stride) THEN a 1x1 stride-1
                # conv, instead of a 1x1 stride-2 conv that samples 1-in-4 and
                # discards ~75% of the shortcut (He et al., "Bag of Tricks", 2019).
                downsample = nn.Sequential(
                    (nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
                     if stride != 1 else nn.Identity()),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                # Original 1x1 stride-2 projection shortcut (pre-df55e89).
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
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet18"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet34"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet50"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet101"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls["resnet152"])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model
