#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : resnet3d_cbam.py
Description     : Short description of the file

Created on 2026-03-05 13:52:22

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = MIT Licence
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
import torch.nn.functional as F


__all__ = [
    "ResNet",
    "resnet18_cbam",
    "resnet34_cbam",
    "resnet50_cbam",
    "resnet101_cbam",
    "resnet152_cbam",
]


def conv1x3x3(in_planes, out_planes, stride=1, dilation=1):
    """
    3D convolution with kernel ``(1, 3, 3)`` — spatial only, depth-1 so the
    detector dimension is unchanged.  Used inside residual blocks.
    """
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, 1),
        padding=(0, dilation, dilation),
        dilation=(1, 1, dilation),
        bias=False,
    )


def conv2x1x1(in_planes, out_planes):
    """
    3D convolution with kernel ``(2, 1, 1)`` — collapses the detector
    dimension from 2 to 1 while keeping spatial dimensions intact.  Used
    in :class:`DetectorFusion` to merge H1/L1 feature maps.
    """
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(2, 1, 1),
        stride=1,
        padding=0,
        bias=False,
    )


class ChannelAttention(nn.Module):
    """
    3D CBAM channel-attention gate.

    Computes per-channel weights by applying a shared MLP to both
    average-pooled and max-pooled global 3D descriptors, then combines them
    with a sigmoid.  The ratio-16 bottleneck keeps the parameter count low.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    ratio : int
        Channel reduction ratio for the bottleneck FC (default ``16``).
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    3D CBAM spatial-attention gate.

    Concatenates channel-wise average and max statistics along the channel
    dimension and passes them through a single 3D convolution to produce a
    voxel-wise attention map.

    Parameters
    ----------
    kernel_size : int
        Spatial kernel size for the attention convolution (default ``7``).
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(
            2,
            1,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, kernel_size // 2, kernel_size // 2),
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DetectorAttention(nn.Module):
    """
    Cross-detector channel-wise attention gate (3D tensor, D=2 detectors).

    Pools over the spatial (H, W) dimensions, applies a per-detector FC to
    get importance weights, normalises with softmax over the detector axis,
    and rescales the input feature map.  Allows the network to suppress one
    detector's contribution when it carries less useful information (e.g. a
    glitch).

    Parameters
    ----------
    channels : int
        Number of feature channels (``C`` in the ``B, C, D, H, W`` tensor).
    """

    def __init__(self, channels):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        B, C, D, H, W = x.shape

        # average spatially
        pooled = x.mean(dim=[3, 4])  # (B,C,D)
        pooled = pooled.permute(0, 2, 1)  # (B,D,C)
        weights = self.fc(pooled)  # (B,D,C)
        weights = weights.permute(0, 2, 1)  # (B,C,D)
        weights = self.softmax(weights)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return x * weights


class DetectorFusion(nn.Module):
    """
    Learned cross-detector feature mixing via a temporal (D=2) convolution.

    Applies a 3D conv with kernel ``(2, 1, 1)`` and output padding ``(1, 0,
    0)`` to mix the two detector feature maps, then trims the result back to
    ``D=2`` so the shape is preserved.  This lets the network learn how to
    combine H1 and L1 features before deeper processing.

    Parameters
    ----------
    channels : int
        Number of feature channels.
    """

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(2, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )

        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = x[:, :, 0:2, :, :]
        x = self.bn(x)
        x = self.relu(x)

        return x


class CoherenceAttention(nn.Module):
    """
    Inter-detector coherence gating module.

    Computes a voxel-wise coherence map from the normalised dot product of the
    two detector feature volumes (``D=0`` and ``D=1``), then uses a 1×1×1
    convolution + sigmoid to gate the full feature tensor.  Bins where the two
    detectors are incoherent (noise-only) are suppressed; bins where they agree
    (signal) are amplified.

    Parameters
    ----------
    channels : int
        Number of feature channels.
    """

    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 1, 1),
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x: (B,C,2,H,W)
        det0 = x[:, :, 0, :, :]
        det1 = x[:, :, 1, :, :]

        det0 = F.normalize(det0, dim=1)
        det1 = F.normalize(det1, dim=1)

        coherence = det0 * det1
        coherence = coherence.unsqueeze(2)
        coherence = coherence.repeat(1, 1, 2, 1, 1)
        coherence = self.conv(coherence)
        coherence = self.sigmoid(coherence)

        return x * coherence


class TemporalTransformer(nn.Module):
    """
    Lightweight multi-head self-attention block over the frequency (W) axis.

    Collapses the spatial (D, H) dimensions via mean pooling, runs a
    multi-head attention + FFN layer over the W (time/frequency) tokens, and
    re-expands the result to a shape compatible with subsequent pooling.
    Provides long-range temporal context that conv layers alone cannot model.

    Parameters
    ----------
    channels : int
        Feature channel dimension (``embed_dim`` for the attention layer).
    heads : int
        Number of attention heads (default ``4``).
    """

    def __init__(self, channels, heads=4):

        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=heads, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)

        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x):

        B, C, D, H, W = x.shape

        # collapse spatial dims
        x = x.mean(dim=[2, 3])  # (B,C,W)
        x = x.permute(0, 2, 1)  # (B,W,C)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm(x + self.ff(x))
        x = x.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)

        return x


class BasicBlock(nn.Module):
    """
    3D ResNet-18/34 residual block with CBAM + DetectorAttention.

    Two ``1×3×3`` convolutions with BN/ReLU, followed by channel attention,
    spatial attention, and detector attention gates before the residual add.
    The ``1×`` depth kernel preserves the detector dimension (D=2) so each
    detector's features remain independent until later fusion layers.

    Parameters
    ----------
    inplanes : int
        Input channel count.
    planes : int
        Output channel count.
    stride : int
        Spatial stride applied in the first convolution (default ``1``).
    downsample : nn.Module or None
        Optional projection shortcut to match dimensions.
    dilation : int
        Dilation for the 3×3 spatial kernel (default ``1``).
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes, 1, dilation)
        self.bn2 = nn.BatchNorm3d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.da = DetectorAttention(planes)

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
        out = self.da(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    3D ResNet-50/101/152 bottleneck residual block with CBAM + DetectorAttention.

    Three-layer bottleneck (``1×3×3 → 1×3×3 → 1×1×1``) with expansion
    factor 4, followed by channel attention, spatial attention, and detector
    attention gates before the residual add.  Dilation is supported in both
    convolutional layers to increase the receptive field without striding.

    Parameters
    ----------
    inplanes : int
        Input channel count.
    planes : int
        Intermediate channel count; output is ``planes * 4``.
    stride : int
        Spatial stride applied in the first convolution (default ``1``).
    downsample : nn.Module or None
        Optional projection shortcut to match dimensions.
    dilation : int
        Dilation for spatial 3×3 kernels (default ``1``).
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = conv1x3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.conv2 = nn.Conv3d(
        #    planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        # )
        self.conv2 = conv1x3x3(planes, planes, 1, dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
        self.da = DetectorAttention(planes * 4)

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

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.da(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    3D ResNet backbone with CBAM + detector-aware attention modules.

    An extension of the standard 2D ResNet that processes detector feature
    volumes of shape ``(B, 1, D=2, H, W)`` with 3D convolutions.  In addition
    to the per-block CBAM gates, the backbone integrates:

    - **DetectorAttention** in every residual block to weight each detector's
      contribution channel-wise.
    - **DetectorFusion** after layer 2 to mix H1/L1 features.
    - **CoherenceAttention** after layer 2 to gate on inter-detector agreement.
    - **TemporalTransformer** after layer 4 for long-range context over the
      frequency axis.
    - **Learned detector embedding** added to the stem output for positional
      discrimination of the two detector channels.

    Parameters
    ----------
    block : type
        Residual block class — :class:`BasicBlock` or :class:`Bottleneck`.
    layers : list[int]
        Number of blocks in each of the four stages.
    num_classes : int
        Output feature dimension of the final FC layer (default ``512``).
    in_channels : int
        Number of input channels (default ``1`` for single-channel 3D volume).
    """

    def __init__(self, block, layers, num_classes=512, in_channels=1):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        self.detector_embed = nn.Embedding(2, 64)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.detector_fusion = DetectorFusion(128 * block.expansion)
        self.coherence_attention = CoherenceAttention(128 * block.expansion)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=4)
        self.temporal_transformer = TemporalTransformer(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        # detector embedding
        B, C, D, H, W = x.shape
        det_ids = torch.arange(D, device=x.device)
        det_embed = self.detector_embed(det_ids)  # (D,C)
        det_embed = det_embed.permute(1, 0).view(1, C, D, 1, 1)

        x = x + 0.1 * det_embed

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.detector_fusion(x)
        x = self.coherence_attention(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Temporal transformer layer
        x = self.temporal_transformer(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
