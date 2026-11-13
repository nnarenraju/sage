#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-attention between two detectors' 2D frontend feature maps, full
or axial.

Filename        : cross_attention.py
Description     : Short description of the file

Created on 2026-03-19 23:36:04

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import torch
import torch.nn as nn


class CrossAttention2D(nn.Module):
    """
    Full cross-attention between two 2D feature maps.

    Each map is projected to ``embed_dim`` channels and flattened to a
    sequence; each detector attends to the other with one shared
    :class:`torch.nn.MultiheadAttention`. Memory scales as (H W)^2.

    Parameters
    ----------
    embed_dim : int
        Projection and attention dimension.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Project the CNN outputs (C=1) to embed_dim
        self.proj1 = nn.Conv2d(1, embed_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(1, embed_dim, kernel_size=1)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, f1, f2):
        """
        Cross-attend two detector feature maps.

        Parameters
        ----------
        f1 : torch.Tensor, shape ``(B, 1, H, W)``
            Feature map of the first detector.
        f2 : torch.Tensor, shape ``(B, 1, H, W)``
            Feature map of the second detector.

        Returns
        -------
        f1_attn : torch.Tensor, shape ``(B, embed_dim, H, W)``
            First detector attending to the second.
        f2_attn : torch.Tensor, shape ``(B, embed_dim, H, W)``
            Second detector attending to the first.
        """
        B, C, H, W = f1.shape

        f1 = self.proj1(f1)  # (B, embed_dim, H, W)
        f2 = self.proj2(f2)

        f1_flat = f1.view(B, self.embed_dim, H * W).permute(
            0, 2, 1
        )  # (B, seq_len=H*W, embed_dim)
        f2_flat = f2.view(B, self.embed_dim, H * W).permute(0, 2, 1)

        f1_attn, _ = self.attn(f1_flat, f2_flat, f2_flat)
        f2_attn, _ = self.attn(f2_flat, f1_flat, f1_flat)

        f1_attn = f1_attn.permute(0, 2, 1).view(B, self.embed_dim, H, W)
        f2_attn = f2_attn.permute(0, 2, 1).view(B, self.embed_dim, H, W)

        return f1_attn, f2_attn


class AxialCrossAttention2D(nn.Module):
    """
    Axial cross-attention between two 2D feature maps.

    Attends along the feature axis (H) and the time axis (W) separately,
    each with its own attention, and sums the two.

    Parameters
    ----------
    embed_dim : int
        Projection and attention dimension.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Project the frontend output (C=1) to embed_dim
        self.proj1 = nn.Conv2d(1, embed_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(1, embed_dim, kernel_size=1)

        self.attn_H = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_W = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, f1, f2):
        """
        Cross-attend two detector feature maps along H and along W.

        Parameters
        ----------
        f1 : torch.Tensor, shape ``(B, 1, H, W)``
            Feature map of the first detector.
        f2 : torch.Tensor, shape ``(B, 1, H, W)``
            Feature map of the second detector.

        Returns
        -------
        f1_attn : torch.Tensor, shape ``(B, embed_dim, H, W)``
            First detector attending to the second.
        f2_attn : torch.Tensor, shape ``(B, embed_dim, H, W)``
            Second detector attending to the first.
        """
        B, C, H, W = f1.shape  # C=1

        f1 = self.proj1(f1)  # (B, embed_dim, H, W)
        f2 = self.proj2(f2)

        # Attention along H, with W folded into the batch
        f1_H = f1.permute(0, 3, 2, 1).reshape(
            B * W, H, self.embed_dim
        )  # (B*W, H, embed_dim)
        f2_H = f2.permute(0, 3, 2, 1).reshape(B * W, H, self.embed_dim)

        f1_H_attn, _ = self.attn_H(f1_H, f2_H, f2_H)
        f2_H_attn, _ = self.attn_H(f2_H, f1_H, f1_H)

        f1_H_attn = f1_H_attn.view(B, W, H, self.embed_dim).permute(
            0, 3, 2, 1
        )  # (B, embed_dim, H, W)
        f2_H_attn = f2_H_attn.view(B, W, H, self.embed_dim).permute(0, 3, 2, 1)

        # Attention along W, with H folded into the batch
        f1_W = f1.permute(0, 2, 3, 1).reshape(
            B * H, W, self.embed_dim
        )  # (B*H, W, embed_dim)
        f2_W = f2.permute(0, 2, 3, 1).reshape(B * H, W, self.embed_dim)

        f1_W_attn, _ = self.attn_W(f1_W, f2_W, f2_W)
        f2_W_attn, _ = self.attn_W(f2_W, f1_W, f1_W)

        f1_W_attn = f1_W_attn.view(B, H, W, self.embed_dim).permute(
            0, 3, 1, 2
        )  # (B, embed_dim, H, W)
        f2_W_attn = f2_W_attn.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

        # Sum of the H and W attention
        f1_attn = f1_H_attn + f1_W_attn
        f2_attn = f2_H_attn + f2_W_attn

        return f1_attn, f2_attn
