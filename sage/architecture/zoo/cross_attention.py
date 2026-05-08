#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : cross_attention.py
Description     : Short description of the file

Created on 2026-03-19 23:36:04

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
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
import torch
import torch.nn as nn


class CrossAttention2D(nn.Module):
    """
    Full cross-attention between two 2D feature maps.

    Projects each detector's 2D feature map ``(B, 1, H, W)`` to
    ``embed_dim`` channels, flattens the spatial grid into a sequence, and
    lets each detector attend to the other using
    :class:`torch.nn.MultiheadAttention`.  Both detectors receive mutually
    informed representations: H1 attends to L1 and vice versa.

    Memory scales as O(H²W²) — use :class:`AxialCrossAttention2D` for
    large feature maps.

    Parameters
    ----------
    embed_dim : int
        Projection and attention dimension (default 128).
    num_heads : int
        Number of parallel attention heads (default 4).
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
        Apply cross-attention between two detector feature maps.

        Parameters
        ----------
        f1 : torch.Tensor, shape ``(B, 1, H, W)``
            Feature map for detector 1 (e.g. H1).
        f2 : torch.Tensor, shape ``(B, 1, H, W)``
            Feature map for detector 2 (e.g. L1).

        Returns
        -------
        f1_attn : torch.Tensor, shape ``(B, embed_dim, H, W)``
        f2_attn : torch.Tensor, shape ``(B, embed_dim, H, W)``
        """
        # f1, f2: (B, C=1, H, W)
        B, C, H, W = f1.shape

        # Project to embed_dim
        f1 = self.proj1(f1)  # (B, embed_dim, H, W)
        f2 = self.proj2(f2)

        # Flatten spatial dims
        f1_flat = f1.view(B, self.embed_dim, H * W).permute(
            0, 2, 1
        )  # (B, seq_len=H*W, embed_dim)
        f2_flat = f2.view(B, self.embed_dim, H * W).permute(0, 2, 1)

        # Cross attention: f1 attends to f2 and vice versa
        f1_attn, _ = self.attn(f1_flat, f2_flat, f2_flat)
        f2_attn, _ = self.attn(f2_flat, f1_flat, f1_flat)

        # Reshape back to (B, embed_dim, H, W)
        f1_attn = f1_attn.permute(0, 2, 1).view(B, self.embed_dim, H, W)
        f2_attn = f2_attn.permute(0, 2, 1).view(B, self.embed_dim, H, W)

        return f1_attn, f2_attn


class AxialCrossAttention2D(nn.Module):
    """
    Axial cross-attention for 2-detector 2D features.
    Attends along features (H) and time (W) separately to save memory.
    Input: f1, f2 -> (B, 1, H, W)
    Output: f1_attn, f2_attn -> same shape
    """

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Project CNN output (C=1) to embed_dim
        self.proj1 = nn.Conv2d(1, embed_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(1, embed_dim, kernel_size=1)

        # Multihead attention along sequence dimension
        self.attn_H = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_W = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, f1, f2):
        B, C, H, W = f1.shape  # C=1

        # Project to embed_dim
        f1 = self.proj1(f1)  # (B, embed_dim, H, W)
        f2 = self.proj2(f2)

        # -----------------------------
        # Attention along features (H)
        # -----------------------------
        # Treat W as batch, H as seq_len
        f1_H = f1.permute(0, 3, 2, 1).reshape(
            B * W, H, self.embed_dim
        )  # (B*W, H, embed_dim)
        f2_H = f2.permute(0, 3, 2, 1).reshape(B * W, H, self.embed_dim)

        f1_H_attn, _ = self.attn_H(f1_H, f2_H, f2_H)
        f2_H_attn, _ = self.attn_H(f2_H, f1_H, f1_H)

        # Reshape back
        f1_H_attn = f1_H_attn.view(B, W, H, self.embed_dim).permute(
            0, 3, 2, 1
        )  # (B, embed_dim, H, W)
        f2_H_attn = f2_H_attn.view(B, W, H, self.embed_dim).permute(0, 3, 2, 1)

        # -----------------------------
        # Attention along time (W)
        # -----------------------------
        # Treat H as batch, W as seq_len
        f1_W = f1.permute(0, 2, 3, 1).reshape(
            B * H, W, self.embed_dim
        )  # (B*H, W, embed_dim)
        f2_W = f2.permute(0, 2, 3, 1).reshape(B * H, W, self.embed_dim)

        f1_W_attn, _ = self.attn_W(f1_W, f2_W, f2_W)
        f2_W_attn, _ = self.attn_W(f2_W, f1_W, f1_W)

        # Reshape back
        f1_W_attn = f1_W_attn.view(B, H, W, self.embed_dim).permute(
            0, 3, 1, 2
        )  # (B, embed_dim, H, W)
        f2_W_attn = f2_W_attn.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

        # Combine feature & time attention
        f1_attn = f1_H_attn + f1_W_attn
        f2_attn = f2_H_attn + f2_W_attn

        return f1_attn, f2_attn
