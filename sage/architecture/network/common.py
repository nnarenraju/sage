"""Pieces every network shares.

Norm and backend tables, the input normalisation and output heads run in full
precision, and the frontend tap that hard-noise mining reads.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..backend.resnet2d_cbam import (
    resnet18_cbam,
    resnet34_cbam,
    resnet50_cbam,
    resnet101_cbam,
    resnet152_cbam,
)

NORMS = {
    "instancenorm": lambda d: nn.InstanceNorm1d(d, affine=True),  # each detector on its own
    "groupnorm": lambda d: nn.GroupNorm(1, d),  # detectors jointly, keeping their relative scale
    "batchnorm": nn.BatchNorm1d,
}
RESNETS = {18: resnet18_cbam, 34: resnet34_cbam, 50: resnet50_cbam, 101: resnet101_cbam, 152: resnet152_cbam}
FEATURES = 512


def norm_layer(norm_type: str, num_detectors: int) -> nn.Module:
    """
    Input normalisation over the detector channels.

    Parameters
    ----------
    norm_type : str
        ``instancenorm``, ``groupnorm`` or ``batchnorm``.
    num_detectors : int
        Detector channels.

    Returns
    -------
    nn.Module
        Norm layer.

    Raises
    ------
    ValueError
        If the norm type is unknown.
    """
    if norm_type not in NORMS:
        raise ValueError(f"norm_type: unknown {norm_type!r}; choose from {sorted(NORMS)}")
    return NORMS[norm_type](num_detectors)


def resnet_backend(size: int, **kwargs) -> nn.Module:
    """
    2D ResNet-CBAM backend of the given depth.

    Parameters
    ----------
    size : int
        Depth: 18, 34, 50, 101 or 152.
    **kwargs
        Passed to the ResNet factory.

    Returns
    -------
    nn.Module
        Backend.

    Raises
    ------
    ValueError
        If the depth is not one of those.
    """
    if size not in RESNETS:
        raise ValueError(f"backend_resnet_size: {size} is not one of {sorted(RESNETS)}")
    return RESNETS[size](pretrained=False, **kwargs)


def make_heads(num_estimates: int, uncertainty: bool) -> tuple:
    """
    Ranking-statistic head and one head per estimated parameter.

    Parameters
    ----------
    num_estimates : int
        Estimated parameters.
    uncertainty : bool
        Whether each head also gives a raw width.

    Returns
    -------
    ranking : nn.Linear
        Ranking-statistic head.
    estimates : nn.ModuleList
        Estimate heads, 2 outputs each with uncertainty, else 1.
    """
    outputs = 2 if uncertainty else 1
    ranking = nn.Linear(FEATURES, 1)
    estimates = nn.ModuleList([nn.Linear(FEATURES, outputs) for _ in range(num_estimates)])
    return ranking, estimates


def initialise_heads(ranking: nn.Linear, estimates: nn.ModuleList) -> None:
    """Small normal weights and zero biases on every head, ranking head first."""
    for layer in (ranking, *estimates):
        nn.init.normal_(layer.weight, 0, 0.01)
        nn.init.zeros_(layer.bias)


def normalise(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Input normalisation in float32 with autocast off.

    InstanceNorm's per-channel variance is unstable under bf16 autocast on
    whitened input with large dynamic range (NaN, then poisoned weights).
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        return norm(x.float())


def run_heads(features: torch.Tensor, ranking: nn.Linear, estimates: nn.ModuleList) -> tuple:
    """
    Heads in float32 with autocast off.

    A bf16 output quantises the ranking statistic in steps visible at the
    false-alarm threshold (a comb in the background); the body stays in the
    autocast dtype.

    Returns
    -------
    ranking_statistic : torch.Tensor, shape ``(B, 1)``
    point_estimates : torch.Tensor, shape ``(B, P)`` or ``(B, 2P)``
        Estimates; with uncertainty, all means then all raw widths.
    """
    with torch.autocast(device_type=features.device.type, enabled=False):
        features = features.float()
        ranking_statistic = ranking(features)
        raw = [layer(features) for layer in estimates]
        if raw[0].shape[1] == 2:
            raw = [r[:, :1] for r in raw] + [r[:, 1:] for r in raw]
        return ranking_statistic, torch.cat(raw, dim=1)


class FrontendTap:
    """
    Hard-noise mining tap: the per-detector frontend descriptor.

    A model calls :meth:`tap` on its per-detector frontend outputs. With
    ``return_frontend_embedding`` set (the mining pass, on the eager model) it
    stores ``frontend_embedding``, shape ``(B, D*K)``: each detector's feature
    map pooled to a ``DESCRIPTOR_GRID`` grid, flattened and L2-normalised.
    Compiled training leaves the flag off, so the branch is pruned; no hook is
    used, as ``fullgraph=True`` forbids them.
    """

    return_frontend_embedding = False
    frontend_embedding = None
    DESCRIPTOR_GRID = 8

    def tap(self, cnn_outputs: list) -> None:
        """Store the descriptor of ``cnn_outputs`` when the flag is set."""
        if not self.return_frontend_embedding:
            return
        g = self.DESCRIPTOR_GRID
        pooled = []
        for o in cnn_outputs:
            o = o.float()
            if o.dim() == 4:                       # (B, C, H, W)
                o = F.adaptive_avg_pool2d(o, (g, g))
            elif o.dim() == 3:                     # (B, C, L)
                o = F.adaptive_avg_pool1d(o, g)
            pooled.append(F.normalize(o.flatten(1), dim=-1))
        self.frontend_embedding = torch.stack(pooled, dim=1).flatten(1)
