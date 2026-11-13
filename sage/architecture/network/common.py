"""Pieces every network shares.

Norm and backend tables, the input normalisation and output heads run
in full precision, and the frontend tap that hard-noise mining reads.
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
    # Each detector on its own
    "instancenorm": lambda d: nn.InstanceNorm1d(d, affine=True),
    # Detectors jointly, keeping their relative scale
    "groupnorm": lambda d: nn.GroupNorm(1, d),
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


def make_heads(num_estimates: int, uncertainty: bool, non_astrophysical: bool) -> tuple:
    """
    Ranking-statistic, estimate and non-astrophysical heads.

    Parameters
    ----------
    num_estimates : int
        Estimated parameters.
    uncertainty : bool
        Whether each head also gives a raw width.
    non_astrophysical : bool
        Whether the run has the non-astrophysical class.

    Returns
    -------
    ranking : nn.Linear
        Ranking-statistic head: the signal vs noise logit.
    estimates : nn.ModuleList
        Estimate heads, 2 outputs each with uncertainty, else 1.
    auxiliary : nn.Linear or None
        The non-astrophysical vs noise logit; None without the class.
    """
    outputs = 2 if uncertainty else 1
    ranking = nn.Linear(FEATURES, 1)
    estimates = nn.ModuleList([nn.Linear(FEATURES, outputs) for _ in range(num_estimates)])
    auxiliary = nn.Linear(FEATURES, 1) if non_astrophysical else None
    return ranking, estimates, auxiliary


def initialise_heads(ranking: nn.Linear, estimates: nn.ModuleList, auxiliary) -> None:
    """
    Normal weights (std 0.01) and zero biases on every head.

    Parameters
    ----------
    ranking : nn.Linear
        Ranking-statistic head.
    estimates : nn.ModuleList
        Estimate heads.
    auxiliary : nn.Linear or None
        Non-astrophysical head; None without the class.
    """
    for layer in (ranking, *estimates, *([] if auxiliary is None else [auxiliary])):
        nn.init.normal_(layer.weight, 0, 0.01)
        nn.init.zeros_(layer.bias)


def normalise(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Input normalisation in float32 with autocast off.

    The per-channel variance is unstable under bf16 autocast on whitened
    input with a large dynamic range.

    Parameters
    ----------
    norm : nn.Module
        Norm layer from :func:`norm_layer`.
    x : torch.Tensor, shape ``(B, D, L)``
        Input strain.

    Returns
    -------
    torch.Tensor, shape ``(B, D, L)``
        Normalised input, float32.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        return norm(x.float())


def run_heads(features: torch.Tensor, ranking: nn.Linear, estimates: nn.ModuleList, auxiliary) -> tuple:
    """
    Heads in float32 with autocast off.

    A bf16 output would quantise the ranking statistic in steps visible
    at the false-alarm threshold; the body stays in the autocast dtype.

    Parameters
    ----------
    features : torch.Tensor, shape ``(B, 512)``
        Backend features.
    ranking : nn.Linear
        Ranking-statistic head.
    estimates : nn.ModuleList
        Estimate heads, 1 output each, or 2 with uncertainty.
    auxiliary : nn.Linear or None
        Non-astrophysical head; None without the class.

    Returns
    -------
    ranking_statistic : torch.Tensor, shape ``(B, 1)``
        Signal vs noise logit.
    point_estimates : torch.Tensor, shape ``(B, P)`` or ``(B, 2P)``
        Estimates; with uncertainty, all means then all raw widths.
    non_astrophysical_logit : torch.Tensor, shape ``(B, 1)``
        Non-astrophysical vs noise logit; ``(B, 0)`` without the class.
    """
    with torch.autocast(device_type=features.device.type, enabled=False):
        features = features.float()
        ranking_statistic = ranking(features)
        raw = [layer(features) for layer in estimates]
        if raw[0].shape[1] == 2:
            raw = [r[:, :1] for r in raw] + [r[:, 1:] for r in raw]
        logit = features.new_zeros(len(features), 0) if auxiliary is None else auxiliary(features)
        return ranking_statistic, torch.cat(raw, dim=1), logit


class FrontendTap:
    """
    Hard-noise mining tap: the per-detector frontend descriptor.

    With ``return_frontend_embedding`` set, :meth:`tap` stores
    ``frontend_embedding``, shape ``(B, D*K)``: each detector's feature
    map pooled to a ``DESCRIPTOR_GRID`` grid, flattened, L2-normalised.
    """

    return_frontend_embedding = False
    frontend_embedding = None
    DESCRIPTOR_GRID = 8

    def tap(self, cnn_outputs: list) -> None:
        """
        Store the descriptor of ``cnn_outputs`` when the flag is set.

        Parameters
        ----------
        cnn_outputs : list of torch.Tensor
            Per-detector frontend outputs, ``(B, C, L)`` or
            ``(B, C, H, W)``.
        """
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
