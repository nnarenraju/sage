"""Per-detector and shared halves of a trained network.

The per-detector half is computed once per detector and reused across
time slides; the shared half combines the detectors. Both call the
network's own layers, as ``model.forward`` does.
"""

from typing import List

import torch

from sage.architecture.network.common import normalise, run_heads


def frontend(
    model: torch.nn.Module, strain: torch.Tensor, detector: int
) -> torch.Tensor:
    """Frontend features of one detector.

    The input is normalised as a whole before the detector's channel is
    taken, as in ``model.forward``.

    Parameters
    ----------
    model : torch.nn.Module
        Trained network.
    strain : torch.Tensor
        Input batch, shape ``(batch, detectors, samples)``.
    detector : int
        Detector channel.

    Returns
    -------
    torch.Tensor
        Features of that detector.
    """
    normed = normalise(model.norm, strain)
    return model.frontend[detector](normed[:, detector : detector + 1])


def backend(model: torch.nn.Module, features: List[torch.Tensor]) -> tuple:
    """Every network output from per-detector features.

    Parameters
    ----------
    model : torch.nn.Module
        Trained network.
    features : list of torch.Tensor
        Frontend features of each detector, in channel order.

    Returns
    -------
    tuple of torch.Tensor
        Ranking statistic, point estimates and non-astrophysical logit,
        as ``model.forward`` returns them.
    """
    pooled = model.backend(torch.cat(features, dim=1))
    pooled = model.flatten(model.avg_pool_1d(pooled))
    return run_heads(
        pooled,
        model.get_ranking_statistic,
        model.point_estimate_layers,
        model.get_non_astrophysical_logit,
    )


def check_split(model: torch.nn.Module, strain: torch.Tensor) -> None:
    """Check the split is exact and keeps the detectors separate.

    Both checks are exact.

    Parameters
    ----------
    model : torch.nn.Module
        Trained network in evaluation mode.
    strain : torch.Tensor
        Finite test input, shape ``(batch, detectors, samples)``.

    Raises
    ------
    ValueError
        If the network mixes detectors before its backend, the halves
        do not reproduce ``model.forward``, or replacing one detector's
        data changes another detector's features.
    """
    if not model.separable:
        raise ValueError(
            f"{type(model).__name__} mixes detectors before its backend, "
            "so it has no per-detector half"
        )
    n_detectors = strain.shape[1]
    generator = torch.Generator().manual_seed(20260821)
    with torch.no_grad():
        expected = model(strain)
        features = [frontend(model, strain, index) for index in range(n_detectors)]
        actual = backend(model, features)
        for want, got in zip(expected, actual, strict=True):
            if not torch.equal(want, got):
                raise ValueError("the network halves do not reproduce model.forward")

        for changed in range(n_detectors):
            replaced = strain.clone()
            batch, _, samples = strain.shape
            noise = torch.randn(batch, samples, generator=generator, dtype=strain.dtype)
            replaced[:, changed] = noise.to(strain.device)
            for other in range(n_detectors):
                if other == changed:
                    continue
                if not torch.equal(frontend(model, replaced, other), features[other]):
                    raise ValueError(f"detector {other} depends on detector {changed}")
