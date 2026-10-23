"""Per-detector and shared halves of a trained network.

The per-detector half is computed once per detector and reused across
time slides; the shared half combines the detectors.
"""

from typing import List, Tuple

import torch


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
    with torch.autocast(device_type=strain.device.type, enabled=False):
        normed = model.norm(strain.float())
    return model.frontend[detector](normed[:, detector : detector + 1])


def backend(
    model: torch.nn.Module, features: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ranking statistic and point estimates from per-detector features.

    The output heads run in full precision, as in ``model.forward``.

    Parameters
    ----------
    model : torch.nn.Module
        Trained network.
    features : list of torch.Tensor
        Frontend features of each detector, in channel order.

    Returns
    -------
    statistic : torch.Tensor
        Ranking statistic per window.
    estimates : torch.Tensor
        Point-estimate means followed by their widths, per window.
    """
    pooled = model.backend(torch.cat(features, dim=1))
    pooled = model.flatten(model.avg_pool_1d(pooled))
    with torch.autocast(device_type=pooled.device.type, enabled=False):
        pooled = pooled.float()
        statistic = model.get_ranking_statistic(pooled)
        outputs = [layer(pooled) for layer in model.point_estimate_layers]
        means = torch.cat([output[:, :1] for output in outputs], dim=1)
        widths = torch.cat([output[:, 1:] for output in outputs], dim=1)
    return statistic, torch.cat([means, widths], dim=1)


def check_split(model: torch.nn.Module, strain: torch.Tensor) -> None:
    """Check the halves reproduce the network and detectors stay separate.

    Both checks are exact. Run once per checkpoint before reusing
    per-detector features across time slides.

    Parameters
    ----------
    model : torch.nn.Module
        Trained network in evaluation mode.
    strain : torch.Tensor
        Finite test input, shape ``(batch, detectors, samples)``.

    Raises
    ------
    ValueError
        If the halves do not reproduce ``model.forward``, or replacing
        one detector's data changes another detector's features.
    """
    n_detectors = strain.shape[1]
    generator = torch.Generator().manual_seed(20260821)
    with torch.no_grad():
        expected = model(strain)
        features = [frontend(model, strain, index) for index in range(n_detectors)]
        actual = backend(model, features)
        for want, got in zip(expected, actual):
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
