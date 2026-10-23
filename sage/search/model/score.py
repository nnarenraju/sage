"""Run a trained network on analysis windows, in float32.

The forward path is the one training uses: rfft, fiducial whitening,
multirate sampling, then the network.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from sage.core.pipeline import GWBatch, Grid, ProcessingState
from sage.factory.contract import forward_batch
from sage.search.model.split import backend, frontend


def window_spectra(
    strain: np.ndarray, window_samples: int, stride_samples: int, device: str
) -> torch.Tensor:
    """rfft of every window in a contiguous stretch of strain.

    Parameters
    ----------
    strain : numpy.ndarray
        Float32 ``(detectors, samples)`` from ``window_batches``.
    window_samples : int
        Window length in samples.
    stride_samples : int
        Step between window starts in samples.
    device : str
        Device to compute on.

    Returns
    -------
    torch.Tensor
        Complex ``(windows, detectors, frequencies)``, normalised with
        ``norm="forward"``.
    """
    block = torch.from_numpy(strain).to(device=device, non_blocking=True)
    windows = block.unfold(-1, window_samples, stride_samples).permute(1, 0, 2)
    return torch.fft.rfft(windows, dim=-1, norm="forward")


def score(
    network: Dict[str, object], spectra: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Ranking statistic and raw point estimates for frequency-domain windows.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    spectra : torch.Tensor
        Complex ``(windows, detectors, frequencies)``, the rfft of each
        window with ``norm="forward"``.

    Returns
    -------
    statistic : numpy.ndarray
        Ranking statistic per window.
    estimates : numpy.ndarray
        Raw point-estimate head output per window.
    """
    with torch.inference_mode():
        output = forward_batch(
            spectra,
            network["model"],
            network["processor"],
            autocast=False,
            device_type=spectra.device.type,
        )
        statistic = output[:, 0].float().cpu().numpy()
        estimates = output[:, 1:].float().cpu().numpy()
    return statistic, estimates


def score_strain(
    network: Dict[str, object],
    strain: np.ndarray,
    stride_samples: int,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Score every window in a contiguous stretch of strain.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    strain : numpy.ndarray
        Float32 ``(detectors, samples)`` from ``window_batches``.
    stride_samples : int
        Step between window starts in samples.
    device : str
        Device to score on.

    Returns
    -------
    statistic, estimates : numpy.ndarray
        Output of ``score`` for each window.
    """
    spectra = window_spectra(strain, network["window_samples"], stride_samples, device)
    return score(network, spectra)


def frontend_features(
    network: Dict[str, object], spectra: torch.Tensor
) -> List[torch.Tensor]:
    """Per-detector frontend features of frequency-domain windows.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    spectra : torch.Tensor
        Complex ``(windows, detectors, frequencies)``.

    Returns
    -------
    list of torch.Tensor
        Features of each detector, in channel order.
    """
    batch = GWBatch(spectra, state=ProcessingState(Grid.FD_UNIFORM))
    with torch.inference_mode():
        prepared = network["processor"](batch).to_network_input()
        return [
            frontend(network["model"], prepared, index)
            for index in range(prepared.shape[1])
        ]


def backend_scores(
    network: Dict[str, object], features: List[torch.Tensor]
) -> Tuple[np.ndarray, np.ndarray]:
    """Ranking statistic and raw point estimates from frontend features.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    features : list of torch.Tensor
        Frontend features of each detector, in channel order.

    Returns
    -------
    statistic, estimates : numpy.ndarray
        Ranking statistic and raw point estimates per window.
    """
    with torch.inference_mode():
        statistic, estimates = backend(network["model"], features)
        statistic = statistic.reshape(statistic.shape[0]).float().cpu().numpy()
        estimates = estimates.float().cpu().numpy()
    return statistic, estimates
