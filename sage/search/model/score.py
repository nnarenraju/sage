"""Run a trained network on analysis windows, in float32.

The forward path is the one training uses (``forward_batch`` without
autocast): rfft, fiducial whitening, multirate sampling, then the network.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from sage.core.pipeline import GWBatch
from sage.search.model.split import backend, check_split, frontend


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


def network_input(network: Dict[str, object], spectra: torch.Tensor) -> torch.Tensor:
    """Preprocessed network input of frequency-domain windows.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    spectra : torch.Tensor
        Complex ``(windows, detectors, frequencies)``, the rfft of each
        window with ``norm="forward"``.

    Returns
    -------
    torch.Tensor
        Real ``(windows, detectors, samples)`` network input.
    """
    batch = GWBatch(spectra, state=network["processor"].input_state)
    with torch.inference_mode():
        return network["processor"](batch).to_network_input()


def network_outputs(network: Dict[str, object], spectra: torch.Tensor) -> np.ndarray:
    """Every output of the network for frequency-domain windows.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    spectra : torch.Tensor
        Complex ``(windows, detectors, frequencies)``, the rfft of each
        window with ``norm="forward"``.

    Returns
    -------
    numpy.ndarray
        Float32 ``(windows, columns)``, columns as ``network["columns"]``:
        ranking statistic, the non-astrophysical logit if the network
        has that class, the raw point-estimate means, then raw widths.
    """
    prepared = network_input(network, spectra)
    with torch.inference_mode():
        statistic, estimates, logit = network["model"](prepared)
        outputs = torch.cat([statistic, logit, estimates], dim=1)
        return outputs.float().cpu().numpy()


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
    outputs = network_outputs(network, spectra)
    n_estimates = 2 * len(network["targets"])
    return outputs[:, 0], outputs[:, -n_estimates:]


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
    network: Dict[str, object], spectra: torch.Tensor, channels: List[int]
) -> List[torch.Tensor]:
    """Per-detector frontend features of frequency-domain windows.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    spectra : torch.Tensor
        Complex ``(windows, detectors, frequencies)``.
    channels : list of int
        Detector channels whose features are computed.

    Returns
    -------
    list of torch.Tensor
        Features of each channel asked for, in that order.
    """
    prepared = network_input(network, spectra)
    with torch.inference_mode():
        return [frontend(network["model"], prepared, channel) for channel in channels]


def backend_outputs(
    network: Dict[str, object], features: List[torch.Tensor]
) -> np.ndarray:
    """Every network output from per-detector frontend features.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    features : list of torch.Tensor
        Frontend features of each detector, in channel order.

    Returns
    -------
    numpy.ndarray
        Float32 ``(windows, columns)``, columns as ``network_outputs``.
    """
    with torch.inference_mode():
        statistic, estimates, logit = backend(network["model"], features)
        outputs = torch.cat([statistic, logit, estimates], dim=1)
        return outputs.float().cpu().numpy()


def check_separable(network: Dict[str, object], spectra: torch.Tensor) -> None:
    """Check the split network is exact on real windows.

    The halves must reproduce the network, and each detector's features
    must not depend on another detector's data, preprocessing included.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    spectra : torch.Tensor
        Complex ``(windows, detectors, frequencies)`` of real windows.

    Raises
    ------
    ValueError
        If either check fails.
    """
    with torch.inference_mode():
        check_split(network["model"], network_input(network, spectra))

    channels = list(range(spectra.shape[1]))
    features = frontend_features(network, spectra, channels)
    for channel in channels:
        others = [other for other in channels if other != channel]
        replaced = spectra.clone()
        replaced[:, others] = 2 * spectra[:, others]
        alone = frontend_features(network, replaced, [channel])[0]
        if not torch.equal(alone, features[channel]):
            raise ValueError(
                f"{network['detectors'][channel]} features depend on another "
                "detector's data"
            )
