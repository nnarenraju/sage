"""
Processing state of a gravitational-wave batch (grid, whitening), its
valid transitions, and the channel layout of the network input. An
invalid transition raises ``PipelineError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class Grid(str, Enum):
    """Frequency or time grid of the data."""

    FD_UNIFORM   = "fd_uniform"    # full uniform FD grid, IFFT-able
    TD_UNIFORM   = "td_uniform"    # uniform time domain, multirate-able
    TD_MULTIRATE = "td_multirate"  # decimated time domain, network input


class PipelineError(RuntimeError):
    """Invalid processing step for the current state."""
    pass


@dataclass(frozen=True)
class ProcessingState:
    """
    Immutable processing state of a ``GWBatch``.

    Each transition returns a new state, or raises ``PipelineError``
    before any tensor operation when the step is invalid.

    Parameters
    ----------
    grid : Grid
        Current grid.
    whitened : bool
        Whether the batch has been whitened (divided by the ASD).
    """

    grid:     Grid
    whitened: bool = False

    def is_fd(self) -> bool:
        """
        Whether the data are on the uniform frequency grid.

        Returns
        -------
        bool
        """
        return self.grid == Grid.FD_UNIFORM

    def is_td(self) -> bool:
        """
        Whether the data are on a time grid, uniform or multirate.

        Returns
        -------
        bool
        """
        return self.grid in (Grid.TD_UNIFORM, Grid.TD_MULTIRATE)

    def n_channels(self) -> int:
        """
        Network input channels per detector: 1 for TD, 2 for FD.

        Returns
        -------
        int
        """
        return 1 if self.is_td() else 2

    def after_whiten(self) -> ProcessingState:
        """
        State after whitening.

        Returns
        -------
        ProcessingState
            Same grid, with ``whitened=True``.

        Raises
        ------
        PipelineError
            If the data are already whitened.
        """
        if self.whitened:
            raise PipelineError(
                f"Data is already whitened ({self}). Cannot whiten twice."
            )
        return ProcessingState(self.grid, whitened=True)

    def after_ifft(self) -> ProcessingState:
        """
        State after an inverse FFT.

        Returns
        -------
        ProcessingState
            ``TD_UNIFORM`` grid, whitening unchanged.

        Raises
        ------
        PipelineError
            If the grid is not ``FD_UNIFORM``.
        """
        if self.grid != Grid.FD_UNIFORM:
            raise PipelineError(
                f"Cannot IFFT from {self.grid}: only FD_UNIFORM supports IFFT."
            )
        return ProcessingState(Grid.TD_UNIFORM, self.whitened)

    def after_multirate(self) -> ProcessingState:
        """
        State after multirate sampling.

        Returns
        -------
        ProcessingState
            ``TD_MULTIRATE`` grid, whitening unchanged.

        Raises
        ------
        PipelineError
            If the grid is not ``TD_UNIFORM``.
        """
        if self.grid != Grid.TD_UNIFORM:
            raise PipelineError(
                f"Cannot apply multirate sampling to {self.grid}.\n"
                "Multirate sampling requires TD_UNIFORM (uniform time-domain) data.\n"
                "FD data must first be converted to TD via IFFT."
            )
        return ProcessingState(Grid.TD_MULTIRATE, self.whitened)

    def __str__(self) -> str:
        parts = [self.grid.value]
        if self.whitened:
            parts.append("whitened")
        return f"ProcessingState({', '.join(parts)})"


@dataclass
class GWBatch:
    """
    A batch of gravitational-wave data paired with its processing state.

    Parameters
    ----------
    data : torch.Tensor
        Shape ``(B, D, F)`` complex for FD grids, or ``(B, D, T)`` real
        for TD grids.
    state : ProcessingState
        Grid and whitening state of ``data``.
    """

    data:  torch.Tensor
    state: ProcessingState

    @property
    def n_channels(self) -> int:
        """
        Network input channels per detector: 1 for TD, 2 for FD.

        Returns
        -------
        int
        """
        return self.state.n_channels()

    def to_network_input(self) -> torch.Tensor:
        """
        Real tensor for the network.

        Returns
        -------
        torch.Tensor
            TD data ``(B, D, T)`` as is; FD data ``(B, D, F)`` as all
            real parts, then all imaginary parts, ``(B, 2 D, F)``.
        """
        if self.state.is_td():
            return self.data                                    # (B, D, T)
        return torch.cat([self.data.real, self.data.imag], dim=1)  # (B, 2D, F)

    def __repr__(self) -> str:
        return f"GWBatch(shape={tuple(self.data.shape)}, state={self.state})"


def blocked_detector_channel_indices(d, num_detectors, channels_per_det=2):
    """
    Channels of detector ``d`` in the FD network input layout.

    :meth:`GWBatch.to_network_input` stacks all real parts, then all
    imaginary parts, so part ``k`` of detector ``d`` is at channel
    ``k * num_detectors + d``; channels are not interleaved by detector.

    Parameters
    ----------
    d : int
        Detector index.
    num_detectors : int
        Number of detectors.
    channels_per_det : int
        Parts per detector (default 2, real and imaginary).

    Returns
    -------
    list of int
        Channel index of each part of detector ``d``.
    """
    return [d + k * num_detectors for k in range(channels_per_det)]
