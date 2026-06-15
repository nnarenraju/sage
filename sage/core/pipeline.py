"""
Sage pipeline state tracking.

Grid, ProcessingState, PipelineError, and GWBatch form a lightweight
state machine that tracks how a batch of gravitational-wave data has been
processed and raises informative errors when an invalid operation is attempted.

The three supported pipeline paths are:

    FD_UNIFORM  →  FiducialWhitening  →  TD_UNIFORM  →  MultirateSampler  →  TD_MULTIRATE
                   (whiten + IFFT)           (decimate)

    FD_COARSE   →  FiducialWhitening  →  FD_COARSE (whitened)
                   (whiten, no IFFT —
                    non-uniform grid)

    FD_UNIFORM  →  FiducialWhitening  →  TD_UNIFORM
                   (no multirate)

FD_COARSE data cannot be IFFTed (the grid is non-uniform) and cannot be
multirate-sampled (which requires uniform time-domain data).  Attempting
either raises PipelineError immediately, before any computation begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


# ── Grid type ─────────────────────────────────────────────────────────────────


class Grid(str, Enum):
    """The frequency/time grid type of the current data representation."""

    FD_UNIFORM   = "fd_uniform"    # full uniform FD grid — IFFT-able
    FD_COARSE    = "fd_coarse"     # worst-case non-uniform FD grid — stays FD
    TD_UNIFORM   = "td_uniform"    # uniform time domain — multirate-able
    TD_MULTIRATE = "td_multirate"  # decimated time domain — network input


# ── Error type ────────────────────────────────────────────────────────────────


class PipelineError(RuntimeError):
    """Raised when an invalid processing step is attempted given the current state."""
    pass


# ── Immutable processing state ────────────────────────────────────────────────


@dataclass(frozen=True)
class ProcessingState:
    """
    Immutable descriptor of a GWBatch's processing history.

    Each transition method returns a **new** ProcessingState or raises
    PipelineError when the requested operation is incompatible with the
    current state.  Validation fires at call time — before any tensor
    operations — so invalid pipelines fail fast.

    Parameters
    ----------
    grid : Grid
        Current frequency/time grid type.
    whitened : bool
        Whether the batch has been whitened (divided by the ASD).
    """

    grid:     Grid
    whitened: bool = False

    # ── Compatibility queries ──────────────────────────────────────────────

    def is_fd(self) -> bool:
        return self.grid in (Grid.FD_UNIFORM, Grid.FD_COARSE)

    def is_td(self) -> bool:
        return self.grid in (Grid.TD_UNIFORM, Grid.TD_MULTIRATE)

    def n_channels(self) -> int:
        """Number of network input channels: 1 for TD (real), 2 for FD (real+imag)."""
        return 1 if self.is_td() else 2

    # ── State transitions ──────────────────────────────────────────────────

    def after_whiten(self) -> ProcessingState:
        if self.whitened:
            raise PipelineError(
                f"Data is already whitened ({self}). Cannot whiten twice."
            )
        return ProcessingState(self.grid, whitened=True)

    def after_ifft(self) -> ProcessingState:
        if self.grid == Grid.FD_COARSE:
            raise PipelineError(
                "Cannot convert FD_COARSE to time domain.\n"
                "The worst-case multibanding grid is non-uniform — IFFT requires "
                "a uniform frequency grid.\n"
                "Options:\n"
                "  • Keep data in FD and use a 2-channel network (real + imag).\n"
                "  • Switch to multiband_mode='none' or 'per_signal' to get "
                "a uniform FD grid that can be IFFTed."
            )
        if self.grid != Grid.FD_UNIFORM:
            raise PipelineError(
                f"Cannot IFFT from {self.grid}: only FD_UNIFORM supports IFFT."
            )
        return ProcessingState(Grid.TD_UNIFORM, self.whitened)

    def after_multirate(self) -> ProcessingState:
        if self.grid != Grid.TD_UNIFORM:
            raise PipelineError(
                f"Cannot apply multirate sampling to {self.grid}.\n"
                "Multirate sampling requires TD_UNIFORM (uniform time-domain) data.\n"
                "FD data must first be converted to TD via IFFT — which is only "
                "possible from FD_UNIFORM grids (i.e. not worst-case multibanding)."
            )
        return ProcessingState(Grid.TD_MULTIRATE, self.whitened)

    def __str__(self) -> str:
        parts = [self.grid.value]
        if self.whitened:
            parts.append("whitened")
        return f"ProcessingState({', '.join(parts)})"


# ── Batch wrapper ─────────────────────────────────────────────────────────────


@dataclass
class GWBatch:
    """
    A batch of gravitational-wave data paired with its processing state.

    Parameters
    ----------
    data : torch.Tensor
        Shape ``(B, D, F)`` complex for FD grids, or ``(B, D, T)`` float32
        for TD grids.
    state : ProcessingState
        Current processing state — tracks the grid type and whether the
        batch has been whitened.
    freqs : torch.Tensor or None
        Frequency array in Hz, shape ``(F,)``.  Non-None for FD grids.
        None for TD grids.
    coarse_indices : torch.Tensor or None
        Integer indices into the full uniform FD array (0-to-Nyquist) that
        correspond to the coarse grid points.  Non-None only when
        ``state.grid == Grid.FD_COARSE`` (worst-case multibanding).
        Allows whitening and other FD operations to select the correct
        coefficients from full-resolution buffers without recomputing
        frequencies.
    """

    data:           torch.Tensor
    state:          ProcessingState
    freqs:          torch.Tensor | None = None
    coarse_indices: torch.Tensor | None = None

    @property
    def n_channels(self) -> int:
        """Network input channels: 1 (TD, real) or 2 (FD, real + imag)."""
        return self.state.n_channels()

    def to_network_input(self) -> torch.Tensor:
        """
        Convert to a real float tensor ready for the neural network.

        Returns
        -------
        torch.Tensor
            TD data ``(B, D, T)`` — returned as-is (already float32).
            FD data ``(B, D, F)`` complex — real and imaginary parts of
            every detector are stacked into ``(B, 2·D, F)`` float32.
        """
        if self.state.is_td():
            return self.data                                    # (B, D, T)
        return torch.cat([self.data.real, self.data.imag], dim=1)  # (B, 2D, F)

    def __repr__(self) -> str:
        return f"GWBatch(shape={tuple(self.data.shape)}, state={self.state})"
