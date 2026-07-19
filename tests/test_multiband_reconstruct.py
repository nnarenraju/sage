"""
Tests for sage.dsp.multiband_reconstruct — the LAL-equivalent order-1 multiband
reconstruction (linear amplitude + linear continuous phase) in PyTorch.

These are torch-only (no LALSim) so they run in CI.  The faithfulness of the
reconstruction to LAL's built-in multibanding (~1e-8 overlap) is validated
separately against lalsimulation.
"""

import math

import numpy as np
import torch

from sage.dsp.multiband_reconstruct import (
    unwrap,
    continuous_coarse_phase,
    multiband_reconstruct,
    MultibandReconstructor,
)


def _grids():
    """Fine uniform grid and a non-uniform coarse subset of it."""
    fine = torch.arange(20.0, 200.0, 1.0 / 8.0, dtype=torch.float64)
    # coarse: a strictly-increasing subset of fine indices (widening spacing)
    idx = torch.unique(torch.cat([
        torch.arange(0, 400, 3),
        torch.arange(400, 900, 11),
        torch.arange(900, fine.numel(), 29),
        torch.tensor([fine.numel() - 1]),
    ]))
    return fine, idx


def test_unwrap_matches_numpy():
    rng = np.random.default_rng(0)
    raw = np.cumsum(rng.uniform(-2.5, 2.5, 3000))
    wrapped = np.angle(np.exp(1j * raw))
    got = unwrap(torch.tensor(wrapped)).numpy()
    assert np.abs(got - np.unwrap(wrapped)).max() == 0.0


def test_reconstruct_passes_through_coarse_points():
    fine, idx = _grids()
    cfreqs = fine[idx]
    rng = np.random.default_rng(1)
    camp   = torch.tensor(rng.uniform(0.1, 1.0, idx.numel()))
    cphase = torch.tensor(np.cumsum(rng.uniform(0.0, 5.0, idx.numel())))  # continuous
    h = multiband_reconstruct(cfreqs, camp, cphase, fine)
    # At the coarse frequencies the reconstruction must equal the samples exactly.
    h_at_coarse = h[idx]
    expected = torch.polar(camp, cphase)
    assert torch.max((h_at_coarse - expected).abs()).item() < 1e-10


def test_reconstruct_exact_for_linear_amp_and_phase():
    # amplitude and phase both globally linear in f -> piecewise-linear interp is
    # exact, so the reconstruction must equal the full signal to float precision.
    fine, idx = _grids()
    a0, a1 = 0.7, 3.0e-3
    p0, p1 = -1.2, 0.9
    amp_full   = a0 + a1 * fine
    phase_full = p0 + p1 * fine
    h_full = torch.polar(amp_full, phase_full)
    camp   = amp_full[idx]
    cphase = phase_full[idx]
    h_rec  = multiband_reconstruct(fine[idx], camp, cphase, fine)
    assert torch.max((h_rec - h_full).abs()).item() < 1e-10


def test_continuous_coarse_phase_roundtrip():
    fine, idx = _grids()
    # a chirp whose phase winds many cycles between coarse points
    phase = 0.02 * (fine - fine[0]) ** 2
    h_full = torch.polar(torch.ones_like(fine), phase)
    cphase = continuous_coarse_phase(h_full, idx)
    # recovered continuous phase must match the true (continuous) phase at coarse pts
    assert torch.max((cphase - phase[idx]).abs()).item() < 1e-9


def test_module_matches_functional():
    fine, idx = _grids()
    rng = np.random.default_rng(2)
    camp   = torch.tensor(rng.uniform(0.1, 1.0, idx.numel()))
    cphase = torch.tensor(np.cumsum(rng.uniform(0.0, 5.0, idx.numel())))
    mod = MultibandReconstructor(fine[idx], fine)
    h_mod = mod(camp, cphase)
    h_fn  = multiband_reconstruct(fine[idx], camp, cphase, fine)
    assert torch.max((h_mod - h_fn).abs()).item() == 0.0
