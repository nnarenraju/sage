#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : contract.py
Description   : The trained model's input and forward contract, in one place.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

How strain becomes a network output is a contract between training and everything that
runs a trained network afterwards: validation, testing, benchmarking and the search. If a
consumer re-derives it, the two drift and the network is quietly fed something it was not
trained on -- a failure that produces plausible numbers rather than an error.

So the contract lives here once and is called, not copied.

Two pieces:

* :func:`make_processor` builds the preprocessing chain. It was duplicated verbatim in
  eight training drivers.
* :func:`forward_batch` runs the input through the processor and the model, preserving the
  boundary that matters: the preprocessor runs OUTSIDE autocast, because it consumes
  complex frequency-domain input which must not be cast, and only the model runs inside.

The input is already frequency-domain: ``torch.fft.rfft(x, dim=-1, norm="forward")`` over
the padded window. The ``norm="forward"`` is not incidental -- it divides by N, and the
fiducial whitening buffer is scaled to match. Feeding an unnormalised transform produces
an output that looks reasonable and is wrong by a factor of N.

The ranking statistic is column 0 of the returned tensor; the remaining columns are the
raw point-estimate means followed by their raw sigmas.
"""

from contextlib import nullcontext
from typing import Optional, Sequence

import torch

from sage.core.graph import Preprocessor
from sage.core.pipeline import GWBatch, Grid, ProcessingState
from sage.dsp.multirate_sampling import DyadicPyramidBinning, MultirateSampler
from sage.dsp.whiten import FiducialWhitening


def make_processor(bounds: Sequence) -> Preprocessor:
    """
    Build the preprocessing chain a Sage network is trained and evaluated with.

    Whitening by the fiducial spectra, then multirate sampling on a dyadic pyramid over
    the parameter bounds.

    Parameters
    ----------
    bounds : sequence
        Parameter bounds from the signal sampler, which fix the dyadic binning. They come
        from ``read_from_config(...).bounds`` on the training path and, for a search, from
        the configuration the checkpoint was trained under.

    Returns
    -------
    Preprocessor
        The chain, ready to be applied to a :class:`GWBatch`.

    Notes
    -----
    Whitening is the network's input normalisation, so the fiducial spectra are part of
    the contract too: they must be the ones the network was trained with, not the ones
    belonging to the run being analysed. The two differ whenever a network trained on one
    observing run is used to search another.
    """
    whitener = FiducialWhitening()
    mrsampler = MultirateSampler(binning_method=DyadicPyramidBinning(bounds))
    return Preprocessor([whitener, mrsampler])


def forward_batch(
    x: torch.Tensor,
    model: torch.nn.Module,
    processor: Preprocessor,
    *,
    state: Optional[ProcessingState] = None,
    selector=None,
    freqs=None,
    coarse_indices=None,
    amp_dtype: torch.dtype = torch.bfloat16,
    autocast: bool = True,
    device_type: str = "cuda",
) -> torch.Tensor:
    """
    Run a frequency-domain strain batch through the preprocessor and the network.

    Parameters
    ----------
    x : Tensor
        Complex ``(B, D, F)``, already ``rfft(..., norm="forward")`` of the padded window.
    model : Module
        The trained network. Returns ``(ranking_statistic, point_estimates)``.
    processor : Preprocessor
        Built by :func:`make_processor`.
    state : ProcessingState, optional
        Grid the input is on. Defaults to the uniform frequency-domain grid.
    selector : optional
        Multiband selector, applied before the batch is formed when present.
    amp_dtype : torch.dtype
        Autocast dtype for the model.
    autocast : bool
        Whether the model runs under autocast. Taken from the training configuration; a
        search must use the same setting the network was trained and validated with.
    device_type : str
        Autocast device type. Left as ``"cuda"`` by default rather than derived from the
        input, because deriving it would silently enable CPU autocast on a CPU tensor and
        change the numerics of any CPU-side comparison.

    Returns
    -------
    Tensor
        ``(B, C)`` in float32: column 0 the ranking statistic, then the raw point-estimate
        means and their raw sigmas.

    Notes
    -----
    The output is cast to float32 here, and the network's own heads already run in float32
    with autocast disabled. That is deliberate: a reduced-precision logit is quantised in
    steps coarse enough to be visible right where a false-alarm threshold sits, which shows
    up as a comb in the background distribution.
    """
    if selector is not None:
        x = selector(x)
    if state is None:
        state = ProcessingState(Grid.FD_UNIFORM)
    batch = GWBatch(x, state=state, freqs=freqs, coarse_indices=coarse_indices)
    # Outside autocast: the preprocessor consumes complex frequency-domain input, which
    # must not be cast to a reduced dtype.
    batch = processor(batch)
    context = (
        torch.autocast(device_type=device_type, dtype=amp_dtype)
        if autocast
        else nullcontext()
    )
    with context:
        out = model(batch.to_network_input())
    return torch.cat([*out], dim=1).float()
