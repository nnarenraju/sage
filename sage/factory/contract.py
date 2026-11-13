#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The forward path of a trained network: preprocessing outside
autocast, then the model inside it.

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
"""

from contextlib import nullcontext
from typing import Optional

import torch

from sage.core.graph import Preprocessor
from sage.core.pipeline import GWBatch, ProcessingState


def forward_batch(
    x: torch.Tensor,
    model: torch.nn.Module,
    processor: Preprocessor,
    *,
    state: Optional[ProcessingState] = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    autocast: bool = True,
    device_type: str = "cuda",
) -> torch.Tensor:
    """
    Run a frequency-domain strain batch through preprocessing and model.

    The preprocessor runs outside autocast and only the model inside;
    the output is cast to float32.

    Parameters
    ----------
    x : torch.Tensor
        Complex ``(B, D, F)``: ``torch.fft.rfft(..., norm="forward")``
        of the padded window; the fiducial whitening is scaled to it.
    model : torch.nn.Module
        The trained network, returning ``(ranking_statistic,
        point_estimates, non_astrophysical_logit)``.
    processor : Preprocessor
        Built by :func:`sage.factory.build.build_preprocess`.
    state : ProcessingState, optional
        Grid the input is on; defaults to the processor's input state.
    amp_dtype : torch.dtype
        Autocast dtype of the model.
    autocast : bool
        Whether the model runs under autocast; a search uses the
        setting the network was trained and validated with.
    device_type : str
        Autocast device type; not derived from the input, so a CPU
        tensor does not enable CPU autocast.

    Returns
    -------
    torch.Tensor
        ``(B, C)`` float32: column 0 the ranking statistic, then the
        raw point-estimate means and their raw sigmas.
    """
    if state is None:
        state = processor.input_state
    batch = GWBatch(x, state=state)
    # Outside autocast: the complex input must not be cast to a reduced
    # dtype.
    batch = processor(batch)
    context = (
        torch.autocast(device_type=device_type, dtype=amp_dtype)
        if autocast
        else nullcontext()
    )
    with context:
        out = model(batch.to_network_input())
    return torch.cat([out[0], out[1]], dim=1).float()
