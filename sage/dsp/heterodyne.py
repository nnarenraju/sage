#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : heterodyne.py
Description     : General signal heterodyning (complex demodulation).

    Heterodyning shifts a (complex) signal to baseband by removing a reference
    (template) phase:

        out = signal * exp(-i * angle(template))

    This is *phase-only* demodulation: the template's instantaneous phase is
    subtracted; the signal's envelope and the residual (signal-minus-template)
    phase are preserved.  A signal with instantaneous phase Phi_s heterodyned
    against a template with phase Phi_r yields phase (Phi_s - Phi_r), whose
    instantaneous frequency is f_s(t) - f_r(t) — zero when the signal matches
    the template, a slowly varying (two-sided) residual otherwise.

    Both ``signal`` and ``template`` must be COMPLEX (an analytic time series,
    or a complex / one-sided frequency series).  This keeps the hot path
    minimal — no Hilbert transforms, no format juggling.  For a real time
    series, build the analytic signal first with the ``analytic_signal`` helper
    in ``tests/dsp_helpers.py`` (kept out of the DSP module since it is only
    used by demos and tests).

    Domain-agnostic (time or frequency domain — the same array op),
    backend-agnostic (numpy or torch), and batched (arbitrary leading dims).

Created on 2026-06-24

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__status__      = inProgress
"""

import numpy as np

try:
    import torch
except ImportError:  # torch is optional for the pure-numpy path
    torch = None


def _is_torch(x):
    return torch is not None and isinstance(x, torch.Tensor)


# ----------------------------------------------------------------------
# low-level primitive: apply a known phase
# ----------------------------------------------------------------------
def apply_phase(data, phase):
    """
    Multiply complex ``data`` by ``exp(-i * phase)`` (phase in radians).

    Use this when the reference phase is already known (e.g. a pre-computed
    chirp phase).  ``data`` must be complex.

    Parameters
    ----------
    data : complex numpy.ndarray or torch.Tensor
    phase : real array broadcastable to ``data``

    Returns
    -------
    Complex array of the same shape/backend as ``data``.
    """
    if _is_torch(data):
        if not torch.is_complex(data):
            raise TypeError("apply_phase expects complex `data`.")
        phase = torch.as_tensor(phase, device=data.device)
        return data * torch.exp(-1j * phase)

    if not np.iscomplexobj(data):
        raise TypeError("apply_phase expects complex `data`.")
    return data * np.exp(-1j * np.asarray(phase))


# ----------------------------------------------------------------------
# main primitive: heterodyne a signal against a template
# ----------------------------------------------------------------------
def heterodyne(signal, template):
    """
    Heterodyne (demodulate) complex ``signal`` by the phase of complex
    ``template``:  ``signal * exp(-i * angle(template))``.

    Removes the template's instantaneous phase, leaving the signal envelope and
    the residual (signal-minus-template) phase.  Time or frequency domain (the
    same operation), numpy or torch, batched.

    Both inputs must be complex (an analytic time series, or a complex /
    one-sided frequency series); ``template`` must broadcast against
    ``signal``.  For a real time series, build its analytic signal first (see
    ``analytic_signal`` in ``tests/dsp_helpers.py``).

    Notes
    -----
    Phase-only: ``exp(-i * angle(template))`` has unit modulus everywhere, so it
    does not blow up where the template amplitude vanishes, and it does not
    weight by the template envelope — this is demodulation, not matched
    filtering.
    """
    if _is_torch(signal):
        return signal * torch.exp(-1j * torch.angle(template))
    return signal * np.exp(-1j * np.angle(template))
