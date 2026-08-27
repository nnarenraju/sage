#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""O3a HL search -- thin wrapper; all settings live in config_base.py.

Searched with the O3b-trained HL network, which is the out-of-domain pairing
config_base.SEARCH_NETWORK sets. Every path follows from those two facts and is derived
there; what is stated here is what this campaign has *earned*.

__license__ = GPL-3.0-or-later
"""

from config_base import search_spec


def get_spec():
    """The O3a HL campaign specification."""
    return search_spec(
        observing_run="O3a",
        detectors=['H1', 'L1'],
        # Proven, not assumed. `diagnose_separability` was run against these weights on
        # real strain, on CPU and again on the GPU under bfloat16 autocast: perturbing one
        # detector left every other detector's frontend output **bitwise** unchanged while
        # moving its own, so the probe reached the network. A different checkpoint has to
        # earn this again -- `./submit.sh separability <config>`.
        engine=dict(use_frontend_cache=True),
    )
