#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""O3b HL search -- thin wrapper; all settings live in config_base.py.

Searched with the O3a-trained HL network, which is the out-of-domain pairing
config_base.SEARCH_NETWORK sets. Every path follows from those two facts and is derived
there; what is stated here is what this campaign has *earned*.

__license__ = GPL-3.0-or-later
"""

from config_base import search_spec


def get_spec():
    """The O3b HL campaign specification."""
    return search_spec(
        observing_run="O3b",
        detectors=['H1', 'L1'],
    )
