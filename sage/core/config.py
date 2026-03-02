#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : config.py
Description     : Short description of the file

Created on 2025-11-27 00:24:23

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, Sage
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import torch
import pathlib
import matplotlib as mpl

# LOCAL
from sage.core.logger import get_logger

# Logging
logger = get_logger(__name__)

# Directory containing style files (bundled with the core package)
_STYLES_DIR = pathlib.Path(__file__).parent / "styles"

# Pre-built styles
AVAILABLE_STYLES = {
    "classic": "classic.mplstyle",
    "dark": "dark.mplstyle",
    "publication": "publication.mplstyle",
    "minimalist": "minimal.mplstyle",
}

_CFG = None
_DATA_CFG = None


def register_configs(cfg, data_cfg):
    global _CFG, _DATA_CFG
    _CFG = cfg
    _DATA_CFG = data_cfg


def get_cfg():
    if _CFG is None:
        raise RuntimeError("cfg has not been registered.")
    return _CFG


def get_data_cfg():
    if _DATA_CFG is None:
        raise RuntimeError("data_cfg has not been registered.")
    return _DATA_CFG


def inject_configs(cfg_cls, data_cfg_cls):
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # attach config instances
            self.cfg = cfg_cls()
            self.data_cfg = data_cfg_cls()

            # call original __init__
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator


class ConfiguredModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = get_cfg()
        self.data_cfg = get_data_cfg()


class StyleConfig:
    """
    Base configuration class for Sage.

    Users should subclass this and can optionally set:
        1. mplstyle : str   (e.g., "dark")
           Upon instantiation, the chosen style is automatically applied.

    """

    # Default if user doesn't specify one
    mplstyle: str = "classic"

    def __init__(self):
        self.apply_style()

    def apply_style(self):
        """Load the user's chosen matplotlib style file."""
        style_key = self.mplstyle

        if style_key not in AVAILABLE_STYLES:
            logger.warning(
                f"Unknown mplstyle '{style_key}'. "
                f"Available: {list(AVAILABLE_STYLES.keys())}"
                f"Defaulting to classic."
            )
            style_key = "classic"

        style_path = _STYLES_DIR / AVAILABLE_STYLES[style_key]
        mpl.style.use(str(style_path))
