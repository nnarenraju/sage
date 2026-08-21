#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : product.py
Description   : The figure data product: the numbers behind one figure.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Separate from the package ``__init__`` so that the builders can import the container
without importing the package that dispatches to them. Every builder needs this type, and
the dispatcher needs every builder, so leaving it in ``__init__`` makes the two import
each other.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class FigData:
    """
    The numbers behind one figure.

    Written by an analysis stage and read by a plotting function that computes nothing,
    so a figure can be redrawn without rerunning the analysis, the numbers behind it can
    be released alongside it, and a plot cannot disagree with what it depicts.
    """

    figure: str
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    scalars: Dict[str, object] = field(default_factory=dict)
    attrs: Dict[str, object] = field(default_factory=dict)

    def require(self, *names: str) -> None:
        """
        Assert the named arrays are present before drawing.

        Raises naming every missing array at once rather than failing on the first, since
        a builder that dropped one field has usually dropped several.
        """
        missing = [name for name in names if name not in self.arrays]
        if missing:
            raise KeyError(
                f"figure {self.figure!r} is missing {missing}; it holds "
                f"{sorted(self.arrays)}. The builder did not write everything the "
                "declaration says this figure needs"
            )

    def save(self, path: str | Path) -> Path:
        """
        Write atomically, so an interrupted build leaves no half-written product.

        Arrays and scalars are kept apart on disk as they are in memory. A scalar folded
        into a zero-dimensional dataset reads back as an array, and a plotting function
        that indexed it would then be indexing a number.
        """
        from sage.utils.atomic_io import atomic_h5

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with atomic_h5(target, mode="w") as handle:
            handle.attrs["figure"] = str(self.figure)
            for key, value in (self.attrs or {}).items():
                handle.attrs[key] = value
            arrays = handle.create_group("arrays")
            for name, values in (self.arrays or {}).items():
                values = np.asarray(values)
                if values.dtype.kind in "SUO":
                    import h5py

                    values = np.asarray(
                        [str(v) for v in values],
                        dtype=h5py.string_dtype(encoding="utf-8"),
                    )
                arrays.create_dataset(name, data=values)
            scalars = handle.create_group("scalars")
            for name, value in (self.scalars or {}).items():
                scalars.attrs[name] = value
        return target

    @classmethod
    def load(cls, path: str | Path) -> "FigData":
        """
        Read a figure data product.

        The figure key is taken from the file rather than from the caller: a product
        loaded under the wrong name would be drawn by the wrong plotting function, which
        fails only if the arrays happen to disagree.
        """
        import h5py

        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"no figure data at {target}")
        with h5py.File(target, "r") as handle:
            if "figure" not in handle.attrs:
                raise ValueError(
                    f"{target} records no figure key, so which figure it belongs to "
                    "cannot be established"
                )
            arrays = {}
            for name, dataset in handle.get("arrays", {}).items():
                values = dataset[()]
                if values.dtype.kind in "SO":
                    values = np.asarray(
                        [
                            v.decode() if isinstance(v, bytes) else str(v)
                            for v in values
                        ]
                    )
                arrays[name] = values
            scalars = dict(handle["scalars"].attrs) if "scalars" in handle else {}
            return cls(
                figure=str(handle.attrs["figure"]),
                arrays=arrays,
                scalars=scalars,
                attrs={
                    key: handle.attrs[key]
                    for key in handle.attrs
                    if key != "figure"
                },
            )
