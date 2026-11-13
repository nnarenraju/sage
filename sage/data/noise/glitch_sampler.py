#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Noise sampler that replaces a fraction of each batch's windows with
windows centred on GravitySpy glitches, optionally class-balanced.
"""

import csv
import json
import threading
import numpy as np

from pathlib import Path
from collections import defaultdict

from .real_noise import MemmapNoiseSampler


class GlitchOversampledNoiseSampler(MemmapNoiseSampler):
    """
    ``MemmapNoiseSampler`` that oversamples GravitySpy glitch windows.

    Other positional and keyword arguments go to ``MemmapNoiseSampler``.
    A glitch window is centred on the glitch peak and must lie inside
    one segment.

    Parameters
    ----------
    catalog_files : list of tuple
        ``(channel index, GravitySpy CSV path)`` pairs.
    min_snr : float
        Lowest GravitySpy SNR used.
    glitch_frac : float
        Fraction of each batch replaced by glitch windows.
    class_balanced : bool
        Draw a class uniformly before a glitch; otherwise draw uniformly
        over all glitches.
    exclude_classes : set of str or None
        GravitySpy labels left out; ``None`` is ``{"No_Glitch"}``.

    Raises
    ------
    ValueError
        If a channel index is out of range.
    """

    def __init__(
        self,
        *args,
        catalog_files: list,
        min_snr:         float = 15.0,
        glitch_frac:     float = 0.10,
        class_balanced:  bool  = True,
        exclude_classes: set   = None,
        **kwargs,
    ):
        # The parent starts the prefetch thread, which calls
        # _sample_starts_batch; glitches are used once loaded
        self._glitch_ready  = False
        self.glitch_frac    = glitch_frac
        self.class_balanced = class_balanced

        super().__init__(*args, **kwargs)

        _exclude = exclude_classes or {'No_Glitch'}

        # det_idx -> {class name -> memmap start indices}
        self._glitch_by_class: dict[int, dict[str, np.ndarray]] = {}
        # det_idx -> start indices of all classes
        self._glitch_flat:     dict[int, np.ndarray]            = {}

        for det_idx, csv_path in catalog_files:
            if det_idx >= len(self.bin_files):
                raise ValueError(
                    f"catalog_files det_idx={det_idx} out of range "
                    f"(sampler has {len(self.bin_files)} detectors)"
                )
            gps_segs  = self._reload_gps_segs(det_idx)
            by_class  = self._csv_to_starts(
                csv_path, gps_segs, min_snr, _exclude
            )
            self._glitch_by_class[det_idx] = by_class
            flat = (
                np.concatenate(list(by_class.values()))
                if by_class else np.array([], dtype=np.int64)
            )
            self._glitch_flat[det_idx] = flat

            total = sum(len(v) for v in by_class.values())
            print(
                f"  [GlitchSampler] det={det_idx}  "
                f"{total:,} glitch windows across {len(by_class)} classes "
                f"(SNR≥{min_snr})"
            )
            for cls, arr in sorted(by_class.items(), key=lambda x: -len(x[1])):
                print(f"    {cls:<25}  {len(arr):>6,}")

        self._glitch_ready = True

    def _reload_gps_segs(self, det_idx: int) -> np.ndarray:
        """
        GPS spans of a detector's segments, from its sidecar.

        Parameters
        ----------
        det_idx : int
            Index into ``bin_files``.

        Returns
        -------
        numpy.ndarray
            Structured array (``t0``, ``t1``, ``sidx``, ``sr``) sorted
            by ``t0``.
        """
        p         = self.bin_files[det_idx]
        meta_path = p.parent / f"{p.stem}_segments.json"
        with open(meta_path) as f:
            meta = json.load(f)
        segs = np.array(
            [
                (s["gps_start"], s["gps_end"], s["sample_start_idx"], s["sample_rate"])
                for s in meta
            ],
            dtype=[("t0", "f8"), ("t1", "f8"), ("sidx", "i8"), ("sr", "f8")],
        )
        segs.sort(order="t0")
        return segs

    def _gps_to_start(
        self,
        gps_segs:  np.ndarray,
        gps_time:  float,
    ):
        """
        Memmap start of the window centred on a GPS time.

        Parameters
        ----------
        gps_segs : numpy.ndarray
            Segment spans (:meth:`_reload_gps_segs`).
        gps_time : float
            Glitch peak GPS time.

        Returns
        -------
        int or None
            Start index, or ``None`` if the window leaves its segment.
        """
        i = np.searchsorted(gps_segs["t0"], gps_time, side="right") - 1
        if i < 0 or i >= len(gps_segs):
            return None
        s = gps_segs[i]
        if gps_time >= s["t1"]:
            return None

        sr              = s["sr"]
        offset_samples  = int((gps_time - s["t0"]) * sr)
        centre          = s["sidx"] + offset_samples
        start           = centre - self.seq_len // 2

        seg_end_sample  = s["sidx"] + int((s["t1"] - s["t0"]) * sr)
        if start < s["sidx"] or start + self.seq_len > seg_end_sample:
            return None
        return int(start)

    def _csv_to_starts(
        self,
        csv_path:       str,
        gps_segs:       np.ndarray,
        min_snr:        float,
        exclude_classes: set,
    ) -> dict[str, np.ndarray]:
        """
        Memmap starts of the glitches in a GravitySpy CSV.

        Rows without a valid ``snr`` or ``ml_label`` are skipped.

        Parameters
        ----------
        csv_path : str
            GravitySpy CSV file.
        gps_segs : numpy.ndarray
            Segment spans (:meth:`_reload_gps_segs`).
        min_snr : float
            Lowest SNR used.
        exclude_classes : set of str
            Labels left out.

        Returns
        -------
        dict of str to numpy.ndarray
            Start indices per class label.
        """
        by_class: dict[str, list] = defaultdict(list)

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    snr   = float(row["snr"])
                    label = row["ml_label"]
                except (KeyError, ValueError):
                    continue

                if snr < min_snr or label in exclude_classes:
                    continue

                gps = float(row["peak_time"])
                ns  = float(row.get("peak_time_ns", "0"))
                gps += ns * 1e-9

                start = self._gps_to_start(gps_segs, gps)
                if start is not None:
                    by_class[label].append(start)

        return {k: np.array(v, dtype=np.int64) for k, v in by_class.items()}

    def _sample_starts_batch(self, batch_size: int):
        """
        Random starts with ``glitch_frac`` of the positions glitches.

        The same positions are replaced in every detector with a
        catalogue; segment ids are not changed.

        Parameters
        ----------
        batch_size : int
            Number of windows.

        Returns
        -------
        start_indices, segment_indices : list of numpy.ndarray
            Per-detector start indices and segment ids.
        """
        start_indices, segment_indices = super()._sample_starts_batch(batch_size)

        if not self._glitch_ready or not self._glitch_by_class:
            return start_indices, segment_indices

        n_glitch = int(batch_size * self.glitch_frac)
        if n_glitch == 0:
            return start_indices, segment_indices

        positions = self.rng.choice(batch_size, size=n_glitch, replace=False)

        for det_idx, by_class in self._glitch_by_class.items():
            if det_idx >= len(start_indices) or not by_class:
                continue

            classes = list(by_class.keys())
            for pos in positions:
                if self.class_balanced:
                    cls  = classes[self.rng.integers(0, len(classes))]
                    pool = by_class[cls]
                else:
                    pool = self._glitch_flat[det_idx]
                start_indices[det_idx][pos] = pool[self.rng.integers(0, len(pool))]

        return start_indices, segment_indices
