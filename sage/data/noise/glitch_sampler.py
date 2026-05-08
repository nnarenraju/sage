#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GlitchOversampledNoiseSampler

Subclass of MemmapNoiseSampler that injects known high-SNR glitch windows
into a controlled fraction of each training batch.

Motivation
----------
The O3b noise training data is dominated by Scattered_Light glitches (~65 % of
H1 triggers), leaving rare-but-dangerous classes like Repeating_Blips
underrepresented.  When evaluating on O3a test data, these minority classes
produce the loudest background triggers.

Solution
--------
Override _sample_starts_batch so that `glitch_frac` of each batch is drawn
from a GPS-aligned pool of high-SNR O3b glitch windows (sourced from the
GravitySpy CSV catalogs).  All downstream processing — DYN_RANGE_FAC
normalisation, TD→FD conversion, and RecolourPostprocess — runs unchanged,
so the injected glitch windows receive exactly the same treatment as regular
noise windows.

Class-balanced sampling
-----------------------
With `class_balanced=True` (default), a glitch class is chosen uniformly at
random before selecting a specific event.  This gives equal weight to each
morphological class regardless of how common it is in the catalogue, directly
addressing the majority-class dominance problem.
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
    MemmapNoiseSampler with class-balanced O3b glitch oversampling.

    Parameters
    ----------
    catalog_files : list of (det_idx, csv_path)
        GravitySpy CSV files keyed by detector index (0=H1, 1=L1).
        Example: [(0, '/path/H1_O3b.csv'), (1, '/path/L1_O3b.csv')]
    min_snr : float
        Only glitches with GravitySpy SNR >= min_snr are used.
    glitch_frac : float
        Fraction of each batch replaced by glitch windows. 0.10 = 10 %.
    class_balanced : bool
        If True, sample uniformly over glitch classes before sampling an
        event, giving equal representation to rare classes.  If False,
        sample uniformly over all qualifying events.
    exclude_classes : set of str, optional
        GravitySpy class labels to exclude (e.g. {'No_Glitch'}).
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
        # Parent __init__ starts the prefetch thread.  The overridden
        # _sample_starts_batch is called from that thread, so we set
        # _glitch_ready=False first and flip it only after our data is loaded.
        self._glitch_ready  = False
        self.glitch_frac    = glitch_frac
        self.class_balanced = class_balanced

        super().__init__(*args, **kwargs)

        _exclude = exclude_classes or {'No_Glitch'}

        # det_idx → {class_name → np.ndarray of sample_start_indices}
        self._glitch_by_class: dict[int, dict[str, np.ndarray]] = {}
        # det_idx → flat np.ndarray (all classes concatenated)
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

    # ------------------------------------------------------------------
    # GPS helpers
    # ------------------------------------------------------------------

    def _reload_gps_segs(self, det_idx: int) -> np.ndarray:
        """Re-read segments JSON for a detector to obtain GPS time ranges."""
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
        Map a glitch peak GPS time to a memmap start index for a window
        of length self.seq_len centred on that time.  Returns None if the
        window falls outside valid segment coverage.
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

        # Segment bounds
        seg_end_sample  = s["sidx"] + int((s["t1"] - s["t0"]) * sr)
        if start < s["sidx"] or start + self.seq_len > seg_end_sample:
            return None
        return int(start)

    # ------------------------------------------------------------------
    # Catalogue loading
    # ------------------------------------------------------------------

    def _csv_to_starts(
        self,
        csv_path:       str,
        gps_segs:       np.ndarray,
        min_snr:        float,
        exclude_classes: set,
    ) -> dict[str, np.ndarray]:
        """
        Parse a GravitySpy CSV, filter by SNR and class, map GPS to memmap
        start indices.  Returns {class_label: np.array(start_indices)}.
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

    # ------------------------------------------------------------------
    # Override: inject glitch start indices
    # ------------------------------------------------------------------

    def _sample_starts_batch(self, batch_size: int):
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
