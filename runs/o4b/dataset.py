#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : dataset.py
Description     : O4b noise download + ASD generation.

O4b public data covers H1, L1 and V1 (Virgo joined for O4b). Everything is
written into its OWN release sub-folder (data_release_o4b) so it can never
collide with other concurrent downloads.

__author__        = Narenraju Nagarajan
__license__       = GPL-3.0-or-later
__status__        = ['inUsage']
"""

# Packages
import math

# LOCAL
from sage.data.primer import DataReleaseDownloader
from sage.data.primer import TimelineQuery
from sage.core.config import get_cfg, get_data_cfg
from sage.data.primer.retry import retry_detector
from sage.utils.servers import get_server
from config import set_configs

_RUN = "O4b"
_DETECTORS = ["H1", "L1", "V1"]  # O4b: full network (Virgo joined)

# Server-specific paths come from the single registry in sage/utils/servers.py.
# Switch machines with the SAGE_SERVER env var (or hostname auto-detect).
_SRV = get_server()

# Data root — all downloaded files land under this directory
_DATA_DIR = _SRV.data_root

# Keep O4b in its OWN release sub-folder, completely separate from every other
# run's directory, so concurrent downloads can never touch each other.
_RELEASE_DIRNAME = "data_release_o4b"

# The O4b noise .bin files and their *_segments.json sidecars live here
_DATASET_DIR = f"{_DATA_DIR}/{_RELEASE_DIRNAME}"


def _get_timeline(data_cfg):

    tq = TimelineQuery(
        detector=_DETECTORS,
        observing_run=[_RUN],
        auto_clean_empty_timelines=True,
    )

    tq.download_segments()

    # Fail loudly if any detector came back empty (e.g. proxy dropped mid-query)
    for record in tq.timeline:
        det = record["detector"]
        segs = record["segments"]
        if det in _DETECTORS and (segs is None or len(segs) == 0):
            raise RuntimeError(
                f"Segment query returned 0 segments for {det}. "
                "Check GWOSC connectivity and retry."
            )

    tq.prune_segments(
        rm_short_segments=True,
        rm_min_duration=22.0,
        rm_allevents=True,
        rm_window_length=30,
    )

    buffer = _get_buffer(data_cfg)

    tq.split_into_mini_segments(
        mini_segment_length=512.0 + (buffer * 2.0),
        minimum_segment_duration=16.0,
    )

    tq.sanity_check_mini_segments(
        mini_segment_length=512.0 + (buffer * 2.0),
        minimum_segment_duration=16.0,
        verbose=True,
    )

    return tq


def _get_buffer(data_cfg):

    # We trim the edges after downsamping; so trim length must be accounted for
    # However it can't be arbitrary since we won't get a proper integer removal of samples
    # So we get the nearest trim length above a threshold that will give us an integer for a given sample rate
    return math.ceil(0.2 * data_cfg.sample_rate) / data_cfg.sample_rate


def _download_data_release(tq, data_cfg):

    buffer = _get_buffer(data_cfg)

    drd = DataReleaseDownloader(
        segments_metadata=tq.timeline,
        save_parent_dir=_DATA_DIR,
        release_dirname=_RELEASE_DIRNAME,
        noise_low_freq_cutoff=15.0,
        minimum_segment_duration=22.0,
        corrupt_trim_length=buffer,
        max_download_retries=15,
        retry_delay=5.0,
        num_workers=16,
        proxy_reset_every=50,
        proxy_reset_sleep=90.0,
        make_monolithic_file=True,
        sample_rate=data_cfg.sample_rate,
        save_bin=True,
    )

    drd.download()


def _make_asds(detector, data_cfg):
    from sage.data.primer import EstimateASD, NoBlackout
    from sage.dsp.welch import TorchWelch
    from sage.data.noise import MemmapSingleNoiseSampler
    from sage.data.asd import smoothing

    torch_welch = TorchWelch(
        delta_t=1 / 2048,
        seg_len=int(2048.0 * 4),
        seg_stride=int(2048.0 * 2),
        avg_method="median",
    )

    asd_smoothener = smoothing.LogSplineSmoothing(
        smooth_factor=None,
        noise_low_frequency_cutoff=data_cfg.noise_low_frequency_cutoff,
    )

    easd = EstimateASD(
        # Detector
        detector=detector,
        # Inverse spectrum truncation
        apply_inverse_spectrum_truncation=False,
        max_filter_len=int(round(2048.0 * 2)),
        low_frequency_cutoff=15.0,
        trunc_method="hann",
        # Spectral estimator (Welch); EstimateASD square-roots its output
        asd_method=torch_welch,
        num_samples=250_000,
        store_asds_as_bin=True,
        # Fiducial ASD parameters
        blackout_policy=NoBlackout(),
        # Interpolation
        interpolate_asd=True,
        training_sample_length=int(2048.0 * 16),
        # ASD smoothing
        asd_smoothener=asd_smoothener,
    )

    # This is used for whitening with the exact segment ASD before recolouring
    easd.estimate_segment_asds(
        noise_segments_file=f"{_DATASET_DIR}/data_{detector}_{_RUN}.bin"
    )

    # With this we make num_samples random ASDs from the given data
    # We use this for recolouring augmentation
    # We also do blackout and aggregate the ASDs to produce the fiducial ASD
    noise_sampler = MemmapSingleNoiseSampler(
        f"{_DATASET_DIR}/data_{detector}_{_RUN}.bin",
        return_tensor=True,
    )
    easd.estimate_raw_asds(
        noise_sampler=noise_sampler, duration=int(round(2048.0 * 16))
    )


def download_single_detector(detector, num_workers=8):
    """Download one detector only — safe to run when others are already complete."""
    set_configs()
    data_cfg = get_data_cfg()
    buffer = _get_buffer(data_cfg)

    tq = TimelineQuery(
        detector=[detector],
        observing_run=[_RUN],
        auto_clean_empty_timelines=True,
    )
    tq.download_segments()

    if not any(r["detector"] == detector and len(r["segments"]) > 0 for r in tq.timeline):
        raise RuntimeError(f"Segment query returned 0 segments for {detector}.")

    tq.prune_segments(
        rm_short_segments=True,
        rm_min_duration=22.0,
        rm_allevents=True,
        rm_window_length=30,
    )
    tq.split_into_mini_segments(
        mini_segment_length=512.0 + (buffer * 2.0),
        minimum_segment_duration=16.0,
    )
    tq.sanity_check_mini_segments(
        mini_segment_length=512.0 + (buffer * 2.0),
        minimum_segment_duration=16.0,
        verbose=False,
    )

    drd = DataReleaseDownloader(
        segments_metadata=tq.timeline,
        save_parent_dir=_DATA_DIR,
        release_dirname=_RELEASE_DIRNAME,
        noise_low_freq_cutoff=15.0,
        minimum_segment_duration=22.0,
        corrupt_trim_length=buffer,
        max_download_retries=15,
        retry_delay=5.0,
        num_workers=num_workers,
        proxy_reset_every=50,
        proxy_reset_sleep=90.0,
        make_monolithic_file=True,
        sample_rate=data_cfg.sample_rate,
        save_bin=True,
    )
    drd.download()


def make_dataset():

    set_configs()

    # Shared configs
    cfg, data_cfg = get_cfg(), get_data_cfg()

    # Make datasets
    tq = _get_timeline(data_cfg)
    _download_data_release(tq, data_cfg)
    for det in _DETECTORS:
        _make_asds(det, data_cfg)


def make_asds_only():
    """Generate fiducial / recolour / segment ASDs for all detectors.

    Assumes the noise .bin files are already downloaded. Does NOT re-download.
    Run from this directory so the relative ``export_dir`` (./run_export)
    resolves to runs/o4b/run_export for the fiducial ASDs.
    """
    set_configs()
    _, data_cfg = get_cfg(), get_data_cfg()
    for det in _DETECTORS:
        _make_asds(det, data_cfg)


def make_asds_single(detector):
    """Generate ASDs for a single detector in its own process."""
    set_configs()
    _, data_cfg = get_cfg(), get_data_cfg()
    _make_asds(detector, data_cfg)


def retry_dataset(detectors=None, num_workers=8):

    set_configs()

    data_dir = _DATASET_DIR
    for det in (detectors or _DETECTORS):
        print(f"\n{'='*60}")
        print(f" Retrying {det} / {_RUN}")
        print(f"{'='*60}")
        retry_detector(det, _RUN, data_dir, num_workers=num_workers)
