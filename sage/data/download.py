"""
Observing-run data in the data directory: the dataset's fixed numbers, the
strain's download from GWOSC with resume, the ASDs estimated from it, and the
records written beside them.
"""

import json
import math
import os
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from gwosc.datasets import find_datasets
from pycbc import DYN_RANGE_FAC

from sage.config.machine import (
    MissingData,
    asds_file,
    bad_time_file,
    dataset_file,
    failed_file,
    fiducial_dir,
    fiducial_file,
    observing_run_dir,
    recolour_bank_file,
    segment_asd_file,
    segments_file,
    strain_file,
)
from sage.core.hardcode import _DETECTORS
from sage.data.asd.make_fiducial import K_DEPTH, PERCENTILE, pooled_fiducial
from sage.data.asd.smoothing import LogSplineSmoothing
from sage.data.noise import MemmapSingleNoiseSampler
from sage.data.noise.data_quality import read_bad_time, write_bad_time
from sage.data.primer import DataReleaseDownloader, EstimateASD, NoBlackout, TimelineQuery
from sage.data.primer.retry import retry_detector
from sage.dsp.welch import TorchWelch

# Source: runs/<run>/dataset.py, the same for O3a, O3b, O4a and O4b.
SAMPLE_RATE = 2048.0
TRIM = math.ceil(0.2 * SAMPLE_RATE) / SAMPLE_RATE
DATASET = {
    "gwosc_files": "<run>_4KHZ_R1",
    "segments": "<detector>_DATA",
    "event_cut_in_s": 30.0,
    "minimum_segment_in_s": 22.0,
    "piece_in_s": 512.0,
    "minimum_piece_in_s": 16.0,
    "sample_rate_in_hz": SAMPLE_RATE,
    "highpass_in_hz": 15.0,
    "trim_in_s": TRIM,
    "scale": float(DYN_RANGE_FAC),
    "dtype": "float32",
}
# Source: runs/<run>/dataset.py ``_make_asds``, the same for O3a, O3b, O4a and O4b.
ASDS = {
    "welch_segment_in_s": 4.0,
    "welch_stride_in_s": 2.0,
    "welch_average": "median",
    "length_in_s": 16.0,
    "smoothing_cutoff_in_hz": 15.0,
    "taper_cutoff_in_hz": 15.0,
    "bank_size": 250_000,
}
BANK_SEED = 0
# Source: runs/build_fiducial.py.
FIDUCIAL = {
    "psd_aggregation": "median",
    "worst_case": f"p{PERCENTILE}",
    "blackout_policy": "LocalLineNotch",
    "local_line_notch": {
        "k_depth": K_DEPTH, "window_hz": 8.0, "thresh": 2.0, "taper_bins": 1.5, "f_low": 15.0, "f_high": 1024.0,
    },
}
RETRIES = 15
RETRY_DELAY = 5.0
PROXY_RESET_EVERY = 50
PROXY_RESET_SLEEP = 90.0


def gwosc_runs() -> dict:
    """GWOSC's observing runs, by name in lower case."""
    return {name.lower(): name for name in find_datasets(type="run")}


def gwosc_run(run: str) -> str:
    """
    GWOSC's name of an observing run given in any case.

    Raises
    ------
    ValueError
        If GWOSC lists no such run.
    """
    names = gwosc_runs()
    if run.lower() not in names:
        raise ValueError(f"observing run {run!r}: GWOSC lists {', '.join(sorted(names.values()))}")
    return names[run.lower()]


def recorded_gwosc(data_dir: Path, run: str) -> dict | None:
    """GWOSC's name and detectors of an observing run as recorded in its dataset record, or None."""
    path = dataset_file(data_dir, run)
    return yaml.safe_load(path.read_text()).get("gwosc") if path.exists() else None


def gwosc_record(data_dir: Path, run: str) -> dict:
    """
    GWOSC's name of an observing run and the detectors it lists with strain.

    Queried once and recorded in the run's dataset record, so later checks need no network.
    """
    recorded = recorded_gwosc(data_dir, run)
    if recorded is not None:
        return recorded
    name = gwosc_run(run)
    detectors = [detector for detector in sorted(_DETECTORS) if name in find_datasets(detector=detector, type="run")]
    path = dataset_file(data_dir, run)
    record = yaml.safe_load(path.read_text()) if path.exists() else {**DATASET, "detectors": {}}
    record["gwosc"] = {"name": name, "detectors": detectors}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    return record["gwosc"]


def run_detectors(data_dir: Path, run: str) -> list:
    """Detectors with strain in an observing run, as GWOSC lists them (``gwosc_record``)."""
    return gwosc_record(data_dir, run)["detectors"]


def gwosc_name(data_dir: Path, run: str) -> str:
    """GWOSC's name of an observing run (``gwosc_record``)."""
    return gwosc_record(data_dir, run)["name"]


def fetch_detector(data_dir: Path, run: str, detector: str, workers: int) -> None:
    """
    Download one detector's strain into ``training.part``, then move it into ``training``.

    Raises
    ------
    RuntimeError
        If GWOSC returns no segments for the detector.
    """
    name = gwosc_name(data_dir, run)
    timeline = TimelineQuery(detector=[detector], observing_run=[name], auto_clean_empty_timelines=True)
    timeline.download_segments()
    if len(timeline.timeline) == 0:
        raise RuntimeError(f"GWOSC returned no {detector}_DATA segments for {name}")
    timeline.prune_segments(
        rm_short_segments=True,
        rm_min_duration=DATASET["minimum_segment_in_s"],
        rm_allevents=True,
        rm_window_length=DATASET["event_cut_in_s"],
    )
    piece = DATASET["piece_in_s"] + 2.0 * TRIM
    timeline.split_into_mini_segments(mini_segment_length=piece, minimum_segment_duration=DATASET["minimum_piece_in_s"])
    timeline.sanity_check_mini_segments(
        mini_segment_length=piece, minimum_segment_duration=DATASET["minimum_piece_in_s"], verbose=True
    )

    DataReleaseDownloader(
        segments_metadata=timeline.timeline,
        save_parent_dir=str(Path(data_dir) / run.lower()),
        noise_low_freq_cutoff=DATASET["highpass_in_hz"],
        corrupt_trim_length=TRIM,
        max_download_retries=RETRIES,
        retry_delay=RETRY_DELAY,
        num_workers=workers,
        proxy_reset_every=PROXY_RESET_EVERY,
        proxy_reset_sleep=PROXY_RESET_SLEEP,
        make_monolithic_file=True,
        save_bin=True,
        sample_rate=SAMPLE_RATE,
        release_dirname="training.part",
    ).download()

    part = observing_run_dir(data_dir, run, "training.part")
    path = strain_file(data_dir, run, detector)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name in (path.name, segments_file(path).name, failed_file(path).name):
        os.replace(part / name, path.parent / name)
    if not any(part.iterdir()):
        part.rmdir()


def retry_pieces(data_dir: Path, run: str, detector: str, workers: int) -> None:
    """Fetch a detector's failed pieces again; its failed list then holds those still missing."""
    path = strain_file(data_dir, run, detector)
    retry_detector(
        detector,
        gwosc_name(data_dir, run),
        path.parent,
        num_workers=workers,
        sample_rate=SAMPLE_RATE,
        noise_low_freq_cutoff=DATASET["highpass_in_hz"],
        max_download_retries=RETRIES,
        retry_delay=RETRY_DELAY,
        proxy_reset_every=PROXY_RESET_EVERY,
        proxy_reset_sleep=PROXY_RESET_SLEEP,
        mini_segment_duration=DATASET["piece_in_s"],
        split_minimum_duration=DATASET["minimum_piece_in_s"],
        prune_minimum_duration=DATASET["minimum_segment_in_s"],
        rm_window_length=DATASET["event_cut_in_s"],
    )
    os.replace(path.parent / f"{path.stem}_retry_failed.json", failed_file(path))


def check_strain(data_dir: Path, run: str, detector: str) -> dict:
    """
    Check one detector's strain file against its sidecar and the dataset's numbers.

    Returns
    -------
    dict
        ``pieces``, ``days`` and ``gps`` (first start, last end) of the file.

    Raises
    ------
    MissingData
        If pieces failed to download.
    ValueError
        If a record or the file size does not match.
    """
    path = strain_file(data_dir, run, detector)
    failed = json.loads(failed_file(path).read_text())
    if failed:
        raise MissingData(f"{path}: {len(failed)} pieces failed to download")
    records = json.loads(segments_file(path).read_text())

    offset = 0
    for index, record in enumerate(records):
        expected = {
            "segment_index": index,
            "detector": detector,
            "sample_start_idx": offset // 4,
            "sample_rate": SAMPLE_RATE,
            "dtype": DATASET["dtype"],
            "endianness": "<",
            "byte_offset": offset,
            "byte_length": 4 * record["nsamples"],
            "dyn_range_fac": DATASET["scale"],
            "noise_low_freq_cutoff": DATASET["highpass_in_hz"],
        }
        differ = {key: record[key] for key, value in expected.items() if record[key] != value}
        if differ or record["observing_run"].lower() != run.lower():
            raise ValueError(f"{segments_file(path)} piece {index}: {differ or record['observing_run']} differs")
        offset += record["byte_length"]
    if path.stat().st_size != offset:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, its pieces hold {offset}")

    return {
        "pieces": len(records),
        "days": sum(record["nsamples"] for record in records) / SAMPLE_RATE / 86400.0,
        "gps": [min(record["gps_start"] for record in records), max(record["gps_end"] for record in records)],
    }


def estimator(detector: str, asd_dir: Path) -> EstimateASD:
    """``EstimateASD`` with the dataset's ASD settings, writing under ``asd_dir``."""
    welch = TorchWelch(
        delta_t=1.0 / SAMPLE_RATE,
        seg_len=int(SAMPLE_RATE * ASDS["welch_segment_in_s"]),
        seg_stride=int(SAMPLE_RATE * ASDS["welch_stride_in_s"]),
        avg_method=ASDS["welch_average"],
    )
    return EstimateASD(
        detector=detector,
        sample_rate=SAMPLE_RATE,
        data_dir=asd_dir,
        fiducial_dir=asd_dir,
        apply_inverse_spectrum_truncation=False,
        max_filter_len=int(round(SAMPLE_RATE * 2.0)),
        low_frequency_cutoff=ASDS["taper_cutoff_in_hz"],
        trunc_method="hann",
        asd_method=welch,
        num_samples=ASDS["bank_size"],
        store_asds_as_bin=True,
        blackout_policy=NoBlackout(),
        interpolate_asd=True,
        training_sample_length=int(SAMPLE_RATE * ASDS["length_in_s"]),
        asd_smoothener=LogSplineSmoothing(
            smooth_factor=None, noise_low_frequency_cutoff=ASDS["smoothing_cutoff_in_hz"]
        ),
    )


def move_built(part: Path, pairs: list) -> None:
    """Move built files out of ``part`` to their places, then remove the directories left empty."""
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
    for directory in dict.fromkeys([*(source.parent for source, _ in pairs), part]):
        if not any(directory.iterdir()):
            directory.rmdir()


def build_segment_asds(data_dir: Path, run: str, detector: str) -> None:
    """Estimate the ASD of every strain piece into ``asds.part``, then move it into ``asds``."""
    part = observing_run_dir(data_dir, run, "asds.part")
    estimator(detector, part).estimate_segment_asds(noise_segments_file=strain_file(data_dir, run, detector))
    built = segment_asd_file(part, detector)
    path = segment_asd_file(observing_run_dir(data_dir, run, "asds"), detector)
    move_built(part, [(built, path), (segments_file(built), segments_file(path))])


def build_bank(data_dir: Path, run: str, detector: str) -> None:
    """Draw the recolour bank from the strain, seeded, into ``asds.part``, then move it into ``asds``."""
    part = observing_run_dir(data_dir, run, "asds.part")
    sampler = MemmapSingleNoiseSampler(strain_file(data_dir, run, detector), return_tensor=True)
    estimator(detector, part).write_raw_asds(
        noise_sampler=partial(sampler.sample, rng=np.random.default_rng(BANK_SEED)),
        duration=int(SAMPLE_RATE * ASDS["length_in_s"]),
    )
    built = recolour_bank_file(part, detector)
    path = recolour_bank_file(observing_run_dir(data_dir, run, "asds"), detector)
    move_built(part, [(built, path), (built.with_suffix(".json"), path.with_suffix(".json"))])


def check_segment_asds(data_dir: Path, run: str, detector: str) -> int:
    """
    Check one detector's per-segment ASDs against its strain pieces and the ASD settings.

    Returns
    -------
    int
        Number of ASDs.

    Raises
    ------
    ValueError
        If a record or the file size does not match.
    """
    path = segment_asd_file(observing_run_dir(data_dir, run, "asds"), detector)
    pieces = json.loads(segments_file(strain_file(data_dir, run, detector)).read_text())
    records = json.loads(segments_file(path).read_text())
    if len(records) != len(pieces):
        raise ValueError(f"{segments_file(path)}: {len(records)} ASDs for {len(pieces)} strain pieces")

    bins = int(SAMPLE_RATE * ASDS["length_in_s"]) // 2 + 1
    for index, (piece, record) in enumerate(zip(pieces, records)):
        expected = {
            "noise_segment_index": piece["segment_index"],
            "gps_start": piece["gps_start"],
            "gps_end": piece["gps_end"],
            "sample_rate": SAMPLE_RATE,
            "psd_len": bins,
            "byte_offset": 4 * bins * index,
            "byte_length": 4 * bins,
            "delta_f": 1.0 / ASDS["length_in_s"],
            "seg_len": int(SAMPLE_RATE * ASDS["welch_segment_in_s"]),
            "seg_stride": int(SAMPLE_RATE * ASDS["welch_stride_in_s"]),
            "window": "hann",
            "inverse_spectrum_truncation": 0,
            "low_frequency_cutoff": ASDS["taper_cutoff_in_hz"],
            "interpolation": 1,
        }
        differ = {key: record[key] for key, value in expected.items() if record[key] != value}
        if differ:
            raise ValueError(f"{segments_file(path)} ASD {index}: {differ} differs")
    if path.stat().st_size != 4 * bins * len(records):
        raise ValueError(f"{path}: {path.stat().st_size} bytes, its {len(records)} ASDs hold {4 * bins * len(records)}")
    return len(records)


def check_bank(data_dir: Path, run: str, detector: str) -> None:
    """
    Check one detector's recolour bank against its sidecar and the ASD settings.

    Raises
    ------
    ValueError
        If the sidecar or the file size does not match.
    """
    path = recolour_bank_file(observing_run_dir(data_dir, run, "asds"), detector)
    record = json.loads(path.with_suffix(".json").read_text())
    bins = int(SAMPLE_RATE * ASDS["length_in_s"]) // 2 + 1
    expected = {
        "detector": detector,
        "num_psds": ASDS["bank_size"],
        "num_freq_bins": bins,
        "dtype": "float32",
        "byte_order": "little",
        "layout": "row-major",
        "sample_rate": SAMPLE_RATE,
        "delta_f": 1.0 / ASDS["length_in_s"],
        "freq_start": 0.0,
        "freq_end": SAMPLE_RATE / 2.0,
        "psd_method": "TorchWelch",
        "apply_inverse_spectrum_truncation": False,
        "low_frequency_cutoff": ASDS["taper_cutoff_in_hz"],
    }
    differ = {key: record[key] for key, value in expected.items() if record[key] != value}
    if differ:
        raise ValueError(f"{path.with_suffix('.json')}: {differ} differs")
    if path.stat().st_size != 4 * bins * ASDS["bank_size"]:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, its {ASDS['bank_size']} ASDs hold "
                         f"{4 * bins * ASDS['bank_size']}")


def fiducial_runs(data_dir: Path, runs: list, detector: str) -> list:
    """
    The fiducial's observing runs with strain from ``detector``, as GWOSC lists them, sorted.

    Raises
    ------
    ValueError
        If none has.
    """
    pooled = [run for run in sorted({run.lower() for run in runs}) if detector in run_detectors(data_dir, run)]
    if not pooled:
        raise ValueError(f"no fiducial run of {runs} has {detector} strain")
    return pooled


def fiducial_record(data_dir: Path, detector: str, pooled: list) -> dict:
    """The fiducial sidecar's fixed fields."""
    bins = int(SAMPLE_RATE * ASDS["length_in_s"]) // 2 + 1
    return {
        "detector": detector,
        "num_freq_bins": bins,
        "dtype": "float32",
        "byte_order": "little",
        "sample_rate": SAMPLE_RATE,
        "delta_f": 1.0 / ASDS["length_in_s"],
        "freq_start": 0.0,
        "freq_end": SAMPLE_RATE / 2.0,
        "num_samples_used": ASDS["bank_size"] * len(pooled),
        "runs_pooled": [gwosc_name(data_dir, run) for run in pooled],
        **FIDUCIAL,
    }


def build_fiducial(data_dir: Path, runs: list, detector: str) -> None:
    """Build one detector's fiducial from the pooled recolour banks into ``<fiducial>.part``, then move it."""
    pooled = fiducial_runs(data_dir, runs, detector)
    record = fiducial_record(data_dir, detector, pooled)
    for run in pooled:
        check_bank(data_dir, run, detector)
    banks = [
        np.memmap(recolour_bank_file(observing_run_dir(data_dir, run, "asds"), detector), dtype="<f4", mode="r",
                  shape=(ASDS["bank_size"], record["num_freq_bins"]))
        for run in pooled
    ]
    asd, lines = pooled_fiducial(banks, record["delta_f"])

    directory = fiducial_dir(data_dir, runs)
    part = directory.with_name(f"{directory.name}.part")
    part.mkdir(parents=True, exist_ok=True)
    built = fiducial_file(part, detector)
    asd.tofile(built)
    freqs = np.arange(record["num_freq_bins"], dtype=np.float64) * record["delta_f"]
    band = (freqs >= 15.0) & (freqs <= 1024.0)
    built.with_suffix(".json").write_text(json.dumps(
        {**record, "num_line_bins": int(len(lines)), "line_frac_inband": float(len(lines) / max(1, int(band.sum())))},
        indent=2,
    ))
    path = fiducial_file(directory, detector)
    move_built(part, [(built, path), (built.with_suffix(".json"), path.with_suffix(".json"))])


def check_fiducial(data_dir: Path, runs: list, detector: str) -> None:
    """
    Check one detector's fiducial against its sidecar and the dataset's numbers.

    Raises
    ------
    ValueError
        If the sidecar or the size does not match, or a value is not finite.
    """
    path = fiducial_file(fiducial_dir(data_dir, runs), detector)
    expected = fiducial_record(data_dir, detector, fiducial_runs(data_dir, runs, detector))
    record = json.loads(path.with_suffix(".json").read_text())
    differ = {key: record.get(key) for key, value in expected.items() if record.get(key) != value}
    if differ:
        raise ValueError(f"{path.with_suffix('.json')}: {differ} differs")
    asd = np.fromfile(path, dtype="<f4")
    if asd.size != expected["num_freq_bins"]:
        raise ValueError(f"{path}: {asd.size} values, not {expected['num_freq_bins']}")
    if not np.isfinite(asd).all():
        raise ValueError(f"{path}: {np.sum(~np.isfinite(asd))} values are not finite")


def download_fiducial(data_dir: Path, runs: list, detectors: list) -> Path:
    """
    Build or adopt the fiducial of ``runs`` for each detector; the runs' recolour banks must be on disk.

    Returns
    -------
    Path
        The fiducial's directory.
    """
    for detector in detectors:
        if not fiducial_file(fiducial_dir(data_dir, runs), detector).exists():
            build_fiducial(data_dir, runs, detector)
        check_fiducial(data_dir, runs, detector)
    return fiducial_dir(data_dir, runs)


def write_record(path: Path, numbers: dict, detectors: dict) -> Path:
    """
    Write fixed numbers and each detector's summary to a record beside the data.

    Detectors recorded earlier are kept.

    Raises
    ------
    ValueError
        If an existing record holds other numbers.
    """
    record = yaml.safe_load(path.read_text()) if path.exists() else {**numbers, "detectors": {}}
    if {key: record.get(key) for key in numbers} != numbers:
        raise ValueError(f"{path} records numbers other than this Sage's: {record}")
    for detector, entry in detectors.items():
        record["detectors"].setdefault(detector, {}).update(entry)
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    return path


def download_run(data_dir: Path, run: str, detectors: list, workers: int, asds: bool, bank: bool) -> dict:
    """
    Download, resume or adopt each detector's strain of an observing run, check it and save its
    data-quality time beside it; build or adopt its per-segment ASDs and recolour bank if asked;
    record both.

    Parameters
    ----------
    data_dir : Path
        Directory from the machine file.
    run : str
        Observing run, e.g. ``"o3b"``.
    detectors : list of str
        Detectors to download.
    workers : int
        Parallel downloads.
    asds : bool
        Also build or adopt the per-segment ASDs.
    bank : bool
        Also build or adopt the recolour bank; an adopted bank's seed is recorded as unknown.

    Returns
    -------
    dict of str to dict
        ``check_strain`` summary per detector.
    """
    for detector in detectors:
        path = strain_file(data_dir, run, detector)
        if not path.exists():
            fetch_detector(data_dir, run, detector, workers)
        elif json.loads(failed_file(path).read_text()):
            retry_pieces(data_dir, run, detector, workers)

    summaries = {detector: check_strain(data_dir, run, detector) for detector in detectors}
    for detector in detectors:
        path = strain_file(data_dir, run, detector)
        if not bad_time_file(path).exists():
            write_bad_time(path, gwosc_name(data_dir, run), detector)
        read_bad_time(path)
    write_record(dataset_file(data_dir, run), DATASET, summaries)

    record = asds_file(data_dir, run)
    recorded = yaml.safe_load(record.read_text())["detectors"] if record.exists() else {}
    entries = {detector: {} for detector in detectors}
    asd_dir = observing_run_dir(data_dir, run, "asds")
    for detector in detectors:
        if asds:
            if not segment_asd_file(asd_dir, detector).exists():
                build_segment_asds(data_dir, run, detector)
            entries[detector]["segment_asds"] = check_segment_asds(data_dir, run, detector)
        if bank:
            seed = recorded.get(detector, {}).get("bank_seed")
            if not recolour_bank_file(asd_dir, detector).exists():
                build_bank(data_dir, run, detector)
                seed = BANK_SEED
            check_bank(data_dir, run, detector)
            entries[detector]["bank_seed"] = seed
    if asds or bank:
        write_record(record, ASDS, entries)
    return summaries
