"""Inject population signals into real noise and score them, as in sgwc-1.

Each signal is generated with PyCBC, added to a random pair of windows of
the searched run's training noise and scored; every score is kept.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from pycbc import DYN_RANGE_FAC
from pycbc.inject.inject import injection_func_map, projector
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform

from sage.data.noise.data_quality import Interval, clean
from sage.search.dataprep.windows import load_segments
from sage.search.model.score import score
from sage.search.timeslides.random_pairs import random_pairs
from sage.search.zerolag.triggers import concatenate, trigger_columns, write_triggers


class TrainingNoise(torch.utils.data.Dataset):
    """Noise windows from a training noise release, one flat ``.bin`` per detector.

    Known events were cut from the release when it was built.

    Parameters
    ----------
    noise_dir : Path
        Training noise release directory.
    run : str
        Observing run.
    segments : dict of str to list of dict
        Sidecar records per detector, in the network's channel order.
    windows : dict of str to numpy.ndarray
        Output of ``random_pairs`` on ``segments``.
    window_samples : int
        Window length in samples.
    """

    def __init__(
        self,
        noise_dir: Path,
        run: str,
        segments: Dict[str, List[dict]],
        windows: Dict[str, np.ndarray],
        window_samples: int,
    ) -> None:
        """Store what each read needs; files are mapped later."""
        self.noise_dir = noise_dir
        self.run = run
        self.detectors = list(segments)
        self.windows = windows
        self.window_samples = window_samples
        self.offsets = {
            detector: {r["segment_index"]: r["byte_offset"] // 4 for r in records}
            for detector, records in segments.items()
        }
        self.files = None

    def __len__(self) -> int:
        """Number of windows."""
        return len(self.windows["gps"])

    def __getitem__(self, index: int) -> np.ndarray:
        """Float32 strain of one window, shape ``(detectors, samples)``.

        Parameters
        ----------
        index : int
            Window index.

        Returns
        -------
        numpy.ndarray
            Strain with ``DYN_RANGE_FAC`` divided out.
        """
        if self.files is None:
            self.files = {
                detector: np.memmap(
                    self.noise_dir / f"data_{detector}_{self.run}.bin", "<f4", "r"
                )
                for detector in self.detectors
            }

        strain = np.empty((len(self.detectors), self.window_samples), dtype=np.float32)
        for row, detector in enumerate(self.detectors):
            segment = self.windows[f"{detector}_segment"][index]
            first = self.offsets[detector][segment]
            first += self.windows[f"{detector}_sample"][index]
            strain[row] = self.files[detector][first : first + self.window_samples]

        np.divide(strain, DYN_RANGE_FAC, out=strain, dtype=np.float32)
        return strain


class InjectionStrain(torch.utils.data.Dataset):
    """Noise windows of each injection with its signal added, by PyCBC.

    As ``pycbc.inject.CBCHDFInjectionSet.apply``: ``get_td_waveform``, the
    ``projector`` at ``injection_time``, and LAL's ``SimAddInjection`` into
    the window, which starts ``padding_s + tc`` before the merger.

    Parameters
    ----------
    noise : TrainingNoise
        Noise windows, one per injection.
    parameters : dict of str to numpy.ndarray
        Output of ``injection_parameters``.
    approximant : str
        PyCBC waveform model.
    f_lower, f_ref : float
        Lower and reference frequencies of the waveform, in Hz.
    padding_s : float
        Whitening padding at each end of a window, in seconds.
    sample_rate : float
        Sample rate of the strain.
    """

    def __init__(
        self,
        noise: TrainingNoise,
        parameters: Dict[str, np.ndarray],
        approximant: str,
        f_lower: float,
        f_ref: float,
        padding_s: float,
        sample_rate: float,
    ) -> None:
        """Store what each injection needs."""
        self.noise = noise
        self.parameters = parameters
        self.waveform = {
            "approximant": approximant,
            "f_lower": f_lower,
            "f_ref": f_ref,
            "delta_t": 1.0 / sample_rate,
        }
        self.padding_s = padding_s
        self.names = ["mass1", "mass2", "spin1x", "spin1y", "spin1z", "spin2x"]
        self.names += ["spin2y", "spin2z", "distance", "inclination", "coa_phase"]

    def __len__(self) -> int:
        """Number of injections."""
        return len(self.noise)

    def __getitem__(self, index: int) -> np.ndarray:
        """Float32 noise plus signal of one injection, ``(detectors, samples)``.

        Parameters
        ----------
        index : int
            Injection index.

        Returns
        -------
        numpy.ndarray
            Strain of every detector.
        """
        values = {name: column[index] for name, column in self.parameters.items()}
        hp, hc = get_td_waveform(
            **{name: values[name] for name in self.names}, **self.waveform
        )
        location = SimpleNamespace(
            tc=values["injection_time"],
            ra=values["ra"],
            dec=values["dec"],
            polarization=values["polarization"],
        )
        start = values["injection_time"] - self.padding_s - values["tc"]

        strain = self.noise[index].astype(np.float64)
        for row, detector in enumerate(self.noise.detectors):
            signal = projector(detector, location, hp.copy(), hc.copy())
            window = TimeSeries(
                strain[row], delta_t=self.waveform["delta_t"], epoch=start
            ).lal()
            injection_func_map[np.dtype(np.float64)](window, signal.lal(), None)
            strain[row] = window.data.data
        return strain.astype(np.float32)


def injection_study(
    network: Dict[str, object],
    noise_dir: Path,
    run: str,
    parameters: Dict[str, np.ndarray],
    output: Path,
    vetoes: Dict[str, List[Interval]],
    f_lower: float,
    f_ref: float,
    approximant: str = "IMRPhenomXPHM",
    min_separation_s: float = 100.0,
    seed: int = 0,
    batch_size: int = 4096,
    device: str = "cuda",
    workers: int = 8,
) -> None:
    """Score every injection in real noise and write one file.

    Row ``i`` of the file is injection ``i``: its parameters, the noise
    windows it was added to, and the network's score and estimates. Noise
    segments that touch bad time are not used, as in training.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    noise_dir : Path
        Training noise release of the searched run.
    run : str
        Observing run.
    parameters : dict of str to numpy.ndarray
        Output of ``injection_parameters``.
    output : Path
        File to write.
    vetoes : dict of str to list of tuple of float
        ``sage.data.noise.data_quality.bad_segments`` per detector.
    f_lower, f_ref : float
        Lower and reference frequencies of the waveforms, in Hz.
    approximant : str
        PyCBC waveform model.
    min_separation_s : float
        Smallest time between the noise windows of an injection.
    seed : int
        Seed of the noise windows.
    batch_size : int
        Maximum injections scored at once.
    device : str
        Device to score on.
    workers : int
        Processes reading strain and generating signals.
    """
    detectors = network["detectors"]
    segments = {}
    for detector in detectors:
        records = load_segments(noise_dir, run, detector)
        keep = clean(records, vetoes[detector])
        segments[detector] = [record for record, ok in zip(records, keep) if ok]

    n_injections = len(parameters["mass1"])
    noise = random_pairs(
        segments, n_injections, network["window_samples"], min_separation_s, seed
    )

    dataset = InjectionStrain(
        TrainingNoise(noise_dir, run, segments, noise, network["window_samples"]),
        parameters,
        approximant,
        f_lower,
        f_ref,
        network["padding_s"],
        segments[detectors[0]][0]["sample_rate"],
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=workers
    )

    kept = []
    first = 0
    for strain in loader:
        batch = slice(first, first + len(strain))
        first += len(strain)

        spectra = torch.fft.rfft(strain.to(device), dim=-1, norm="forward")
        statistic, estimates = score(network, spectra)
        columns = trigger_columns(
            estimates,
            noise["gps"][batch],
            network["targets"],
            network["offsets"],
            network["scales"],
            network["padding_s"],
        )
        columns["stat"] = statistic
        columns.update({name: values[batch] for name, values in noise.items()})
        kept.append(columns)

    columns = concatenate(kept)
    shared = set(columns) & set(parameters)
    if shared:
        raise ValueError(f"scores and parameters share the names {sorted(shared)}")
    columns.update(parameters)

    attrs = {
        **network["provenance"],
        "noise_dir": str(Path(noise_dir).resolve()),
        "run": run,
        "detectors": detectors,
        "n_injections": n_injections,
        "approximant": approximant,
        "f_lower": f_lower,
        "f_ref": f_ref,
        "min_separation_s": min_separation_s,
        "seed": seed,
    }
    write_triggers(output, columns, attrs)
