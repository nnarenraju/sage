"""Read strain for batches of analysis windows from a strain release.

Consecutive windows in the same segments are read as one stretch per
detector, which the caller unfolds into overlapping windows.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import h5py
import numpy as np
import torch
from pycbc import DYN_RANGE_FAC


def contiguous_runs(
    windows: Dict[str, np.ndarray], detectors: List[str], stride_samples: int
) -> List[Tuple[int, int]]:
    """Ranges of windows that are one stride apart in every detector.

    Parameters
    ----------
    windows : dict of str to numpy.ndarray
        Output of ``analysis_windows``.
    detectors : list of str
        Detectors to read.
    stride_samples : int
        Step between window starts in samples.

    Returns
    -------
    list of tuple of int
        Start and end window index of each run.
    """
    n_windows = len(windows["gps"])
    breaks = np.zeros(n_windows, dtype=bool)
    breaks[:1] = True

    for detector in detectors:
        segment = windows[f"{detector}_segment"]
        sample = windows[f"{detector}_sample"]
        breaks[1:] |= segment[1:] != segment[:-1]
        breaks[1:] |= np.diff(sample) != stride_samples

    starts = np.flatnonzero(breaks)
    ends = np.append(starts[1:], n_windows)

    return list(zip(starts.tolist(), ends.tolist()))


def window_batches(
    release_dir: Path,
    run: str,
    windows: Dict[str, np.ndarray],
    detectors: List[str],
    window_samples: int,
    stride_samples: int = 204,
    batch_size: int = 4096,
) -> Iterator[Tuple[slice, np.ndarray]]:
    """Strain for consecutive batches of windows.

    Each batch lies inside one segment of every detector, so each
    detector is read as one stretch. ``DYN_RANGE_FAC`` is divided out.

    Parameters
    ----------
    release_dir : Path
        Strain release directory.
    run : str
        Observing run.
    windows : dict of str to numpy.ndarray
        Output of ``analysis_windows``.
    detectors : list of str
        Detectors in the channel order the network expects.
    window_samples : int
        Window length in samples.
    stride_samples : int
        Step between window starts in samples.
    batch_size : int
        Maximum windows per batch.

    Yields
    ------
    windows : slice
        Indices of the windows in the batch.
    strain : numpy.ndarray
        Float32 strain of shape ``(detectors, samples)`` covering them.
    """
    files = {
        detector: h5py.File(release_dir / f"data_{detector}_{run}.h5", "r")
        for detector in detectors
    }
    
    try:
        for run_start, run_end in contiguous_runs(windows, detectors, stride_samples):
            for first in range(run_start, run_end, batch_size):
                last = min(first + batch_size, run_end)
                n_samples = window_samples + stride_samples * (last - first - 1)

                strain = np.empty((len(detectors), n_samples), dtype=np.float32)
                for row, detector in enumerate(detectors):
                    segment = windows[f"{detector}_segment"][first]
                    start = windows[f"{detector}_sample"][first]
                    dataset = files[detector][f"segments/{segment:06d}"]
                    strain[row] = dataset[start : start + n_samples]

                np.divide(strain, DYN_RANGE_FAC, out=strain, dtype=np.float32)
                yield slice(first, last), strain
    finally:
        for handle in files.values():
            handle.close()


class WindowStrain(torch.utils.data.Dataset):
    """Strain of individual windows, for reading scattered windows.

    Each worker process opens its own file handles on first use.

    Parameters
    ----------
    release_dir : Path
        Strain release directory.
    run : str
        Observing run.
    windows : dict of str to numpy.ndarray
        Windows in the format of ``analysis_windows``.
    detectors : list of str
        Detectors in the channel order the network expects.
    window_samples : int
        Window length in samples.
    """

    def __init__(
        self,
        release_dir: Path,
        run: str,
        windows: Dict[str, np.ndarray],
        detectors: List[str],
        window_samples: int,
    ) -> None:
        """Store what each read needs; files are opened later."""
        self.release_dir = release_dir
        self.run = run
        self.windows = windows
        self.detectors = detectors
        self.window_samples = window_samples
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
                detector: h5py.File(self.release_dir / f"data_{detector}_{self.run}.h5")
                for detector in self.detectors
            }

        strain = np.empty((len(self.detectors), self.window_samples), dtype=np.float32)
        for row, detector in enumerate(self.detectors):
            segment = self.windows[f"{detector}_segment"][index]
            start = self.windows[f"{detector}_sample"][index]
            dataset = self.files[detector][f"segments/{segment:06d}"]
            strain[row] = dataset[start : start + self.window_samples]

        np.divide(strain, DYN_RANGE_FAC, out=strain, dtype=np.float32)
        return strain


def scattered_window_batches(
    release_dir: Path,
    run: str,
    windows: Dict[str, np.ndarray],
    detectors: List[str],
    window_samples: int,
    batch_size: int = 4096,
    workers: int = 8,
) -> Iterator[Tuple[slice, torch.Tensor]]:
    """Strain for batches of windows, each window read on its own.

    For windows that are not one stride apart, such as random pairs.
    Reads run in parallel worker processes.

    Parameters
    ----------
    release_dir : Path
        Strain release directory.
    run : str
        Observing run.
    windows : dict of str to numpy.ndarray
        Windows in the format of ``analysis_windows``.
    detectors : list of str
        Detectors in the channel order the network expects.
    window_samples : int
        Window length in samples.
    batch_size : int
        Maximum windows per batch.
    workers : int
        Reading processes.

    Yields
    ------
    windows : slice
        Indices of the windows in the batch.
    strain : torch.Tensor
        Float32 strain of shape ``(windows, detectors, window_samples)``.
    """
    dataset = WindowStrain(release_dir, run, windows, detectors, window_samples)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=workers
    )

    first = 0
    for strain in loader:
        yield slice(first, first + len(strain)), strain
        first += len(strain)
