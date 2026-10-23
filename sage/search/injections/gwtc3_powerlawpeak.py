"""GWTC-3 Power-Law + Peak population hyperposterior.

Downloads the O1-O3 Power-Law + Peak result (Phys. Rev. X 13, 011048)
from Zenodo, reads its samples and picks representative samples.
"""

import json
import shutil
import tarfile
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Version 3 of the record, which ships each population model separately.
_ARCHIVE_URL = (
    "https://zenodo.org/records/11254021/files/analyses_PowerLawPeak.tar.gz"
)


def download_hyperposterior(
    directory: Path,
    url: str = _ARCHIVE_URL,
    member: str = (
        "analyses/PowerLawPeak/"
        "o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json"
    ),
) -> Path:
    """Download the release archive and extract the Power-Law + Peak result.

    Files already in ``directory`` are reused.

    Parameters
    ----------
    directory : Path
        Directory for the archive and the extracted result.
    url : str
        Release archive URL.
    member : str
        Path of the result file inside the archive.

    Returns
    -------
    Path
        Extracted bilby result file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / Path(url).name
    if not archive.is_file():
        partial = archive.with_suffix(".partial")
        with urlopen(url, timeout=120) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
        partial.replace(archive)

    result = directory / Path(member).name
    if not result.is_file():
        with tarfile.open(archive, "r:gz") as handle:
            result.write_bytes(handle.extractfile(member).read())
    return result


def read_hyperposterior(path: Path) -> pd.DataFrame:
    """Hyperposterior samples from a bilby result file, without bilby.

    Parameters
    ----------
    path : Path
        Bilby result file.

    Returns
    -------
    pandas.DataFrame
        One row per hyperposterior sample.
    """
    posterior = json.loads(path.read_text())["posterior"]
    return pd.DataFrame(posterior.get("content", posterior))


def max_likelihood_index(samples: pd.DataFrame) -> int:
    """Index of the sample with the highest log likelihood.

    Parameters
    ----------
    samples : pandas.DataFrame
        Output of ``read_hyperposterior``.

    Returns
    -------
    int
        Row of the maximum-likelihood sample.
    """
    return int(np.argmax(samples["log_likelihood"].to_numpy()))


def joint_map_index(
    samples: pd.DataFrame,
    bandwidths: Sequence[float] = tuple(np.logspace(-1.5, 0.7, 12)),
    folds: int = 5,
    auxiliary: Sequence[str] = (
        "log_likelihood",
        "log_prior",
        "log_noise_evidence",
        "log_evidence",
        "log_bayes_factor",
        "selection",
        "pdet_n_effective",
        "min_event_n_effective",
        "surveyed_hypervolume",
        "log_10_rate",
    ),
) -> int:
    """Index of the densest sample under a kernel density estimate.

    The hyperparameters are standardised and the Gaussian kernel width is
    chosen by cross-validated likelihood.

    Parameters
    ----------
    samples : pandas.DataFrame
        Output of ``read_hyperposterior``.
    bandwidths : sequence of float
        Kernel widths tried, in standardised units.
    folds : int
        Cross-validation folds.
    auxiliary : sequence of str
        Sampler bookkeeping columns left out of the density.

    Returns
    -------
    int
        Row of the densest sample.
    """
    columns = [
        name
        for name in samples.columns
        if name not in auxiliary and samples[name].nunique() > 1
    ]
    values = samples[columns].to_numpy(dtype=np.float64)

    search = GridSearchCV(
        make_pipeline(StandardScaler(), KernelDensity(kernel="gaussian")),
        {"kerneldensity__bandwidth": list(bandwidths)},
        cv=folds,
        n_jobs=-1,
    )
    search.fit(values)
    return int(np.argmax(search.best_estimator_.score_samples(values)))
