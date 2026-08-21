#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : gwtc3_powerlawpeak.py
Description   : GWTC-3 Power-Law + Peak hyperposterior, fetched and reduced.

Created on 2026-08-21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The population sgwc-1 draws its injections from: the Power-Law + Peak fit to O1+O2+O3,
published with *Population of Merging Compact Binaries Inferred Using Gravitational Waves
through GWTC-3* (Phys. Rev. X 13, 011048).

**Version 3 of the record, not version 1.** v1 ships one 10.1 GB archive holding every
model; v3 splits it per model, so the Power-Law + Peak fit is a 794 MB download instead.
The file inside is the same one ``injection_study.ipynb`` reads.

Injections are drawn by marginalising over the whole hyperposterior by default, so the
population's own uncertainty reaches ``p(x | signal)`` rather than being replaced by a
point. Where a single representative *is* wanted, this release publishes ``log_likelihood``
and the densest sample is therefore read rather than estimated; see :data:`METHODS` for
what that leaves and what was removed.

sgwc-1 conditioned on one sample chosen by marginal MAP, and recorded the answer --
sample 10677. That method was reproduced exactly during the port and then dropped: it
sits 2.16 nats below the likelihood maximum, and the population it names is roughly nine
times less likely than the one the data prefers.

Reading needs neither ``bilby`` nor ``gwpopulation``: a bilby result JSON stores its
posterior as a plain serialised frame, and this reads that directly. What it writes is
the canonical hyperposterior form described in :mod:`sage.search.sources`, so the search
itself never sees this release's layout.
"""

import json
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

#: The Zenodo record this reads. Concept DOI ``10.5281/zenodo.5655785`` resolves to
#: whatever is latest, which is precisely what a reproducible analysis must not follow.
RECORD = "https://zenodo.org/records/11254021"
DOI = "10.5281/zenodo.11254021"
VERSION = "v3"
ARCHIVE_URL = f"{RECORD}/files/analyses_PowerLawPeak.tar.gz"

#: The one file wanted out of the archive. Named in full because the archive holds the
#: leave-one-out refits and the waveform-systematics refits alongside it, and those are
#: different populations with confusingly similar names.
MEMBER = (
    "analyses/PowerLawPeak/"
    "o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json"
)

#: What :mod:`sage.search.injection.population` reads off a hyperposterior sample. The
#: canonical output carries exactly these; the release's other columns -- rates,
#: selection diagnostics, likelihoods -- are kept beside them but are not the population.
HYPERPARAMETERS: Tuple[str, ...] = (
    "alpha", "beta", "mmin", "mmax", "lam", "mpp", "sigpp", "delta_m",
    "mu_chi", "sigma_chi", "amax", "xi_spin", "sigma_spin", "lamb",
)

#: Bandwidth grid for the joint-MAP KDE, from the Thyme notebook (cell 18).
JOINT_BANDWIDTH_LOGSPACE = (-1.5, 0.7, 12)
CV_FOLDS = 5

#: Columns that record how the sampler ran rather than what the population is. Excluded
#: from a density estimate over the hyperposterior: they are deterministic functions of
#: the sample or of the analysis, so including them puts the sampler's bookkeeping into
#: the definition of "the densest point".
AUXILIARY_COLUMNS = frozenset({
    "log_likelihood", "log_prior", "log_noise_evidence", "log_evidence",
    "log_bayes_factor", "selection", "pdet_n_effective", "min_event_n_effective",
    "surveyed_hypervolume", "log_10_rate",
})

#: How a single representative sample is chosen, best first. Both return a real posterior
#: sample rather than a constructed point: the population models are evaluated at a
#: hyperparameter vector, and a vector assembled coordinate-by-coordinate need not be one
#: the posterior supports anywhere.
#:
#: Two methods were built from the Thyme notebook and then removed, because both are
#: strictly worse than what remains and keeping a bad default reachable is how it gets
#: used. ``joint_map_gaussian_kde`` estimates the density at Scott's bandwidth without
#: standardising, which Thyme's own notes record over-smoothing into the wrong mode in
#: three dimensions and which refuses outright at fourteen. ``marginal_map`` -- sgwc-1's
#: choice -- assembles a point out of per-parameter modes that the joint posterior need
#: not support, then finds the nearest sample to it by unstandardised Euclidean distance;
#: measured on this release it lands 2.16 nats below the maximum, and its answer is
#: driven by ``rate`` and ``surveyed_hypervolume`` rather than by the population (it
#: selected sample 10677, reproducing sgwc-1 exactly, and 8222 once the sampler's
#: bookkeeping columns were excluded).
METHODS = ("max_likelihood", "joint_map_kde")

#: Second choice, and the best available point. With a likelihood column in hand the
#: densest sample is known exactly, so estimating it with a kernel density is answering a
#: question that does not need asking -- and in fourteen dimensions, answering it badly.
#: First choice is not a point at all: see ``InjectionSpec.population_mode``.
DEFAULT_METHOD = "max_likelihood"


def fetch(dest, url: str = ARCHIVE_URL, refresh: bool = False) -> Path:
    """
    Download the release archive, streaming it to disk.

    Streamed rather than read into memory: the archive is ~794 MB, and a login node that
    happens to be busy is not the place to hold that twice over.

    An archive already on disk is returned untouched unless ``refresh`` is set. Zenodo
    records are immutable per version, so re-fetching one gains nothing and the record is
    pinned by version for exactly that reason.
    """
    import shutil
    from urllib.request import urlopen

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(url).name
    if target.is_file() and not refresh:
        return target

    # Through a temporary in the same directory, so an interrupted download cannot be
    # mistaken for a complete archive by the next run.
    partial = target.with_suffix(target.suffix + ".partial")
    with urlopen(url, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    partial.replace(target)
    return target


def extract(archive, dest, member: str = MEMBER) -> Path:
    """
    Pull the one result file out of the archive.

    Only the named member is extracted. The archive expands to 840 MB of models this does
    not use, and on a quota measured in gigabytes that is worth not spending.

    Members are checked against the destination before extraction: a tar entry may name
    an absolute or parent-relative path, and extracting one blindly writes outside the
    directory it was pointed at.
    """
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(member).name
    if target.is_file():
        return target

    with tarfile.open(archive, "r:gz") as handle:
        try:
            entry = handle.getmember(member)
        except KeyError:
            names = [n for n in handle.getnames() if n.endswith(".json")][:5]
            raise KeyError(
                f"{archive.name} holds no member {member!r}. The record's layout may "
                f"have changed; JSON members present include {names}"
            ) from None
        resolved = (dest / entry.name).resolve()
        if not resolved.is_relative_to(dest.resolve()):
            raise ValueError(
                f"archive member {entry.name!r} resolves outside {dest}, which an "
                "archive is not permitted to do"
            )
        extracted = handle.extractfile(entry)
        if extracted is None:
            raise ValueError(f"archive member {entry.name!r} is not a regular file")
        target.write_bytes(extracted.read())
    return target


def read_posterior(path):
    """
    The hyperposterior samples, as a frame, without ``bilby``.

    ``bilby.core.result.read_in_result`` is what sgwc-1 calls, but a bilby result JSON
    stores its posterior as ``{"__dataframe__": ..., "content": {column: values}}`` --
    plain JSON that reconstructs to the same frame. Reading it directly keeps a heavy
    and version-sensitive dependency out of the search: bilby 1.1.3 wrote this file and
    the current release is 2.6, which is a wide gap to ask a reader to span for a
    dictionary of lists.
    """
    import pandas as pd

    payload = json.loads(Path(path).read_text())
    posterior = payload.get("posterior")
    if posterior is None:
        raise KeyError(
            f"{Path(path).name} carries no 'posterior'; it is not a bilby result file"
        )
    content = posterior.get("content", posterior)
    frame = pd.DataFrame(content)
    missing = [name for name in HYPERPARAMETERS if name not in frame.columns]
    if missing:
        raise KeyError(
            f"{Path(path).name} is missing hyperparameters {missing}; the Power-Law + "
            f"Peak population cannot be drawn from it. Columns present: "
            f"{sorted(frame.columns)}"
        )
    return frame


def population_columns(samples) -> list:
    """
    The columns that define the population, with the sampler's bookkeeping dropped.

    A density estimate over the hyperposterior must run on the hyperparameters. Leaving
    ``rate``, ``surveyed_hypervolume`` and the likelihood columns in makes the "densest
    point" partly a statement about how the sampler ran, and their numerical scales --
    thousands, against spin widths of 0.03 -- dominate any distance or bandwidth.
    """
    return [
        c for c in samples.columns
        if c not in AUXILIARY_COLUMNS and samples[c].nunique() > 1
    ]


def select_max_likelihood(samples):
    """
    The posterior sample of highest likelihood.

    Thyme's ``get_true_joint_map`` (``hyperposterior_to_intrinsic_params.ipynb`` cell 11),
    and the default here. The hyperposterior carries its own ``log_likelihood``, so the
    densest sample is known rather than estimated -- no bandwidth, no dimensionality
    problem, no cross-validation, and reproducible exactly.

    Strictly this is the maximum-likelihood sample, and the maximum *a posteriori* one is
    the maximum of ``log_likelihood + log_prior``. Both are computed and the posterior one
    is returned; on the GWTC-3 release they coincide, because that run sampled under a
    prior which is constant over its support, so ``log_prior`` takes one value at every
    sample.
    """
    import numpy as np

    total = np.asarray(samples["log_likelihood"], dtype=np.float64)
    if "log_prior" in samples.columns:
        total = total + np.asarray(samples["log_prior"], dtype=np.float64)
    index = int(np.argmax(total))
    return samples.iloc[index], index


def select_joint_map_kde(
    samples,
    bandwidth_grid=None,
    cv_folds: int = CV_FOLDS,
    n_jobs=-1,
    columns=None,
):
    """
    The densest sample under a standardised, cross-validated kernel density estimate.

    Thyme's ``get_joint_map_as_pdseries`` (cell 18). Standardising first puts the
    bandwidth in units of each parameter's own spread, which is what makes one scalar
    bandwidth meaningful across parameters whose scales differ by four orders of
    magnitude; the bandwidth itself is chosen by cross-validated likelihood rather than a
    rule of thumb.

    The estimate is over the *joint* density, so it cannot return a point the posterior
    does not support. It is still a density estimate in as many dimensions as there are
    hyperparameters, and the notebook that introduced it says so: kernel density
    estimation degrades quickly with dimension whatever the bandwidth. The worst case --
    used only when the release publishes no likelihood to read.
    """
    import numpy as np
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if bandwidth_grid is None:
        bandwidth_grid = np.logspace(*JOINT_BANDWIDTH_LOGSPACE)
    columns = list(columns or population_columns(samples))
    matrix = np.asarray(samples[columns].values, dtype=np.float64)

    search = GridSearchCV(
        make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            KernelDensity(kernel="gaussian"),
        ),
        {"kerneldensity__bandwidth": list(bandwidth_grid)},
        cv=cv_folds,
        n_jobs=n_jobs,
    )
    search.fit(matrix)
    index = int(np.argmax(search.best_estimator_.score_samples(matrix)))
    return samples.iloc[index], index


#: Dispatch for :data:`METHODS`.
SELECTORS = {
    "max_likelihood": select_max_likelihood,
    "joint_map_kde": select_joint_map_kde,
}


def select(samples, method: str = DEFAULT_METHOD, **kwargs):
    """
    Pick one representative hyperposterior sample by the named method.

    Returns ``(sample, index)``. The index identifies the row in the release, which is
    what makes a selection quotable and checkable against another analysis.
    """
    if method not in SELECTORS:
        raise ValueError(
            f"unknown selection method {method!r}; known methods are {sorted(SELECTORS)}"
        )
    if method == "max_likelihood" and "log_likelihood" not in samples.columns:
        raise ValueError(
            "'max_likelihood' needs a log_likelihood column and this release has none; "
            "use 'joint_map_kde', which estimates the density instead of reading it"
        )
    return SELECTORS[method](samples, **kwargs)


def build(
    dest,
    archive_dir=None,
    url: str = ARCHIVE_URL,
    method: str = DEFAULT_METHOD,
    expect_index: Optional[int] = None,
    keep_archive: bool = True,
    store_population: bool = True,
) -> Path:
    """
    Fetch, select and write the canonical hyperposterior.

    Parameters
    ----------
    dest : path
        Where the canonical JSON is written. A campaign passes
        ``spec.path("injections", "hyperposterior_gwtc3_pp.json")``.
    archive_dir : path
        Where the release archive and the extracted result file are staged. Defaults to
        a ``sources`` directory beside ``dest``, so a campaign holds its own inputs.
    method : str
        One of :data:`METHODS`. Chooses the single representative sample; the whole
        posterior is written regardless, so a campaign can marginalise instead without
        re-fetching anything.
    expect_index : int or None
        Refuse to write unless the selection lands on this sample. For pinning a campaign
        to a checked answer; ``None`` by default, because a first build has nothing to
        check against.
    keep_archive : bool
        Keep the 794 MB tarball. Turned off once the result file is extracted and the
        archive is not wanted again.
    store_population : bool
        Write every hyperposterior sample alongside the representative. This is what
        makes marginalising over the hyperposterior possible offline, and it costs a few
        megabytes against a 794 MB archive that would otherwise have to be kept.

    Returns
    -------
    Path
        The canonical JSON: the fourteen hyperparameters of the representative sample,
        the whole selected sample, optionally the whole posterior, and the provenance to
        identify what produced them.
    """
    import hashlib

    from datetime import datetime, timezone

    import numpy as np

    dest = Path(dest)
    staging = Path(archive_dir) if archive_dir else dest.parent / "sources"

    archive = fetch(staging, url=url)
    result = extract(archive, staging)
    samples = read_posterior(result)

    sample, index = select(samples, method=method)

    if expect_index is not None and index != int(expect_index):
        raise ValueError(
            f"selection by {method!r} landed on sample {index}, not the {expect_index} "
            f"asked for. Either the release differs from the one that value was measured "
            f"against, or the selection has drifted -- both change the population "
            f"injections are drawn from, so neither is written silently"
        )

    columns = population_columns(samples)
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    payload = {
        "model": "PowerLawPeak",
        "observing_runs": ["O1", "O2", "O3"],
        "hyperparameters": {name: float(sample[name]) for name in HYPERPARAMETERS},
        "sample": {str(k): float(v) for k, v in sample.items()},
        "selection": {
            "method": method,
            "description": _METHOD_NOTES[method],
            "index": int(index),
            "n_samples": int(len(samples)),
            "log_likelihood": (
                float(sample["log_likelihood"])
                if "log_likelihood" in samples.columns
                else None
            ),
        },
        "source": {
            "record": RECORD,
            "doi": DOI,
            "version": VERSION,
            "archive_url": url,
            "member": MEMBER,
            "sha256": digest,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    if store_population:
        # The whole posterior, so marginalising needs neither the archive nor a network.
        # Column-major and named, because a campaign reading this must not depend on the
        # release's column order.
        payload["population"] = {
            "columns": list(HYPERPARAMETERS),
            "n_samples": int(len(samples)),
            "samples": [
                [float(v) for v in np.asarray(samples[name], dtype=np.float64)]
                for name in HYPERPARAMETERS
            ],
            "log_likelihood": [
                float(v)
                for v in np.asarray(samples["log_likelihood"], dtype=np.float64)
            ]
            if "log_likelihood" in samples.columns
            else None,
            "hyperparameter_columns": columns,
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=1))

    if not keep_archive:
        archive.unlink(missing_ok=True)
    return dest


#: One line per method, written into the canonical file so a campaign's own record says
#: what "the representative population" meant, without anyone reading this module.
_METHOD_NOTES = {
    "max_likelihood": (
        "posterior sample of highest log_likelihood + log_prior; read from the release "
        "rather than estimated"
    ),
    "joint_map_kde": (
        "densest sample under a standardised kernel density estimate with a "
        "cross-validated bandwidth"
    ),
}


def population(path):
    """
    Every hyperposterior sample, as a list of dicts, for marginalising over.

    Conditioning on one sample states a population the data merely prefers; drawing each
    injection under a different posterior sample propagates the uncertainty on the
    population into the injection set instead. Thyme's pipeline defaults to it for the
    same reason.

    Returns
    -------
    list of dict
        One mapping per posterior sample, each carrying the fourteen hyperparameters.
    """
    payload = json.loads(Path(path).read_text())
    stored = payload.get("population")
    if not stored:
        raise KeyError(
            f"{Path(path).name} holds no population block, so it can only be used at its "
            "representative sample. Rebuild it with store_population=True"
        )
    names = list(stored["columns"])
    columns = stored["samples"]
    return [
        {name: float(columns[j][i]) for j, name in enumerate(names)}
        for i in range(int(stored["n_samples"]))
    ]


def load(path) -> Dict[str, float]:
    """
    The fourteen hyperparameters, from a canonical file this module wrote.

    The one function the search calls. Kept separate from :func:`build` so that scoring a
    campaign needs neither the release nor a network -- only the small JSON that was
    written once and travels with the campaign.
    """
    payload = json.loads(Path(path).read_text())
    values = payload.get("hyperparameters")
    if not isinstance(values, dict):
        raise KeyError(
            f"{Path(path).name} carries no 'hyperparameters' mapping; it was not written "
            f"by {__name__}.build"
        )
    missing = [name for name in HYPERPARAMETERS if name not in values]
    if missing:
        raise KeyError(f"{Path(path).name} is missing hyperparameters {missing}")
    return {name: float(values[name]) for name in HYPERPARAMETERS}
