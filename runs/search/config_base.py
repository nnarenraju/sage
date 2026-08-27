#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : config_base.py
Description   : Shared search-campaign configuration.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One campaign covers one observing run. Whitening is the network's input normalisation, so
the fiducial spectra must belong to the run being searched; background, false-alarm rates
and the noise density are likewise per-run. Cross-run figures are assembled afterwards
from the per-run products.

Training configuration comes from the run that produced the checkpoint, imported through
the usual mechanism so that the search reproduces the geometry the network was trained
with. Search-only settings are applied on top without touching the training run.
"""

import hashlib
from pathlib import Path
from typing import Optional, Sequence

SEARCH_ROOT = Path("/work/nagarajan/sage_runs/search")

#: Search-grade strain per observing run.
#:
#: **Not the training releases.** Those were built with ``prune_segments(rm_allevents=
#: True)``, which excises every published event and a 30 s guard either side. A search
#: against one recovers nothing and raises no error -- the events are simply not in the
#: data, and the livetime cost of their removal is 0.027%, far too small to notice. The
#: search-grade releases keep the events, apply the category-1 and transient-injection
#: vetoes, and store one HDF5 dataset per segment rather than one flat ``.bin``.
#:
#: A run with no search-grade release yet is absent rather than pointed at its training
#: release, so ``make_spec`` refuses it by name instead of silently searching data with
#: the answers cut out.
def _search_release(run: str) -> Path:
    """
    Where the search-grade release for one observing run lives.

    Derived from the same naming :mod:`sage.search.dataprep` writes it under, rather than
    listed here. A hardcoded table has to be edited every time a run is built, and the
    edit is invisible until a campaign refuses -- which is a manual step standing between
    a finished download and a runnable campaign, for no information the code does not
    already have.

    Always the three-detector release. One HLV build serves every arm of a run: an HL
    search reads H1 and L1 out of it and ignores V1, so a two-detector arm needs no
    release of its own.
    """
    from sage.search.dataprep import SearchDataSpec

    return SearchDataSpec(
        observing_run=run, detectors=("H1", "L1", "V1"), dq_flag="DATA"
    ).release_dir()


#: Search-grade releases, keyed by observing run. Present here means *built*: the value is
#: derived, so what this records is which releases exist on disk, and a run whose download
#: has not finished is absent for exactly that reason.
RELEASE_DIRS = {
    run: path
    for run in ("O3a", "O3b", "O4a", "O4b")
    for path in (_search_release(run),)
    if path.is_dir()
}

#: Training releases, kept for provenance and for tooling that compares the two. Never
#: used as a search target; see :data:`RELEASE_DIRS`.
TRAINING_RELEASE_DIRS = {
    "O3a": Path("/work/nagarajan/data_release_o3a"),
    "O3b": Path("/work/nagarajan/data_release"),
    "O4a": Path("/work/nagarajan/data_release_o4a"),
    "O4b": Path("/work/nagarajan/data_release_o4b"),
}


#: Repository root, from this file's own location. Campaign paths are resolved against it
#: rather than against the working directory: a submit script embeds ``cd`` into its
#: sbatch wrap and array tasks do not share a working directory, so a relative path here
#: resolves differently in the job that writes a product and the job that reads it.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _sibling_gwconfig(training_config: str | Path) -> Path:
    """
    The ``gwconfig.yaml`` beside a training config module, as an absolute path.

    Every training run keeps its parameter prior next to its config, so the location is a
    convention rather than a guess. Anchored to :data:`REPO_ROOT` when the config module is
    given relatively, because the campaign is launched from ``runs/search`` and read back
    on a compute node: the relative form passed ``validate()`` and then failed at
    ``tc_prior()``, which is two stages later and nowhere near the cause.
    """
    path = Path(training_config)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return (path.parent / "gwconfig.yaml").resolve()


def make_spec(
    observing_run: str,
    checkpoint: str | Path,
    training_config: str,
    fiducial_dir: str | Path,
    gwconfig: str | Path = "",
    detectors: Sequence[str] = ("H1", "L1"),
    tag: Optional[str] = None,
    background_yr: float = 10.0,
    **overrides,
):
    """
    Build the search specification for one observing run.

    Parameters
    ----------
    observing_run : str
        Key into the release directories.
    checkpoint : path
        Trained weights; its stored configuration is validated against the live one.
    training_config : str
        Config module of the run that produced the checkpoint. Recorded on the spec for
        provenance, and the anchor for ``gwconfig``.
    gwconfig : path
        The parameter prior the network was trained under. Defaults to the
        ``gwconfig.yaml`` beside ``training_config``, which is where every training run
        keeps it. It carries two things the checkpoint does not record at all: the
        coalescence-time band, which fixes how a decoded ``tc`` becomes a GPS time, and
        the mass bounds, which fix the multirate binning and therefore what the network is
        fed.
    fiducial_dir : path
        Fiducial spectra for this observing run.
    background_yr : float
        How much background to make, in years. Not a slide count: a campaign's foreground
        depends on its run and its detector network, so a fixed count would give each
        search a different depth and their false-alarm rates would not be comparable.
        The count needed is derived from this and the measured foreground, and the plan
        records what was built.
    **overrides
        Applied last, onto the assembled :class:`~sage.search.spec.SearchSpec`. Top-level
        field names only, so a campaign can pin one thing without restating the rest. A
        dict given for a sub-specification updates its named fields and leaves the rest,
        rather than replacing the object.
        ``data`` is refused: the release directory is the invariant this function exists
        to hold, and an override replacing the whole ``DataSpec`` would walk straight
        past it.

    Returns
    -------
    SearchSpec
        Frozen configuration, ready for :func:`sage.search.stages.run_stage`.

    Notes
    -----
    The release directory is looked up through :data:`RELEASE_DIRS` rather than taken from
    the caller, so two campaigns over the same run cannot silently read different strain.
    An unknown run raises here rather than producing a spec that points nowhere.

    The assembled spec is validated before it is returned. A campaign is launched by a
    submit script that reads the config and queues an array; a spec that fails validation
    at the first array task has already cost a scheduling round trip, and every task fails
    the same way.
    """
    import dataclasses

    from sage.search.spec import (
        DataSpec,
        EngineSpec,
        GeometrySpec,
        SearchSpec,
        SlideSpec,
    )

    run = str(observing_run)
    if run not in RELEASE_DIRS:
        known = f"search-grade releases are registered for {sorted(RELEASE_DIRS)}"
        if run in TRAINING_RELEASE_DIRS:
            raise ValueError(
                f"{run} has a training release but no search-grade one, and the two are "
                "not interchangeable: the training release has every published event "
                "excised, so a search against it recovers nothing and reports no error. "
                f"Build the search-grade release first. {known}"
            )
        raise ValueError(f"unknown observing run {run!r}; {known}")
    if isinstance(detectors, str):
        raise TypeError(
            f"detectors must be a sequence of names, got the string {detectors!r}; a "
            "string is a sequence of characters, so 'H1L1' would build the four-detector "
            "network ('H', '1', 'L', '1')"
        )
    detectors = tuple(str(d) for d in detectors)
    name = tag or f"{run.lower()}_{''.join(d[0] for d in detectors)}"
    spec = SearchSpec(
        tag=name,
        out_dir=SEARCH_ROOT / name,
        data=DataSpec(
            observing_run=run,
            detectors=detectors,
            release_dir=RELEASE_DIRS[run],
            fiducial_dir=Path(fiducial_dir),
        ),
        engine=EngineSpec(
            checkpoint=Path(checkpoint),
            training_config=str(training_config),
            gwconfig=Path(gwconfig or _sibling_gwconfig(training_config)),
        ),
        # Taken from the checkpoint, which records the geometry the weights were trained
        # under. Stating it here instead would let a search run a window the network was
        # never trained on, and nothing downstream would notice.
        geometry=GeometrySpec(tc_source="checkpoint"),
        # Reference detector from the network, not a fixed "H1". Slides are measured
        # relative to it and it must be in the network, so an LV campaign built on the
        # old default failed at construction -- correctly, but it meant every two-detector
        # arm without Hanford needed the field stated by hand, and stating it by hand is
        # how it comes to disagree with the network it belongs to.
        slides=SlideSpec(
            target_background_yr=float(background_yr),
            reference_detector=detectors[0],
        ),
    )
    data_overrides = overrides.pop("data", None)
    if isinstance(data_overrides, dict):
        # Field-level tuning of the DataSpec, which the whole-object override below
        # forbids: the release directory stays the one looked up from the observing run,
        # so a campaign can state its flag policy without being able to redirect its
        # strain.
        forbidden = sorted(set(data_overrides) & {"release_dir", "observing_run"})
        if forbidden:
            raise ValueError(
                f"data fields {forbidden} cannot be overridden; the release directory is "
                "looked up from the observing run so two campaigns over one run cannot "
                "read different strain"
            )
        replacement = data_overrides.get("detectors")
        if replacement is not None and tuple(replacement) != detectors:
            # The default tag and out_dir are built from the detector network, so an
            # override that changes it leaves an HLV campaign living in a directory
            # called o3a_HL -- and a second campaign that states HLV up front lands in
            # the same one. Naming it explicitly is the whole fix.
            raise ValueError(
                f"data override sets detectors to {tuple(replacement)} while the "
                f"campaign was built for {detectors}. The tag and out_dir are derived "
                "from the network, so this would file one campaign under another's "
                "name; pass detectors= to make_spec instead"
            )
        spec = dataclasses.replace(
            spec, data=dataclasses.replace(spec.data, **data_overrides)
        )
    elif data_overrides is not None:
        overrides["data"] = data_overrides

    # A dict given for a sub-specification means "change these fields", not "replace the
    # whole object with a dict". Without this, `injection=dict(hyperposterior_path=...)`
    # set spec.injection to a plain dict and the campaign failed several stages later at
    # the first attribute access, with nothing pointing back here.
    for key, value in list(overrides.items()):
        current = getattr(spec, key, None)
        if isinstance(value, dict) and dataclasses.is_dataclass(current):
            unknown = sorted(
                set(value) - {f.name for f in dataclasses.fields(current)}
            )
            if unknown:
                raise ValueError(
                    f"unknown {key} fields {unknown}; {type(current).__name__} holds "
                    f"{sorted(f.name for f in dataclasses.fields(current))}"
                )
            overrides[key] = dataclasses.replace(current, **value)

    if overrides:
        if "data" in overrides:
            raise ValueError(
                "data cannot be overridden; the release directory is looked up from the "
                "observing run so that two campaigns over one run cannot read different "
                "strain, and replacing the whole DataSpec would defeat that. Override "
                "the campaign's other fields, or register the release in RELEASE_DIRS"
            )
        unknown = [
            key for key in overrides if key not in {f.name for f in dataclasses.fields(spec)}
        ]
        if unknown:
            raise ValueError(
                f"unknown specification fields {sorted(unknown)}; make_spec overrides "
                f"top-level fields only, of which there are "
                f"{sorted(f.name for f in dataclasses.fields(spec))}"
            )
        spec = dataclasses.replace(spec, **overrides)
    spec.validate()
    return spec


def register(spec):
    """
    Load the training configuration and apply the search overrides on top of it.

    Returns the live ``(cfg, data_cfg)`` pair the training run used, with the search-only
    fields shadowed. Two things need it: the modules that read configuration globally --
    :class:`~sage.dsp.whiten.FiducialWhitening` reads the fiducial directory and the padded
    length from there rather than from arguments -- and
    :func:`sage.search.checkpoint.validate_geometry`, which has nothing to compare the
    checkpoint against without a live configuration.

    The training config module is imported and its ``set_configs()`` called if it has one,
    which is the mechanism every training run already uses. Nothing in the training run is
    modified: the overrides are set on the per-process wrapper, so a live run's export
    directory is untouched.

    Returns
    -------
    tuple
        ``(cfg, data_cfg)``, also registered globally through
        :func:`sage.core.config.register_configs`.
    """
    import importlib.util
    import sys

    from sage.core.config import get_cfg, get_data_cfg, register_configs

    path = Path(spec.engine.training_config)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"no training config at {path}; it is what supplies the live configuration "
            "the checkpoint's own geometry is validated against"
        )

    name = "sage_search_training_%s" % hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"{path} could not be imported as a Python module")
    module = importlib.util.module_from_spec(module_spec)
    parent = str(path.resolve().parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    sys.modules[name] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        if added:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass

    setter = getattr(module, "set_configs", None)
    if setter is None:
        raise ValueError(
            f"{path} exposes no set_configs(), so the training configuration it describes "
            "cannot be registered; every runs/*/config_*.py defines one"
        )
    setter()
    cfg, data_cfg = get_cfg(), get_data_cfg()
    spec.apply_shadow_overrides(cfg, data_cfg)
    register_configs(cfg, data_cfg)
    return cfg, data_cfg


#: Which run's network searches which run.
#:
#: Out of domain by default, which is the whole point: a network validated on the run it
#: was trained on measures how well it fits its own noise, and the O3a and O3b noise
#: differ enough that the distinction is not academic. Searching O3a with the O3b-trained
#: weights, and O3b with the O3a-trained ones, keeps every reported sensitivity a
#: statement about generalisation.
#:
#: A campaign may override it with ``network_run=``, and must for a run with no
#: counterpart.
SEARCH_NETWORK = {"O3a": "O3b", "O3b": "O3a"}

#: Fiducial spectra per *network* run, not per searched run. Whitening follows the weights:
#: the network was trained against these buffers and a search that whitened against others
#: would present the model with a distribution it never saw. The O3 networks share the
#: combined O3a+O3b fiducial, which is what makes an O3b-trained network on O3a strain
#: coherent rather than a domain shift.
FIDUCIAL_DIRS = {
    "O3a": Path("/work/nagarajan/sage_runs/fiducial_psds_o3ab"),
    "O3b": Path("/work/nagarajan/sage_runs/fiducial_psds_o3ab"),
}

#: Where the training runs export their weights.
SAGE_RUNS = Path("/work/nagarajan/sage_runs")


def search_spec(observing_run, detectors, network_run=None, **overrides):
    """
    One search campaign, with everything derivable derived.

    The thin-wrapper half of the pattern the training runs use: a per-arm config states
    the run and the network and nothing else, and every path that follows from those two
    is built here. What a per-arm config still states is what it has *earned* -- the
    frontend cache is a measured property of a particular set of weights, not a default.

    Derived: the checkpoint and its training config from the network run and the detector
    set, the fiducial spectra from the network run, the campaign tag and directory from
    the searched run and the detector set, the hyperposterior from the campaign directory,
    and the slide reference detector from the network.

    Parameters
    ----------
    observing_run : str
        The run being *searched*.
    network_run : str, optional
        The run whose network does the searching. Defaults to :data:`SEARCH_NETWORK`.
    """
    detectors = tuple(detectors)
    arm = "".join(d[0] for d in detectors)
    net = str(network_run or SEARCH_NETWORK.get(str(observing_run), ""))
    if not net:
        raise ValueError(
            f"no network run registered for a search of {observing_run}; SEARCH_NETWORK "
            f"holds {sorted(SEARCH_NETWORK)}. Pass network_run= to say which trained "
            "network should search it"
        )
    if net not in FIDUCIAL_DIRS:
        raise ValueError(
            f"no fiducial spectra registered for the {net} networks; FIDUCIAL_DIRS holds "
            f"{sorted(FIDUCIAL_DIRS)}. Whitening follows the weights, so this cannot be "
            "guessed from the run being searched"
        )
    name = str(overrides.pop("tag", None) or f"{str(observing_run).lower()}_{arm}")

    # CBC_CAT1 was measured against DATA over 600 ks of each run and coincides with it
    # exactly (100.0%), so requiring it would remove nothing and only assert a veto the
    # release did not apply. Stated rather than defaulted: the livetime must not claim a
    # vetoing it did not do.
    data = {"apply_cat1": False, **(overrides.pop("data", None) or {})}
    injection = {
        "hyperposterior_path": SEARCH_ROOT
        / name
        / "injections"
        / "hyperposterior_gwtc3_pp.json",
        **(overrides.pop("injection", None) or {}),
    }
    return make_spec(
        observing_run=observing_run,
        checkpoint=SAGE_RUNS / net.lower() / f"production_run_{arm}" / "CHECKPOINTS" / "best.pt",
        training_config=f"runs/{net.lower()}/config_{arm}.py",
        fiducial_dir=FIDUCIAL_DIRS[net],
        detectors=detectors,
        tag=name,
        data=data,
        injection=injection,
        **overrides,
    )
