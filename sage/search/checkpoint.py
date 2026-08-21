#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : checkpoint.py
Description   : The checkpoint loader and stored-vs-live geometry validation.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Consolidates the three ad-hoc loaders in runs/o3b (eval_efficiency_snr.py,
benchmark_mlgwsc1.py, validate_checkpoint.py), which each re-implemented the
torch.compile prefix strip and none of which validated the stored config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Distinguishes "this key is absent" from "this key is present and None", which are
# different statements about a configuration and would otherwise compare equal.
_ABSENT = object()

# Architectures a checkpoint may name, as ``factory(cfg, data_cfg) -> nn.Module``. Keyed
# by the string a training config records, so a checkpoint identifies the class it was
# trained as rather than inheriting whichever class is current.
ARCHITECTURES: Dict[str, Any] = {}


def register_architecture(name: str, factory) -> None:
    """Register an architecture under the name a training config records."""
    if name in ARCHITECTURES:
        raise ValueError(f"architecture {name!r} is already registered")
    ARCHITECTURES[str(name)] = factory


GEOMETRY_KEYS: Tuple[str, ...] = (
    "sample_rate",
    "sample_length_in_s",
    "padding_length_in_s",
    "detectors",
    "norm_type",
    "do_point_estimate",
    "noise_low_frequency_cutoff",
    "signal_low_frequency_cutoff",
)




def _production_bbh(cfg, data_cfg):
    """
    The architecture every current Sage BBH checkpoint was trained as.

    Registered here rather than recorded in the checkpoint because the checkpoints that
    exist do not record it: ``cfg`` carries neither ``architecture`` nor ``model``. The
    training configs could record it in one line, but training is not this package's to
    change, so the search names the class it expects and
    :func:`validate_geometry` checks the weights actually fit it -- ``strict`` loading
    turns a wrong guess into a refusal rather than a partly-random network.

    Imported inside the factory, not at module scope, so that registering an architecture
    costs nothing until one is built.
    """
    from sage.architecture.network.networks import MSCNN1D_2DResNetCBAM_HardMining

    return MSCNN1D_2DResNetCBAM_HardMining(
        frontend_filters=int(getattr(cfg, "frontend_filters", 32)),
        frontend_kernel=int(getattr(cfg, "frontend_kernel", 64)),
        backend_resnet_size=int(getattr(cfg, "backend_resnet_size", 50)),
        norm_type=str(getattr(cfg, "norm_type", "instancenorm")),
        dropout=float(getattr(cfg, "dropout", 0.0)),
    )


#: Name a campaign selects when its checkpoint does not record one. Chosen by the spec,
#: not defaulted inside the loader: a search must state which class it is loading weights
#: into, so that the statement is in the provenance rather than in this file.
PRODUCTION_BBH: str = "mscnn1d_2dresnetcbam_hardmining"

register_architecture(PRODUCTION_BBH, _production_bbh)

@dataclass
class LoadedCheckpoint:
    """A checkpoint plus the flattened configs it was trained under."""

    path: Path
    sha256: str
    state_dict: Dict[str, Any]
    cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    epoch: int
    val_loss: float

    @property
    def norm_type(self) -> str:
        """
        Normalisation layer the weights were trained with.

        Read for reporting, never as the separability gate. Whether a detector's frontend
        output depends on that detector alone is a property of the module graph, and a
        string in a config survives a refactor that changes the graph;
        :func:`assert_separable` measures it instead.
        """
        return str(self.cfg.get("norm_type", ""))

    @property
    def detectors(self) -> Tuple[str, ...]:
        """
        Detector ordering baked into the weights.

        Ordering, not membership. Channel ``i`` of the input tensor is the detector at
        position ``i`` of this tuple, so a search reading the same detectors in a
        different order feeds each frontend the wrong strain -- and the result looks
        entirely ordinary, because every channel still holds real data.
        """
        stored = self.cfg.get("detectors") or self.data_cfg.get("detectors")
        if not stored:
            raise ValueError(
                f"{self.path} records no detector list, so the channel ordering the "
                "weights expect is unknown and cannot be reconstructed"
            )
        return tuple(str(name) for name in stored)

    def tc_prior(self) -> Tuple[float, float]:
        """
        Coalescence-time prior bounds recorded at training time.

        The window positions the network was taught to expect a merger in. The search's
        own ``tc`` bounds are taken from here rather than configured, so a decoded time
        cannot be mapped through a prior the weights never saw.
        """
        for source in (self.data_cfg, self.cfg):
            lower, upper = source.get("tc_lower_s"), source.get("tc_upper_s")
            if lower is not None and upper is not None:
                return float(lower), float(upper)
            low, high = source.get("tc_inject_lower"), source.get("tc_inject_upper")
            if low is not None and high is not None:
                return float(low), float(high)
        raise ValueError(
            f"{self.path} records no coalescence-time prior; the search cannot infer the "
            "window positions the network was trained to expect a merger in"
        )


def read_checkpoint(path: str | Path, map_location: str = "cpu") -> LoadedCheckpoint:
    """
    Load a ``.pt``, strip the ``_orig_mod.`` compile prefix and hash the file.

    Refuses a checkpoint whose ``cfg`` or ``data_cfg`` is anything but a mapping of
    primitives. Every checkpoint written before the flat-dict change pickled a *live
    config object*, and once that class was refactored away the file became unopenable --
    the failure surfaces as ``ModuleNotFoundError`` from inside pickle, naming a module
    that has nothing to do with the search. Checking the type here means the message names
    the checkpoint and says what is wrong with it.

    ``sha256`` is of the file, not of the weights, so it identifies the artefact a result
    was produced from even if two training runs happened to converge to the same tensors.

    Parameters
    ----------
    map_location : str
        Passed to ``torch.load``. Defaults to CPU so a checkpoint can be inspected on a
        login node with no GPU present.
    """
    import hashlib

    import torch

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no checkpoint at {target}")
    digest = hashlib.sha256()
    with open(target, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    try:
        payload = torch.load(target, map_location=map_location, weights_only=False)
    except Exception as error:  # noqa: BLE001 - re-raised with the file named
        raise ValueError(
            f"{target} could not be unpickled ({type(error).__name__}: {error}). A "
            "checkpoint that stored a live config object cannot be reopened once that "
            "class is refactored away; such files are unrecoverable without "
            "reconstructing the deleted class"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"{target} holds a {type(payload).__name__}, not a checkpoint dictionary"
        )
    for key in ("model_state_dict", "cfg", "data_cfg"):
        if key not in payload:
            raise ValueError(f"{target} carries no {key!r}; it is not a Sage checkpoint")
    for key in ("cfg", "data_cfg"):
        block = payload[key]
        if not isinstance(block, dict):
            raise ValueError(
                f"{target} stores {key!r} as a {type(block).__name__} rather than a dict. "
                "It was written before configurations were flattened, so it pickles a "
                "live config object whose class no longer exists; the file cannot be "
                "loaded without reconstructing that class"
            )
        bad = sorted(
            name
            for name, value in block.items()
            if not isinstance(
                value, (str, int, float, bool, list, tuple, dict, type(None))
            )
        )
        if bad:
            raise ValueError(
                f"{target} stores non-primitive values in {key!r} for {bad}; a "
                "configuration that cannot be written as data cannot be compared against "
                "the live one"
            )

    # torch.compile wraps the module, so every key gains this prefix. Left in place, the
    # weights simply fail to load into an uncompiled model, one name at a time.
    state = {
        (name[len("_orig_mod.") :] if name.startswith("_orig_mod.") else name): tensor
        for name, tensor in payload["model_state_dict"].items()
    }
    return LoadedCheckpoint(
        path=target,
        sha256=digest.hexdigest(),
        state_dict=state,
        cfg=dict(payload["cfg"]),
        data_cfg=dict(payload["data_cfg"]),
        epoch=int(payload.get("epoch", -1)),
        val_loss=float(payload.get("val_loss", float("nan"))),
    )


def validate_geometry(
    ckpt: LoadedCheckpoint,
    cfg,
    data_cfg,
    keys: Tuple[str, ...] = GEOMETRY_KEYS,
    strict: bool = True,
) -> List[str]:
    """
    Compare the stored config against the live one.

    Returns the list of mismatches; raises on any when ``strict``. The list is
    recorded in output provenance either way.

    Compared against the *stored* config, which is what the weights were trained under.
    A mismatch in any of :data:`GEOMETRY_KEYS` means the search is feeding the network a
    different input from the one it learned on -- a different sample rate, window length,
    detector set or low-frequency cutoff -- and the network still produces a number for
    every window. Nothing downstream can tell that number apart from a good one, which is
    why this is checked once, loudly, rather than trusted.

    ``strict=False`` is for inspection and for reporting a mismatch a caller has already
    decided to accept; the list is stamped into provenance either way, so a product built
    under a known mismatch says so.

    Values are compared after normalising sequences to tuples of strings: a config that
    stores ``detectors`` as a list and another that stores a tuple describe the same
    network, and a difference of container type is not a difference of geometry.
    """

    def _live(name: str):
        for source in (data_cfg, cfg):
            if source is None:
                continue
            if isinstance(source, dict):
                if name in source:
                    return source[name]
            elif hasattr(source, name):
                return getattr(source, name)
        return _ABSENT

    def _comparable(value):
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value)
        if isinstance(value, float):
            return float(value)
        return value

    mismatches: List[str] = []
    for name in keys:
        stored = ckpt.data_cfg.get(name, ckpt.cfg.get(name, _ABSENT))
        live = _live(name)
        if stored is _ABSENT and live is _ABSENT:
            continue
        if stored is _ABSENT or live is _ABSENT:
            missing = "the checkpoint" if stored is _ABSENT else "the live configuration"
            mismatches.append(f"{name}: absent from {missing}")
            continue
        if _comparable(stored) != _comparable(live):
            mismatches.append(f"{name}: checkpoint {stored!r} against live {live!r}")

    if mismatches and strict:
        raise ValueError(
            f"{ckpt.path} was trained under a different geometry from the one being run: "
            + "; ".join(mismatches)
            + ". The network will still produce a number for every window, and nothing "
            "downstream can tell it from a good one"
        )
    return mismatches


def as_config(block: Dict[str, Any]):
    """
    Wrap a flat config dict so the architecture can read it by attribute.

    The network reads ``cfg.detectors`` and ``getattr(cfg, "use_blurpool", True)``, while
    a checkpoint stores a flat dict. Wrapping rather than reconstructing the original
    class is deliberate: the class may have been refactored since, and the values that
    matter -- including every ``cached_property`` -- were already materialised into the
    dict when the checkpoint was written.

    ``dtype`` is rehydrated. A checkpoint has to store primitives -- ``read_checkpoint``
    refuses anything else, because a configuration that cannot be written as data cannot
    be compared against the live one -- so a ``torch.dtype`` is flattened to the string
    ``"torch.float32"`` on the way in. Verified on the production O3b checkpoint, which
    stores exactly that. Handed back as a string it reaches ``torch.tensor(dtype=...)``
    inside the parameter sampler, which raises there rather than here.
    """
    from types import SimpleNamespace

    values = dict(block)
    dtype = values.get("dtype")
    if isinstance(dtype, str):
        values["dtype"] = _as_torch_dtype(dtype)
    return SimpleNamespace(**values)


def _as_torch_dtype(name: str):
    """
    Resolve ``"torch.float32"`` or ``"float32"`` to the dtype object.

    Named rather than ``eval``-ed, and an unknown name is an error: a dtype silently
    defaulted to float32 would change the numerics of every window without changing
    anything that fails.
    """
    import torch

    resolved = getattr(torch, name.split(".")[-1], None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(
            f"the checkpoint records dtype {name!r}, which names no torch dtype; it "
            "cannot be resolved, and guessing one would silently change the numerics"
        )
    return resolved


def build_search_model(
    ckpt: LoadedCheckpoint,
    device: str,
    dtype: str = "float32",
    factory=None,
    architecture: str = "",
):
    """
    Instantiate the architecture from the stored config and load the weights.

    The architecture is **not guessed.** A checkpoint records the configuration it was
    trained under but not the class it was trained as, so either ``factory`` is supplied
    or ``cfg["architecture"]`` names an entry in :data:`ARCHITECTURES`. Defaulting to
    whichever architecture is current would load a year-old checkpoint into a refactored
    class, and a shape mismatch is the *lucky* outcome -- a class that changed only in
    behaviour loads cleanly and scores every window wrongly.

    The checkpoint's own configs are registered globally first, through
    ``sage.core.config.register_configs``, because the architecture reads them during
    construction. The registration is left in place: the model holds references to it and
    the processor built alongside reads it too.

    ``strict=True`` on the state-dict load. A missing or unexpected key means the weights
    do not belong to this class, and loading the subset that happens to match gives a
    partly-random network that still returns a ranking statistic for every window.

    Parameters
    ----------
    factory : callable, optional
        ``factory(cfg, data_cfg)`` returning an ``nn.Module``. Takes precedence over the
        registry.
    architecture : str, optional
        Registry name to use when the checkpoint records none, which every current Sage
        checkpoint is. It comes from the campaign spec so the choice is a stated part of
        the configuration -- and therefore of the hash and the provenance -- rather than a
        default buried here. The checkpoint's own record still wins where it has one.
    """
    import torch

    from sage.core.config import register_configs

    cfg = as_config(ckpt.cfg)
    data_cfg = as_config(ckpt.data_cfg)
    register_configs(cfg, data_cfg)

    if factory is not None:
        model = factory(cfg, data_cfg)
    else:
        name = (
            ckpt.cfg.get("architecture") or ckpt.cfg.get("model") or architecture
        )
        if not name:
            raise ValueError(
                f"{ckpt.path} does not record which architecture it was trained as, and "
                "no factory was supplied. Pass factory=..., or record 'architecture' in "
                f"the training config; known architectures are {sorted(ARCHITECTURES)}. "
                "Guessing would load these weights into whatever class is current, and a "
                "class that changed only in behaviour loads cleanly and scores wrongly"
            )
        if name not in ARCHITECTURES:
            import difflib

            close = difflib.get_close_matches(str(name), ARCHITECTURES, n=3, cutoff=0.4)
            hint = f"; did you mean {close}?" if close else ""
            raise ValueError(
                f"{ckpt.path} names architecture {name!r}, which is not registered. "
                f"Known: {sorted(ARCHITECTURES)}{hint}"
            )
        model = ARCHITECTURES[name](cfg, data_cfg)

    # strict=False tolerates absent and extra keys but still raises on a SHAPE
    # disagreement, so both routes have to be caught to give one message.
    try:
        loaded = model.load_state_dict(ckpt.state_dict, strict=False)
    except RuntimeError as error:
        raise ValueError(
            f"the weights in {ckpt.path} do not match {type(model).__name__}: {error}"
        ) from error
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError(
            f"the weights in {ckpt.path} do not match {type(model).__name__}: "
            f"missing {sorted(loaded.missing_keys)[:8]}, "
            f"unexpected {sorted(loaded.unexpected_keys)[:8]}. Loading only the keys that "
            "happen to match would leave a partly-random network that still returns a "
            "ranking statistic for every window"
        )
    model = model.to(device=device, dtype=getattr(torch, str(dtype)))
    model.eval()
    return model


def assert_separable(model, sample_input=None) -> None:
    """
    Refuse a model whose detectors couple, so the frontend cache cannot be misapplied.

    Delegates to :meth:`sage.search.network.SplitNetwork.separability`, which composes the
    per-detector path from the model's own submodules and checks that composition against
    ``model.forward`` before measuring anything. The measurement itself perturbs one
    detector and compares every other detector's frontend output bitwise.

    Called only by a campaign that intends to use the cache. The default search path
    rescores every slide from raw strain and needs no such property, which is why it works
    for any architecture.

    Parameters
    ----------
    sample_input : tensor, optional
        Input to probe with, shaped ``(B, D, L)``. Built from the model's own detector
        count when omitted, at the shortest length that model accepts. Deterministic
        random values rather than zeros or a constant: a constant input has zero variance,
        so a variance-normalising layer maps it to a degenerate output dominated by its
        epsilon and a coupling that only appears on structured input would be missed.

    Raises
    ------
    ValueError
        The per-detector path cannot be isolated, the split does not reproduce
        ``forward``, or perturbing one detector changed another's frontend output. The
        message names the pair, since which pair couples says where in the graph the
        coupling is.
    """
    from sage.search.network import SplitNetwork

    try:
        split = SplitNetwork(model, verify=True, sample_input=sample_input)
    except AttributeError as error:
        raise ValueError(str(error)) from error
    report = split.separability(sample_input)
    if not report.separable:
        perturbed, other = report.worst_pair
        raise ValueError(
            f"the frontend is not separable: perturbing detector {perturbed} changed "
            f"detector {other}'s frontend output by up to {report.worst_gap}. Every "
            "detector's features would then depend on what the others held, so a feature "
            "cached at one pairing is wrong at every other -- the background would be "
            "subtly unlike the zero-lag it is compared against, in a way no check made on "
            "the numbers alone would find. Run with use_frontend_cache=False, which "
            "rescores every slide and needs no such property"
        )


def load_search_model(
    path: str | Path,
    cfg,
    data_cfg,
    device: str = "cuda",
    strict: bool = True,
    require_separable: bool = False,
    factory=None,
    architecture: str = "",
) -> Tuple[Any, LoadedCheckpoint]:
    """
    Read, validate, build and (optionally) prove separability in one call.

    The order is the point: geometry is compared before the weights are built, and
    separability proved before anything is scored. Doing it the other way round would
    spend a GPU-hour discovering that the configuration disagrees.

    ``require_separable`` must be set by any caller that intends to use the frontend
    feature cache, and only by those. The cache reuses a detector's features across every
    slide, so on a model that couples detectors it produces a background subtly unlike the
    zero-lag it is compared against -- the kind of error that survives every check made of
    the numbers alone.

    Returns
    -------
    tuple
        ``(model, checkpoint)``. The checkpoint travels with the model so provenance can
        record the digest the result was produced under.
    """
    ckpt = read_checkpoint(path, map_location="cpu" if device == "cpu" else device)
    if cfg is not None or data_cfg is not None:
        # Comparing the checkpoint against a live configuration is the point of the
        # check, so a caller with no live configuration to compare against skips it
        # rather than comparing the checkpoint with itself and calling that agreement.
        validate_geometry(ckpt, cfg, data_cfg, strict=strict)
    model = build_search_model(
        ckpt, device=device, factory=factory, architecture=architecture
    )
    if require_separable:
        assert_separable(model)
    return model, ckpt
