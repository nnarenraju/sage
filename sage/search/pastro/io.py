#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : io.py
Description   : Products and contract enforcement for the p_astro stage.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The background used for the noise density is recorded with the products and asserted to
match the one behind the false-alarm rates. If the two differed, a candidate's
significance and its probability would rest on different noise models and could disagree
precisely where it matters most.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


class ContractViolation(ValueError):
    """Raised when an input breaks a documented precondition of the stage."""


def require_clustered(table) -> None:
    """
    Refuse a trigger set that has not been clustered.

    The mixture likelihood treats triggers as independent draws. An unclustered glitch
    supplies one draw per analysis window instead of one, so every inferred rate is
    multiplied by the number of windows per event -- and nothing downstream looks wrong,
    because a rate that is too large by a constant produces probabilities that are merely
    too confident.
    """
    attrs = getattr(table, "attrs", {}) or {}
    if "clustered" not in attrs:
        raise ContractViolation(
            "this trigger set does not declare whether it was clustered; the mixture "
            "likelihood is only valid on clustered triggers and a default would be "
            "believed rather than checked"
        )
    if not bool(attrs["clustered"]):
        raise ContractViolation(
            "refusing an unclustered trigger set: each glitch would contribute one draw "
            "per analysis window rather than one event, multiplying every inferred rate "
            "by the number of windows per event"
        )


def require_matching_background(
    pastro_attrs: Dict[str, object], far_attrs: Dict[str, object]
) -> None:
    """
    Assert the noise density and the false-alarm rates share a background.

    If the two rested on different backgrounds, a candidate's FAR and its p_astro would
    describe different noise, and they would disagree precisely where it matters -- at the
    top of the list, where the two backgrounds differ most.

    Compared on the fields that identify a background rather than on the whole attribute
    block: the removal mode, the livetime and the provenance hash. Two products of the
    same background written by different stages differ in incidental fields, and requiring
    those to match would fail on nothing.
    """
    keys = ("spec_hash", "observing_run", "removal", "background_livetime_s")
    missing = [
        key
        for key in keys
        if key not in (pastro_attrs or {}) or key not in (far_attrs or {})
    ]
    if missing:
        raise ContractViolation(
            f"cannot compare the two backgrounds: {sorted(set(missing))} is absent from "
            "one of the attribute blocks, so agreement could not be established either way"
        )
    disagree = {
        key: (pastro_attrs[key], far_attrs[key])
        for key in keys
        if pastro_attrs[key] != far_attrs[key]
    }
    if disagree:
        raise ContractViolation(
            f"the p_astro noise model and the FAR curve describe different backgrounds: "
            f"{disagree}. A candidate's rate and its probability would then rest on "
            "different noise and could disagree at the top of the list"
        )


def _density_nodes(nodes, density):
    """
    Support grid, plus a node either side of a blended density's join.

    One ULP apart, so linear interpolation between them is a step to within a rounding
    error rather than a ramp across the nearest grid spacing.
    """
    import numpy as np

    join = getattr(density, "join", None)
    if join is None or getattr(density, "body_empty", False):
        return np.asarray(nodes, dtype=np.float64)
    join = float(join)
    if not (nodes[0] < join < nodes[-1]):
        return np.asarray(nodes, dtype=np.float64)
    extra = [np.nextafter(join, -np.inf), np.nextafter(join, np.inf)]
    return np.unique(np.concatenate([np.asarray(nodes, dtype=np.float64), extra]))


def save_model(
    path: str | Path,
    densities: Dict[str, object],
    support,
    posterior,
    validation,
    attrs: Optional[Dict[str, object]] = None,
) -> Path:
    """
    Write the fitted model, its support and its validation record.

    The densities are stored as their values on a grid rather than as their samples. A
    reader then evaluates exactly what was fitted, without needing the injection set and
    the background that produced them, and without a bandwidth rule changing what a
    persisted model means.

    **The grid is per density, and carries the join.** A tail-blended noise density is
    discontinuous where the kernel estimate hands over to the fitted tail -- mass
    anchoring does not force continuity, and ``step`` measures the jump. Sampled on the
    plain support grid and interpolated back, that step becomes a ramp across whichever
    two nodes happen to straddle the join, and the reloaded density is not the one that
    was validated. Two extra nodes one ULP either side of the join are stored, so linear
    interpolation reproduces the discontinuity rather than smoothing it.

    Written under ``atomic_h5``, so a kill mid-write leaves the previous model rather
    than a truncated file that would load as a different one.
    """
    import numpy as np
    from sage.utils.atomic_io import atomic_h5

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    nodes = support.grid()[0]
    with atomic_h5(target, mode="w") as handle:
        for key, value in (attrs or {}).items():
            handle.attrs[key] = value
        handle.attrs["categories"] = list(posterior.categories)
        handle.attrs["prior"] = str(posterior.prior)
        handle.attrs["n_triggers"] = int(posterior.n_triggers)
        group = handle.create_group("support")
        for field in (
            "stat_lo", "stat_hi", "n_stat", "threshold_far_per_day", "threshold_stat",
        ):
            group.attrs[field] = getattr(support, field)
        # Optional in the support and therefore optional here: a model fitted on the
        # statistic alone has no chirp-mass axis, and writing zeros for the absent fields
        # would make a one-dimensional model reload as a degenerate two-dimensional one.
        for field in ("mchirp_lo", "mchirp_hi", "n_mchirp"):
            value = getattr(support, field, None)
            if value is not None:
                group.attrs[field] = value
        group.create_dataset("stat", data=np.asarray(nodes, dtype=np.float64))
        densities_group = handle.create_group("densities")
        for name, density in densities.items():
            entry = densities_group.create_group(name)
            grid = _density_nodes(nodes, density)
            entry.create_dataset("stat", data=np.asarray(grid, dtype=np.float64))
            entry.create_dataset(
                "log_prob", data=np.asarray(density.log_prob(grid), dtype=np.float64)
            )
            # Provenance for the blend, so a reader can see where the handover was and
            # how large the discontinuity is without re-deriving either.
            for field in ("join", "tail_mass", "step"):
                value = getattr(density, field, None)
                if value is not None:
                    entry.attrs[field] = float(value)
        rates = handle.create_group("posterior")
        rates.create_dataset("total_grid", data=posterior.total_grid)
        rates.create_dataset("fraction_grid", data=posterior.fraction_grid)
        rates.create_dataset(
            "log_posterior", data=posterior.log_posterior, compression="gzip"
        )
        if validation is not None:
            import json

            handle.attrs["validation"] = json.dumps(validation.as_dict())
    return target


def load_model(path: str | Path):
    """
    Read a persisted model.

    Returns a mapping of everything the file holds. The densities come back as
    :class:`GriddedDensity`, which interpolates the stored log values -- the same numbers
    the fit produced, so a reloaded model cannot drift from the one that was validated.
    """
    import json

    import h5py
    import numpy as np

    from sage.search.pastro.support import CommonSupport

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no p_astro model at {target}")
    with h5py.File(target, "r") as handle:
        group = handle["support"]
        optional = {
            field: group.attrs[field]
            for field in ("mchirp_lo", "mchirp_hi", "n_mchirp")
            if field in group.attrs
        }
        if "n_mchirp" in optional:
            optional["n_mchirp"] = int(optional["n_mchirp"])
        for field in ("mchirp_lo", "mchirp_hi"):
            if field in optional:
                optional[field] = float(optional[field])
        support = CommonSupport(
            **optional,
            stat_lo=float(group.attrs["stat_lo"]),
            stat_hi=float(group.attrs["stat_hi"]),
            n_stat=int(group.attrs["n_stat"]),
            threshold_far_per_day=float(group.attrs["threshold_far_per_day"]),
            threshold_stat=float(group.attrs["threshold_stat"]),
        )
        nodes = np.asarray(group["stat"])
        densities = {
            name: GriddedDensity(
                np.asarray(entry["stat"]),
                np.asarray(entry["log_prob"]),
                support,
                join=(
                    float(entry.attrs["join"]) if "join" in entry.attrs else None
                ),
                tail_mass=(
                    float(entry.attrs["tail_mass"])
                    if "tail_mass" in entry.attrs
                    else None
                ),
                step=(
                    float(entry.attrs["step"]) if "step" in entry.attrs else None
                ),
            )
            for name, entry in handle["densities"].items()
        }
        rates = handle["posterior"]
        stored = {
            "support": support,
            "densities": densities,
            "categories": tuple(
                value.decode() if isinstance(value, bytes) else str(value)
                for value in handle.attrs["categories"]
            ),
            "prior": str(handle.attrs["prior"]),
            "n_triggers": int(handle.attrs["n_triggers"]),
            "total_grid": np.asarray(rates["total_grid"]),
            "fraction_grid": np.asarray(rates["fraction_grid"]),
            "log_posterior": np.asarray(rates["log_posterior"]),
        }
        if "validation" in handle.attrs:
            stored["validation"] = json.loads(handle.attrs["validation"])
    return stored


@dataclass
class GriddedDensity:
    """
    A density stored as its log values on a grid.

    Interpolated linearly in the log, which keeps it positive everywhere and reproduces
    the fitted values exactly at the nodes. Outside the support it is ``-inf``, as the
    fitted densities are: the model says nothing there and a persisted copy must not say
    more than the original did.

    ``stat`` is the grid the values were written on, not necessarily the support grid: a
    blended density carries two extra nodes one ULP either side of its join so the
    handover survives the round trip as a step rather than a ramp. ``join``, ``tail_mass``
    and ``step`` are carried through for provenance and are ``None`` for a density with no
    tail.
    """

    stat: "object"
    log_values: "object"
    support: "object"
    join: "object" = None
    tail_mass: "object" = None
    step: "object" = None

    def log_prob(self, stat, mchirp=None):
        """Log density at the given points."""
        import numpy as np

        query = np.asarray(stat, dtype=np.float64)
        out = np.interp(query.ravel(), self.stat, self.log_values)
        out = np.where(self.support.contains(query.ravel()), out, -np.inf)
        return np.reshape(out, query.shape)

    def normalisation(self) -> float:
        """
        Integral over the common support.

        Taken on the *support* grid rather than on the stored one, so it is the same
        quadrature the density reported before it was written and the two numbers are
        comparable. Integrating over the augmented grid instead would change the answer
        by the treatment of two nodes one ULP apart.
        """
        import numpy as np

        nodes = self.support.grid()[0]
        return float(
            np.sum(np.exp(self.log_prob(nodes)) * self.support.cell_volume())
        )
