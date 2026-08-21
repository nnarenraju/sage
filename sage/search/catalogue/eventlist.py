#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : eventlist.py
Description   : A generic event list, however the times happen to be written down.

Created on 2026-08-20

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Every external comparison reduces to the same question: *did anyone else see something at
this time, and how significant did they call it?* Answering it needs a list of times and,
where the source publishes one, a significance. Nothing else about a source's file format
survives into the comparison.

So there is one input here and no per-source parsers. A published catalogue, a
subthreshold data product, an iDQ glitch list and a table read off a paper all arrive the
same way: as times, optionally with names and significances. Where a source needs
unpacking -- a Zenodo tarball, a trigger XML, whatever this year's layout is -- that is a
short wrapper written when the source is used, which produces times and calls in here.

**This is deliberate rather than lazy.** Every one of those sources changes layout between
releases: Zenodo restructures, XML tables gain columns, iDQ's archive is not promised to
stay as it is. A parser written against one release is a thing to maintain against all
future ones, and the part that would break is the part this analysis does not care about.
Keeping the format-specific code in throwaway wrappers means a layout change costs the
wrapper and not the comparison.

Times may be given as GPS, as a UTC string, or as an event name -- ``GW190412`` carries
its own UTC stamp. Converting by hand and comparing by eye is the step this replaces, and
it is also where the transcription errors were.
"""

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np

from sage.search.catalogue.record import CatalogueEvent, Conventions, ExternalCatalogue

#: Column spellings accepted for each quantity, lower-cased. A file whose header names
#: none of them for the time is rejected with the names it does hold, rather than being
#: read positionally -- a mis-read time column produces plausible times that are wrong.
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "gps": ("gps", "gps_time", "time", "t", "geocent_time", "tc", "utc", "date"),
    "name": ("name", "event", "event_name", "id", "label"),
    "far_per_yr": ("far_per_yr", "far", "far_yr", "false_alarm_rate"),
    "ifar_yr": ("ifar_yr", "ifar", "inverse_far"),
    "p_astro": ("p_astro", "pastro", "p_a", "probability"),
    "network_snr": ("network_snr", "snr", "rho"),
    "chirp_mass": ("chirp_mass", "mchirp", "mc"),
}


def to_gps(value) -> float:
    """
    A GPS time from whatever form it was written in.

    Accepts a GPS number, a UTC or ISO string, or an event name carrying a UTC stamp
    (``GW190412_053044``, ``SGW190412_053044``). Converting these by hand and comparing
    by eye is the step this exists to replace -- and the step the transcription errors
    were in.

    A bare number is taken as GPS, never as a UTC epoch: the two differ by decades, so a
    misreading is obvious rather than subtle, and every source in this field quotes GPS.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        seconds = float(value)
        if not np.isfinite(seconds):
            raise ValueError(f"{value!r} is not a finite time")
        return seconds

    text = str(value).strip()
    if not text:
        raise ValueError("an empty string is not a time")
    try:
        return float(text)
    except ValueError:
        pass

    from sage.search.naming import gps_from_name

    try:
        return gps_from_name(text)
    except ValueError:
        pass

    from astropy.time import Time

    try:
        return float(Time(text.replace(" ", "T"), scale="utc").gps)
    except Exception as error:
        raise ValueError(
            f"{value!r} is not a time this understands. Give a GPS number, a UTC "
            f"timestamp such as '2019-04-12 05:30:44', or an event name carrying one "
            f"such as 'GW190412_053044' ({error})"
        ) from None


def from_times(
    key: str,
    times: Iterable,
    far_per_yr: Optional[Sequence[float]] = None,
    p_astro: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
    conventions: Optional[Conventions] = None,
    reference: str = "",
    version: str = "",
    **columns,
) -> ExternalCatalogue:
    """
    An event list from times, and whatever else the source published.

    The one entry point every comparison goes through. A wrapper for a particular source
    does its unpacking and calls this; a person with a list of times off a paper calls it
    directly.

    ``conventions`` says what the significance means. It defaults to a FAR in per-year
    when ``far_per_yr`` is given and to p_astro when only that is -- and a source
    publishing p_astro should also name its prior, since two p_astro values under
    different priors are not the same quantity.
    """
    gps = np.asarray([to_gps(value) for value in times], dtype=np.float64)
    if gps.size == 0:
        raise ValueError(f"{key!r} was given no event times")

    def column(values, default=None):
        if values is None:
            return [default] * gps.size
        values = list(values)
        if len(values) != gps.size:
            raise ValueError(
                f"{key!r} has {gps.size} times and {len(values)} values in a companion "
                "column; they are paired elementwise, so a mismatch would attach a "
                "significance to the wrong event"
            )
        return values

    if conventions is None:
        conventions = Conventions(
            significance="far_per_yr" if far_per_yr is not None else "p_astro"
        )
    if names is None:
        from sage.search.naming import name_from_gps

        names = [name_from_gps(float(t), prefix="") for t in gps]

    far = column(far_per_yr)
    prob = column(p_astro)
    extra = {name: column(values) for name, values in columns.items()}

    events = []
    for index in range(gps.size):
        events.append(
            CatalogueEvent(
                name=str(names[index]),
                gps=float(gps[index]),
                source=str(key),
                far_per_yr=_optional(far[index]),
                ifar_yr=(
                    None
                    if _optional(far[index]) is None or far[index] == 0
                    else 1.0 / float(far[index])
                ),
                p_astro=_optional(prob[index]),
                extra={name: values[index] for name, values in extra.items()},
            )
        )
    return ExternalCatalogue(
        key=str(key),
        events=events,
        conventions=conventions,
        reference=reference,
        version=version,
    )


def _optional(value):
    """``None`` for a missing or non-finite entry, a float otherwise."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(out) else out


def read_event_times(
    path: Union[str, Path],
    key: Optional[str] = None,
    conventions: Optional[Conventions] = None,
    **kwargs,
) -> ExternalCatalogue:
    """
    Read an event list a person wrote down.

    Accepts what someone actually produces when copying a table out of a paper:

    - a header line naming columns, comma- or whitespace-separated, in any order and
      under any of the spellings in :data:`COLUMN_ALIASES`;
    - no header at all, in which case the first field of each line is the time and a
      second numeric field, if present, is the FAR;
    - blank lines and ``#`` comments anywhere.

    Times may be GPS, UTC strings or event names -- see :func:`to_gps`.

    A header naming no recognised time column is refused, listing what it does hold. The
    alternative is reading it positionally, which succeeds on the wrong column and gives
    plausible times that are wrong by hours.
    """
    path = Path(path)
    lines = [
        line.split("#", 1)[0].strip()
        for line in path.read_text().splitlines()
    ]
    rows = [line for line in lines if line]
    if not rows:
        raise ValueError(f"{path} holds no event lines")

    delimiter = "," if "," in rows[0] else None
    header = [
        field.strip().lower() for field in _split(rows[0], delimiter)
    ]
    resolved = _resolve_header(header)

    if resolved is None:
        # No header: first field is the time, an optional second numeric field the FAR.
        times, far = [], []
        for row in rows:
            fields = _split(row, delimiter)
            times.append(fields[0])
            far.append(fields[1] if len(fields) > 1 else None)
        return from_times(
            key or path.stem,
            times,
            far_per_yr=far if any(f is not None for f in far) else None,
            conventions=conventions,
            **kwargs,
        )

    columns: Dict[str, list] = {name: [] for name in resolved.values()}
    for row in rows[1:]:
        fields = _split(row, delimiter)
        if len(fields) != len(header):
            raise ValueError(
                f"{path}: a row has {len(fields)} fields against {len(header)} in the "
                f"header, so its columns cannot be identified: {row!r}"
            )
        for index, quantity in resolved.items():
            columns[quantity].append(fields[index])
    times = columns.pop("gps")
    return from_times(
        key or path.stem,
        times,
        far_per_yr=columns.pop("far_per_yr", None),
        p_astro=columns.pop("p_astro", None),
        names=columns.pop("name", None),
        conventions=conventions,
        **{**columns, **kwargs},
    )


def _split(row: str, delimiter: Optional[str]) -> list:
    """Fields of one row."""
    return [f.strip() for f in (row.split(delimiter) if delimiter else row.split())]


def _resolve_header(header: Sequence[str]) -> Optional[Dict[int, str]]:
    """
    Map column positions to quantities, or ``None`` when this is not a header.

    A row is a header only if it names a time column; otherwise it is data, and a file
    whose first line happens to name a *significance* but not a time is refused rather
    than being read as one row short.
    """
    found: Dict[int, str] = {}
    for index, field in enumerate(header):
        for quantity, aliases in COLUMN_ALIASES.items():
            if field in aliases and quantity not in found.values():
                found[index] = quantity
                break
    if "gps" in found.values():
        return found
    if found:
        raise ValueError(
            f"the header names {sorted(set(found.values()))} but no time column. "
            f"Accepted spellings for the time are {list(COLUMN_ALIASES['gps'])}"
        )
    return None
