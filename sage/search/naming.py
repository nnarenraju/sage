#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : naming.py
Description   : Candidate naming and prefix policy.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The bare ``GW`` prefix conventionally denotes a published detection, so Sage candidates
carry their own prefix and never mint one.

Every candidate is named the same way regardless of tier. A name that encoded confidence
would have to change when a tier changed -- and tiers do change, since vetting can demote
a candidate after the fact -- which would break every filename, table row and figure
caption already referring to it. Confidence lives in the tier column; a cross-matched
event reports its catalogue name in a separate column beside its Sage name.

The stamp is UTC and therefore leap-second dependent, so it is computed with a real time
scale rather than by arithmetic on the GPS epoch. An implementation that skips that is
wrong by the current offset and still produces a plausible-looking timestamp.
"""

import re
from typing import Optional

DEFAULT_PREFIX: str = "SGW"

# <letters><YYMMDD>_<HHMMSS>, with the leading prefix optional and an optional
# disambiguating suffix. The suffix exists because the stamp names a *second*: two
# published events never shared one, but a search's candidate list runs to thousands of
# sub-threshold triggers over a run, and clustering only guarantees they are 0.35 s apart.
_NAME_PATTERN = re.compile(
    r"^(?P<prefix>[A-Za-z]*)(?P<date>\d{6})_(?P<time>\d{6})(?:-(?P<suffix>\d+))?$"
)


def name_from_gps(gps: float, prefix: str = DEFAULT_PREFIX) -> str:
    """
    Format a candidate name as ``<prefix>YYMMDD_HHMMSS`` from its GPS time.

    The sub-second part is truncated, not rounded: the stamp names the second the event
    falls in, and rounding would push an event late in a second into the next one and
    give it a name disagreeing with its own recorded time.
    """
    if gps < 0:
        raise ValueError(f"GPS time must not be negative, got {gps}")
    # Deferred: astropy is a core dependency but a heavy import, and naming a candidate
    # is the only thing here that needs it.
    from astropy.time import Time

    stamp = Time(int(gps), format="gps", scale="tai").utc.strftime("%y%m%d_%H%M%S")
    return f"{prefix}{stamp}"


def disambiguate(names, gps) -> list:
    """
    Make a list of candidate names unique, keeping the GWTC form wherever it already is.

    ``name_from_gps`` names the second an event falls in, which is unique for published
    detections and is not unique for a search's candidate list: clustering separates
    events by 0.35 s, so several can share a second. The name is the identity every later
    join uses -- the trials record, the data-quality verdict, the catalogue crossmatch --
    so a collision is not cosmetic. It makes those joins ambiguous, and
    ``trials._records_by_name`` refuses outright rather than picking one.

    The earliest candidate in a colliding second keeps the bare name and the rest take
    ``-1``, ``-2``, ... in time order. Ordering by time rather than by position makes the
    assignment a property of the data, so a re-run that returns the same candidates in a
    different order gives them the same names.
    """
    import numpy as np

    names = [str(name) for name in names]
    gps = np.asarray(gps, dtype=np.float64)
    if len(names) != gps.size:
        raise ValueError(
            f"{len(names)} names against {gps.size} times; they are paired elementwise"
        )
    order = np.argsort(gps, kind="stable")
    seen: dict = {}
    out = list(names)
    for index in order:
        base = names[index]
        count = seen.get(base, 0)
        seen[base] = count + 1
        out[index] = base if count == 0 else f"{base}-{count}"
    return out


def gps_from_name(name: str) -> float:
    """
    Parse the UTC stamp in a candidate name back to GPS, to the second.

    Accepts an external ``GW`` name as readily as a Sage one, so a catalogue row can be
    placed in time from its label when no explicit time is published. The short form
    without a time is refused, being ambiguous to within a day.
    """
    from astropy.time import Time

    match = _NAME_PATTERN.match(str(name).strip())
    if match is None:
        raise ValueError(
            f"{name!r} is not a candidate name of the form <prefix>YYMMDD_HHMMSS; "
            "the short form without a time is ambiguous within a day"
        )
    # The disambiguating suffix is deliberately ignored: it distinguishes candidates
    # inside one second and carries no time information of its own.
    date, clock = match.group("date"), match.group("time")
    isot = (
        f"20{date[0:2]}-{date[2:4]}-{date[4:6]}T"
        f"{clock[0:2]}:{clock[2:4]}:{clock[4:6]}"
    )
    try:
        return float(Time(isot, format="isot", scale="utc").gps)
    except Exception as exc:
        raise ValueError(f"{name!r} does not encode a real UTC time: {exc}") from None


def check_prefix_policy(
    prefix: str, p_astro: float, force_reason: Optional[str] = None
) -> None:
    """
    Refuse a bare ``GW`` prefix, which denotes a published detection.

    Sage does not assign ``GW`` names at any probability: the prefix is a claim about
    provenance rather than about significance, and a search producing its own candidate
    list is not the body that assigns it. An override exists for reproducing a published
    catalogue, and requires a written reason so the exception is visible.
    """
    if prefix.upper() != "GW":
        return
    if force_reason is None:
        raise ValueError(
            "the bare 'GW' prefix denotes a published detection and is not assigned by "
            f"this search; use {DEFAULT_PREFIX!r}, or pass force_reason to override"
        )
    if not str(force_reason).strip():
        raise ValueError("force_reason must explain why a 'GW' prefix is being assigned")


def normalise_external_name(name: str) -> str:
    """
    Strip prefixes and separators from an external catalogue name.

    Names are labels only; cross-matching is done on GPS time, because the same event
    appears with second-level differences in its name across catalogues.
    """
    text = str(name).strip().lower().replace("-", "_")
    return re.sub(r"^[a-z]+", "", text)
