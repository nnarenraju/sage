#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : references.py
Description   : Registry of the local reference documents behind the search methods.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A convenience index of the documents behind the methods here, so a reader can find the
sources locally and a bibliography can be generated for the write-up. Where a specific
result is taken from one of them, the docstring at that site says so.

This is a reading aid, not a constraint on the code: nothing requires a given function to
name a given paper, and the registry can be extended or trimmed freely.

Titles below were read from the first page of each stored file rather than transcribed.
The PDFs are not checked in; ``docs/references/fetch.py`` restores them.
"""

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "docs" / "references"


@dataclass(frozen=True)
class Reference:
    """One stored document and the results this subpackage takes from it."""

    key: str
    arxiv_id: str
    title: str
    filename: str
    equations: Dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def path(self) -> Path:
        """Absolute path to the stored document."""
        return REFERENCE_DIR / self.filename

    def exists(self) -> bool:
        """Whether the document is present locally."""
        return self.path.is_file()

    def cite(self, equation: Optional[str] = None) -> str:
        """Render a citation naming the local file and, optionally, an equation."""
        where = f"docs/references/{self.filename}"
        if equation is None:
            return f"{self.title} ({where})"
        return f"{self.title}, Eq. ({equation}) ({where})"


REFERENCES: Dict[str, Reference] = {
    "fgmc": Reference(
        key="fgmc",
        arxiv_id="1302.5341",
        title="Counting And Confusion: Bayesian Rate Estimation With Multiple Populations",
        filename="arxiv_1302.5341.pdf",
        equations={
            "12": "Likelihood of the data given per-event foreground/background flags.",
            "14": "Flag prior; an event is foreground with probability Rf/(Rf+Rb).",
            "21": (
                "Rate posterior with the flags marginalised out: "
                "p(Rf,Rb,th|d,N) proportional to "
                "prod_i [Rf fhat(x_i,th) + Rb bhat(x_i,th)] "
                "exp[-(Rf+Rb)] p(th) / sqrt(Rf Rb). "
                "The 1/sqrt(Rf Rb) factor is the Jeffreys prior on the two rates."
            ),
            "35": (
                "Foreground-dominated limit, Rf^(N-1/2) exp(-Rf), peaked at Rf = N - 1/2; "
                "the half comes from the Jeffreys prior."
            ),
        },
        note="Origin of the mixture-model rate estimator used by every search pipeline.",
    ),
    "unified_pastro": Reference(
        key="unified_pastro",
        arxiv_id="2305.00071",
        title=(
            "A Unified p_astro for Gravitational Waves: Consistently Combining "
            "Information from Multiple Search Pipelines"
        ),
        filename="arxiv_2305.00071.pdf",
        equations={
            "4": "Count parameters: Lambda_s = R_s T and Lambda_n = R_n T.",
            "10": (
                "Rate posterior: p(Ls,Ln|{x},N) proportional to "
                "exp[-(Ln+Ls)] pi(Ls,Ln) prod_i {Ls p(x_i|S) + Ln p(x_i|0)}."
            ),
            "11": (
                "Per-trigger probability, marginalised over the rate posterior: "
                "p_astro(x) = int dLs dLn [Ls p(x|S) / (Ls p(x|S) + Ln p(x|0))] "
                "p(Ls,Ln|{x},N)."
            ),
        },
        note=(
            "Section V adopts a preliminary cut of FAR <= 2 per day, the same threshold "
            "used for the public candidate list."
        ),
    ),
    "pycbc_search": Reference(
        key="pycbc_search",
        arxiv_id="1508.02357",
        title="The PyCBC search for gravitational waves from compact binary coalescence",
        filename="arxiv_1508.02357.pdf",
        equations={},
        note="Time-slide background construction and the conservative false-alarm counting.",
    ),
    "sensitivity_injections": Reference(
        key="sensitivity_injections",
        arxiv_id="2508.10638",
        title=(
            "Compact Binary Coalescence Sensitivity Estimates with Injection Campaigns "
            "during the LIGO-Virgo-KAGRA Collaborations' Fourth Observing Run"
        ),
        filename="arxiv_2508.10638.pdf",
        equations={},
        note=(
            "Importance-sampled sensitivity estimator, the injected distribution, the "
            "found/missed matching rule and the effective-sample requirement."
        ),
    ),
    "idq": Reference(
        key="idq",
        arxiv_id="2005.12761",
        title=(
            "iDQ: Statistical Inference of Non-Gaussian Noise with Auxiliary Degrees of "
            "Freedom in Gravitational-Wave Detectors"
        ),
        filename="arxiv_2005.12761.pdf",
        equations={},
        note=(
            "Method behind the auxiliary-channel glitch inference used to vet candidates. "
            "Uses auxiliary channels only and never the strain, which is what makes its "
            "verdict independent of a strain-derived detection statistic."
        ),
    ),
    "idq_performance": Reference(
        key="idq_performance",
        arxiv_id="2412.04638",
        title="Performance of iDQ ahead of LIGO, Virgo, and KAGRA's fourth observing run",
        filename="arxiv_2412.04638.pdf",
        equations={},
        note=(
            "Performance characterisation, and the scale of the auxiliary-channel input "
            "that makes the pipeline impossible to reproduce from public data."
        ),
    ),
    "open_data_o4a": Reference(
        key="open_data_o4a",
        arxiv_id="2508.18079",
        title="Open Data from LIGO, Virgo, and KAGRA through the First Part of O4",
        filename="arxiv_2508.18079.pdf",
        equations={},
        note=(
            "What the open release contains, including the auxiliary-derived data-quality "
            "time series published alongside the alternate strain."
        ),
    ),
    "beyond_gwtc3": Reference(
        key="beyond_gwtc3",
        arxiv_id="2401.08709",
        title=(
            "Beyond GWTC-3: Analysing and verifying new gravitational-wave events from "
            "community catalogues"
        ),
        filename="arxiv_2401.08709.pdf",
        equations={},
        note="Verification of externally claimed events through a common analysis.",
    ),
    "gwtc2p1": Reference(
        key="gwtc2p1",
        arxiv_id="2108.01045",
        title=(
            "GWTC-2.1: Deep Extended Catalog of Compact Binary Coalescences Observed by "
            "LIGO and Virgo During the First Half of the Third Observing Run"
        ),
        filename="arxiv_2108.01045.pdf",
        equations={},
        note="Introduced the sub-threshold candidate list at FAR < 2 per day.",
    ),
    "gwtc3": Reference(
        key="gwtc3",
        arxiv_id="2111.03606",
        title=(
            "GWTC-3: Compact Binary Coalescences Observed by LIGO and Virgo during the "
            "Second Part of the Third Observing Run"
        ),
        filename="arxiv_2111.03606.pdf",
        equations={},
        note="Category definitions and the astrophysical-probability conventions.",
    ),
    "gwtc4_methods": Reference(
        key="gwtc4_methods",
        arxiv_id="2508.18081",
        title=(
            "GWTC-4.0: Methods for Identifying and Characterizing Gravitational-wave "
            "Transients"
        ),
        filename="arxiv_2508.18081.pdf",
        equations={},
        note="Ranking statistics, background construction, data quality and inclusion criteria.",
    ),
    "gwtc4_results": Reference(
        key="gwtc4_results",
        arxiv_id="2508.18082",
        title=(
            "GWTC-4.0: Updating the Gravitational-wave Transient Catalog with "
            "Observations from the First Part of the Fourth LIGO-Virgo-KAGRA Observing Run"
        ),
        filename="arxiv_2508.18082.pdf",
        equations={},
        note="Catalogue contents and the presentation conventions for candidate tables.",
    ),
    "gwtc5_methods": Reference(
        key="gwtc5_methods",
        arxiv_id="2605.27224",
        title=(
            "GWTC-5.0: Methods for Identifying and Characterizing Gravitational-wave "
            "Transients"
        ),
        filename="arxiv_2605.27224.pdf",
        equations={},
        note=(
            "Current methods reference: hierarchical background removal, the tiered "
            "inclusion criteria, and the section ordering this analysis follows."
        ),
    ),
}


def get(key: str) -> Reference:
    """Look up a reference, raising with suggestions on a typo."""
    try:
        return REFERENCES[key]
    except KeyError:
        close = difflib.get_close_matches(key, REFERENCES, n=3, cutoff=0.5)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown reference {key!r}{hint}") from None


def cite(key: str, equation: Optional[str] = None) -> str:
    """Render a citation for use in a docstring, log line or table caption."""
    return get(key).cite(equation)


def verify_all() -> Dict[str, bool]:
    """Check every registered document is present under ``docs/references``."""
    return {key: ref.exists() for key, ref in REFERENCES.items()}


def bibliography(path: Optional[str | Path] = None) -> str:
    """
    Render the registry as a reference list for the manuscript.

    Ordered by identifier so the output is stable between runs. Written to ``path`` when
    one is given, and returned either way.
    """
    lines = [
        f"[{ref.arxiv_id}] {ref.title}. arXiv:{ref.arxiv_id}."
        for ref in sorted(REFERENCES.values(), key=lambda r: r.arxiv_id)
    ]
    text = "\n".join(lines)
    if path is not None:
        Path(path).write_text(text + "\n", encoding="utf-8")
    return text
