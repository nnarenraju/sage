#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : test_search_sources.py
Description   : Per-release fetchers and handlers for external data.

Created on 2026-08-21

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Each handler converts one published release into the canonical form the search reads.
The release is the part that changes; the canonical form is the part that must not.
"""

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from sage.search.sources import gwtc3_powerlawpeak as pp

#: The extracted release file, where the campaign staged it. Present only after a
#: deliberate fetch, so everything needing it skips rather than reaching the network.
RELEASE = Path(
    "/work/nagarajan/sage_runs/search/o3a_HL/injections/sources"
) / Path(pp.MEMBER).name

requires_release = pytest.mark.skipif(
    not RELEASE.is_file(), reason=f"{RELEASE} not fetched in this checkout"
)


def _result_json(path, columns, n=32, seed=0):
    """A bilby-shaped result file holding a small posterior."""
    rng = np.random.default_rng(seed)
    content = {name: list(rng.normal(size=n)) for name in columns}
    path.write_text(
        json.dumps({"posterior": {"__dataframe__": True, "content": content}})
    )
    return path


class TestPosteriorReading:
    """A bilby result JSON is read without bilby."""

    def test_content_is_unwrapped(self, tmp_path):
        """
        The samples are one level below ``posterior``; that level is the serialised
        frame. Stopping short of it hands on ``{"__dataframe__", "content"}``, which
        fails as a missing hyperparameter two calls later rather than here.
        """
        path = _result_json(tmp_path / "r.json", pp.HYPERPARAMETERS)
        frame = pp.read_posterior(path)

        assert list(frame.columns) == list(pp.HYPERPARAMETERS)
        assert len(frame) == 32

    def test_missing_hyperparameter_named(self, tmp_path):
        """
        A release lacking one of the fourteen cannot draw the population, and says which
        one rather than failing inside the sampler.
        """
        columns = [c for c in pp.HYPERPARAMETERS if c != "sigpp"]
        path = _result_json(tmp_path / "r.json", columns)

        with pytest.raises(KeyError, match="sigpp"):
            pp.read_posterior(path)

    def test_non_result_refused(self, tmp_path):
        """A file that is not a bilby result is refused by name."""
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"samples": {}}))

        with pytest.raises(KeyError, match="no 'posterior'"):
            pp.read_posterior(path)


class TestCanonicalForm:
    """What the search reads, and what it refuses."""

    def _canonical(self, path):
        payload = {
            "model": "PowerLawPeak",
            "hyperparameters": {name: 1.0 for name in pp.HYPERPARAMETERS},
            "source": {"doi": pp.DOI},
        }
        path.write_text(json.dumps(payload))
        return path

    def test_load_returns_the_fourteen(self, tmp_path):
        """Exactly what sage.search.injection.population reads, and nothing else."""
        values = pp.load(self._canonical(tmp_path / "h.json"))

        assert set(values) == set(pp.HYPERPARAMETERS)
        assert all(isinstance(v, float) for v in values.values())

    def test_release_file_refused(self, tmp_path):
        """
        Handing a release file straight to the search is the failure the handler exists
        to prevent, so it is named rather than parsed.
        """
        path = _result_json(tmp_path / "r.json", pp.HYPERPARAMETERS)

        with pytest.raises(KeyError, match="no 'hyperparameters'"):
            pp.load(path)


class TestArchiveSafety:
    """A tar member may name any path it likes."""

    def test_escaping_member_refused(self, tmp_path):
        """
        Extracting a member whose name resolves outside the destination writes wherever
        the archive says. Refused by resolved path, not by inspecting the name.
        """
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        archive = tmp_path / "a.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(outside, arcname="../escaped.json")

        with pytest.raises((ValueError, KeyError)):
            pp.extract(archive, tmp_path / "dest", member="../escaped.json")

    def test_absent_member_lists_alternatives(self, tmp_path):
        """A record whose layout moved says what it does hold."""
        member = tmp_path / "other.json"
        member.write_text("{}")
        archive = tmp_path / "a.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(member, arcname="analyses/PowerLawPeak/other.json")

        with pytest.raises(KeyError, match="other.json"):
            pp.extract(archive, tmp_path / "dest")


class TestAgainstRelease:
    """Fidelity to sgwc-1, on the file sgwc-1 read."""

    @requires_release
    def test_release_shape(self):
        """The published fit: 11184 hyperposterior samples over 22 columns."""
        frame = pp.read_posterior(RELEASE)

        assert frame.shape == (11184, 22)

    @requires_release
    def test_max_likelihood_reads_the_maximum(self):
        """
        The densest sample of this posterior, read rather than estimated. sgwc-1's
        marginal-MAP choice -- sample 10677, now removed -- sat 2.16 nats below it, a
        population roughly nine times less likely than the one the data prefers.
        """
        frame = pp.read_posterior(RELEASE)
        best, index = pp.select(frame, method="max_likelihood")

        assert index == 11182
        assert float(best["log_likelihood"]) == pytest.approx(107.9093, abs=1e-3)
        assert float(best["log_likelihood"]) == float(frame["log_likelihood"].max())

    @requires_release
    @pytest.mark.slow
    def test_joint_kde_is_the_usable_fallback(self):
        """
        What a release with no likelihood column would fall back to. It has to land near
        the true maximum to be worth having: measured, 0.31 nats below it, against the
        2.16 of the method it replaces.
        """
        frame = pp.read_posterior(RELEASE)
        best, index = pp.select(frame, method="joint_map_kde")

        assert index == 11175
        assert float(frame["log_likelihood"].max()) - float(best["log_likelihood"]) < 1.0

    @requires_release
    def test_prior_is_constant_on_this_release(self):
        """
        Which is why the maximum-likelihood and maximum-a-posteriori samples coincide
        here. Asserted rather than assumed: a release with a varying prior would make the
        two different, and the code adds the prior for that reason.
        """
        frame = pp.read_posterior(RELEASE)

        assert frame["log_prior"].nunique() == 1


class TestSelectionMethods:
    """Four ways to name a representative sample, and what separates them."""

    def _frame(self, n=400, seed=3):
        """A posterior with a known likelihood peak away from the marginal modes."""
        import pandas as pd

        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(
            {
                "alpha": rng.normal(3.0, 0.5, n),
                "beta": rng.normal(1.0, 0.3, n),
                "rate": rng.normal(20.0, 5.0, n),
            }
        )
        frame["log_likelihood"] = rng.normal(100.0, 1.0, n)
        frame.loc[7, "log_likelihood"] = 200.0
        frame["log_prior"] = -20.0
        return frame

    def test_max_likelihood_reads_the_column(self):
        """No bandwidth, no dimensionality, no cross-validation -- it is published."""
        frame = self._frame()
        sample, index = pp.select(frame, method="max_likelihood")

        assert index == 7
        assert sample["log_likelihood"] == 200.0

    def test_max_likelihood_includes_the_prior(self):
        """
        The maximum *a posteriori* sample maximises likelihood plus prior. They coincide
        on the GWTC-3 release because its log_prior is constant, but a release with a
        varying prior must not be handed the maximum-likelihood sample under the name.
        """
        frame = self._frame()
        frame.loc[7, "log_prior"] = -500.0
        frame.loc[11, "log_likelihood"] = 150.0

        _, index = pp.select(frame, method="max_likelihood")
        assert index == 11

    def test_max_likelihood_refused_without_a_likelihood(self):
        """Named rather than silently falling back to an estimate of what it can read."""
        frame = self._frame().drop(columns=["log_likelihood"])

        with pytest.raises(ValueError, match="log_likelihood"):
            pp.select(frame, method="max_likelihood")

    def test_rejected_methods_are_gone(self):
        """
        Both were built, measured and removed: an unstandardised Gaussian KDE that
        over-smooths into the wrong mode, and the marginal MAP that assembles a point the
        joint posterior need not support. A bad option left reachable is one that gets
        used.
        """
        for method in ("joint_map_gaussian_kde", "marginal_map"):
            assert method not in pp.METHODS
            with pytest.raises(ValueError, match="unknown selection method"):
                pp.select(self._frame(), method=method)

    def test_auxiliary_columns_dropped(self):
        """
        The sampler's bookkeeping is not the population. Measured on the release: leaving
        it in moved the joint MAP from sample 11175 to 5789, a likelihood 5.3 nats worse.
        """
        frame = self._frame()
        frame["surveyed_hypervolume"] = 4200.0 + np.arange(len(frame))
        frame["amax"] = 1.0

        columns = pp.population_columns(frame)
        assert "surveyed_hypervolume" not in columns
        assert "log_likelihood" not in columns
        # Constant columns go too: they carry no information and have zero variance to
        # standardise by.
        assert "amax" not in columns
        assert set(columns) == {"alpha", "beta", "rate"}

    def test_every_method_returns_a_real_sample(self):
        """
        A hyperparameter vector assembled coordinate-by-coordinate need not be one the
        joint posterior supports, so each method returns a row rather than a point.
        """
        frame = self._frame()
        for method in pp.METHODS:
            sample, index = pp.select(frame, method=method)
            assert 0 <= index < len(frame)
            assert sample["alpha"] == frame.iloc[index]["alpha"]

    def test_unknown_method_refused(self):
        """The population is not something to fall back to a default on."""
        with pytest.raises(ValueError, match="unknown selection method"):
            pp.select(self._frame(), method="whatever")


class TestStoredPopulation:
    """The whole posterior travels with the representative, for marginalising."""

    def _payload(self, tmp_path, with_population=True):
        payload = {
            "hyperparameters": {name: 1.0 for name in pp.HYPERPARAMETERS},
            "selection": {"method": "max_likelihood", "index": 3},
        }
        if with_population:
            payload["population"] = {
                "columns": list(pp.HYPERPARAMETERS),
                "n_samples": 5,
                "samples": [
                    [float(i + j) for i in range(5)]
                    for j in range(len(pp.HYPERPARAMETERS))
                ],
            }
        path = tmp_path / "h.json"
        path.write_text(json.dumps(payload))
        return path

    def test_population_round_trips(self, tmp_path):
        """One mapping per posterior sample, in the same names the representative uses."""
        samples = pp.population(self._payload(tmp_path))

        assert len(samples) == 5
        assert set(samples[0]) == set(pp.HYPERPARAMETERS)
        assert samples[2]["alpha"] == 2.0

    def test_absent_population_refused(self, tmp_path):
        """
        Marginalising over a file that holds one sample would silently become
        conditioning on it, which is the opposite of what was asked for.
        """
        with pytest.raises(KeyError, match="no population block"):
            pp.population(self._payload(tmp_path, with_population=False))
