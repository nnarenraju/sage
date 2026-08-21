#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : campaign.py
Description   : Draw signals, inject them into real noise, and score them.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

What this stage produces is ``p(x | signal)``: the distribution of the ranking statistic
over signals drawn from the astrophysical population and scored the way the search scores
everything else. It is the only thing p_astro consumes from here.

**This follows sgwc-1's injection study, which has no timeline.** Each injection is added
to a *randomly chosen* stretch of real noise from the search release, not placed at a
scheduled GPS time and looked for afterwards. There is consequently no found/missed
outcome, no association window and no matching: an injection is scored, and its statistic
is the measurement. That is self-consistent with sgwc-1 having no sensitive-volume
calculation, since ``VT`` needs found/missed against an analysed timeline to be defined at
all.

A lattice-scheduled campaign -- injections overlaid on the analysed stream through the
same reader, with those outside analysed segments counted as missed -- is what
:mod:`sage.search.injection.overlay` and :mod:`sage.search.injection.matching` are shaped
for, and is what a PyCBC-style sensitivity would need. It is a later addition, kept
separate deliberately: the two answer different questions and mixing them would make the
signal density depend on the run's duty cycle.

Two deliberate differences from sgwc-1's code, both settled:

- **Waveforms come from Sage's own IMRPhenomPv2**, not from the LALSimulation wrapper
  sgwc-1 carries locally. It is the generator the network was trained with, so the density
  describes signals as the network learned them; it is verified against LAL to a worst
  mismatch of 1.14e-7, so it agrees with what that wrapper produces; and it avoids a
  second waveform implementation that would have to be kept in agreement with the first.
- **Extrinsic parameters come from the training run's own prior** through Sage's
  ``DistributionSampler``, where sgwc-1 read a PyCBC ``.ini``. It is the same prior; the
  ``.ini`` was how the old layout expressed it.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass
class CampaignReport:
    """Completion accounting for one injection pass."""

    stream: int
    n_injections: int
    n_scored: int
    n_outside_segments: int
    wall_seconds: float

    def as_dict(self) -> dict:
        """
        Flat dict for the manifest.

        ``n_outside_segments`` is reported and is zero by construction on this campaign:
        noise is drawn from inside the analysed segments, so no injection can land
        outside one. It is kept in the record rather than dropped so that a
        lattice-scheduled campaign, where it is the number that carries duty cycle into
        the sensitivity, reports it in the same field.
        """
        return {
            "stream": int(self.stream),
            "n_injections": int(self.n_injections),
            "n_scored": int(self.n_scored),
            "n_outside_segments": int(self.n_outside_segments),
            "wall_seconds": float(self.wall_seconds),
            "injections_per_second": (
                float(self.n_scored / self.wall_seconds)
                if self.wall_seconds > 0
                else 0.0
            ),
        }


class InjectionCampaign:
    """
    Score one injection stream against one observing run's noise.

    Parameters
    ----------
    engine : SearchEngine
        The same engine the zero-lag and background passes use. Sharing it is what makes
        the signal density comparable with the noise density: a separate forward path
        could differ in the preprocessor, the autocast policy or the decode, and the
        ratio of two densities measured through different paths is not a likelihood ratio.
    noise : NoiseSlices
        Supplies a real-noise window per injection.
    """

    def __init__(self, spec, engine, injections, noise, writer) -> None:
        self.spec = spec
        self.engine = engine
        self.injections = injections
        self.noise = noise
        self.writer = writer

    def run(self, resume: bool = True) -> CampaignReport:
        """
        Generate, inject and score every injection in the stream.

        Batched, and each batch is committed as one block, so a requeued job resumes at a
        batch boundary and cannot rescore an injection it already wrote -- which would put
        the same signal into ``p(x | signal)`` twice.
        """
        started = time.perf_counter()
        batch_size = int(self.spec.engine.batch_size)
        done = set(self.writer.completed_blocks()) if resume else set()
        total = len(self.injections)
        scored = 0

        for block_id, lo in enumerate(range(0, total, batch_size)):
            if block_id in done:
                continue
            hi = min(lo + batch_size, total)
            strain, params = self.injections.build(lo, hi, self.noise)
            stat, point = self.engine.forward(strain)
            self.writer.append(_table(lo, stat, point, params, self.engine))
            self.writer.complete_block(block_id)
            scored += int(stat.size)

        return CampaignReport(
            stream=int(self.injections.stream),
            n_injections=int(total),
            n_scored=int(scored),
            n_outside_segments=0,
            wall_seconds=time.perf_counter() - started,
        )


class NoiseSlices:
    """
    Real-noise windows drawn at random from the analysed lattice.

    sgwc-1 injects into a randomly chosen stretch of search-era noise rather than at a
    scheduled time, so what a signal is added to is a fair draw from the noise the search
    actually reads. Drawing from the *lattice* rather than from the raw segments is what
    keeps that true: a window sampled anywhere in a segment could straddle a boundary or
    fall in the band no window start can reach, and neither is noise the search ever sees.

    Seeded, and the seed is part of the campaign's configuration, so the same injection
    set lands on the same noise on a re-run. Without that a resumed campaign would score
    its later batches against different noise from its earlier ones, and the density would
    be a mixture of two experiments.
    """

    def __init__(self, spec, grid, seed: int = 0) -> None:
        from sage.search.reader import read_segment_span

        self._read = read_segment_span
        self._grid = grid
        self._geometry = spec.geometry_object()
        self._detectors = tuple(spec.data.detectors)
        self._rng = np.random.default_rng(int(seed))
        self._release = Path(spec.data.release_dir)
        self._run = str(spec.data.observing_run)
        self._maps = {}
        # One flat list of (segment, first_local) per detector, so a draw is an index.
        self._spans = {
            detector: [
                (span.segment, span.first_local, span.n_windows)
                for span in grid.spans_by_detector[detector]
                if span.n_windows
            ]
            for detector in self._detectors
        }
        self._weights = {
            detector: np.array([n for _, _, n in spans], dtype=np.float64)
            for detector, spans in self._spans.items()
        }
        for detector, weights in self._weights.items():
            if weights.sum() <= 0:
                raise ValueError(
                    f"{detector} hosts no analysed windows, so there is no noise to "
                    "inject into"
                )
            self._weights[detector] = weights / weights.sum()

    def _mmap(self, detector: str):
        """The detector's sample stream, opened once."""
        if detector not in self._maps:
            path = self._release / f"data_{detector}_{self._run}.bin"
            if not path.is_file():
                raise FileNotFoundError(
                    f"no strain for {detector} at {path}; injections are added to real "
                    "noise from the search release, which must be built first"
                )
            self._maps[detector] = np.memmap(path, dtype=np.float32, mode="r")
        return self._maps[detector]

    def draw(self, n: int) -> "tuple":
        """
        ``(noise, segment_index, local_start)`` for ``n`` injections.

        Each detector is drawn independently. That matches sgwc-1, whose noise index array
        pairs an H1 slice with an L1 slice chosen separately -- the point is a fair sample
        of each detector's noise, not a coincident stretch, since the signal supplies the
        coherence.
        """
        window = int(self._geometry.window_samples)
        stride = int(self._geometry.stride_samples)
        noise = np.empty((n, len(self._detectors), window), dtype=np.float64)
        segment_index = np.empty(n, dtype=np.int64)
        local_start = np.empty(n, dtype=np.int64)

        for row in range(n):
            for column, detector in enumerate(self._detectors):
                spans = self._spans[detector]
                choice = int(
                    self._rng.choice(len(spans), p=self._weights[detector])
                )
                segment, first_local, n_windows = spans[choice]
                offset = first_local + stride * int(self._rng.integers(0, n_windows))
                noise[row, column] = self._read(
                    self._mmap(detector), segment, offset, window
                )
                if column == 0:
                    segment_index[row] = int(segment.segment_index)
                    local_start[row] = int(offset)
        return noise, segment_index, local_start

    def close(self) -> None:
        """Release the memory maps."""
        self._maps.clear()


def _table(first: int, stat, point, params, engine):
    """
    Assemble the shard rows for one batch.

    ``gps`` holds the injection *index*, not a time. There is no timeline here, so a GPS
    column would be a fabricated one; the index is what identifies an injection and what
    joins this shard back to the drawn parameter set. The column keeps its name because
    the shard schema is shared with the search, and a reader that took it for a time would
    be reading a number that is plainly not one.
    """
    from sage.search.triggers import TriggerTable

    n = int(np.asarray(stat).size)
    index = np.arange(first, first + n, dtype=np.int64)
    columns = {
        "stat": np.asarray(stat, dtype=np.float64),
        "gps": index.astype(np.float64),
        "segment_index": np.asarray(params["segment_index"], dtype=np.int64),
        "local_start": np.asarray(params["local_start"], dtype=np.int64),
        "slide_id": np.zeros(n, dtype=np.int64),
    }
    if engine.decoder is not None:
        columns.update(engine.decoder.trigger_columns(np.asarray(point), columns["gps"]))
    return TriggerTable(columns=columns, attrs={"clustered": False})


class InjectionSet:
    """
    The drawn injections, and how a batch of them becomes strain.

    Holds the parameter table and the generator; :meth:`build` turns a slice of it into
    ``(strain, params)`` ready for the engine's forward pass.
    """

    def __init__(self, spec, table, generator, sampler, stream: int = 0) -> None:
        self.spec = spec
        self.table = table
        self.generator = generator
        self.sampler = sampler
        self.stream = int(stream)

    def __len__(self) -> int:
        """Number of injections in the stream."""
        return int(self.table.shape[0])

    def build(self, lo: int, hi: int, noise):
        """
        Signal plus real noise, for injections ``[lo, hi)``.

        The signal is generated through the same approximant, projection and SNR
        convention the network was trained under, and added to the noise in the strain
        domain -- which is where a signal is actually added to a detector's output. The
        whitening and multirate binning then run inside the engine, on the sum, exactly as
        they do for a search window.

        The sampler is sought to ``lo`` rather than read sequentially, so a resumed
        campaign that skips completed batches still gives every injection its own row.
        """
        import torch

        self.sampler.seek(int(lo))
        n = int(hi) - int(lo)
        strain_noise, segment_index, local_start = noise.draw(n)
        with torch.no_grad():
            signal = _detach(self.generator(n))
        signal = np.asarray(signal, dtype=np.float64)
        if signal.shape != strain_noise.shape:
            raise ValueError(
                f"the generator returned {signal.shape} against noise of "
                f"{strain_noise.shape}; the two are added sample by sample, so a "
                "mismatch would broadcast a signal across the wrong detectors or times"
            )
        return (
            strain_noise + signal,
            {"segment_index": segment_index, "local_start": local_start},
        )


def _detach(value):
    """A tensor, or the first element of a tuple of them, as a numpy array."""
    import torch

    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _generator(sampler):
    """
    The approximant the network was trained with, reading the injection table.

    Sage's own IMRPhenomPv2 rather than the LALSimulation wrapper sgwc-1 carries: it is
    the generator the weights were fitted against, and it is verified to a worst mismatch
    of 1.14e-7 against LAL, so it produces what that wrapper would.
    """
    from sage.data.waveform import ConstantProjection, IMRPhenomPv2

    return IMRPhenomPv2(sampler, ConstantProjection(), augment=None)


def _hyperposterior(spec) -> dict:
    """
    The Power-Law + Peak hyperposterior sample the population is drawn at.

    Reads the canonical form a :mod:`sage.search.sources` handler writes -- a flat
    mapping of hyperparameter to value, with the release it came from recorded beside
    it. Required rather than defaulted: the population is the published one, and
    inventing hyperparameters would produce a plausible population that is not it.

    A release file is refused rather than parsed here. Every catalogue restructures
    between versions, and the GWTC-3 and GWTC-4.0 releases already disagree on format
    for this same model -- bilby JSON against ``popsummary`` HDF5. Keeping that knowledge
    in one module per release is what stops a campaign from silently reading the wrong
    level of a nested file: ``payload["posterior"]`` on a bilby result is the serialised
    frame, not the samples, and handing it on produced a ``KeyError`` two calls later.
    """
    import json

    path = getattr(spec.injection, "hyperposterior_path", None)
    if not path:
        raise FileNotFoundError(
            "no injection.hyperposterior_path is set. Injections are drawn from the "
            "Power-Law + Peak model at its MAP hyperposterior sample; build one with "
            "sage.search.sources.gwtc3_powerlawpeak.build() and point this at what it "
            "wrote. There is no defensible default for it"
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no hyperposterior at {path}")

    payload = json.loads(path.read_text())
    if "hyperparameters" in payload:
        return {str(k): float(v) for k, v in payload["hyperparameters"].items()}
    if "posterior" in payload:
        raise ValueError(
            f"{path.name} looks like a raw release file rather than a canonical "
            "hyperposterior. Reduce it first -- sage.search.sources holds one handler "
            "per release, each of which selects the sample and writes the flat form "
            "this reads"
        )
    return {str(k): float(v) for k, v in payload.items()}


def _population(spec):
    """
    Every hyperposterior sample, for the marginalised draw.

    Read from the same canonical file as the representative, so a campaign cannot
    marginalise over one release while quoting another as its population.
    """
    from sage.search.sources.gwtc3_powerlawpeak import population

    path = Path(spec.injection.hyperposterior_path)
    try:
        return population(path)
    except KeyError as error:
        raise ValueError(
            f"injection.population_mode is 'marginalise' but {path.name} carries only "
            "its representative sample. Rebuild it with store_population=True -- "
            "runs/search/fetch_sources.py does by default"
        ) from error


def _population_digest(hyperposterior) -> str:
    """
    Short digest of the population a set was drawn from.

    Two campaigns drawing at different hyperparameters produce different populations
    under the same seed, so the seed alone is not enough to decide a staged table is
    still the right one. Taken over whatever was actually drawn from -- the whole
    posterior when marginalising, the one sample when not -- because a digest of the
    representative would let two campaigns marginalising over different releases share a
    staged table.
    """
    import hashlib
    import json

    payload = json.dumps(hyperposterior, sort_keys=True, default=repr).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _staged_table(spec, stream: int, columns, provenance_attrs, build):
    """
    The drawn parameter set, from the campaign's own directory where it is already there.

    Drawing is seeded, so this is a cache rather than a source of truth -- but a campaign
    that scored a set has to keep the set it scored. Without it the injections exist only
    inside the process that scored them, and the parameters behind ``p(x | signal)``
    cannot be read back, plotted, or compared against what was recovered.

    Reused only when the stored provenance matches the one asked for: the draw count, the
    seed, the hyperposterior it was drawn at and the sampler's column names. Any of those
    differing describes a different population, so the set is redrawn and replaced rather
    than a stale table being scored under the current configuration's name.
    """
    import h5py
    import numpy as np
    import torch

    path = spec.injection.staged_path or spec.path(
        "injections", f"injection_table_{stream:02d}.h5"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        with h5py.File(path, "r") as handle:
            stored = {key: handle.attrs.get(key) for key in provenance_attrs}
            stored_columns = [str(c) for c in handle.attrs.get("columns", ())]
            matches = stored_columns == list(columns) and all(
                stored.get(key) == value for key, value in provenance_attrs.items()
            )
            if matches:
                return torch.as_tensor(
                    np.asarray(handle["parameters"]), device=spec.engine.device
                )

    table = build()
    from sage.utils.atomic_io import atomic_h5

    with atomic_h5(path, mode="w") as handle:
        handle.create_dataset(
            "parameters", data=np.asarray(_detach(table), dtype=np.float64)
        )
        handle.attrs["columns"] = list(columns)
        for key, value in provenance_attrs.items():
            handle.attrs[key] = value
    return table


def run_campaign(spec, stream: int = 0, **kwargs) -> CampaignReport:
    """
    Score one injection stream: draw, inject, and run the search's own forward path.

    The keep threshold is ``-inf``. Every injection's statistic is kept, because the
    product is the *distribution* of the statistic over signals -- ``p(x | signal)`` --
    and thresholding it would truncate the density on one side only while the noise
    density kept its full range.
    """
    import torch

    from sage.search.checkpoint import as_config, load_search_model
    from sage.search.decode import PEDecoder
    from sage.search.engine import SearchEngine, build_param_sampler, build_processor
    from sage.search.grid import AnalysisGrid
    from sage.search.injection.population import (
        sample_intrinsic_marginalised,
        sample_intrinsic_torch,
    )
    from sage.search.injection.waveforms import (
        TabulatedSampler,
        build_injection_table,
        in_training_prior,
    )
    from sage.search.manifest import provenance
    from sage.search.segments import coincident_intervals, load_segments
    from sage.search.triggers import TriggerWriter

    geometry = spec.geometry_object()
    segments = {
        detector: load_segments(
            Path(spec.data.release_dir)
            / f"data_{detector}_{spec.data.observing_run}_segments.json"
        )
        for detector in spec.data.detectors
    }
    grid = AnalysisGrid.build(geometry, segments, coincident_intervals(segments))

    model, ckpt = load_search_model(
        spec.engine.checkpoint, cfg=None, data_cfg=None,
        device=spec.engine.device, architecture=spec.engine.architecture,
    )
    cfg, data_cfg = as_config(ckpt.cfg), as_config(ckpt.data_cfg)
    spec.apply_shadow_overrides(cfg, data_cfg)
    base_sampler = build_param_sampler(
        cfg, data_cfg, spec.engine.gwconfig, seed=int(spec.engine.sampler_seed)
    )

    hyperposterior = _hyperposterior(spec)
    draw_seed = int(spec.seed) + int(stream)
    low, high = base_sampler.bounds["mass1"]
    mode = str(spec.injection.population_mode)

    def _draw():
        n_draw = int(spec.injection.n_draw)
        if mode == "marginalise":
            intrinsic = sample_intrinsic_marginalised(
                _population(spec),
                n_draw,
                n_hyper=spec.injection.n_hyper,
                device=spec.engine.device,
                seed=draw_seed,
            )
        else:
            intrinsic = sample_intrinsic_torch(
                hyperposterior, n_draw, device=spec.engine.device
            )
        drawn = build_injection_table(base_sampler, _detach(intrinsic), seed=draw_seed)
        keep = in_training_prior(drawn, base_sampler, float(low), float(high))
        return drawn[torch.as_tensor(np.asarray(keep), device=drawn.device)]

    # Staged after the chirp-mass cut, so what is stored is the set that was scored.
    columns = sorted(base_sampler.param_index, key=base_sampler.param_index.get)
    table = _staged_table(
        spec,
        stream,
        columns,
        {
            "n_draw": int(spec.injection.n_draw),
            "draw_seed": draw_seed,
            "sampler_seed": int(spec.engine.sampler_seed),
            "hyperposterior": _population_digest(
                _population(spec) if mode == "marginalise" else hyperposterior
            ),
            "population_mode": mode,
            "n_hyper": int(spec.injection.n_hyper or 0),
            "mass1_bounds": f"{float(low)}:{float(high)}",
        },
        _draw,
    )

    sampler = TabulatedSampler(base_sampler, table)
    decoder = PEDecoder(
        targets=tuple(ckpt.cfg.get("do_point_estimate", ("tc", "mchirp"))),
        param_sampler=base_sampler,
        pe_target_minmax=bool(ckpt.cfg.get("pe_target_minmax", False)),
        geometry=geometry,
    )
    engine = SearchEngine(
        model, build_processor(base_sampler), geometry,
        device=spec.engine.device, amp_dtype=spec.engine.amp_dtype,
        keep_threshold=float("-inf"),
        autocast=bool(ckpt.cfg.get("autocast", True)),
        decoder=decoder,
    )

    shard = spec.path("injections", f"injection_triggers_{stream:02d}.h5")
    shard.parent.mkdir(parents=True, exist_ok=True)
    attrs = dict(provenance(spec))
    attrs.update(
        clustered=False, slide_id=0, stage="injections", stream=int(stream),
        keep_threshold=float("-inf"), n_draw=int(spec.injection.n_draw),
        n_kept=int(table.shape[0]),
    )
    writer = TriggerWriter(shard, attrs)
    noise = NoiseSlices(spec, grid, seed=int(spec.seed) + int(stream))
    try:
        report = InjectionCampaign(
            spec, engine,
            InjectionSet(spec, table, _generator(sampler), sampler, stream=stream),
            noise, writer,
        ).run()
    finally:
        writer.close()
        noise.close()
    return report


def run(spec, **kwargs) -> dict:
    """
    Stage entry point :func:`sage.search.stages.run_stage` dispatches to.

    One stream per call. Multiple streams exist so a campaign can be split across array
    tasks; each writes its own shard and p_astro reads them together.
    """
    reports = [
        run_campaign(spec, stream=int(stream), **kwargs)
        for stream in (spec.injection.streams or (0,))
    ]
    shards = [
        str(spec.path("injections", f"injection_triggers_{r.stream:02d}.h5"))
        for r in reports
    ]
    from sage.search.fingerprint import combine, digest_h5

    return {
        "shards": shards,
        "streams": [r.stream for r in reports],
        "n_scored": int(sum(r.n_scored for r in reports)),
        "reports": [r.as_dict() for r in reports],
        "fingerprint": combine(
            sum(r.n_scored for r in reports), digest_h5(shards)
        ),
    }
