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


#: Injections per committed block, rounded down to a whole number of generator batches.
#:
#: A commit is crash-atomic: it snapshots the shard, appends to the copy and renames. The
#: snapshot costs the *whole* shard, so committing every generator batch made the campaign
#: quadratic in its own length -- 4.4 M injections at a 2,048 batch is 2,129 commits of a
#: shard growing to 235 MB, about 250 GB copied to protect 235 MB of work. At this size it
#: is ~33 commits and 4 GB, and a killed job replays at most this many injections.
COMMIT_ROWS = 131_072


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

        Scored in the approximant's own batches and committed in larger blocks, so a
        requeued job resumes at a block boundary and cannot rescore an injection it
        already wrote -- which would put the same signal into ``p(x | signal)`` twice.

        The two sizes are separate because they are bounded by different things. A batch
        is what the generator emits and what fits on the card. A commit snapshots the
        shard before appending to it, so its cost grows with the shard while the work it
        protects does not: committing every batch made the campaign quadratic in its own
        length. See :data:`COMMIT_ROWS`.
        """
        started = time.perf_counter()
        # The approximant's batch is fixed when its frequency grid is built, and
        # `forward` takes no size -- it returns `generator.B` signals whatever is asked
        # of it. Batching at anything else silently misaligns signals against noise.
        batch_size = int(self.injections.batch_size)
        per_block = max(1, COMMIT_ROWS // batch_size) * batch_size
        done = set(self.writer.completed_blocks()) if resume else set()
        total = len(self.injections)
        scored = 0

        for block_id, block_lo in enumerate(range(0, total, per_block)):
            if block_id in done:
                continue
            block_hi = min(block_lo + per_block, total)
            for lo in range(block_lo, block_hi, batch_size):
                hi = min(lo + batch_size, block_hi)
                spectra, params = self.injections.build(lo, hi, self.noise)
                stat, point = self.engine.forward_frequency(spectra)
                self.writer.append(_table(lo, stat, point, params, self.engine))
                scored += int(stat.size)
            self.writer.complete_block(block_id)

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
        self._grid = grid
        self._geometry = spec.geometry_object()
        self._detectors = tuple(spec.data.detectors)
        self._rng = np.random.default_rng(int(seed))
        self._release = Path(spec.data.release_dir)
        self._run = str(spec.data.observing_run)
        self._maps = {}
        # One flat list of (segment, first_local, n_windows) per detector, so a draw is
        # an index. Taken from the lattice's own per-detector runs rather than from
        # grid.spans_by_detector, which holds the reference detector alone: a follower's
        # windows live on its own segments at its own local offsets, and there is no
        # entry there to read them from.
        self._spans = {
            detector: [
                (run.segment, run.first_local, run.n_windows)
                for run in grid.runs_for_detector(detector)
                if run.n_windows
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

    def _stream(self, detector: str):
        """
        The detector's strain, opened once.

        Opened through the reader's own chooser so that the noise an injection is added to
        comes off the release by the same route the search reads it. The layout is decided
        by the sidecar, and the search-grade release is one HDF5 dataset per segment
        rather than the training releases' flat stream.
        """
        if detector not in self._maps:
            from sage.search.reader import open_stream

            self._maps[detector] = open_stream(
                self._release, detector, self._grid.segments_by_detector.get(detector)
            )
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
        # float32, which is what the release stores and what the engine transforms in.
        # A float64 buffer here widened every window on the way in only for the signal
        # addition to narrow it again, at 1 GB allocated and freed per batch.
        noise = np.empty((n, len(self._detectors), window), dtype=np.float32)
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
                noise[row, column] = self._stream(detector).read(
                    segment, offset, window
                )
                if column == 0:
                    segment_index[row] = int(segment.segment_index)
                    local_start[row] = int(offset)
        return noise, segment_index, local_start

    def close(self) -> None:
        """Release the open strain streams."""
        for stream in self._maps.values():
            stream.close()
        self._maps.clear()


def scored_shards(spec):
    """
    The injection shards this campaign declares, in stream order.

    One shard per stream, named where they are written. Two readers had each spelled the
    name themselves and both spelled it without the stream, so p_astro and its figure
    looked for a file no campaign has ever produced.
    """
    return [
        spec.path("injections", f"injection_triggers_{int(stream):02d}.h5")
        for stream in tuple(spec.injection.streams or (0,))
    ]


def scored_stats(spec) -> np.ndarray:
    """
    Ranking statistics of every scored injection, which are ``p(x | signal)``.

    All streams together. A campaign is split across streams so it can be spread over
    array tasks, and every stream draws from the same population into the same run's
    noise, so reading one would fit the signal density on a fraction of the injections
    while reporting the whole campaign.
    """
    from sage.search.triggers import read_shard

    shards = scored_shards(spec)
    missing = [str(path) for path in shards if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"no scored injections at {', '.join(missing)}; p(x | signal) is the "
            "distribution of the ranking statistic over recovered injections, so the "
            "injections stage must run first -- for every stream the campaign declares, "
            "since the density is fitted on all of them. It is not a sensitive volume "
            "and cannot be substituted by one"
        )
    stats = [
        np.asarray(read_shard(shard)[0].columns["stat"], dtype=np.float64)
        for shard in shards
    ]
    return np.concatenate(stats) if stats else np.zeros(0, dtype=np.float64)


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

    @property
    def batch_size(self) -> int:
        """
        Injections per forward pass, fixed by the approximant.

        The frequency grid the approximant is built on carries the batch dimension, so
        ``forward`` returns that many signals and takes no count. Reading it here is what
        keeps the campaign's batching and the generator's agreeing.
        """
        return int(self.generator.B)

    def build(self, lo: int, hi: int, noise):
        """
        Signal plus real noise, for injections ``[lo, hi)``, as spectra.

        The signal is generated through the same approximant, projection and SNR
        convention the network was trained under. The approximant emits the projected
        strain in the frequency domain, and the noise is transformed with the engine's own
        convention before the two are added -- which is where training adds them, and the
        transform is linear, so this is adding a signal to a detector's output. The
        whitening and multirate binning then run inside the engine, on the sum, exactly as
        they do for a search window.

        The sampler is sought to ``lo`` rather than read sequentially, so a resumed
        campaign that skips completed batches still gives every injection its own row.
        """
        import torch

        n = int(hi) - int(lo)
        size = self.batch_size
        total = len(self)
        if n > size:
            raise ValueError(
                f"asked for {n} injections in one batch against a generator that "
                f"produces {size}; the campaign batches at generator.B"
            )
        if n == size:
            self.sampler.seek(int(lo))
            keep = slice(0, size)
        else:
            # The table's last rows do not fill a batch, and the approximant emits a whole
            # one whatever is asked. Rather than drop the remainder from p(x | signal),
            # the generator is run over the batch that *ends* at the table's end and only
            # its last `n` rows are kept. The rows before them were scored by the previous
            # batch and are discarded here, so nothing is counted twice and nothing is
            # lost.
            if int(hi) != total:
                raise ValueError(
                    f"a short batch of {n} ends at row {hi} of {total}; only the table's "
                    "last batch may be short, and the rows kept from a short batch are "
                    "taken from the end of the table"
                )
            if total < size:
                raise ValueError(
                    f"the campaign drew {total} injections against a generator batch of "
                    f"{size}; a stream shorter than one batch cannot be scored, so raise "
                    "n_draw or lower the training batch the approximant was built with"
                )
            self.sampler.seek(total - size)
            keep = slice(size - n, size)
        strain_noise, segment_index, local_start = noise.draw(n)
        with torch.no_grad():
            # No count: `forward`'s only positional argument is `return_theta`, so a size
            # passed here was read as a truthy flag and quietly returned a third tensor.
            signal = _tensor(self.generator())[keep]
        spectra = self.noise_spectra(strain_noise, signal)
        if signal.shape[0] != n:
            raise ValueError(
                f"the generator produced {signal.shape[0]} signals for a batch of {n}; "
                "the campaign batches at generator.B so that every injection meets its "
                "own noise, and a mismatch would pair them off by position"
            )
        if signal.shape != spectra.shape:
            raise ValueError(
                f"the generator returned {tuple(signal.shape)} against noise of "
                f"{tuple(spectra.shape)}; the two are added bin by bin, so a mismatch "
                "would broadcast a signal across the wrong detectors or frequencies"
            )
        return (
            spectra + signal.to(spectra.dtype),
            {"segment_index": segment_index, "local_start": local_start},
        )

    def noise_spectra(self, strain_noise, like):
        """
        Real noise in the domain the signal is generated in.

        ``norm="forward"`` and float32, matching :meth:`SearchEngine._to_frequency_domain`
        exactly: the fiducial whitening buffer is scaled to that convention, and an
        unnormalised transform gives an output that looks reasonable and is wrong by a
        factor of N.
        """
        import torch

        noise = torch.as_tensor(
            np.ascontiguousarray(strain_noise), device=like.device, dtype=torch.float32
        )
        return torch.fft.rfft(noise, dim=-1, norm="forward")


def _detach(value):
    """A tensor, or the first element of a tuple of them, as a numpy array."""
    import torch

    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tensor(value):
    """
    The strain a generator returned, still a tensor and still on its device.

    ``forward`` returns ``(hf, targets)``; only the first is strain. Kept on the device
    and in the complex dtype it was generated in, because the noise is transformed to meet
    it rather than the other way round.
    """
    if isinstance(value, (tuple, list)):
        value = value[0]
    return value


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
    seed, the hyperposterior it was drawn at, the mass frame it was built in and the
    sampler's column names. Any of those differing describes a different population, so
    the set is redrawn and replaced rather than a stale table being scored under the
    current configuration's name.

    Returns
    -------
    (table, reused)
        ``reused`` is False when the set was redrawn. The caller needs it: a shard's rows
        are indexed by row number into this table, so if the table changed then every row
        already scored describes a different binary, and resuming onto it would build
        ``p(x | signal)`` out of two different populations.
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
                return (
                    torch.as_tensor(
                        np.asarray(handle["parameters"]), device=spec.engine.device
                    ),
                    True,
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
    return table, False


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
    grid = AnalysisGrid.build(
        geometry,
        segments,
        coincident_intervals(segments),
        reference_detector=spec.slides.reference_detector,
        coverage=False,
    )

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
                hyperposterior, n_draw, device=spec.engine.device, seed=draw_seed
            )
        drawn = build_injection_table(base_sampler, _detach(intrinsic), seed=draw_seed)
        keep = in_training_prior(drawn, base_sampler, float(low), float(high))
        return drawn[torch.as_tensor(np.asarray(keep), device=drawn.device)]

    # Staged after the chirp-mass cut, so what is stored is the set that was scored.
    columns = sorted(base_sampler.param_index, key=base_sampler.param_index.get)
    table, reused = _staged_table(
        spec,
        stream,
        columns,
        {
            # The convention the masses were written in. Not decoration: the same draw
            # count, seed and hyperposterior describe a different table either side of
            # the source-to-detector frame fix, so without this a campaign would reuse a
            # source-frame table under the current configuration's name. See SB-50.
            "mass_frame": "detector",
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
    if not reused and shard.is_file():
        # A shard's rows are indexed by row number into the staged table. The table was
        # just redrawn, so every row already in the shard describes a different binary,
        # and TriggerWriter would resume onto it -- reporting a complete campaign whose
        # statistics came from two different populations. The stage is idempotent on an
        # unchanged configuration precisely because this case is the one that is not.
        shard.unlink()
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
