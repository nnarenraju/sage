#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : engine.py
Description   : The inference loop; mirrors the trained forward contract exactly.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The forward path reproduces sage.factory.testing.SageVanillaTesting._forward:
rfft(norm="forward") -> GWBatch(Grid.FD_UNIFORM) -> Preprocessor([FiducialWhitening,
MultirateSampler]) -> autocast -> model. The ranking head is fp32 because a bf16
output cast quantises the logit at the scale where the FAR threshold sits.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from sage.search.fingerprint import combine, digest_h5
from sage.search.geometry import SearchGeometry
from sage.search.grid import AnalysisGrid, Block
from sage.search.spec import SearchSpec


@dataclass
class EngineReport:
    """Throughput and completion accounting for one engine run."""

    n_windows: int
    n_triggers: int
    wall_seconds: float
    windows_per_second: float
    blocks_completed: int

    def as_dict(self) -> dict:
        """Flat dict for the manifest."""
        return {
            "n_windows": int(self.n_windows),
            "n_triggers": int(self.n_triggers),
            "wall_seconds": float(self.wall_seconds),
            "windows_per_second": float(self.windows_per_second),
            "blocks_completed": int(self.blocks_completed),
        }


_AMP_DTYPES = ("bfloat16", "float16", "float32")


class SearchEngine:
    """
    Score a window lattice and emit thresholded triggers plus an exact histogram.

    The forward path is not reimplemented here. It is
    :func:`sage.factory.contract.forward_batch`, the same function training, validation
    and benchmarking call, so the network is fed exactly what it was trained on. A search
    that re-derived the path would drift from it silently -- the failure produces plausible
    numbers rather than an error.

    Parameters
    ----------
    keep_threshold : float
        Ranking-statistic threshold above which individual triggers are written.
        Derived once from the complete zero-lag histogram and frozen for the whole
        campaign, so it is never calibrated on a subsample.
    cache : FrontendCache, optional
        Per-detector frontend features, reused across slides. Valid only for a separable
        frontend; :func:`sage.search.checkpoint.assert_separable` is what establishes that,
        and it is the caller's job to have run it.
    """

    def __init__(
        self,
        model,
        processor,
        geometry: SearchGeometry,
        device: str = "cuda",
        amp_dtype: str = "bfloat16",
        keep_threshold: float = 0.0,
        cache=None,
        autocast: bool = True,
        decoder=None,
    ) -> None:
        import torch

        if amp_dtype not in _AMP_DTYPES:
            raise ValueError(
                f"amp_dtype must be one of {_AMP_DTYPES}, got {amp_dtype!r}"
            )
        self.model = model
        self.processor = processor
        self.geometry = geometry
        self.device = torch.device(device)
        self.amp_dtype = getattr(torch, amp_dtype)
        self.keep_threshold = float(keep_threshold)
        self.cache = cache
        self.autocast = bool(autocast)
        self.decoder = decoder
        self._split_network = None
        self.model.eval()

    def _to_frequency_domain(self, strain):
        """
        Raw ``(B, D, T)`` strain to the complex ``(B, D, F)`` the contract expects.

        ``norm="forward"`` divides by N and the fiducial whitening buffer is scaled to
        match; an unnormalised transform produces an output that looks reasonable and is
        wrong by a factor of N. The window must be the padded length the network was
        trained on -- a shorter one changes the frequency resolution, which the whitening
        buffer is indexed by.
        """
        import torch

        if not torch.is_tensor(strain):
            strain = torch.as_tensor(np.ascontiguousarray(strain))
        if strain.ndim != 3:
            raise ValueError(
                f"strain must be (batch, detectors, samples); got {tuple(strain.shape)}"
            )
        expected = self.geometry.window_samples
        if strain.shape[-1] != expected:
            raise ValueError(
                f"strain windows are {strain.shape[-1]} samples but the geometry says "
                f"{expected}; the whitening buffer is indexed by frequency, so a window "
                "of a different length is whitened by the wrong bins"
            )
        strain = strain.to(device=self.device, dtype=torch.float32, non_blocking=True)
        return torch.fft.rfft(strain, dim=-1, norm="forward")

    def forward(self, strain) -> Tuple["np.ndarray", "np.ndarray"]:
        """Score a raw strain batch; returns ``(ranking_statistic, point_estimates)``."""
        import torch

        from sage.factory.contract import forward_batch

        with torch.inference_mode():
            output = forward_batch(
                self._to_frequency_domain(strain),
                self.model,
                self.processor,
                amp_dtype=self.amp_dtype,
                autocast=self.autocast,
                device_type=self.device.type,
            )
            # forward_batch concatenates the network's two outputs and casts to float32:
            # column 0 is the ranking statistic, the rest the raw PE means then sigmas.
            stat = output[:, 0].float().cpu().numpy()
            point = output[:, 1:].float().cpu().numpy()
        return stat, point

    def forward_frontend(self, strain, detector: int):
        """
        Run the per-detector path only, for the frontend cache.

        Delegates to :class:`~sage.search.network.SplitNetwork`, which composes the half
        from the model's own submodules and has already checked that composition against
        ``model.forward`` bitwise. The engine deliberately does not re-derive it: two
        copies of a network's internals drift the moment the architecture is refactored,
        and both keep returning plausible numbers.

        Valid only because the frontend is separable -- the input normalisation is per
        channel, so one detector's features do not depend on what the others held. That is
        what makes a feature computed once reusable across every slide that re-pairs this
        detector. :func:`sage.search.checkpoint.assert_separable` is what establishes it,
        and ``EngineSpec.use_frontend_cache`` is off by default precisely because the
        property is a fact about the trained network rather than a given.
        """
        import torch

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.autocast,
            ):
                return self._split().frontend(self._prepared(strain), detector)

    def forward_backend(self, features):
        """
        Run the shared backend on re-paired cached features.

        ``features`` is the per-detector frontend output in network order, either as a
        sequence or already concatenated on the channel axis.
        """
        import torch

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.autocast,
            ):
                stat, point = self._split().backend(features)
            stat = stat.reshape(stat.shape[0]).float().cpu().numpy()
            point = point.float().cpu().numpy()
        return stat, point

    def _split(self):
        """
        The model's two halves, wrapped and verified once per engine.

        Built lazily and cached: the verification is a forward pass on a two-window probe,
        which is negligible against a campaign but not against a batch.
        """
        if self._split_network is None:
            from sage.search.network import SplitNetwork

            self._split_network = SplitNetwork(self.model, verify=True)
        return self._split_network

    def _prepared(self, strain):
        """Whitened, multirate-sampled input: everything up to the network itself."""
        import torch

        from sage.core.pipeline import GWBatch, Grid, ProcessingState

        spectra = self._to_frequency_domain(strain)
        batch = GWBatch(spectra, state=ProcessingState(Grid.FD_UNIFORM))
        return self.processor(batch).to_network_input()

    def run_block(self, reader, block: Block, writer) -> EngineReport:
        """
        Score one block and append to its shard.

        Every window is counted in the histogram; only those above ``keep_threshold`` are
        written as rows. The histogram is the exact count the false-alarm rate is measured
        from, so it must see the whole lattice -- thresholding before counting would make
        the denominator describe less data than the search covered.
        """
        from sage.search.triggers import TriggerTable, histogram_stats

        started = time.perf_counter()
        n_windows = 0
        n_triggers = 0
        for batch in reader.iter_block(block):
            stat, point = self.forward(batch.strain)
            n_windows += stat.size
            writer.add_histogram(histogram_stats(stat, clustered=False))
            if writer.keep_stream:
                # The statistic alone. The stream carries one row per window of the whole
                # run, which is what makes it the largest product the search writes, and
                # it exists to hold the quantile the keep threshold is frozen from and to
                # show the zero-lag distribution. Adding the decoded parameters would
                # multiply that by the number of heads to store values the trigger table
                # already holds for every window anyone will look at.
                writer.add_stream(stat)
            loud = stat > self.keep_threshold
            if not np.any(loud):
                continue
            table = self._table(batch, stat, point, loud)
            writer.append(table)
            n_triggers += len(table)
        writer.complete_block(block.block_id)
        elapsed = time.perf_counter() - started
        return EngineReport(
            n_windows=n_windows,
            n_triggers=n_triggers,
            wall_seconds=elapsed,
            windows_per_second=n_windows / elapsed if elapsed > 0 else 0.0,
            blocks_completed=1,
        )

    def _table(self, batch, stat, point, loud) -> "object":
        """Assemble the schema columns for the windows that passed the threshold."""
        from sage.search.triggers import TriggerTable

        columns: Dict[str, np.ndarray] = {
            "stat": np.asarray(stat, dtype=np.float64)[loud],
            "gps": np.asarray(batch.gps, dtype=np.float64)[loud],
            "segment_index": np.asarray(batch.segment_index, dtype=np.int64)[loud],
            "local_start": np.asarray(batch.local_start, dtype=np.int64)[loud],
            "slide_id": np.full(int(np.count_nonzero(loud)), batch.slide_id, np.int64),
        }
        if self.decoder is not None:
            decoded = self.decoder.trigger_columns(
                np.asarray(point)[loud], columns["gps"]
            )
            columns.update(decoded)
        return TriggerTable(columns=columns, attrs={"clustered": False})

    def run(self, reader, grid: AnalysisGrid, writer, resume: bool = True) -> EngineReport:
        """
        Score a whole lattice, skipping blocks already marked complete.

        Resumption is by block because the block is what the writer commits atomically:
        a shard holds exactly the blocks that finished, so replaying one that did not
        cannot double-count and skipping one that did cannot drop anything.
        """
        done = set(writer.completed_blocks()) if resume else set()
        started = time.perf_counter()
        n_windows = 0
        n_triggers = 0
        blocks = 0
        if len(grid) == 0:
            # A slide whose lag carries every window off the end of the run analyses no
            # time at all. That is a measurement, not a failure: it contributes zero
            # events over zero livetime, and both the numerator and the denominator
            # already say so. The shard is still written and finalised, because "measured,
            # empty" and "never ran" are exactly the two the collation must tell apart.
            return EngineReport(
                n_windows=0,
                n_triggers=0,
                wall_seconds=time.perf_counter() - started,
                windows_per_second=0.0,
                blocks_completed=0,
            )
        for block in _blocks_of(reader, grid):
            if block.block_id in done:
                continue
            report = self.run_block(reader, block, writer)
            n_windows += report.n_windows
            n_triggers += report.n_triggers
            blocks += 1
        elapsed = time.perf_counter() - started
        return EngineReport(
            n_windows=n_windows,
            n_triggers=n_triggers,
            wall_seconds=elapsed,
            windows_per_second=n_windows / elapsed if elapsed > 0 else 0.0,
            blocks_completed=blocks,
        )


def _blocks_of(reader, grid: AnalysisGrid):
    """
    The blocks to walk: the reader's own, so both sides agree on what a block id names.

    Taken rather than recomputed. Deriving the partition from the blocks themselves --
    ``max(block.duration_s)`` -- was wrong and silently so: ``duration_s`` is a block's
    **wall span**, gaps included, while the partition is budgeted in **livetime**. On the
    O3a lattice the largest wall span is 254,401 s against a 32,768 s budget, so the
    engine re-partitioned into 5 blocks where the reader held 30.

    Everything was still scored, because a block carries the span slice both sides index
    through -- which is why nothing failed. What broke was the bookkeeping: the shard
    recorded 5 completed blocks against an ``n_blocks`` of 30, resume granularity was a
    fifth of the run rather than a thirtieth, and the frontend cache residency for one
    coarse block came to 117 GB on an 80 GB card.
    """
    blocks = getattr(reader, "blocks", None)
    if blocks:
        return list(blocks)
    block_seconds = getattr(reader, "block_seconds", None)
    if not block_seconds:
        raise ValueError(
            "this reader exposes neither blocks nor the block_seconds it was built "
            "with, so the engine cannot walk the same partition it does; a block id "
            "would name different data in each"
        )
    return grid.blocks(float(block_seconds))


def build_param_sampler(cfg, data_cfg, gwconfig: str | Path, seed: int):
    """
    The training run's parameter sampler, with its encoding buffers compiled.

    One sampler serves both consumers: the dyadic binning reads its ``bounds``, and
    :class:`~sage.search.decode.PEDecoder` reads the buffers that invert the target
    encoding. Building it once is not an optimisation -- two samplers could be given
    different priors or different seeds, and the binning would then be laid out for one
    prior while the decode inverted another.

    ``register_configs`` comes first because the sampler reads the global device, dtype
    and ``do_point_estimate`` from there rather than from arguments. That indirection is
    the training path's, and the search matches it rather than working around it.

    Both compile steps are called explicitly. In training they are called by the waveform
    generator's constructor (``IMRPhenomPv2.__init__``), and the search builds no
    generator because it generates no waveforms -- so without these the buffers are never
    registered and the decode fails on a missing attribute.

    Parameters
    ----------
    gwconfig : path
        The parameter prior the network was trained under. It fixes both the bin layout
        and the decode, so it must be the training run's own ``gwconfig.yaml`` and is
        required rather than defaulted: a different mass prior gives a different bin
        layout, which changes what the network is fed without changing anything that
        would fail.
    seed : int
        Seed for the sampler's generator. It reaches the decode through
        ``_compile_batch_standardiser``, which estimates each target's mean and standard
        deviation from a million draws -- so the buffers carry Monte Carlo noise and the
        seed is part of the result.

    Notes
    -----
    **There is no exactly-right seed, and the spread is measured rather than assumed.**
    ``runs/*/train_hard.py`` derives its seed from the resume epoch
    (``BASE_SEED + SEED_STRIDE * K``), so a run that resumed recompiled these buffers
    against a different draw each time and the network was trained across all of them.
    Measured on ``runs/o3b/gwconfig.yaml`` between seeds 150914 and 170817, the resulting
    decode differs by at most 1e-4 s in ``tc`` -- a thousandth of the 0.1 s stride and a
    hundredth of the H1-L1 light travel time -- and by at most 0.013 solar masses in
    ``mchirp``. The min-max encoding used when ``pe_target_minmax`` is set is built from
    theoretical bounds and is seed-independent; only the standardised path is affected.

    So the seed is recorded rather than reasoned about: whatever value the campaign used
    is in the provenance block, and the number above says what a different one would have
    cost.
    """
    from sage.core.config import register_configs
    from sage.data.waveform import read_from_config

    path = Path(gwconfig)
    if not path.is_file():
        raise FileNotFoundError(
            f"no parameter prior at {path}; the multirate binning is built from the mass "
            "prior the network was trained under, and cannot be guessed from the "
            "checkpoint -- it records the geometry, not the prior"
        )
    register_configs(cfg, data_cfg)
    sampler = read_from_config(str(path), seed=int(seed))
    sampler._compile_batch_normaliser()
    sampler._compile_batch_standardiser()
    return sampler


def build_processor(param_sampler):
    """
    Assemble the FiducialWhitening + MultirateSampler graph used in training.

    Takes the sampler rather than a path, so the bin layout and the decode are provably
    built from one prior. :func:`build_param_sampler` has already registered the configs
    that :class:`~sage.dsp.whiten.FiducialWhitening` reads the fiducial directory, the
    padded length and the sample rate from.
    """
    from sage.factory.contract import make_processor

    return make_processor(param_sampler.bounds)


def run_search(
    spec: SearchSpec,
    stage: str = "zerolag",
    slide_id: int = 0,
    offsets_s: Optional[dict] = None,
    window_shift: Optional[dict] = None,
    **kwargs,
) -> dict:
    """
    Stage driver: build everything from ``spec`` and score one pass.

    One pass is one slide. ``slide_id=0`` is the zero-lag pass over the observing run; any
    other is one rung of the background ladder, and differs only in the offsets the lattice
    is built with. The same driver serves the ``zerolag`` and ``background`` stages for
    that reason -- two drivers would be two forward paths that could drift.

    The pairing -- ``offsets_s`` for a lag ladder, ``window_shift`` for a roll along the
    lattice -- is supplied by the caller rather than read from the stored ladder. The
    background driver owns the plan and is scheduled after ``slides``; ``zerolag`` is
    scheduled before it. An engine that loaded the plan itself would make the earliest
    scoring stage depend on a product built two stages later, which is the ordering the
    stage graph exists to prevent.

    The order is deliberate and each step is cheap relative to the one after it: resolve
    the geometry, read and validate the checkpoint, prove separability if the frontend
    cache is wanted, build the processor from the training prior, then open the strain.
    Discovering a configuration mismatch after a GPU-hour of scoring is the failure this
    ordering exists to prevent.

    Returns
    -------
    dict
        The engine report plus the shard path and a ``fingerprint`` over what was written,
        which is what :func:`sage.search.stages.run_stage` compares to decide whether the
        downstream chain needs rebuilding.
    """
    from sage.search.checkpoint import as_config, load_search_model
    from sage.search.decode import PEDecoder
    from sage.search.grid import AnalysisGrid
    from sage.search.manifest import provenance
    from sage.search.reader import StreamingStrainReader
    from sage.search.segments import coincident_intervals, load_segments
    from sage.search.triggers import TriggerWriter

    # Frozen once for the whole campaign from the complete zero-lag histogram, and passed
    # in rather than derived here: a threshold calibrated inside a slide job would be
    # calibrated on that slide's own triggers, and every slide would keep a different
    # amount of its own tail. Read before anything is built, so the value stamped on the
    # shard and the value the engine thresholds on are the same one by construction.
    keep_threshold = float(kwargs.pop("keep_threshold", 0.0))

    geometry = spec.geometry_object()
    release = Path(spec.data.release_dir)
    run_name = spec.data.observing_run
    segments = {
        detector: load_segments(
            release / f"data_{detector}_{run_name}_segments.json"
        )
        for detector in spec.data.detectors
    }

    if slide_id != 0 and not offsets_s and not window_shift:
        raise ValueError(
            f"slide {slide_id} was requested with neither offsets nor a window shift; "
            "the pairing comes from the "
            "stored ladder and are supplied by the background driver, which owns it. "
            "The engine deliberately does not read slide_plan.h5: zerolag runs before "
            "slides in the stage graph, so an engine that depended on the plan would "
            "invert the campaign's own ordering"
        )

    grid = AnalysisGrid.build(
        geometry,
        segments,
        coincident_intervals(segments),
        offsets_s=dict(offsets_s) if offsets_s else None,
        window_shift=dict(window_shift) if window_shift else None,
        slide_id=int(slide_id),
        reference_detector=spec.slides.reference_detector,
    )

    model, ckpt = load_search_model(
        spec.engine.checkpoint,
        cfg=None,
        data_cfg=None,
        device=spec.engine.device,
        require_separable=bool(spec.engine.use_frontend_cache),
        architecture=spec.engine.architecture,
    )
    # The checkpoint's configs describe the training run, and the sampler, the whitener
    # and the multirate binning all read them from the global registry rather than from
    # arguments. So the campaign's own device, batch size, fiducial spectra and output
    # directory are shadowed onto the per-process wrappers first; without this the search
    # runs on whatever device the training job used and writes through the training run's
    # export_dir. The geometry is left alone -- it is what validate_geometry checks.
    cfg, data_cfg = as_config(ckpt.cfg), as_config(ckpt.data_cfg)
    spec.apply_shadow_overrides(cfg, data_cfg)
    param_sampler = build_param_sampler(
        cfg, data_cfg, spec.engine.gwconfig, seed=int(spec.engine.sampler_seed)
    )
    processor = build_processor(param_sampler)
    decoder = PEDecoder(
        targets=tuple(ckpt.cfg.get("do_point_estimate", ("tc", "mchirp"))),
        param_sampler=param_sampler,
        pe_target_minmax=bool(ckpt.cfg.get("pe_target_minmax", False)),
        geometry=geometry,
    )

    engine = SearchEngine(
        model,
        processor,
        geometry,
        device=spec.engine.device,
        amp_dtype=spec.engine.amp_dtype,
        keep_threshold=keep_threshold,
        autocast=bool(ckpt.cfg.get("autocast", True)),
        decoder=decoder,
    )

    shard = spec.path(stage, f"{stage}_slide{int(slide_id):04d}.h5")
    shard.parent.mkdir(parents=True, exist_ok=True)
    attrs = dict(provenance(spec, ckpt))
    attrs["clustered"] = False
    attrs["slide_id"] = int(slide_id)
    attrs["stage"] = str(stage)
    # The pairing this shard was scored under, so a background collation can tell a
    # rolled slide from a lagged one without consulting the plan. A slide_id alone is
    # only meaningful against the plan that assigned it.
    attrs["pairing"] = "roll" if window_shift else "lag"
    if window_shift:
        for detector, shift in sorted(window_shift.items()):
            attrs[f"window_shift_{detector}"] = int(shift)
    # Part of the shard's identity, not a note about it: it decides which windows became
    # rows, so two rungs thresholded differently hold different fractions of their tails
    # and must not be collated into one background. Checked by COMPATIBILITY_KEYS.
    attrs["keep_threshold"] = keep_threshold
    # How many blocks this slide has. Stamped so a reader can tell a shard that finished
    # from one that was merely opened: "finalised" alone is set by a close that happens
    # to land on a block boundary.
    attrs["n_blocks"] = len(grid.blocks(float(spec.engine.block_seconds)))

    reader = StreamingStrainReader(
        release,
        grid,
        geometry,
        batch_size=int(spec.engine.batch_size),
        block_seconds=float(spec.engine.block_seconds),
        prefetch=2,
        pin_memory=True,
    )
    # The per-window stream is a zero-lag diagnostic and the writer refuses it for a
    # slide: it holds one value per analysed window, so a ladder would write one copy of
    # the whole run per rung. Gated here rather than left to fail, because the flag is a
    # campaign-wide setting and a campaign that wanted the zero-lag stream should not
    # find its background stage unrunnable as a result.
    writer = TriggerWriter(
        shard, attrs, keep_stream=bool(spec.engine.keep_stream) and slide_id == 0
    )
    try:
        report = engine.run(reader, grid, writer)
    finally:
        writer.close()
        reader.close()

    payload = report.as_dict()
    payload["shard"] = str(shard)
    payload["slide_id"] = int(slide_id)
    payload["n_lattice_windows"] = int(len(grid))
    # What a re-run would have to reproduce for the downstream chain to still describe
    # this data: the same windows scored and the same triggers kept -- so the shard's
    # contents, not a count of its rows. Counts are blind to every number the chain
    # actually reads. A different checkpoint, a changed preprocessor or a re-decoded tc
    # move every ranking statistic in the shard while leaving its row count exactly
    # where it was, and significance would then be computed from new triggers against
    # products fitted to the old ones.
    payload["fingerprint"] = combine(
        payload["n_windows"],
        payload["n_triggers"],
        len(grid),
        slide_id,
        digest_h5(shard),
    )
    return payload


def run(spec: SearchSpec, **kwargs) -> dict:
    """
    Entry point :func:`sage.search.stages.run_stage` dispatches to.

    Named ``run`` because that is the contract every stage module exposes; the work is
    :func:`run_search`, which is also callable directly for a single slide.
    """
    return run_search(spec, **kwargs)
