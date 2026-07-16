#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HDF5-backed hard-mining bank -- the single, file-resident source of truth for
the CMA-MAE hard-noise miner.

Design goals (set by the project):
  * Nothing persists in RAM between mining rounds.  Start-times, the diverse
    embedding bank, the IncrementalPCA state, and the per-round re-evaluation
    history all live in ONE HDF5 file under /work.
  * Continuous learning: the file is appended to forever after a single cold
    start; the miner is never rebuilt from scratch.
  * Generic / config-agnostic: ``runs`` and ``detectors`` are passed in (sourced
    from cfg), never hardcoded here.  The filename and root attributes are
    derived from them so each (runs, detectors) training setup gets its own bank.

Layout (see ``_ensure_file``)::

    attrs: detectors, runs, sample_rate, seq_len, descriptor_dim, cold_epoch
    start_times       (N, D) int64   append-only, stable row index
    start_segments    (N, D) int64
    start_runs        (N, D) int64    which pooled run each window came from
    start_found_epoch (N,)   int32
    start_found_score (N,)   f32
    embeddings        (M, P) f32      diverse subset (distance-gated, PCA-reduced)
    emb_epoch         (M,)   int32
    emb_score         (M,)   f32
    pca/{components,mean,var,n_seen}  persisted IncrementalPCA state
    eval_stats        (N, R) f32      re-eval ranking stat of every start per round
    eval_model_epoch  (R,)   int32    model epoch that produced each eval column

Concurrency: writes happen only during mining (between epochs); the sampler
reads during epochs.  The two never overlap, so a single file accessed by one
party at a time is safe -- no SWMR needed.
"""

import os
import numpy as np
import h5py

try:
    from scipy.spatial import cKDTree as _KDTree
except Exception:                                   # pragma: no cover
    _KDTree = None


def detectors_tag(detectors):
    """['H1','L1','V1'] -> 'HLV' (first letter of each)."""
    return "".join(str(d)[0].upper() for d in detectors)


def runs_tag(runs):
    """'O3b' or ['O3a','O3b','O4a'] -> 'O3b' / 'O3aO3bO4a'."""
    if isinstance(runs, str):
        return runs
    return "".join(str(r) for r in runs)


def default_bank_path(root, runs, detectors):
    """Standard per-(runs, detectors) bank path, e.g.
    ``/work/.../hard_mining/hardbank_O3aO3bO4a_HLV.h5``."""
    fname = f"hardbank_{runs_tag(runs)}_{detectors_tag(detectors)}.h5"
    return os.path.join(root, fname)


class HardMiningBank:
    """File-resident bank for hard-noise mining (one HDF5 file).

    Parameters
    ----------
    path : str
        HDF5 file path (use :func:`default_bank_path` to build a standard one).
    detectors : list[str]
        Detector names, ``D = len(detectors)``.  Sourced from cfg.
    runs : str or list[str]
        Observing run(s) the model trains on (e.g. ``"O3b"`` or
        ``["O3a","O3b","O4a"]``).  Sourced from cfg.  Stored as metadata only.
    seq_len, sample_rate, bin_files :
        Window/length/source metadata, mirrored into emitted datasets.
    descriptor_dim : int
        Width ``P`` of the stored (PCA-reduced) embeddings.
    novelty_dist : float
        Minimum distance a new embedding must have from every bank embedding to
        be kept (diversity gate).  Cosine distance by default.
    max_embeddings : int
        Soft cap on the embedding bank; when exceeded, the lowest-score (least
        hard) embeddings are dropped so the bank stays "diverse AND hard" and
        novelty queries stay fast.
    metric : {'cosine','euclidean'}
        Distance used for the novelty gate.
    """

    def __init__(self, path, detectors, runs, seq_len, sample_rate, bin_files,
                 descriptor_dim=8, novelty_dist=0.1, max_embeddings=50_000,
                 metric="cosine"):
        self.path = str(path)
        self.detectors = [str(d) for d in detectors]
        self.D = len(self.detectors)
        self.runs = runs_tag(runs)
        self.seq_len = int(seq_len)
        self.sample_rate = float(sample_rate)
        self.bin_files = [str(b) for b in bin_files]
        self.P = int(descriptor_dim)
        self.novelty_dist = float(novelty_dist)
        self.max_embeddings = int(max_embeddings)
        self.metric = metric
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._ensure_file()

    # ------------------------------------------------------------------ setup
    def _ensure_file(self):
        with h5py.File(self.path, "a") as f:
            if "start_times" in f:
                self._validate_existing(f)     # fail fast on incompatible reopen
                # Backfill start_runs on a pre-multi-run bank (all run 0) so an
                # older single-run bank keeps working after the schema change.
                if "start_runs" not in f:
                    n = f["start_times"].shape[0]
                    ds = f.create_dataset("start_runs", (n, self.D), dtype="i8",
                                          maxshape=(None, self.D), chunks=True)
                    if n:
                        ds[:] = 0
                return
            f.attrs["detectors"] = detectors_tag(self.detectors)
            f.attrs["runs"] = self.runs
            f.attrs["sample_rate"] = self.sample_rate
            f.attrs["seq_len"] = self.seq_len
            f.attrs["descriptor_dim"] = self.P
            f.attrs["cold_epoch"] = -1
            f.attrs["bin_files"] = np.array(self.bin_files, dtype=h5py.string_dtype())
            ck = lambda *s: dict(maxshape=tuple(None if d is None else d for d in s),
                                 chunks=True)
            f.create_dataset("start_times", (0, self.D), dtype="i8", **ck(None, self.D))
            f.create_dataset("start_segments", (0, self.D), dtype="i8", **ck(None, self.D))
            f.create_dataset("start_runs", (0, self.D), dtype="i8", **ck(None, self.D))
            f.create_dataset("start_found_epoch", (0,), dtype="i4", **ck(None))
            f.create_dataset("start_found_score", (0,), dtype="f4", **ck(None))
            f.create_dataset("embeddings", (0, self.P), dtype="f4", **ck(None, self.P))
            f.create_dataset("emb_epoch", (0,), dtype="i4", **ck(None))
            f.create_dataset("emb_score", (0,), dtype="f4", **ck(None))
            # eval_stats is appended one COLUMN per round and read by latest
            # column; column-width-1 chunks avoid read/write amplification as N
            # and the round count grow (the "append forever" design).
            f.create_dataset("eval_stats", (0, 0), dtype="f4",
                             maxshape=(None, None), chunks=(8192, 1))
            f.create_dataset("eval_model_epoch", (0,), dtype="i4", **ck(None))

    def _validate_existing(self, f):
        """Fail fast if a reopened bank is incompatible with this config (a
        config typo must never silently append into a mismatched bank)."""
        checks = {
            "descriptor_dim": (int(f.attrs["descriptor_dim"]), self.P),
            "seq_len": (int(f.attrs["seq_len"]), self.seq_len),
            "detectors": (str(f.attrs["detectors"]), detectors_tag(self.detectors)),
            "runs": (str(f.attrs["runs"]), self.runs),
        }
        for name, (stored, want) in checks.items():
            if stored != want:
                raise ValueError(
                    f"hard bank {self.path} was created with {name}={stored!r} "
                    f"but this run wants {want!r}; use a different bank file."
                )
        if int(f["embeddings"].shape[1]) != self.P:
            raise ValueError(
                f"hard bank {self.path} embeddings width "
                f"{int(f['embeddings'].shape[1])} != descriptor_dim {self.P}."
            )

    # -------------------------------------------------------------- properties
    @property
    def n_starts(self):
        with h5py.File(self.path, "r") as f:
            return int(f["start_times"].shape[0])

    @property
    def n_embeddings(self):
        with h5py.File(self.path, "r") as f:
            return int(f["embeddings"].shape[0])

    @property
    def n_eval_rounds(self):
        with h5py.File(self.path, "r") as f:
            return int(f["eval_model_epoch"].shape[0])

    @property
    def is_cold(self):
        """True before the first mining round has written anything."""
        return self.n_starts == 0

    def has_epoch(self, epoch):
        """True if this epoch's mining round is already persisted (mined starts
        tagged ``found_epoch == epoch`` or an eval column ``model_epoch ==
        epoch``). Used to make a crash-resume that re-runs the epoch idempotent
        instead of double-appending its hard windows + eval column."""
        e = int(epoch)
        with h5py.File(self.path, "r") as f:
            fe = f["start_found_epoch"]
            if fe.shape[0] and e in np.asarray(fe[...]):
                return True
            me = f["eval_model_epoch"]
            if me.shape[0] and e in np.asarray(me[...]):
                return True
        return False

    # ---------------------------------------------------------- start-times io
    @staticmethod
    def _append(ds, rows):
        """Append ``rows`` along axis 0 of a resizable dataset; return start idx."""
        n0 = ds.shape[0]
        ds.resize(n0 + len(rows), axis=0)
        ds[n0:] = rows
        return n0

    def append_starts(self, starts, segs, runs, scores, epoch):
        """Append newly-mined hard start-times (append-only, stable index).

        ``runs`` (N, D) records which pooled run each window came from, so it is
        later read from the correct file's mmap.
        """
        starts = np.asarray(starts, dtype=np.int64).reshape(-1, self.D)
        segs = np.asarray(segs, dtype=np.int64).reshape(-1, self.D)
        runs = np.asarray(runs, dtype=np.int64).reshape(-1, self.D)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(starts) == 0:
            return
        with h5py.File(self.path, "a") as f:
            i0 = self._append(f["start_times"], starts)
            self._append(f["start_segments"], segs)
            self._append(f["start_runs"], runs)
            self._append(f["start_found_epoch"],
                         np.full(len(starts), int(epoch), np.int32))
            self._append(f["start_found_score"], scores)
            if int(f.attrs["cold_epoch"]) < 0:
                f.attrs["cold_epoch"] = int(epoch)
            return i0

    def sample_starts(self, n, rng):
        """Random ``n`` (start, seg, run) rows from the bank for genotype seeding."""
        with h5py.File(self.path, "r") as f:
            N = f["start_times"].shape[0]
            if N == 0:
                empty = np.zeros((0, self.D), np.int64)
                return empty, empty, empty
            idx = np.sort(rng.choice(N, size=min(n, N), replace=False))
            return f["start_times"][idx], f["start_segments"][idx], f["start_runs"][idx]

    def iter_starts(self, batch_size):
        """Yield ``(sl, starts, segs, runs)`` over the whole bank for re-eval."""
        with h5py.File(self.path, "r") as f:
            N = f["start_times"].shape[0]
            for i in range(0, N, batch_size):
                sl = slice(i, min(i + batch_size, N))
                yield sl, f["start_times"][sl], f["start_segments"][sl], f["start_runs"][sl]

    def read_starts(self, indices):
        """Read specific rows (used by the sampler's hot path).

        Robust to unordered and duplicate indices: h5py point selection needs
        strictly-increasing indices, so we read the sorted-unique set and map
        back to the caller's order (preserving duplicates). Returns
        ``(starts, segs, runs)``.
        """
        indices = np.asarray(indices, dtype=np.int64)
        uniq, inv = np.unique(indices, return_inverse=True)   # sorted, unique
        with h5py.File(self.path, "r") as f:
            s = f["start_times"][uniq]
            g = f["start_segments"][uniq]
            r = f["start_runs"][uniq]
        return s[inv], g[inv], r[inv]

    # ----------------------------------------------------------- embeddings io
    def _distances_to_bank(self, cand, bank):
        """Nearest-neighbour distance of each ``cand`` row to ``bank``.

        For ``metric='cosine'`` returns true cosine distance ``1 - cos_sim`` (so
        ``novelty_dist`` is on the cosine scale the docstrings advertise): on
        L2-normalised vectors the chord distance ``d`` satisfies
        ``1 - cos_sim = d**2 / 2``.
        """
        cosine = self.metric == "cosine"
        if cosine:
            cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12)
            bank = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)
        if _KDTree is not None:
            d, _ = _KDTree(bank).query(cand, k=1)
        else:
            d2 = ((cand[:, None, :] - bank[None, :, :]) ** 2).sum(-1)
            d = np.sqrt(d2.min(1))
        return (d * d / 2.0) if cosine else d

    def add_embeddings(self, emb, scores, epoch):
        """Distance-gated, capped append of a diverse, hard embedding subset.

        ``emb`` are PCA-reduced embeddings (already filtered to scores >=
        threshold by the caller).  A candidate is kept only if it is at least
        ``novelty_dist`` from every existing bank embedding AND from the others
        accepted this round (greedy).  When the bank would exceed
        ``max_embeddings`` the lowest-score rows are dropped.
        Returns the number of embeddings added.
        """
        emb = np.asarray(emb, dtype=np.float32).reshape(-1, self.P)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(emb) == 0:
            return 0
        # Hardest-first so ties in a niche keep the harder window.
        order = np.argsort(-scores)
        emb, scores = emb[order], scores[order]

        with h5py.File(self.path, "a") as f:
            bank = f["embeddings"][:]
            accepted, acc_emb = [], (bank.copy() if len(bank) else None)
            for i in range(len(emb)):
                ref = acc_emb
                if ref is None or len(ref) == 0:
                    accepted.append(i)
                    acc_emb = emb[i:i + 1].copy()
                    continue
                d = self._distances_to_bank(emb[i:i + 1], ref)[0]
                if d >= self.novelty_dist:
                    accepted.append(i)
                    acc_emb = np.concatenate([acc_emb, emb[i:i + 1]], 0)
            if not accepted:
                return 0
            accepted = np.array(accepted)
            self._append(f["embeddings"], emb[accepted])
            self._append(f["emb_epoch"], np.full(len(accepted), int(epoch), np.int32))
            self._append(f["emb_score"], scores[accepted])
            self._enforce_cap(f)
            return len(accepted)

    def _enforce_cap(self, f):
        """Drop lowest-score embeddings if over the soft cap (keep hardest)."""
        M = f["embeddings"].shape[0]
        if M <= self.max_embeddings:
            return
        keep = np.sort(np.argsort(-f["emb_score"][:])[: self.max_embeddings])
        for name in ("embeddings", "emb_epoch", "emb_score"):
            data = f[name][:][keep]
            f[name].resize(len(keep), axis=0)
            f[name][:] = data

    def read_embeddings(self):
        """Return ``(embeddings (M,P), scores (M,), epochs (M,))``."""
        with h5py.File(self.path, "r") as f:
            return f["embeddings"][:], f["emb_score"][:], f["emb_epoch"][:]

    # ------------------------------------------------------------------ PCA io
    # Fitted IncrementalPCA attributes that must round-trip so a reloaded model
    # can both .transform() and continue .partial_fit() (sklearn's incremental
    # SVD update reads singular_values_/components_/mean_/var_/n_samples_seen_).
    _PCA_ARRAYS = ("components_", "mean_", "var_", "singular_values_",
                   "explained_variance_", "explained_variance_ratio_")

    def save_pca(self, ipca):
        """Persist a fitted sklearn IncrementalPCA (continuous across rounds)."""
        with h5py.File(self.path, "a") as f:
            g = f.require_group("pca")
            for k in self._PCA_ARRAYS:
                v = getattr(ipca, k, None)
                if v is None:
                    continue
                if k in g:
                    del g[k]
                g.create_dataset(k, data=np.asarray(v, np.float64))
            g.attrs["n_seen"] = int(getattr(ipca, "n_samples_seen_", 0))
            g.attrs["n_components"] = int(ipca.components_.shape[0])
            g.attrs["n_features_in"] = int(ipca.mean_.shape[0])
            g.attrs["noise_variance"] = float(getattr(ipca, "noise_variance_", 0.0))

    def load_pca(self):
        """Reconstruct the persisted IncrementalPCA, or None if not yet fitted."""
        from sklearn.decomposition import IncrementalPCA
        with h5py.File(self.path, "r") as f:
            if "pca" not in f or "components_" not in f["pca"]:
                return None
            g = f["pca"]
            ipca = IncrementalPCA(n_components=int(g.attrs["n_components"]))
            for k in self._PCA_ARRAYS:
                if k in g:
                    setattr(ipca, k, g[k][:].astype(np.float64))
            ipca.n_samples_seen_ = int(g.attrs["n_seen"])
            ipca.n_components_ = ipca.components_.shape[0]
            ipca.n_features_in_ = int(g.attrs["n_features_in"])
            ipca.noise_variance_ = float(g.attrs["noise_variance"])
            return ipca

    def active_start_indices(self, threshold):
        """Indices whose LATEST re-eval ranking stat is >= ``threshold``.

        This is the set the sampler draws from: starts that are *currently* hard
        under the latest model.  Nothing is removed from storage -- a start that
        dropped below threshold is simply absent from this set, and reappears
        automatically once a later re-eval lifts it back above threshold.
        If no re-eval has run yet, every start is considered active.
        """
        with h5py.File(self.path, "r") as f:
            N = f["start_times"].shape[0]
            R = f["eval_stats"].shape[1]
            if N == 0:
                return np.zeros(0, np.int64)
            if R == 0:
                return np.arange(N, dtype=np.int64)
            latest = f["eval_stats"][:, R - 1]
        active = np.where(latest >= float(threshold))[0].astype(np.int64)
        # NaN (start not yet scored in the latest round) fails >= -> excluded;
        # that is correct: it has no current-model evidence of being hard.
        return active

    # ----------------------------------------------------- re-eval history io
    def append_eval(self, stats, model_epoch):
        """Add one re-evaluation column: the current model's ranking stat for
        EVERY start in the bank, tagged with the model epoch that produced it.

        ``stats`` must have length == current ``n_starts``.  Rows that did not
        exist in earlier rounds are back-filled with NaN.
        """
        stats = np.asarray(stats, dtype=np.float32).reshape(-1)
        with h5py.File(self.path, "a") as f:
            es = f["eval_stats"]
            N, R = es.shape
            n = len(stats)
            if n < N:
                raise ValueError(f"eval stats {n} < n_starts {N}")
            # grow rows (new starts get NaN for past columns)
            if n > N:
                es.resize(n, axis=0)
                if R > 0:
                    es[N:n, :] = np.nan
            es.resize(R + 1, axis=1)
            es[:, R] = stats
            self._append(f["eval_model_epoch"], np.array([int(model_epoch)], np.int32))

    def hard_sample_ages(self, threshold):
        """Per-start hardness *longevity*, derived from the persisted re-eval
        history (``start_found_epoch`` + ``eval_stats`` + ``eval_model_epoch``).

        "Age" = how long a start has stayed at/above ``threshold``. Windows that
        remain hard for a very long time (large ``streak_rounds``, or found long
        ago yet ``currently_active``) are the interesting ones to study. Returns a
        length-``n_starts`` structured array (empty if the bank is cold):

          found_epoch          : mine epoch the window was first found
          first/last_active_epoch : model epoch it first/last scored >= threshold
          n_active_rounds      : total re-eval rounds it was >= threshold
          streak_rounds        : consecutive most-recent rounds it stayed hard
          age_epochs           : last_active_epoch - found_epoch (span it stayed hard)
          currently_active     : >= threshold in the latest round
        """
        dt = np.dtype([("found_epoch", "i4"), ("first_active_epoch", "i4"),
                       ("last_active_epoch", "i4"), ("n_active_rounds", "i4"),
                       ("streak_rounds", "i4"), ("age_epochs", "i4"),
                       ("currently_active", "?")])
        with h5py.File(self.path, "r") as f:
            N = f["start_times"].shape[0]
            if N == 0:
                return np.zeros(0, dtype=dt)
            found = np.asarray(f["start_found_epoch"][:], np.int32)
            R = f["eval_stats"].shape[1]
            es = f["eval_stats"][:] if R > 0 else np.full((N, 1), np.nan, np.float32)
            mep = (np.asarray(f["eval_model_epoch"][:], np.int32) if R > 0
                   else np.array([-1], np.int32))
        act = np.nan_to_num(es, nan=-np.inf) >= float(threshold)     # (N, R)
        anyact = act.any(1)
        first_idx = np.argmax(act, axis=1)                            # first True col
        last_idx = act.shape[1] - 1 - np.argmax(act[:, ::-1], axis=1)  # last True col
        streak = np.cumprod(act[:, ::-1], axis=1).sum(1).astype(np.int32)  # trailing run
        out = np.zeros(N, dtype=dt)
        out["found_epoch"] = found
        out["n_active_rounds"] = act.sum(1).astype(np.int32)
        out["streak_rounds"] = streak
        out["currently_active"] = act[:, -1]
        out["first_active_epoch"] = np.where(anyact, mep[first_idx], -1)
        out["last_active_epoch"] = np.where(anyact, mep[last_idx], -1)
        out["age_epochs"] = np.where(anyact, mep[last_idx] - found, -1)
        return out
