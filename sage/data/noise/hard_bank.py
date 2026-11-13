"""
HDF5 bank of the CMA-MAE hard-noise miner: mined start times, a diverse
set of embeddings, the IncrementalPCA state and the re-evaluation
history of every start, in one file.
"""

import os
import shutil
from contextlib import contextmanager

import numpy as np
import h5py

try:
    from scipy.spatial import cKDTree as _KDTree
except Exception:                                   # pragma: no cover
    _KDTree = None


def detectors_tag(detectors):
    """
    First letter of each detector, e.g. ``["H1", "L1"] -> "HL"``.

    Parameters
    ----------
    detectors : list of str
        Detector names.

    Returns
    -------
    str
        The tag.
    """
    return "".join(str(d)[0].upper() for d in detectors)


def runs_tag(runs):
    """
    Observing runs joined, e.g. ``["O3a", "O3b"] -> "O3aO3b"``.

    Parameters
    ----------
    runs : str or list of str
        One run, returned as is, or several.

    Returns
    -------
    str
        The tag.
    """
    if isinstance(runs, str):
        return runs
    return "".join(str(r) for r in runs)


class HardMiningBank:
    """
    Hard-noise mining bank held in one HDF5 file.

    Every method opens the file, so nothing is held in memory between
    rounds. On opening, a leftover ``<path>.tmp`` of an interrupted
    round is removed, and an existing bank is checked against the run.

    Parameters
    ----------
    path : str
        HDF5 file path (a run keeps its bank at ``mining/bank.h5``).
    detectors : list of str
        Detector names.
    runs : str or list of str
        Observing runs the model trains on; stored as metadata.
    seq_len : int
        Window length in samples.
    sample_rate : float
        Sample rate (Hz).
    bin_files : list of str
        Noise ``.bin`` files the starts index into.
    descriptor_dim : int
        Width ``P`` of the stored PCA-reduced embeddings.
    novelty_dist : float
        Minimum distance of a new embedding from every kept one.
    max_embeddings : int
        Cap on the embeddings; the lowest-score ones above it are
        dropped.
    metric : {"cosine", "euclidean"}
        Distance of the novelty gate.

    Raises
    ------
    ValueError
        If an existing bank was created with other settings.
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
        # A leftover "<bank>.tmp" is an interrupted round (atomic_round)
        _stale = self.path + ".tmp"
        if os.path.exists(_stale):
            try:
                os.remove(_stale)
            except OSError:
                pass
        self._ensure_file()

    @contextmanager
    def atomic_round(self):
        """
        Context in which all bank I/O goes to a copy of the bank.

        On a clean exit the copy replaces the bank with ``os.replace``;
        on an exception the copy is removed and the bank is unchanged.
        """
        real = self.path
        tmp = real + ".tmp"
        if os.path.exists(real):
            shutil.copy2(real, tmp)              # copy of the committed bank
        self.path = tmp                          # all I/O goes to the copy
        committed = False
        try:
            yield
            os.replace(tmp, real)                # atomic commit
            committed = True
        finally:
            self.path = real
            if not committed and os.path.exists(tmp):
                try:
                    os.remove(tmp)               # discard the round
                except OSError:
                    pass

    def _ensure_file(self):
        """
        Create the bank file, or check an existing one.

        An existing bank without ``start_runs`` gets them as run 0.

        Raises
        ------
        ValueError
            If an existing bank was created with other settings.
        """
        if os.path.exists(self.path):
            with h5py.File(self.path, "r") as f:
                if "start_times" in f:
                    self._validate_existing(f)
                    if "start_runs" in f:
                        return
        # Creation and backfill are written atomically, like a round
        with self.atomic_round(), h5py.File(self.path, "a") as f:
            if "start_times" in f:
                # Bank without start_runs: all starts are run 0
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
            # eval_stats grows by one column per round and is read by
            # its last column, so chunks are one column wide
            f.create_dataset("eval_stats", (0, 0), dtype="f4",
                             maxshape=(None, None), chunks=(8192, 1))
            f.create_dataset("eval_model_epoch", (0,), dtype="i4", **ck(None))

    def _validate_existing(self, f):
        """
        Check a reopened bank against this run's settings.

        Parameters
        ----------
        f : h5py.File
            The open bank.

        Raises
        ------
        ValueError
            If the descriptor width, window length, detectors or runs
            differ.
        """
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

    @property
    def n_starts(self):
        """Number of banked start rows."""
        with h5py.File(self.path, "r") as f:
            return int(f["start_times"].shape[0])

    @property
    def n_embeddings(self):
        """Number of banked embeddings."""
        with h5py.File(self.path, "r") as f:
            return int(f["embeddings"].shape[0])

    @property
    def n_eval_rounds(self):
        """Number of re-evaluation rounds."""
        with h5py.File(self.path, "r") as f:
            return int(f["eval_model_epoch"].shape[0])

    @property
    def is_cold(self):
        """True while the bank holds no start."""
        return self.n_starts == 0

    def has_epoch(self, epoch):
        """
        Whether the bank holds results of ``epoch``.

        Parameters
        ----------
        epoch : int
            Training epoch.

        Returns
        -------
        bool
            True if a start was found or a re-evaluation was made at
            ``epoch``.
        """
        e = int(epoch)
        with h5py.File(self.path, "r") as f:
            fe = f["start_found_epoch"]
            if fe.shape[0] and e in np.asarray(fe[...]):
                return True
            me = f["eval_model_epoch"]
            if me.shape[0] and e in np.asarray(me[...]):
                return True
        return False

    @staticmethod
    def _append(ds, rows):
        """
        Append rows along axis 0 of a resizable dataset.

        Parameters
        ----------
        ds : h5py.Dataset
            Resizable dataset.
        rows : numpy.ndarray
            Rows to append.

        Returns
        -------
        int
            Index of the first appended row.
        """
        n0 = ds.shape[0]
        ds.resize(n0 + len(rows), axis=0)
        ds[n0:] = rows
        return n0

    def append_starts(self, starts, segs, runs, scores, epoch):
        """
        Append mined windows; rows are never reordered.

        The first append sets the ``cold_epoch`` attribute.

        Parameters
        ----------
        starts, segs, runs : array-like, shape ``(N, D)``
            Absolute start indices, segment ids and pooled run ids.
        scores : array-like, shape ``(N,)``
            Ranking statistics when found.
        epoch : int
            Epoch of the mining round.

        Returns
        -------
        int or None
            Row index of the first appended window; ``None`` if there
            are none.
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
        """
        Up to ``n`` distinct random rows, in bank order.

        Parameters
        ----------
        n : int
            Number of rows.
        rng : numpy.random.Generator
            Random generator.

        Returns
        -------
        starts, segs, runs : numpy.ndarray, shape ``(min(n, N), D)``
            Start indices, segment ids and run ids.
        """
        with h5py.File(self.path, "r") as f:
            N = f["start_times"].shape[0]
            if N == 0:
                empty = np.zeros((0, self.D), np.int64)
                return empty, empty, empty
            idx = np.sort(rng.choice(N, size=min(n, N), replace=False))
            return f["start_times"][idx], f["start_segments"][idx], f["start_runs"][idx]

    def iter_starts(self, batch_size):
        """
        Iterate over the whole bank in batches.

        Parameters
        ----------
        batch_size : int
            Rows per batch.

        Yields
        ------
        tuple
            ``(slice, starts, segs, runs)`` of each batch.
        """
        with h5py.File(self.path, "r") as f:
            N = f["start_times"].shape[0]
            for i in range(0, N, batch_size):
                sl = slice(i, min(i + batch_size, N))
                yield sl, f["start_times"][sl], f["start_segments"][sl], f["start_runs"][sl]

    def read_starts(self, indices):
        """
        Read the given rows, in the given order.

        h5py needs increasing indices, so the unique sorted rows are
        read and mapped back, duplicates included.

        Parameters
        ----------
        indices : array-like of int
            Bank rows.

        Returns
        -------
        starts, segs, runs : numpy.ndarray, shape ``(K, D)``
            Start indices, segment ids and run ids.
        """
        indices = np.asarray(indices, dtype=np.int64)
        uniq, inv = np.unique(indices, return_inverse=True)   # sorted, unique
        with h5py.File(self.path, "r") as f:
            s = f["start_times"][uniq]
            g = f["start_segments"][uniq]
            r = f["start_runs"][uniq]
        return s[inv], g[inv], r[inv]

    def _distances_to_bank(self, cand, bank):
        """
        Nearest-neighbour distance of each ``cand`` row to ``bank``.

        For the cosine metric this is ``1 - cos_sim``, from the chord
        distance ``d`` of L2-normalised vectors as ``d**2 / 2``.

        Parameters
        ----------
        cand : numpy.ndarray, shape ``(K, P)``
            Candidate embeddings.
        bank : numpy.ndarray, shape ``(M, P)``
            Reference embeddings.

        Returns
        -------
        numpy.ndarray, shape ``(K,)``
            Distances.
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
        """
        Append the embeddings far enough from all kept ones.

        Candidates are taken hardest first; one is kept if it is at
        least ``novelty_dist`` from the bank and from those kept before
        it. Then the lowest-score rows above ``max_embeddings`` are
        dropped.

        Parameters
        ----------
        emb : array-like, shape ``(K, P)``
            PCA-reduced embeddings.
        scores : array-like, shape ``(K,)``
            Their ranking statistics.
        epoch : int
            Epoch of the mining round.

        Returns
        -------
        int
            Number of embeddings appended.
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
        """
        Drop the lowest-score embeddings above ``max_embeddings``.

        Parameters
        ----------
        f : h5py.File
            The open bank.
        """
        M = f["embeddings"].shape[0]
        if M <= self.max_embeddings:
            return
        keep = np.sort(np.argsort(-f["emb_score"][:])[: self.max_embeddings])
        for name in ("embeddings", "emb_epoch", "emb_score"):
            data = f[name][:][keep]
            f[name].resize(len(keep), axis=0)
            f[name][:] = data

    def read_embeddings(self):
        """
        All banked embeddings.

        Returns
        -------
        embeddings : numpy.ndarray, shape ``(M, P)``
            Embeddings.
        scores : numpy.ndarray, shape ``(M,)``
            Their ranking statistics.
        epochs : numpy.ndarray, shape ``(M,)``
            Epochs they were added.
        """
        with h5py.File(self.path, "r") as f:
            return f["embeddings"][:], f["emb_score"][:], f["emb_epoch"][:]

    # IncrementalPCA attributes stored so that a reloaded model can
    # transform and continue partial_fit
    _PCA_ARRAYS = ("components_", "mean_", "var_", "singular_values_",
                   "explained_variance_", "explained_variance_ratio_")

    def save_pca(self, ipca):
        """
        Store a fitted IncrementalPCA, replacing the stored one.

        Parameters
        ----------
        ipca : sklearn.decomposition.IncrementalPCA
            Fitted model.
        """
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
        """
        The stored IncrementalPCA.

        Returns
        -------
        sklearn.decomposition.IncrementalPCA or None
            The model, or ``None`` if none is stored.
        """
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
        """
        Rows whose latest re-evaluation score is >= ``threshold``.

        Without a re-evaluation every row is active. Rows are never
        removed; inactive rows can become active again.

        Parameters
        ----------
        threshold : float
            Lowest active score.

        Returns
        -------
        numpy.ndarray of int64
            Active row indices.
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
        # NaN (not scored in the latest round) fails >= and is excluded
        return active

    def latest_eval(self, indices=None):
        """
        Latest re-evaluation score of the given rows.

        Non-finite scores, and all scores before a re-evaluation, are 0.

        Parameters
        ----------
        indices : array-like of int or None
            Bank rows; ``None`` is every row.

        Returns
        -------
        numpy.ndarray of float32
            Scores.
        """
        with h5py.File(self.path, "r") as f:
            N = f["start_times"].shape[0]
            R = f["eval_stats"].shape[1]
            if N == 0 or R == 0:
                n = N if indices is None else len(np.asarray(indices))
                return np.zeros(n, np.float32)
            latest = f["eval_stats"][:, R - 1]
        if indices is not None:
            latest = latest[np.asarray(indices, dtype=np.int64)]
        latest = np.asarray(latest, dtype=np.float32)
        return np.nan_to_num(latest, nan=0.0, posinf=0.0, neginf=0.0)

    def append_eval(self, stats, model_epoch):
        """
        Append one re-evaluation column, tagged with the model epoch.

        Rows beyond the current ``eval_stats`` rows get NaN in earlier
        columns.

        Parameters
        ----------
        stats : array-like, shape ``(N,)``
            Score of every banked start.
        model_epoch : int
            Epoch of the model that scored them.

        Raises
        ------
        ValueError
            If ``stats`` is shorter than ``eval_stats``.
        """
        stats = np.asarray(stats, dtype=np.float32).reshape(-1)
        with h5py.File(self.path, "a") as f:
            es = f["eval_stats"]
            N, R = es.shape
            n = len(stats)
            if n < N:
                raise ValueError(f"eval stats {n} < n_starts {N}")
            # New starts get NaN in the earlier columns
            if n > N:
                es.resize(n, axis=0)
                if R > 0:
                    es[N:n, :] = np.nan
            es.resize(R + 1, axis=1)
            es[:, R] = stats
            self._append(f["eval_model_epoch"], np.array([int(model_epoch)], np.int32))

    def hard_sample_ages(self, threshold):
        """
        How long each start has scored >= ``threshold``.

        Parameters
        ----------
        threshold : float
            Lowest score counted as hard.

        Returns
        -------
        numpy.ndarray
            Structured array of length ``n_starts`` with fields
            ``found_epoch``, ``first_active_epoch``,
            ``last_active_epoch`` (model epochs, -1 if never hard),
            ``n_active_rounds``, ``streak_rounds`` (trailing hard
            rounds), ``age_epochs`` (last active minus found, -1 if
            never hard) and ``currently_active``.
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
        first_idx = np.argmax(act, axis=1)                            # first True
        last_idx = act.shape[1] - 1 - np.argmax(act[:, ::-1], axis=1)  # last True
        streak = np.cumprod(act[:, ::-1], axis=1).sum(1).astype(np.int32)  # trailing
        out = np.zeros(N, dtype=dt)
        out["found_epoch"] = found
        out["n_active_rounds"] = act.sum(1).astype(np.int32)
        out["streak_rounds"] = streak
        out["currently_active"] = act[:, -1]
        out["first_active_epoch"] = np.where(anyact, mep[first_idx], -1)
        out["last_active_epoch"] = np.where(anyact, mep[last_idx], -1)
        out["age_epochs"] = np.where(anyact, mep[last_idx] - found, -1)
        return out
