"""False-alarm rates of zero-lag triggers against the background.

Reads only finished rows, so it runs while a search is still scoring.
Windows touching bad time are dropped, each stream is clustered on its
own, then rates follow PyCBC: inclusive background, ``n_louder``, and
hierarchical removal of triggers louder than all background.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from pycbc.conversions import sec_to_year

from sage.config.machine import search_strain_file
from sage.core.progress import step
from sage.data.noise.data_quality import read_bad_time
from sage.search.clustering.greedy import greedy_triggers
from sage.search.clustering.sliding_window import cluster_triggers
from sage.search.far.far import false_alarm_rate
from sage.search.model.load import load_search_network
from sage.search.records import read_stream
from sage.search.removal.vetoes import clear_windows
from sage.search.settings import read_settings, search_dir
from sage.search.zerolag.triggers import decode
from sage.search.zerolag.zerolag import SCORES, WINDOW_TABLE

CLUSTERING = {"greedy": greedy_triggers, "pycbc": cluster_triggers}
REMOVE_NA = ("before-clustering", "after-clustering", "off")
# Sources: pycbc_coinc_statmap --cluster-window and
# --hierarchical-removal-window defaults; NA cut, user ruling
# 2026-11-01.
CLUSTER_WINDOW_S = 10.0
REMOVAL_WINDOW_S = 1.0
NA_CUT = 0.0

STEPS = (
    "Read the finished zero-lag windows and drop those touching bad time",
    "Cluster the zero-lag windows",
    "Read the finished background rolls, add up their live time and cluster "
    "each roll on its own",
    "Compute every zero-lag trigger's false-alarm rate against the background",
    "Remove zero-lag triggers louder than all background, one at a time, "
    "recomputing the rates after each",
    "Write statmap.h5",
)


def stream_triggers(
    network: Dict[str, object],
    rows: np.ndarray,
    columns: List[str],
    done: int,
    gps: np.ndarray,
    clear: Dict[str, np.ndarray],
    shift: int,
    stride_s: float,
    clustering: str,
    remove_na: str,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Clustered triggers and live time of a stream's finished rows.

    Row ``i`` pairs reference window ``i`` with the other detector's
    window ``(i + shift) mod N``; zero-lag has ``shift`` 0.

    Parameters
    ----------
    network : dict of str to object
        Output of ``load_search_network``.
    rows : numpy.ndarray
        Float32 rows of the stream, ``(N, columns)``.
    columns : list of str
        Column names of the rows.
    done : int
        Windows finished.
    gps : numpy.ndarray
        GPS start of every zero-lag window.
    clear : dict of str to numpy.ndarray
        ``clear_windows`` of the zero-lag windows.
    shift : int
        Shift of the other detector in windows.
    stride_s : float
        Step between window starts in seconds.
    clustering : str
        Key of ``CLUSTERING``.
    remove_na : str
        One of ``REMOVE_NA``: when to drop triggers whose NA logit
        exceeds the ranking statistic by more than ``NA_CUT``.

    Returns
    -------
    triggers : dict of str to numpy.ndarray
        Clustered trigger columns.
    live_time_s : float
        Finished pairs clear of bad time, times the stride.
    """
    reference, other = network["detectors"]
    n_windows = len(gps)
    window = np.arange(done)
    partner = (window + shift) % n_windows
    live = clear[reference][window] & clear[other][partner]

    window, partner = window[live], partner[live]
    outputs = np.asarray(rows[window], dtype=np.float64)
    statistic = outputs[:, columns.index("ranking_statistic")]
    keep = np.ones(len(window), dtype=bool)
    if remove_na == "before-clustering":
        logit = outputs[:, columns.index("non_astrophysical_logit")]
        keep = logit - statistic <= NA_CUT

    targets = network["targets"]
    estimate_columns = [columns.index(f"{target}_raw_mean") for target in targets]
    estimate_columns += [columns.index(f"{target}_raw_width") for target in targets]
    means, widths = decode(
        outputs[:, estimate_columns],
        network["offsets"],
        network["scales"],
        *network["width_bounds"],
    )
    offset = network["padding_s"] + means[:, targets.index("delta_tc")]
    triggers = {
        "window": window,
        "partner": partner,
        f"{reference}_tc": gps[window] + offset,
        f"{other}_tc": gps[partner] + offset,
        "ranking_statistic": statistic,
    }
    if "non_astrophysical_logit" in columns:
        triggers["non_astrophysical_logit"] = outputs[
            :, columns.index("non_astrophysical_logit")
        ]
    for index, target in enumerate(targets):
        if target != "delta_tc":
            triggers[target] = means[:, index]
        name = "tc" if target == "delta_tc" else target
        triggers[f"{name}_sigma"] = widths[:, index]

    triggers = {name: values[keep] for name, values in triggers.items()}
    triggers["tc"] = triggers[f"{reference}_tc"]
    triggers["stat"] = triggers["ranking_statistic"]
    triggers = CLUSTERING[clustering](triggers, CLUSTER_WINDOW_S)
    del triggers["tc"], triggers["stat"]
    if remove_na == "after-clustering":
        excess = triggers["non_astrophysical_logit"] - triggers["ranking_statistic"]
        triggers = {name: values[excess <= NA_CUT] for name, values in triggers.items()}
    return triggers, float(live.sum() * stride_s)


def hierarchical_rates(
    foreground: Dict[str, np.ndarray],
    background: Dict[str, np.ndarray],
    background_time_s: float,
    detectors: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hierarchical-removal false-alarm rates, as pycbc_coinc_statmap.

    While the loudest remaining zero-lag trigger is louder than all
    background, it keeps its current rate and is removed, along with
    background triggers within ``REMOVAL_WINDOW_S`` in any detector.

    Parameters
    ----------
    foreground : dict of str to numpy.ndarray
        Clustered zero-lag triggers.
    background : dict of str to numpy.ndarray
        Clustered background triggers of every roll.
    background_time_s : float
        Background live time in seconds.
    detectors : list of str
        Detectors, the reference first.

    Returns
    -------
    rates : numpy.ndarray
        Per year, ``(rounds + 1, foreground)``: before any removal, then
        after each; NaN for triggers removed by then.
    far : numpy.ndarray
        Per year: a removed trigger's rate when it was removed, any
        other trigger's rate after the last removal.
    removed_round : numpy.ndarray
        Per background trigger, the removal it went in (0: kept).
    """
    statistic = foreground["ranking_statistic"]
    centres = foreground[f"{detectors[0]}_tc"]
    remaining = np.ones(len(statistic), dtype=bool)
    removed_round = np.zeros(len(background["ranking_statistic"]), dtype=np.int64)

    rates = []
    while True:
        kept = removed_round == 0
        current = np.full(len(statistic), np.nan)
        current[remaining] = false_alarm_rate(
            statistic[remaining],
            background["ranking_statistic"][kept],
            background_time_s,
        )
        rates.append(current)
        if not remaining.any():
            break
        loudest = np.flatnonzero(remaining)[np.argmax(statistic[remaining])]
        louder = background["ranking_statistic"][kept].max(initial=-np.inf)
        if louder >= statistic[loudest]:
            break

        remaining[loudest] = False
        near = np.zeros(len(kept), dtype=bool)
        for detector in detectors:
            distance = np.abs(background[f"{detector}_tc"] - centres[loudest])
            near |= distance <= REMOVAL_WINDOW_S
        removed_round[kept & near] = len(rates)

    rates = np.array(rates)
    far = rates[-1].copy()
    for row in range(len(rates) - 1):
        removed_now = ~np.isnan(rates[row]) & np.isnan(rates[row + 1])
        far[removed_now] = rates[row][removed_now]
    return rates, far, removed_round


def write_statmap(
    path: Path,
    foreground: Dict[str, np.ndarray],
    background: Dict[str, np.ndarray],
    rates: Tuple[np.ndarray, np.ndarray, np.ndarray],
    rolls: Dict[str, np.ndarray],
    attrs: dict,
) -> None:
    """
    Write the triggers, their rates and the settings to one HDF5 file.

    Parameters
    ----------
    path : Path
        Output file; replaced in one step.
    foreground, background : dict of str to numpy.ndarray
        Clustered zero-lag and background triggers.
    rates : tuple of numpy.ndarray
        Output of ``hierarchical_rates``.
    rolls : dict of str to numpy.ndarray
        Per roll: file, number, shift, windows done and live time.
    attrs : dict
        Settings and live times.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    with h5py.File(part, "w") as handle:
        for name, values in foreground.items():
            handle[f"zerolag/{name}"] = values
        handle["zerolag/far"] = rates[1]
        handle["zerolag/far_per_round"] = rates[0]
        for name, values in background.items():
            handle[f"background/{name}"] = values
        handle["background/removed_round"] = rates[2]
        for name, values in rolls.items():
            handle[f"rolls/{name}"] = values
        handle.attrs.update(attrs)
    part.replace(path)


def run_statmap(
    run_dir: Path, name: str, data_dir: Path, clustering: str, remove_na: str
) -> Path:
    """
    False-alarm rates of a search's finished rows; write ``statmap.h5``.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    name : str
        Search name.
    data_dir : Path
        Directory from the machine file.
    clustering : str
        Key of ``CLUSTERING``.
    remove_na : str
        One of ``REMOVE_NA``.

    Returns
    -------
    Path
        The statmap file.

    Raises
    ------
    ValueError
        If NA removal is asked of a network without the NA class, or
        no background roll has finished rows.
    """
    directory = search_dir(run_dir, name)
    settings = read_settings(directory)
    checkpoint = Path(run_dir) / "checkpoints" / f"{settings['checkpoint']}.pt"
    network = load_search_network(checkpoint, "cpu")
    detectors = network["detectors"]
    stride_s = settings["stride_in_samples"] / network["sample_rate"]

    with step(1, STEPS):
        table = np.load(directory / "zerolag" / WINDOW_TABLE, mmap_mode="r")
        gps = np.asarray(table["gps"])
        vetoes = {
            detector: read_bad_time(
                search_strain_file(data_dir, settings["observing_run"], detector)
            )
            for detector in detectors
        }
        clear = clear_windows(
            network, {f"{detector}_gps": gps for detector in detectors}, vetoes
        )
        rows, progress = read_stream(directory / "zerolag", SCORES)
        columns = progress["columns"]
        if remove_na != "off" and "non_astrophysical_logit" not in columns:
            raise ValueError(f"{checkpoint} has no NA class to remove by")
        print(f"  {progress['done']:,} of {len(gps):,} windows finished")
    with step(2, STEPS):
        foreground, foreground_s = stream_triggers(
            network,
            rows,
            columns,
            progress["done"],
            gps,
            clear,
            0,
            stride_s,
            clustering,
            remove_na,
        )
        print(
            f"  {len(foreground['window']):,} triggers in "
            f"{sec_to_year(foreground_s):.4f} years of live time"
        )

    with step(3, STEPS):
        parts, rolls = [], []
        number = 0
        for file in sorted((directory / "background").glob("year_*.json")):
            rows, progress = read_stream(file.parent, file.stem)
            for index, (shift,) in enumerate(progress["attrs"]["shifts"]):
                number += 1
                if progress["done"] == 0:
                    continue
                triggers, live_s = stream_triggers(
                    network,
                    rows[index * len(gps) : (index + 1) * len(gps)],
                    progress["columns"],
                    progress["done"],
                    gps,
                    clear,
                    shift,
                    stride_s,
                    clustering,
                    remove_na,
                )
                triggers["roll"] = np.full(len(triggers["window"]), number)
                parts.append(triggers)
                year = int(file.stem.split("_")[1])
                rolls.append((year, number, shift, progress["done"], live_s))
        if not parts:
            raise ValueError(f"{directory / 'background'} has no finished rows yet")
        background = {
            key: np.concatenate([part[key] for part in parts]) for key in parts[0]
        }
        rolls = dict(zip(("file", "roll", "shift", "done", "live_time_s"),
                         map(np.array, zip(*rolls))))
        background_s = float(rolls["live_time_s"].sum())
        print(
            f"  {len(background['window']):,} triggers in {len(parts)} rolls, "
            f"{sec_to_year(background_s):.4f} years of live time"
        )

    with step(4, STEPS):
        before = false_alarm_rate(
            foreground["ranking_statistic"],
            background["ranking_statistic"],
            background_s,
        )
        print(f"  lowest rate {before.min():.3g} per year")
    with step(5, STEPS):
        rates = hierarchical_rates(foreground, background, background_s, detectors)
        print(f"  {len(rates[0]) - 1} triggers removed")

    path = directory / "statmap" / f"{clustering}_remove-na-{remove_na}" / "statmap.h5"
    attrs = {
        "search": name,
        "checkpoint": str(checkpoint.resolve()),
        "clustering": clustering,
        "cluster_window_s": CLUSTER_WINDOW_S,
        "remove_na": remove_na,
        "na_cut": NA_CUT,
        "removal_window_s": REMOVAL_WINDOW_S,
        "zerolag_live_time_s": foreground_s,
        "background_live_time_s": background_s,
        "settings": json.dumps(settings),
    }
    with step(6, STEPS):
        write_statmap(path, foreground, background, rates, rolls, attrs)
    return path
