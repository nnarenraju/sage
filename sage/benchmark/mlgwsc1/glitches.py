"""
Glitch rejection on MLGWSC-1 dataset 4: the run's and PyCBC's loudest
background trigger on each Gravity Spy glitch of the test data.
"""

import importlib.util
import itertools
import sys
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sage.benchmark.mlgwsc1 import release
from sage.benchmark.mlgwsc1.evaluator import reachable_thresholds
from sage.core.logger import get_logger
from sage.data.noise.data_quality import clean, noise_bad_segments
from sage.factory.build import noise_runs, selected_records

logger = get_logger(__name__)

# Gravity Spy's O3 classes (Zenodo 5649212): cell 90's 22 in its order,
# Blip_Low_Frequency and Fast_Scattering inserted alphabetically
GLITCH_NAMES = [
    "1400Ripples", "1080Lines", "Air_Compressor", "Blip", "Blip_Low_Frequency",
    "Chirp", "Extremely_Loud", "Fast_Scattering", "Helix", "Koi_Fish",
    "Light_Modulation", "Low_Frequency_Burst", "Low_Frequency_Lines",
    "No_Glitch", "None_of_the_Above", "Paired_Doves", "Power_Line",
    "Repeating_Blips", "Scattered_Light", "Scratchy", "Tomte", "Violin_Mode",
    "Wandering_Line", "Whistle",
]
GLITCH_ALIAS = [
    "1400R", "1080L", "AC", "B", "BLF", "C", "EL", "FS", "H", "KF", "LM",
    "LFB", "LFL", "NG", "NA", "PD", "PL", "RB", "SL", "S", "T", "VM", "WL",
    "W",
]
NO_GLITCH = GLITCH_NAMES.index("No_Glitch")
DETECTORS = ["H1", "L1"]
DETFILES = ["H1_O3a", "L1_O3a"]  # dataset 4's noise is O3a
TEAMS = ["Sage", "PyCBC"]


def read_glitches(path: Path) -> dict:
    """
    GPS start, duration, SNR and Gravity Spy label of every glitch in a
    Gravity Spy file.

    Parameters
    ----------
    path : Path
        Gravity Spy file, e.g. ``H1_O3a.csv``.

    Returns
    -------
    dict
        ``time``, ``duration``, ``snr`` and ``type`` (index into
        ``GLITCH_NAMES`` of ``ml_label``) arrays.

    Raises
    ------
    ValueError
        If a label is not in ``GLITCH_NAMES``.
    """
    gfile = pd.read_csv(
        path, usecols=["start_time", "start_time_ns", "duration", "snr", "ml_label"]
    )
    unknown = set(gfile["ml_label"]) - set(GLITCH_NAMES)
    if unknown:
        raise ValueError(f"{path}: ml_label {sorted(map(str, unknown))} unknown")
    map_to_idx = dict(zip(GLITCH_NAMES, range(len(GLITCH_NAMES))))
    return {
        "time": np.array(gfile["start_time"]) + np.array(gfile["start_time_ns"]) * 1e-9,
        "duration": np.array(gfile["duration"]),
        "snr": np.array(gfile["snr"]),
        "type": np.array([map_to_idx[label] for label in gfile["ml_label"]]),
    }


def slide_shifts(code: Path, background: Path) -> dict:
    """
    How far L1 is slid in each segment of dataset 4.

    MLGWSC-1's ``get_real_noise`` keeps H1 at GPS time and reads L1 from
    GPS time + shift (``OverlapSegment.get3``); its segments and seeds
    are replayed here from the settings stored in the background file.

    Parameters
    ----------
    code : Path
        MLGWSC-1's code folder.
    background : Path
        Dataset 4's ``background.hdf``.

    Returns
    -------
    dict
        Shift in s, keyed by segment start.
    """
    with h5py.File(background, "r") as fp:
        attrs = dict(fp.attrs)
    # generate_data imports its sibling segments.py
    sys.path.insert(0, str(code))
    try:
        spec = importlib.util.spec_from_file_location(
            "generate_data", code / "generate_data.py"
        )
        generate_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_data)
    finally:
        sys.path.remove(str(code))

    segments = generate_data.restrict_segments(
        start_offset=attrs["start_offset"],
        duration=attrs["duration"],
        min_segment_duration=attrs["min_segment_duration"],
        slide_buffer=attrs["slide_buffer"],
    )
    delta_t = 1 / attrs["sample_rate"]
    rs = np.random.RandomState(attrs["seed"])
    shifts = {}
    for seg in segments:
        tmpseed = rs.randint(0, int(1e6))
        seg_rs = np.random.RandomState(seed=tmpseed)
        samples = int((seg[1] - seg[0]) / delta_t)
        kept = int((seg[1] - seg[0] - attrs["slide_buffer"]) / delta_t)
        shifts[float(seg[0])] = seg_rs.randint(0, samples - kept) * delta_t
    return shifts


def test_segments(background: Path) -> list[tuple[float, float]]:
    """
    Start and end of each segment of dataset 4, as in its H1 data.

    Parameters
    ----------
    background : Path
        Dataset 4's ``background.hdf``.

    Returns
    -------
    list of tuple of float
    """
    with h5py.File(background, "r") as fp:
        return [
            (
                float(ds.attrs["start_time"]),
                ds.attrs["start_time"] + len(ds) * ds.attrs["delta_t"],
            )
            for ds in fp["H1"].values()
        ]


def glitches_in_testset(
    glitches: dict, segments: list, shifts: dict, detector: str
) -> dict:
    """
    Glitches inside the test data, at their time in the data files.

    Parameters
    ----------
    glitches : dict
        Output of ``read_glitches``.
    segments : list of tuple of float
        Output of ``test_segments``.
    shifts : dict
        Output of ``slide_shifts``; applied to L1 only.
    detector : str
        ``H1`` or ``L1``.

    Returns
    -------
    dict
        As ``glitches``, ``time`` being the time in the data files.
    """
    times = glitches["time"]
    file_times = times.astype(float)
    inside = np.zeros(len(times), dtype=bool)
    for start, end in segments:
        shift = shifts[start] if detector == "L1" else 0.0
        present = (start <= times - shift) & (times - shift <= end)
        inside |= present
        file_times[present] = times[present] - shift
    in_testset = {key: values[inside] for key, values in glitches.items()}
    in_testset["time"] = file_times[inside]
    return in_testset


def relevant_glitches(
    bg_events: np.ndarray, glitches: dict, threshold: float
) -> np.ndarray:
    """
    The loudest background trigger on each glitch, above a threshold.

    A glitch's triggers are those from its start to its end; glitches of
    the class No_Glitch are left out.

    Parameters
    ----------
    bg_events : numpy.ndarray, shape (2, N)
        Time and ranking statistic of each background event.
    glitches : dict
        Output of ``glitches_in_testset``.
    threshold : float
        Ranking statistic of 1000 false alarms per month.

    Returns
    -------
    numpy.ndarray, shape (M, 4)
        Per glitch: FAR rank (number of louder background triggers + 1),
        glitch SNR, time and class.
    """
    order = bg_events[0].argsort()
    bg_trigger_times, bg_trigger_stats = bg_events[0][order], bg_events[1][order]
    starts = np.searchsorted(bg_trigger_times, glitches["time"], "left")
    ends = np.searchsorted(
        bg_trigger_times, glitches["time"] + glitches["duration"], "right"
    )

    glitch_stat_snr = np.array(
        [
            (np.max(bg_trigger_stats[start:end]), gsnr, gtime, gtype)
            for start, end, gsnr, gtime, gtype in zip(
                starts, ends, glitches["snr"], glitches["time"], glitches["type"]
            )
            if end > start
        ]
    ).reshape(-1, 4)
    ridxs = np.argwhere(
        (glitch_stat_snr[:, 0] > threshold) & (glitch_stat_snr[:, 3] != NO_GLITCH)
    ).flatten()
    above = glitch_stat_snr[ridxs]
    above[:, 0] = [
        len(bg_trigger_stats[bg_trigger_stats > stat]) + 1 for stat in above[:, 0]
    ]
    return above


def training_glitch_counts(
    source: dict, data_dir: Path, window_samples: int, directory: Path
) -> np.ndarray | None:
    """
    Gravity Spy glitches of each class in the run's training noise, both
    detectors together.

    The segments are those the run's noise sampler draws from: its
    selection, clear of bad time and at least one window long.

    Parameters
    ----------
    source : dict
        The run's ``noise.training`` settings.
    data_dir : Path
        Directory from the machine file.
    window_samples : int
        The run's padded window length in samples.
    directory : Path
        The MLGWSC-1 folder of the data directory.

    Returns
    -------
    numpy.ndarray or None
        Count per class of ``GLITCH_NAMES``; None if the noise is not
        real or a training run has no Gravity Spy file.
    """
    if source["name"] != "real":
        logger.info("No training glitch counts: noise.training is %s", source["name"])
        return None

    types = []
    for run in noise_runs(source, data_dir, DETECTORS):
        records = selected_records(run, DETECTORS)
        bad = noise_bad_segments(run["bins"])
        for detector in DETECTORS:
            observing_run = records[detector][0]["observing_run"]
            name = f"{detector}_{observing_run}.csv"
            if name not in release.GRAVITYSPY_MD5:
                logger.info("No training glitch counts: no Gravity Spy %s", name)
                return None

            # As MemmapNoiseSampler: clean and at least one window long
            good = clean(records[detector], bad[(observing_run, detector)])
            kept = [
                record
                for record, is_good in zip(records[detector], good)
                if is_good and record["nsamples"] >= window_samples
            ]
            starts = np.array([record["gps_start"] for record in kept])
            ends = np.array([record["gps_end"] for record in kept])
            order = starts.argsort()
            starts, ends = starts[order], ends[order]

            # The segments do not overlap: the last one starting at or
            # before a glitch is the only one that can hold it
            glitches = read_glitches(release.fetch_gravityspy(directory, name))
            last = np.searchsorted(starts, glitches["time"], side="right") - 1
            inside = (last >= 0) & (glitches["time"] <= ends[np.maximum(last, 0)])
            types.append(glitches["type"][inside])
    return np.bincount(np.concatenate(types), minlength=len(GLITCH_NAMES))


def plot_glitch_triggers(found: dict, path: str) -> None:
    """
    Glitch SNR against the log10 FAR rank of its loudest trigger, per
    detector, for Sage and PyCBC.

    Parameters
    ----------
    found : dict
        Output of ``relevant_glitches`` by team and ``DETFILES`` entry.
    path : str
        Figure file.
    """
    with plt.rc_context({"font.size": 22, "font.family": "sans-serif"}):
        nrows = 1
        fig, axs = plt.subplots(nrows, 2, figsize=(13.0 * 2, 9.0 * nrows))
        axs = np.concatenate((axs.flatten(), np.tile(axs.flatten(), 1)))
        coords = itertools.product(["PyCBC", "Sage"], DETFILES)

        for (tname, detfile), ax in zip(coords, axs):
            ax.set_ylabel(
                r"log$_{{10}}$(FAR) per month ({})".format(detfile.replace("_", " ")),
                fontsize=22,
            )
            ax.set_xlabel("Glitch SNR", fontsize=22)
            glitches_above_1000_per_month = found[tname][detfile]

            # Plot all glitch triggers alongside the thresh vals
            xmin = 5.0
            xmax = 10_000

            # Scatter plot of glitch triggers
            color = "#2F67B1" if tname == "PyCBC" else "#BF2C23"
            ax.scatter(
                glitches_above_1000_per_month[:, 1],
                np.log10(glitches_above_1000_per_month[:, 0]),
                s=256,
                alpha=1.0,
                c=color,
                edgecolors="k",
                label=tname,
            )

            if tname == "Sage":
                ax.grid(which="both")
                ax.set_ylim(0, 3.1)
                ax.set_ylim(ax.get_ylim()[::-1])
                ax.set_xlim(xmin, xmax)
                ax.set_xscale("log")
                ax.tick_params(axis="both", which="major", labelsize=22)
                ax.tick_params(axis="both", which="minor", labelsize=22)

            pycbc_patch = mpatches.Patch(color="#2F67B1", label="PyCBC")
            sage_patch = mpatches.Patch(color="#BF2C23", label="Sage")
            ax.legend(handles=[pycbc_patch, sage_patch])

        plt.subplots_adjust(wspace=0.2)
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()


def top_glitches(team: np.ndarray, alt_team: np.ndarray) -> tuple[dict, dict]:
    """
    The three glitch classes a team gives the lowest FAR, with both
    teams' log10 FAR ranks on them.

    Parameters
    ----------
    team, alt_team : numpy.ndarray
        Outputs of ``relevant_glitches``.

    Returns
    -------
    tuple of dict
        The team's and the other team's log10 FAR ranks below 4, by
        class alias.
    """
    team_triggers = np.column_stack((team[:, 3], np.log10(team[:, 0])))
    alt_team_triggers = np.column_stack((alt_team[:, 3], np.log10(alt_team[:, 0])))

    # Iterate through all glitch types
    save_team_triggers = {}
    save_team_triggers_nums = {}
    save_alt_team_triggers = {}
    for i in range(len(GLITCH_NAMES)):
        save_team_triggers[i] = team_triggers[:, 1][team_triggers[:, 0] == i]
        save_alt_team_triggers[i] = alt_team_triggers[:, 1][
            alt_team_triggers[:, 0] == i
        ]
        if len(save_team_triggers[i]):
            save_team_triggers_nums[i] = np.min(save_team_triggers[i])

    # Which triggers types have the highest trigger values
    highest_team_triggers = dict(
        sorted(save_team_triggers_nums.items(), key=lambda item: item[1], reverse=False)
    )
    top10_glitches_team = list(highest_team_triggers.keys())[:3]
    top10_glitches_alt_team = top10_glitches_team.copy()

    team_swarm_data = [
        save_team_triggers[i][save_team_triggers[i] < 4] for i in top10_glitches_team
    ]
    alt_team_swarm_data = [
        save_alt_team_triggers[i][save_alt_team_triggers[i] < 4]
        for i in top10_glitches_alt_team
    ]
    team_labels = [GLITCH_ALIAS[i] for i in top10_glitches_team]
    alt_team_labels = [GLITCH_ALIAS[i] for i in top10_glitches_alt_team]
    team_swarm_datadict = dict(zip(team_labels, team_swarm_data))
    alt_team_swarm_datadict = dict(zip(alt_team_labels, alt_team_swarm_data))
    return team_swarm_datadict, alt_team_swarm_datadict


def plot_glitch_classes(counts: np.ndarray | None, found: dict, path: str) -> None:
    """
    Glitch classes in the training data and the classes each team gives
    the lowest FAR, per detector.

    Parameters
    ----------
    counts : numpy.ndarray or None
        Output of ``training_glitch_counts``; None leaves the bar chart
        out.
    found : dict
        Output of ``relevant_glitches`` by team and ``DETFILES`` entry.
    path : str
        Figure file.
    """
    fnt = 26
    coords = itertools.product(TEAMS, DETFILES)

    with plt.rc_context({"font.size": 22, "font.family": "sans-serif"}):
        # figsize - (width, height)
        fig = plt.figure(figsize=(10.0 * 3, 7.5 * 3), constrained_layout=True)
        gs = fig.add_gridspec(
            3,
            3,
            width_ratios=[2, 2, 1],
            height_ratios=[2, 2, 2],
            wspace=0.03,
            hspace=0.04,
        )

        ## Glitch type frequency bar plot for training data
        if counts is not None:
            ax1 = fig.add_subplot(gs[0, :])
            ax1.bar(
                GLITCH_ALIAS,
                counts,
                width=1.0,
                color="#025196",
                edgecolor="k",
                linewidth=1,
            )
            ax1.set_ylabel("N Glitches in Training Data", fontsize=fnt)
            ax1.tick_params(labelsize=24)
            ax1.set_yscale("log")
            ax1.set_axisbelow(True)
            grid = dict(color="gray", linestyle="dashed")
            ax1.yaxis.grid(which="major", linewidth=2.0, **grid)
            ax1.yaxis.grid(which="minor", linewidth=0.5, **grid)

        ## Glitch type acronyms and legend for swarmplot
        ax2 = fig.add_subplot(gs[1:, -1])
        for n, (galias, gname) in enumerate(zip(GLITCH_ALIAS, GLITCH_NAMES)):
            ax2.text(
                0.05,
                0.75 - n * 0.03,
                "{} = {}".format(galias, gname),
                horizontalalignment="left",
                fontsize=fnt,
            )
        sage_patch = mpatches.Patch(color="#512888", label="Sage")
        pycbc_patch = mpatches.Patch(color="#EB6123", label="PyCBC")
        # left/right, up/down, width, height
        ax2.legend(
            handles=[sage_patch, pycbc_patch],
            bbox_to_anchor=(0.15, -2.4, 1.0, 6.5),
            loc="center left",
            frameon=False,
            fontsize=28,
            handlelength=2.6,
        )
        # Remove borders and ticks
        ax2.set_xticks([])
        ax2.set_yticks([])
        for pos in ["right", "top", "bottom", "left"]:
            ax2.spines[pos].set_visible(False)

        ## Top glitches comparison
        topgs = [fig.add_subplot(gs[row, col]) for row in (1, 2) for col in (0, 1)]
        for (team_name, detfile), ax in zip(coords, topgs):
            alt_team_name = "PyCBC" if team_name == "Sage" else "Sage"
            team_swarm_datadict, alt_team_swarm_datadict = top_glitches(
                found[team_name][detfile], found[alt_team_name][detfile]
            )
            ylabel = "Glitch Types ({}, {})".format(
                team_name, detfile.replace("_", " ")
            )
            xlabel = r"$\mathregular{log}_{\mathregular{10}}(\mathregular{FAR/month})$"
            swarmdf = pd.DataFrame(
                [
                    [label, val, team_name]
                    for label, values in team_swarm_datadict.items()
                    for val in values
                ]
                + [
                    [label, val, alt_team_name]
                    for label, values in alt_team_swarm_datadict.items()
                    for val in values
                ],
                columns=[ylabel, xlabel, "source"],
            )
            colors = ["#512888", "#EB6123"]
            if team_name != "Sage":
                colors = colors[::-1]
            swarm = sns.swarmplot(
                swarmdf,
                dodge=False,
                legend=False,
                ax=ax,
                orient="h",
                y=ylabel,
                x=xlabel,
                size=10,
                palette=colors,
                hue="source",
            )

            # Thesholds in FAR
            for thresh in np.log10([1, 10, 100, 1000, 10000]):
                ax.axvline(thresh, ls="--", linewidth=1.0, color="k")

            ax.set_xlabel(xlabel, fontsize=fnt)
            ax.set_ylabel(ylabel, fontsize=fnt)
            ax.tick_params(labelsize=fnt)

            ax = swarm.axes
            ax.invert_xaxis()
            ax.grid(axis="x")

        plt.savefig(path)
        plt.close()


def read_background(path: Path) -> dict:
    """
    A team's background events and the ranking statistics of its FAR
    thresholds, as ``get_stats``.

    Parameters
    ----------
    path : Path
        Background event file with ``time`` and ``stat``.

    Returns
    -------
    dict
        ``bgevents`` (time and ranking statistic) and
        ``far_thresholds``.
    """
    with h5py.File(path, "r") as fp:
        bgevents = np.vstack([fp["time"], fp["stat"]])
    noise_stats = bgevents[1].copy()
    noise_stats.sort()
    return {"bgevents": bgevents, "far_thresholds": reachable_thresholds(noise_stats)}


def plot_glitches(
    run_background: Path,
    pycbc_background: Path,
    dataset: Path,
    code: Path,
    counts: np.ndarray | None,
    directory: Path,
    output_dir: Path,
) -> None:
    """
    Draw the glitch figures into ``output_dir/PAPER``.

    Both figures are left out when a team's background does not reach
    1000 false alarms per month.

    Parameters
    ----------
    run_background, pycbc_background : Path
        The run's and PyCBC's background events.
    dataset : Path
        Dataset 4's folder.
    code : Path
        MLGWSC-1's code folder.
    counts : numpy.ndarray or None
        Output of ``training_glitch_counts``.
    directory : Path
        The MLGWSC-1 folder of the data directory.
    output_dir : Path
        Directory of the evaluation.
    """
    teams = {"Sage": read_background(run_background)}
    teams["PyCBC"] = read_background(pycbc_background)
    for name, team in teams.items():
        if len(team["far_thresholds"]) < 5:
            logger.info("No glitch figures: %s's background misses 1000/month", name)
            return

    background = dataset / "background.hdf"
    segments = test_segments(background)
    shifts = slide_shifts(code, background)
    found = {name: {} for name in teams}
    for detector, detfile in zip(DETECTORS, DETFILES):
        glitches = read_glitches(release.fetch_gravityspy(directory, f"{detfile}.csv"))
        in_testset = glitches_in_testset(glitches, segments, shifts, detector)
        for name, team in teams.items():
            found[name][detfile] = relevant_glitches(
                team["bgevents"], in_testset, team["far_thresholds"][4]
            )

    paper_dir = Path(output_dir) / "PAPER"
    paper_dir.mkdir(exist_ok=True)
    plot_glitch_triggers(found, str(paper_dir / "glitch_triggers_poster.png"))
    plot_glitch_classes(
        counts, found, str(paper_dir / "glitch_swarmplot_pycbc_sage.png")
    )
