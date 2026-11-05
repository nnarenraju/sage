"""
The paper's figures of a run against PyCBC and the MLGWSC-1 teams on
dataset 4: found signals, sensitive distance and ROC.
"""

import itertools
import os

import h5py
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
from sklearn import metrics

from sage.benchmark.mlgwsc1.evaluator import MONTH
from sage.core.logger import get_logger

logger = get_logger(__name__)

COLORS_HISTOGRAM = [  # PyCBC, Sage
    mpl.colors.rgb2hex(mpl.colormaps["turbo"](0.05), keep_alpha=True),
    mpl.colors.rgb2hex(mpl.colormaps["turbo"](0.95), keep_alpha=True),
]
# Midpoint of the diverging palette of cells 39 and 50
BLACK = sns.diverging_palette(260, 15, sep=1, center="dark", as_cmap=True)([0.5])
PARAMS = ["mchirp", "q", "spin1z", "spin2z", "snr"]  # corner plot
LABELS = {
    "mchirp": r"$\mathcal{M}_c$",
    "q": "q",
    "spin1z": "Spin1z",
    "spin2z": "Spin2z",
    "snr": r"$\rho$",
}
TEAM_STYLES = {  # colour, line width, line style, label
    "PyCBC": ("#855e46", 1.5, "dashed", "PyCBC"),
    "TPI_FSU_Jena": ("green", 1.0, "solid", "TPI FSU Jena"),
    "Virgo-AUTh": ("blueviolet", 1.0, "solid", "Virgo-AUTh"),
    "CNN-Coinc": ("blue", 1.0, "solid", "CNN-Coinc"),
    "MFCNN": ("orange", 1.0, "solid", "MFCNN"),
}
ROC_WINDOW_IN_S = (-0.3, 0.3)  # events around each injection time


def found_parameters(team: dict) -> dict:
    """
    Parameters of the injections a team finds above 1 false alarm per
    month.

    Parameters
    ----------
    team : dict
        A team as filled by ``get_stats``.

    Returns
    -------
    dict
        Found values of each of ``PARAMS``.
    """
    found = team["found_idx"][team["found_stats"] > team["far_thresholds"][0]]
    return {param: team[param][found] for param in PARAMS}


def found_difference(found_by: list) -> int:
    """
    Number found by Sage minus number found by PyCBC in a hexbin cell.

    Parameters
    ----------
    found_by : list of int
        1 for each injection found by Sage, 0 for each found by PyCBC.

    Returns
    -------
    int
    """
    found_by = np.array(found_by)
    num_pycbc = len(found_by[found_by == 0])
    num_sage = len(found_by[found_by == 1])
    return num_sage - num_pycbc


def hexbin_data(
    found_sage: dict, found_pycbc: dict, x_param: str, y_param: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Both pipelines' found values of two parameters and who found each.

    Parameters
    ----------
    found_sage, found_pycbc : dict
        As ``found_parameters``.
    x_param, y_param : str
        The two parameters.

    Returns
    -------
    x_data, y_data : numpy.ndarray
        PyCBC's found values, then Sage's.
    found_by : numpy.ndarray
        0 for PyCBC's, 1 for Sage's.
    """
    x_data = np.hstack((found_pycbc[x_param], found_sage[x_param]))
    y_data = np.hstack((found_pycbc[y_param], found_sage[y_param]))
    found_by = np.hstack(
        (
            np.full(len(found_pycbc[x_param]), 0),
            np.full(len(found_sage[x_param]), 1),
        )
    )
    return x_data, y_data, found_by


def hexbin_limits(found_sage: dict, found_pycbc: dict) -> tuple[int, int]:
    """
    Smallest and largest found-count difference of the (mchirp, q)
    hexbin.

    Parameters
    ----------
    found_sage, found_pycbc : dict
        As ``found_parameters``.

    Returns
    -------
    tuple of int
    """
    x_data, y_data, found_by = hexbin_data(found_sage, found_pycbc, "mchirp", "q")
    fig, ax = plt.subplots()
    hexb = ax.hexbin(
        x_data,
        y_data,
        C=found_by,
        reduce_C_function=found_difference,
        gridsize=32,
    )
    plt.close(fig)

    values = hexb.get_array()
    return int(min(values)), int(max(values))


def blue_red_colormap(limits: tuple[int, int]) -> ListedColormap:
    """
    One colour per found-count difference: blue below zero, black at
    zero, red above.

    Parameters
    ----------
    limits : tuple of int
        As ``hexbin_limits``.

    Returns
    -------
    matplotlib.colors.ListedColormap
    """
    lower, upper = limits
    colors_blue = plt.cm.Blues(np.linspace(0.5, 1.0, abs(lower)))
    colors_red = plt.cm.Reds_r(np.linspace(0.0, 0.5, abs(upper)))
    colors = np.vstack((colors_blue, BLACK, colors_red))
    custom = mpl.colors.LinearSegmentedColormap.from_list("my_colormap", colors)
    return ListedColormap(custom(np.linspace(0, 1, abs(upper) + abs(lower) + 1)))


def found_counts(found_sage: dict, found_pycbc: dict) -> tuple[int, int, int]:
    """
    Injections found by both pipelines, only by Sage and only by PyCBC.

    Parameters
    ----------
    found_sage, found_pycbc : dict
        As ``found_parameters``.

    Returns
    -------
    tuple of int
    """
    sage_set = set(found_sage["mchirp"])
    pycbc_set = set(found_pycbc["mchirp"])
    found_by_both = len(set.intersection(sage_set, pycbc_set))
    found_only_by_sage = len(sage_set - pycbc_set)
    found_only_by_pycbc = len(pycbc_set - sage_set)
    return found_by_both, found_only_by_sage, found_only_by_pycbc


def plot_mchirp_histogram(found_sage: dict, found_pycbc: dict, path: str) -> None:
    """
    Histogram of the chirp masses each pipeline finds.

    Parameters
    ----------
    found_sage, found_pycbc : dict
        As ``found_parameters``.
    path : str
        Figure file.
    """
    with plt.rc_context({"font.size": 18, "font.family": "sans-serif"}):
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
        bins = np.histogram(
            np.hstack((found_pycbc["mchirp"], found_sage["mchirp"])), bins=40
        )[1]
        ax.hist(
            found_pycbc["mchirp"],
            density=False,
            bins=bins,
            alpha=0.8,
            histtype="step",
            color=COLORS_HISTOGRAM[0],
            linewidth=2.0,
            linestyle="dashed",
            label="PyCBC",
        )
        ax.hist(
            found_sage["mchirp"],
            density=False,
            bins=bins,
            alpha=0.8,
            histtype="step",
            color=COLORS_HISTOGRAM[1],
            linewidth=2.0,
            linestyle="solid",
            label="Sage",
        )

        plt.xlabel(r"Chirp Mass ($\mathregular{M}_\odot$)")
        plt.ylabel("N Detected Signals")
        plt.legend()

        plt.savefig(path, bbox_inches="tight", dpi=500)
        plt.close()


def plot_mchirp_q(
    found_sage: dict,
    found_pycbc: dict,
    limits: tuple[int, int],
    cmap: ListedColormap,
    path: str,
) -> None:
    """
    Hexbin of the found-count difference over (mchirp, q), with the
    marginal histograms and the found counts.

    Parameters
    ----------
    found_sage, found_pycbc : dict
        As ``found_parameters``.
    limits : tuple of int
        As ``hexbin_limits``.
    cmap : matplotlib.colors.ListedColormap
        As ``blue_red_colormap``.
    path : str
        Figure file.
    """
    lower_hexbin, upper_hexbin = limits
    with plt.rc_context({"font.size": 16, "font.family": "sans-serif"}):
        fig = plt.figure(figsize=(8, 7.5))
        gs = gridspec.GridSpec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0, hspace=0
        )

        # 2D hexbin (main)
        ax_hex = plt.subplot(gs[2])
        x_data, y_data, found_by = hexbin_data(found_sage, found_pycbc, "mchirp", "q")
        hexb = ax_hex.hexbin(
            x_data,
            y_data,
            C=found_by,
            reduce_C_function=found_difference,
            gridsize=32,
            cmap=cmap,
            norm=mpl.colors.Normalize(vmin=lower_hexbin - 0.5, vmax=upper_hexbin + 0.5),
        )
        ax_hex.set_xlabel(r"$\mathcal{M}_c\;(\mathregular{in}\;\mathregular{M}_\odot)$")
        ax_hex.set_ylabel(r"$q$")

        ax_hex.xaxis.set_minor_locator(AutoMinorLocator())
        ax_hex.yaxis.set_minor_locator(AutoMinorLocator())
        sides = dict(bottom=True, left=True, top=True, right=True, direction="in")
        for which, length in (("minor", 2.5), ("major", 5)):
            for axis in ("x", "y"):
                ax_hex.tick_params(axis=axis, which=which, length=length, **sides)

        # 1D histogram (dim x)
        ax_dim1 = plt.subplot(gs[0])
        bins = np.histogram(
            np.hstack((found_pycbc["mchirp"], found_sage["mchirp"])), bins=40
        )[1]
        ax_dim1.hist(
            found_pycbc["mchirp"],
            density=False,
            bins=bins,
            alpha=0.8,
            histtype="step",
            color=COLORS_HISTOGRAM[0],
            linewidth=1.5,
            linestyle="dashed",
        )
        ax_dim1.hist(
            found_sage["mchirp"],
            density=False,
            bins=bins,
            alpha=1.0,
            histtype="step",
            color=COLORS_HISTOGRAM[1],
            linewidth=1.5,
            linestyle="solid",
        )
        ax_dim1.set_xticklabels([])
        yticks = ax_dim1.yaxis.get_major_ticks()
        yticks[0].set_visible(False)
        ax_dim1.tick_params(
            bottom=False, left=True, right=True, top=True, direction="in"
        )
        ax_dim1.set_ylabel(r"$\mathregular{N}_{F}$")

        # 1D histogram (dim y)
        ax_dim2 = plt.subplot(gs[3])
        bins = np.histogram(np.hstack((found_pycbc["q"], found_sage["q"])), bins=40)[1]
        ax_dim2.hist(
            found_pycbc["q"],
            density=False,
            bins=bins,
            alpha=0.8,
            histtype="step",
            color=COLORS_HISTOGRAM[0],
            linewidth=1.5,
            linestyle="dashed",
            orientation="horizontal",
        )
        ax_dim2.hist(
            found_sage["q"],
            density=False,
            bins=bins,
            alpha=1.0,
            histtype="step",
            color=COLORS_HISTOGRAM[1],
            linewidth=1.5,
            linestyle="solid",
            orientation="horizontal",
        )
        ax_dim2.set_yticklabels([])
        ax_dim2.tick_params(
            bottom=True, left=False, right=True, top=True, direction="in"
        )
        xticks = ax_dim2.xaxis.get_major_ticks()
        xticks[0].set_visible(False)
        ax_dim2.set_xlabel(r"$\mathregular{N}_{F}$")

        # Legend for 1D histograms
        sage_line = mlines.Line2D(
            [], [], color=COLORS_HISTOGRAM[1], label="Sage", linewidth=2.0
        )
        pycbc_line = mlines.Line2D(
            [],
            [],
            color=COLORS_HISTOGRAM[0],
            label="PyCBC",
            linewidth=2.0,
            linestyle="dashed",
        )
        # left/right, up/down, width, height
        plt.legend(
            handles=[sage_line, pycbc_line],
            bbox_to_anchor=(-1.9, 0.6, 0.5, 0.5),
            loc="center left",
            frameon=False,
            handlelength=2.6,
            fontsize=18,
        )

        # left/right, up/down, width, height
        cax = ax_hex.inset_axes([5.0, -0.9, 40.0, 0.4], transform=ax_hex.transData)
        cbar = fig.colorbar(
            hexb,
            cax=cax,
            orientation="horizontal",
            ticks=np.arange(lower_hexbin, upper_hexbin + 1, 3.0),
        )
        cbar.set_label(r"$\mathregular{N^F_{sage} - N^F_{pycbc}}$")
        cbar.ax.tick_params(labelsize=12)
        tick_rotation = 0
        plt.setp(
            cbar.ax.get_xticklabels(),
            rotation=tick_rotation,
            horizontalalignment="center",
        )

        # Found and missed numbers
        both, only_sage, only_pycbc = found_counts(found_sage, found_pycbc)
        text = dict(transform=ax_hex.transAxes, horizontalalignment="right")
        plt.text(0.9, 0.7, f"Found by both = {both}", fontsize=11, **text)
        plt.text(
            0.9, 0.7 - 0.060, f"Found only by Sage = {only_sage}  ", fontsize=11, **text
        )
        plt.text(
            0.9, 0.7 - 0.120, f"Found only by PyCBC = {only_pycbc}", fontsize=11, **text
        )

        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()


def plot_corner(
    found_sage: dict,
    found_pycbc: dict,
    limits: tuple[int, int],
    cmap: ListedColormap,
    path: str,
) -> None:
    """
    Corner plot of the found-count difference over ``PARAMS``, with
    each pipeline's histograms on the diagonal.

    Parameters
    ----------
    found_sage, found_pycbc : dict
        As ``found_parameters``.
    limits : tuple of int
        As ``hexbin_limits``.
    cmap : matplotlib.colors.ListedColormap
        As ``blue_red_colormap``.
    path : str
        Figure file.
    """
    lower_hexbin, upper_hexbin = limits
    num_rows = len(PARAMS)
    num_cols = len(PARAMS)
    with plt.rc_context({"font.size": 22, "font.family": "sans-serif"}):
        fig = plt.figure(figsize=(4 * num_rows, 4 * num_cols), facecolor="white")
        gs = gridspec.GridSpec(num_rows, num_cols, wspace=0, hspace=0)
        tick_rotation = 45.0  # degrees
        needs_colorbar = True

        labelsize = 18
        fontsize = 18
        legendsize = 22

        coords = itertools.product(np.arange(num_rows), np.arange(num_cols))
        names = itertools.product(PARAMS, PARAMS)
        for (param_1, param_2), coord in zip(names, coords):
            if coord[1] > coord[0]:
                continue

            if coord[0] == coord[1]:
                ax = plt.subplot(gs[int(num_cols * coord[0] + coord[1])])
                # Make histogram
                bins = np.histogram(
                    np.hstack((found_pycbc[param_1], found_sage[param_1])), bins=40
                )[1]
                ax.hist(
                    found_pycbc[param_1],
                    density=False,
                    bins=bins,
                    alpha=0.8,
                    histtype="step",
                    color=COLORS_HISTOGRAM[0],
                    linewidth=1.5,
                    linestyle="dashed",
                )
                ax.hist(
                    found_sage[param_1],
                    density=False,
                    bins=bins,
                    alpha=1.0,
                    histtype="step",
                    color=COLORS_HISTOGRAM[1],
                    linewidth=1.5,
                    linestyle="solid",
                )

                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.tick_params(axis="y", which="major", labelsize=16)
                plt.setp(ax.get_yticklabels()[0], visible=False)
                ax.set_ylabel(r"$N^F$", fontsize=16)

                plt.setp(
                    ax.get_xticklabels(),
                    rotation=tick_rotation,
                    horizontalalignment="center",
                )
                plt.setp(
                    ax.get_yticklabels(),
                    rotation=tick_rotation,
                    horizontalalignment="left",
                    va="center",
                )

                if coord[1] == 0:
                    ax.set_xticklabels([])
                elif coord[0] == num_cols - 1:
                    ax.set_xlabel(LABELS[param_2])
                else:
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                continue

            ppaxis = plt.subplot(gs[int(num_cols * coord[0] + coord[1])])
            x_data, y_data, found_by = hexbin_data(
                found_sage, found_pycbc, param_2, param_1
            )
            hexb = plt.hexbin(
                x_data,
                y_data,
                C=found_by,
                reduce_C_function=found_difference,
                gridsize=32,
                cmap=cmap,
                norm=mpl.colors.Normalize(
                    vmin=lower_hexbin - 0.5, vmax=upper_hexbin + 0.5
                ),
            )

            if needs_colorbar:
                # left/right, up/down, width, height
                cax = ppaxis.inset_axes(
                    [190.0, 3.0, 3.0, 7.0], transform=ppaxis.transData
                )
                cbar = fig.colorbar(
                    hexb,
                    cax=cax,
                    orientation="vertical",
                    ticks=np.arange(lower_hexbin, upper_hexbin + 1, 3.0),
                )
                cbar.set_label(r"$\mathregular{N^F_{sage} - N^F_{pycbc}}$", rotation=90)
                cbar.ax.tick_params(labelsize=labelsize)
                needs_colorbar = False

            # Rotate tick labels
            plt.setp(
                ppaxis.get_xticklabels(),
                rotation=tick_rotation,
                horizontalalignment="center",
            )
            plt.setp(
                ppaxis.get_yticklabels(),
                rotation=tick_rotation,
                horizontalalignment="right",
                va="top" if param_1 == "spin2z" else "center",
            )

            ppaxis.yaxis.get_major_locator().set_params(integer=True)

            if coord[1] == 0:
                ppaxis.set_ylabel(LABELS[param_1])
                if coord[0] != num_rows - 1:
                    ppaxis.set_xticklabels([])
                    ppaxis.set_xticks([])
            if coord[0] == num_cols - 1:
                ppaxis.set_xlabel(LABELS[param_2])
                if coord[1] != 0:
                    ppaxis.set_yticklabels([])
                    ppaxis.set_yticks([])
            if coord[1] != 0 and coord[0] != num_cols - 1:
                ppaxis.set_xticklabels([])
                ppaxis.set_yticklabels([])
                ppaxis.set_xticks([])
                ppaxis.set_yticks([])

        fig.subplots_adjust(wspace=0)
        fig.subplots_adjust(hspace=0)
        # Legend
        sage_line = mlines.Line2D(
            [], [], color=COLORS_HISTOGRAM[1], label="Sage", linewidth=4.0
        )
        pycbc_line = mlines.Line2D(
            [],
            [],
            color=COLORS_HISTOGRAM[0],
            label="PyCBC",
            linewidth=4.0,
            linestyle="dashed",
        )
        # left/right, up/down, width, height
        plt.legend(
            handles=[sage_line, pycbc_line],
            bbox_to_anchor=(-0.45, 0.9, 1.0, 6.5),
            loc="center left",
            frameon=False,
            fontsize=legendsize,
            handlelength=2.6,
        )

        # Found and missed numbers
        both, only_sage, only_pycbc = found_counts(found_sage, found_pycbc)
        text = dict(transform=ax.transAxes, horizontalalignment="right")
        plt.text(0.35, 3.8, f"Found by both = {both}", fontsize=fontsize, **text)
        plt.text(
            0.35,
            3.8 - 0.150,
            f"Found only by Sage = {only_sage}  ",
            fontsize=fontsize,
            **text,
        )
        plt.text(
            0.35,
            3.8 - 0.300,
            f"Found only by PyCBC = {only_pycbc}  ",
            fontsize=fontsize,
            **text,
        )

        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()


def by_far(stats: dict, key: str) -> np.ndarray:
    """
    One of the run's statistics by increasing FAR, zero FAR left out.

    Parameters
    ----------
    stats : dict
        The run's statistics, as returned by ``get_stats``.
    key : str
        The statistic.

    Returns
    -------
    numpy.ndarray
    """
    return stats[key][stats["far"].argsort()][1:]


def read_sensitivity(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    FAR (per s) and sensitive distance of a team's ``eval.hdf``.

    Parameters
    ----------
    path : str
        The team's ``eval.hdf``.

    Returns
    -------
    tuple of numpy.ndarray
    """
    with h5py.File(path, "r") as fp:
        return np.array(fp["far"]), np.array(fp["sensitive-distance"])


def plot_sensitivity_pycbc(
    stats: dict, duration: float, pycbc_eval: str, path: str
) -> None:
    """
    Sensitive distance against FAR per month of the run and of PyCBC.

    Parameters
    ----------
    stats : dict
        The run's statistics, as returned by ``get_stats``.
    duration : float
        Duration of the foreground data in s.
    pycbc_eval : str
        PyCBC's ``eval.hdf``.
    path : str
        Figure file.
    """
    far = by_far(stats, "far")[:1000] * duration * (MONTH / duration)
    sens = by_far(stats, "sensitive-distance")[:1000]
    far_team2, sens_team2 = read_sensitivity(pycbc_eval)

    with plt.rc_context({"font.size": 16, "font.family": "sans-serif"}):
        plt.figure(figsize=(12.0, 8.0))
        plt.plot(
            far_team2 * MONTH,
            sens_team2,
            color="#D44842",
            linewidth=2.0,
            linestyle="dashed",
            label="PyCBC",
        )
        plt.plot(far, sens, linewidth=2.0, color="#65156E", label="Sage")

        plt.grid(True, which="both")
        plt.xlim(1000, 1)
        plt.ylim(1400, 2050)
        plt.xscale("log")
        plt.xlabel("False Alarm Rate (FAR) per month")
        plt.ylabel("Sensitive Distance [MPc]")
        plt.legend()
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def plot_sensitivity_teams(
    stats: dict, duration: float, teams: list[dict], path: str
) -> None:
    """
    Sensitive distance against FAR per month of the run and every
    MLGWSC-1 team.

    Parameters
    ----------
    stats : dict
        The run's statistics, as returned by ``get_stats``.
    duration : float
        Duration of the foreground data in s.
    teams : list of dict
        The MLGWSC-1 teams, each with its ``eval.hdf`` as ``eval``.
    path : str
        Figure file.
    """
    far = by_far(stats, "far") * duration * (MONTH / duration)
    sens = by_far(stats, "sensitive-distance")
    evals = {team["name"]: team["eval"] for team in teams}

    with plt.rc_context({"font.size": 16, "font.family": "sans-serif"}):
        plt.figure(figsize=(12.0, 7.0))
        plt.plot(far, sens, color="red", linewidth=1.0, label="Sage")
        for name, (color, linewidth, linestyle, label) in TEAM_STYLES.items():
            far_team, sens_team = read_sensitivity(evals[name])
            plt.plot(
                far_team * MONTH,
                sens_team,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=label,
            )

        plt.grid(True, which="both")
        plt.xlim(1000, 1)
        plt.ylim(500, 2100)
        plt.xscale("log")
        plt.xlabel("False Alarm Rate (FAR) per month")
        plt.ylabel("Sensitive Distance [MPc]")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()


def loudest_near(events: np.ndarray, injection_times: np.ndarray) -> np.ndarray:
    """
    Loudest ranking statistic of the events within ``ROC_WINDOW_IN_S``
    of each injection time; -inf where there is none.

    Parameters
    ----------
    events : numpy.ndarray, shape (3, N)
        Time, ranking statistic and time tolerance of each event.
    injection_times : numpy.ndarray
        Injection times.

    Returns
    -------
    numpy.ndarray
    """
    order = events[0].argsort()
    times, ranking_stats = events[0][order], events[1][order]
    starts = np.searchsorted(times, injection_times + ROC_WINDOW_IN_S[0], "left")
    ends = np.searchsorted(times, injection_times + ROC_WINDOW_IN_S[1], "right")
    return np.array(
        [
            np.max(ranking_stats[start:end]) if end > start else -np.inf
            for start, end in zip(starts, ends)
        ]
    )


def roc(team: dict, injection_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ROC of a team from its loudest foreground (signal) and background
    (noise) event near each injection time.

    Parameters
    ----------
    team : dict
        A team with ``fgevents`` and ``bgevents``.
    injection_times : numpy.ndarray
        Injection times.

    Returns
    -------
    fpr, tpr : numpy.ndarray
        False and true alarm probabilities.
    """
    outputs = np.concatenate(
        (
            loudest_near(team["fgevents"], injection_times),
            loudest_near(team["bgevents"], injection_times),
        )
    )
    labels = np.concatenate(
        (np.full(len(injection_times), 1.0), np.full(len(injection_times), 0.0))
    )
    # Prediction probabilities from the ranking statistics
    outputs = 1.0 / (1.0 + np.exp(-outputs))
    fpr, tpr, _ = metrics.roc_curve(labels, outputs)
    return fpr, tpr


def plot_roc(run: dict, pycbc: dict, path: str) -> None:
    """
    ROC of the run and of PyCBC on the injection times.

    Parameters
    ----------
    run, pycbc : dict
        The run and PyCBC, as filled by ``get_stats``.
    path : str
        Figure file.
    """
    fpr_sage, tpr_sage = roc(run, run["tc"])
    fpr_pycbc, tpr_pycbc = roc(pycbc, run["tc"])

    with plt.rc_context({"font.size": 18, "font.family": "sans-serif"}):
        plt.figure(figsize=(12.0, 8.0))
        plt.loglog(fpr_sage, tpr_sage, c="#65156E", label="Sage", linewidth=2.0)
        plt.loglog(fpr_pycbc, tpr_pycbc, c="#D44842", label="PyCBC", linewidth=2.0)
        plt.plot([0, 1], [0, 1], label="Random Classifier", c="k", linestyle="dashed")
        plt.xlabel("False Alarm Probability")
        plt.ylabel("True Alarm Probability")
        plt.xlim(0.0, 1.0)
        plt.ylim(1e-2, 1.0)
        plt.grid(which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()


def plot_paper(
    run: dict, teams: list[dict], stats: dict, duration: float, output_dir: str
) -> None:
    """
    Draw the paper's figures into ``output_dir/PAPER``.

    The found-signal figures are left out when either background has
    no event, since no FAR threshold is then reached.

    Parameters
    ----------
    run : dict
        The run, as filled by ``get_stats``.
    teams : list of dict
        The MLGWSC-1 teams, as ``run``; PyCBC is the reference.
    stats : dict
        The run's statistics, as returned by ``get_stats``.
    duration : float
        Duration of the foreground data in s.
    output_dir : str
        Directory of the evaluation.

    Raises
    ------
    ValueError
        If no team is named PyCBC.
    """
    names = [team["name"] for team in teams]
    if "PyCBC" not in names:
        raise ValueError(f"No PyCBC among the MLGWSC-1 teams: {names}")
    pycbc = teams[names.index("PyCBC")]
    paper_dir = os.path.join(output_dir, "PAPER")
    os.makedirs(paper_dir, exist_ok=False)

    if len(run["far_thresholds"]) and len(pycbc["far_thresholds"]):
        found_sage = found_parameters(run)
        found_pycbc = found_parameters(pycbc)
        limits = hexbin_limits(found_sage, found_pycbc)
        cmap = blue_red_colormap(limits)
        plot_mchirp_histogram(
            found_sage,
            found_pycbc,
            os.path.join(paper_dir, "compare_sage_pycbc_mchirp.png"),
        )
        plot_mchirp_q(
            found_sage,
            found_pycbc,
            limits,
            cmap,
            os.path.join(paper_dir, "compare_hexbinned_2params.png"),
        )
        plot_corner(
            found_sage,
            found_pycbc,
            limits,
            cmap,
            os.path.join(paper_dir, "compare_hexbinned_sage_pycbc.png"),
        )
    else:
        logger.info("Leaving out the found-signal figures: a background is empty")

    plot_sensitivity_pycbc(
        stats,
        duration,
        pycbc["eval"],
        os.path.join(paper_dir, "sensitive_distance_pycbc_sage.png"),
    )
    plot_sensitivity_teams(
        stats,
        duration,
        teams,
        os.path.join(paper_dir, "sensitive_distance_ML_comparison.png"),
    )
    plot_roc(run, pycbc, os.path.join(paper_dir, "roc_pycbc_sage.png"))
