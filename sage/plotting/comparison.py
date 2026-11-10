"""
The paper's figures comparing several runs, drawn by ``sage compare``:
from ``compare.h5`` and from each run's MLGWSC-1 benchmark.
"""

import itertools
import string
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from numpy.lib.stride_tricks import sliding_window_view

from sage.benchmark.mlgwsc1.evaluator import FOUND_FILE, MONTH
from sage.benchmark.mlgwsc1.paper_plots import COLORS_HISTOGRAM
from sage.core.logger import get_logger

logger = get_logger(__name__)

BENCHMARK = Path("results") / "mlgwsc1" / "best" / "evaluation"  # of best.pt
WINDOW_LENGTH = 1000  # signals in each sliding window (cells 64, 123)
OUTPUT_PARAMS = ["network_snr", "mchirp", "q"]  # cell 64
PARAM_TO_LABEL = {
    "network_snr": r"$\mathregular{Signal-to-Noise\;Ratio}$",
    "mchirp": r"$\mathregular{Chirp\;Mass}\;(\mathregular{M}_\odot)$",
    "q": r"$\mathregular{Mass\;Ratio}$",
}
FILL_COLORS = ["blue", "red"]  # cell 64, one per run
PAIR_COLORS = ["#EB6123", "#2F67B1"]  # cell 123: a panel's two runs
PAIR_ALPHAS = [0.2, 0.4]  # cell 123, their bands
SENSITIVITY_COLORS = ["#6A4A3C", "#65156E", "#025196", "#F6A482"]  # cell 74
ABLATION_COLORS = [  # cell 149
    "#742881", "#1B7939", "#855e46", "#E30B5C", "#DA957B", "m"
]


def read_compare(path: Path) -> dict:
    """
    Every run's ranking statistics and estimates in ``compare.h5``.

    Parameters
    ----------
    path : Path
        A complete ``compare.h5``.

    Returns
    -------
    dict
        ``names``; per run the ``noise`` ranking statistics, the
        ``signal`` outputs and their ``columns``; the signals'
        ``params`` by name and their ``snr``.
    """
    with h5py.File(path, "r") as file:
        names = [str(name) for name in file.attrs["runs"]]
        params = file["foreground/signal_params"]
        return {
            "names": names,
            "noise": {name: file["background"][name][:, 0] for name in names},
            "signal": {name: file["foreground"][name][:] for name in names},
            "columns": {
                name: [str(c) for c in file["foreground"][name].attrs["columns"]]
                for name in names
            },
            "params": dict(zip([str(n) for n in params.attrs["names"]], params[:].T)),
            "snr": file["foreground/target_snr"][:, 0],
        }


def loud_signals(data: dict, name: str, param: np.ndarray) -> np.ndarray | None:
    """
    A run's signals louder than its loudest noise window, by parameter.

    Parameters
    ----------
    data : dict
        Output of ``read_compare``.
    name : str
        The run.
    param : numpy.ndarray
        The parameter of every signal.

    Returns
    -------
    numpy.ndarray or None
        Rows of ranking statistic and parameter, by increasing
        parameter; None (logged) if fewer than ``WINDOW_LENGTH``.
    """
    signal_data = np.column_stack((data["signal"][name][:, 0], param))
    req_signal_data = signal_data[signal_data[:, 0] > max(data["noise"][name])]
    req_signal_data = req_signal_data[req_signal_data[:, 1].argsort()]
    if len(req_signal_data) < WINDOW_LENGTH:
        logger.info(
            "%s: %d signals above the loudest noise window, fewer than %d; "
            "its curve is left out",
            name,
            len(req_signal_data),
            WINDOW_LENGTH,
        )
        return None
    return req_signal_data


def sliding_bands(req_signal_data: np.ndarray, relative: bool) -> np.ndarray:
    """
    Median and 5 and 95 percentiles of the ranking statistic in sliding
    windows of ``WINDOW_LENGTH`` signals.

    Parameters
    ----------
    req_signal_data : numpy.ndarray
        Output of ``loud_signals``.
    relative : bool
        Take each window's statistics above its smallest (cell 123).

    Returns
    -------
    numpy.ndarray
        Per window: its first parameter, median, 5 and 95 percentile.
    """
    split = sliding_window_view(req_signal_data, WINDOW_LENGTH, axis=0)
    if relative:
        return np.array(
            [
                [
                    chunk[1][0],
                    np.median(chunk[0] - min(chunk[0])),
                    np.percentile(chunk[0] - min(chunk[0]), 5),
                    np.percentile(chunk[0] - min(chunk[0]), 95),
                ]
                for chunk in split
            ]
        )
    return np.array(
        [
            [
                chunk[1][0],
                np.median(chunk[0]),
                np.percentile(chunk[0], 5),
                np.percentile(chunk[0], 95),
            ]
            for chunk in split
        ]
    )


def plot_output_vs_parameters(data: dict, path: Path) -> None:
    """
    Ranking statistic of loud signals against their SNR, chirp mass and
    mass ratio, per run (cell 64).

    Parameters
    ----------
    data : dict
        Output of ``read_compare``.
    path : Path
        Figure file.
    """
    names = data["names"]
    if len(names) > len(FILL_COLORS):
        logger.info(
            "No %s: cell 64 has colours for %d runs", path.name, len(FILL_COLORS)
        )
        return

    signal_params = {
        "network_snr": data["snr"],
        "mchirp": data["params"]["mchirp"],
        "q": data["params"]["mass1"] / data["params"]["mass2"],
    }
    params = OUTPUT_PARAMS
    ncols = 3
    nrows = len(params) // ncols + int(len(params) % ncols or False)

    with plt.rc_context({"font.size": 22, "font.family": "sans-serif"}):
        fig, ax = plt.subplots(
            nrows, ncols, figsize=(8.0 * ncols, 5.0 * nrows), squeeze=False
        )
        pidxs = list(itertools.product(range(nrows), range(ncols)))
        num_fin = 0
        for param, (i, j) in zip(params, pidxs):
            for name, color in zip(names, FILL_COLORS):
                req_signal_data = loud_signals(data, name, signal_params[param])
                if req_signal_data is None:
                    continue
                # Smoothing and calculating error bars
                plot_data = sliding_bands(req_signal_data, relative=False)
                ax[i][j].plot(plot_data[:, 0], plot_data[:, 1], label=name)
                ax[i][j].set_xlim(min(plot_data[:, 0]), max(plot_data[:, 0]))
                ax[i][j].fill_between(
                    plot_data[:, 0],
                    plot_data[:, 2],
                    plot_data[:, 3],
                    color=color,
                    alpha=0.2,
                )

            ax[i][j].set_xlabel(PARAM_TO_LABEL[param])
            ax[i][j].set_ylabel("Network Output")
            ax[i][j].grid(True)
            ax[i][j].legend()
            num_fin += 1

        for i, j in pidxs[num_fin:]:
            ax[i][j].set_visible(False)

        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


def compare_vout(
    fig, gs, outputs: list, labels: list, limits: list, plot_xlabel: bool
) -> None:
    """
    Distributions of two runs' ranking statistics on noise (left) and
    signals (right), as histograms and kernel density estimates.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    gs : matplotlib.gridspec.GridSpecFromSubplotSpec
        Its two cells.
    outputs : list of tuple of numpy.ndarray
        Per run: noise and signal ranking statistics.
    labels : list of str
        The two runs.
    limits : list of list of float
        x limits of the noise and of the signal panel.
    plot_xlabel : bool
        Label the x axes.
    """
    cell_1 = gs[0]
    cell_2 = gs[1]

    ### CELL 1 ###
    compare_1 = outputs[0][0]
    compare_2 = outputs[1][0]

    nbins = 1024
    bins = np.histogram(np.hstack((compare_1, compare_2)), bins=nbins)[1]

    ax_1 = fig.add_subplot(cell_1)
    step = dict(bins=bins, histtype="step", linewidth=0.5, density=True)
    ax_1.hist(compare_1, color="blue", **step)
    sns.kdeplot(data=compare_1, cut=3, gridsize=512, label=labels[0], ax=ax_1)
    ax_1.hist(compare_2, color="orange", **step)
    sns.kdeplot(data=compare_2, cut=3, gridsize=512, label=labels[1], ax=ax_1)

    # We change the fontsize of minor ticks label
    ax_1.tick_params(axis="both", which="major", labelsize=16)
    ax_1.tick_params(axis="both", which="minor", labelsize=16)
    ax_1.set_xlim(limits[0][0], limits[0][1])
    ax_1.set_ylabel("Density", fontsize=16)

    ### CELL 2 ###
    compare_1 = outputs[0][1]
    compare_2 = outputs[1][1]

    nbins = 1024
    bins = np.histogram(np.hstack((compare_1, compare_2)), bins=nbins)[1]

    ax_2 = fig.add_subplot(cell_2)
    step = dict(bins=bins, histtype="step", linewidth=0.5, density=True)
    ax_2.hist(compare_1, color="blue", **step)
    sns.kdeplot(data=compare_1, cut=3, gridsize=512, label=labels[0], ax=ax_2)
    ax_2.hist(compare_2, color="orange", **step)
    sns.kdeplot(data=compare_2, cut=3, gridsize=512, label=labels[1], ax=ax_2)

    # We change the fontsize of minor ticks label
    ax_2.tick_params(axis="both", which="major", labelsize=16)
    ax_2.tick_params(axis="both", which="minor", labelsize=16)
    ax_2.set_xlim(limits[1][0], limits[1][1])
    ax_2.set_ylabel("Density", fontsize=16)

    # hide the spines between ax and ax2
    ax_1.spines["right"].set_linestyle((0, (4, 4)))
    ax_2.spines["left"].set_linestyle((0, (4, 4)))
    ax_1.yaxis.tick_left()
    ax_2.yaxis.tick_right()
    ax_2.yaxis.set_label_position("right")
    ax_1.grid(True)
    ax_2.grid(True)
    ax_2.legend(fontsize=14)

    if plot_xlabel:
        ax_1.set_xlabel("Network Output", fontsize=16)
        ax_2.set_xlabel("Network Output", fontsize=16)

    plt.setp(ax_1.get_xticklabels()[-1], visible=False)
    plt.setp(ax_2.get_xticklabels()[0], visible=False)


def plot_output_distributions(data: dict, path: Path) -> None:
    """
    ``compare_vout`` of the first run against every other run, one row
    each (cells 108 to 110), x limits spanning both runs' data.

    Parameters
    ----------
    data : dict
        Output of ``read_compare``.
    path : Path
        Figure file.
    """
    names = data["names"]
    if len(names) < 2:
        logger.info("No %s: it compares two runs", path.name)
        return

    outputs = {
        name: (data["noise"][name], data["signal"][name][:, 0]) for name in names
    }
    rows = len(names) - 1
    with plt.rc_context({"font.size": 22, "font.family": "sans-serif"}):
        # Make outer gridspec
        fig = plt.figure(figsize=(10.0 * 2, 5.0 * rows))
        plt.subplots_adjust(wspace=0.01)
        outer = fig.add_gridspec(rows, 1, height_ratios=[2] * rows)
        for row, name in enumerate(names[1:]):
            # Make nested gridspecs
            gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row])
            pair = [outputs[names[0]], outputs[name]]
            limits = [
                [float(min(a.min() for a, _ in pair)), float(max(a.max() for a, _ in pair))],
                [float(min(b.min() for _, b in pair)), float(max(b.max() for _, b in pair))],
            ]
            compare_vout(
                fig,
                gs,
                pair,
                labels=[names[0], name],
                limits=limits,
                plot_xlabel=row == rows - 1,
            )

        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_output_vs_mchirp(data: dict, path: Path) -> None:
    """
    Ranking statistic of loud signals above each window's smallest,
    against chirp mass; two runs per panel (cell 123, bottom row).

    Parameters
    ----------
    data : dict
        Output of ``read_compare``.
    path : Path
        Figure file.
    """
    names = data["names"]
    panels = [names[k : k + 2] for k in range(0, len(names), 2)]
    with plt.rc_context({"font.size": 22, "font.family": "sans-serif"}):
        fig, ax = plt.subplots(
            1, len(panels), figsize=(9 * len(panels), 8 * 1.0), squeeze=False
        )
        plt.subplots_adjust(wspace=0.0)
        for k, panel in enumerate(panels):
            axis = ax[0][k]
            for name, color, alpha in zip(panel, PAIR_COLORS, PAIR_ALPHAS):
                req_signal_data = loud_signals(data, name, data["params"]["mchirp"])
                if req_signal_data is None:
                    continue
                # Smoothing and calculating error bars
                plot_data = sliding_bands(req_signal_data, relative=True)
                axis.plot(plot_data[:, 0], plot_data[:, 1], label=name, c=color)
                axis.set_xlim(min(plot_data[:, 0]), max(plot_data[:, 0]))
                axis.fill_between(
                    plot_data[:, 0],
                    plot_data[:, 2],
                    plot_data[:, 3],
                    color=color,
                    alpha=alpha,
                )
            axis.set_ylim(0, 10)
            axis.set_xlabel(r"Chirp Mass (M$_{\odot}$)")
            if k == 0:
                axis.set_ylabel("Network Output")
            else:
                axis.set_yticklabels([])
                axis.set_yticks([])
            axis.legend()

        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def prior_bounds(scaling: dict) -> tuple:
    """
    A target's prior bounds from its scaling.

    Parameters
    ----------
    scaling : dict
        The target's ``offset`` (middle of the bounds) and ``scale``
        (half their width), from the run's ``derived.yaml``.

    Returns
    -------
    tuple of float
        Lower and upper bound.
    """
    return scaling["offset"] - scaling["scale"], scaling["offset"] + scaling["scale"]


def plot_point_estimates(data: dict, run_dirs: list, path: Path) -> None:
    """
    Predicted against actual chirp mass and time of coalescence in
    physical units, coloured by network SNR; one row per run (cells
    127 to 130), each axis spanning the run's prior bounds.

    Parameters
    ----------
    data : dict
        Output of ``read_compare``.
    run_dirs : list of Path
        The runs, in the order of ``data["names"]``.
    path : Path
        Figure file.
    """
    rows, bounds = [], []
    for name, run_dir in zip(data["names"], run_dirs):
        columns = data["columns"][name]
        if not {"mu_mchirp", "mu_delta_tc"} <= set(columns):
            logger.info(
                "%s: no mchirp and delta_tc estimates; left out of %s", name, path.name
            )
            continue
        scaling = yaml.safe_load((run_dir / "resolved" / "derived.yaml").read_text())
        scaling = scaling["target_scaling"]
        signal = data["signal"][name]
        mc_labels = data["params"]["mchirp"]
        tc_labels = data["params"]["delta_tc"]
        mc_output = signal[:, columns.index("mu_mchirp")]
        tc_output = signal[:, columns.index("mu_delta_tc")]
        snrs = data["snr"]
        values = np.column_stack((snrs, mc_labels, mc_output, tc_labels, tc_output))
        rows.append(values[values[:, 0].argsort()])
        bounds.append(
            (prior_bounds(scaling["mchirp"]), prior_bounds(scaling["delta_tc"]))
        )
    if not rows:
        return

    with plt.rc_context({"font.size": 12, "font.family": "sans-serif"}):
        fig, ax = plt.subplots(
            len(rows),
            2,
            figsize=(8 * 2.0, 7 * len(rows)),
            gridspec_kw={"width_ratios": [1.6, 2]},
            squeeze=False,
        )
        cmap = mpl.colormaps["RdYlBu_r"].resampled(10)
        for row, (sorted_run, (mc_bounds, tc_bounds)) in enumerate(zip(rows, bounds)):
            points = dict(marker=".", s=25.0, c=sorted_run[:, 0], cmap=cmap)
            ax[row][0].scatter(sorted_run[:, 2], sorted_run[:, 1], **points)
            ax[row][0].plot(mc_bounds, mc_bounds, linestyle="dashed", color="k")
            ax[row][0].set_xlabel(r"Predicted Chirp Mass ($\mathregular{M}_\odot$)")
            ax[row][0].set_ylabel(r"Actual Chirp Mass ($\mathregular{M}_\odot$)")
            ax[row][0].grid(True, which="both")
            ax[row][0].set_xlim(*mc_bounds)
            ax[row][0].set_ylim(*mc_bounds)

            foo = ax[row][1].scatter(sorted_run[:, 4], sorted_run[:, 3], **points)
            ax[row][1].plot(tc_bounds, tc_bounds, linestyle="dashed", color="k")
            ax[row][1].set_xlabel("Predicted Time of Coalescence (s)")
            ax[row][1].set_ylabel("Actual Time of Coalescence (s)")
            ax[row][1].grid(True, which="both")
            ax[row][1].set_xlim(*tc_bounds)
            ax[row][1].set_ylim(*tc_bounds)

            cbar = fig.colorbar(foo)
            cbar.set_label("Network SNR", rotation=270, labelpad=40)

        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)


def read_found(path: Path) -> dict:
    """
    Chirp masses of the injections found above 1 false alarm per month,
    for the run and PyCBC, from a benchmark's ``found.hdf``.

    Parameters
    ----------
    path : Path
        ``found.hdf`` of an evaluation.

    Returns
    -------
    dict
        ``run`` and ``PyCBC`` chirp masses (None where the background
        reaches no threshold) and the foreground ``duration``.
    """
    with h5py.File(path, "r") as fp:
        mchirp = fp["mchirp"][:]
        found = {"duration": float(fp.attrs["duration"])}
        for name in ("run", "PyCBC"):
            thresholds = fp[name]["far_thresholds"][:]
            found_idx = fp[name]["found_idx"][:]
            found_stats = fp[name]["found_stats"][:]
            found[name] = None
            if len(thresholds):
                found[name] = mchirp[found_idx[found_stats > thresholds[0]]]
    return found


def plot_sensitivity_runs(
    evaluations: dict, found: dict, pycbc_eval: Path, path: Path
) -> None:
    """
    Sensitive distance against FAR per month of every run and of PyCBC
    (cell 74).

    Parameters
    ----------
    evaluations : dict
        Each run's evaluation folder, by name.
    found : dict
        Output of ``read_found`` by run.
    pycbc_eval : Path
        PyCBC's ``eval.hdf``.
    path : Path
        Figure file.
    """
    if len(evaluations) > len(SENSITIVITY_COLORS):
        logger.info(
            "No %s: cell 74 has colours for %d runs",
            path.name,
            len(SENSITIVITY_COLORS),
        )
        return

    with plt.rc_context({"font.size": 16, "font.family": "sans-serif"}):
        plt.figure(figsize=(12.0, 8.0))
        with h5py.File(pycbc_eval, "r") as fp:
            sens_team2 = np.array(fp["sensitive-distance"])
            far_team2 = np.array(fp["far"])
            plt.plot(
                far_team2 * MONTH,
                sens_team2,
                color="#D44842",
                linewidth=2.0,
                linestyle="dashed",
                label="PyCBC",
            )

        for (name, evaluation), color in zip(evaluations.items(), SENSITIVITY_COLORS):
            dur = found[name]["duration"]
            with h5py.File(evaluation / "eval.hdf", "r") as fp:
                far = fp["far"][()]
                sens = fp["sensitive-distance"][()]
                sidxs = far.argsort()
                far = far[sidxs][1:] * dur
                sens = sens[sidxs][1:]
                plt.plot(
                    far * (MONTH / dur), sens, color=color, linewidth=2.0, label=name
                )

        plt.grid(True, which="both")
        plt.xlim(1000, 1)
        plt.ylim(1400, 2050)
        plt.xscale("log")
        plt.xlabel("False Alarm Rate (FAR) per month")
        plt.ylabel("Sensitive Distance [MPc]")
        plt.legend()
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def plot_found_mchirp(found: dict, path: Path) -> None:
    """
    Chirp masses found above 1 false alarm per month by PyCBC and two
    runs per panel (cell 123, top row).

    Parameters
    ----------
    found : dict
        Output of ``read_found`` by run.
    path : Path
        Figure file.
    """
    names = list(found)
    mchirp_pycbc = found[names[0]]["PyCBC"]
    panels = [names[k : k + 2] for k in range(0, len(names), 2)]
    with plt.rc_context({"font.size": 22, "font.family": "sans-serif"}):
        fig, ax = plt.subplots(
            1, len(panels), figsize=(9 * len(panels), 8 * 1.0), squeeze=False
        )
        plt.subplots_adjust(wspace=0.0)
        for k, panel in enumerate(panels):
            axis = ax[0][k]
            runs = [found[name]["run"] for name in panel]
            bins = np.histogram(np.hstack((*runs, mchirp_pycbc)), bins=40)[1]
            axis.hist(
                mchirp_pycbc,
                density=False,
                bins=bins,
                alpha=1,
                histtype="step",
                color="#0A5C36",
                linewidth=2.0,
                linestyle="dashed",
                label="D4 PyCBC",
            )
            for name, mchirp, color in zip(panel, runs, PAIR_COLORS):
                axis.hist(
                    mchirp,
                    density=False,
                    bins=bins,
                    alpha=1,
                    histtype="step",
                    color=color,
                    linewidth=2.0,
                    linestyle="solid",
                    label=name,
                )
            axis.set_ylim(0.0, 255.0)
            if k == 0:
                axis.set_ylabel("N Detected Signals")
            else:
                axis.set_yticklabels([])
                axis.set_yticks([])
            axis.set_xlabel(r"Chirp Mass ($\mathregular{M}_\odot$)")
            axis.text(0.01, 0.99, "D4", ha="left", va="top", transform=axis.transAxes)
            axis.legend()

        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_ablation_histograms(found: dict, path: Path) -> None:
    """
    Chirp masses found above 1 false alarm per month by each other run,
    with PyCBC and the first run in every panel (cells 144 to 149).

    Parameters
    ----------
    found : dict
        Output of ``read_found`` by run; the first run is the reference.
    path : Path
        Figure file.
    """
    names = list(found)
    if len(names) < 2 or len(names) - 1 > len(ABLATION_COLORS):
        logger.info(
            "No %s: it needs 2 to %d runs", path.name, len(ABLATION_COLORS) + 1
        )
        return

    ref = {"pycbc": found[names[0]]["PyCBC"], "best": found[names[0]]["run"]}
    ablation_mchirp = {name: found[name]["run"] for name in names[1:]}
    hex_left, hex_right = COLORS_HISTOGRAM

    with plt.rc_context({"font.size": 20, "font.family": "sans-serif"}):
        fig, axs = plt.subplots(2, 3, figsize=(9 * 3, 6 * 2), sharex=True, sharey=True)
        coords = list(itertools.product([0, 1], [0, 1, 2]))
        bins = np.histogram(np.hstack(tuple(ablation_mchirp.values())), bins=40)[1]
        for n, (key, val) in enumerate(ablation_mchirp.items()):
            i, j = coords[n]
            color = ABLATION_COLORS[n]
            axs[i][j].hist(
                ref["pycbc"],
                bins=bins,
                histtype="step",
                linewidth=2.0,
                linestyle="dashed",
                color=hex_left,
            )
            axs[i][j].hist(
                ref["best"],
                bins=bins,
                histtype="step",
                linewidth=2.0,
                linestyle="solid",
                color=hex_right,
            )
            axs[i][j].hist(
                val,
                bins=bins,
                histtype="step",
                linewidth=2.0,
                linestyle="solid",
                label=key,
                color=color,
            )
            axs[i][j].legend()
            axs[i][j].set_ylim(0, 218)
            axs[i][j].tick_params(
                bottom=True, left=True, right=True, top=True, direction="in"
            )
            axs[i][j].text(
                0.03,
                0.96,
                "({})".format(list(string.ascii_lowercase)[n]),
                ha="left",
                va="top",
                transform=axs[i][j].transAxes,
            )
            if j == 0:
                axs[i][j].set_ylabel("N Detected Signals")
            if i == 1:
                axs[i][j].set_xlabel(r"Chirp Mass ($\mathregular{M}_\odot$)")

        # Fewer than six runs: the bottom row's x axis moves up
        for i, j in coords[len(ablation_mchirp) :]:
            axs[i][j].set_visible(False)
            if i == 1:
                axs[0][j].xaxis.set_tick_params(labelbottom=True)
                axs[0][j].set_xlabel(r"Chirp Mass ($\mathregular{M}_\odot$)")

        sage_patch = mpatches.Patch(
            color=hex_right, label=names[0], fill=None, linewidth=2.0
        )
        pycbc_patch = mpatches.Patch(
            color=hex_left, label="PyCBC", fill=None, linestyle="dashed", linewidth=2.0
        )
        fig.legend(
            handles=[sage_patch, pycbc_patch],
            loc="upper center",
            bbox_to_anchor=(0.195, 0.94),
            ncol=2,
        )
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_comparison(path: Path, run_dirs: list, data_dir: Path) -> None:
    """
    Draw the comparison figures beside ``compare.h5``.

    The benchmark figures need every run's MLGWSC-1 evaluation of
    ``best.pt``; without one they are left out, naming the runs.

    Parameters
    ----------
    path : Path
        A complete ``compare.h5``.
    run_dirs : list of Path
        The compared runs, in its order.
    data_dir : Path
        The machine's data directory.
    """
    output_dir = path.parent
    data = read_compare(path)
    plot_output_vs_parameters(data, output_dir / "compare_network_output.png")
    plot_output_distributions(data, output_dir / "compare_vout.png")
    plot_output_vs_mchirp(data, output_dir / "compare_main_runs_output.png")
    plot_point_estimates(data, run_dirs, output_dir / "parameter_point_estimate.png")

    evaluations = {
        name: Path(run_dir) / BENCHMARK
        for name, run_dir in zip(data["names"], run_dirs)
    }
    missing = [
        name
        for name, evaluation in evaluations.items()
        if not all((evaluation / file).exists() for file in ("eval.hdf", FOUND_FILE))
    ]
    if missing:
        logger.info(
            "No benchmark figures: no MLGWSC-1 evaluation of best.pt with %s for %s",
            FOUND_FILE,
            ", ".join(missing),
        )
        return

    found = {
        name: read_found(evaluation / FOUND_FILE)
        for name, evaluation in evaluations.items()
    }
    pycbc_eval = data_dir / "mlgwsc1" / "results" / "PyCBC" / "ds4" / "eval.hdf"
    plot_sensitivity_runs(
        evaluations,
        found,
        pycbc_eval,
        output_dir / "sensitive_distance_pycbc_sage_others.png",
    )

    unreached = [
        name
        for name in found
        if found[name]["run"] is None or found[name]["PyCBC"] is None
    ]
    if unreached:
        logger.info(
            "No found-signal figures: no FAR threshold reached for %s",
            ", ".join(unreached),
        )
        return
    plot_found_mchirp(found, output_dir / "compare_main_runs_found.png")
    plot_ablation_histograms(found, output_dir / "compare_all_ablation_histograms.png")
