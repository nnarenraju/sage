"""
MLGWSC-1's evaluate.py (statistics unchanged) with additions: injection
SNRs in the plots and found/missed comparisons with every team.
"""

import os
import h5py
import itertools
import numpy as np

from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})

from tqdm import tqdm

from sage.core.constants import GM
from sage.core.logger import get_logger

logger = get_logger(__name__)

FAR_INDICES = [0, 3, 29, 99, 999]  # 1, 4, 30, 100 and 1000 false alarms per month
MARGINS = (30, 30)  # no injections in a segment's first and last 30 s
MONTH = 30.0 * 24.0 * 60.0 * 60.0
FOUND_FILE = "found.hdf"  # the run's and PyCBC's found injections
COLORS = {  # sensitivity plots
    "PyCBC": "orange", "TPI_FSU_Jena": "red", "Virgo-AUTh": "blueviolet",
    "CNN-Coinc": "green", "MFCNN": "blue",
}


def reachable_thresholds(noise_stats):
    """
    Ranking statistics at the FAR thresholds the background reaches.

    Parameters
    ----------
    noise_stats : numpy.ndarray
        Background ranking statistics, sorted ascending.

    Returns
    -------
    numpy.ndarray
        The statistic at each ``FAR_INDICES`` entry below
        ``len(noise_stats)``, counted from the loudest.
    """
    reached = [index for index in FAR_INDICES if index < len(noise_stats)]
    return noise_stats[::-1][reached]


def find_injection_times(fgfiles, injfile, padding_start=0, padding_end=0):
    """
    Determine injections which are contained in the file.

    Parameters
    ----------
    fgfiles : list of str
        Paths to the files containing the foreground data (noise +
        injections).
    injfile : str
        Path to the file containing information on the injections in the
        foreground files.
    padding_start : float
        The amount of time (in seconds) at the start of each segment
        where no injections are present.
    padding_end : float
        The amount of time (in seconds) at the end of each segment
        where no injections are present.

    Returns
    -------
    duration : float
        The total duration (in seconds) of all foreground files.
    numpy.ndarray of bool
        Which injections are contained in the provided foreground
        files.
    """
    duration = 0
    times = []
    for fpath in fgfiles:
        with h5py.File(fpath, "r") as fp:
            det = list(fp.keys())[0]

            for key in fp[det].keys():
                ds = fp[f"{det}/{key}"]
                start = ds.attrs["start_time"]
                end = start + len(ds) * ds.attrs["delta_t"]
                duration += end - start
                start += padding_start
                end -= padding_end
                if end > start:
                    times.append([start, end])

    with h5py.File(injfile, "r") as fp:
        injtimes = fp["tc"][()]

    ret = np.full((len(times), len(injtimes)), False)
    for i, (start, end) in enumerate(times):
        ret[i] = np.logical_and(start <= injtimes, injtimes <= end)

    return duration, np.any(ret, axis=0)


def find_closest_index(array, value, assume_sorted=False):
    """
    Find the index of the closest element in the array for the given
    value(s).

    Parameters
    ----------
    array : numpy.ndarray
        1D numpy array.
    value : number or numpy.ndarray
        The value(s) of which the closest array element should be found.
    assume_sorted : bool
        Assume that the array is sorted. May improve evaluation speed.

    Returns
    -------
    numpy.ndarray of int
        Index, into ``array`` sorted ascending, of the element closest
        to each value.

    Raises
    ------
    ValueError
        If ``array`` is empty.
    """
    if len(array) == 0:
        raise ValueError("Cannot find closest index for empty input array.")
    if not assume_sorted:
        array = array.copy()
        array.sort()
    ridxs = np.searchsorted(array, value, side="right")
    lidxs = np.maximum(ridxs - 1, 0)
    comp = np.fabs(array[lidxs] - value) < np.fabs(
        array[np.minimum(ridxs, len(array) - 1)] - value
    )  # noqa: E127, E501
    lisbetter = np.logical_or((ridxs == len(array)), comp)
    ridxs[lisbetter] -= 1
    return ridxs


def mchirp(mass1, mass2):
    """
    Chirp mass from the component masses.

    Parameters
    ----------
    mass1, mass2 : float or numpy.ndarray
        Component masses.

    Returns
    -------
    float or numpy.ndarray
        Chirp mass, in the units of the masses.
    """
    return (mass1 * mass2) ** (3.0 / 5.0) / (mass1 + mass2) ** (1.0 / 5.0)


def figure(title="", size_x=16.0, size_y=14.0):
    """
    Single-panel figure with a centred title; sets a serif font.

    Parameters
    ----------
    title : str
        Figure super-title.
    size_x : float
        Figure width in inches.
    size_y : float
        Figure height in inches.

    Returns
    -------
    axs : matplotlib.axes.Axes
        The panel.
    fig : matplotlib.figure.Figure
        The figure.
    """
    plt.rc("font", family="serif")
    plt.rc("xtick", labelsize="medium")
    plt.rc("ytick", labelsize="medium")
    fig, axs = plt.subplots(1, figsize=(size_x, size_y))
    fig.suptitle(title, fontsize=28, y=0.92)
    return axs, fig


def _plot(
    ax,
    x=None,
    y=None,
    xlabel="x-axis",
    ylabel="y-axis",
    ls="solid",
    label="",
    c=None,
    yscale="linear",
    xscale="linear",
    histogram=False,
    scatter=False,
    save_file="",
):
    """
    Histogram, scatter or line of ``y`` against ``x`` on ``ax``.

    A legend is drawn when ``label`` is given.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    x, y : array-like or None
        Data; a histogram uses ``y`` only.
    xlabel, ylabel : str
        Axis labels.
    ls : str
        Line style of a line.
    label : str
        Legend label of a histogram or line.
    c : str or None
        Colour of a line.
    yscale, xscale : str
        Axis scales.
    histogram : bool
        Draw a 100-bin histogram of ``y``.
    scatter : bool
        Draw a scatter of ``y`` against ``x``.
    save_file : str
        Save the current figure there and close it; empty keeps it.
    """
    if histogram:
        ax.hist(y, bins=100, label=label, alpha=0.8)
    elif scatter:
        ax.scatter(x, y, marker=".", s=100.0)
    else:
        ax.plot(x, y, ls=ls, c=c, linewidth=3.0, label=label)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid(True, which="both")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label:
        ax.legend()

    if save_file != "":
        plt.savefig(save_file)
        plt.close()


def param_vs_param(output_dir, injparams, found_injections):
    """
    Scatter plots of pairs of parameters of the found injections.

    Saved under ``output_dir/PARAM_VS_PARAM``; the signal duration is
    the leading-order chirp time ``tau_0`` from 20 Hz.

    Parameters
    ----------
    output_dir : str
        Directory of the evaluation.
    injparams : dict
        Injection parameters by name.
    found_injections : numpy.ndarray, shape (2, N_found)
        Row 0: injection indices; row 1: ranking statistics.
    """
    vs_dir = os.path.join(output_dir, "PARAM_VS_PARAM")
    if not os.path.exists(vs_dir):
        os.makedirs(vs_dir, exist_ok=False)

    plot_mchirp = injparams["mchirp"][found_injections[0].astype(int)]
    plot_distance = injparams["distance"][found_injections[0].astype(int)]
    plot_q = injparams["q"][found_injections[0].astype(int)]
    plot_dchirp = injparams["chirp_distance"][found_injections[0].astype(int)]

    # Signal duration
    lf = 20.0  # Hz
    plot_signal_duration = (
        5.0
        * (8.0 * np.pi * lf) ** (-8.0 / 3.0)
        * (plot_mchirp * GM) ** (-5.0 / 3.0)
    )

    ax, _ = figure("mchirp vs distance", 12.0, 12.0)
    spath = os.path.join(vs_dir, "mchirp_vs_distance.png")
    _plot(
        ax,
        plot_mchirp,
        plot_distance,
        "Chirp Mass",
        "Distance",
        scatter=True,
        save_file=spath,
    )

    ax, _ = figure("mchirp vs q", 12.0, 12.0)
    spath = os.path.join(vs_dir, "mchirp_vs_q.png")
    _plot(
        ax,
        plot_mchirp,
        plot_q,
        "Chirp Mass",
        "Mass Ratio (m1/m2)",
        scatter=True,
        save_file=spath,
    )

    ax, _ = figure("mchirp vs dchirp", 12.0, 12.0)
    spath = os.path.join(vs_dir, "mchirp_vs_dchirp.png")
    _plot(
        ax,
        plot_mchirp,
        plot_dchirp,
        "Chirp Mass",
        "Chirp Distance",
        scatter=True,
        save_file=spath,
    )

    ax, _ = figure("q vs dchirp", 12.0, 12.0)
    spath = os.path.join(vs_dir, "q_vs_dchirp.png")
    _plot(
        ax,
        plot_q,
        plot_dchirp,
        "Mass Ratio (m1/m2)",
        "Chirp Distance",
        scatter=True,
        save_file=spath,
    )

    # Signal duration plots
    ax, _ = figure("Tau_0 vs q", 12.0, 12.0)
    spath = os.path.join(vs_dir, "tau0_vs_q.png")
    _plot(
        ax,
        plot_signal_duration,
        plot_q,
        "Signal Duration [s]",
        "Mass Ratio (m1/m2)",
        scatter=True,
        save_file=spath,
    )

    ax, _ = figure("Tau_0 vs mchirp", 12.0, 12.0)
    spath = os.path.join(vs_dir, "tau0_vs_mchirp.png")
    _plot(
        ax,
        plot_signal_duration,
        plot_mchirp,
        "Signal Duration [s]",
        "Chirp Mass",
        scatter=True,
        save_file=spath,
    )

    ax, _ = figure("Tau_0 vs dchirp", 12.0, 12.0)
    spath = os.path.join(vs_dir, "tau0_vs_dchirp.png")
    _plot(
        ax,
        plot_signal_duration,
        plot_dchirp,
        "Signal Duration [s]",
        "Chirp Distance",
        scatter=True,
        save_file=spath,
    )

    ax, _ = figure("Tau_0 vs distance", 12.0, 12.0)
    spath = os.path.join(vs_dir, "tau0_vs_distance.png")
    _plot(
        ax,
        plot_signal_duration,
        plot_distance,
        "Signal Duration [s]",
        "Distance",
        scatter=True,
        save_file=spath,
    )


def found_param_plots(noise_stats, output_dir, injparams, found_injections):
    """
    Histograms of every injection parameter, all injections against
    those found at each reached FAR threshold.

    Saved under ``output_dir/FOUND_INJECTIONS/<param>``. When the
    background reaches 1000 per month, the injections found between
    100 and 1000 per month with 10 < SNR < 100 go to ``HIGH_SNR_BAD``.

    Parameters
    ----------
    noise_stats : numpy.ndarray, shape (N_bg,)
        Background ranking statistics, sorted ascending.
    output_dir : str
        Directory of the evaluation.
    injparams : dict
        Injection parameters by name.
    found_injections : numpy.ndarray, shape (2, N_found)
        Row 0: injection indices; row 1: ranking statistics.

    Raises
    ------
    FileExistsError
        If ``HIGH_SNR_BAD/COMPARE`` exists.
    """
    # Thresholds for different false alarm rates; for one month of
    # data: 1 per month, week and day, 100 and 1000 per month
    far_thresholds = reachable_thresholds(noise_stats)
    thresh_names = [
        "1-per-month",
        "1-per-week",
        "1-per-day",
        "100-per-month",
        "1000-per-month",
    ]
    # How many signals are present above given threshold?
    far_found_idx = {
        thresh_names[n]: found_injections[0][found_injections[1] > thresh]
        for n, thresh in enumerate(far_thresholds)
    }
    # The 100 to 1000 per month band needs a background reaching 1000
    bad = len(far_thresholds) == len(FAR_INDICES)
    if bad:
        badfar_idx = found_injections[0][
            (found_injections[1] <= noise_stats[::-1][99])
            & (found_injections[1] > noise_stats[::-1][999])
        ].astype(int)
        highsnr_badfar_idx = np.argwhere(
            (injparams["snr"][badfar_idx] > 10.0)
            & (injparams["snr"][badfar_idx] < 100.0)
        )
    highsnr_badfar_params = {}

    # Injected and found histograms of every parameter
    save_dir = os.path.join(output_dir, "FOUND_INJECTIONS")
    bad_dir = os.path.join(output_dir, "HIGH_SNR_BAD")
    for param in injparams.keys():
        param_dir = os.path.join(save_dir, "{}".format(param))
        if not os.path.exists(param_dir):
            os.makedirs(param_dir, exist_ok=False)

        all_param = injparams[param]
        if bad:
            if not os.path.exists(bad_dir):
                os.makedirs(bad_dir, exist_ok=False)
            highsnr_badfar_params[param] = all_param[badfar_idx][highsnr_badfar_idx]

            # plot params of highsnr bad signals if present
            plt.figure(figsize=(12.0, 12.0))
            plt.title("SNR>10 and 1000/month>FAR>100/month {}".format(param))
            plt.hist(
                highsnr_badfar_params[param],
                bins=100,
                label="{}-bad".format(param),
                alpha=0.8,
            )
            plt.grid(True, which="both")
            plt.xlabel("{}".format(param))
            plt.ylabel("Number of Occurences")
            plt.legend()
            plt.savefig(os.path.join(bad_dir, "{}-highsnr_bad.png".format(param)))
            plt.close()

        for key, value in far_found_idx.items():
            found_param = all_param[value.astype(int)]
            # Plotting the overlap histograms of all and found data
            plt.figure(figsize=(12.0, 12.0))
            plt.title("Injected vs Found (FAR = {}) - {}".format(key, param))
            plt.hist(all_param, bins=100, label="{}-all".format(param), alpha=0.8)
            plt.hist(found_param, bins=100, label="{}-found".format(param), alpha=0.8)
            plt.grid(True, which="both")
            plt.xlabel("{}".format(param))
            plt.ylabel("Number of Occurences")
            plt.legend()
            plt.savefig(
                os.path.join(param_dir, "{}-compare_FAR_{}.png".format(param, key))
            )
            plt.close()

    if not bad:
        return

    # plotting param vs param for bad signals
    # Calculate signal duration
    lf = 20.0  # Hz
    duration = (
        lambda mchirp: 5.0
        * (8.0 * np.pi * lf) ** (-8.0 / 3.0)
        * (mchirp * GM) ** (-5.0 / 3.0)
    )
    team = {}
    team.update(highsnr_badfar_params)
    team["duration"] = duration(team["mchirp"])

    compare_bad_dir = os.path.join(bad_dir, "COMPARE")
    os.makedirs(compare_bad_dir, exist_ok=False)
    params = [
        "duration",
        "mchirp",
        "distance",
        "q",
        "chirp_distance",
        "snr",
        "mass1",
        "mass2",
    ]
    ncols = 3
    plots = list(itertools.combinations(params, 2))
    nsubplots = len(plots)
    nrows = nsubplots // ncols + int(nsubplots % ncols or False)

    fig, ax = plt.subplots(nrows, ncols, figsize=(9.0 * ncols, 6.0 * nrows))
    kwargs = {}

    pidxs = list(itertools.product(range(nrows), range(ncols)))
    num_fin = 0
    for (param_1, param_2), (i, j) in zip(plots, pidxs):
        x = team[param_1]
        y = team[param_2]

        kwargs.update({"color": "blue", "s": 100.0, "alpha": 0.7})
        ax[i][j].scatter(x, y, **kwargs)
        ax[i][j].set_xlabel(param_1)
        ax[i][j].set_ylabel(param_2)
        ax[i][j].grid(True)
        num_fin += 1

    for i, j in pidxs[num_fin:]:
        ax[i][j].set_visible(False)

    save_name = "param_vs_param.png"
    save_path = os.path.join(compare_bad_dir, save_name)
    plt.savefig(save_path)
    plt.close()


def network_output(
    found_injections, noise_stats, output_dir, team_name, lower_threshold=0.0
):
    """
    Log-count histograms of the ranking statistics of the found
    injections and of the background.

    Parameters
    ----------
    found_injections : numpy.ndarray, shape (2, N_found)
        Row 0: injection indices; row 1: ranking statistics.
    noise_stats : numpy.ndarray, shape (N_bg,)
        Background ranking statistics.
    output_dir : str
        Saves ``network_output_{team_name}.png`` in it.
    team_name : str
        Team the statistics belong to.
    lower_threshold : float
        Only statistics above it are drawn.
    """
    # Plotting the noise and signals stats for found samples
    plt.figure(figsize=(12.0, 12.0))
    foo = found_injections[1][found_injections[1] > lower_threshold]
    plt.hist(foo, label="found_injections", bins=100, alpha=0.8)
    noise_stats = noise_stats[noise_stats > lower_threshold]
    plt.hist(noise_stats, label="noise", bins=100, alpha=0.8)
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "network_output_{}.png".format(team_name)))
    plt.close()


def parameter_learning(injparams, noise_stats, found_injections, output_dir):
    """
    Ranking statistic of the found injections against each parameter,
    with the reached FAR thresholds as lines.

    Saved under ``output_dir/LEARNING``; the parameters include the
    chirp time ``signal_duration`` from 20 Hz.

    Parameters
    ----------
    injparams : dict
        Injection parameters by name.
    noise_stats : numpy.ndarray, shape (N_bg,)
        Background ranking statistics, sorted ascending.
    found_injections : numpy.ndarray, shape (2, N_found)
        Row 0: injection indices; row 1: ranking statistics.
    output_dir : str
        Directory of the evaluation.
    """
    learning_dir = os.path.join(output_dir, "LEARNING")
    if not os.path.exists(learning_dir):
        os.makedirs(learning_dir, exist_ok=False)
    source_params = {
        key: injparams[key][found_injections[0].astype(int)] for key in injparams.keys()
    }
    lf = 20.0  # Hz
    source_params["signal_duration"] = (
        5.0
        * (8.0 * np.pi * lf) ** (-8.0 / 3.0)
        * (source_params["mchirp"] * GM) ** (-5.0 / 3.0)
    )
    predicted_outputs = found_injections[1]
    save_name = "raw_value"

    # Define FAR thresholds
    far_thresholds = reachable_thresholds(noise_stats)
    thresh_names = [
        "1 per month",
        "1 per week",
        "1 per day",
        "100 per month",
        "1000 per month",
    ]
    for key in source_params.keys():
        assert len(source_params[key]) == len(predicted_outputs)
        ax, fig = figure(title="Learning {}".format(key))
        _plot(
            ax,
            x=source_params[key],
            y=predicted_outputs,
            xlabel=key,
            ylabel=save_name,
            label=key,
            yscale="linear",
            xscale="linear",
            scatter=True,
        )
        # Plotting FAR thresholds
        min_x = min(source_params[key])
        max_x = max(source_params[key])
        ax.set_xlim(min_x, max_x)
        for fthresh, nthresh in zip(far_thresholds, thresh_names):
            ax.plot([min_x, max_x], [fthresh, fthresh], label=nthresh, linewidth=2.0)
        save_path = os.path.join(
            learning_dir, "learning_{}_{}.png".format(save_name, key)
        )
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def read_data(injection_file, run, teams, idxs):
    """
    Load injection parameters and the run's and every team's events.

    The events go into each team's dict as ``fgevents`` and
    ``bgevents``: rows of time, ranking statistic and ``var``.

    Parameters
    ----------
    injection_file : str
        MLGWSC-1 injection file.
    run : dict
        The run searched here: ``name``, ``fgpath`` and ``bgpath``
        (lists of event files).
    teams : list of dict
        Every MLGWSC-1 team, as ``run``.
    idxs : numpy.ndarray of bool
        Injections inside the analysed segments.

    Returns
    -------
    injparams : dict
        Every injection parameter, restricted to ``idxs``.
    use_chirp_distance : bool
        Whether the injection file has ``chirp_distance``.
    """
    # Read injection parameters
    logger.info(f"Reading injections from {injection_file}")

    injparams = {}
    with h5py.File(injection_file, "r") as fp:
        params = list(fp.keys())
        for param in params:
            data = fp[param][()]
            injparams[param] = data[idxs]

        use_chirp_distance = "chirp_distance" in params

    for team in [run, *teams]:
        # Read foreground events
        logger.info(f'Reading foreground events from {team["fgpath"]}')
        fg_events = []
        for fpath in team["fgpath"]:
            with h5py.File(fpath, "r") as fp:
                fg_events.append(
                    np.vstack([fp["time"], fp["stat"], np.array(fp["var"])])
                )
        team["fgevents"] = np.concatenate(fg_events, axis=-1)

        # Read background events
        logger.info(f'Reading background events from {team["bgpath"]}')
        bg_events = []
        for fpath in team["bgpath"]:
            with h5py.File(fpath, "r") as fp:
                bg_events.append(
                    np.vstack([fp["time"], fp["stat"], np.array(fp["var"])])
                )
        team["bgevents"] = np.concatenate(bg_events, axis=-1)

    return injparams, use_chirp_distance


def compare_plot_1(team_1, team_2, save_dir):
    """
    Histograms of each parameter of the injections each team finds, per
    FAR threshold both teams reach.

    Parameters
    ----------
    team_1, team_2 : dict
        Teams as filled by :func:`get_stats`: ``name``, ``params``,
        ``found_idx``, ``found_stats``, ``far_thresholds`` and the
        injection parameters.
    save_dir : str
        Directory of the plots; must not exist.

    Raises
    ------
    FileExistsError
        If ``save_dir`` exists.
    """
    # Plot 1: histograms of the injections each team finds
    os.makedirs(save_dir, exist_ok=False)
    params = team_1["params"]
    ncols = 3
    nrows = len(params) // ncols + int(len(params) % ncols or False)

    thresh_names = [
        "1-per-month",
        "1-per-week",
        "1-per-day",
        "100-per-month",
        "1000-per-month",
    ]
    # How many signals are present above given threshold?
    for n, thresh in enumerate(team_1["far_thresholds"]):
        team_1[thresh_names[n]] = team_1["found_idx"][team_1["found_stats"] > thresh]
    for n, thresh in enumerate(team_2["far_thresholds"]):
        team_2[thresh_names[n]] = team_2["found_idx"][team_2["found_stats"] > thresh]

    shared = min(len(team_1["far_thresholds"]), len(team_2["far_thresholds"]))
    for thresh_name in thresh_names[:shared]:
        fig, ax = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 6.0 * nrows))
        kwargs = dict(histtype="stepfilled", alpha=0.5)

        pidxs = list(itertools.product(range(nrows), range(ncols)))
        num_fin = 0
        for param, (i, j) in zip(params, pidxs):
            bins = np.histogram(
                np.hstack(
                    (
                        team_2[param][team_2[thresh_name]],
                        team_1[param][team_1[thresh_name]],
                    )
                ),
                bins=64,
            )[
                1
            ]  # get the bin edges
            ax[i][j].hist(
                team_2[param][team_2[thresh_name]],
                bins=bins,
                label=team_2["name"],
                color="red",
                histtype="stepfilled",
                alpha=0.5,
            )
            ax[i][j].hist(
                team_1[param][team_1[thresh_name]],
                bins=bins,
                label=team_1["name"],
                color="blue",
                **kwargs,
            )
            ax[i][j].set_title(param)
            ax[i][j].grid(True)
            ax[i][j].legend()
            num_fin += 1

        for i, j in pidxs[num_fin:]:
            ax[i][j].set_visible(False)

        plt.tight_layout()
        save_name = "compare_histogram_{}_and_{}-{}.png".format(
            team_1["name"], team_2["name"], thresh_name
        )
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        plt.close()


def compare_plot_2(team_1, team_2, save_dir):
    """
    Scatter plots of pairs of parameters of the found injections, per
    FAR threshold both teams reach.

    Blue: found by ``team_1`` only; red: by ``team_2`` only; grey: by
    both. A team's found points must be unique (asserted).

    Parameters
    ----------
    team_1, team_2 : dict
        Teams as filled by :func:`get_stats`: ``name``, ``params``,
        ``found_idx``, ``found_stats``, ``far_thresholds`` and the
        injection parameters.
    save_dir : str
        Directory of the plots; must not exist.

    Raises
    ------
    FileExistsError
        If ``save_dir`` exists.
    """
    # Plot 2: pairs of parameters, coloured by which team finds them
    # Calculate signal duration
    lf = 20.0  # Hz
    duration = (
        lambda mchirp: 5.0
        * (8.0 * np.pi * lf) ** (-8.0 / 3.0)
        * (mchirp * GM) ** (-5.0 / 3.0)
    )
    team_1["duration"] = duration(team_1["mchirp"])
    team_2["duration"] = duration(team_2["mchirp"])

    os.makedirs(save_dir, exist_ok=False)
    params = ["duration", "mchirp", "distance", "q", "chirp_distance", "snr"]
    ncols = 3
    plots = list(itertools.combinations(params, 2))
    nsubplots = len(plots)
    nrows = nsubplots // ncols + int(nsubplots % ncols or False)

    thresh_names = [
        "1-per-month",
        "1-per-week",
        "1-per-day",
        "100-per-month",
        "1000-per-month",
    ]
    # How many signals are present above given threshold?
    for n, thresh in enumerate(team_1["far_thresholds"]):
        team_1[thresh_names[n]] = team_1["found_idx"][team_1["found_stats"] > thresh]
    for n, thresh in enumerate(team_2["far_thresholds"]):
        team_2[thresh_names[n]] = team_2["found_idx"][team_2["found_stats"] > thresh]

    shared = min(len(team_1["far_thresholds"]), len(team_2["far_thresholds"]))
    for thresh_name in thresh_names[:shared]:
        fig, ax = plt.subplots(nrows, ncols, figsize=(8.0 * ncols, 5.0 * nrows))
        kwargs = {}

        pidxs = list(itertools.product(range(nrows), range(ncols)))
        num_fin = 0
        for (param_1, param_2), (i, j) in zip(plots, pidxs):
            # Team 1
            x = team_1[param_1][team_1[thresh_name]]
            y = team_1[param_2][team_1[thresh_name]]
            team1_set = set(zip(x, y))
            # The set operations need unique points
            assert len(list(team1_set)) == len(x)
            # Team 2
            x = team_2[param_1][team_2[thresh_name]]
            y = team_2[param_2][team_2[thresh_name]]
            team2_set = set(zip(x, y))
            assert len(list(team2_set)) == len(x)
            # Plots: A-B, B-A, A&B
            unique_team1 = np.array(list(team1_set - team2_set))
            unique_team2 = np.array(list(team2_set - team1_set))
            found_both = np.array(list(team1_set.intersection(team2_set)))

            kwargs.update(
                {
                    "color": "blue",
                    "s": 100.0,
                    "label": "Unique {}".format(team_1["name"]),
                    "alpha": 0.7,
                }
            )
            if len(unique_team1) != 0:
                ax[i][j].scatter(unique_team1[:, 0], unique_team1[:, 1], **kwargs)
            kwargs.update(
                {
                    "color": "red",
                    "s": 100.0,
                    "label": "Unique {}".format(team_2["name"]),
                    "alpha": 0.7,
                }
            )
            if len(unique_team2) != 0:
                ax[i][j].scatter(unique_team2[:, 0], unique_team2[:, 1], **kwargs)
            kwargs.update(
                {"color": "darkgrey", "s": 30.0, "label": "Found by Both", "alpha": 0.3}
            )
            if len(found_both) != 0:
                ax[i][j].scatter(found_both[:, 0], found_both[:, 1], **kwargs)

            ax[i][j].set_xlabel(param_1)
            ax[i][j].set_ylabel(param_2)
            ax[i][j].grid(True)
            num_fin += 1

        for i, j in pidxs[num_fin:]:
            ax[i][j].set_visible(False)

        fig.suptitle(
            "{} = Blue, {} = Red, Found by Both = Grey".format(
                team_1["name"], team_2["name"]
            )
        )
        save_name = "param_vs_param_{}_and_{}-{}.png".format(
            team_1["name"], team_2["name"], thresh_name
        )
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        plt.close()


def compare_plot_3(team_1, team_2, save_dir):
    """
    Colour strips of the ratio of injections found by ``team_1`` only to
    those found by ``team_2`` only, in bins of each parameter.

    One figure per FAR threshold both teams reach; the colour map is
    centred on a ratio of 1, red where ``team_1`` finds more.

    Parameters
    ----------
    team_1, team_2 : dict
        Teams as filled by :func:`get_stats`: ``name``, ``params``,
        ``found_idx``, ``found_stats``, ``far_thresholds`` and the
        injection parameters.
    save_dir : str
        Directory of the plots; must not exist.

    Raises
    ------
    FileExistsError
        If ``save_dir`` exists.
    """
    # Plot 3: ratio of the unique finds in each bin
    os.makedirs(save_dir, exist_ok=False)
    params = team_1["params"] + ["duration"]
    ncols = 3
    nrows = len(params) // ncols + int(len(params) % ncols or False)

    # Calculate signal duration
    lf = 20.0  # Hz
    duration = (
        lambda mchirp: 5.0
        * (8.0 * np.pi * lf) ** (-8.0 / 3.0)
        * (mchirp * GM) ** (-5.0 / 3.0)
    )
    team_1["duration"] = duration(team_1["mchirp"])
    team_2["duration"] = duration(team_2["mchirp"])

    thresh_names = [
        "1-per-month",
        "1-per-week",
        "1-per-day",
        "100-per-month",
        "1000-per-month",
    ]
    # How many signals are present above given threshold?
    for n, thresh in enumerate(team_1["far_thresholds"]):
        team_1[thresh_names[n]] = team_1["found_idx"][team_1["found_stats"] > thresh]
    for n, thresh in enumerate(team_2["far_thresholds"]):
        team_2[thresh_names[n]] = team_2["found_idx"][team_2["found_stats"] > thresh]

    shared = min(len(team_1["far_thresholds"]), len(team_2["far_thresholds"]))
    for thresh_name in thresh_names[:shared]:
        fig, ax = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.0 * nrows))

        pidxs = list(itertools.product(range(nrows), range(ncols)))
        num_fin = 0
        for param, (i, j) in zip(params, pidxs):
            team1_set = set(team_1[param][team_1[thresh_name]])
            team2_set = set(team_2[param][team_2[thresh_name]])
            # Equal values merge into one set element
            unique_team1 = np.array(list(team1_set - team2_set))
            unique_team2 = np.array(list(team2_set - team1_set))
            # Bin both teams before taking the ratio
            bins = np.linspace(
                min(team_1[param]), max(team_1[param]), 40, dtype=int, endpoint=True
            )
            team1_counts, _ = np.histogram(unique_team1, bins=bins)
            team2_counts, _ = np.histogram(unique_team2, bins=bins)
            # 1e-3 added to every count to avoid division by zero
            team1_counts = team1_counts.astype(np.float32)
            team2_counts = team2_counts.astype(np.float32)
            team1_counts += 1e-3
            team2_counts += 1e-3

            ratio = team1_counts / team2_counts

            # Making the color strip plot
            height = 25
            divnorm = colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)
            kwargs = dict(cmap="seismic", norm=divnorm)
            axes = ax[i][j]
            cstr = ax[i][j].imshow(
                np.repeat(ratio, height).reshape(-1, height).T, **kwargs
            )
            ax[i][j].set_title(param)
            ax[i][j].set_yticks([])
            divider = make_axes_locatable(axes)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(cstr, cax=cax, orientation="vertical")
            num_fin += 1

        for i, j in pidxs[num_fin:]:
            ax[i][j].set_visible(False)

        fig.suptitle(
            "Ratio = N_unique_{}/N_unique_{}".format(team_1["name"], team_2["name"])
        )
        save_name = "colour_strip_{}_and_{}-{}.png".format(
            team_1["name"], team_2["name"], thresh_name
        )
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        plt.close()


def compare_plot_4(team_1, team_2, save_dir):
    """
    Fraction of injections found against optimal SNR, both teams, per
    FAR threshold both teams reach.

    Parameters
    ----------
    team_1, team_2 : dict
        Teams as filled by :func:`get_stats`: ``name``, ``params``,
        ``found_idx``, ``found_stats``, ``far_thresholds`` and the
        injection parameters.
    save_dir : str
        Directory of the plots; must not exist.

    Raises
    ------
    FileExistsError
        If ``save_dir`` exists.
    """
    # Plot 4 (Efficiency curves made for each of the two groups)
    os.makedirs(save_dir, exist_ok=False)

    thresh_names = [
        "1-per-month",
        "1-per-week",
        "1-per-day",
        "100-per-month",
        "1000-per-month",
    ]
    # How many signals are present above given threshold?
    for n, thresh in enumerate(team_1["far_thresholds"]):
        team_1[thresh_names[n]] = team_1["found_idx"][team_1["found_stats"] > thresh]
    for n, thresh in enumerate(team_2["far_thresholds"]):
        team_2[thresh_names[n]] = team_2["found_idx"][team_2["found_stats"] > thresh]

    # (0, (3, 1, 1, 1, 1, 1)) is densely dashdotdotted in parameterised
    # form (matplotlib gallery: lines_bars_and_markers/linestyles)
    linestyles = ["solid", "dotted", "dashed", "dashdot", (0, (3, 1, 1, 1, 1, 1))]
    shared = min(len(team_1["far_thresholds"]), len(team_2["far_thresholds"]))
    for n, thresh_name in enumerate(thresh_names[:shared]):
        plt.figure(figsize=(12.0, 9.0))
        # Plotting the efficiency curve for each FAR threshold
        team1_data = team_1["snr"][team_1[thresh_name]]
        team2_data = team_2["snr"][team_2[thresh_name]]
        # Bin both teams before taking the true alarm probability
        bins = np.linspace(
            min(team_1["snr"]), max(team_1["snr"]), 20, dtype=int, endpoint=True
        )
        all_counts, _ = np.histogram(team_1["snr"], bins=bins)
        team1_counts, _ = np.histogram(team1_data, bins=bins)
        team2_counts, _ = np.histogram(team2_data, bins=bins)
        xbins = (bins[1:] + bins[:-1]) / 2.0
        kwargs = dict(
            marker="o", markersize=12, fillstyle="none", linestyle=linestyles[n]
        )
        plt.plot(
            xbins,
            team1_counts / all_counts,
            markerfacecolor="blue",
            color="blue",
            label="{}, {}".format(team_1["name"], thresh_name),
            **kwargs,
        )
        plt.plot(
            xbins,
            team2_counts / all_counts,
            markerfacecolor="red",
            color="red",
            label="{}, {}".format(team_2["name"], thresh_name),
            **kwargs,
        )

        plt.grid(True, which="both")
        plt.xlabel("Optimal SNR")
        plt.ylabel("True Alarm Probability")
        plt.title(
            "Efficiency Curves ({} and {})".format(team_1["name"], team_2["name"])
        )
        save_name = "efficiency_curves_{}_and_{}-{}.png".format(
            team_1["name"], team_2["name"], thresh_name
        )
        save_path = os.path.join(save_dir, save_name)
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def compare_groups(team_1, team_2, output_dir):
    """
    Compare the injections found by two teams.

    Histograms, parameter pairs, ratio strips and efficiency curves
    (:func:`compare_plot_1` to :func:`compare_plot_4`) under
    ``output_dir/FOUND_AND_MISSED``.

    Parameters
    ----------
    team_1, team_2 : dict
        Teams as filled by :func:`get_stats`.
    output_dir : str
        Directory of the comparison.

    Raises
    ------
    FileExistsError
        If ``FOUND_AND_MISSED`` or one of its plot folders exists.
    """

    save_dir = os.path.join(output_dir, "FOUND_AND_MISSED")
    os.makedirs(save_dir, exist_ok=False)

    compare_plot_1(team_1, team_2, os.path.join(save_dir, "histogram"))
    compare_plot_2(team_1, team_2, os.path.join(save_dir, "param_vs_param"))
    compare_plot_3(team_1, team_2, os.path.join(save_dir, "uniqueness_color_strips"))
    compare_plot_4(team_1, team_2, os.path.join(save_dir, "efficiency_curves"))


def get_stats(run, teams, injparams, chirp_distance, duration, output_dir):
    """
    Calculate the false-alarm rate and sensitivity of a search
    algorithm, for the run searched here and for every team.

    Each team gets ``found_idx``, ``found_stats``, ``far_thresholds``,
    ``params`` and the injection parameters; the run is then compared
    with every team (:func:`compare_groups`).

    Parameters
    ----------
    run : dict
        The run, as :func:`read_data` fills it. ``fgevents`` and
        ``bgevents`` hold rows of event time, ranking statistic and the
        maximum distance to an injection for a true positive.
    teams : list of dict
        Every MLGWSC-1 team, as ``run``.
    injparams : dict
        Injection parameters, at least ``tc`` and ``distance``, and
        ``mass1`` and ``mass2`` when ``chirp_distance``.
    chirp_distance : bool
        Whether the injections are drawn in chirp distance.
    duration : float or None
        The duration of the analyzed background. If None the injections
        are used to infer the duration.
    output_dir : str
        Directory of the plots.

    Returns
    -------
    dict
        The run's statistics; the most important are the keys ``far``
        and ``sensitive-distance``.
    """
    ret = {}

    injtimes = injparams["tc"]
    dist = injparams["distance"]

    # Get chirp mass from the source masses
    if chirp_distance:
        massc = mchirp(injparams["mass1"], injparams["mass2"])
    # Set duration if nothing is passed
    if duration is None:
        duration = injtimes.max() - injtimes.min()

    for team in [run, *teams]:

        logger.info("Sorting foreground event times")
        sidxs = team["fgevents"][0].argsort()
        fgevents = team["fgevents"].T[sidxs].T

        logger.info("Finding injection times closest to event times")
        idxs = find_closest_index(injtimes, fgevents[0])
        diff = np.abs(injtimes[idxs] - fgevents[0])

        # A trigger within its var of the closest injection is a true
        # positive (there may be duplicate triggers)
        logger.info("Finding true- and false-positives")
        tpbidxs = diff <= fgevents[2]
        tpidxs = np.arange(len(fgevents[0]))[tpbidxs]
        fpbidxs = diff > fgevents[2]
        fpidxs = np.arange(len(fgevents[0]))[fpbidxs]

        tpevents = fgevents.T[tpidxs].T
        fpevents = fgevents.T[fpidxs].T

        # The run's statistics
        if team is run:
            ret["fg-events"] = fgevents
            ret["found-indices"] = np.arange(len(injtimes))[idxs]
            ret["missed-indices"] = np.setdiff1d(
                np.arange(len(injtimes)), ret["found-indices"]
            )
            ret["true-positive-event-indices"] = tpidxs
            ret["false-positive-event-indices"] = fpidxs
            ret["sorting-indices"] = sidxs
            ret["true-positive-diffs"] = diff[tpidxs]
            ret["false-positive-diffs"] = diff[fpidxs]
            ret["true-positives"] = tpevents
            ret["false-positives"] = fpevents

        # Calculate foreground FAR
        logger.info("Calculating foreground FAR")
        noise_stats_fg = fpevents[1].copy()
        noise_stats_fg.sort()
        fgfar = len(noise_stats_fg) - np.arange(len(noise_stats_fg)) - 1
        fgfar = fgfar / duration
        if team is run:
            ret["fg-far"] = fgfar

        # Calculate background FAR
        logger.info("Calculating background FAR")
        noise_stats = team["bgevents"][1].copy()
        noise_stats.sort()
        far = len(noise_stats) - np.arange(len(noise_stats)) - 1
        far = far / duration
        if team is run:
            ret["far"] = far

        # Find best true-positive for each injection
        found_injections = []
        tmpsidxs = idxs.argsort()
        sorted_idxs = idxs[tmpsidxs]
        iidxs = np.full(len(idxs), False)
        for i in tqdm(
            range(len(injtimes)), ascii=True, desc="Determining found injections"
        ):
            L = np.searchsorted(sorted_idxs, i, side="left")
            if L >= len(idxs) or sorted_idxs[L] != i:
                continue
            R = np.searchsorted(sorted_idxs, i, side="right")
            # All indices that point to the same injection
            iidxs[tmpsidxs[L:R]] = True
            # Indices of the true-positives that belong to the same
            # injection
            eidxs = np.logical_and(iidxs[tmpsidxs[L:R]], tpbidxs[tmpsidxs[L:R]])
            if eidxs.any():
                found_injections.append([i, np.max(fgevents[1][tmpsidxs[L:R]][eidxs])])
            iidxs[tmpsidxs[L:R]] = False

        # Number of injections found within given testing data
        found_injections = np.array(found_injections).T

        # Calculate sensitivity
        # CARE! THIS APPLIES ONLY WHEN THE DISTRIBUTION IS CHOSEN
        # CORRECTLY
        logger.info("Calculating sensitivity")
        sidxs = found_injections[1].argsort()  # Sort found injections
        found_injections = found_injections.T[sidxs].T

        if chirp_distance:
            found_mchirp_total = massc[found_injections[0].astype(int)]
            mchirp_max = massc.max()
            if team is run:
                # Found against all injections
                found_param_plots(noise_stats, output_dir, injparams, found_injections)
                # Plotting all param vs param
                param_vs_param(output_dir, injparams, found_injections)

        max_distance = dist.max()
        vtot = (4.0 / 3.0) * np.pi * max_distance**3.0
        Ninj = len(dist)

        # Params to calculate the sensitive volume
        if chirp_distance:
            mc_norm = mchirp_max ** (5.0 / 2.0) * len(massc)
        else:
            mc_norm = Ninj

        prefactor = vtot / mc_norm
        nfound = len(found_injections[1]) - np.searchsorted(
            found_injections[1], noise_stats, side="right"
        )

        if chirp_distance:
            # Get found chirp-mass indices for given threshold
            fidxs = np.searchsorted(found_injections[1], noise_stats, side="right")
            # Plotting the network output
            network_output(found_injections, noise_stats, output_dir, team["name"])
            if team is run:
                # Parameter learning
                parameter_learning(injparams, noise_stats, found_injections, output_dir)

            found_mchirp_total = np.flip(found_mchirp_total)

            # Calculate sum(found_mchirp ** (5/2))
            # with found_mchirp = found_mchirp_total[i:]
            # and i looped over fidxs
            # Code below is a vectorized form of that
            cumsum = np.flip(np.cumsum(found_mchirp_total ** (5.0 / 2.0)))
            cumsum = np.concatenate([cumsum, np.zeros(1)])
            mc_sum = cumsum[fidxs]
            Ninj = np.sum((mchirp_max / massc) ** (5.0 / 2.0))

            cumsumsq = np.flip(np.cumsum(found_mchirp_total**5))
            cumsumsq = np.concatenate([cumsumsq, np.zeros(1)])
            sample_variance_prefactor = cumsumsq[fidxs]
            sample_variance = (
                sample_variance_prefactor / Ninj - (mc_sum / Ninj) ** 2
            )  # noqa: E127
        else:
            mc_sum = nfound
            sample_variance = nfound / Ninj - (nfound / Ninj) ** 2

        vol = prefactor * mc_sum
        vol_err = prefactor * (Ninj * sample_variance) ** 0.5
        rad = (3 * vol / (4 * np.pi)) ** (1.0 / 3.0)

        if team is run:
            ret["sensitive-volume"] = vol
            ret["sensitive-distance"] = rad
            ret["sensitive-volume-error"] = vol_err
            ret["sensitive-fraction"] = nfound / Ninj

        # Update plotting params for each group
        team["found_idx"] = found_injections[0].astype(int)
        team["found_stats"] = found_injections[1]
        # Every injection parameter, for the plots
        team["params"] = list(injparams.keys())
        team.update(injparams)
        # FAR_INDICES: false alarms per month minus 1
        team["far_thresholds"] = reachable_thresholds(noise_stats)
        team["sens_dist"] = rad
        team["sens_frac"] = nfound / Ninj

    # Compare the run with every team
    for team in teams:
        compare_groups(run, team, os.path.join(output_dir, team["name"]))

    return ret


def write_found(path: str, run: dict, teams: list, duration: float) -> None:
    """
    The run's and PyCBC's found injections, for figures across runs.

    Parameters
    ----------
    path : str
        New HDF5 file: attribute ``duration``, the injections'
        ``mchirp``, and per group ``run`` and ``PyCBC`` the
        ``found_idx``, ``found_stats`` and ``far_thresholds`` of
        ``get_stats``.
    run : dict
        The run, as filled by ``get_stats``.
    teams : list of dict
        Every MLGWSC-1 team, as ``run``; one is PyCBC.
    duration : float
        Duration of the foreground data in s.
    """
    pycbc = {team["name"]: team for team in teams}["PyCBC"]
    with h5py.File(path, "x") as fp:
        fp.attrs["duration"] = duration
        fp.create_dataset("mchirp", data=run["mchirp"])
        for name, team in (("run", run), ("PyCBC", pycbc)):
            for key in ("found_idx", "found_stats", "far_thresholds"):
                fp.create_dataset(f"{name}/{key}", data=team[key])


def evaluate(dataset_dir, run_name, fg_events, bg_events, results_dir, output_dir):
    """
    MLGWSC-1 evaluation of the run searched here, against every team.

    Writes the statistics to ``output_dir/eval.hdf``, the found
    injections to ``output_dir/found.hdf`` and the figures beside them;
    each team's comparison goes into ``output_dir/<team>`` and the
    paper's figures into ``output_dir/PAPER``.

    Parameters
    ----------
    dataset_dir : str
        Dataset 4: ``injections.hdf``, ``foreground.hdf``, ``snr.hdf``.
    run_name : str
        Label of the run.
    fg_events, bg_events : str
        The run's events on the foreground and background data.
    results_dir : str
        Every team's results, ``<team>/ds4/{fg,bg,eval}.hdf``.
    output_dir : str
        Directory of the statistics and figures.

    Raises
    ------
    RuntimeError
        If the foreground data holds no injection.
    """
    injection_file = os.path.join(dataset_dir, "injections.hdf")
    foreground_file = os.path.join(dataset_dir, "foreground.hdf")
    with h5py.File(os.path.join(dataset_dir, "snr.hdf"), "r") as fp:
        snrs = fp["snr"][()]

    # Find indices contained in foreground
    logger.info("Finding injections contained in data")
    padding_start, padding_end = MARGINS
    dur, idxs = find_injection_times(
        [foreground_file],
        injection_file,
        padding_start=padding_start,
        padding_end=padding_end,
    )
    if np.sum(idxs) == 0:
        msg = "The foreground data contains no injections! "
        msg += "Probably a too small section of data was generated. "
        msg += "Please make sure to generate at least {} seconds of data. "
        msg += "Otherwise a sensitive distance cannot be calculated."
        msg = msg.format(padding_start + padding_end + 24)
        raise RuntimeError(msg)

    # Get data from fg and bg events file
    run = {"name": run_name, "fgpath": [fg_events], "bgpath": [bg_events]}
    # Teams that published fg, bg and eval files
    teams = []
    for name in sorted(os.listdir(results_dir)):
        files = {
            kind: os.path.join(results_dir, name, "ds4", f"{kind}.hdf")
            for kind in ("fg", "bg", "eval")
        }
        if not all(os.path.isfile(path) for path in files.values()):
            logger.info(f"Leaving out {name}: not all of fg, bg, eval.hdf in ds4")
            continue
        teams.append(
            {
                "name": name,
                "fgpath": [files["fg"]],
                "bgpath": [files["bg"]],
                "eval": files["eval"],
            }
        )
    injparams, chirp_distance = read_data(injection_file, run, teams, idxs)
    # SNRs in injparams, and so in most plots
    injparams["snr"] = snrs[idxs]

    stats = get_stats(run, teams, injparams, chirp_distance, dur, output_dir)

    # Store results
    output_file = os.path.join(output_dir, "eval.hdf")
    logger.info(f"Writing output to {output_file}")
    with h5py.File(output_file, "x") as fp:
        for key, val in stats.items():
            fp.create_dataset(key, data=np.array(val))

    # Sensitivity against FAR per month; FAR is scaled by the duration
    far_scaling_factor = dur
    far = stats["far"]
    sidxs = far.argsort()
    far = far[sidxs][1:] * far_scaling_factor
    sens = stats["sensitive-distance"][sidxs][1:]

    curves = {}
    for team in teams:
        with h5py.File(team["eval"], "r") as fp:
            curves[team["name"]] = (
                np.array(fp["far"]),
                np.array(fp["sensitive-distance"]),
            )

    plt.figure(figsize=(18.0, 12.0))
    plt.title("Sensitivity Measure for Dataset 4")
    plt.plot(far * (MONTH / dur), sens, color="m", linewidth=3.0, label=run_name)
    for name, (far_team, sens_team) in curves.items():
        plt.plot(
            far_team * MONTH,
            sens_team,
            color=COLORS[name],
            linewidth=2.5,
            linestyle="dashed",
            label=name,
        )
    plt.grid(True, which="both")
    plt.xlim(1000, 1)
    plt.ylim(0, 3500)
    plt.xscale("log")
    plt.xlabel("False Alarm Rate (FAR) per month")
    plt.ylabel("Sensitive Distance [MPc]")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "sensitivity_all_teams.png"))
    plt.close()

    for name, (far_team, sens_team) in curves.items():
        plt.figure(figsize=(18.0, 12.0))
        plt.title("Sensitivity Measure for Dataset 4")
        plt.plot(far * (MONTH / dur), sens, color="m", linewidth=3.0, label=run_name)
        plt.plot(
            far_team * MONTH,
            sens_team,
            color=COLORS[name],
            linewidth=2.5,
            linestyle="dashed",
            label=name,
        )
        plt.grid(True, which="both")
        plt.xlim(1000, 1)
        plt.ylim(1500, 2100)
        plt.xscale("log")
        plt.xlabel("False Alarm Rate (FAR) per month")
        plt.ylabel("Sensitive Distance [MPc]")
        plt.legend()
        plt.savefig(os.path.join(output_dir, name, "sensitivity_compare.png"))
        plt.close()

    # Imported here: paper_plots imports this module
    from sage.benchmark.mlgwsc1.paper_plots import plot_paper

    plot_paper(run, teams, stats, dur, output_dir)
    write_found(os.path.join(output_dir, FOUND_FILE), run, teams, dur)
