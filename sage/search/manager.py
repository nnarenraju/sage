"""Run or resume one search: zero-lag, then the roll background.

Called by ``sage search``; every step resumes after its last batch.
"""

from pathlib import Path

from pycbc.conversions import sec_to_year

from sage.config.machine import gwosc_label, observing_run_dir, search_strain_file
from sage.core.progress import step
from sage.data.noise.data_quality import read_bad_time
from sage.search.dataprep.windows import analysis_windows, load_segments
from sage.search.model.load import load_search_network
from sage.search.removal.vetoes import clear_windows
from sage.search.settings import read_settings, search_dir
from sage.search.timeslides.roll import plan_rolls, score_roll_file, step_windows
from sage.search.zerolag.zerolag import score_zerolag

STEPS = (
    "Load the network and its preprocessing from the checkpoint",
    "Find the zero-lag windows: one every stride where every detector has a whole "
    "window",
    "Score every zero-lag window",
    "Plan the background: roll the window list until the live time is reached",
    "Score the background, one file of about a year at a time",
)


def run_search(run_dir: Path, name: str, data_dir: Path) -> Path:
    """
    Run or resume a search: every zero-lag window, then every roll.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    name : str
        Search name, a directory under ``<run_dir>/search``.
    data_dir : Path
        Directory from the machine file.

    Returns
    -------
    Path
        The search directory.

    Raises
    ------
    SettingsChanged
        If ``search.yaml`` changed after scoring started.
    """
    directory = search_dir(run_dir, name)
    settings = read_settings(directory)
    run, label = settings["observing_run"], gwosc_label(settings["observing_run"])
    stride, batch_size = settings["stride_in_samples"], settings["batch_size"]
    device = settings["device"]
    checkpoint = Path(run_dir) / "checkpoints" / f"{settings['checkpoint']}.pt"
    release = observing_run_dir(data_dir, run, "search")
    with step(1, STEPS):
        network = load_search_network(checkpoint, device)
    detectors = network["detectors"]
    attrs = {
        "settings": settings,
        "checkpoint": str(checkpoint.resolve()),
        "release_dir": str(release.resolve()),
    }

    with step(2, STEPS):
        segments = {
            detector: load_segments(release, label, detector) for detector in detectors
        }
        windows = analysis_windows(segments, network["window_samples"], stride)
        print(f"  {len(windows['gps']):,} windows")
    with step(3, STEPS):
        score_zerolag(
            network,
            release,
            label,
            windows,
            directory / "zerolag",
            stride,
            batch_size,
            device,
            attrs,
        )

    with step(4, STEPS):
        vetoes = {
            detector: read_bad_time(search_strain_file(data_dir, run, detector))
            for detector in detectors
        }
        background = settings["background"]
        rate = network["sample_rate"]
        files = plan_rolls(
            clear_windows(network, windows, vetoes),
            detectors,
            step_windows(background["step_in_s"], stride, rate),
            background["step_in_s"],
            background["live_time_in_years"],
            background["years_per_file"],
            stride,
            rate,
        )
        n_rolls = sum(len(plan["shifts"]) for plan in files)
        live_s = sum(sum(plan["live_time_s"]) for plan in files)
        print(
            f"  {n_rolls} rolls in {len(files)} files, "
            f"{sec_to_year(live_s):.3f} years of live time"
        )

    with step(5, STEPS):
        first_roll = 1
        for index, plan in enumerate(files, 1):
            name = f"year_{index:02d}"
            last_roll = first_roll + len(plan["shifts"]) - 1
            years = sec_to_year(sum(plan["live_time_s"]))
            print(f"  {name}: rolls {first_roll}-{last_roll}, {years:.3f} years")
            score_roll_file(
                network,
                release,
                label,
                windows,
                directory / "background",
                name,
                plan["shifts"],
                stride,
                batch_size,
                device,
                {**attrs, **plan},
            )
            first_roll = last_roll + 1
    return directory
