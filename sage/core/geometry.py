"""The analysis window geometry, computed once from the ``data`` settings."""

from dataclasses import dataclass

from sage.core.checks import check_whole_samples, exact


@dataclass(frozen=True)
class Window:
    """
    Sizes of the padded analysis window, shared by every component.

    Parameters
    ----------
    sample_rate_in_hz : float
        Sample rate.
    sample_length_in_s, sample_length_in_samples : float, int
        Analysed window, without padding.
    padding_length_in_s, padding_in_samples : float, int
        Padding on each side.
    padded_length_in_s, padded_length_in_samples : float, int
        Analysed window plus both paddings.
    delta_f_in_hz : float
        Frequency spacing of the padded window.
    n_frequency_bins : int
        Bins of the real FFT of the padded window.
    """

    sample_rate_in_hz: float
    sample_length_in_s: float
    sample_length_in_samples: int
    padding_length_in_s: float
    padding_in_samples: int
    padded_length_in_s: float
    padded_length_in_samples: int
    delta_f_in_hz: float
    n_frequency_bins: int


def window_from_data(data: dict) -> Window:
    """
    Build the window geometry from the ``data`` settings.

    Parameters
    ----------
    data : dict
        Checked ``data`` section of a run file.

    Returns
    -------
    Window
        Window geometry.

    Raises
    ------
    ValueError
        If a length is not a whole number of samples.
    """
    rate = data["sample_rate_in_hz"]
    check_whole_samples("sample_length_in_s", data["sample_length_in_s"], rate)
    check_whole_samples("padding_length_in_s", data["padding_length_in_s"], rate)

    sample_length = exact(data["sample_length_in_s"])
    padding = exact(data["padding_length_in_s"])
    padded = sample_length + 2 * padding
    padded_samples = int(padded * exact(rate))

    return Window(
        sample_rate_in_hz=rate,
        sample_length_in_s=float(sample_length),
        sample_length_in_samples=int(sample_length * exact(rate)),
        padding_length_in_s=float(padding),
        padding_in_samples=int(padding * exact(rate)),
        padded_length_in_s=float(padded),
        padded_length_in_samples=padded_samples,
        delta_f_in_hz=float(1 / padded),
        n_frequency_bins=padded_samples // 2 + 1,
    )
