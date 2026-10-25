"""Start-up checks that span several settings.

Each raises ``ValueError`` with the offending value and, where one
exists, the nearest valid value. Decimal settings are compared exactly.
"""

import math
from fractions import Fraction


def exact(value: float) -> Fraction:
    """
    The decimal a setting was written as, as an exact fraction.

    Parameters
    ----------
    value : float
        Setting value.

    Returns
    -------
    Fraction
        ``repr(value)`` read exactly, so ``1.1`` is ``11/10``.
    """
    return Fraction(repr(value))


def check_whole_samples(name: str, length_in_s: float, sample_rate_in_hz: float) -> None:
    """
    Check that a length is a whole number of samples.

    Parameters
    ----------
    name : str
        Setting name, for the message.
    length_in_s : float
        Length in seconds.
    sample_rate_in_hz : float
        Sample rate in Hz.

    Raises
    ------
    ValueError
        If the length is not a whole number of samples.
    """
    rate = exact(sample_rate_in_hz)
    samples = exact(length_in_s) * rate
    if samples.denominator != 1:
        nearest = float(round(samples) / rate)
        raise ValueError(
            f"{name} {length_in_s} s is {float(samples)} samples at "
            f"{sample_rate_in_hz} Hz; nearest whole length is {nearest} s"
        )


def check_on_grid(name: str, frequency_in_hz: float, padded_length_in_s: float) -> None:
    """
    Check that a frequency lies on the grid of the padded window.

    Parameters
    ----------
    name : str
        Setting name, for the message.
    frequency_in_hz : float
        Frequency in Hz.
    padded_length_in_s : float
        Padded window length; the grid spacing is its inverse.

    Raises
    ------
    ValueError
        If the frequency is not a whole number of grid steps.
    """
    delta_f = 1 / exact(padded_length_in_s)
    steps = exact(frequency_in_hz) / delta_f
    if steps.denominator != 1:
        nearest = float(round(steps) * delta_f)
        raise ValueError(
            f"{name} {frequency_in_hz} Hz is not on the {delta_f} Hz grid; "
            f"nearest is {nearest}"
        )


def check_power_of_two(name: str, value: float) -> None:
    """
    Check that a setting is a whole power of two.

    Parameters
    ----------
    name : str
        Setting name, for the message.
    value : float
        Setting value.

    Raises
    ------
    ValueError
        If the value is not 1, 2, 4, 8, ...
    """
    if value >= 1 and float(value).is_integer() and int(value) & (int(value) - 1) == 0:
        return
    message = f"{name} {value} is not a power of two"
    if math.isfinite(value) and value > 0:
        lower = 1 << max(0, math.floor(math.log2(value)))
        nearest = lower if value - lower <= 2 * lower - value else 2 * lower
        message += f"; nearest is {nearest}"
    raise ValueError(message)


def check_whole_batches(name: str, samples: int, batch_size: int) -> None:
    """
    Check that a sample count is a whole number of batches.

    Raises
    ------
    ValueError
        If not; the message gives the nearest whole number of batches.
    """
    if samples % batch_size:
        nearest = max(1, round(samples / batch_size)) * batch_size
        raise ValueError(
            f"{name} {samples} is not a whole number of batches of {batch_size}; nearest is {nearest}"
        )


def check_whole_signals(signal_fraction: float, batch_size: int) -> None:
    """
    Check that the signal fraction gives a whole number of signals per batch.

    Raises
    ------
    ValueError
        If the fraction is outside (0, 1] or gives a fractional count; the
        message gives the nearest fraction that does not.
    """
    if not 0 < signal_fraction <= 1:
        raise ValueError(f"train.signal_fraction {signal_fraction} must be in (0, 1]")
    signals = exact(signal_fraction) * batch_size
    if signals.denominator != 1:
        nearest = max(1, round(signals))
        raise ValueError(
            f"train.signal_fraction {signal_fraction} of a batch of {batch_size} is "
            f"{float(signals)} signals; nearest whole is {nearest / batch_size} ({nearest} per batch)"
        )


def check_unique_detectors(detectors: list) -> None:
    """
    Check that no detector is listed twice.

    Parameters
    ----------
    detectors : list of str
        Detector names.

    Raises
    ------
    ValueError
        If a detector appears more than once.
    """
    repeated = sorted({d for d in detectors if detectors.count(d) > 1})
    if repeated:
        raise ValueError(f"detectors listed more than once: {', '.join(repeated)}")


def check_tc_window(tc_min: float, tc_max: float, sample_length_in_s: float) -> None:
    """
    Check that the tc prior lies inside the analysed window.

    tc is measured from the end of the leading padding, so the analysed
    window is ``[0, sample_length_in_s]``.

    Parameters
    ----------
    tc_min, tc_max : float
        Bounds of the tc prior in seconds.
    sample_length_in_s : float
        Length of the analysed window in seconds.

    Raises
    ------
    ValueError
        If either bound lies outside the analysed window.
    """
    if tc_min < 0 or tc_max > sample_length_in_s:
        raise ValueError(
            f"tc prior [{tc_min}, {tc_max}] s is outside the analysed window "
            f"[0, {sample_length_in_s}] s"
        )
