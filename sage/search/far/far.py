"""False-alarm rates against a background, using PyCBC.

The rate is the number of background triggers at least as loud, plus
one, divided by the background time.
"""

import numpy as np
from pycbc.conversions import sec_to_year
from pycbc.events.significance import get_far


def false_alarm_rate(
    statistic: np.ndarray, background: np.ndarray, background_time_s: float
) -> np.ndarray:
    """False-alarm rate per year of each ranking statistic value.

    Parameters
    ----------
    statistic : numpy.ndarray
        Ranking statistic of the triggers to assess.
    background : numpy.ndarray
        Ranking statistic of the clustered background triggers.
    background_time_s : float
        Background time in seconds.

    Returns
    -------
    numpy.ndarray
        False-alarm rate per year.
    """
    weights = np.ones(len(background))
    _, far_hz, _ = get_far(
        background, statistic, weights, background_time_s, method="n_louder"
    )
    return far_hz / sec_to_year(1.0)
