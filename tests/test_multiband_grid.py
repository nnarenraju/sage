"""sage.data.waveform.multiband_grid against LALSimulation 6.1.0."""

import numpy as np
import pytest

from sage.core.constants import GM
from sage.data.waveform import multiband_grid as mb

# 22-mode arguments (zero spins, 1024 Hz, 1/16 Hz) and the sub-grids XLALSimIMRPhenomXMultibandingGrid
# returns for them in LAL's binary, as (intdfRatio, Length, xStart, deltax).
MODE22 = (0.021497332614512636, 0.11528197698673585)  # fMECO, fRING + 2 fDAMP
SPACING = (1.8333333333333333, 0.1798507193740161, 0.0007341551132675033, 0.0009240704180787725)
CASES = {
    "pre-grid, inspiral": (
        (0.0002758274930679109, *MODE22, 0.014122367645077039, 8.619609158372216e-07),
        [(1, 1137, 0.0002758274930679109, 8.619609158372216e-07), (1, 671, 0.0012558770543748318, 8.619609158372216e-07),
         (1, 979, 0.0018342528289016075, 8.619609158372216e-07), (1, 1429, 0.0026781125655062476, 8.619609158372216e-07),
         (1, 2086, 0.003909854714237637, 8.619609158372216e-07), (1, 3044, 0.005707905184674081, 8.619609158372216e-07),
         (3, 1482, 0.008333438134314258, 2.585882747511665e-06), (7, 928, 0.012169164209789895, 6.033726410860551e-06),
         (15, 633, 0.017775358006395188, 1.2929413737558324e-05), (30, 3455, 0.025972606316007164, 2.5858827475116648e-05)],
    ),
    "inspiral, merger, ringdown": (
        (0.0059105891371695205, *MODE22, 0.3, 1.847059105365475e-05),
        [(1, 43, 0.0059105891371695205, 1.847059105365475e-05), (1, 168, 0.0067048245524766745, 1.847059105365475e-05),
         (3, 83, 0.009844825031597983, 5.541177316096425e-05), (6, 62, 0.014499413977118979, 0.0001108235463219285),
         (12, 46, 0.021481297395400473, 0.000221647092643857), (24, 190, 0.031898710749661746, 0.000443294185287714),
         (50, 200, 0.11660484132172243, 0.0009235295526827375)],
    ),
    "no inspiral, merger capped": (
        (0.0295529456858476, *MODE22, 0.3, 3.078431842275792e-05),
        [(18, 156, 0.0295529456858476, 0.0005541177316096425), (30, 200, 0.11636472363802491, 0.0009235295526827375)],
    ),
}


@pytest.mark.parametrize("arguments, expected", CASES.values(), ids=CASES)
def test_grid_matches_lal(arguments, expected):
    grids = mb.multibanding_grid(*arguments[:5], *SPACING)
    assert [(g["int_df_ratio"], g["length"], g["x_start"], g["deltax"]) for g in grids] == expected


def test_coarse_frequencies():
    grids = [{**mb.grid_comp(0.0, 1.0, 0.25), "int_df_ratio": 5}]
    used, points, freqs = mb.coarse_frequencies(grids, 0.1, 1.05, 10.0)  # ends within one step of mf_max
    assert used[0]["length"] == 6 and points.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    assert np.array_equal(freqs, points * (1.0 / (10.0 * GM)))
    with pytest.raises(ValueError, match="0 multibanding sub-grids"):
        mb.coarse_frequencies([], 0.1, 1.05, 10.0)


def test_int4_overflow():
    with pytest.raises(ValueError, match="overflows LAL's INT4 conversion"):
        mb.multibanding_grid(0.013985028791545114, 4.274830115023252, 0.1307953941733732, 0.19570734231051892,
                             1.8676842519656006e-07, 11 / 6, 35.31845994636517, 5.135628861561841e-05, 0.0036534469958082196)
