"""Named networks: a frontend per detector feeding a shared backend.

``MSCNN1D`` frontends have no attention; ``MSCNN1Datt`` frontends have channel
and temporal attention; ``catt`` marks cross-attention between detectors.

Every network declares ``uncertainty`` (estimate heads also give a raw width)
and ``separable`` (detectors meet only at the backend input, so the search can
compute each detector's frontend once and reuse it across time slides).
"""

from .mc_dropout import enable_mc_dropout, mc_predict
from .mscnn1d_att_catt_resnet2d_cbam import MSCNN1Datt_catt_2DResNetCBAM
from .mscnn1d_att_resnet2d_cbam import MSCNN1Datt_2DResNetCBAM
from .mscnn1d_resnet2d_cbam import MSCNN1D_2DResNetCBAM

__all__ = [
    "MSCNN1D_2DResNetCBAM",
    "MSCNN1Datt_2DResNetCBAM",
    "MSCNN1Datt_catt_2DResNetCBAM",
    "enable_mc_dropout",
    "mc_predict",
]
