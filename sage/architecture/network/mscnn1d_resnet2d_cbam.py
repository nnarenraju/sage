"""Multi-scale 1D CNN frontend (no attention) per detector + 2D ResNet-CBAM backend."""

import torch
from torch import nn

from ..frontend.mscnn1d import ConvBlock, _initialize_frontend_weights
from .common import FrontendTap, initialise_heads, make_heads, norm_layer, normalise, resnet_backend, run_heads


class MSCNN1D_2DResNetCBAM(FrontendTap, nn.Module):
    """
    Plain multi-scale frontend per detector, 2D ResNet-CBAM backend.

    Outputs the ranking statistic and one estimate per parameter (no uncertainty).

    Parameters
    ----------
    num_detectors : int
        Detector channels; from ``data.detectors``.
    num_estimates : int
        Estimated parameters; from ``signal.targets``.
    frontend_filters, frontend_kernel : int
        Base filter count and kernel size of the frontend.
    backend_resnet_size : int
        ResNet depth: 18, 34, 50, 101 or 152.
    norm_type : str
        Input normalisation: ``instancenorm``, ``groupnorm`` or ``batchnorm``.
    dropout : float
        Backend dropout rate.
    use_blurpool : bool
        Anti-aliased downsampling in the backend.
    use_resnet_cd : bool
        ResNet-C deep stem and ResNet-D average-pool downsampling in the backend.
    """

    uncertainty = False
    separable = True

    def __init__(
        self,
        num_detectors: int,
        num_estimates: int,
        frontend_filters: int = 32,
        frontend_kernel: int = 64,
        backend_resnet_size: int = 50,
        norm_type: str = "instancenorm",
        dropout: float = 0.0,
        use_blurpool: bool = True,
        use_resnet_cd: bool = True,
    ):
        super().__init__()
        self.num_detectors = num_detectors
        self.norm = norm_layer(norm_type, num_detectors)
        self.frontend = nn.ModuleList(
            [ConvBlock(frontend_filters, frontend_kernel) for _ in range(num_detectors)]
        )
        self.backend = resnet_backend(
            backend_resnet_size, in_channels=num_detectors, dropout=dropout,
            use_blurpool=use_blurpool, use_resnet_cd=use_resnet_cd,
        )
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(512)
        self.flatten = nn.Flatten(start_dim=1)
        self.get_ranking_statistic, self.point_estimate_layers = make_heads(
            num_estimates, self.uncertainty
        )
        initialise_heads(self.get_ranking_statistic, self.point_estimate_layers)
        for detector in self.frontend:
            _initialize_frontend_weights(detector)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, L)``

        Returns
        -------
        ranking_statistic : torch.Tensor, shape ``(B, 1)``
        point_estimates : torch.Tensor, shape ``(B, P)``
        """
        x = normalise(self.norm, x)
        cnn_outputs = [detector(x[:, i : i + 1]) for i, detector in enumerate(self.frontend)]
        self.tap(cnn_outputs)
        features = self.backend(torch.cat(cnn_outputs, dim=1))
        features = self.flatten(self.avg_pool_1d(features))
        return run_heads(features, self.get_ranking_statistic, self.point_estimate_layers)
