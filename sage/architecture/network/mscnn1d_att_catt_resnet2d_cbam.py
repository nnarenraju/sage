"""Attention frontend per detector, cross-attention between two detectors, 2D ResNet-CBAM backend."""

import torch
from torch import nn

from ..frontend.mscnn1d_cbam import ConvBlock, _initialize_frontend_weights
from ..zoo.cross_attention import AxialCrossAttention2D
from .common import FrontendTap, initialise_heads, make_heads, norm_layer, normalise, resnet_backend, run_heads


class MSCNN1Datt_catt_2DResNetCBAM(FrontendTap, nn.Module):
    """
    Multi-scale frontend with attention per detector, gated axial cross-attention
    between the two detectors, 2D ResNet-CBAM backend on ``[f1, f2, |f1 - f2|]``.

    Exactly two detectors. Outputs the ranking statistic and one estimate per
    parameter (no uncertainty).

    Parameters
    ----------
    num_detectors : int
        Detector channels; from ``data.detectors``; must be 2.
    num_estimates : int
        Estimated parameters; from ``signal.targets``.
    frontend_filters, frontend_kernel : int
        Base filter count and kernel size of the frontend.
    cross_attention_heads : int
        Attention heads of the cross-attention.
    backend_resnet_size : int
        ResNet depth: 18, 34, 50, 101 or 152.
    norm_type : str
        Input normalisation: ``instancenorm``, ``groupnorm`` or ``batchnorm``.
    use_blurpool : bool
        Anti-aliased downsampling in the frontend and backend.
    use_resnet_cd : bool
        ResNet-C deep stem and ResNet-D average-pool downsampling in the backend.

    Raises
    ------
    ValueError
        If ``num_detectors`` is not 2.
    """

    uncertainty = False
    separable = False

    def __init__(
        self,
        num_detectors: int,
        num_estimates: int,
        frontend_filters: int = 32,
        frontend_kernel: int = 64,
        cross_attention_heads: int = 4,
        backend_resnet_size: int = 50,
        norm_type: str = "instancenorm",
        use_blurpool: bool = True,
        use_resnet_cd: bool = True,
    ):
        super().__init__()
        if num_detectors != 2:
            raise ValueError(
                f"MSCNN1Datt_catt_2DResNetCBAM cross-attends exactly two detectors, got {num_detectors}"
            )
        self.num_detectors = num_detectors
        self.norm = norm_layer(norm_type, num_detectors)
        self.frontend = nn.ModuleList(
            [
                ConvBlock(frontend_filters, frontend_kernel, use_blurpool=use_blurpool)
                for _ in range(num_detectors)
            ]
        )

        channels = 128
        self.cross_attention = AxialCrossAttention2D(channels, cross_attention_heads)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.proj_gate = nn.Conv2d(1, channels, kernel_size=1)
        self.reduce = nn.Conv2d(channels, 1, kernel_size=1)

        self.backend = resnet_backend(
            backend_resnet_size, in_channels=3, use_blurpool=use_blurpool, use_resnet_cd=use_resnet_cd
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
        x : torch.Tensor, shape ``(B, 2, L)``

        Returns
        -------
        ranking_statistic : torch.Tensor, shape ``(B, 1)``
        point_estimates : torch.Tensor, shape ``(B, P)``
        """
        x = normalise(self.norm, x)
        cnn_outputs = [detector(x[:, i : i + 1]) for i, detector in enumerate(self.frontend)]
        self.tap(cnn_outputs)

        f1_attn, f2_attn = self.cross_attention(*cnn_outputs)
        # Gate from both detectors' projected features; the residual keeps each
        # detector's own features if the attention contributes nothing.
        proj_f1 = self.proj_gate(cnn_outputs[0])
        proj_f2 = self.proj_gate(cnn_outputs[1])
        g = self.gate(torch.cat([proj_f1, proj_f2], dim=1))
        f1 = self.reduce(cnn_outputs[0] + g * f1_attn)
        f2 = self.reduce(cnn_outputs[1] + g * f2_attn)

        features = self.backend(torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1))
        features = self.flatten(self.avg_pool_1d(features))
        return run_heads(features, self.get_ranking_statistic, self.point_estimate_layers)
