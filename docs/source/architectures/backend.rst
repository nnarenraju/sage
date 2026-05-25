Backend: 2D ResNet with CBAM
=============================

The backend receives the stacked per-detector feature maps from the frontend and
produces a compact 512-dimensional feature vector used by the output heads.

ResNet variants
----------------

Five standard ResNet depths are available, each augmented with CBAM attention at
every residual block. The factory functions live in
:mod:`sage.architecture.backend.resnet2d_cbam`:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - ``backend_resnet_size``
     - Factory function
     - Total parameters (approx.)
   * - 18
     - ``resnet18_cbam``
     - ~11 M (backbone only)
   * - 34
     - ``resnet34_cbam``
     - ~21 M
   * - 50
     - ``resnet50_cbam``
     - ~25 M  ← default in the O3b run
   * - 101
     - ``resnet101_cbam``
     - ~44 M
   * - 152
     - ``resnet152_cbam``
     - ~60 M

CBAM attention
--------------

Each residual block in the backend is followed by a CBAM (Convolutional Block
Attention Module) that applies sequential **channel attention** and **spatial
attention**:

* **Channel attention** — squeeze the spatial dimensions to a single vector via
  global average pooling *and* global max pooling, combine them through a small MLP,
  and use the result as per-channel gating weights.  This lets the network suppress
  feature maps that are uninformative for the current input.

* **Spatial attention** — pool across the channel dimension to produce
  a 2D map, convolve it to produce per-location attention weights, and apply them
  to the feature map.  This directs the backend to focus on the time–frequency
  region containing the merger.

CBAM adds a small overhead (~2% of backbone parameters) but consistently improves
detection performance by allowing the network to attend to the merger rather than
treating all positions equally.

Feature pooling and output
---------------------------

After the final residual stage, the backend output ``(B, C, H, W)`` is pooled to a
fixed 512-dimensional vector:

.. code-block:: python

    # Inside MSCNN1D_2DResNetCBAM_Heteroscedastic.forward():
    features = self.backend(cnn_output)             # (B, C, H, W)
    features = self.flatten(self.avg_pool_1d(features))   # (B, 512)

:class:`~torch.nn.AdaptiveAvgPool1d` with output size 512 collapses the spatial
dimensions regardless of the input time length. This makes the network independent
of the compressed sequence length, which varies with the mass prior.

The 512-dimensional feature vector is shared between all output heads (ranking
statistic + one head per regression target).

Input format
-------------

The backend expects a 4D tensor ``(B, n_detectors, C, T_down)``, which is the
concatenated frontend output. For a 2-detector run with ``filters_start=32``, the
frontend produces ``128`` channels after Stage 3, so the backend input shape is
``(B, 2, 128, T_down)``.
