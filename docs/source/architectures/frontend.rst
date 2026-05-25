Frontend: Multi-Scale 1D CNN
=============================

The frontend processes each detector's compressed time-domain waveform independently
and converts it into a 2D feature map that the backend ResNet can consume.

The two key classes are :class:`~sage.architecture.frontend.mscnn1d.ConcatBlockConv5`
(a single multi-scale inception block) and
:class:`~sage.architecture.frontend.mscnn1d.ConvBlock` (three stages of those blocks
with downsampling).

ConcatBlockConv5 — the multi-scale block
------------------------------------------

Each :class:`~sage.architecture.frontend.mscnn1d.ConcatBlockConv5` applies **five
parallel 1D convolutions** at different kernel scales and concatenates their outputs
together with the identity skip connection:

.. code-block:: text

    input ──┬── Conv1d(k)    ──┐
            ├── Conv1d(2k)   ──┤
            ├── Conv1d(k//2) ──┤── cat ── Conv1d(1)  ── output
            ├── Conv1d(k//4) ──┤
            ├── Conv1d(4k)   ──┤
            └──────────────────┘  (identity)

All convolutions use ``padding='same'`` so that every branch produces the same time
length as the input. The five branches capture features at very different time scales:
``k//4`` resolves fast oscillations near merger, while ``4k`` integrates over long
inspiral trends. The final 1×1 pointwise convolution fuses the concatenated channels.

Each conv–BN–SiLU sequence uses :class:`~sage.architecture.frontend.mscnn1d.Conv1dSame`
so no manual padding calculations are needed.

ConvBlock — the per-detector frontend
---------------------------------------

:class:`~sage.architecture.frontend.mscnn1d.ConvBlock` stacks three stages of paired
:class:`~sage.architecture.frontend.mscnn1d.ConcatBlockConv5` blocks with spatial
downsampling between stages:

.. code-block:: text

    input (1, T)
        │
    Stage 1: ConcatBlock(k) + ConcatBlock(k//2+1) → MaxPool1d(8)
        │
    Stage 2: ConcatBlock(k//2+1) + ConcatBlock(k//4+1) → MaxPool1d(4)
        │
    Stage 3: ConcatBlock(k//4+1) + ConcatBlock(k//4+1)
        │
    unsqueeze → (1, C, T_down)  — 2D feature map

The ``unsqueeze`` at the end adds a height dimension so the output is a 2D image
``(batch, 1, channels, time_compressed)`` suitable for the 2D ResNet backend.

One :class:`~sage.architecture.frontend.mscnn1d.ConvBlock` is instantiated per
detector, and the outputs are concatenated along the channel dimension before being
passed to the backend:

.. code-block:: text

    H1 frontend output  (B, 1, C, T_down)
                                            → cat → (B, n_det, C, T_down)
    L1 frontend output  (B, 1, C, T_down)

.. figure:: /_static/full_frontend.png
   :align: center
   :alt: Multiscale frontend feature extractor architecture

   **(a)** A single multiscale residual block. The input :math:`x^i` is analysed
   simultaneously by :math:`N_s` parallel 1D convolutions with kernel sizes
   :math:`sk^i`, where :math:`s` is a set of constant prefactors applied to the base
   kernel size :math:`k^i`; outputs are element-wise summed (:math:`\oplus`) with a
   skip connection. **(b)** The full frontend architecture: three stages of paired
   multiscale blocks with progressive spatial downsampling, one
   :class:`~sage.architecture.frontend.mscnn1d.ConvBlock` per detector.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Effect
   * - ``filters_start``
     - 32
     - Base filter count. Stage 1 outputs 32 channels, Stage 2 64, Stage 3 128.
   * - ``kernel_start``
     - 64
     - Base kernel size ``k``. Larger kernels integrate over more time at the cost of
       compute and receptive field granularity.
   * - ``in_channels``
     - 1
     - Input channels per detector (always 1 for the compressed waveform).

Weight initialisation
----------------------

:func:`~sage.architecture.frontend.mscnn1d._initialize_frontend_weights` applies
Kaiming-normal initialisation to all ``Conv1d`` layers, ones/zeros to all
``BatchNorm1d`` layers, and small-normal initialisation to all ``Linear`` layers.
It is called automatically on construction.
