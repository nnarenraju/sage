Network Architectures
======================

Sage uses a two-stage neural network: a **frontend** that extracts per-detector
temporal features from the compressed time-domain input, and a **backend** that
fuses the multi-detector feature maps and produces a compact feature vector.

Two complete networks are provided, differing only in the uncertainty output of the
regression head:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Class
     - Regression head
   * - :class:`~sage.architecture.network.MSCNN1D_2DResNetCBAM`
     - Point estimates only (used with :class:`~sage.architecture.custom_losses.BCEWithPEregLoss`)
   * - :class:`~sage.architecture.network.MSCNN1D_2DResNetCBAM_Heteroscedastic`
     - Mean + log-variance (used with :class:`~sage.architecture.custom_losses.BCEWithPEsigmaLoss`)

.. toctree::
   :maxdepth: 1

   frontend
   backend
   networks
