BBH Parameter Distributions on GPU
=====================================

Sage draws binary black hole parameters directly on the GPU at each training iteration,
eliminating any CPU-side bottleneck. The prior is fully specified in a YAML file, and
all sampling is reproducible via a seeded :class:`torch.Generator`.

.. toctree::
   :maxdepth: 1

   distributions
   gwconfig
   sampling
