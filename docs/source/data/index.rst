Data Handling
=============

Before a run can train, Sage needs the detectors' strain and the amplitude
spectral densities (ASDs) built from it. Everything it downloads and builds goes into
one data directory, shared by every run on the machine. These pages show how
to set it up and fill it.

:doc:`data_dir`
   Choosing the data directory with ``sage setup``, and seeing what it holds.

:doc:`download`
   Downloading the strain a run needs with ``sage download``.

:doc:`asds`
   Building the ASDs a run needs with ``sage build-asds``.

Downloading strain to search an observing run is on :doc:`../searching`.

.. toctree::
   :hidden:

   data_dir
   download
   asds
