Benchmarks — MLGWSC-1 Results
==============================

All results below are evaluated on the D4 testing dataset from the Machine Learning
Gravitational-Wave Search Challenge
(`Schäfer et al. 2023 <https://doi.org/10.1103/PhysRevD.107.023021>`__,
`arXiv:2209.11146 <https://arxiv.org/abs/2209.11146>`__), which provides
standardised noise and injection sets for direct comparison between pipelines.

Sensitive distance vs. false alarm rate
-----------------------------------------

.. figure:: /_static/sensitive_distance_pycbc_sage.png
   :align: center
   :alt: Sensitive distance vs FAR for Sage and PyCBC

   Sensitive distance as a function of FAR for PyCBC and the Sage — Broad run.
   The solid violet line is the mean sensitive distance over 3 independent runs with
   the Broad configuration; error bars indicate the min/max range across those runs.

.. figure:: /_static/sensitive_distance_ML_comparison.png
   :align: center
   :alt: Sensitive distance vs FAR across all ML pipelines

   Sensitive distance as a function of FAR per month for Sage — Broad, other ML
   pipelines, and PyCBC on the D4 testing dataset. AresGW results from
   `Nousi et al. (2023) <https://doi.org/10.1103/PhysRevD.108.024022>`__
   (`arXiv:2211.01520 <https://arxiv.org/abs/2211.01520>`__);
   Aframe and TPI FSU Jena (2024) results via private communication; all other
   results from `Schäfer et al. (2023) <https://doi.org/10.1103/PhysRevD.107.023021>`__.
   Aframe results are marked with an asterisk (✱) as their pipeline is not optimised
   for the D4 mass distribution in MLGWSC-1.

ROC curves
-----------

.. figure:: /_static/roc_pycbc_sage.png
   :align: center
   :alt: ROC curves for Sage and PyCBC

   True alarm probability as a function of false alarm probability (ROC curves) for
   PyCBC and the Sage — Broad run that achieved the highest sensitive distance on the
   D4 testing dataset.

Per-parameter detection comparisons
--------------------------------------

The following hexbin plots show 1D and 2D histograms of detected signals at a false
alarm rate of one per month, comparing Sage against each competing pipeline. The
diagonal subplots contain 1D histograms of detected signals for individual parameters.
Off-diagonal subplots are hexagonally binned, coloured by :math:`N^\mathrm{F}_\mathrm{Sage} - N^\mathrm{F}_\mathrm{other}`:
black indicates no difference, progressively brighter **red** indicates more signals
detected by Sage, and progressively brighter **blue** indicates more signals detected
by the comparison pipeline.

.. figure:: /_static/compare_hexbinned_sage_pycbc.png
   :align: center
   :alt: Sage vs PyCBC per-parameter detection comparison

   1D and 2D detection histograms at FAR = 1/month comparing Sage — Broad and PyCBC.
   Parameters were chosen where learning bias is expected to be visible.
   Red regions indicate parameter combinations where Sage detects more signals;
   blue regions where PyCBC detects more.

.. figure:: /_static/compare_hexbinned_sage_aresgw.png
   :align: center
   :alt: Sage vs AresGW per-parameter detection comparison

   1D and 2D detection histograms at FAR = 1/month comparing Sage — Broad and AresGW
   (`Nousi et al. 2023 <https://doi.org/10.1103/PhysRevD.108.024022>`__).
   Red regions indicate parameter combinations where Sage detects more signals;
   blue regions where AresGW detects more.

.. figure:: /_static/compare_hexbinned_sage_aframe.png
   :align: center
   :alt: Sage vs Aframe per-parameter detection comparison

   1D and 2D detection histograms at FAR = 1/month comparing Sage — Broad and Aframe.
   Red regions indicate parameter combinations where Sage detects more signals;
   blue regions where Aframe detects more.
