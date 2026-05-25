BCEWithPEsigmaLoss
===================

:class:`~sage.architecture.custom_losses.BCEWithPEsigmaLoss` is the production
loss used in the O3b run. It extends the simpler regression loss with
**heteroscedastic uncertainty estimation**: the network predicts both the mean
:math:`\mu` and the log-variance :math:`\log \sigma^2` for each parameter, and
the loss penalises the negative log-likelihood under a Gaussian model.

This is the loss paired with the
:class:`~sage.architecture.network.MSCNN1D_2DResNetCBAM_Heteroscedastic` network.

.. code-block:: python

    from sage.architecture.custom_losses import BCEWithPEsigmaLoss

    loss_function = BCEWithPEsigmaLoss(
        regression_weight=0.005,
        coupling_weight=0.005,
    )

Three-term formula
-------------------

.. math::

    \mathcal{L} = \mathcal{L}_\text{BCE}
                + \lambda_r \cdot \mathcal{L}_\text{NLL}
                + \lambda_c \cdot \mathcal{L}_\text{coupling}

**Classification loss** — standard binary cross-entropy:

.. math::

    \mathcal{L}_\text{BCE} = \text{BCE}(\hat{y},\, y)

**Heteroscedastic NLL regression loss** — Gaussian negative log-likelihood:

.. math::

    \mathcal{L}_\text{NLL}
    = \frac{1}{N_s} \sum_{i \in \text{signals}}
      p_i^2 \cdot \left[
          \frac{1}{2} \log \sigma_i^2
        + \frac{(\theta_i - \mu_i)^2}{2\sigma_i^2}
      \right]
    + \lambda_v \cdot \frac{1}{N_s} \sum_{i \in \text{signals}} p_i^2 \cdot \sigma_i^2

The second sum is a **variance regulariser** (weight ``λ_v = 1e-3``) that penalises
sigma explosion — without it the network can trivially minimise the NLL by making
:math:`\sigma` very large.

The confidence weighting uses :math:`p_i^2 = \sigma(\hat{y}_i)^2` (squared
probability), making the curriculum stricter than in ``BCEWithPEregLoss``: only
high-confidence signal predictions receive strong regression gradient.

**Coupling loss** — prevents the network from escaping the classification task by
exploiting the regression uncertainty head:

.. math::

    \mathcal{L}_\text{coupling}
    = \frac{1}{B} \sum_{i=1}^{B}
      \bar{\sigma}_i \cdot \sigma(\hat{y}_i)

where :math:`\bar{\sigma}_i = \sqrt{\text{mean}(\sigma_i^2)}` is the average
predicted uncertainty across all PE targets for sample :math:`i`. This term penalises
any combination of high predicted uncertainty *and* high predicted signal probability,
forcing the network to either be uncertain (low ranking stat) *or* confident (low
sigma). It prevents the degenerate solution where the network classifies everything as
a signal while predicting large uncertainty.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``regression_weight`` (default 1.0)
     - Weight :math:`\lambda_r` for the heteroscedastic NLL term. Very small
       values (0.005) are typical in production — the NLL gradient is much
       larger than the BCE gradient for well-separated classes.
   * - ``coupling_weight`` (default 1.0)
     - Weight :math:`\lambda_c` for the coupling term. Values in ``[0.001, 0.01]``
       are typical.
   * - ``eps`` (default 1e-6)
     - Small constant added to :math:`\exp(\log \sigma^2)` for numerical stability.

Outputs
-------

.. code-block:: python

    loss = loss_function(model_output, targets)
    # loss: torch.Tensor, shape (num_pe + 2,)
    # loss[0] = total_loss
    # loss[1] = bce_loss
    # loss[2] = reg_loss    (heteroscedastic NLL + variance regulariser)
    # loss[3] = coupling_loss

``loss_function.num_components`` equals ``len(cfg.do_point_estimate) + 2``.

The log-variance is clamped to ``[-10, 6]`` before exponentiation to prevent
numerical overflow (``exp(6) ≈ 400``) and underflow (``exp(-10) ≈ 5e-5``).

When to use
-----------

Use ``BCEWithPEsigmaLoss`` when:

* You want per-prediction uncertainty estimates for downstream ranking and
  parameter recovery diagnostics.
* You are using the heteroscedastic network
  :class:`~sage.architecture.network.MSCNN1D_2DResNetCBAM_Heteroscedastic`.
* You are doing a production run aimed at matched-filter-competitive performance.

For a quick experiment or ablation where uncertainty is not needed, prefer
:doc:`bce_reg` with its simpler tuning surface.
