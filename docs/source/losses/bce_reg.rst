BCEWithPEregLoss
=================

:class:`~sage.architecture.custom_losses.BCEWithPEregLoss` is the simpler of the
two losses. It combines binary cross-entropy for detection with a Smooth-L1
(Huber) regression term for parameter estimation. The regression term does not
model prediction uncertainty — each target gets a single point estimate.

.. code-block:: python

    from sage.architecture.custom_losses import BCEWithPEregLoss

    loss_function = BCEWithPEregLoss(regression_weight=0.3)

Formula
-------

.. math::

    \mathcal{L} = \mathcal{L}_\text{BCE}(\hat{y}, y)
                + \lambda_r \cdot \mathcal{L}_\text{reg}

where the regression term is:

.. math::

    \mathcal{L}_\text{reg}
    = \frac{1}{N_s} \sum_{i \in \text{signals}}
      p_i \cdot \text{SmoothL1}(\hat{\theta}_i,\, \theta_i)

* :math:`N_s` — number of signal samples in the batch.
* :math:`p_i = \sigma(\hat{y}_i)` — the network's current predicted signal
  probability (detached from the graph).  This acts as a **confidence curriculum**:
  the regression update focuses on samples the network is already reasonably
  confident are signals, which prevents the regression head from being trained on
  noise samples that are misclassified early in training.
* The Smooth-L1 loss transitions from squared error below ``|error| < 1`` to
  absolute error above, making it more robust to outliers than plain MSE.

The regression term is computed **only for signal samples** (``class_target == 1``).
Noise samples contribute to the BCE term but are excluded from the regression.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``regression_weight`` (default 1.0)
     - Relative weight of :math:`\mathcal{L}_\text{reg}` vs. BCE. Values in
       ``[0.1, 1.0]`` are typical; lower values de-emphasise parameter estimation
       during early training when the detector is still learning to classify.

Outputs
-------

.. code-block:: python

    loss = loss_function(model_output, targets)
    # loss: torch.Tensor, shape (num_pe + 1,)
    # loss[0] = total_loss
    # loss[1] = bce_loss
    # loss[2] = reg_loss  (one entry per PE target in cfg.do_point_estimate)

``loss_function.num_components`` equals ``len(cfg.do_point_estimate) + 1``.

When to use
-----------

Use ``BCEWithPEregLoss`` when:

* You want a simpler loss with fewer hyperparameters to tune.
* Your regression targets are well-conditioned and do not require uncertainty
  quantification.
* You are using the non-heteroscedastic network
  :class:`~sage.architecture.network.MSCNN1D_2DResNetCBAM`.
