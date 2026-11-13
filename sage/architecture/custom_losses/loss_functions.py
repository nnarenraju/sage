"""Classification and parameter-estimation losses.

Each loss returns its named components (``component_names``): ``total``,
``classification``, ``regression_<target>`` per target and, for
:class:`BCE_PE_Sigma`, ``coupling``. Class labels: 0 noise, 1 signal,
2 non-astrophysical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def width_to_sigma(raw: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """
    Standard deviation from a raw width output.

    Softplus plus ``sigma_min``, clamped to ``[sigma_min, sigma_max]``.

    Parameters
    ----------
    raw : torch.Tensor
        Raw width outputs.
    sigma_min, sigma_max : float
        Bounds of the standard deviation.

    Returns
    -------
    torch.Tensor
        Standard deviations, shape of ``raw``.
    """
    return torch.clamp(F.softplus(raw) + sigma_min, sigma_min, sigma_max)


def classification(ranking_statistic: torch.Tensor, non_astrophysical_logit: torch.Tensor, label: torch.Tensor):
    """
    Classification loss and the signal probability of each sample.

    With the non-astrophysical class the logits are ``(0, r, a)``, noise
    the reference, and the loss is their multinomial cross-entropy;
    without it, BCE on ``r``.

    Parameters
    ----------
    ranking_statistic : torch.Tensor, shape ``(B, 1)``
        Signal vs noise logit ``r``.
    non_astrophysical_logit : torch.Tensor
        Non-astrophysical vs noise logit ``a``, shape ``(B, 1)``, or
        ``(B, 0)`` without the class.
    label : torch.Tensor, shape ``(B,)``
        Class label.

    Returns
    -------
    loss : torch.Tensor
        Mean over the batch.
    signal_probability : torch.Tensor, shape ``(B,)``
        Softmax signal probability, or ``sigmoid(r)`` without the class.
    """
    r = ranking_statistic.reshape(-1)
    if non_astrophysical_logit.shape[1] == 0:
        return F.binary_cross_entropy_with_logits(r, label.to(r.dtype)), torch.sigmoid(r)
    logits = torch.stack([torch.zeros_like(r), r, non_astrophysical_logit.reshape(-1).to(r.dtype)], dim=1)
    return F.cross_entropy(logits, label.long()), torch.softmax(logits, dim=1)[:, 1]


def component_names(targets: list, coupling: bool) -> list:
    """
    Names of a loss's components, in output order.

    Parameters
    ----------
    targets : list of str
        Estimated parameters.
    coupling : bool
        Whether the loss has a coupling term.

    Returns
    -------
    list of str
        Component names.
    """
    names = ["total", "classification", *[f"regression_{target}" for target in targets]]
    return names + ["coupling"] if coupling else names


class EstimateLoss(nn.Module):
    """
    Classification plus a regression of the estimates (no widths).

    Loss: classification + regression_weight * :meth:`regression` summed
    over signal samples / n_signal; with ``confidence_weighting`` each
    sample is weighted by the detached signal probability.

    Parameters
    ----------
    targets : list of str
        Estimated parameters, in head order.
    regression_weight : float
        Weight of the regression against the classification.
    confidence_weighting : bool
        Weight each sample's regression by its signal probability.
    """

    uncertainty = False

    def __init__(self, targets: list, regression_weight: float, confidence_weighting: bool):
        super().__init__()
        self.regression_weight = regression_weight
        self.confidence_weighting = confidence_weighting
        self.component_names = component_names(targets, coupling=False)
        self.num_components = len(self.component_names)

    def regression(self, estimates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Elementwise regression loss; defined by each subclass.

        Parameters
        ----------
        estimates : torch.Tensor, shape ``(B, P)``
            Point estimates.
        targets : torch.Tensor, shape ``(B, P)``
            Scaled targets.

        Returns
        -------
        torch.Tensor, shape ``(B, P)``
            Loss per sample and parameter.

        Raises
        ------
        NotImplementedError
            Always, on this base class.
        """
        raise NotImplementedError

    def forward(self, outputs: tuple, targets: torch.Tensor) -> torch.Tensor:
        """
        Loss components of a batch.

        Parameters
        ----------
        outputs : tuple
            ``(ranking_statistic, estimates, non_astrophysical_logit)``,
            shapes ``(B, 1)``, ``(B, P)`` and ``(B, 1)`` or ``(B, 0)``.
        targets : torch.Tensor, shape ``(B, P + 1)``
            Scaled estimate targets, then the class label.

        Returns
        -------
        torch.Tensor, shape ``(P + 2,)``
            ``[total, classification, regression_<target>...]``.
        """
        ranking_stat, point_estimates, non_astrophysical_logit = outputs
        bce_loss, signal_probability = classification(ranking_stat, non_astrophysical_logit, targets[:, -1])

        signal_mask = (targets[:, -1] == 1).to(point_estimates.dtype).unsqueeze(1)
        reg = self.regression(point_estimates, targets[:, :-1]) * signal_mask
        if self.confidence_weighting:
            reg = reg * signal_probability.detach().unsqueeze(1)
        num_signal = signal_mask.sum().clamp_min(1)

        total_loss = bce_loss + self.regression_weight * (reg.sum() / num_signal)
        return torch.stack([total_loss, bce_loss, *(reg.sum(dim=0) / num_signal)])


class BCE_PE_Huber(EstimateLoss):
    """
    Classification plus smooth-L1 (Huber) regression of the estimates.

    Parameters
    ----------
    targets : list of str
        Estimated parameters, in head order; from ``signal.targets``.
    regression_weight : float
        Weight of the regression against the classification.
    confidence_weighting : bool
        Weight each sample's regression by its signal probability.
    """

    def __init__(self, targets: list, regression_weight: float = 1.0, confidence_weighting: bool = True):
        super().__init__(targets, regression_weight, confidence_weighting)

    def regression(self, estimates, targets):
        """
        Elementwise smooth-L1 loss.

        Parameters
        ----------
        estimates, targets : torch.Tensor, shape ``(B, P)``
            Point estimates and scaled targets.

        Returns
        -------
        torch.Tensor, shape ``(B, P)``
            Loss per sample and parameter.
        """
        return F.smooth_l1_loss(estimates, targets, reduction="none")


class BCE_PE_MSE(EstimateLoss):
    """
    Classification plus mean-squared-error regression of the estimates.

    Parameters
    ----------
    targets : list of str
        Estimated parameters, in head order; from ``signal.targets``.
    regression_weight : float
        Weight of the regression against the classification.
    confidence_weighting : bool
        Weight each sample's regression by its signal probability.
    """

    def __init__(self, targets: list, regression_weight: float = 1.0, confidence_weighting: bool = False):
        super().__init__(targets, regression_weight, confidence_weighting)

    def regression(self, estimates, targets):
        """
        Elementwise squared error.

        Parameters
        ----------
        estimates, targets : torch.Tensor, shape ``(B, P)``
            Point estimates and scaled targets.

        Returns
        -------
        torch.Tensor, shape ``(B, P)``
            Loss per sample and parameter.
        """
        return F.mse_loss(estimates, targets, reduction="none")


class BCE_PE_Sigma(nn.Module):
    """
    Classification, beta-NLL regression and coupling; production loss.

    Gaussian NLL on signal samples times ``stop_grad(sigma^(2 beta))``
    (beta-NLL), sigma from :func:`width_to_sigma`;
    coupling: batch mean of mean sigma times signal probability.

    Parameters
    ----------
    targets : list of str
        Estimated parameters, in head order; from ``signal.targets``.
    regression_weight : float
        Weight of the regression against the classification.
    coupling_weight : float
        Weight of the coupling between width and signal probability.
    beta : float
        beta-NLL exponent; 0 gives the plain Gaussian NLL.
    sigma_min, sigma_max : float
        Bounds of the standard deviation.
    confidence_weighting : bool
        Weight each sample's regression by its signal probability.
    """

    uncertainty = True

    def __init__(
        self,
        targets: list,
        regression_weight: float = 0.005,
        coupling_weight: float = 0.005,
        beta: float = 0.5,
        sigma_min: float = 1e-3,
        sigma_max: float = 10.0,
        confidence_weighting: bool = True,
    ):
        super().__init__()
        self.regression_weight = regression_weight
        self.coupling_weight = coupling_weight
        self.beta = float(beta)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.confidence_weighting = confidence_weighting
        self.component_names = component_names(targets, coupling=True)
        self.num_components = len(self.component_names)

    def forward(self, outputs: tuple, targets: torch.Tensor) -> torch.Tensor:
        """
        Loss components of a batch.

        Parameters
        ----------
        outputs : tuple
            ``(ranking_statistic, estimates, non_astrophysical_logit)``,
            shapes ``(B, 1)``, ``(B, 2P)`` (all means, then all raw
            widths) and ``(B, 1)`` or ``(B, 0)``.
        targets : torch.Tensor, shape ``(B, P + 1)``
            Scaled estimate targets, then the class label.

        Returns
        -------
        torch.Tensor, shape ``(P + 3,)``
            ``[total, classification, regression_<target>...,
            coupling]``.
        """
        ranking_stat, point_estimates, non_astrophysical_logit = outputs
        bce_loss, signal_probability = classification(ranking_stat, non_astrophysical_logit, targets[:, -1])

        pe_targets = targets[:, :-1]
        signal_mask = (targets[:, -1] == 1).to(targets.dtype).unsqueeze(1)
        num_pe = pe_targets.shape[1]

        # float32: 1/sigma^2 can exceed the fp16 range under autocast
        mu = point_estimates[:, :num_pe].float()
        sigma = width_to_sigma(point_estimates[:, num_pe:].float(), self.sigma_min, self.sigma_max)
        var = sigma * sigma
        nll = 0.5 * (mu - pe_targets.float()) ** 2 / var + torch.log(sigma)
        # beta-NLL damps the 1/sigma^2 gradient, optimum unchanged
        if self.beta > 0.0:
            nll = nll * (var**self.beta).detach()

        nll = nll * signal_mask.float()
        if self.confidence_weighting:
            nll = nll * signal_probability.float().detach().unsqueeze(1)
        num_signal = signal_mask.float().sum().clamp_min(1.0)

        coupling_loss = (sigma.mean(dim=1) * signal_probability.float()).mean()

        bce_loss = bce_loss.float()
        total_loss = (
            bce_loss
            + (self.regression_weight * (nll.sum() / num_signal))
            + (self.coupling_weight * coupling_loss)
        )
        return torch.stack([total_loss, bce_loss, *(nll.sum(dim=0) / num_signal), coupling_loss])
