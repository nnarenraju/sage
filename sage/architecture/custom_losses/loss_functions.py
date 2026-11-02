"""Classification + parameter-estimation losses.

Each loss returns a vector of named components (``component_names``): ``total``
(the one back-propagated), ``classification``, one ``regression_<target>`` per
target and, for :class:`BCE_PE_Sigma`, ``coupling``. The total is built from the
regression summed over every target and sample, as before the per-target split.

The class label is 0 for noise, 1 for a signal and 2 for a non-astrophysical pair.
Without the non-astrophysical class the classification is BCE on the ranking
statistic; with it, the multinomial cross-entropy of :func:`classification`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def width_to_sigma(raw: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """
    Standard deviation from a raw width output: softplus plus ``sigma_min``,
    clamped to ``[sigma_min, sigma_max]``.

    No ``exp``, so a momentarily large output cannot collapse the width and blow
    up ``(mu - y)^2 / sigma^2``.
    """
    return torch.clamp(F.softplus(raw) + sigma_min, sigma_min, sigma_max)


def classification(ranking_statistic: torch.Tensor, non_astrophysical_logit: torch.Tensor, label: torch.Tensor):
    """
    Classification loss and the signal probability of each sample.

    With the non-astrophysical class the logits of noise, signal and non-astrophysical are
    ``(0, r, a)``, noise the reference category (baseline-category logits), so the ranking
    statistic ``r`` stays the signal vs noise log odds; the loss is their multinomial
    cross-entropy (multinomial density-ratio estimation, Srivastava et al. 2023, TMLR).
    Without it, ``a`` is empty and the loss is BCE on ``r``.

    Parameters
    ----------
    ranking_statistic : torch.Tensor, shape ``(B, 1)``
    non_astrophysical_logit : torch.Tensor, shape ``(B, 1)`` or ``(B, 0)``
    label : torch.Tensor, shape ``(B,)``
        Class label.

    Returns
    -------
    loss : torch.Tensor
        Mean over the batch.
    signal_probability : torch.Tensor, shape ``(B,)``
    """
    r = ranking_statistic.reshape(-1)
    if non_astrophysical_logit.shape[1] == 0:
        return F.binary_cross_entropy_with_logits(r, label.to(r.dtype)), torch.sigmoid(r)
    logits = torch.stack([torch.zeros_like(r), r, non_astrophysical_logit.reshape(-1).to(r.dtype)], dim=1)
    return F.cross_entropy(logits, label.long()), torch.softmax(logits, dim=1)[:, 1]


def component_names(targets: list, coupling: bool) -> list:
    """Names of a loss's components, in output order."""
    names = ["total", "classification", *[f"regression_{target}" for target in targets]]
    return names + ["coupling"] if coupling else names


class EstimateLoss(nn.Module):
    """
    BCE on the ranking statistic + a regression of the estimates (no widths).

        L = BCE + regression_weight * sum(w * r) / n_signal

    ``r`` is the elementwise regression of :meth:`regression`, kept on signal
    samples only; with ``confidence_weighting`` each sample is weighted by the
    detached signal probability. BCE is the multinomial cross-entropy with the
    non-astrophysical class (:func:`classification`).
    """

    uncertainty = False

    def __init__(self, targets: list, regression_weight: float, confidence_weighting: bool):
        super().__init__()
        self.regression_weight = regression_weight
        self.confidence_weighting = confidence_weighting
        self.component_names = component_names(targets, coupling=False)
        self.num_components = len(self.component_names)

    def regression(self, estimates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, outputs: tuple, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        outputs : tuple
            ``(ranking_statistic, estimates, non_astrophysical_logit)``, shapes ``(B, 1)``,
            ``(B, P)`` and ``(B, 1)`` or ``(B, 0)``.
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
    BCE + smooth-L1 (Huber) regression of the estimates.

    Parameters
    ----------
    targets : list of str
        Estimated parameters, in head order; from ``signal.targets``.
    regression_weight : float
        Weight of the regression against BCE.
    confidence_weighting : bool
        Weight each sample's regression by the network's detection confidence.
    """

    def __init__(self, targets: list, regression_weight: float = 1.0, confidence_weighting: bool = True):
        super().__init__(targets, regression_weight, confidence_weighting)

    def regression(self, estimates, targets):
        return F.smooth_l1_loss(estimates, targets, reduction="none")


class BCE_PE_MSE(EstimateLoss):
    """
    BCE + mean-squared-error regression of the estimates.

    Parameters
    ----------
    targets : list of str
        Estimated parameters, in head order; from ``signal.targets``.
    regression_weight : float
        Weight of the regression against BCE.
    confidence_weighting : bool
        Weight each sample's regression by the network's detection confidence.
    """

    def __init__(self, targets: list, regression_weight: float = 1.0, confidence_weighting: bool = False):
        super().__init__(targets, regression_weight, confidence_weighting)

    def regression(self, estimates, targets):
        return F.mse_loss(estimates, targets, reduction="none")


class BCE_PE_Sigma(nn.Module):
    """
    BCE + heteroscedastic beta-NLL regression + coupling. The production loss.

        L = BCE + regression_weight * sum(w * nll) / n_signal
                + coupling_weight * mean(mean_sigma * signal_probability)

    ``nll = 0.5 (mu - y)^2 / sigma^2 + log sigma``, scaled by
    ``stop_grad(sigma^(2 beta))`` (beta-NLL, Seitzer et al. 2022), on signal
    samples only; with ``confidence_weighting`` each sample is weighted by the
    detached signal probability. ``sigma`` comes from :func:`width_to_sigma`. BCE
    is the multinomial cross-entropy with the non-astrophysical class
    (:func:`classification`).

    Parameters
    ----------
    targets : list of str
        Estimated parameters, in head order; from ``signal.targets``.
    regression_weight : float
        Weight of the regression against BCE.
    coupling_weight : float
        Weight of the coupling between width and detection confidence.
    beta : float
        beta-NLL exponent; 0 gives the plain Gaussian NLL.
    sigma_min, sigma_max : float
        Bounds of the standard deviation.
    confidence_weighting : bool
        Weight each sample's regression by the network's detection confidence.
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
        Parameters
        ----------
        outputs : tuple
            ``(ranking_statistic, estimates, non_astrophysical_logit)``, shapes ``(B, 1)``,
            ``(B, 2P)`` (all means, then all raw widths) and ``(B, 1)`` or ``(B, 0)``.
        targets : torch.Tensor, shape ``(B, P + 1)``
            Scaled estimate targets, then the class label.

        Returns
        -------
        torch.Tensor, shape ``(P + 3,)``
            ``[total, classification, regression_<target>..., coupling]``.
        """
        ranking_stat, point_estimates, non_astrophysical_logit = outputs
        bce_loss, signal_probability = classification(ranking_stat, non_astrophysical_logit, targets[:, -1])

        pe_targets = targets[:, :-1]
        signal_mask = (targets[:, -1] == 1).to(targets.dtype).unsqueeze(1)
        num_pe = pe_targets.shape[1]

        # float32 throughout: 1/sigma^2 can exceed fp16 range under autocast.
        mu = point_estimates[:, :num_pe].float()
        sigma = width_to_sigma(point_estimates[:, num_pe:].float(), self.sigma_min, self.sigma_max)
        var = sigma * sigma
        nll = 0.5 * (mu - pe_targets.float()) ** 2 / var + torch.log(sigma)
        # beta-NLL tempers the ~1/sigma^2 gradient that lets a briefly
        # overconfident head explode, without moving the optimum.
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
