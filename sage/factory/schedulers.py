"""Learning-rate schedules and the adapter that steps them."""

from torch.optim.lr_scheduler import LambdaLR


class WarmupPolynomialDecayFlatTail(LambdaLR):
    """
    Linear warmup, polynomial decay to a floor, then a flat tail at the floor.

    Sage's production schedule, stepped every batch; its peak is the
    optimiser's learning rate. With ``floor = min_lr / peak`` and
    ``decay_steps = total_steps - warmup_steps - flat tail``, the factor on the
    peak at ``step`` is::

        warmup_start_factor + (1 - warmup_start_factor) * step / warmup_steps   step < warmup_steps
        floor + (1 - floor) * (1 - s / decay_steps) ** decay_power              s = step - warmup_steps < decay_steps
        floor                                                                   after

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser whose learning rate is the peak.
    total_steps : int
        Batches in the run.
    steps_per_epoch : int
        Batches per epoch.
    warmup_steps : int
        Batches of linear warmup.
    warmup_start_factor : float
        Fraction of the peak the warmup starts from.
    decay_power : float
        Power of the polynomial decay.
    min_lr : float
        Floor reached at the end of the decay and held in the flat tail.
    flat_tail_epochs : int
        Epochs at the floor at the end of the run.

    Raises
    ------
    ValueError
        If the warmup is longer than half the run, or the warmup and flat tail
        together are longer than the run.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        steps_per_epoch: int,
        warmup_steps: int = 20000,
        warmup_start_factor: float = 1e-3,
        decay_power: float = 2.0,
        min_lr: float = 1e-6,
        flat_tail_epochs: int = 4,
    ):
        half = max(1, total_steps // 2)
        if warmup_steps > half:
            raise ValueError(
                f"schedule.warmup_steps: {warmup_steps} is more than half the run "
                f"({total_steps} batches); use at most {half}"
            )
        flat_steps = flat_tail_epochs * steps_per_epoch
        if warmup_steps + flat_steps > total_steps:
            raise ValueError(
                f"schedule: warmup ({warmup_steps} batches) and flat tail ({flat_steps} batches) "
                f"are longer than the run ({total_steps} batches)"
            )
        floor = min_lr / optimizer.param_groups[0]["lr"]
        decay_steps = max(1, total_steps - warmup_steps - flat_steps)

        def factor(step):
            if step < warmup_steps:
                return warmup_start_factor + (1.0 - warmup_start_factor) * (step / max(1, warmup_steps))
            s = step - warmup_steps
            if s >= decay_steps:
                return floor
            return floor + (1.0 - floor) * (1.0 - s / decay_steps) ** decay_power

        super().__init__(optimizer, factor)


class ManageScheduler:
    """
    Steps a learning-rate scheduler at the unit it counts in.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler.LRScheduler
        The scheduler.
    mode : str
        ``"batch"`` steps after every batch, ``"epoch"`` once per epoch,
        ``"metric"`` once per epoch with the validation loss (e.g.
        :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`).
    """

    MODES = ("batch", "epoch", "metric")

    def __init__(self, scheduler, mode="batch"):
        if mode not in self.MODES:
            raise ValueError(
                f"unknown scheduler mode {mode!r}; expected one of {self.MODES}"
            )
        self.scheduler = scheduler
        self.mode = mode

    def batch_step(self):
        """Advance a ``"batch"``-mode scheduler at the end of a batch."""
        if self.mode == "batch":
            self.scheduler.step()

    def epoch_step(self, metric=None):
        """
        Advance an ``"epoch"`` or ``"metric"``-mode scheduler at the end of an epoch.

        Parameters
        ----------
        metric : float or None
            Validation loss, required in ``"metric"`` mode.
        """
        if self.mode == "epoch":
            self.scheduler.step()

        elif self.mode == "metric":
            if metric is None:
                raise ValueError("scheduler mode 'metric' needs a validation metric")
            self.scheduler.step(metric)
