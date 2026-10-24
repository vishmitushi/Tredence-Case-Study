"""Pruning sparsity schedules.

A schedule maps the current training step to a target sparsity fraction
in [0, 1].  Gradual pruning (e.g. cubic ramp) is preferred over one-shot
because it lets the network adapt its remaining weights after each round
of pruning — see Zhu & Gupta (2017), "To Prune, or Not to Prune".
"""

from __future__ import annotations


class CubicSparsitySchedule:
    """Cubic sparsity ramp from 0 to ``target_sparsity``.

    s(t) = s_final * (1 - (1 - (t - t_start) / (t_end - t_start))^3 )

    for t in [t_start, t_end], clamped to 0 before t_start and s_final
    after t_end.

    Properties:
    - Starts slow (small pruning increments early)
    - Accelerates through the middle
    - Decelerates near the end (fine adjustments)

    This matches the schedule used in the TensorFlow Model Optimization
    Toolkit and several pruning papers.

    Parameters
    ----------
    target_sparsity : float
        Final fraction of weights to prune (e.g. 0.9 for 90%).
    total_steps : int
        Total number of training steps.
    start_frac : float
        Fraction of training at which pruning begins (default 0.1 = 10%).
    end_frac : float
        Fraction of training at which pruning ends (default 0.8 = 80%).
        The remaining training is fine-tuning at final sparsity.
    prune_every : int
        Apply pruning every N steps (default 100).
    """

    def __init__(
        self,
        target_sparsity: float,
        total_steps: int,
        start_frac: float = 0.1,
        end_frac: float = 0.8,
        prune_every: int = 100,
    ):
        self.target_sparsity = target_sparsity
        self.total_steps = total_steps
        self.t_start = int(start_frac * total_steps)
        self.t_end = int(end_frac * total_steps)
        self.prune_every = prune_every

    def __call__(self, step: int) -> float:
        """Return the target sparsity at the given training step."""
        if step < self.t_start:
            return 0.0
        if step >= self.t_end:
            return self.target_sparsity

        # Fraction of the pruning window elapsed
        frac = (step - self.t_start) / (self.t_end - self.t_start)

        # Cubic ramp
        return self.target_sparsity * (1.0 - (1.0 - frac) ** 3)

    def should_prune(self, step: int) -> bool:
        """Whether pruning should be applied at this step."""
        if step < self.t_start or step > self.t_end:
            return False
        return (step - self.t_start) % self.prune_every == 0
