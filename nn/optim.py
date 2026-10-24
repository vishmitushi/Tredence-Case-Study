"""Optimisers: Adam and SGD with momentum, implemented from scratch.

The critical design choice here is how the optimiser interacts with pruning
masks.  See the docstring of :meth:`Adam.step` for a detailed explanation.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict
from engine.tensor import Tensor


class Adam:
    """Adam optimiser (Kingma & Ba, 2015) with correct pruning-mask handling.

    Parameters
    ----------
    params : list of Tensor
        Learnable parameters.
    lr : float
        Learning rate (default 1e-3).
    betas : tuple of float
        Coefficients for running averages of gradient (beta1) and
        its square (beta2).  Default (0.9, 0.999).
    eps : float
        Term added to denominator for numerical stability (default 1e-8).
    weight_decay : float
        L2 regularisation coefficient (default 0).

    Notes on mask handling
    ----------------------
    When weights are pruned (mask=0), the forward pass already produces zero
    gradient for those entries (because the mask is applied *inside* the
    computation graph — see ``Linear.forward``).  This means Adam's first-
    and second-moment estimates for pruned entries naturally decay toward zero
    at rates beta1 and beta2.

    **The problem with revival**: if a weight is later *revived* (mask flipped
    from 0 to 1), its moment estimates ``m`` and ``v`` have been decaying for
    many steps and are near zero, but the bias-correction denominators
    ``1 - beta^t`` use the *global* step counter ``t``.  This creates a
    mismatch: the bias-corrected moments will be tiny, causing Adam to take an
    inappropriately large step because the effective learning rate is
    ``lr / (sqrt(v_hat) + eps)`` ≈ ``lr / eps`` — potentially thousands of
    times the intended scale.

    **Our solution**: after every pruning step, the ``Pruner`` calls
    ``optimizer.reset_state_for_mask_changes(param, old_mask, new_mask)``
    which zeroes out ``m`` and ``v`` for any entry whose mask changed (either
    pruned→active or active→pruned) **and** resets a per-element step counter.
    This way Adam treats a revived weight as if it were newly initialised,
    with no stale momentum.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # global step counter

        # Per-parameter state
        self.state: Dict[int, dict] = {}
        for p in params:
            pid = id(p)
            self.state[pid] = {
                "m": np.zeros_like(p.data),          # first moment
                "v": np.zeros_like(p.data),          # second moment
                "step": np.zeros_like(p.data, dtype=np.int64),  # per-element step
            }

    def step(self) -> None:
        """Perform a single optimisation step.

        Gradients must already be computed (via ``backward()``).
        """
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue

            pid = id(p)
            state = self.state[pid]
            m, v, elem_step = state["m"], state["v"], state["step"]

            g = p.grad.copy()

            # Optional L2 weight decay (decoupled, like AdamW)
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data

            # Identify which elements actually received a non-zero gradient
            # (this naturally covers masked weights whose gradient is 0).
            active = (g != 0.0) | (m != 0.0) | (v != 0.0)
            elem_step[active] += 1

            # Update moments
            m[:] = self.beta1 * m + (1.0 - self.beta1) * g
            v[:] = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)

            # Bias-corrected moments using per-element step count
            # This is critical for correctness after weight revival.
            safe_step = np.maximum(elem_step, 1)  # avoid division by zero
            m_hat = m / (1.0 - self.beta1 ** safe_step)
            v_hat = v / (1.0 - self.beta2 ** safe_step)

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Reset all parameter gradients."""
        for p in self.params:
            p.zero_grad()

    def reset_state_for_mask_changes(
        self,
        param: Tensor,
        old_mask: np.ndarray,
        new_mask: np.ndarray,
    ) -> None:
        """Reset Adam state for entries whose mask status changed.

        This must be called by the pruner AFTER updating the mask but BEFORE
        the next ``step()`` call.  It prevents stale moment estimates from
        corrupting training for revived or newly pruned connections.

        Parameters
        ----------
        param : Tensor
            The weight tensor whose mask changed.
        old_mask, new_mask : ndarray
            Binary masks (same shape as ``param.data``).
        """
        pid = id(param)
        if pid not in self.state:
            return

        changed = old_mask != new_mask  # entries that were pruned or revived
        if not changed.any():
            return

        state = self.state[pid]
        state["m"][changed] = 0.0
        state["v"][changed] = 0.0
        state["step"][changed] = 0


class SGDMomentum:
    """SGD with momentum and correct pruning-mask handling.

    Parameters
    ----------
    params : list of Tensor
        Learnable parameters.
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient (default 0.9).
    weight_decay : float
        L2 regularisation coefficient (default 0).
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.velocity: Dict[int, np.ndarray] = {}
        for p in params:
            self.velocity[id(p)] = np.zeros_like(p.data)

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            pid = id(p)
            v = self.velocity[pid]
            g = p.grad.copy()
            if self.weight_decay > 0:
                g = g + self.weight_decay * p.data

            v[:] = self.momentum * v + g
            p.data -= self.lr * v

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def reset_state_for_mask_changes(
        self,
        param: Tensor,
        old_mask: np.ndarray,
        new_mask: np.ndarray,
    ) -> None:
        """Reset velocity for entries whose mask status changed."""
        pid = id(param)
        if pid not in self.velocity:
            return
        changed = old_mask != new_mask
        if changed.any():
            self.velocity[pid][changed] = 0.0
