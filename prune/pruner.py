"""Pruner: orchestrates gradual structured pruning with regrowth support.

This module ties together the importance criterion, sparsity schedule,
mask management, and optimizer-state hygiene into a single ``Pruner``
object that is called once per training step.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict

from nn.linear import Linear
from nn.optim import Adam, SGDMomentum
from prune.importance import IMPORTANCE_CRITERIA
from prune.schedule import CubicSparsitySchedule


class Pruner:
    """Gradual unstructured pruner with optional regrowth.

    Parameters
    ----------
    layers : list of Linear
        The layers whose weights will be pruned.
    optimizer : Adam or SGDMomentum
        The optimizer — needed so we can reset state for mask changes.
    schedule : CubicSparsitySchedule
        Controls the target sparsity over training.
    criterion : str
        Name of importance criterion: ``'magnitude'``, ``'taylor'``, or
        ``'movement'``.
    regrowth : bool
        If ``True``, allow pruned connections to be revived when their
        estimated importance exceeds that of the least-important active
        connection.  Default ``True``.
    regrowth_fraction : float
        When regrowing, allow at most this fraction of newly pruned
        connections to come back.  Default 0.1 (10%).

    Mask semantics
    --------------
    ``layer.mask[i,j] == 1`` → connection is active (contributes to forward).
    ``layer.mask[i,j] == 0`` → connection is pruned (exactly zero in forward,
    zero gradient flows back, optimizer state is reset on transition).

    The pruner enforces that after masking, ``layer.weight.data[mask==0]`` is
    set to exactly zero.  This is *not* strictly necessary for correctness
    (the mask already blocks the forward path), but it makes sparsity auditing
    unambiguous and avoids "phantom weights" that appear sparse but are not.
    """

    def __init__(
        self,
        layers: List[Linear],
        optimizer,
        schedule: CubicSparsitySchedule,
        criterion: str = "taylor",
        regrowth: bool = True,
        regrowth_fraction: float = 0.1,
    ):
        self.layers = layers
        self.optimizer = optimizer
        self.schedule = schedule
        self.criterion_name = criterion
        self.criterion_fn = IMPORTANCE_CRITERIA[criterion]
        self.regrowth = regrowth
        self.regrowth_fraction = regrowth_fraction

        # Store initial weights for movement-based importance
        self.initial_weights: Dict[int, np.ndarray] = {}
        for layer in layers:
            self.initial_weights[id(layer)] = layer.weight.data.copy()

    def step(self, global_step: int) -> None:
        """Potentially prune/regrow at the current training step.

        Should be called once per training step, after ``backward()`` and
        before ``optimizer.step()``, so that weight gradients are available
        for Taylor importance but weights have not yet been updated.
        """
        if not self.schedule.should_prune(global_step):
            return

        target_sparsity = self.schedule(global_step)
        if target_sparsity <= 0:
            return

        # --- Compute importance scores across ALL layers (global pruning) ---
        all_scores = []
        all_layers = []
        all_indices = []

        for layer in self.layers:
            kwargs = {
                "weight_data": layer.weight.data,
                "grad_data": layer.weight.grad if layer.weight.grad is not None else np.zeros_like(layer.weight.data),
                "initial_weight_data": self.initial_weights[id(layer)],
            }
            scores = self.criterion_fn(**kwargs)

            # For already-pruned weights: if no regrowth, give them -inf
            # importance so they stay pruned.  If regrowth is enabled,
            # compute their importance from the *unmasked* gradient so we
            # can decide whether to revive them.
            if not self.regrowth:
                scores[layer.mask == 0] = -np.inf

            all_scores.append(scores.ravel())
            all_layers.append(layer)
            all_indices.append(scores.shape)

        flat_scores = np.concatenate(all_scores)

        # --- Determine the global pruning threshold ---
        total_weights = flat_scores.size
        n_to_prune = int(np.round(target_sparsity * total_weights))
        n_to_prune = min(n_to_prune, total_weights - 1)  # keep at least 1

        if n_to_prune <= 0:
            return

        # The threshold: weights with score below this are pruned
        # Use partition for O(n) instead of full sort
        threshold = np.partition(flat_scores, n_to_prune)[n_to_prune]

        # --- Apply masks ---
        offset = 0
        for layer in self.layers:
            n_elem = layer.weight.data.size
            layer_scores = flat_scores[offset : offset + n_elem].reshape(
                layer.weight.data.shape
            )
            offset += n_elem

            old_mask = layer.mask.copy()
            new_mask = (layer_scores >= threshold).astype(np.float64)

            # Handle regrowth budget: if regrowth is enabled but we are
            # regrowing too many, keep only the top regrowth_fraction
            if self.regrowth:
                revived = (old_mask == 0) & (new_mask == 1)
                n_revived = revived.sum()
                max_regrowth = int(
                    self.regrowth_fraction * (old_mask == 0).sum()
                )
                if n_revived > max_regrowth and n_revived > 0:
                    # Keep only the top-scoring revived connections
                    revived_scores = layer_scores[revived]
                    cutoff = np.partition(revived_scores, -max_regrowth)[
                        -max_regrowth
                    ] if max_regrowth > 0 else np.inf
                    # Un-revive connections below the cutoff
                    too_weak = revived & (layer_scores < cutoff)
                    new_mask[too_weak] = 0.0

            layer.mask = new_mask

            # Zero out pruned weight values (honest zeros)
            layer.weight.data[new_mask == 0] = 0.0

            # Reset optimizer state for mask changes
            self.optimizer.reset_state_for_mask_changes(
                layer.weight, old_mask, new_mask
            )

    def get_total_sparsity(self) -> float:
        """Return the current aggregate sparsity across all layers."""
        total = 0
        pruned = 0
        for layer in self.layers:
            total += layer.mask.size
            pruned += (layer.mask == 0).sum()
        return pruned / total if total > 0 else 0.0

    def get_active_params(self) -> int:
        """Total active (non-pruned) weight parameters."""
        return sum(layer.active_params for layer in self.layers)

    def get_total_params(self) -> int:
        """Total weight parameters (including pruned)."""
        return sum(layer.total_params for layer in self.layers)
