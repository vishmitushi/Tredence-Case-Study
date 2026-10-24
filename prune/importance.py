"""Importance criteria for deciding which weights to prune.

Each function takes a Linear layer and returns an importance score array of
the same shape as ``layer.weight.data``.  Higher score → more important →
less likely to be pruned.

Derivation of Taylor importance
-------------------------------
Removing weight w_{ij} (setting it to zero) changes the loss by approximately:

    Delta L  approx  (dL / dw_{ij}) * (0 - w_{ij})  =  -g_{ij} * w_{ij}

where g_{ij} = dL/dw_{ij} is the gradient.  The first-order Taylor expansion
of the loss around the current weights gives us:

    L(w - delta_w)  approx  L(w) + nabla L^T * (-delta_w)

Setting delta_w = w_{ij} * e_{ij} (removing one weight):

    |Delta L|  approx  |g_{ij} * w_{ij}|

So ``|weight * gradient|`` is a first-order approximation of the absolute
loss change from pruning that connection.  Weights with small |w * g| can
be removed with little impact on the loss.

This is more informative than magnitude alone because a large weight with
a near-zero gradient is sitting at a local optimum for that parameter and
contributes stably to the output, while a small weight with a large gradient
is being actively adjusted and may become important.
"""

from __future__ import annotations

import numpy as np


def magnitude_importance(weight_data: np.ndarray, **kwargs) -> np.ndarray:
    """Importance = |w|.  The trivial baseline."""
    return np.abs(weight_data)


def taylor_importance(
    weight_data: np.ndarray,
    grad_data: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Importance = |w * g|.

    First-order Taylor approximation of the loss change from removing
    each connection.  See module docstring for derivation.
    """
    return np.abs(weight_data * grad_data)


def movement_importance(
    weight_data: np.ndarray,
    initial_weight_data: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Importance = |w - w_0|.

    Weights that have moved far from their initialisation are assumed
    to have been shaped by training signal and are therefore important.
    This criterion (Sanh et al., 2020 — "Movement Pruning") does not
    require gradients and is complementary to magnitude.
    """
    return np.abs(weight_data - initial_weight_data)


# Registry for easy lookup
IMPORTANCE_CRITERIA = {
    "magnitude": magnitude_importance,
    "taylor": taylor_importance,
    "movement": movement_importance,
}
