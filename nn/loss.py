"""Loss functions."""

from engine.tensor import Tensor, softmax_cross_entropy as _sce
import numpy as np


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Numerically stable softmax cross-entropy loss.

    See :func:`engine.tensor.softmax_cross_entropy` for details.
    """
    return _sce(logits, targets)
