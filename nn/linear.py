"""Linear (fully-connected) layer with optional pruning mask support."""

from __future__ import annotations

import numpy as np
from engine.tensor import Tensor
from nn.module import Module


class Linear(Module):
    """Fully-connected layer: y = x @ W + b.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool
        If ``True`` (default), add a learnable bias.
    init : str
        Weight initialisation: ``'he'`` (default, good for ReLU),
        ``'xavier'`` (good for tanh/sigmoid), or ``'lecun'``.

    Attributes
    ----------
    weight : Tensor, shape (in_features, out_features)
        Learnable weight matrix.
    bias : Tensor, shape (out_features,)
        Learnable bias vector (if enabled).
    mask : ndarray, shape (in_features, out_features)
        Binary mask — 1 means active, 0 means pruned.  Applied in
        ``forward()`` so that pruned connections contribute exactly zero.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "he",
    ):
        self.in_features = in_features
        self.out_features = out_features

        # --- weight initialisation ---
        if init == "he":
            # He et al. 2015 — suited for ReLU networks
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        elif init == "lecun":
            std = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"Unknown init: {init!r}")

        w = np.random.randn(in_features, out_features) * std
        self.weight = Tensor(w, requires_grad=True)

        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

        # Pruning mask: all ones → all connections active
        self.mask = np.ones((in_features, out_features), dtype=np.float64)

    def forward(self, x: Tensor) -> Tensor:
        """Compute y = x @ (W * mask) + b.

        The mask is applied as a *constant* element-wise multiply before the
        matmul.  Because ``W * mask`` participates in the computation graph,
        the autodiff engine naturally produces ``dW = (dL/d(W*mask)) * mask``
        — meaning pruned (mask=0) weights receive exactly zero gradient.
        This is the correct behaviour: the gradient of a pruned weight is
        zero because it does not influence the loss.
        """
        # Apply the mask — this is a Tensor * ndarray operation, which we
        # handle via element-wise multiply where the mask is treated as a
        # constant (not requiring grad).
        masked_weight = Tensor(
            self.weight.data * self.mask,
            requires_grad=self.weight.requires_grad,
            _parents=(self.weight,),
            _op="mask_apply",
        )

        # Backward for the masking operation: dW = d(W*mask) * mask
        _mask = self.mask  # capture in closure

        def _mask_backward():
            if self.weight.requires_grad:
                self.weight.grad += masked_weight.grad * _mask

        masked_weight._backward_fn = _mask_backward

        out = x.matmul(masked_weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    @property
    def sparsity(self) -> float:
        """Fraction of weights that are pruned (mask == 0)."""
        return 1.0 - self.mask.mean()

    @property
    def active_params(self) -> int:
        """Number of non-pruned weight parameters."""
        count = int(self.mask.sum())
        if self.bias is not None:
            count += self.out_features
        return count

    @property
    def total_params(self) -> int:
        """Total weight parameters (including pruned)."""
        count = self.in_features * self.out_features
        if self.bias is not None:
            count += self.out_features
        return count
