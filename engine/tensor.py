"""
Reverse-mode automatic differentiation engine built on NumPy.

This module provides a Tensor class that tracks computation graphs and supports
reverse-mode (backpropagation) automatic differentiation.  Every differentiable
operation records its parents and a local backward function; calling
`backward()` on a scalar loss walks the graph in reverse-topological order and
accumulates gradients via the chain rule.

Key design decisions
--------------------
* **float64 everywhere** — needed to pass finite-difference gradient checks at
  tight tolerances (rtol=1e-5, atol=1e-6).
* **Gradient is `None` until `backward()` is called** — avoids silent
  accumulation bugs; each `backward()` call freshly initialises grad buffers
  to zero before accumulation.
* **`_unbroadcast` helper** — whenever NumPy broadcasting expands a tensor
  during the forward pass (e.g. (N,D)+(D,)), the backward function sums over
  the broadcast axes so that `grad.shape == data.shape`.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Helper: undo broadcasting in the backward pass
# ---------------------------------------------------------------------------

def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum *grad* over axes that were broadcast to reach *shape*.

    Examples
    --------
    >>> _unbroadcast(np.ones((3,4)), (4,)).shape
    (4,)
    >>> _unbroadcast(np.ones((3,4)), (1,4)).shape
    (1, 4)
    """
    if shape == ():
        return grad.sum()

    # 1. Leading dimensions that were implicitly added (e.g. (4,) -> (3,4))
    ndim_added = grad.ndim - len(shape)
    axes_to_sum = list(range(ndim_added))

    # 2. Existing dimensions that were 1 and got broadcast (e.g. (1,4) -> (3,4))
    for i, s in enumerate(shape):
        if s == 1:
            axes_to_sum.append(i + ndim_added)

    if axes_to_sum:
        grad = grad.sum(axis=tuple(axes_to_sum)).reshape(shape)

    return grad


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class Tensor:
    """An N-dimensional array with automatic differentiation support.

    Parameters
    ----------
    data : array-like
        The underlying numerical data (will be cast to float64).
    requires_grad : bool
        Whether gradients should be tracked for this tensor.
    """

    __slots__ = (
        "data", "grad", "requires_grad", "shape",
        "_backward_fn", "_parents", "_op",
    )

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _parents: Tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64) if data.dtype != np.float64 else data.copy()
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self.shape = self.data.shape
        self._backward_fn = lambda: None
        self._parents = _parents
        self._op = _op

    # ---- helpers ----------------------------------------------------------

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, req_grad={self.requires_grad}, op={self._op!r})"

    # ---- backward ---------------------------------------------------------

    def backward(self) -> None:
        """Run reverse-mode AD starting from *self* (must be a scalar)."""
        if self.data.size != 1:
            raise RuntimeError("backward() can only be called on a scalar tensor")

        # Build reverse-topological ordering via DFS
        topo: list[Tensor] = []
        visited: set[int] = set()

        def _build_topo(v: Tensor):
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for parent in v._parents:
                    _build_topo(parent)
                topo.append(v)

        _build_topo(self)

        # Initialise all grads to zero
        for v in topo:
            if v.requires_grad:
                v.grad = np.zeros_like(v.data)

        # Seed gradient of loss = 1
        self.grad = np.ones_like(self.data)

        # Walk in reverse
        for v in reversed(topo):
            v._backward_fn()

    def zero_grad(self) -> None:
        """Reset gradient to None."""
        self.grad = None

    # ---- element-wise arithmetic ------------------------------------------

    def __add__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="add",
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.shape)  # type: ignore[operator]
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.shape)

        out._backward_fn = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        return self * (-1.0)

    def __sub__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="sub",
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(-out.grad, other.shape)

        out._backward_fn = _backward
        return out

    def __rsub__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        return other - self

    def __mul__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="mul",
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.shape)

        out._backward_fn = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="div",
        )

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad / other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(
                    out.grad * (-self.data / (other.data ** 2)), other.shape
                )

        out._backward_fn = _backward
        return out

    def __rtruediv__(self, other) -> "Tensor":
        other = _ensure_tensor(other)
        return other / self

    def __pow__(self, exponent: float) -> "Tensor":
        """Element-wise power (scalar exponent only)."""
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op=f"pow({exponent})",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * exponent * (self.data ** (exponent - 1))

        out._backward_fn = _backward
        return out

    # ---- matrix multiplication -------------------------------------------

    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication: self @ other."""
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op="matmul",
        )

        def _backward():
            if self.requires_grad:
                # (N,K) @ (K,M).T -> (N,M) — but handle batch dims
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward_fn = _backward
        return out

    def __matmul__(self, other) -> "Tensor":
        return self.matmul(_ensure_tensor(other))

    # ---- reductions -------------------------------------------------------

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="sum",
        )

        def _backward():
            if self.requires_grad:
                g = out.grad
                # Expand back to self.shape so the gradient broadcasts correctly
                if axis is not None and not keepdims:
                    # Re-insert the summed axis so np broadcast works
                    if isinstance(axis, int):
                        g = np.expand_dims(g, axis=axis)
                    else:
                        for ax in sorted(axis):
                            g = np.expand_dims(g, axis=ax)
                self.grad += np.broadcast_to(g, self.shape)

        out._backward_fn = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="mean",
        )

        # Number of elements averaged over
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = 1
            for ax in axis:
                n *= self.data.shape[ax]

        def _backward():
            if self.requires_grad:
                g = out.grad / n
                if axis is not None and not keepdims:
                    if isinstance(axis, int):
                        g = np.expand_dims(g, axis=axis)
                    else:
                        for ax in sorted(axis):
                            g = np.expand_dims(g, axis=ax)
                self.grad += np.broadcast_to(g, self.shape)

        out._backward_fn = _backward
        return out

    # ---- transpose --------------------------------------------------------

    def transpose(self, axes=None) -> "Tensor":
        if axes is None:
            out_data = self.data.T
        else:
            out_data = np.transpose(self.data, axes)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="transpose",
        )

        def _backward():
            if self.requires_grad:
                if axes is None:
                    self.grad += out.grad.T
                else:
                    # Invert the permutation
                    inv = [0] * len(axes)
                    for i, ax in enumerate(axes):
                        inv[ax] = i
                    self.grad += np.transpose(out.grad, inv)

        out._backward_fn = _backward
        return out

    # ---- reshape ----------------------------------------------------------

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="reshape",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)

        out._backward_fn = _backward
        return out

    # ---- non-linearities --------------------------------------------------

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(self.data, 0.0),
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="relu",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0).astype(np.float64)

        out._backward_fn = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="tanh",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1.0 - t ** 2)

        out._backward_fn = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(
            s,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="sigmoid",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * s * (1.0 - s)

        out._backward_fn = _backward
        return out

    def gelu(self) -> "Tensor":
        """Exact GELU: x * Phi(x) where Phi is the standard normal CDF."""
        from scipy.special import erf as _erf   # only for CDF — no autodiff

        x = self.data
        phi = 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))          # Phi(x)
        pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)  # phi(x)
        out_data = x * phi

        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="gelu",
        )

        def _backward():
            if self.requires_grad:
                # d/dx [x Phi(x)] = Phi(x) + x phi(x)
                self.grad += out.grad * (phi + x * pdf)

        out._backward_fn = _backward
        return out

    # ---- indexing (for cross-entropy) -------------------------------------

    def __getitem__(self, idx) -> "Tensor":
        out = Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="getitem",
        )

        def _backward():
            if self.requires_grad:
                # Scatter-add the gradient back
                np.add.at(self.grad, idx, out.grad)

        out._backward_fn = _backward
        return out

    # ---- exp / log --------------------------------------------------------

    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(
            e,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="exp",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * e

        out._backward_fn = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="log",
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data

        out._backward_fn = _backward
        return out

    # ---- clamp ------------------------------------------------------------

    def clamp(self, min_val=None, max_val=None) -> "Tensor":
        out_data = self.data.copy()
        if min_val is not None:
            out_data = np.maximum(out_data, min_val)
        if max_val is not None:
            out_data = np.minimum(out_data, max_val)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="clamp",
        )

        def _backward():
            if self.requires_grad:
                mask = np.ones_like(self.data)
                if min_val is not None:
                    mask *= (self.data >= min_val)
                if max_val is not None:
                    mask *= (self.data <= max_val)
                self.grad += out.grad * mask

        out._backward_fn = _backward
        return out

    # ---- detach -----------------------------------------------------------

    def detach(self) -> "Tensor":
        """Return a copy that does not track gradients."""
        return Tensor(self.data.copy(), requires_grad=False)


# ---------------------------------------------------------------------------
# Numerically stable softmax + cross-entropy (combined for stability)
# ---------------------------------------------------------------------------

def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Numerically stable softmax → cross-entropy loss.

    Parameters
    ----------
    logits : Tensor, shape (N, C)
        Raw (un-normalised) class scores.
    targets : ndarray of int, shape (N,)
        Ground-truth class indices.

    Returns
    -------
    Tensor (scalar)
        Mean cross-entropy loss over the batch.

    Notes
    -----
    The combined forward and backward avoids computing ``log(softmax)``
    separately, which is the standard trick for numerical stability.

    Forward:
        shifted = logits - max(logits, axis=1, keepdims=True)
        log_sum_exp = log(sum(exp(shifted), axis=1))
        loss = mean( -shifted[i, targets[i]] + log_sum_exp[i] )

    Backward (well-known result):
        d_logits = (softmax(logits) - one_hot(targets)) / N
    """
    N = logits.data.shape[0]
    C = logits.data.shape[1]

    # --- forward (numerically stable) ---
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    sum_exp = exp_shifted.sum(axis=1, keepdims=True)
    log_sum_exp = np.log(sum_exp)

    # loss_i = -shifted[i, target_i] + log_sum_exp[i]
    correct_class_scores = shifted[np.arange(N), targets]
    loss_per_sample = -correct_class_scores + log_sum_exp.squeeze()
    loss_val = loss_per_sample.mean()

    loss = Tensor(
        np.array(loss_val),
        requires_grad=logits.requires_grad,
        _parents=(logits,),
        _op="softmax_ce",
    )

    # --- backward ---
    softmax_probs = exp_shifted / sum_exp  # (N, C)

    def _backward():
        if logits.requires_grad:
            grad_logits = softmax_probs.copy()  # p_j
            grad_logits[np.arange(N), targets] -= 1.0  # p_j - 1{j=y}
            grad_logits /= N  # mean over batch
            logits.grad += loss.grad * grad_logits

    loss._backward_fn = _backward
    return loss


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _ensure_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)
