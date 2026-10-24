"""Gradient-checking suite for the autodiff engine.

For every differentiable operation we implemented, we verify that the
analytic gradient (from backward()) agrees with the finite-difference
numerical gradient to within tight tolerances.

Finite-difference formula (central difference):
    df/dx_i ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps)

We use eps=1e-5 and require:
    |analytic - numerical| / (|analytic| + |numerical| + 1e-8) < 1e-5

Usage:
    python -m pytest tests/test_gradients.py -v
"""

import numpy as np
import pytest
from engine.tensor import Tensor, softmax_cross_entropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numerical_gradient(f, tensor, eps=1e-5):
    """Compute the numerical gradient of scalar function f w.r.t. tensor.

    Parameters
    ----------
    f : callable
        A function that takes no arguments, uses `tensor` via closure,
        and returns a scalar Tensor.
    tensor : Tensor
        The parameter to differentiate with respect to.
    eps : float
        Finite-difference step size.

    Returns
    -------
    grad : ndarray
        Numerical gradient, same shape as tensor.data.
    """
    grad = np.zeros_like(tensor.data)
    it = np.nditer(tensor.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old_val = tensor.data[idx]

        tensor.data[idx] = old_val + eps
        loss_plus = float(f().data)

        tensor.data[idx] = old_val - eps
        loss_minus = float(f().data)

        tensor.data[idx] = old_val  # restore
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        it.iternext()
    return grad


def check_gradient(f, tensor, rtol=1e-5, atol=1e-6):
    """Compare analytic and numerical gradients.

    Parameters
    ----------
    f : callable
        Returns a scalar Tensor that depends on `tensor`.
    tensor : Tensor
        Parameter to check.
    rtol, atol : float
        Tolerance for np.allclose.
    """
    # Analytic gradient
    loss = f()
    loss.backward()
    analytic = tensor.grad.copy()

    # Numerical gradient
    numeric = numerical_gradient(f, tensor)

    # Relative comparison
    max_diff = np.max(np.abs(analytic - numeric))
    denom = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-8)
    max_rel = np.max(np.abs(analytic - numeric) / denom)

    ok = np.allclose(analytic, numeric, rtol=rtol, atol=atol)
    if not ok:
        print(f"  max abs diff: {max_diff:.2e}")
        print(f"  max rel diff: {max_rel:.2e}")
        print(f"  analytic:\n{analytic}")
        print(f"  numeric:\n{numeric}")
    return ok


# ---------------------------------------------------------------------------
# Tests for each operation
# ---------------------------------------------------------------------------

class TestElementwiseOps:
    """Test gradients of element-wise operations."""

    def test_add(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: (a + b).sum(), a)
        assert check_gradient(lambda: (a + b).sum(), b)

    def test_add_broadcast(self):
        """(N,D) + (D,) — the broadcasting case called out in the spec."""
        a = Tensor(np.random.randn(5, 3), requires_grad=True)
        b = Tensor(np.random.randn(3), requires_grad=True)
        assert check_gradient(lambda: (a + b).sum(), a)
        assert check_gradient(lambda: (a + b).sum(), b)

    def test_add_broadcast_keepdims(self):
        """(N,D) + (1,D) broadcasting."""
        a = Tensor(np.random.randn(4, 3), requires_grad=True)
        b = Tensor(np.random.randn(1, 3), requires_grad=True)
        assert check_gradient(lambda: (a + b).sum(), a)
        assert check_gradient(lambda: (a + b).sum(), b)

    def test_sub(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: (a - b).sum(), a)
        assert check_gradient(lambda: (a - b).sum(), b)

    def test_sub_broadcast(self):
        a = Tensor(np.random.randn(4, 3), requires_grad=True)
        b = Tensor(np.random.randn(3), requires_grad=True)
        assert check_gradient(lambda: (a - b).sum(), a)
        assert check_gradient(lambda: (a - b).sum(), b)

    def test_mul(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: (a * b).sum(), a)
        assert check_gradient(lambda: (a * b).sum(), b)

    def test_mul_broadcast(self):
        a = Tensor(np.random.randn(4, 3), requires_grad=True)
        b = Tensor(np.random.randn(3), requires_grad=True)
        assert check_gradient(lambda: (a * b).sum(), a)
        assert check_gradient(lambda: (a * b).sum(), b)

    def test_div(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(3, 4) * 2 + 3, requires_grad=True)  # avoid near-zero
        assert check_gradient(lambda: (a / b).sum(), a)
        assert check_gradient(lambda: (a / b).sum(), b)

    def test_div_broadcast(self):
        a = Tensor(np.random.randn(4, 3), requires_grad=True)
        b = Tensor(np.random.randn(3) * 2 + 3, requires_grad=True)
        assert check_gradient(lambda: (a / b).sum(), a)
        assert check_gradient(lambda: (a / b).sum(), b)

    def test_neg(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: (-a).sum(), a)

    def test_pow(self):
        a = Tensor(np.abs(np.random.randn(3, 4)) + 0.5, requires_grad=True)
        assert check_gradient(lambda: (a ** 2).sum(), a)
        assert check_gradient(lambda: (a ** 0.5).sum(), a)


class TestMatmul:
    """Test gradients of matrix multiplication."""

    def test_basic(self):
        a = Tensor(np.random.randn(5, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 3), requires_grad=True)
        assert check_gradient(lambda: (a @ b).sum(), a)
        assert check_gradient(lambda: (a @ b).sum(), b)

    def test_vector(self):
        """Matrix @ vector."""
        a = Tensor(np.random.randn(5, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 1), requires_grad=True)
        assert check_gradient(lambda: (a @ b).sum(), a)
        assert check_gradient(lambda: (a @ b).sum(), b)


class TestReductions:
    """Test gradients of sum and mean."""

    def test_sum_all(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: a.sum(), a)

    def test_sum_axis0(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: a.sum(axis=0).sum(), a)

    def test_sum_axis1(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: a.sum(axis=1).sum(), a)

    def test_mean_all(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: a.mean(), a)

    def test_mean_axis0(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: a.mean(axis=0).sum(), a)

    def test_mean_axis1(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: a.mean(axis=1).sum(), a)


class TestActivations:
    """Test gradients of non-linearities."""

    def test_relu(self):
        # Use values away from 0 where ReLU is non-differentiable
        a = Tensor(np.random.randn(4, 5) * 2, requires_grad=True)
        a.data[np.abs(a.data) < 0.1] = 0.5  # avoid the kink
        assert check_gradient(lambda: a.relu().sum(), a)

    def test_tanh(self):
        a = Tensor(np.random.randn(4, 5), requires_grad=True)
        assert check_gradient(lambda: a.tanh().sum(), a)

    def test_sigmoid(self):
        a = Tensor(np.random.randn(4, 5), requires_grad=True)
        assert check_gradient(lambda: a.sigmoid().sum(), a)

    def test_gelu(self):
        a = Tensor(np.random.randn(4, 5), requires_grad=True)
        assert check_gradient(lambda: a.gelu().sum(), a)


class TestExpLog:
    """Test gradients of exp and log."""

    def test_exp(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: a.exp().sum(), a)

    def test_log(self):
        a = Tensor(np.abs(np.random.randn(3, 4)) + 0.1, requires_grad=True)
        assert check_gradient(lambda: a.log().sum(), a)


class TestSoftmaxCrossEntropy:
    """Test gradient of the combined softmax + cross-entropy."""

    def test_basic(self):
        np.random.seed(123)
        logits = Tensor(np.random.randn(8, 5), requires_grad=True)
        targets = np.array([0, 1, 2, 3, 4, 0, 1, 2])

        def f():
            return softmax_cross_entropy(logits, targets)

        assert check_gradient(f, logits, rtol=1e-4, atol=1e-5)

    def test_single_sample(self):
        logits = Tensor(np.random.randn(1, 4), requires_grad=True)
        targets = np.array([2])

        def f():
            return softmax_cross_entropy(logits, targets)

        assert check_gradient(f, logits, rtol=1e-4, atol=1e-5)

    def test_large_logits(self):
        """Test numerical stability with large logits."""
        logits = Tensor(np.random.randn(4, 3) * 100, requires_grad=True)
        targets = np.array([0, 1, 2, 0])

        def f():
            return softmax_cross_entropy(logits, targets)

        loss = f()
        assert np.isfinite(loss.data), "Loss should be finite even with large logits"
        assert check_gradient(f, logits, rtol=1e-4, atol=1e-5)


class TestComposedGradients:
    """Test gradients through composed operations (multi-layer)."""

    def test_linear_layer(self):
        """y = x @ W + b → loss = sum(y)."""
        np.random.seed(42)
        x = Tensor(np.random.randn(4, 3), requires_grad=False)
        W = Tensor(np.random.randn(3, 5), requires_grad=True)
        b = Tensor(np.random.randn(5), requires_grad=True)

        def f():
            return (x @ W + b).sum()

        assert check_gradient(f, W)
        assert check_gradient(f, b)

    def test_two_layer_mlp(self):
        """Two-layer MLP: ReLU(x @ W1 + b1) @ W2 + b2."""
        np.random.seed(42)
        x = Tensor(np.random.randn(4, 3), requires_grad=False)
        W1 = Tensor(np.random.randn(3, 5), requires_grad=True)
        b1 = Tensor(np.random.randn(5), requires_grad=True)
        W2 = Tensor(np.random.randn(5, 2), requires_grad=True)
        b2 = Tensor(np.random.randn(2), requires_grad=True)

        def f():
            h = (x @ W1 + b1).relu()
            return (h @ W2 + b2).sum()

        assert check_gradient(f, W1)
        assert check_gradient(f, b1)
        assert check_gradient(f, W2)
        assert check_gradient(f, b2)

    def test_gradient_accumulation(self):
        """A tensor used twice should accumulate gradients."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        # Use a in two different operations
        def f():
            return (a + a).sum()

        loss = f()
        loss.backward()
        # d/da (a + a) = 2
        expected = 2.0 * np.ones_like(a.data)
        assert np.allclose(a.grad, expected), \
            f"Expected grad=2, got {a.grad}"

    def test_reuse_in_product(self):
        """a * a should give gradient 2*a."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)

        def f():
            return (a * a).sum()

        assert check_gradient(f, a)


class TestTransposeReshape:
    """Test gradients through transpose and reshape."""

    def test_transpose(self):
        a = Tensor(np.random.randn(3, 5), requires_grad=True)
        assert check_gradient(lambda: a.transpose().sum(), a)

    def test_reshape(self):
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        assert check_gradient(lambda: a.reshape(12).sum(), a)
        assert check_gradient(lambda: a.reshape(2, 6).sum(), a)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
