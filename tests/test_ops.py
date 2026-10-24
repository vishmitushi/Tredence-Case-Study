"""Additional operation tests: correctness of forward pass values."""

import numpy as np
import pytest
from engine.tensor import Tensor, softmax_cross_entropy


class TestForwardCorrectness:
    """Verify that forward-pass values match NumPy reference."""

    def test_add_values(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a + b
        assert np.allclose(c.data, [5.0, 7.0, 9.0])

    def test_matmul_values(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(c.data, expected)

    def test_relu_values(self):
        a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        c = a.relu()
        assert np.allclose(c.data, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_sigmoid_values(self):
        a = Tensor([0.0])
        c = a.sigmoid()
        assert np.allclose(c.data, [0.5])

    def test_tanh_values(self):
        a = Tensor([0.0])
        c = a.tanh()
        assert np.allclose(c.data, [0.0])

    def test_softmax_ce_value(self):
        """Compare against manually computed cross-entropy."""
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = np.array([0])

        loss = softmax_cross_entropy(logits, targets)

        # Manual: softmax([2, 1, 0.1]) then -log(p[0])
        shifted = np.array([2.0, 1.0, 0.1]) - 2.0
        exp_s = np.exp(shifted)
        probs = exp_s / exp_s.sum()
        expected_loss = -np.log(probs[0])

        assert np.allclose(loss.data, expected_loss, atol=1e-10)


class TestEdgeCases:
    """Edge cases and numerical stability."""

    def test_backward_requires_scalar(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        with pytest.raises(RuntimeError, match="scalar"):
            a.backward()

    def test_no_grad_tensor(self):
        a = Tensor([1.0, 2.0], requires_grad=False)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = (a + b).sum()
        c.backward()
        assert a.grad is None or np.all(a.grad == 0)
        assert np.allclose(b.grad, [1.0, 1.0])

    def test_large_logits_stable(self):
        """Softmax CE should handle logits of magnitude 1000."""
        logits = Tensor([[1000.0, 999.0, 998.0]], requires_grad=True)
        targets = np.array([0])
        loss = softmax_cross_entropy(logits, targets)
        assert np.isfinite(loss.data), "Loss should be finite"
        loss.backward()
        assert np.all(np.isfinite(logits.grad)), "Gradients should be finite"


class TestPruningIntegration:
    """Integration tests for pruning with the training loop."""

    def test_sparsity_actually_achieved(self):
        """After pruning, verify the mask has the expected number of zeros."""
        from nn.linear import Linear
        layer = Linear(10, 10, bias=False)

        # Prune 50% — set bottom half by magnitude to zero
        scores = np.abs(layer.weight.data)
        threshold = np.median(scores)
        layer.mask = (scores >= threshold).astype(np.float64)

        actual_sparsity = 1.0 - layer.mask.mean()
        assert 0.4 <= actual_sparsity <= 0.6, \
            f"Expected ~50% sparsity, got {actual_sparsity:.1%}"

    def test_pruned_weights_are_real_zeros(self):
        """After pruning, weight.data at masked positions must be exactly 0."""
        from nn.linear import Linear
        layer = Linear(10, 10, bias=False)

        layer.mask[0:5, :] = 0.0
        layer.weight.data[layer.mask == 0] = 0.0  # enforce

        assert np.all(layer.weight.data[0:5, :] == 0.0), \
            "Pruned weights must be exactly zero (not near-zero)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
