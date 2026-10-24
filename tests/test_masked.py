"""Tests for masked-weight correctness (Part 1 requirement).

These tests verify that:
1. A masked weight contributes exactly zero to the forward pass.
2. A masked weight receives exactly zero gradient.
3. Adam optimizer state (m, v) is correctly reset when a weight is
   pruned and then revived.
4. A revived weight does not produce a corrupted update.

This is the "subtle thing" most quick implementations get wrong.

Usage:
    python -m pytest tests/test_masked.py -v
"""

import numpy as np
import pytest
from engine.tensor import Tensor, softmax_cross_entropy
from nn.linear import Linear
from nn.optim import Adam


class TestMaskedForward:
    """Verify that masked weights produce zero output contribution."""

    def test_masked_weight_zero_contribution(self):
        """Setting mask[i,j]=0 should make weight[i,j] invisible."""
        np.random.seed(42)
        layer = Linear(4, 3, bias=False)

        # Pick a specific weight to mask
        layer.mask[1, 2] = 0.0
        layer.weight.data[1, 2] = 999.0  # should have NO effect

        x = Tensor(np.random.randn(2, 4))
        y = layer(x)

        # Compare with a layer where that weight is simply zero
        layer2 = Linear(4, 3, bias=False)
        layer2.weight.data = layer.weight.data.copy()
        layer2.weight.data[1, 2] = 0.0
        # No mask on layer2 — just zeroed weight
        y2 = layer2(x)

        assert np.allclose(y.data, y2.data), \
            "Masked weight should produce same result as a zeroed weight"

    def test_full_row_masked(self):
        """Masking an entire input feature should zero its contribution."""
        np.random.seed(42)
        layer = Linear(4, 3, bias=False)
        layer.mask[2, :] = 0.0  # mask all outgoing connections from input 2

        x = Tensor(np.ones((1, 4)))
        y = layer(x)

        # Manually compute: only rows 0, 1, 3 of W should contribute
        expected = x.data @ (layer.weight.data * layer.mask)
        assert np.allclose(y.data, expected)


class TestMaskedGradient:
    """Verify gradient treatment for masked weights."""

    def test_masked_weight_zero_gradient(self):
        """A masked weight should receive zero gradient."""
        np.random.seed(42)
        layer = Linear(4, 3, bias=False)
        layer.mask[1, 2] = 0.0

        x = Tensor(np.random.randn(5, 4))
        targets = np.array([0, 1, 2, 0, 1])

        logits = layer(x)
        loss = softmax_cross_entropy(logits, targets)
        loss.backward()

        # The gradient at the masked position should be exactly zero
        assert layer.weight.grad[1, 2] == 0.0, \
            f"Expected zero gradient for masked weight, got {layer.weight.grad[1, 2]}"

    def test_unmasked_weights_get_gradient(self):
        """Non-masked weights should receive non-trivial gradients."""
        np.random.seed(42)
        layer = Linear(4, 3, bias=False)
        layer.mask[1, 2] = 0.0  # only this one is masked

        x = Tensor(np.random.randn(5, 4))
        targets = np.array([0, 1, 2, 0, 1])

        logits = layer(x)
        loss = softmax_cross_entropy(logits, targets)
        loss.backward()

        # Other weights should have non-zero gradients
        unmasked_grads = layer.weight.grad.copy()
        unmasked_grads[1, 2] = 0.0  # exclude the masked one
        assert np.any(unmasked_grads != 0.0), \
            "Unmasked weights should receive gradients"

    def test_gradient_check_with_mask(self):
        """Finite-difference check that masked gradients are correct."""
        np.random.seed(42)
        layer = Linear(4, 3, bias=False)
        layer.mask[0, 1] = 0.0
        layer.mask[2, 0] = 0.0

        x = Tensor(np.random.randn(3, 4))
        targets = np.array([0, 1, 2])

        def f():
            logits = layer(x)
            return softmax_cross_entropy(logits, targets)

        # Analytic gradient
        loss = f()
        loss.backward()
        analytic = layer.weight.grad.copy()

        # Numerical gradient
        eps = 1e-5
        numeric = np.zeros_like(layer.weight.data)
        it = np.nditer(layer.weight.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = layer.weight.data[idx]

            layer.weight.data[idx] = old + eps
            l_plus = float(f().data)

            layer.weight.data[idx] = old - eps
            l_minus = float(f().data)

            layer.weight.data[idx] = old
            numeric[idx] = (l_plus - l_minus) / (2 * eps)
            it.iternext()

        # Masked positions: both analytic and numeric should be ~0
        for pos in [(0, 1), (2, 0)]:
            assert abs(analytic[pos]) < 1e-12, \
                f"Analytic grad at masked {pos} should be 0, got {analytic[pos]}"
            assert abs(numeric[pos]) < 1e-8, \
                f"Numeric grad at masked {pos} should be ~0, got {numeric[pos]}"

        # Unmasked: should agree
        mask_bool = layer.mask.astype(bool)
        assert np.allclose(
            analytic[mask_bool], numeric[mask_bool], rtol=1e-4, atol=1e-6
        ), "Analytic and numeric gradients should agree for unmasked weights"


class TestAdamMaskHandling:
    """Verify that Adam handles pruned-then-revived connections correctly."""

    def test_adam_moments_reset_on_prune(self):
        """When a weight is pruned, its Adam moments should be reset."""
        np.random.seed(42)
        layer = Linear(4, 3, bias=False)
        optimizer = Adam([layer.weight], lr=0.01)

        # Train for a few steps to build up momentum
        x = Tensor(np.random.randn(5, 4))
        targets = np.array([0, 1, 2, 0, 1])

        for _ in range(5):
            optimizer.zero_grad()
            logits = layer(x)
            loss = softmax_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        # Check that moments are non-zero
        state = optimizer.state[id(layer.weight)]
        assert np.any(state["m"] != 0), "m should be non-zero after training"
        assert np.any(state["v"] != 0), "v should be non-zero after training"

        # Now prune weight[1,2]
        old_mask = layer.mask.copy()
        new_mask = old_mask.copy()
        new_mask[1, 2] = 0.0
        layer.mask = new_mask

        optimizer.reset_state_for_mask_changes(
            layer.weight, old_mask, new_mask
        )

        # The moments at [1,2] should be zeroed
        assert state["m"][1, 2] == 0.0, \
            f"m[1,2] should be 0 after pruning, got {state['m'][1, 2]}"
        assert state["v"][1, 2] == 0.0, \
            f"v[1,2] should be 0 after pruning, got {state['v'][1, 2]}"
        assert state["step"][1, 2] == 0, \
            f"step[1,2] should be 0 after pruning, got {state['step'][1, 2]}"

        # Other positions should be unchanged
        assert state["m"][0, 0] != 0.0, "m[0,0] should be unchanged"

    def test_adam_revival_no_corruption(self):
        """A revived weight should train cleanly (no momentum corruption).

        The failure mode: if we do NOT reset moments, a revived weight
        has decayed moments m≈0, v≈0, and the bias-corrected update
        lr * m_hat / (sqrt(v_hat) + eps) → lr / eps ≈ 1e5 * lr,
        causing a catastrophic spike.
        """
        np.random.seed(42)
        layer = Linear(4, 3, bias=False)
        optimizer = Adam([layer.weight], lr=0.01)

        x = Tensor(np.random.randn(10, 4))
        targets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

        # Phase 1: Train normally for 20 steps
        for _ in range(20):
            optimizer.zero_grad()
            logits = layer(x)
            loss = softmax_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        # Phase 2: Prune weight[1,2] for 20 steps
        old_mask = layer.mask.copy()
        new_mask = old_mask.copy()
        new_mask[1, 2] = 0.0
        layer.mask = new_mask
        layer.weight.data[1, 2] = 0.0
        optimizer.reset_state_for_mask_changes(
            layer.weight, old_mask, new_mask
        )

        for _ in range(20):
            optimizer.zero_grad()
            logits = layer(x)
            loss = softmax_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            layer.weight.data[1, 2] = 0.0  # enforce mask

        # Phase 3: Revive weight[1,2]
        old_mask2 = layer.mask.copy()
        new_mask2 = np.ones_like(layer.mask)
        layer.mask = new_mask2
        optimizer.reset_state_for_mask_changes(
            layer.weight, old_mask2, new_mask2
        )

        # The revived weight should start from 0 with fresh moments
        assert layer.weight.data[1, 2] == 0.0

        # Do one training step — the update should be reasonable
        optimizer.zero_grad()
        logits = layer(x)
        loss = softmax_cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        # The update should be on the order of lr, NOT lr/eps
        update_magnitude = abs(layer.weight.data[1, 2])
        assert update_magnitude < 1.0, \
            f"Revived weight update should be reasonable, got {update_magnitude}"
        assert np.all(np.isfinite(layer.weight.data)), \
            "All weights should remain finite after revival"

    def test_pruned_weight_stays_zero_in_forward(self):
        """Even after optimizer step, pruned weight output is zero."""
        np.random.seed(42)
        layer = Linear(4, 3, bias=False)
        optimizer = Adam([layer.weight], lr=0.01)

        layer.mask[1, 2] = 0.0

        x = Tensor(np.random.randn(5, 4))
        targets = np.array([0, 1, 2, 0, 1])

        for _ in range(5):
            optimizer.zero_grad()
            logits = layer(x)
            loss = softmax_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            # Forward output should be same as if weight[1,2] doesn't exist
            y_with_mask = layer(x)
            # Manually zero the weight and compute without mask
            w_copy = layer.weight.data.copy()
            w_copy[1, 2] = 0.0
            y_expected = x.data @ w_copy
            assert np.allclose(y_with_mask.data, y_expected, atol=1e-10), \
                "Pruned weight should not affect forward pass"


class TestMaskGradientSemantics:
    """Test the exact gradient semantics of masked weights.

    Our design choice: the gradient of a masked weight is ZERO.

    Rationale: since the masked weight does not participate in the forward
    pass (it is multiplied by 0 via the mask), it has no influence on the
    loss.  Therefore dL/dw = 0 is mathematically correct — it reflects the
    actual computation that was performed.

    For REGROWTH decisions, we separately compute importance scores that
    may use unmasked gradient information, but the optimizer never sees
    non-zero gradients for pruned weights.
    """

    def test_gradient_is_exactly_zero(self):
        """Masked weight gradient = 0, verified across multiple inputs."""
        np.random.seed(42)
        layer = Linear(8, 5, bias=True)
        # Mask out half the weights
        layer.mask[:4, :] = 0.0

        for trial in range(3):
            x = Tensor(np.random.randn(10, 8))
            targets = np.random.randint(0, 5, size=10)

            logits = layer(x)
            loss = softmax_cross_entropy(logits, targets)
            loss.backward()

            masked_grads = layer.weight.grad[:4, :]
            assert np.all(masked_grads == 0.0), \
                f"Trial {trial}: masked gradients should be exactly 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
