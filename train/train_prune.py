"""Part 3 — Train a self-pruning MLP.

Same network as Part 2, but with gradual pruning during training using
a Taylor-based importance criterion and a cubic sparsity schedule.

Usage:
    python -m train.train_prune [--target-sparsity 0.9] [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np

from engine.tensor import Tensor
from nn.module import Sequential
from nn.linear import Linear
from nn.activations import ReLU
from nn.loss import softmax_cross_entropy
from nn.optim import Adam
from prune.pruner import Pruner
from prune.schedule import CubicSparsitySchedule
from train.train_base import load_data, evaluate


def build_model(input_dim: int = 64, hidden: int = 128, n_classes: int = 10):
    """Build the same 3-layer MLP as Part 2."""
    return Sequential(
        Linear(input_dim, hidden, init="he"),
        ReLU(),
        Linear(hidden, hidden // 2, init="he"),
        ReLU(),
        Linear(hidden // 2, n_classes, init="he"),
    )


def get_linear_layers(model: Sequential):
    """Extract all Linear layers from a Sequential model."""
    return [layer for layer in model.layers if isinstance(layer, Linear)]


def sparse_forward_flops(model: Sequential, input_dim: int) -> int:
    """Count multiply-accumulate ops using only active (non-pruned) connections.

    For a Linear layer y = x @ (W * mask) + b:
      FLOPs = 2 * (number of 1s in mask) * batch_size_1
    We report per-sample FLOPs (batch=1).
    """
    flops = 0
    for layer in model.layers:
        if isinstance(layer, Linear):
            active = int(layer.mask.sum())
            flops += 2 * active  # multiply + add per connection
    return flops


def dense_forward_flops(model: Sequential) -> int:
    """Count multiply-accumulate ops assuming all weights are active."""
    flops = 0
    for layer in model.layers:
        if isinstance(layer, Linear):
            flops += 2 * layer.in_features * layer.out_features
    return flops


def train_with_pruning(
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 128,
    target_sparsity: float = 0.9,
    criterion: str = "taylor",
    regrowth: bool = True,
    save_plot: bool = True,
):
    """Train with gradual pruning.  Returns the model and history."""
    np.random.seed(seed)

    X_train, y_train, X_test, y_test = load_data(seed)
    model = build_model(input_dim=X_train.shape[1], hidden=hidden)
    optimizer = Adam(model.parameters(), lr=lr)

    n_train = len(X_train)
    steps_per_epoch = max(1, n_train // batch_size)
    total_steps = epochs * steps_per_epoch

    linear_layers = get_linear_layers(model)

    schedule = CubicSparsitySchedule(
        target_sparsity=target_sparsity,
        total_steps=total_steps,
        start_frac=0.1,
        end_frac=0.75,
        prune_every=max(1, steps_per_epoch // 2),  # prune ~twice per epoch
    )

    pruner = Pruner(
        layers=linear_layers,
        optimizer=optimizer,
        schedule=schedule,
        criterion=criterion,
        regrowth=regrowth,
        regrowth_fraction=0.1,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "sparsity": [],
        "active_params": [],
        "sparse_flops": [],
    }

    print(f"Training with {criterion} pruning → target sparsity {target_sparsity:.0%}")
    print(f"{'Epoch':>5} | {'TrLoss':>7} | {'TrAcc':>6} | "
          f"{'TeLoss':>7} | {'TeAcc':>6} | {'Spars':>6} | {'Active':>7} | {'Time':>5}")
    print("-" * 72)

    global_step = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        rng = np.random.RandomState(seed + epoch)
        perm = rng.permutation(n_train)

        for step in range(steps_per_epoch):
            batch_idx = perm[step * batch_size : (step + 1) * batch_size]
            xb = Tensor(X_train[batch_idx])
            yb = y_train[batch_idx]

            # Forward
            logits = model(xb)
            loss = softmax_cross_entropy(logits, yb)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Pruning step (before optimizer, so gradients are available)
            pruner.step(global_step)

            # Optimizer step
            optimizer.step()

            # Enforce mask after optimizer step (belt-and-suspenders)
            for layer in linear_layers:
                layer.weight.data[layer.mask == 0] = 0.0

            global_step += 1

        train_loss, train_acc = evaluate(model, X_train, y_train)
        test_loss, test_acc = evaluate(model, X_test, y_test)
        sparsity = pruner.get_total_sparsity()
        active = pruner.get_active_params()
        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)
        history["active_params"].append(active)
        history["sparse_flops"].append(sparse_forward_flops(model, X_train.shape[1]))

        print(f"{epoch:5d} | {train_loss:7.4f} | {train_acc:6.3f} | "
              f"{test_loss:7.4f} | {test_acc:6.3f} | {sparsity:6.1%} | "
              f"{active:7d} | {dt:4.1f}s")

    # --- summary ---
    dense_fl = dense_forward_flops(model)
    sparse_fl = sparse_forward_flops(model, X_train.shape[1])
    print(f"\nFinal sparsity:    {pruner.get_total_sparsity():.1%}")
    print(f"Final test acc:    {history['test_acc'][-1]:.4f}")
    print(f"Dense FLOPs:       {dense_fl}")
    print(f"Sparse FLOPs:      {sparse_fl}")
    print(f"FLOP reduction:    {1 - sparse_fl / dense_fl:.1%}")
    print(f"Active / total:    {pruner.get_active_params()} / {pruner.get_total_params()}")

    # --- save plot ---
    if save_plot:
        _save_training_plot(history, target_sparsity, criterion)

    return model, history, pruner


def _save_training_plot(history, target_sparsity, criterion):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(history["epoch"], history["train_loss"], label="Train")
        axes[0].plot(history["epoch"], history["test_loss"], label="Test")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history["epoch"], history["train_acc"], label="Train")
        axes[1].plot(history["epoch"], history["test_acc"], label="Test")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(history["epoch"], [s * 100 for s in history["sparsity"]])
        axes[2].axhline(y=target_sparsity * 100, color="r", linestyle="--",
                        label=f"Target ({target_sparsity:.0%})")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Sparsity %")
        axes[2].set_title("Sparsity Schedule")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        os.makedirs("results", exist_ok=True)
        plt.tight_layout()
        fname = f"results/part3_pruning_{criterion}_{int(target_sparsity*100)}pct.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"\nPlot saved to {fname}")
    except ImportError:
        print("\nmatplotlib not available — skipping plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 3: Self-pruning MLP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--target-sparsity", type=float, default=0.9)
    parser.add_argument("--criterion", type=str, default="taylor",
                        choices=["magnitude", "taylor", "movement"])
    parser.add_argument("--no-regrowth", action="store_true")
    args = parser.parse_args()

    model, history, pruner = train_with_pruning(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        target_sparsity=args.target_sparsity,
        criterion=args.criterion,
        regrowth=not args.no_regrowth,
    )
