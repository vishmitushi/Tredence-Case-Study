"""Part 2 — Train a small MLP on a classification task (no pruning).

Uses sklearn ONLY to load the digits dataset (8x8 images, 10 classes,
1797 samples).  All training, optimisation, and evaluation is our own code.

Usage:
    python -m train.train_base [--seed 42] [--epochs 40] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

# sklearn is used ONLY for loading a standard dataset — not for training,
# models, or gradients.
from sklearn.datasets import load_digits

# Our own modules
from engine.tensor import Tensor
from nn.module import Sequential
from nn.linear import Linear
from nn.activations import ReLU
from nn.loss import softmax_cross_entropy
from nn.optim import Adam


def load_data(seed: int = 42):
    """Load and preprocess the digits dataset.

    Returns
    -------
    X_train, y_train, X_test, y_test : ndarrays
        Normalised features and integer labels.
    """
    digits = load_digits()
    X = digits.data.astype(np.float64)  # (1797, 64)
    y = digits.target.astype(np.int64)  # (1797,)

    # Normalise to zero-mean, unit-variance (per feature)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    # Train/test split (80/20)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    return X_train, y_train, X_test, y_test


def build_model(input_dim: int = 64, hidden: int = 128, n_classes: int = 10):
    """Build a 3-layer MLP: 64 → 128 → 64 → 10.

    Weight initialisation: He (2015) for ReLU layers.
    """
    return Sequential(
        Linear(input_dim, hidden, init="he"),
        ReLU(),
        Linear(hidden, hidden // 2, init="he"),
        ReLU(),
        Linear(hidden // 2, n_classes, init="he"),
    )


def evaluate(model, X: np.ndarray, y: np.ndarray) -> tuple:
    """Compute loss and accuracy (no gradient tracking needed)."""
    logits = model(Tensor(X))
    loss = softmax_cross_entropy(logits, y)
    preds = logits.data.argmax(axis=1)
    acc = (preds == y).mean()
    return float(loss.data), float(acc)


def train(
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 128,
    save_plot: bool = True,
):
    """Full training loop.  Returns the trained model and history."""
    np.random.seed(seed)

    X_train, y_train, X_test, y_test = load_data(seed)
    model = build_model(input_dim=X_train.shape[1], hidden=hidden)
    optimizer = Adam(model.parameters(), lr=lr)

    n_train = len(X_train)
    steps_per_epoch = max(1, n_train // batch_size)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Test Loss':>9} | {'Test Acc':>8} | {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        rng = np.random.RandomState(seed + epoch)
        perm = rng.permutation(n_train)

        epoch_loss = 0.0
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

            # Update
            optimizer.step()

            epoch_loss += float(loss.data)

        epoch_loss /= steps_per_epoch
        train_loss, train_acc = evaluate(model, X_train, y_train)
        test_loss, test_acc = evaluate(model, X_test, y_test)
        dt = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:9.4f} | "
              f"{test_loss:9.4f} | {test_acc:8.4f} | {dt:5.1f}s")

    # --- save learning curve plot ---
    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(history["epoch"], history["train_loss"], label="Train")
            ax1.plot(history["epoch"], history["test_loss"], label="Test")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Learning Curve — Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(history["epoch"], history["train_acc"], label="Train")
            ax2.plot(history["epoch"], history["test_acc"], label="Test")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Learning Curve — Accuracy")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            os.makedirs("results", exist_ok=True)
            plt.tight_layout()
            plt.savefig("results/part2_learning_curve.png", dpi=150)
            plt.close()
            print(f"\nPlot saved to results/part2_learning_curve.png")
        except ImportError:
            print("\nmatplotlib not available — skipping plot")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2: Train MLP (no pruning)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    model, history = train(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
    )

    print(f"\nFinal test accuracy: {history['test_acc'][-1]:.4f}")
