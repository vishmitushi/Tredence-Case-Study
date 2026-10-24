# AQUA Self-Pruning Network

A neural network library built from scratch in pure Python and NumPy that trains a classifier capable of pruning its own least-important connections during training, ending under a target sparsity budget while maintaining accuracy.

**scikit-learn is used ONLY to load the `digits` dataset** — not for training, models, or gradients.

## Project Structure

```
├── engine/
│   ├── __init__.py
│   └── tensor.py          # Reverse-mode autodiff engine (Part 1)
├── nn/
│   ├── __init__.py
│   ├── module.py           # Base Module and Sequential
│   ├── linear.py           # Linear layer with mask support
│   ├── activations.py      # ReLU, Tanh, Sigmoid, GELU
│   ├── loss.py             # Softmax cross-entropy
│   └── optim.py            # Adam and SGD with momentum
├── prune/
│   ├── __init__.py
│   ├── importance.py       # Importance criteria (magnitude, Taylor, movement)
│   ├── schedule.py         # Cubic sparsity schedule
│   └── pruner.py           # Pruner with regrowth support
├── train/
│   ├── __init__.py
│   ├── train_base.py       # Part 2: baseline training (no pruning)
│   ├── train_prune.py      # Part 3: training with self-pruning
│   └── sweep.py            # Part 4: Pareto sweep
├── tests/
│   ├── __init__.py
│   ├── test_gradients.py   # Gradient checking for every operation
│   ├── test_masked.py      # Masked-weight correctness tests
│   └── test_ops.py         # Forward-pass and integration tests
├── results/                # Plots and raw numbers (generated)
├── README.md
├── DESIGN.md
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd aqua-self-pruning

# Install dependencies (Python 3.10+)
pip install -r requirements.txt

# Or with uv:
uv pip install -r requirements.txt
```

## Quick Start — Reproduce All Results

### 1. Run gradient-check tests (Part 1)

```bash
python -m pytest tests/test_gradients.py -v
```

### 2. Run masked-weight correctness tests (Part 1)

```bash
python -m pytest tests/test_masked.py -v
```

### 3. Run all tests

```bash
python -m pytest tests/ -v
```

### 4. Reproduce Part 2 training (baseline MLP, no pruning)

```bash
python -m train.train_base --seed 42 --epochs 40
```

Produces `results/part2_learning_curve.png`.

### 5. Reproduce Part 3 self-pruning run

```bash
python -m train.train_prune --seed 42 --epochs 50 --target-sparsity 0.9 --criterion taylor
```

Produces `results/part3_pruning_taylor_90pct.png`.

### 6. Reproduce Part 4 Pareto sweep

```bash
python -m train.sweep --seeds 3 --epochs 50
```

Produces:
- `results/pareto_curve.png` — Sparsity–accuracy Pareto curve
- `results/pareto_raw.csv` — Raw numbers
- `results/cost_comparison.txt` — FLOP analysis
- `results/falsifiable_claim.txt` — Quantitative claim

## Key Design Decisions

### Autodiff Engine
- Float64 throughout for gradient-check numerical stability
- `_unbroadcast` helper correctly handles all NumPy broadcasting patterns
- Combined softmax + cross-entropy for numerical stability (log-sum-exp trick)
- Topological-sort backward pass with gradient accumulation

### Masked-Weight Handling
- Mask is applied **inside the computation graph**: `output = x @ (W * mask) + b`
- This naturally produces **zero gradient** for pruned weights (correct by construction)
- Adam optimizer tracks **per-element step counts** to avoid bias-correction corruption on revival
- On any mask change (prune or revive), `m`, `v`, and step count are reset for affected entries

### Importance Criterion
- Primary: Taylor importance `|w * g|` — first-order approximation of loss change from removal
- Baseline: Magnitude `|w|`
- Alternative: Movement `|w - w_0|`

### Pruning Schedule
- Cubic sparsity ramp (gradual, not one-shot)
- Starts at 10% of training, ends at 75%
- Remaining 25% is fine-tuning at final sparsity
- Optional regrowth (capped at 10% of pruned connections per round)

## Reproducibility

All random seeds are fixed. Running the commands above with the default seeds will reproduce the reported numbers. The `results/pareto_raw.csv` file contains per-seed results for statistical analysis.

## Dataset

We use sklearn's `load_digits` (8×8 grayscale images, 10 classes, 1797 samples) with an 80/20 train/test split. This dataset is small enough for the custom NumPy engine to train in reasonable time while being a genuine classification challenge.
