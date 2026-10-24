"""Part 4 — Pareto sweep: sparsity vs. accuracy for multiple criteria.

Sweeps target sparsity levels {0%, 50%, 75%, 90%, 95%} across multiple
random seeds, comparing Taylor pruning vs. magnitude pruning.

Produces:
  - results/pareto_curve.png
  - results/pareto_raw.csv
  - results/cost_comparison.txt
  - results/falsifiable_claim.txt

Usage:
    python -m train.sweep [--seeds 3]
"""

from __future__ import annotations

import argparse
import os
import time
import csv
import numpy as np

from train.train_prune import (
    train_with_pruning,
    dense_forward_flops,
    sparse_forward_flops,
    build_model,
    get_linear_layers,
)
from train.train_base import load_data, evaluate


def run_sweep(
    sparsity_levels=(0.0, 0.50, 0.75, 0.90, 0.95),
    criteria=("taylor", "magnitude"),
    n_seeds: int = 3,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 128,
):
    """Run the full Pareto sweep.

    Returns
    -------
    results : list of dict
        One entry per (criterion, sparsity, seed) combination.
    """
    results = []
    base_seeds = list(range(42, 42 + n_seeds))

    total_runs = len(criteria) * len(sparsity_levels) * n_seeds
    run_idx = 0

    for criterion in criteria:
        for target_sparsity in sparsity_levels:
            for seed in base_seeds:
                run_idx += 1
                print(f"\n{'='*60}")
                print(f"Run {run_idx}/{total_runs}: "
                      f"criterion={criterion}, sparsity={target_sparsity:.0%}, seed={seed}")
                print(f"{'='*60}")

                if target_sparsity == 0.0:
                    # No pruning — use baseline training
                    from train.train_base import train as train_base
                    _, hist = train_base(
                        seed=seed, epochs=epochs, batch_size=batch_size,
                        lr=lr, hidden=hidden, save_plot=False,
                    )
                    # Compute dense FLOPs
                    model_tmp = build_model(hidden=hidden)
                    d_flops = dense_forward_flops(model_tmp)

                    results.append({
                        "criterion": criterion,
                        "target_sparsity": 0.0,
                        "achieved_sparsity": 0.0,
                        "seed": seed,
                        "test_acc": hist["test_acc"][-1],
                        "test_loss": hist["test_loss"][-1],
                        "active_params": model_tmp.layers[0].total_params +
                                         model_tmp.layers[2].total_params +
                                         model_tmp.layers[4].total_params,
                        "dense_flops": d_flops,
                        "sparse_flops": d_flops,
                    })
                else:
                    model, hist, pruner = train_with_pruning(
                        seed=seed, epochs=epochs, batch_size=batch_size,
                        lr=lr, hidden=hidden,
                        target_sparsity=target_sparsity,
                        criterion=criterion,
                        regrowth=True,
                        save_plot=False,
                    )
                    d_flops = dense_forward_flops(model)
                    s_flops = sparse_forward_flops(model, 64)

                    results.append({
                        "criterion": criterion,
                        "target_sparsity": target_sparsity,
                        "achieved_sparsity": pruner.get_total_sparsity(),
                        "seed": seed,
                        "test_acc": hist["test_acc"][-1],
                        "test_loss": hist["test_loss"][-1],
                        "active_params": pruner.get_active_params(),
                        "dense_flops": d_flops,
                        "sparse_flops": s_flops,
                    })

    return results


def save_results(results):
    """Save CSV, plots, cost analysis, and falsifiable claim."""
    os.makedirs("results", exist_ok=True)

    # --- CSV ---
    csv_path = "results/pareto_raw.csv"
    fieldnames = [
        "criterion", "target_sparsity", "achieved_sparsity",
        "seed", "test_acc", "test_loss",
        "active_params", "dense_flops", "sparse_flops",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nRaw data saved to {csv_path}")

    # --- aggregate by (criterion, target_sparsity) ---
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        key = (r["criterion"], r["target_sparsity"])
        agg[key].append(r)

    summary_lines = []
    summary_lines.append(
        f"{'Criterion':<12} {'Target%':>8} {'Achieved%':>10} "
        f"{'Acc Mean':>9} {'Acc Std':>8} {'FLOPs':>10} {'FLOP Red':>9}"
    )
    summary_lines.append("-" * 75)

    agg_data = {}
    for key in sorted(agg.keys()):
        entries = agg[key]
        accs = [e["test_acc"] for e in entries]
        achieved = [e["achieved_sparsity"] for e in entries]
        sparse_flops = [e["sparse_flops"] for e in entries]
        dense_flops = entries[0]["dense_flops"]
        flop_red = 1.0 - np.mean(sparse_flops) / dense_flops if dense_flops > 0 else 0

        agg_data[key] = {
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "achieved_mean": np.mean(achieved),
            "flop_mean": np.mean(sparse_flops),
            "flop_reduction": flop_red,
        }

        line = (f"{key[0]:<12} {key[1]:>7.0%} {np.mean(achieved):>10.1%} "
                f"{np.mean(accs):>9.4f} {np.std(accs):>8.4f} "
                f"{int(np.mean(sparse_flops)):>10} {flop_red:>9.1%}")
        summary_lines.append(line)

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    # --- Cost comparison ---
    cost_path = "results/cost_comparison.txt"
    with open(cost_path, "w") as f:
        f.write("COST COMPARISON: PRUNED vs DENSE\n")
        f.write("=" * 60 + "\n\n")
        f.write("We measure cost as forward-pass FLOPs (multiply-accumulate\n")
        f.write("operations) counting ONLY active (non-zero-mask) connections.\n")
        f.write("This is NOT a dense-times-zero shortcut — our sparse_forward_flops\n")
        f.write("function counts mask.sum() per layer, so a pruned connection\n")
        f.write("genuinely contributes zero FLOPs to the count.\n\n")
        f.write("However, our actual forward pass still uses dense matmul (NumPy\n")
        f.write("does not have sparse BLAS). The FLOP count is a *theoretical*\n")
        f.write("measure of work that a sparse-aware kernel WOULD save. In a\n")
        f.write("production system (e.g. AQUA), you would use a sparse GEMM\n")
        f.write("kernel or a compressed sparse row (CSR) representation to\n")
        f.write("realise these savings on real hardware.\n\n")
        f.write(summary_text + "\n")
    print(f"Cost comparison saved to {cost_path}")

    # --- Falsifiable claim ---
    # Compare taylor vs magnitude at 90% sparsity
    taylor_90 = agg_data.get(("taylor", 0.9))
    mag_90 = agg_data.get(("magnitude", 0.9))

    claim_path = "results/falsifiable_claim.txt"
    with open(claim_path, "w") as f:
        f.write("FALSIFIABLE CLAIM\n")
        f.write("=" * 60 + "\n\n")
        if taylor_90 and mag_90:
            n = len(agg[("taylor", 0.9)])
            f.write(
                f"At 90% target sparsity, Taylor-based pruning (|w * g|) retains\n"
                f"{taylor_90['acc_mean']:.2%} test accuracy (std={taylor_90['acc_std']:.4f})\n"
                f"versus {mag_90['acc_mean']:.2%} (std={mag_90['acc_std']:.4f}) for\n"
                f"magnitude pruning, averaged across {n} seeds.\n\n"
            )
            diff = taylor_90["acc_mean"] - mag_90["acc_mean"]
            if diff > 0:
                f.write(
                    f"Taylor pruning outperforms magnitude pruning by "
                    f"{diff:.2%} absolute accuracy.\n"
                )
            elif diff < 0:
                f.write(
                    f"Magnitude pruning outperforms Taylor pruning by "
                    f"{-diff:.2%} absolute accuracy at this sparsity level.\n"
                )
            else:
                f.write("The two criteria are statistically indistinguishable.\n")

            f.write(
                f"\nThis claim is reproducible by running:\n"
                f"  python -m train.sweep --seeds {n}\n"
                f"with the random seeds documented in results/pareto_raw.csv.\n"
            )
        else:
            f.write("Insufficient data for 90% sparsity comparison.\n")

    print(f"Falsifiable claim saved to {claim_path}")

    # --- Pareto plot ---
    _save_pareto_plot(agg, agg_data)


def _save_pareto_plot(agg, agg_data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        markers = {"taylor": "o", "magnitude": "s", "movement": "^"}
        colors = {"taylor": "#2196F3", "magnitude": "#FF5722", "movement": "#4CAF50"}

        for criterion in ["taylor", "magnitude"]:
            sparsities = []
            accs = []
            acc_stds = []
            for key in sorted(agg.keys()):
                if key[0] == criterion:
                    d = agg_data[key]
                    sparsities.append(d["achieved_mean"])
                    accs.append(d["acc_mean"])
                    acc_stds.append(d["acc_std"])

            ax1.errorbar(
                [s * 100 for s in sparsities],
                [a * 100 for a in accs],
                yerr=[s * 100 for s in acc_stds],
                marker=markers.get(criterion, "o"),
                color=colors.get(criterion, "gray"),
                label=criterion.capitalize(),
                capsize=4,
                linewidth=2,
                markersize=8,
            )

        ax1.set_xlabel("Sparsity (%)", fontsize=12)
        ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax1.set_title("Sparsity–Accuracy Pareto Curve", fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-5, 100)

        # FLOP reduction plot
        for criterion in ["taylor", "magnitude"]:
            sparsities = []
            flop_reds = []
            for key in sorted(agg.keys()):
                if key[0] == criterion:
                    d = agg_data[key]
                    sparsities.append(d["achieved_mean"])
                    flop_reds.append(d["flop_reduction"])

            ax2.plot(
                [s * 100 for s in sparsities],
                [f * 100 for f in flop_reds],
                marker=markers.get(criterion, "o"),
                color=colors.get(criterion, "gray"),
                label=criterion.capitalize(),
                linewidth=2,
                markersize=8,
            )

        ax2.set_xlabel("Sparsity (%)", fontsize=12)
        ax2.set_ylabel("FLOP Reduction (%)", fontsize=12)
        ax2.set_title("Sparsity vs. Compute Savings", fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-5, 100)

        plt.tight_layout()
        plt.savefig("results/pareto_curve.png", dpi=150)
        plt.close()
        print(f"Pareto plot saved to results/pareto_curve.png")
    except ImportError:
        print("matplotlib not available — skipping Pareto plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 4: Pareto sweep")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds per configuration")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    results = run_sweep(
        n_seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
    )
    save_results(results)
