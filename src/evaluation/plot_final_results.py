from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

METHODS = [
    "Attack\nBaseline",
    "Filter\nOnly",
    "Filter +\nCrossEncoder",
    "Filter + CE\n+ Evidence",
]

COLORS = [
    "#e74c3c",
    "#95a5a6",
    "#2980b9",
    "#8e44ad",
]


def add_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2,
            h + 0.015,
            f"{h:.3f}",
            ha="center",
            fontsize=11,
            fontweight="bold"
        )


def plot_metric(values, title, ylabel, output_path, ylim=(0,1.1)):
    fig, ax = plt.subplots(figsize=(8,5))

    x = np.arange(len(METHODS))

    bars = ax.bar(
        x,
        values,
        color=COLORS,
        width=0.65,
        alpha=0.9,
        zorder=3,
    )

    add_labels(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(*ylim)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/defense/multistage_defense_metrics.json")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures")
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.metrics.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    A = metrics["conditions"]["A"]
    B = metrics["conditions"]["B"]
    C = metrics["conditions"]["C"]
    D = metrics["conditions"]["D"]

    recall5 = [
        A["recall@5"],
        B["recall@5"],
        C["recall@5"],
        D["recall@5"],
    ]

    recall20 = [
        A["recall@20"],
        B["recall@20"],
        C["recall@20"],
        D["recall@20"],
    ]

    spoofwin = [
        A["top1_spoof_win_rate"],
        B["top1_spoof_win_rate"],
        C["top1_spoof_win_rate"],
        D["top1_spoof_win_rate"],
    ]

    goldpool = [
        A["real_in_pool_pct"],
        B["real_in_pool_pct"],
        C["real_in_pool_pct"],
        D["real_in_pool_pct"],
    ]

    print("\nGenerating plots...\n")

    plot_metric(
        recall5,
        "Defense Comparison — Recall@5",
        "Recall@5",
        args.output_dir / "final_recall5.png"
    )

    plot_metric(
        recall20,
        "Defense Comparison — Recall@20",
        "Recall@20",
        args.output_dir / "final_recall20.png"
    )

    plot_metric(
        spoofwin,
        "Defense Comparison — Top1 Spoof Win Rate",
        "Spoof Win Rate",
        args.output_dir / "final_spoofwin.png"
    )

    plot_metric(
        goldpool,
        "Defense Comparison — Gold Chunk In Pool",
        "GoldPool",
        args.output_dir / "final_goldpool.png"
    )

    print("\nDone.")
    print(f"Graphs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()