from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Presentation style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        13,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  13,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

RETRIEVERS       = ["minilm", "bm25", "hybrid"]
RETRIEVER_LABELS = ["MiniLM\n(dense)", "BM25\n(sparse)", "Hybrid"]

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {
    "Clean Top-5":             "#2ecc71",
    "Attack Naive 5/20":       "#f39c12",
    "No-query filter":         "#95a5a6",
    "Query-aware defense":     "#2980b9",
    "Attack Top-20 (ceiling)": "#e74c3c",
    "Clean + Defense":         "#1a9850",
}

# ── File name mapping (matches actual pipeline output) ────────────────────────
METHOD_FILES = {
    "Clean Top-5":             "{ret}_baseline_metrics.json",
    "Attack Top-20 (ceiling)": "{ret}_attack_top20_metrics.json",
    "Attack Naive 5/20":       "{ret}_attack_naive_top5_metrics.json",
    "No-query filter":         "{ret}_no_query_metrics.json",
    "Query-aware defense":     "{ret}_defense_metrics.json",
    "Clean + Defense":         "{ret}_baseline_defense_metrics.json",
}

SPOOF_METHODS = {
    "Attack Top-20 (ceiling)",
    "Attack Naive 5/20",
    "No-query filter",
    "Query-aware defense",
}

RECALL_ORDER = [
    "Clean Top-5",
    "Attack Naive 5/20",
    "No-query filter",
    "Query-aware defense",
]

SPOOF_ORDER = [
    "Attack Top-20 (ceiling)",
    "Attack Naive 5/20",
    "No-query filter",
    "Query-aware defense",
]


def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"  [missing] {path.name}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bar_label(ax, bars, fontsize=11, pad=0.015):
    for bar in bars:
        h = bar.get_height()
        if h > 0.005:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + pad,
                f"{h:.2f}",
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold",
                color="#333333",
            )


def plot_recall(data: Dict[str, List], output: Path) -> None:
    methods  = RECALL_ORDER
    n        = len(methods)
    fig, ax  = plt.subplots(figsize=(12, 6))
    x        = np.arange(len(RETRIEVERS))
    total_w  = 0.72
    w        = total_w / n
    offsets  = np.linspace(-total_w / 2 + w / 2, total_w / 2 - w / 2, n)

    for off, method in zip(offsets, methods):
        vals = [v if v is not None else 0.0 for v in data[method]]
        bars = ax.bar(x + off, vals, width=w * 0.88,
                      color=COLORS[method], alpha=0.90, label=method, zorder=3)
        _bar_label(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(RETRIEVER_LABELS, fontsize=13)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Recall@5", fontsize=14)
    ax.set_title("Recall@5 Under Attack  —  300 queries", fontsize=16, pad=14)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11, ncol=2)
    for xi in x[1:]:
        ax.axvline(xi - 0.5, color="#cccccc", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output.name}")


def plot_spoof_win(data: Dict[str, List], output: Path) -> None:
    methods  = SPOOF_ORDER
    n        = len(methods)
    fig, ax  = plt.subplots(figsize=(12, 6))
    x        = np.arange(len(RETRIEVERS))
    total_w  = 0.72
    w        = total_w / n
    offsets  = np.linspace(-total_w / 2 + w / 2, total_w / 2 - w / 2, n)

    for off, method in zip(offsets, methods):
        vals = [v if v is not None else 0.0 for v in data[method]]
        bars = ax.bar(x + off, vals, width=w * 0.88,
                      color=COLORS[method], alpha=0.90, label=method, zorder=3)
        _bar_label(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(RETRIEVER_LABELS, fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Top-1 Spoof Win Rate  ↓ lower is better", fontsize=13)
    ax.set_title("Spoof Win Rate  —  300 attacked queries", fontsize=16, pad=14)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11, ncol=2)
    for xi in x[1:]:
        ax.axvline(xi - 0.5, color="#cccccc", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output.name}")


def plot_summary(recall_data: Dict, spoof_data: Dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("RAG Spoof Attack & Defense — Summary", fontsize=18, y=1.01)

    recall_methods = RECALL_ORDER
    spoof_methods  = SPOOF_ORDER
    short_r = ["Clean", "Naive 5/20", "No-query", "Defense"]
    short_s = ["Attack\n(ceiling)", "Naive 5/20", "No-query", "Defense"]

    for col, (ret, label) in enumerate(zip(RETRIEVERS, RETRIEVER_LABELS)):
        # Row 0: Recall
        ax = axes[0][col]
        vals   = [recall_data[m][col] or 0.0 for m in recall_methods]
        colors = [COLORS[m] for m in recall_methods]
        bars   = ax.bar(short_r, vals, color=colors, alpha=0.90, zorder=3, width=0.55)
        _bar_label(ax, bars, fontsize=10, pad=0.015)
        ax.set_ylim(0, 1.10)
        ax.set_title(f"{label}\nRecall@5", fontsize=13)
        ax.set_ylabel("Recall@5" if col == 0 else "", fontsize=12)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_axisbelow(True)

        # Row 1: Spoof win rate
        ax = axes[1][col]
        vals   = [spoof_data[m][col] or 0.0 for m in spoof_methods]
        colors = [COLORS[m] for m in spoof_methods]
        bars   = ax.bar(short_s, vals, color=colors, alpha=0.90, zorder=3, width=0.55)
        _bar_label(ax, bars, fontsize=10, pad=0.015)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{label}\nTop-1 Spoof Win Rate ↓", fontsize=13)
        ax.set_ylabel("Spoof Win Rate" if col == 0 else "", fontsize=12)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=COLORS["Clean Top-5"],             label="Clean baseline"),
        mpatches.Patch(color=COLORS["Attack Top-20 (ceiling)"], label="Attack (ceiling)"),
        mpatches.Patch(color=COLORS["Attack Naive 5/20"],       label="Naive 5/20"),
        mpatches.Patch(color=COLORS["No-query filter"],         label="No-query filter"),
        mpatches.Patch(color=COLORS["Query-aware defense"],     label="Query-aware defense"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(pad=2.5)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output.name}")


def plot_threshold_sweep(sweep_rows: List[Dict], output: Path) -> None:
    thresholds  = [r["threshold"] for r in sweep_rows]
    recall_vals = [r["recall@5"] for r in sweep_rows]
    spoof_vals  = [r["top1_spoof_win_rate"] for r in sweep_rows]
    chosen_th   = 0.30

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, recall_vals, marker="o", linewidth=2.5, markersize=8,
            color=COLORS["Query-aware defense"], label="Recall@5", zorder=4)
    ax.plot(thresholds, spoof_vals, marker="s", linewidth=2.5, markersize=8,
            color=COLORS["Attack Top-20 (ceiling)"], label="Top-1 Spoof Win Rate", zorder=4)
    ax.axvline(chosen_th, color="#555555", linewidth=2.0,
               linestyle="--", label=f"Chosen threshold = {chosen_th}", zorder=3)
    ax.axvspan(0.0, chosen_th, alpha=0.06,
               color=COLORS["Query-aware defense"], zorder=1)

    try:
        idx = thresholds.index(chosen_th)
        ax.annotate(f"Recall={recall_vals[idx]:.2f}",
                    xy=(chosen_th, recall_vals[idx]),
                    xytext=(chosen_th + 0.06, recall_vals[idx] + 0.04),
                    fontsize=11, color=COLORS["Query-aware defense"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["Query-aware defense"]))
        ax.annotate(f"Spoof={spoof_vals[idx]:.2f}",
                    xy=(chosen_th, spoof_vals[idx]),
                    xytext=(chosen_th + 0.06, spoof_vals[idx] - 0.06),
                    fontsize=11, color=COLORS["Attack Top-20 (ceiling)"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["Attack Top-20 (ceiling)"]))
    except ValueError:
        pass

    ax.set_xlabel("Suspicion Threshold", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_ylim(0, 1.10)
    ax.set_xlim(0.05, 0.95)
    ax.set_title("Defense Threshold Sweep\nRecall vs. Spoof Suppression Trade-off",
                 fontsize=15, pad=12)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/retrieval"))
    parser.add_argument("--output-dir",  type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics…")
    recall_data: Dict[str, List] = {m: [] for m in METHOD_FILES}
    spoof_data:  Dict[str, List] = {m: [] for m in METHOD_FILES if m in SPOOF_METHODS}

    for ret in RETRIEVERS:
        for method, tmpl in METHOD_FILES.items():
            m = _load(args.results_dir / tmpl.format(ret=ret))
            recall_data[method].append(m.get("recall@5")           if m else None)
            if method in spoof_data:
                spoof_data[method].append(m.get("top1_spoof_win_rate") if m else None)

    print("\nGenerating plots…")
    plot_recall(recall_data,  args.output_dir / "recall_under_attack.png")
    plot_spoof_win(spoof_data, args.output_dir / "spoof_win_rate.png")
    plot_summary(recall_data, spoof_data, args.output_dir / "summary.png")

    sweep = _load(args.results_dir / "minilm_threshold_sweep.json")
    if sweep and sweep.get("sweep"):
        plot_threshold_sweep(sweep["sweep"], args.output_dir / "threshold_sweep.png")
    else:
        print("  [skip] threshold sweep not found")

    # Console summary
    print("\n" + "=" * 65)
    print(f"{'Method':<28} {'MiniLM':>9} {'BM25':>9} {'Hybrid':>9}  Recall@5")
    print("-" * 65)
    for method in RECALL_ORDER:
        vals = recall_data[method]
        row  = f"{method:<28}"
        for v in vals:
            row += f"  {v:.3f}" if v is not None else "      —  "
        print(row)
    print()
    print(f"{'Method':<28} {'MiniLM':>9} {'BM25':>9} {'Hybrid':>9}  Spoof Win Rate")
    print("-" * 65)
    for method in SPOOF_ORDER:
        vals = spoof_data[method]
        row  = f"{method:<28}"
        for v in vals:
            row += f"  {v:.3f}" if v is not None else "      —  "
        print(row)
    print("=" * 65)


if __name__ == "__main__":
    main()
