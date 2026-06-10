from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils import get_logger

"""
=============================================================================
Purpose:
Aggregates evaluation metrics across all retrievers (MiniLM, BM25, Hybrid) 
and pipeline conditions (Clean, Attacked, Defense). 
Generates comprehensive comparative bar charts and a threshold sweep line plot 
to visualize the trade-offs between recall preservation and spoof suppression.

Inputs:
- --results-dir: Directory containing all `{ret}_{condition}_metrics.json` files.

Outputs:
- --output-dir: Directory where the generated .png plots will be saved.
  Also prints formatted ASCII summary tables to the standard output.
=============================================================================
"""

# Set up logger
script_name = Path(__file__).stem
folder_name = Path(__file__).parent.name
logger = get_logger(name=script_name, group=folder_name)

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
    "Baseline Top-5":          "#2ecc71", 
    "Baseline Top-20":         "#27ae60",
    "Attacked (No Defense)":   "#e74c3c", # אדום רגיל - ההתקפה על ה-Top 5
    "Attack Top-20 (ceiling)": "#c0392b", # אדום כהה - תקרת ההתקפה (יופיע רק ב-Recall)
    "No-query filter":         "#95a5a6",
    "CE Defense":              "#2980b9", 
    "Baseline + Defense":      "#1a9850",
}

# ── File mapping ──────────────────────────────────────────────────────────────
METHOD_FILES = {
    "Baseline Top-5":          "{ret}_baseline_metrics.json",
    "Baseline Top-20":         "{ret}_baseline_top20_metrics.json",
    "Attack Top-20 (ceiling)": "{ret}_attack_top20_metrics.json", # חזר! חייבים אותו ל-Recall
    "Attacked (No Defense)":   "{ret}_attack_naive_top5_metrics.json",
    "No-query filter":         "{ret}_no_query_metrics.json",
    "CE Defense":              "{ret}_defense_metrics.json",
    "Baseline + Defense":      "{ret}_baseline_defense_metrics.json",
}

RESULT_FILES = {
    "Attacked (No Defense)":   "{ret}_attack_naive_top5_results.json",
    "CE Defense":              "{ret}_defense_results.json",
}

SPOOF_METHODS = {
    "Attacked (No Defense)", 
    "No-query filter",
    "CE Defense",
}

RECALL_ORDER = [
    "Baseline Top-5",
    "Attacked (No Defense)",
    "No-query filter",
    "CE Defense",
]

SPOOF_ORDER = [
    "Attacked (No Defense)", 
    "No-query filter",
    "CE Defense",
]

OVERALL_SPOOF_ORDER = [
    "Attacked (No Defense)",
    "CE Defense",
]

RECALL_TOP20_VS_DEFENSE_ORDER = [
    "Baseline Top-20",
    "Attack Top-20 (ceiling)", 
    "CE Defense",
]


def _load(path: Path) -> Optional[Any]:
    if not path.exists():
        logger.warning(f"Missing expected metrics file: {path.name}")
        print(f"  [missing] {path.name}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _metric(m: Optional[Dict], *keys: str) -> Optional[float]:
    if not isinstance(m, dict):
        return None
    for k in keys:
        v = _safe_float(m.get(k))
        if v is not None:
            return v
    return None


def _overall_top1_spoof_from_results(results_path: Path) -> Optional[float]:
    """Compute Top-1 spoof rate over ALL queries from a results JSON file."""
    obj = _load(results_path)
    if not isinstance(obj, dict):
        return None

    total = 0
    spoof_top1 = 0
    for _qid, ranked in obj.items():
        if not isinstance(ranked, list) or not ranked:
            continue
        top = ranked[0]
        if not isinstance(top, dict):
            continue
        total += 1
        if bool(top.get("is_spoof", False)):
            spoof_top1 += 1

    return round(spoof_top1 / total, 4) if total else None


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


def _grouped_bar(data: Dict[str, List], methods: List[str], ylabel: str,
                 title: str, output: Path, ylim_top: float = 1.15) -> None:
    n       = len(methods)
    fig, ax = plt.subplots(figsize=(12, 6))
    x       = np.arange(len(RETRIEVERS))
    total_w = 0.72
    w       = total_w / n
    offsets = np.linspace(-total_w / 2 + w / 2, total_w / 2 - w / 2, n)

    for off, method in zip(offsets, methods):
        vals = [v if v is not None else 0.0 for v in data.get(method, [])]
        bars = ax.bar(
            x + off, vals, width=w * 0.88,
            color=COLORS[method], alpha=0.90, label=method, zorder=3,
        )
        _bar_label(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(RETRIEVER_LABELS, fontsize=13)
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16, pad=14)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11, ncol=2)
    for xi in x[1:]:
        ax.axvline(xi - 0.5, color="#cccccc", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output.name}")
    logger.info(f"Saved plot: {output.name}")


def plot_top20_vs_defense_recall(data: Dict[str, List], output: Path) -> None:
    _grouped_bar(
        data=data,
        methods=RECALL_TOP20_VS_DEFENSE_ORDER,
        ylabel="Recall",
        title="Recall Comparison: Top-20 vs Defense (Top-5/20)",
        output=output,
        ylim_top=1.10,
    )


def plot_recall(data: Dict[str, List], output: Path) -> None:
    _grouped_bar(
        data=data,
        methods=RECALL_ORDER,
        ylabel="Recall@5",
        title="Recall@5 Under Attack",
        output=output,
        ylim_top=1.10,
    )


def plot_spoof_win(data: Dict[str, List], output: Path) -> None:
    _grouped_bar(
        data=data,
        methods=SPOOF_ORDER,
        ylabel="Top-1 Spoof Win Rate on Attacked Queries ↓ lower is better",
        title="Spoof Win Rate — attacked queries only",
        output=output,
        ylim_top=1.15,
    )


def plot_overall_spoof_win(data: Dict[str, List], output: Path) -> None:
    _grouped_bar(
        data=data,
        methods=OVERALL_SPOOF_ORDER,
        ylabel="Overall Top-1 Spoof Rate ↓ lower is better",
        title="Overall Top-1 Spoof Rate — all evaluated queries",
        output=output,
        ylim_top=1.15,
    )


def plot_summary(recall_data: Dict, spoof_data: Dict, overall_spoof_data: Dict, output: Path) -> None:
    """
    Generates a 3x3 grid of subplots summarizing Recall@5, Attacked-query 
    Spoof Win Rate, and Overall Spoof Rate across all three retrievers.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.suptitle("RAG Spoof Attack & Reverse QA Defense — Summary", fontsize=18, y=1.01)

    
    recall_methods = RECALL_ORDER
    spoof_methods  = SPOOF_ORDER
    overall_methods = OVERALL_SPOOF_ORDER
    short_r = ["Baseline", "Attacked", "No-query", "Defense"]
    short_s = ["Attacked", "No-query", "Defense"]  
    short_o = ["Attacked", "Defense"]

    for col, (_ret, label) in enumerate(zip(RETRIEVERS, RETRIEVER_LABELS)):
        # Row 0: Recall
        ax = axes[0][col]
        vals   = [recall_data[m][col] or 0.0 for m in recall_methods]
        colors = [COLORS[m] for m in recall_methods]
        bars   = ax.bar(short_r, vals, color=colors, alpha=0.90, zorder=3, width=0.55)
        _bar_label(ax, bars, fontsize=10, pad=0.015)
        ax.set_ylim(0, 1.10)
        ax.set_title(f"{label}\nRecall@5", fontsize=13)
        ax.set_ylabel("Recall@5" if col == 0 else "", fontsize=12)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_axisbelow(True)

        # Row 1: attacked-query spoof win rate
        ax = axes[1][col]
        vals   = [spoof_data[m][col] or 0.0 for m in spoof_methods]
        colors = [COLORS[m] for m in spoof_methods]
        bars   = ax.bar(short_s, vals, color=colors, alpha=0.90, zorder=3, width=0.55)
        _bar_label(ax, bars, fontsize=10, pad=0.015)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{label}\nAttacked-query Spoof Win ↓", fontsize=13)
        ax.set_ylabel("Spoof Win" if col == 0 else "", fontsize=12)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_axisbelow(True)

        # Row 2: overall top1 spoof rate from result files
        ax = axes[2][col]
        vals   = [overall_spoof_data[m][col] or 0.0 for m in overall_methods]
        colors = [COLORS[m] for m in overall_methods]
        bars   = ax.bar(short_o, vals, color=colors, alpha=0.90, zorder=3, width=0.55)
        _bar_label(ax, bars, fontsize=10, pad=0.015)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{label}\nOverall Top-1 Spoof ↓", fontsize=13)
        ax.set_ylabel("Overall Spoof" if col == 0 else "", fontsize=12)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=COLORS["Baseline Top-5"],          label="Baseline (Safe)"),
        mpatches.Patch(color=COLORS["Attack Top-20 (ceiling)"], label="Attack Ceiling"),
        mpatches.Patch(color=COLORS["Attacked (No Defense)"],   label="Attacked (No Defense)"),
        mpatches.Patch(color=COLORS["No-query filter"],         label="No-query Filter"),
        mpatches.Patch(color=COLORS["CE Defense"],              label="Our Defense (CE+D2Q)"),
    ]

    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(pad=2.2)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output.name}")
    logger.info(f"Saved plot: {output.name}")


def plot_threshold_sweep(sweep_rows: List[Dict], output: Path) -> None:
    """
    Plots the trade-off curve between Recall@5 and Spoof Win Rate across 
    different suspicion thresholds to justify the chosen threshold value.
    """
    thresholds  = [r["threshold"] for r in sweep_rows]
    recall_vals = [r["recall@5"] for r in sweep_rows]
    spoof_vals  = [r["top1_spoof_win_rate"] for r in sweep_rows]
    chosen_th   = 0.30

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, recall_vals, marker="o", linewidth=2.5, markersize=8,
            color=COLORS["CE Defense"], label="Recall@5", zorder=4)
    ax.plot(thresholds, spoof_vals, marker="s", linewidth=2.5, markersize=8,
            color=COLORS["Attack Top-20 (ceiling)"], label="Top-1 Spoof Win Rate", zorder=4)
    ax.axvline(chosen_th, color="#555555", linewidth=2.0,
               linestyle="--", label=f"Chosen threshold = {chosen_th}", zorder=3)
    ax.axvspan(0.0, chosen_th, alpha=0.06,
               color=COLORS["CE Defense"], zorder=1)

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
    logger.info(f"Saved plot: {output.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/retrieval"))
    parser.add_argument("--output-dir",  type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metrics…")
    logger.info("Loading metrics from results directory...")
    recall_data: Dict[str, List[Optional[float]]] = {m: [] for m in METHOD_FILES}
    spoof_data:  Dict[str, List[Optional[float]]] = {m: [] for m in METHOD_FILES if m in SPOOF_METHODS}
    overall_spoof_data: Dict[str, List[Optional[float]]] = {m: [] for m in OVERALL_SPOOF_ORDER}

    for ret in RETRIEVERS:
        for method, tmpl in METHOD_FILES.items():
            m = _load(args.results_dir / tmpl.format(ret=ret))

            if "Top-20" in method:
                recall_data[method].append(_metric(m, "recall@20", "recall_at_20"))
            else:
                recall_data[method].append(_metric(m, "recall@5", "recall_at_5"))            
            
            if method in spoof_data:
                spoof_data[method].append(_metric(m, "top1_spoof_win_rate", "spoof_win"))

        # Overall Top-1 spoof rate is computed from result files, not from attacked-query metrics.
        for method in OVERALL_SPOOF_ORDER:
            tmpl = RESULT_FILES.get(method)
            if tmpl is None:
                overall_spoof_data[method].append(None)
                continue
            overall = _overall_top1_spoof_from_results(args.results_dir / tmpl.format(ret=ret))
            overall_spoof_data[method].append(overall)

    print("\nGenerating plots…")
    logger.info("Generating comparative plots...")
    plot_recall(recall_data, args.output_dir / "recall_under_attack.png")
    plot_spoof_win(spoof_data, args.output_dir / "spoof_win_rate_attacked_queries.png")
    # Backward-compatible filename: this now also uses attacked-query spoof metrics.
    plot_spoof_win(spoof_data, args.output_dir / "spoof_win_rate.png")
    plot_overall_spoof_win(overall_spoof_data, args.output_dir / "overall_top1_spoof_rate.png")
    plot_summary(recall_data, spoof_data, overall_spoof_data, args.output_dir / "summary.png")

    plot_top20_vs_defense_recall(recall_data, args.output_dir / "recall_top20_vs_defense_top5.png")

    sweep = _load(args.results_dir / "minilm_threshold_sweep.json")
    if sweep and isinstance(sweep, dict) and sweep.get("sweep"):
        plot_threshold_sweep(sweep["sweep"], args.output_dir / "threshold_sweep.png")
    else:
        logger.warning("Threshold sweep data not found. Skipping plot.")
        print("  [skip] threshold sweep not found")

    print("\n" + "=" * 76)
    print(f"{'Method':<28} {'MiniLM':>9} {'BM25':>9} {'Hybrid':>9}  Recall@5")
    print("-" * 76)
    for method in RECALL_ORDER:
        vals = recall_data[method]
        row  = f"{method:<28}"
        for v in vals:
            row += f"  {v:.3f}" if v is not None else "      —  "
        print(row)

    print("\n" + f"{'Method':<28} {'MiniLM':>9} {'BM25':>9} {'Hybrid':>9}  Attacked-query spoof win")
    print("-" * 76)
    for method in SPOOF_ORDER:
        vals = spoof_data[method]
        row  = f"{method:<28}"
        for v in vals:
            row += f"  {v:.3f}" if v is not None else "      —  "
        print(row)

    print("\n" + f"{'Method':<28} {'MiniLM':>9} {'BM25':>9} {'Hybrid':>9}  Overall Top-1 spoof")
    print("-" * 76)
    for method in OVERALL_SPOOF_ORDER:
        vals = overall_spoof_data[method]
        row  = f"{method:<28}"
        for v in vals:
            row += f"  {v:.3f}" if v is not None else "      —  "
        print(row)
    print("=" * 76)


if __name__ == "__main__":
    main()
