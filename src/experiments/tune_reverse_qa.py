from __future__ import annotations

"""
src/experiments/tune_reverse_qa.py

Grid-search runner for the Reverse QA defense layer.

Important design choice:
- This script does NOT rebuild the corpus.
- It does NOT rebuild indexes.
- It does NOT regenerate spoof chunks.
- It reuses an existing retrieval-results JSON, reruns only defense_filter,
  then evaluates the defended output.

Typical use:

python -m src.experiments.tune_reverse_qa `
  --input-path results/retrieval/minilm_attack_results.json `
  --queries-path data/processed/val_queries.jsonl `
  --qrels-path data/processed/val_qrels.json `
  --spoof-chunks-path data/processed/spoof_chunks.jsonl `
  --output-dir results/reverse_qa_tuning/minilm `
  --qg-backend heuristic
"""

import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # plotting should never break the tuning run
    plt = None


ROOT = Path(__file__).resolve().parents[2]


def _parse_float_list(values: str) -> List[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run(cmd: List[str]) -> None:
    print("\n" + "=" * 88)
    print("RUN:", " ".join(cmd))
    print("=" * 88)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _metric_score(metrics: Dict[str, Any]) -> float:
    """
    Single scalar for selecting the best config.

    Higher is better:
      + Recall@5 is rewarded.
      - Top1 spoof win is penalized heavily.
      - Avg spoofs in Top5 is also penalized.
    """
    recall5 = float(metrics.get("recall@5", 0.0))
    spoof1 = float(metrics.get("top1_spoof_win_rate", 1.0))
    avg_spoof5 = float(metrics.get("avg_spoofs_in_top5", 1.0))
    return recall5 - 0.70 * spoof1 - 0.30 * avg_spoof5


def _plot_line(rows: List[Dict[str, Any]], x_key: str, y_key: str, output: Path, title: str, ylabel: str) -> None:
    if plt is None or not rows:
        return
    grouped: Dict[float, List[float]] = {}
    for row in rows:
        x = float(row[x_key])
        y = float(row[y_key])
        grouped.setdefault(x, []).append(y)
    xs = sorted(grouped)
    ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff(rows: List[Dict[str, Any]], output: Path) -> None:
    if plt is None or not rows:
        return
    x = [float(r["top1_spoof_win_rate"]) for r in rows]
    y = [float(r["recall@5"]) for r in rows]
    labels = [float(r["reverse_qa_weight"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(x, y, c=labels)
    ax.set_xlabel("Top1 Spoof Win Rate ↓")
    ax.set_ylabel("Recall@5 ↑")
    ax.set_title("Recall vs Spoof Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="ReverseQAWeight")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(rows: List[Dict[str, Any]], output: Path) -> None:
    if plt is None or not rows:
        return
    xs = sorted({float(r["bm25_weight"]) for r in rows})
    ys = sorted({float(r["cross_encoder_weight"]) for r in rows})
    grid = [[0.0 for _ in xs] for _ in ys]
    counts = [[0 for _ in xs] for _ in ys]
    xidx = {v: i for i, v in enumerate(xs)}
    yidx = {v: i for i, v in enumerate(ys)}
    for r in rows:
        xi = xidx[float(r["bm25_weight"])]
        yi = yidx[float(r["cross_encoder_weight"])]
        grid[yi][xi] += float(r["selection_score"])
        counts[yi][xi] += 1
    for yi in range(len(ys)):
        for xi in range(len(xs)):
            if counts[yi][xi]:
                grid[yi][xi] /= counts[yi][xi]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys)
    ax.set_xlabel("BM25 weight")
    ax.set_ylabel("CrossEncoder weight")
    ax.set_title("Parameter Heatmap — average selection score")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Reverse QA defense without regenerating data")

    parser.add_argument("--input-path", type=Path, required=True,
                        help="Existing Top-20 attack retrieval results JSON")
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--qrels-path", type=Path, default=Path("data/processed/val_qrels.json"))
    parser.add_argument("--spoof-chunks-path", type=Path, default=Path("data/processed/spoof_chunks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/reverse_qa_tuning"))

    parser.add_argument("--defense-mode", default="cross_encoder", choices=["cross_encoder", "text", "no_query"])
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--qg-backend", default="heuristic", choices=["heuristic", "openai", "transformers", "auto"])
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--keep-top-k", type=int, default=5)
    parser.add_argument("--suspicion-threshold", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=16)

    # Grid values as comma-separated lists for easy PowerShell usage.
    parser.add_argument("--reverse-qa-weights", default="0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--bm25-weights", default="0.20,0.40,0.60,0.80")
    parser.add_argument("--ce-weights", default="0.20,0.40,0.60,0.80")

    # Existing defense weights. Defaults keep the current defense unchanged.
    parser.add_argument("--semantic-weights", default="0.55")
    parser.add_argument("--retrieval-weights", default="0.15")
    parser.add_argument("--doc2query-weights", default="0.30")
    parser.add_argument("--lexical-penalty-weights", default="0.15")
    parser.add_argument("--use-doc2query", action="store_true")
    parser.add_argument("--doc2query-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")

    parser.add_argument("--max-runs", type=int, default=0,
                        help="Optional cap for quick smoke tests. 0 = run full grid.")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rqa_weights = _parse_float_list(args.reverse_qa_weights)
    bm25_weights = _parse_float_list(args.bm25_weights)
    ce_weights = _parse_float_list(args.ce_weights)
    semantic_weights = _parse_float_list(args.semantic_weights)
    retrieval_weights = _parse_float_list(args.retrieval_weights)
    doc2query_weights = _parse_float_list(args.doc2query_weights)
    lexical_penalty_weights = _parse_float_list(args.lexical_penalty_weights)

    rows: List[Dict[str, Any]] = []
    run_id = 0
    py = sys.executable

    grid = product(
        rqa_weights,
        bm25_weights,
        ce_weights,
        semantic_weights,
        retrieval_weights,
        doc2query_weights,
        lexical_penalty_weights,
    )

    for rqa_w, bm25_w, ce_w, sem_w, ret_w, d2q_w, lex_w in grid:
        # Normalize only the Reverse-QA internal mixture; defense weights remain as configured.
        ev_w = max(0.0, 1.0 - bm25_w - ce_w)
        if bm25_w + ce_w > 1.0:
            continue

        run_id += 1
        if args.max_runs and run_id > args.max_runs:
            break

        run_dir = args.output_dir / f"run_{run_id:04d}"
        defense_output = run_dir / "defense_results.json"
        metrics_output = run_dir / "metrics.json"
        cache_path = args.output_dir / "reverse_qa_cache.jsonl"

        defense_cmd = [
            py, "-m", "src.defense.defense_filter",
            "--input-path", str(args.input_path),
            "--queries-path", str(args.queries_path),
            "--output-path", str(defense_output),
            "--keep-top-k", str(args.keep_top_k),
            "--suspicion-threshold", str(args.suspicion_threshold),
            "--defense-mode", args.defense_mode,
            "--cross-encoder-model", args.cross_encoder_model,
            "--semantic-weight", str(sem_w),
            "--retrieval-weight", str(ret_w),
            "--doc2query-weight", str(d2q_w),
            "--lexical-penalty-weight", str(lex_w),
            "--batch-size", str(args.batch_size),
            "--use-reverse-qa",
            "--reverse-qa-weight", str(rqa_w),
            "--reverse-qa-bm25-weight", str(bm25_w),
            "--reverse-qa-cross-encoder-weight", str(ce_w),
            "--reverse-qa-evidence-weight", str(ev_w),
            "--reverse-qa-qg-backend", args.qg_backend,
            "--reverse-qa-openai-model", args.openai_model,
            "--reverse-qa-cache-path", str(cache_path),
        ]
        if args.use_doc2query:
            defense_cmd.extend([
                "--use-doc2query",
                "--doc2query-embedding-model", args.doc2query_embedding_model,
            ])

        eval_cmd = [
            py, "-m", "src.evaluation.evaluate_retrieval",
            "--results-path", str(defense_output),
            "--qrels-path", str(args.qrels_path),
            "--output-path", str(metrics_output),
            "--spoof-chunks-path", str(args.spoof_chunks_path),
        ]

        print(f"\n[run {run_id}] rqa={rqa_w} bm25={bm25_w} ce={ce_w} ev={ev_w} sem={sem_w} ret={ret_w} d2q={d2q_w} lex={lex_w}")
        if args.dry_run:
            print("DRY RUN defense:", " ".join(defense_cmd))
            print("DRY RUN eval:", " ".join(eval_cmd))
            continue

        _run(defense_cmd)
        _run(eval_cmd)
        metrics = _read_json(metrics_output)
        selection_score = _metric_score(metrics)

        row = {
            "run_id": run_id,
            "reverse_qa_weight": rqa_w,
            "bm25_weight": bm25_w,
            "cross_encoder_weight": ce_w,
            "evidence_weight": ev_w,
            "semantic_weight": sem_w,
            "retrieval_weight": ret_w,
            "doc2query_weight": d2q_w,
            "lexical_penalty_weight": lex_w,
            "recall@5": metrics.get("recall@5"),
            "recall@20": metrics.get("recall@20"),
            "top1_spoof_win_rate": metrics.get("top1_spoof_win_rate"),
            "avg_spoofs_in_top5": metrics.get("avg_spoofs_in_top5"),
            "avg_spoofs_in_top20": metrics.get("avg_spoofs_in_top20"),
            "selection_score": selection_score,
            "defense_results_path": str(defense_output),
            "metrics_path": str(metrics_output),
        }
        rows.append(row)

        # Save after every run so interrupted tuning still leaves useful data.
        csv_path = args.output_dir / "all_runs.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)

        best = max(rows, key=lambda r: float(r["selection_score"]))
        _write_json(args.output_dir / "best_config.json", best)
        print("Current best:", json.dumps(best, indent=2, ensure_ascii=False))

    if not args.dry_run and rows:
        _plot_line(rows, "reverse_qa_weight", "recall@5", args.output_dir / "recall5_vs_reverse_qa_weight.png",
                   "Recall@5 vs ReverseQAWeight", "Recall@5")
        _plot_line(rows, "reverse_qa_weight", "top1_spoof_win_rate", args.output_dir / "spoofwin_vs_reverse_qa_weight.png",
                   "Spoof Win Rate vs ReverseQAWeight", "Top1 Spoof Win Rate")
        _plot_tradeoff(rows, args.output_dir / "recall_vs_spoof_tradeoff.png")
        _plot_heatmap(rows, args.output_dir / "parameter_heatmap.png")
        print(f"\nDone. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
