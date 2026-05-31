"""
Grid-search hyperparameter tuning for the defense reranking stage.

This script is designed to run on existing retrieval/attack result files. It does
not rebuild the corpus, regenerate embeddings, or rerun attack generation.

Expected flexible input formats
-------------------------------
A JSON file containing either:
1) list of query records:
   {"query": "...", "results": [{"text": "...", "score": 0.9, "is_spoof": false, "is_relevant": true}, ...]}
2) dict with a top-level records key: "queries", "records", "data", or "results".

Run example
-----------
python -m src.experiments.tune_defense \
  --input results/retrieval/attack_results.json \
  --output-dir results/tuning_reverse_qa \
  --no-cross-encoder \
  --qg-backend heuristic

For the real run, remove --no-cross-encoder and set qg-backend auto/openai.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.defense.reverse_qa import ReverseQAConfig, rerank_with_reverse_qa


def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    obj = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    for key in ("queries", "records", "data", "results"):
        if isinstance(obj.get(key), list):
            return obj[key]
    raise ValueError(f"Unsupported input format: {path}")


def get_docs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("results", "retrieved", "docs", "documents", "candidates"):
        val = record.get(key)
        if isinstance(val, list):
            return [dict(x) for x in val]
    return []


def get_query(record: Dict[str, Any]) -> str:
    for key in ("query", "question", "user_query"):
        val = record.get(key)
        if isinstance(val, str):
            return val
    return ""


def is_spoof(doc: Dict[str, Any]) -> bool:
    return bool(doc.get("is_spoof") or doc.get("spoof") or doc.get("label") == "spoof" or doc.get("source") == "spoof")


def is_relevant(doc: Dict[str, Any]) -> bool:
    return bool(doc.get("is_relevant") or doc.get("relevant") or doc.get("gold") or doc.get("contains_answer"))


def compute_metrics(reranked_records: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, float]:
    n = 0
    recall_hits = 0
    spoof_top1 = 0
    spoof_in_topk = 0
    avg_spoof_rank_sum = 0.0
    avg_spoof_rank_count = 0

    for record in reranked_records:
        docs = get_docs(record)
        if not docs:
            continue
        n += 1
        top = docs[:top_k]
        if any(is_relevant(d) for d in top):
            recall_hits += 1
        if is_spoof(docs[0]):
            spoof_top1 += 1
        if any(is_spoof(d) for d in top):
            spoof_in_topk += 1
        for i, d in enumerate(docs, start=1):
            if is_spoof(d):
                avg_spoof_rank_sum += i
                avg_spoof_rank_count += 1
                break

    denom = max(1, n)
    return {
        "num_queries": float(n),
        "recall_at_5": recall_hits / denom,
        "top1_spoof_win_rate": spoof_top1 / denom,
        "spoof_in_top5_rate": spoof_in_topk / denom,
        "avg_first_spoof_rank": avg_spoof_rank_sum / max(1, avg_spoof_rank_count),
    }


def run_grid(records: List[Dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    grid = list(itertools.product(
        args.reverse_qa_weight,
        args.bm25_weight,
        args.cross_encoder_weight,
    ))

    for reverse_qa_weight, bm25_weight, cross_encoder_weight in grid:
        cfg = ReverseQAConfig(
            top_k=args.top_k,
            num_questions=args.num_questions,
            reverse_qa_weight=reverse_qa_weight,
            bm25_weight=bm25_weight,
            cross_encoder_weight=cross_encoder_weight,
            qg_backend=args.qg_backend,
            use_cross_encoder=not args.no_cross_encoder,
            cache_path=str(Path(args.output_dir) / "reverse_qa_cache.jsonl"),
            base_score_key=args.base_score_key,
        )

        reranked_records: List[Dict[str, Any]] = []
        for record in records[: args.limit if args.limit else None]:
            query = get_query(record)
            docs = get_docs(record)
            reranked_docs = rerank_with_reverse_qa(query, docs, cfg) if query and docs else docs
            new_record = dict(record)
            # Preserve the most common result key name.
            result_key = "results" if "results" in record else "retrieved"
            new_record[result_key] = reranked_docs
            reranked_records.append(new_record)

        metrics = compute_metrics(reranked_records, top_k=5)
        rows.append({
            "reverse_qa_weight": reverse_qa_weight,
            "bm25_weight": bm25_weight,
            "cross_encoder_weight": cross_encoder_weight,
            **metrics,
        })
        print(rows[-1])

    return pd.DataFrame(rows).sort_values(
        by=["recall_at_5", "top1_spoof_win_rate"], ascending=[False, True]
    )


def plot_results(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "tuning_results.csv", index=False)
    df.head(20).to_json(output_dir / "top20_tuning_results.json", orient="records", indent=2)

    for metric in ["recall_at_5", "top1_spoof_win_rate", "spoof_in_top5_rate", "avg_first_spoof_rank"]:
        plt.figure(figsize=(10, 5))
        labels = [f"rq={r.reverse_qa_weight},bm25={r.bm25_weight},ce={r.cross_encoder_weight}" for r in df.itertuples()]
        plt.bar(range(len(df)), df[metric])
        plt.xticks(range(len(df)), labels, rotation=90, fontsize=7)
        plt.ylabel(metric)
        plt.title(f"Defense tuning: {metric}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=180)
        plt.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="results/tuning_reverse_qa")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--num-questions", type=int, default=5)
    ap.add_argument("--qg-backend", default="auto", choices=["auto", "openai", "transformers", "heuristic"])
    ap.add_argument("--no-cross-encoder", action="store_true")
    ap.add_argument("--base-score-key", default="defense_score")
    ap.add_argument("--limit", type=int, default=0, help="Debug on first N queries; 0 = all")
    ap.add_argument("--reverse-qa-weight", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    ap.add_argument("--bm25-weight", type=float, nargs="+", default=[0.30, 0.50, 0.70])
    ap.add_argument("--cross-encoder-weight", type=float, nargs="+", default=[0.30, 0.50, 0.70])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.input)
    df = run_grid(records, args)
    plot_results(df, output_dir)
    print("\nBest configuration:")
    print(df.head(1).to_string(index=False))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
