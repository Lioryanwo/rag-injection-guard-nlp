"""
Grid-search tuning for the FULL defense reranking stage.

Designed for this project format:
    results/retrieval/<retriever>_attack_results.json

Expected input format:
    {
      "query_id_1": [doc1, doc2, ...],
      "query_id_2": [doc1, doc2, ...]
    }

This script:
- does NOT rebuild corpus/indexes
- does NOT regenerate attacks
- reads existing Top-20 retrieval results
- loads original questions from val_queries.jsonl
- loads qrels from val_qrels.json
- optionally restricts evaluation to attacked queries from spoof_chunks.jsonl
- reruns the FULL defense stage with different hyperparameters
- computes Recall@5/20 and spoof metrics using qrels/doc_id
- saves all_runs.csv and best_config.json

Example quick dry run:
python -m src.experiments.tune_defense \
  --input-path results/retrieval/minilm_attack_results.json \
  --queries-path data/processed/val_queries.jsonl \
  --qrels-path data/processed/val_qrels.json \
  --spoof-chunks-path data/processed/spoof_chunks.jsonl \
  --output-dir results/reverse_qa_tuning/minilm \
  --limit 25 \
  --reverse-qa-qg-backend heuristic \
  --no-reverse-qa-cross-encoder

Example real run:
python -m src.experiments.tune_defense \
  --input-path results/retrieval/minilm_attack_results.json \
  --queries-path data/processed/val_queries.jsonl \
  --qrels-path data/processed/val_qrels.json \
  --spoof-chunks-path data/processed/spoof_chunks.jsonl \
  --output-dir results/reverse_qa_tuning/minilm \
  --reverse-qa-qg-backend openai
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.defense.defense_filter import cross_encoder_rerank


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_retrieval_results(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load project retrieval results as {query_id: ranked_docs}.

    Supports the normal project format directly and also a few list-based
    formats for robustness.
    """
    obj = read_json(path)

    # Native project format: {qid: [docs...]}
    if isinstance(obj, dict) and all(isinstance(v, list) for v in obj.values()):
        return {str(qid): [dict(d) for d in docs] for qid, docs in obj.items()}

    # Wrapped format: {"results": {qid: [docs...]}}
    for key in ("results", "retrieved", "data"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            inner = obj[key]
            if all(isinstance(v, list) for v in inner.values()):
                return {str(qid): [dict(d) for d in docs] for qid, docs in inner.items()}

    # List format: [{query_id, retrieved/results/docs}]
    records: Optional[List[Dict[str, Any]]] = None
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        for key in ("queries", "records", "items"):
            if isinstance(obj.get(key), list):
                records = obj[key]
                break

    if records is not None:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for i, record in enumerate(records):
            qid = str(record.get("query_id") or record.get("qid") or record.get("id") or f"record_{i}")
            docs = []
            for docs_key in ("retrieved", "results", "docs", "documents", "candidates"):
                if isinstance(record.get(docs_key), list):
                    docs = [dict(d) for d in record[docs_key]]
                    break
            out[qid] = docs
        return out

    raise ValueError(
        f"Unsupported input format: {path}. Expected project format {qid: [docs]}"
    )


def load_queries(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in read_jsonl(path):
        qid = str(row.get("query_id") or row.get("id") or "")
        question = row.get("question") or row.get("query")
        if qid and isinstance(question, str):
            out[qid] = question
    return out


def load_qrels(path: Path) -> Dict[str, Set[str]]:
    obj = read_json(path)
    out: Dict[str, Set[str]] = {}
    for qid, rel in obj.items():
        if isinstance(rel, list):
            out[str(qid)] = {str(x) for x in rel}
        elif rel is not None:
            out[str(qid)] = {str(rel)}
    return out


def load_attacked_qids(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None or not path.exists():
        return None
    qids: Set[str] = set()
    for row in read_jsonl(path):
        qid = row.get("spoof_for_query")
        if qid:
            qids.add(str(qid))
    return qids or None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def is_spoof(doc: Dict[str, Any]) -> bool:
    return bool(
        doc.get("is_spoof", False)
        or doc.get("spoof", False)
        or doc.get("label") in {"spoof", "injected"}
        or doc.get("source") in {"spoof", "synthetic_attack"}
    )


def is_relevant(doc: Dict[str, Any], relevant_doc_ids: Set[str]) -> bool:
    doc_id = str(doc.get("doc_id", ""))
    if doc_id in relevant_doc_ids:
        return True

    # Some outputs preserve chunk_id but not doc_id, or chunk_id contains doc_id.
    chunk_id = str(doc.get("chunk_id", ""))
    return any(chunk_id.startswith(f"{rel}::") or chunk_id == rel for rel in relevant_doc_ids)


def compute_metrics(
    results: Dict[str, List[Dict[str, Any]]],
    qrels: Dict[str, Set[str]],
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
) -> Dict[str, Any]:
    qids = [qid for qid in results.keys() if qid in qrels]
    n = len(qids)
    if n == 0:
        return {
            "num_queries": 0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "top1_spoof_win_rate": 0.0,
            "avg_spoofs_in_top5": 0.0,
            "avg_spoofs_in_top20": 0.0,
            "spoof_in_top5_rate": 0.0,
            "spoof_in_top20_rate": 0.0,
            "avg_rank_of_first_spoof": None,
            "avg_rank_of_first_relevant": None,
        }

    recall_hits = {k: 0 for k in k_values}
    top1_spoof = 0
    spoofs_top5: List[float] = []
    spoofs_top20: List[float] = []
    spoof_in_top5 = 0
    spoof_in_top20 = 0
    first_spoof_ranks: List[int] = []
    first_rel_ranks: List[int] = []

    for qid in qids:
        ranked = results.get(qid, [])
        rel = qrels[qid]
        if ranked and is_spoof(ranked[0]):
            top1_spoof += 1

        for k in k_values:
            if any(is_relevant(doc, rel) for doc in ranked[:k]):
                recall_hits[k] += 1

        top5 = ranked[:5]
        top20 = ranked[:20]
        spoofs_top5.append(sum(1 for d in top5 if is_spoof(d)) / max(1, len(top5)))
        spoofs_top20.append(sum(1 for d in top20 if is_spoof(d)) / max(1, len(top20)))
        spoof_in_top5 += int(any(is_spoof(d) for d in top5))
        spoof_in_top20 += int(any(is_spoof(d) for d in top20))

        for i, doc in enumerate(ranked, 1):
            if is_spoof(doc):
                first_spoof_ranks.append(i)
                break
        for i, doc in enumerate(ranked, 1):
            if is_relevant(doc, rel):
                first_rel_ranks.append(i)
                break

    metrics: Dict[str, Any] = {"num_queries": n}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_hits[k] / n
    metrics.update({
        "top1_spoof_win_rate": top1_spoof / n,
        "avg_spoofs_in_top5": sum(spoofs_top5) / max(1, len(spoofs_top5)),
        "avg_spoofs_in_top20": sum(spoofs_top20) / max(1, len(spoofs_top20)),
        "spoof_in_top5_rate": spoof_in_top5 / n,
        "spoof_in_top20_rate": spoof_in_top20 / n,
        "avg_rank_of_first_spoof": (sum(first_spoof_ranks) / len(first_spoof_ranks)) if first_spoof_ranks else None,
        "avg_rank_of_first_relevant": (sum(first_rel_ranks) / len(first_rel_ranks)) if first_rel_ranks else None,
    })
    return metrics


def objective_score(metrics: Dict[str, Any]) -> float:
    """One scalar for choosing best config.

    Prioritize recall@5, then penalize spoof success. Recall@20 is included
    lightly so we do not choose configs that destroy the candidate pool.
    """
    return (
        1.00 * float(metrics.get("recall@5", 0.0))
        + 0.20 * float(metrics.get("recall@20", 0.0))
        - 0.60 * float(metrics.get("top1_spoof_win_rate", 0.0))
        - 0.30 * float(metrics.get("avg_spoofs_in_top5", 0.0))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Defense run
# ─────────────────────────────────────────────────────────────────────────────


def run_one_config(
    base_results: Dict[str, List[Dict[str, Any]]],
    queries: Dict[str, str],
    qids: Sequence[str],
    cross_encoder: CrossEncoder,
    doc2query_embedder: Optional[SentenceTransformer],
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    defended: Dict[str, List[Dict[str, Any]]] = {}

    for i, qid in enumerate(qids, 1):
        query = queries.get(qid, "")
        ranked = base_results.get(qid, [])
        if not query or not ranked:
            defended[qid] = ranked[: args.final_top_k]
            continue

        # Critical latency rule: defense sees only the already-retrieved Top-20.
        candidate_pool = [dict(d) for d in ranked[: args.pool_top_k]]

        reranked = cross_encoder_rerank(
            query=query,
            ranked=candidate_pool,
            cross_encoder=cross_encoder,
            threshold=cfg["suspicion_threshold"],
            semantic_weight=cfg["semantic_weight"],
            retrieval_weight=cfg["retrieval_weight"],
            doc2query_weight=cfg["doc2query_weight"],
            lexical_penalty_weight=cfg["lexical_penalty_weight"],
            batch_size=args.batch_size,
            doc2query_embedder=doc2query_embedder,
            doc2query_model_name=args.doc2query_embedding_model,
            use_reverse_qa=cfg["use_reverse_qa"],
            reverse_qa_weight=cfg["reverse_qa_weight"],
            reverse_qa_num_questions=args.reverse_qa_num_questions,
            reverse_qa_qg_backend=args.reverse_qa_qg_backend,
            reverse_qa_openai_model=args.reverse_qa_openai_model,
            reverse_qa_cache_path=str(Path(args.output_dir) / "reverse_qa_cache.jsonl"),
            reverse_qa_bm25_weight=cfg["reverse_qa_bm25_weight"],
            reverse_qa_cross_encoder_weight=cfg["reverse_qa_cross_encoder_weight"],
            reverse_qa_evidence_weight=cfg["reverse_qa_evidence_weight"],
            reverse_qa_cross_encoder_model=args.reverse_qa_cross_encoder_model,
        )

        defended[qid] = reranked[: args.final_top_k]

        if i % args.log_every == 0:
            print(f"    processed {i}/{len(qids)} queries")

    return defended


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def _plot_metric_by_reverse_weight(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    grouped = df.groupby("reverse_qa_weight", as_index=False)[metric].max()
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["reverse_qa_weight"], grouped[metric], marker="o")
    plt.xlabel("reverse_qa_weight")
    plt.ylabel(metric)
    plt.title(f"Best {metric} by ReverseQA weight")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_vs_reverse_qa_weight.png", dpi=180)
    plt.close()


def _plot_tradeoff(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(df["top1_spoof_win_rate"], df["recall@5"])
    plt.xlabel("Top1 Spoof Win Rate ↓")
    plt.ylabel("Recall@5 ↑")
    plt.title("Recall vs Spoof Tradeoff")
    plt.tight_layout()
    plt.savefig(output_dir / "recall_vs_spoof_tradeoff.png", dpi=180)
    plt.close()


def _plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    # Best objective for each ReverseQA x CE-weight pair.
    pivot = df.pivot_table(
        index="reverse_qa_weight",
        columns="reverse_qa_cross_encoder_weight",
        values="objective",
        aggfunc="max",
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label="objective")
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xlabel("reverse_qa_cross_encoder_weight")
    plt.ylabel("reverse_qa_weight")
    plt.title("Parameter heatmap: best objective")
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_heatmap.png", dpi=180)
    plt.close()


def save_outputs(
    rows: List[Dict[str, Any]],
    best_results: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(
        by=["objective", "recall@5", "top1_spoof_win_rate"],
        ascending=[False, False, True],
    )

    df.to_csv(output_dir / "all_runs.csv", index=False)
    df.to_json(output_dir / "all_runs.json", orient="records", indent=2, force_ascii=False)

    best = df.iloc[0].to_dict() if not df.empty else {}
    config_keys = [
        "suspicion_threshold", "semantic_weight", "retrieval_weight",
        "doc2query_weight", "lexical_penalty_weight", "use_reverse_qa",
        "reverse_qa_weight", "reverse_qa_bm25_weight",
        "reverse_qa_cross_encoder_weight", "reverse_qa_evidence_weight",
    ]
    best_config = {
        "best_config": {k: best.get(k) for k in config_keys if k in best},
        "best_metrics": {k: best.get(k) for k in best.keys() if k not in config_keys},
        "selection_objective": "recall@5 + 0.2*recall@20 - 0.6*top1_spoof_win_rate - 0.3*avg_spoofs_in_top5",
    }
    write_json(output_dir / "best_config.json", best_config)
    write_json(output_dir / "best_results.json", best_results)

    if not df.empty:
        _plot_metric_by_reverse_weight(df, "recall@5", output_dir)
        _plot_metric_by_reverse_weight(df, "top1_spoof_win_rate", output_dir)
        _plot_metric_by_reverse_weight(df, "avg_spoofs_in_top5", output_dir)
        _plot_tradeoff(df, output_dir)
        _plot_heatmap(df, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_float_list(text: Optional[str], default: List[float]) -> List[float]:
    if text is None:
        return default
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tune full RAG spoofing defense on existing Top-20 retrieval results.")

    ap.add_argument("--input-path", required=True, type=Path)
    ap.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    ap.add_argument("--qrels-path", type=Path, default=Path("data/processed/val_qrels.json"))
    ap.add_argument("--spoof-chunks-path", type=Path, default=Path("data/processed/spoof_chunks.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/reverse_qa_tuning"))

    ap.add_argument("--limit", type=int, default=0, help="Debug on first N evaluated queries; 0 = all")
    ap.add_argument("--pool-top-k", type=int, default=20, help="Defense candidate pool. Keep at 20 for latency.")
    ap.add_argument("--final-top-k", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=50)

    # Models
    ap.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--use-doc2query", action="store_true", default=True)
    ap.add_argument("--no-doc2query", action="store_false", dest="use_doc2query")
    ap.add_argument("--doc2query-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")

    # Reverse QA fixed options
    ap.add_argument("--disable-reverse-qa", action="store_true", help="Tune existing defense only, without Reverse QA.")
    ap.add_argument("--reverse-qa-num-questions", type=int, default=5)
    ap.add_argument("--reverse-qa-qg-backend", default="auto", choices=["auto", "openai", "transformers", "heuristic"])
    ap.add_argument("--reverse-qa-openai-model", default="gpt-4o-mini")
    ap.add_argument("--reverse-qa-cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--no-reverse-qa-cross-encoder", action="store_true", help="Set ReverseQA CE weight grid to 0 for faster dry run.")

    # Grid values as comma-separated strings for easier PowerShell usage.
    ap.add_argument("--suspicion-thresholds", default="0.20,0.30,0.40")
    ap.add_argument("--semantic-weights", default="0.55,0.65")
    ap.add_argument("--retrieval-weights", default="0.05,0.10,0.15")
    ap.add_argument("--doc2query-weights", default="0.20,0.30")
    ap.add_argument("--lexical-penalty-weights", default="0.05,0.10,0.15")

    ap.add_argument("--reverse-qa-weights", default="0.10,0.20,0.30,0.40,0.50")
    ap.add_argument("--reverse-qa-bm25-weights", default="0.20,0.35,0.50")
    ap.add_argument("--reverse-qa-cross-encoder-weights", default="0.40,0.55,0.70")
    ap.add_argument("--reverse-qa-evidence-weights", default="0.00,0.10,0.20")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading retrieval results...")
    base_results = load_retrieval_results(args.input_path)
    queries = load_queries(args.queries_path)
    qrels = load_qrels(args.qrels_path)
    attacked_qids = load_attacked_qids(args.spoof_chunks_path)

    qids = [qid for qid in base_results.keys() if qid in queries and qid in qrels]
    if attacked_qids:
        qids = [qid for qid in qids if qid in attacked_qids]
    if args.limit and args.limit > 0:
        qids = qids[: args.limit]

    if not qids:
        raise ValueError("No overlapping query_ids among input results, queries, qrels, and attacked_qids.")

    print(f"Evaluation queries: {len(qids)}")
    print(f"Pool Top-K: {args.pool_top_k} | Final Top-K: {args.final_top_k}")

    print(f"Loading main CrossEncoder: {args.cross_encoder_model}")
    cross_encoder = CrossEncoder(args.cross_encoder_model)

    doc2query_embedder: Optional[SentenceTransformer] = None
    if args.use_doc2query:
        print(f"Loading Doc2Query embedder: {args.doc2query_embedding_model}")
        doc2query_embedder = SentenceTransformer(args.doc2query_embedding_model)

    suspicion_thresholds = parse_float_list(args.suspicion_thresholds, [0.30])
    semantic_weights = parse_float_list(args.semantic_weights, [0.65])
    retrieval_weights = parse_float_list(args.retrieval_weights, [0.10])
    doc2query_weights = parse_float_list(args.doc2query_weights, [0.25])
    lexical_penalty_weights = parse_float_list(args.lexical_penalty_weights, [0.05])

    reverse_qa_weights = [0.0] if args.disable_reverse_qa else parse_float_list(args.reverse_qa_weights, [0.30])
    reverse_qa_bm25_weights = [0.0] if args.disable_reverse_qa else parse_float_list(args.reverse_qa_bm25_weights, [0.35])
    reverse_qa_ce_weights = [0.0] if (args.disable_reverse_qa or args.no_reverse_qa_cross_encoder) else parse_float_list(args.reverse_qa_cross_encoder_weights, [0.55])
    reverse_qa_evidence_weights = [0.0] if args.disable_reverse_qa else parse_float_list(args.reverse_qa_evidence_weights, [0.10])

    grid = list(itertools.product(
        suspicion_thresholds,
        semantic_weights,
        retrieval_weights,
        doc2query_weights,
        lexical_penalty_weights,
        reverse_qa_weights,
        reverse_qa_bm25_weights,
        reverse_qa_ce_weights,
        reverse_qa_evidence_weights,
    ))

    print(f"Grid size: {len(grid)} configurations")
    rows: List[Dict[str, Any]] = []
    best_objective = -math.inf
    best_results: Dict[str, List[Dict[str, Any]]] = {}

    for idx, values in enumerate(grid, 1):
        (
            suspicion_threshold,
            semantic_weight,
            retrieval_weight,
            doc2query_weight,
            lexical_penalty_weight,
            reverse_qa_weight,
            reverse_qa_bm25_weight,
            reverse_qa_cross_encoder_weight,
            reverse_qa_evidence_weight,
        ) = values

        cfg = {
            "suspicion_threshold": suspicion_threshold,
            "semantic_weight": semantic_weight,
            "retrieval_weight": retrieval_weight,
            "doc2query_weight": doc2query_weight,
            "lexical_penalty_weight": lexical_penalty_weight,
            "use_reverse_qa": not args.disable_reverse_qa,
            "reverse_qa_weight": reverse_qa_weight,
            "reverse_qa_bm25_weight": reverse_qa_bm25_weight,
            "reverse_qa_cross_encoder_weight": reverse_qa_cross_encoder_weight,
            "reverse_qa_evidence_weight": reverse_qa_evidence_weight,
        }

        print("\n" + "=" * 80)
        print(f"Config {idx}/{len(grid)}: {cfg}")

        defended = run_one_config(
            base_results=base_results,
            queries=queries,
            qids=qids,
            cross_encoder=cross_encoder,
            doc2query_embedder=doc2query_embedder,
            args=args,
            cfg=cfg,
        )
        metrics = compute_metrics(defended, qrels)
        obj = objective_score(metrics)
        row = {**cfg, **metrics, "objective": obj}
        rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))

        if obj > best_objective:
            best_objective = obj
            best_results = defended
            # Save checkpoint so long runs keep the current best.
            write_json(args.output_dir / "best_results.json", best_results)
            write_json(args.output_dir / "best_config_checkpoint.json", row)

    save_outputs(rows, best_results, args.output_dir)
    print("\nDONE")
    print(f"Saved all outputs to: {args.output_dir}")
    print("Best config:")
    print(json.dumps(read_json(args.output_dir / "best_config.json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
