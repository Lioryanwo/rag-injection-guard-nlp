from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from src.utils import get_logger

"""
=============================================================================
Purpose:
Computes Information Retrieval (IR) and attack-success metrics for a given set 
of retrieval results. It calculates traditional metrics (Recall@K) alongside 
security metrics (Spoof Win Rate, Attack Coverage).

Critically, it enforces a fair comparison by optionally 
restricting the evaluation strictly to the subset of queries that were targeted 
by an attack, ensuring apples-to-apples comparisons across different pipeline runs.

Inputs:
- --results-path: JSON file containing the retrieved chunks to evaluate.
- --qrels-path: JSON file containing the gold-standard document IDs (ground truth).
- --spoof-chunks-path: (Optional) JSONL file of generated spoofs, used to filter 
                       the evaluated queries.

Outputs:
- --output-path: JSON file containing the aggregated numerical metrics.
=============================================================================
"""

# Set up logger
script_name = Path(__file__).stem
folder_name = Path(__file__).parent.name
logger = get_logger(name=script_name, group=folder_name)


# --- helper functions ---
def _load(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# --- evaluation metrics ---
def _attacked_qids_from_spoofs(path: Optional[Path]) -> Optional[Set[str]]:
    """Read the set of query-ids that have at least one generated spoof chunk."""
    if path is None:
        return None
    qids: Set[str] = set()
    for row in _read_jsonl(path):
        qid = row.get("spoof_for_query")
        if qid:
            qids.add(str(qid))
    return qids


def _filter_to_qids(d: Dict, qids: Optional[Set[str]]) -> Dict:
    """Keep only entries whose key is in qids (no-op when qids is None)."""
    if not qids:
        return d
    return {qid: value for qid, value in d.items() if qid in qids}


def recall_at_k(results: Dict, qrels: Dict, k: int) -> float:
    """Calculates the percentage of queries where the gold document appears in the Top-K retrieved results."""
    hits, total = 0, 0
    for qid, rel in qrels.items():
        ranked = results.get(qid, [])
        if not ranked:
            continue
        total += 1
        retrieved_docs = {item["doc_id"] for item in ranked[:k]}
        if any(d in retrieved_docs for d in rel):
            hits += 1
    return hits / total if total else 0.0


def top1_spoof_win_rate(results: Dict) -> float:
    """Calculates the percentage of queries where an attacker-generated spoof took the #1 rank."""
    total = wins = 0
    for ranked in results.values():
        if not ranked:
            continue
        total += 1
        if ranked[0].get("is_spoof", False):
            wins += 1
    return wins / total if total else 0.0


def avg_spoofs_in_top_k(results: Dict, k: int) -> float:
    """Calculates the average proportion of spoofed documents occupying the Top-K slots across all queries."""
    vals = []
    for ranked in results.values():
        topk = ranked[:k]
        if topk:
            vals.append(sum(1 for x in topk if x.get("is_spoof", False)) / len(topk))
    return sum(vals) / len(vals) if vals else 0.0


def avg_rank_of_first_spoof(results: Dict):
    """Calculates the average rank position of the first spoofed document across all queries that contain at least one spoof."""
    ranks = []
    for ranked in results.values():
        for i, item in enumerate(ranked, 1):
            if item.get("is_spoof", False):
                ranks.append(i)
                break
    return sum(ranks) / len(ranks) if ranks else None


# --- main evaluation logic ---
def attack_type_breakdown(results: Dict) -> Dict:
    stats: Dict[str, Any] = defaultdict(lambda: {
        "top1_wins": 0, "top3": 0, "top5": 0, "top20": 0,
        "total": 0, "queries": set(),
    })
    for qid, ranked in results.items():
        for i, item in enumerate(ranked, 1):
            if not item.get("is_spoof", False):
                continue
            at = item.get("attack_type", "unknown")
            stats[at]["total"] += 1
            stats[at]["queries"].add(qid)
            if i == 1:  stats[at]["top1_wins"] += 1
            if i <= 3:  stats[at]["top3"] += 1
            if i <= 5:  stats[at]["top5"] += 1
            if i <= 20: stats[at]["top20"] += 1
    return {
        at: {**{k: v for k, v in d.items() if k != "queries"},
             "queries_affected": len(d["queries"])}
        for at, d in stats.items()
    }


def query_attack_coverage(results: Dict) -> Dict:
    total = affected1 = affected3 = affected5 = affected20 = 0
    for ranked in results.values():
        if not ranked:
            continue
        total += 1
        if any(x.get("is_spoof") for x in ranked[:1]):  affected1  += 1
        if any(x.get("is_spoof") for x in ranked[:3]):  affected3  += 1
        if any(x.get("is_spoof") for x in ranked[:5]):  affected5  += 1
        if any(x.get("is_spoof") for x in ranked[:20]): affected20 += 1
    return {
        "top1":  affected1  / total if total else 0.0,
        "top3":  affected3  / total if total else 0.0,
        "top5":  affected5  / total if total else 0.0,
        "top20": affected20 / total if total else 0.0,
    }


def main() -> None:
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--qrels-path",   type=Path, default=Path("data/processed/val_qrels.json"))
    parser.add_argument("--output-path",  type=Path, required=True)
    parser.add_argument(
        "--spoof-chunks-path", type=Path, default=None,
        help=(
            "If provided, restrict evaluation to the 300 queries that have spoof chunks. "
            "IMPORTANT: this filters BOTH results AND qrels to the same query set, "
            "so recall is computed correctly on the restricted set. "
            "Sasha's requirement: all conditions evaluated on the same N=300 queries."
        ),
    )
    args = parser.parse_args()

    # --- Load data ---
    results = _load(args.results_path)
    qrels   = _load(args.qrels_path)
    logger.info(f"Loaded {len(results)} queries from results and {len(qrels)} queries from qrels.")

    attacked_qids = _attacked_qids_from_spoofs(args.spoof_chunks_path)

    # Enforce fair comparison by restricting to attacked queries if spoof chunks are provided 
    if attacked_qids:
        logger.info(f"Filtering evaluation to the {len(attacked_qids)} attacked queries to ensure fair comparison.") 
        qrels = _filter_to_qids(qrels, attacked_qids)
        results = _filter_to_qids(results, set(qrels.keys()))

        logger.info(f"After filtering, evaluating exactly {len(results)} queries.")
    else:
        logger.info("No spoof chunks provided. Evaluating all available queries in results.")
    # ──────────────────────────────────────────────────────────────────────────

    pool_size = max((len(v) for v in results.values()), default=0)

    metrics = {
        "num_queries":         len(results),
        "evaluation_scope":    "attacked_queries_only" if attacked_qids else "all_queries",
        "retrieval_pool_size": pool_size,

        "recall@1":  recall_at_k(results, qrels, 1),
        "recall@3":  recall_at_k(results, qrels, 3),
        "recall@5":  recall_at_k(results, qrels, 5),
        "recall@10": recall_at_k(results, qrels, 10),
        "recall@20": recall_at_k(results, qrels, 20),

        "top1_spoof_win_rate":     top1_spoof_win_rate(results),
        "avg_spoofs_in_top5":      avg_spoofs_in_top_k(results, 5),
        "avg_spoofs_in_top20":     avg_spoofs_in_top_k(results, 20),
        "avg_rank_of_first_spoof": avg_rank_of_first_spoof(results),

        "query_attack_coverage": query_attack_coverage(results),
        "attack_type_breakdown": attack_type_breakdown(results),
    }

    _save(args.output_path, metrics)
    
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    logger.info(f"Evaluation complete. Metrics saved to {args.output_path}")


if __name__ == "__main__":
    main()
