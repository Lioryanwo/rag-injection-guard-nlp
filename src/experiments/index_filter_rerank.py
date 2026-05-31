from __future__ import annotations

"""
src/defense/index_filter_rerank.py

Combined Defense: Index-Level Filtering + CrossEncoder Reranking
=================================================================

Pipeline:
  1. Filter suspicious chunks from corpus (no is_spoof label used)
  2. Build filtered FAISS index
  3. Retrieve Top-N from filtered index
  4. CrossEncoder rerank → final Top-5

Why this should work:
  - Index filter removes 31.7% of spoofs with only 1.4% false positive
  - Real chunk in pool: 28% → 48% after filtering
  - CrossEncoder then promotes real evidence chunks from pool to Top-5
  - Together: cleaner pool + better ranking = better recall
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ── Import filter logic from index_level_filter ───────────────────────────────
from src.experiments.index_level_filter import (
    filter_corpus,
    build_filtered_index,
    evaluate_filter,
    _read_jsonl,
    _write_jsonl,
    _write_json,
)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_top_n(
    question:   str,
    index:      faiss.Index,
    metadata:   List[Dict],
    embedder:   SentenceTransformer,
    model_name: str,
    top_n:      int = 50,
) -> List[Dict]:
    """Dense retrieval לשאלה אחת — מחזיר Top-N candidates."""
    prefix = f"query: {question}" if "e5" in model_name.lower() else question
    q_emb  = embedder.encode(
        [prefix], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32")

    scores, idxs = index.search(q_emb, top_n)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        results.append({
            **item,
            "retrieval_score": float(score),
            "retrieval_rank":  rank,
        })
    return results


# ── CrossEncoder reranking ────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    import math
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def crossencoder_rerank(
    query:        str,
    candidates:   List[Dict],
    cross_encoder: CrossEncoder,
    top_k:        int = 5,
    batch_size:   int = 16,
) -> Tuple[List[Dict], List[Dict]]:
    """
    CrossEncoder reranking על candidates.

    מחזיר (top_k_after, all_scored).
    """
    if not candidates:
        return [], []

    pairs    = [(query, c.get("text", "")) for c in candidates]
    raw_ce   = cross_encoder.predict(pairs, batch_size=batch_size,
                                     show_progress_bar=False)
    raw_ce   = [float(x) for x in np.asarray(raw_ce).reshape(-1)]
    ce_probs = [_sigmoid(x) for x in raw_ce]

    scored = []
    for item, ce_raw, ce_prob in zip(candidates, raw_ce, ce_probs):
        scored.append({
            **item,
            "ce_score_raw":  round(ce_raw,  4),
            "ce_score_prob": round(ce_prob, 4),
            "final_score":   round(ce_prob, 6),
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, item in enumerate(scored, 1):
        item["rerank_rank"] = rank

    return scored[:top_k], scored


# ── Per-query evaluation ──────────────────────────────────────────────────────

def _first_real_rank(ranked: List[Dict]) -> Optional[int]:
    for i, item in enumerate(ranked, 1):
        if not item.get("is_spoof", False):
            return i
    return None


def _has_real(ranked: List[Dict]) -> bool:
    return any(not item.get("is_spoof", False) for item in ranked)


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def run_evaluation(
    queries:      List[Dict],
    qrels:        Dict[str, List[str]],
    index:        faiss.Index,
    metadata:     List[Dict],
    embedder:     SentenceTransformer,
    model_name:   str,
    cross_encoder: Optional[CrossEncoder],
    top_n:        int = 50,
    top_k:        int = 5,
    attacked_qids: Set[str] = None,
    label:        str = "defense",
) -> Tuple[Dict, List[Dict]]:
    """
    מריץ retrieval + reranking על כל השאלות.
    מחזיר (metrics, debug_rows).
    """
    qs = queries
    if attacked_qids:
        qs = [q for q in queries if q["query_id"] in attacked_qids]

    hits_topk = hits_top20 = spoof_top1 = real_in_pool = 0
    spoof_above_real = 0
    real_ranks_before: List[int] = []
    real_ranks_after:  List[int] = []
    total = 0
    debug_rows = []

    for q in qs:
        qid      = q["query_id"]
        question = q["question"]
        gold     = (q.get("answers") or [""])[0]
        rel      = set(qrels.get(qid, []))

        # Retrieval
        candidates = retrieve_top_n(
            question, index, metadata, embedder, model_name, top_n=top_n,
        )
        if not candidates:
            continue
        total += 1

        # Recall@20 (before rerank)
        top20_docs = {c["doc_id"] for c in candidates[:20]}
        if rel & top20_docs:
            hits_top20 += 1

        # Real chunk rank before rerank
        rr_before = _first_real_rank(candidates)
        if rr_before:
            real_ranks_before.append(rr_before)
        if _has_real(candidates):
            real_in_pool += 1

        # Reranking
        if cross_encoder is not None:
            top_k_results, all_scored = crossencoder_rerank(
                question, candidates, cross_encoder,
                top_k=top_k,
            )
        else:
            top_k_results = candidates[:top_k]
            all_scored    = candidates

        # Metrics after rerank
        topk_docs = {c["doc_id"] for c in top_k_results}
        if rel & topk_docs:
            hits_topk += 1

        if top_k_results and top_k_results[0].get("is_spoof", False):
            spoof_top1 += 1

        rr_after = _first_real_rank(top_k_results)
        if rr_after:
            real_ranks_after.append(rr_after)

        # Spoof above real
        real_rank_in_topk = next(
            (i for i, c in enumerate(top_k_results, 1)
             if not c.get("is_spoof", False)),
            None,
        )
        if real_rank_in_topk is None or top_k_results[0].get("is_spoof", False):
            spoof_above_real += 1

        # Debug row
        debug_rows.append({
            "query_id":    qid,
            "question":    question,
            "gold_answer": gold,
            "label":       label,
            "real_rank_before_rerank": rr_before,
            "real_rank_after_rerank":  rr_after,
            "top1_before_spoof": candidates[0].get("is_spoof") if candidates else None,
            "top1_after_spoof":  top_k_results[0].get("is_spoof") if top_k_results else None,
            "real_in_pool":      _has_real(candidates),
            "top5_after": [
                {
                    "chunk_id":    c.get("chunk_id"),
                    "doc_id":      c.get("doc_id"),
                    "is_spoof":    c.get("is_spoof"),
                    "ce_prob":     c.get("ce_score_prob"),
                    "ret_rank":    c.get("retrieval_rank"),
                    "rerank_rank": c.get("rerank_rank"),
                }
                for c in top_k_results
            ],
        })

    metrics = {
        "label":                 label,
        "num_queries":           total,
        "top_n_pool":            top_n,
        "top_k_return":          top_k,
        "recall@5":              round(hits_topk   / max(1, total), 4),
        "recall@20":             round(hits_top20  / max(1, total), 4),
        "top1_spoof_win_rate":   round(spoof_top1  / max(1, total), 4),
        "real_in_pool_pct":      round(real_in_pool / max(1, total), 4),
        "spoof_above_real_pct":  round(spoof_above_real / max(1, total), 4),
        "avg_real_rank_before":  round(float(np.mean(real_ranks_before)), 2)
                                  if real_ranks_before else None,
        "avg_real_rank_after":   round(float(np.mean(real_ranks_after)),  2)
                                  if real_ranks_after  else None,
    }
    return metrics, debug_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combined Defense: Index-Level Filtering + CrossEncoder Reranking"
    )
    # נתונים
    parser.add_argument("--chunks-path",    type=Path,
                        default=Path("data/processed/augmented_chunks.jsonl"))
    parser.add_argument("--queries-path",   type=Path,
                        default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--qrels-path",     type=Path,
                        default=Path("data/processed/val_qrels.json"))
    parser.add_argument("--spoof-chunks-path", type=Path,
                        default=Path("data/processed/spoof_chunks.jsonl"))
    # index directories
    parser.add_argument("--attack-index-dir",   type=Path,
                        default=Path("indexes/minilm_attack"))
    parser.add_argument("--filtered-index-dir", type=Path,
                        default=Path("indexes/minilm_filtered"))
    parser.add_argument("--filtered-chunks-output", type=Path,
                        default=Path("data/processed/corpus_chunks_filtered.jsonl"))
    # מודלים
    parser.add_argument("--model-name",          type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder-model", type=str,
                        default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    # פרמטרים
    parser.add_argument("--filter-threshold",  type=float, default=0.05,
                        help="Suspicion threshold for index-level filter")
    parser.add_argument("--top-n",             type=int,   default=50,
                        help="Candidates retrieved before reranking")
    parser.add_argument("--top-k",             type=int,   default=5,
                        help="Final results returned")
    parser.add_argument("--max-queries",       type=int,   default=0)
    parser.add_argument("--skip-filter",       action="store_true",
                        help="Skip filtering step, use existing filtered index")
    parser.add_argument("--use-semantic",      action="store_true")
    # פלט
    parser.add_argument("--metrics-output", type=Path,
                        default=Path("results/defense/index_filter_rerank_metrics.json"))
    parser.add_argument("--debug-output",   type=Path,
                        default=Path("results/defense/index_filter_rerank_debug.jsonl"))
    args = parser.parse_args()

    # ── טעינת נתונים ──────────────────────────────────────────────────────────
    print("טוען נתונים…")
    chunks  = _read_jsonl(args.chunks_path)
    queries = _read_jsonl(args.queries_path)
    qrels   = json.load(args.qrels_path.open(encoding="utf-8"))
    spoofs  = _read_jsonl(args.spoof_chunks_path)
    attacked_qids = set(s["spoof_for_query"] for s in spoofs)

    if args.max_queries > 0:
        queries = [q for q in queries if q["query_id"] in attacked_qids][:args.max_queries]
        attacked_qids = set(q["query_id"] for q in queries)

    query_texts = [q["question"] for q in queries]
    print(f"Chunks: {len(chunks)} | Queries: {len(queries)}")

    # ── טעינת מודלים ──────────────────────────────────────────────────────────
    print(f"טוען embedding model: {args.model_name}")
    embedder = SentenceTransformer(args.model_name, local_files_only=True)

    print(f"טוען CrossEncoder: {args.cross_encoder_model}")
    ce = CrossEncoder(args.cross_encoder_model)

    # ── שלב 1: סינון + בניית index ────────────────────────────────────────────
    filter_eval = {}
    if not args.skip_filter:
        print(f"\n── שלב 1: Index-level filtering (threshold={args.filter_threshold}) ──")
        kept, removed, scored = filter_corpus(
            chunks=chunks,
            query_texts=query_texts,
            threshold=args.filter_threshold,
            use_semantic=args.use_semantic,
            embedder=embedder if args.use_semantic else None,
            model_name=args.model_name,
        )
        print(f"  Kept: {len(kept)} | Removed: {len(removed)}")
        filter_eval = evaluate_filter(kept, removed, chunks)
        print(f"  Spoof removal rate:  {filter_eval['spoof_removal_rate']:.1%}")
        print(f"  Real false positive: {filter_eval['real_false_positive']:.1%}")
        print(f"  Precision:           {filter_eval['precision_of_removal']:.1%}")

        _write_jsonl(args.filtered_chunks_output,
                     [{k: v for k, v in c.items()
                       if not k.startswith("suspicion")} for c in kept])

        print(f"\n── שלב 2: בניית filtered index ──")
        build_filtered_index(kept, args.filtered_index_dir, embedder, args.model_name)
    else:
        print("  [skip] שלב הסינון — משתמש ב-filtered index קיים")

    # ── שלב 3: טעינת indexes ──────────────────────────────────────────────────
    print("\n── שלב 3: טעינת indexes ──")
    attack_idx   = faiss.read_index(str(args.attack_index_dir   / "index.faiss"))
    filtered_idx = faiss.read_index(str(args.filtered_index_dir / "index.faiss"))

    with (args.attack_index_dir   / "metadata.pkl").open("rb") as f:
        attack_meta = pickle.load(f)
    with (args.filtered_index_dir / "metadata.pkl").open("rb") as f:
        filtered_meta = pickle.load(f)

    print(f"  Attack index:   {attack_idx.ntotal} chunks")
    print(f"  Filtered index: {filtered_idx.ntotal} chunks")

    # ── שלב 4: הרצת שלוש השוואות ──────────────────────────────────────────────
    print(f"\n── שלב 4: הרצת השוואות (top_n={args.top_n}, top_k={args.top_k}) ──")

    # השוואה 1: attack index, ללא reranking
    print("\n  [1/3] Attack index, ללא reranking…")
    m_attack_naive, d_attack_naive = run_evaluation(
        queries, qrels, attack_idx, attack_meta,
        embedder, args.model_name, cross_encoder=None,
        top_n=args.top_k, top_k=args.top_k,
        attacked_qids=attacked_qids, label="attack_naive",
    )

    # השוואה 2: filtered index, ללא reranking
    print("\n  [2/3] Filtered index, ללא reranking…")
    m_filtered_naive, d_filtered_naive = run_evaluation(
        queries, qrels, filtered_idx, filtered_meta,
        embedder, args.model_name, cross_encoder=None,
        top_n=args.top_k, top_k=args.top_k,
        attacked_qids=attacked_qids, label="filtered_naive",
    )

    # השוואה 3: filtered index + CrossEncoder reranking (top_n גדול)
    print("\n  [3/3] Filtered index + CrossEncoder reranking…")
    m_combined, d_combined = run_evaluation(
        queries, qrels, filtered_idx, filtered_meta,
        embedder, args.model_name, cross_encoder=ce,
        top_n=args.top_n, top_k=args.top_k,
        attacked_qids=attacked_qids, label="filtered_reranked",
    )

    # ── הדפסת תוצאות ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Combined Defense — תוצאות השוואה")
    print("=" * 72)
    header = f"  {'':30} {'R@5':>7} {'R@20':>7} {'SpoofWin':>9} {'RealPool':>9}"
    print(header)
    print("  " + "-" * 68)
    for m, label in [
        (m_attack_naive,   "Attack (no defense)"),
        (m_filtered_naive, "Filter only"),
        (m_combined,       "Filter + CrossEncoder"),
    ]:
        print(
            f"  {label:30} "
            f"{m.get('recall@5', 0):>7.3f} "
            f"{m.get('recall@20', 0):>7.3f} "
            f"{m.get('top1_spoof_win_rate', 0):>9.3f} "
            f"{m.get('real_in_pool_pct', 0):>8.1%}"
        )
    print("=" * 72)

    # ── שמירת תוצאות ──────────────────────────────────────────────────────────
    metrics = {
        "filter_threshold":  args.filter_threshold,
        "top_n":             args.top_n,
        "top_k":             args.top_k,
        "filter_evaluation": filter_eval,
        "comparison": {
            "attack_naive":   m_attack_naive,
            "filtered_naive": m_filtered_naive,
            "combined":       m_combined,
        },
        "delta_from_attack_to_combined": {
            "recall@5":            round(
                m_combined.get("recall@5", 0) -
                m_attack_naive.get("recall@5", 0), 4),
            "top1_spoof_win_rate": round(
                m_combined.get("top1_spoof_win_rate", 0) -
                m_attack_naive.get("top1_spoof_win_rate", 0), 4),
            "real_in_pool_pct":    round(
                m_combined.get("real_in_pool_pct", 0) -
                m_attack_naive.get("real_in_pool_pct", 0), 4),
        },
    }
    _write_json(args.metrics_output, metrics)
    _write_jsonl(args.debug_output, d_combined)

    print(f"\nנשמר → {args.metrics_output}")
    print(f"Debug → {args.debug_output}")


if __name__ == "__main__":
    main()
