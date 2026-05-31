from __future__ import annotations

"""
src/defense/retrieval_agreement_defense.py

Retrieval Agreement Defense (RAD)
==================================

הרעיון המרכזי:
  real evidence chunk אמור להיות עקבי על פני מספר שיטות retrieval.
  spoof chunk לרוב מדורג גבוה בdense retrieval (כי הוא optimized לembedding)
  אבל פחות עקבי ב-BM25 או CrossEncoder.

  הסיגנל: disagreement בין retrievers = סימן ל-spoof.

זרימה:
  1. dense retrieval (FAISS/MiniLM) → Top-K
  2. BM25 retrieval → Top-K
  3. מיזוג candidates לפי chunk_id
  4. חישוב agreement_score לכל candidate
  5. reranking לפי final_score
  6. החזרת Top-5

agreement_score = 1 / (1 + std(normalized_ranks))
disagreement_penalty = std(normalized_ranks)

final_score = retrieval_score
            + lambda_agreement    * agreement_score
            - lambda_disagreement * disagreement_penalty
            + lambda_ce           * crossencoder_score  (אם זמין)
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ── I/O ───────────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tok(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-12:
        return [0.5] * len(values)
    return ((arr - lo) / (hi - lo)).astype(float).tolist()


def _sigmoid(x: float) -> float:
    import math
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# ── Dense retrieval ───────────────────────────────────────────────────────────

def dense_retrieve(
    question: str,
    index: faiss.Index,
    metadata: List[Dict],
    embedder: SentenceTransformer,
    model_name: str,
    top_k: int = 20,
) -> List[Dict]:
    """FAISS dense retrieval לשאלה אחת."""
    prefix = f"query: {question}" if "e5" in model_name.lower() else question
    q_emb = embedder.encode(
        [prefix], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32")
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        results.append({
            "chunk_id":        item["chunk_id"],
            "doc_id":          item["doc_id"],
            "text":            item["text"],
            "title":           item.get("title", ""),
            "is_spoof":        item.get("is_spoof", False),
            "spoof_for_query": item.get("spoof_for_query"),
            "attack_type":     item.get("attack_type"),
            "dense_score":     float(score),
            "dense_rank":      rank,
        })
    return results


# ── BM25 retrieval ────────────────────────────────────────────────────────────

def bm25_retrieve(
    question: str,
    bm25: BM25Okapi,
    metadata: List[Dict],
    top_k: int = 20,
) -> List[Dict]:
    """BM25 retrieval לשאלה אחת."""
    scores = bm25.get_scores(_tok(question))
    top_i  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for rank, idx in enumerate(top_i, 1):
        item = metadata[idx]
        results.append({
            "chunk_id":    item["chunk_id"],
            "bm25_score":  float(scores[idx]),
            "bm25_rank":   rank,
        })
    return results


# ── Candidate pool merger ─────────────────────────────────────────────────────

def merge_candidates(
    dense_results: List[Dict],
    bm25_results:  List[Dict],
    pool_size:     int = 20,
    missing_rank_penalty: int = 30,
) -> List[Dict]:
    """
    מיזוג candidates מdense וBM25 לפי chunk_id.

    chunk שלא מופיע בretriever אחד מקבל rank = missing_rank_penalty.
    זה גדול מכל rank אמיתי ב-Top-20, ולכן מעניש chunks
    שמופיעים רק בretriever אחד.
    """
    # מפה לפי chunk_id
    pool: Dict[str, Dict] = {}

    for item in dense_results:
        cid = item["chunk_id"]
        pool[cid] = {
            **item,
            "bm25_score": 0.0,
            "bm25_rank":  missing_rank_penalty,
        }

    for item in bm25_results:
        cid = item["chunk_id"]
        if cid in pool:
            pool[cid]["bm25_score"] = item["bm25_score"]
            pool[cid]["bm25_rank"]  = item["bm25_rank"]
        else:
            # נמצא רק ב-BM25 — מוסיפים עם dense_rank גדול
            pool[cid] = {
                "chunk_id":        cid,
                "doc_id":          "",
                "text":            "",
                "title":           "",
                "is_spoof":        False,
                "spoof_for_query": None,
                "attack_type":     None,
                "dense_score":     0.0,
                "dense_rank":      missing_rank_penalty,
                "bm25_score":      item["bm25_score"],
                "bm25_rank":       item["bm25_rank"],
            }

    # מוגבל לpool_size הגדולים לפי dense_score
    candidates = sorted(pool.values(), key=lambda x: x["dense_score"], reverse=True)
    return candidates[:pool_size]


# ── Agreement scoring ─────────────────────────────────────────────────────────

def compute_agreement(
    candidate: Dict,
    max_rank: int = 20,
) -> Tuple[float, float, Dict]:
    """
    חישוב agreement_score ו-disagreement_penalty.

    ranks מנורמלים ל-[0,1]:
      0 = מקום ראשון (הכי טוב)
      1 = מקום אחרון / לא נמצא

    agreement_score     = 1 / (1 + std(normalized_ranks))
      גבוה כשה-ranks עקביים בין retrievers.

    disagreement_penalty = std(normalized_ranks)
      גבוה כשיש אי-עקביות (dense גבוה, BM25 נמוך).
    """
    dense_rank = candidate.get("dense_rank", max_rank)
    bm25_rank  = candidate.get("bm25_rank",  max_rank)

    # נרמול ל-[0,1]
    norm_dense = (dense_rank - 1) / max(1, max_rank - 1)
    norm_bm25  = (bm25_rank  - 1) / max(1, max_rank - 1)
    norm_dense = min(1.0, norm_dense)
    norm_bm25  = min(1.0, norm_bm25)

    ranks = [norm_dense, norm_bm25]
    std_ranks = float(np.std(ranks))

    agreement_score      = 1.0 / (1.0 + std_ranks)
    disagreement_penalty = std_ranks

    # spoof signature: dense_rank נמוך (טוב) אבל bm25_rank גבוה (רע)
    # כלומר norm_dense קטן ו-norm_bm25 גדול → std גבוה → disagreement גבוה
    spoof_signature = max(0.0, norm_bm25 - norm_dense)

    return agreement_score, disagreement_penalty, {
        "dense_rank":           dense_rank,
        "bm25_rank":            bm25_rank,
        "norm_dense_rank":      round(norm_dense, 4),
        "norm_bm25_rank":       round(norm_bm25,  4),
        "rank_std":             round(std_ranks,   4),
        "agreement_score":      round(agreement_score,      4),
        "disagreement_penalty": round(disagreement_penalty, 4),
        "spoof_signature":      round(spoof_signature,      4),
    }


# ── CrossEncoder scoring (optional) ──────────────────────────────────────────

def score_with_cross_encoder(
    query:        str,
    candidates:   List[Dict],
    cross_encoder: CrossEncoder,
    batch_size:   int = 16,
) -> List[float]:
    """CrossEncoder scores לכל candidates."""
    pairs = [(query, c.get("text", "")) for c in candidates]
    if not pairs:
        return []
    raw = cross_encoder.predict(pairs, batch_size=batch_size,
                                show_progress_bar=False)
    raw = [float(x) for x in np.asarray(raw).reshape(-1)]
    # נרמול ל-[0,1] עם sigmoid
    return [_sigmoid(x) for x in raw]


# ── Main defense function ─────────────────────────────────────────────────────

def retrieval_agreement_defense(
    query:        str,
    dense_results: List[Dict],
    bm25_results:  List[Dict],
    cross_encoder: Optional[CrossEncoder] = None,
    pool_size:     int   = 20,
    top_k_return:  int   = 5,
    lambda_agreement:    float = 0.3,
    lambda_disagreement: float = 0.4,
    lambda_ce:           float = 0.3,
    missing_rank_penalty: int  = 30,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Retrieval Agreement Defense.

    מחזיר:
      (top_k_after, all_candidates_scored)
    """
    # 1. מיזוג candidates
    candidates = merge_candidates(
        dense_results, bm25_results,
        pool_size=pool_size,
        missing_rank_penalty=missing_rank_penalty,
    )

    if not candidates:
        return [], []

    # 2. נרמול dense scores ל-[0,1]
    dense_scores = [c.get("dense_score", 0.0) for c in candidates]
    dense_norm   = _minmax(dense_scores)

    # 3. CrossEncoder (אם זמין)
    ce_scores: List[float] = []
    if cross_encoder is not None:
        ce_scores = score_with_cross_encoder(query, candidates, cross_encoder)
    ce_norm = _minmax(ce_scores) if ce_scores else [0.0] * len(candidates)

    # 4. Agreement scoring + final score
    scored: List[Dict] = []
    for idx, (cand, d_norm, ce_n) in enumerate(
        zip(candidates, dense_norm, ce_norm)
    ):
        agr_score, dis_pen, agr_detail = compute_agreement(
            cand, max_rank=missing_rank_penalty
        )

        # final_score:
        #   dense score בתור בסיס
        #   + agreement bonus (עקביות בין retrievers)
        #   - disagreement penalty (חוסר עקביות = סימן ל-spoof)
        #   + CE score (אם זמין)
        final_score = (
            d_norm
            + lambda_agreement    * agr_score
            - lambda_disagreement * dis_pen
            + lambda_ce           * ce_n
        )
        final_score = max(0.0, float(final_score))

        scored.append({
            **cand,
            "original_dense_score":  round(d_norm,       4),
            "crossencoder_score":    round(ce_n,          4),
            "agreement_score":       round(agr_score,     4),
            "disagreement_penalty":  round(dis_pen,       4),
            "final_score":           round(final_score,   6),
            "agreement_detail":      agr_detail,
            "score_breakdown": {
                "dense_norm":    round(d_norm,                        4),
                "agr_bonus":     round(lambda_agreement * agr_score,  4),
                "dis_penalty":   round(-lambda_disagreement * dis_pen, 4),
                "ce_bonus":      round(lambda_ce * ce_n,              4),
                "final":         round(final_score,                   4),
            },
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:top_k_return], scored


# ── Evaluation helpers ────────────────────────────────────────────────────────

def recall_at_k(results: Dict[str, List[Dict]], qrels: Dict[str, List[str]], k: int) -> float:
    hits = total = 0
    for qid, rel in qrels.items():
        ranked = results.get(qid, [])
        if not ranked:
            continue
        total += 1
        retrieved = {item["doc_id"] for item in ranked[:k]}
        if any(d in retrieved for d in rel):
            hits += 1
    return hits / total if total else 0.0


def top1_spoof_win_rate(results: Dict[str, List[Dict]]) -> float:
    total = wins = 0
    for ranked in results.values():
        if not ranked:
            continue
        total += 1
        if ranked[0].get("is_spoof", False):
            wins += 1
    return wins / total if total else 0.0


def avg_real_chunk_rank(results: Dict[str, List[Dict]]) -> float:
    ranks = []
    for ranked in results.values():
        for i, item in enumerate(ranked, 1):
            if not item.get("is_spoof", False):
                ranks.append(i)
                break
    return float(np.mean(ranks)) if ranks else float("inf")


def pct_real_in_pool(results: Dict[str, List[Dict]]) -> float:
    total = has_real = 0
    for ranked in results.values():
        if not ranked:
            continue
        total += 1
        if any(not item.get("is_spoof", False) for item in ranked):
            has_real += 1
    return has_real / total if total else 0.0


def pct_spoof_above_real(results: Dict[str, List[Dict]]) -> float:
    total = spoof_above = 0
    for ranked in results.values():
        if not ranked:
            continue
        total += 1
        # ציון ה-real הראשון
        real_rank = next(
            (i for i, item in enumerate(ranked, 1) if not item.get("is_spoof", False)),
            None,
        )
        if real_rank is None:
            spoof_above += 1
            continue
        if ranked[0].get("is_spoof", False):
            spoof_above += 1
    return spoof_above / total if total else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Agreement Defense — multi-retriever consistency scoring."
    )
    parser.add_argument("--queries-path",    type=Path,
                        default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--qrels-path",      type=Path,
                        default=Path("data/processed/val_qrels.json"))
    parser.add_argument("--spoof-chunks-path", type=Path,
                        default=Path("data/processed/spoof_chunks.jsonl"))
    parser.add_argument("--index-dir",       type=Path,
                        default=Path("indexes/minilm_attack"),
                        help="FAISS index directory (augmented/attack index)")
    parser.add_argument("--model-name",      type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--pool-size",       type=int,  default=20)
    parser.add_argument("--top-k",           type=int,  default=5)
    parser.add_argument("--missing-rank-penalty", type=int, default=30)
    parser.add_argument("--lambda-agreement",    type=float, default=0.3)
    parser.add_argument("--lambda-disagreement", type=float, default=0.4)
    parser.add_argument("--lambda-ce",           type=float, default=0.3)
    parser.add_argument("--use-cross-encoder",   action="store_true")
    parser.add_argument("--cross-encoder-model", type=str,
                        default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--max-queries",     type=int,  default=0,
                        help="0 = כל השאלות")
    parser.add_argument("--metrics-output",  type=Path,
                        default=Path("results/defense/agreement_defense_metrics.json"))
    parser.add_argument("--debug-output",    type=Path,
                        default=Path("results/defense/agreement_defense_debug.jsonl"))
    args = parser.parse_args()

    # ── טעינה ─────────────────────────────────────────────────────────────────
    queries  = _read_jsonl(args.queries_path)
    qrels    = _read_json(args.qrels_path)
    spoofs   = _read_jsonl(args.spoof_chunks_path)
    attacked = set(s["spoof_for_query"] for s in spoofs)

    # מגביל לשאלות מותקפות בלבד
    queries = [q for q in queries if q["query_id"] in attacked]
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    qrels = {q["query_id"]: qrels.get(q["query_id"], []) for q in queries}

    print(f"שאלות: {len(queries)} | pool_size={args.pool_size} | top_k={args.top_k}")

    # ── טעינת index ───────────────────────────────────────────────────────────
    faiss_index = faiss.read_index(str(args.index_dir / "index.faiss"))
    with (args.index_dir / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)

    print(f"טוען embedding model: {args.model_name}")
    embedder = SentenceTransformer(args.model_name, local_files_only=True)
    bm25     = BM25Okapi([_tok(item["text"]) for item in metadata])

    cross_encoder: Optional[CrossEncoder] = None
    if args.use_cross_encoder:
        print(f"טוען CrossEncoder: {args.cross_encoder_model}")
        cross_encoder = CrossEncoder(args.cross_encoder_model)

    # ── הרצה ──────────────────────────────────────────────────────────────────
    defended:    Dict[str, List[Dict]] = {}
    debug_rows:  List[Dict]            = []
    q_map = {q["query_id"]: q for q in queries}

    for n, q in enumerate(queries, 1):
        qid      = q["query_id"]
        question = q["question"]
        gold     = (q.get("answers") or [""])[0]

        # Dense retrieval
        dense_res = dense_retrieve(
            question, faiss_index, metadata, embedder,
            args.model_name, top_k=args.pool_size,
        )

        # BM25 retrieval
        bm25_res = bm25_retrieve(
            question, bm25, metadata, top_k=args.pool_size,
        )

        # Agreement defense
        top_k_after, all_scored = retrieval_agreement_defense(
            query=question,
            dense_results=dense_res,
            bm25_results=bm25_res,
            cross_encoder=cross_encoder,
            pool_size=args.pool_size,
            top_k_return=args.top_k,
            lambda_agreement=args.lambda_agreement,
            lambda_disagreement=args.lambda_disagreement,
            lambda_ce=args.lambda_ce,
            missing_rank_penalty=args.missing_rank_penalty,
        )

        defended[qid] = top_k_after

        # Debug row
        real_before = next(
            (c for c in dense_res if not c.get("is_spoof", False)), None
        )
        debug_rows.append({
            "query_id":              qid,
            "question":              question,
            "gold_answer":           gold,
            "real_chunk_id":         real_before["chunk_id"] if real_before else None,
            "real_dense_rank":       real_before["dense_rank"] if real_before else None,
            "top1_before_is_spoof":  dense_res[0].get("is_spoof", False) if dense_res else None,
            "top1_after_is_spoof":   top_k_after[0].get("is_spoof", False) if top_k_after else None,
            "top_candidates_before": [
                {"chunk_id": c["chunk_id"], "dense_rank": c["dense_rank"],
                 "is_spoof": c["is_spoof"]}
                for c in dense_res[:5]
            ],
            "top_candidates_after": [
                {"chunk_id": c["chunk_id"],
                 "dense_rank": c.get("dense_rank"), "bm25_rank": c.get("bm25_rank"),
                 "agreement_score": c.get("agreement_score"),
                 "disagreement_penalty": c.get("disagreement_penalty"),
                 "final_score": c.get("final_score"),
                 "is_spoof": c.get("is_spoof")}
                for c in top_k_after
            ],
        })

        if n % 50 == 0:
            print(f"  {n}/{len(queries)} שאלות")

    # ── מדדים ────────────────────────────────────────────────────────────────
    r5    = recall_at_k(defended, qrels, 5)
    r20_d = recall_at_k(
        {qid: dense_retrieve(q["question"], faiss_index, metadata,
                             embedder, args.model_name, top_k=20)
         for q in queries[:50]
         for qid in [q["query_id"]]},
        {q["query_id"]: qrels.get(q["query_id"], []) for q in queries[:50]},
        20,
    ) if len(queries) <= 50 else None

    swr   = top1_spoof_win_rate(defended)
    avg_r = avg_real_chunk_rank(defended)
    pct_r = pct_real_in_pool(defended)
    pct_s = pct_spoof_above_real(defended)

    metrics = {
        "defense_mode":               "retrieval_agreement",
        "num_queries":                len(defended),
        "pool_size":                  args.pool_size,
        "lambda_agreement":           args.lambda_agreement,
        "lambda_disagreement":        args.lambda_disagreement,
        "lambda_ce":                  args.lambda_ce,
        "use_cross_encoder":          args.use_cross_encoder,
        "recall@5":                   round(r5,    4),
        "top1_spoof_win_rate":        round(swr,   4),
        "avg_real_chunk_rank":        round(avg_r, 4) if avg_r != float("inf") else None,
        "pct_real_in_top5_pool":      round(pct_r, 4),
        "pct_spoof_above_real":       round(pct_s, 4),
        "interpretation": {
            "recall@5":            "Higher is better — did we find the real answer?",
            "top1_spoof_win_rate": "Lower is better — did spoofs stay at Top-1?",
            "pct_real_in_pool":    "Higher is better — is the real chunk in the candidate pool?",
            "pct_spoof_above_real":"Lower is better — how often does spoof beat real?",
        },
    }

    _write_json(args.metrics_output, metrics)
    _write_jsonl(args.debug_output, debug_rows)

    print("\n" + "=" * 60)
    print("Retrieval Agreement Defense — תוצאות")
    print("=" * 60)
    print(f"  Recall@5:               {r5:.3f}")
    print(f"  Top-1 Spoof Win Rate:   {swr:.3f}  (lower is better)")
    print(f"  Avg real chunk rank:    {avg_r:.2f}  (lower is better)")
    print(f"  Real chunk in pool:     {pct_r:.1%}  (higher is better)")
    print(f"  Spoof above real:       {pct_s:.1%}  (lower is better)")
    print(f"\nנשמר → {args.metrics_output}")
    print(f"Debug → {args.debug_output}")


if __name__ == "__main__":
    main()
