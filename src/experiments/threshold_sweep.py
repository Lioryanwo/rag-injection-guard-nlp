from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from sentence_transformers import CrossEncoder, SentenceTransformer

from src.defense.defense_filter import cross_encoder_rerank, load_queries
from src.evaluation.evaluate_retrieval import recall_at_k, top1_spoof_win_rate, avg_spoofs_in_top_k


def _load(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _attacked_qids(spoof_path: Path) -> Set[str]:
    qids: Set[str] = set()
    with spoof_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("spoof_for_query")
            if qid:
                qids.add(str(qid))
    return qids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep lambda (penalty weight) for soft CrossEncoder reranking.\n"
            "מכיוון שציוני suspicion הם כמעט אפס, ה-threshold הישן לא עבד.\n"
            "עכשיו threshold משמש כ-lambda: final = CE_score - lambda * suspicion"
        )
    )
    parser.add_argument("--input-path",    type=Path, required=True)
    parser.add_argument("--queries-path",  type=Path,
                        default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--qrels-path",    type=Path,
                        default=Path("data/processed/val_qrels.json"))
    parser.add_argument("--spoof-chunks-path", type=Path,
                        default=Path("data/processed/spoof_chunks.jsonl"))
    parser.add_argument("--output-path",   type=Path,
                        default=Path("results/retrieval/threshold_sweep.json"))
    parser.add_argument("--keep-top-k",   type=int,  default=5)
    parser.add_argument("--cross-encoder-model", type=str,
                        default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--use-doc2query", action="store_true")
    parser.add_argument("--doc2query-embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--doc2query-weight", type=float, default=0.20)
    parser.add_argument("--semantic-weight",  type=float, default=0.70)
    parser.add_argument("--retrieval-weight", type=float, default=0.30)
    parser.add_argument("--alpha",         type=float, default=0.5,
                        help="Recall@5 - alpha * SpoofWinRate")
    parser.add_argument("--max-queries",   type=int,  default=0)
    args = parser.parse_args()

    # ── טעינה והגבלה ל-300 שאלות מותקפות ────────────────────────────────────
    all_results = _load(args.input_path)
    all_qrels   = _load(args.qrels_path)
    queries     = load_queries(args.queries_path)

    attacked = _attacked_qids(args.spoof_chunks_path) if args.spoof_chunks_path.exists() else set()
    if attacked:
        qrels   = {qid: rel for qid, rel in all_qrels.items()   if qid in attacked}
        results = {qid: r   for qid, r   in all_results.items() if qid in attacked}
    else:
        qrels, results = all_qrels, all_results

    if args.max_queries > 0:
        results = dict(list(results.items())[: args.max_queries])
        qrels   = {qid: rel for qid, rel in qrels.items() if qid in results}

    print(f"Sweep על {len(results)} שאלות")

    ce = CrossEncoder(args.cross_encoder_model)
    dq = SentenceTransformer(args.doc2query_embedding_model) if args.use_doc2query else None

    # ── טווח lambda ───────────────────────────────────────────────────────────
    # ציוני suspicion הם 0.01-0.05.
    # lambda גדול = penalty חזק יותר.
    # טווח: 0.0 (ללא penalty) עד 1.0 (penalty מלא)
    # בשלבים של 0.1 כדי לראות tradeoff ברור
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rows: List[Dict] = []
    best_obj = -999.0
    best_lam = 0.0

    for lam in lambdas:
        defended: Dict[str, List[Dict]] = {}

        for qid, ranked in results.items():
            reranked = cross_encoder_rerank(
                query=queries.get(qid, ""),
                ranked=ranked,
                cross_encoder=ce,
                threshold=lam,                    # threshold = lambda כאן
                semantic_weight=args.semantic_weight,
                retrieval_weight=args.retrieval_weight,
                doc2query_weight=args.doc2query_weight if dq else 0.0,
                doc2query_embedder=dq,
                doc2query_model_name=args.doc2query_embedding_model,
            )
            defended[qid] = reranked[: args.keep_top_k]

        r5   = recall_at_k(defended, qrels, 5)
        swr  = top1_spoof_win_rate(defended)
        asp5 = avg_spoofs_in_top_k(defended, 5)
        obj  = r5 - args.alpha * swr

        row = {
            "threshold":                    lam,
            "lambda":                       lam,
            "num_queries":                  len(defended),
            "recall@5":                     round(r5,   4),
            "top1_spoof_win_rate":          round(swr,  4),
            "avg_spoofs_in_top5":           round(asp5, 4),
            "objective_r5_minus_alpha_swr": round(obj,  4),
        }
        rows.append(row)

        marker = " ← best" if obj > best_obj else ""
        print(
            f"lambda={lam:.2f}  recall={r5:.3f}  "
            f"spoof_win={swr:.3f}  obj={obj:.3f}{marker}"
        )

        if obj > best_obj:
            best_obj = obj
            best_lam = lam

    print(f"\nLambda הכי טוב: {best_lam:.2f}  (objective={best_obj:.4f})")
    print(f"Objective = Recall@5 - {args.alpha} * SpoofWinRate")

    _save(args.output_path, {
        "sweep":           rows,
        "best_threshold":  best_lam,
        "best_objective":  round(best_obj, 4),
        "alpha":           args.alpha,
        "note": (
            "threshold משמש כlambda לsoft penalty. "
            "ציוני suspicion הם 0.01-0.05 לכן ה-penalty קטן. "
            "הסיגנל הראשי הוא CrossEncoder."
        ),
    })
    print(f"נשמר → {args.output_path}")


if __name__ == "__main__":
    main()
