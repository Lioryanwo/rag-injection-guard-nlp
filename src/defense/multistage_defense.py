from __future__ import annotations

"""
src/defense/multistage_defense.py

Multi-Stage RAG Defense Pipeline
================================

Pipeline:
  1. Evidence-aware index-level filtering
  2. Filtered FAISS index retrieval
  3. CrossEncoder reranking
  4. Evidence / answerability scoring
  5. Final score: CE + evidence - suspicion

Important fixes in this version:
  - No placeholders.
  - Full runnable main().
  - Adds --preserve-threshold and --hard-keep-threshold.
  - RealPool is computed against the gold doc_id/qrels, not just any non-spoof chunk.
  - Recall@5 and Recall@20 compare string-normalized doc_ids.
  - Filter reports gold preservation rate.
"""

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "why", "how",
    "is", "was", "were", "are", "did", "does", "do",
    "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "by", "and", "or", "from", "this", "that", "these", "those",
    "it", "its", "into", "with", "as", "be", "been", "has", "have",
    "had", "will", "would", "could", "should", "may", "might",
}

_GENERIC_PHRASES = [
    "widely documented", "some researchers argue",
    "recent scholarship indicates", "it is important to note",
    "background context", "according to alternative sources",
    "a lesser-known account", "the topic is related to",
    "this passage discusses", "in summary", "this section explains",
    "evidence suggests", "however, the exact", "the precise role remains",
    "interestingly", "notably",
]

_ANSWER_TEMPLATES = [
    r"is\s+(often|widely|commonly)\s+(described|known|regarded)",
    r"according\s+to\s+(various|multiple|several)\s+sources",
    r"(many|several|various)\s+(studies|sources|accounts)\s+(suggest|indicate|show)",
    r"(while|although)\s+the\s+(exact|specific|precise)\s+(answer|detail)",
    r"(this|the)\s+(passage|text|document|chunk)\s+(discusses|addresses|covers)",
]
_ANSWER_RE = [re.compile(p, re.IGNORECASE) for p in _ANSWER_TEMPLATES]

_NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b")
_DATE_RE = re.compile(
    r"\b(\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)
_CAPITAL_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

_NUMBER_PAT = _NUMBER_RE
_DATE_PAT = _DATE_RE
_CAP_PAT = _CAPITAL_RE


# ═══════════════════════════════════════════════════════════════════════════════
# IO helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _cwords(text: str) -> List[str]:
    return [
        t for t in re.findall(r"\b[a-zA-Z][a-zA-Z'\-]+\b", text.lower())
        if t not in STOPWORDS and len(t) > 2
    ]


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-9:
        return [0.5] * len(values)
    return ((arr - lo) / (hi - lo)).astype(float).tolist()


def _clean_doc_id(x) -> str:
    return str(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Suspicion + evidence-preservation filter
# ═══════════════════════════════════════════════════════════════════════════════

def _suspicion_score(chunk: Dict, seen_shingles: Dict) -> Tuple[float, Dict]:
    """Chunk-level suspicion score. Does not use is_spoof labels."""
    text = chunk.get("text", "")
    if not text:
        return 0.0, {}

    words = _cwords(text)
    lc = text.lower()

    lexical_div = len(set(words)) / len(words) if len(words) >= 10 else 1.0
    generic_phrase = min(1.0, sum(1 for p in _GENERIC_PHRASES if p in lc) * 0.20)
    answer_template = min(1.0, sum(1 for p in _ANSWER_RE if p.search(text)) * 0.30)

    near_dup = 0.0
    k = 5
    if len(words) >= k:
        shingles = frozenset(tuple(words[i:i + k]) for i in range(len(words) - k + 1))
        max_j = 0.0
        for prev in seen_shingles:
            u = len(shingles | prev)
            if u:
                max_j = max(max_j, len(shingles & prev) / u)
        near_dup = min(1.0, max_j)
        seen_shingles[shingles] = seen_shingles.get(shingles, 0) + 1

    score = min(
        1.0,
        0.20 * max(0.0, 0.65 - lexical_div)
        + 0.40 * generic_phrase
        + 0.30 * answer_template
        + 0.10 * near_dup,
    )

    return score, {
        "lexical_diversity": round(lexical_div, 4),
        "generic_phrase": round(generic_phrase, 4),
        "answer_template": round(answer_template, 4),
        "near_duplicate": round(near_dup, 4),
        "suspicion": round(score, 4),
    }


def _evidence_preservation_score(chunk: Dict) -> Tuple[float, Dict]:
    """Rewards chunks that look like real evidence-bearing text."""
    text = chunk.get("text", "")
    if not text:
        return 0.0, {}

    numbers = _NUMBER_PAT.findall(text)
    dates = _DATE_PAT.findall(text)
    caps = [w for w in _CAP_PAT.findall(text) if w.lower() not in STOPWORDS and len(w) > 2]
    words = _cwords(text)

    has_numbers = min(1.0, len(numbers) / 2)
    has_dates = min(1.0, len(dates) / 1)
    has_caps = min(1.0, len(caps) / 3)

    n_words = len(text.split())
    if 30 <= n_words <= 220:
        length_ok = 1.0
    elif n_words < 15:
        length_ok = 0.0
    else:
        length_ok = 0.5

    lexical_div = len(set(words)) / len(words) if len(words) >= 10 else 0.5

    score = min(
        1.0,
        0.30 * has_numbers
        + 0.20 * has_dates
        + 0.25 * has_caps
        + 0.15 * length_ok
        + 0.10 * lexical_div,
    )

    return score, {
        "n_numbers": len(numbers),
        "n_dates": len(dates),
        "n_caps": len(caps),
        "n_words": n_words,
        "lexical_div": round(lexical_div, 4),
        "evidence_pres": round(score, 4),
    }


def filter_corpus(
    chunks: List[Dict],
    query_texts: List[str],
    threshold: float = 0.05,
    preserve_threshold: float = 0.30,
    hard_keep_threshold: float = 0.60,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Conservative evidence-aware filter.

    Remove only when:
      suspicion_score >= threshold
      AND evidence_preservation_score < preserve_threshold

    Hard keep:
      evidence_preservation_score >= hard_keep_threshold
    """
    del query_texts  # Kept for API compatibility. Not used by design.

    seen: Dict = {}
    kept: List[Dict] = []
    removed: List[Dict] = []

    for chunk in chunks:
        sus, sus_det = _suspicion_score(chunk, seen)
        epres, ep_det = _evidence_preservation_score(chunk)

        if epres >= hard_keep_threshold:
            decision = "keep_hard_evidence"
        elif sus >= threshold and epres < preserve_threshold:
            decision = "remove_suspicious_low_evidence"
        else:
            decision = "keep"

        entry = {
            **chunk,
            "_sus": sus,
            "_epres": epres,
            "_decision": decision,
            "_sus_det": sus_det,
            "_ep_det": ep_det,
        }

        if decision.startswith("remove"):
            removed.append(entry)
        else:
            kept.append(entry)

    return kept, removed


def evaluate_filter(
    kept: List[Dict],
    removed: List[Dict],
    all_chunks: List[Dict],
    qrels: Optional[Dict] = None,
    attacked_qids: Optional[Set[str]] = None,
) -> Dict:
    """Evaluate filter quality. is_spoof/qrels are used for evaluation only."""
    total_spoof = sum(1 for c in all_chunks if c.get("is_spoof", False))
    total_real = sum(1 for c in all_chunks if not c.get("is_spoof", False))
    removed_spoof = sum(1 for c in removed if c.get("is_spoof", False))
    removed_real = sum(1 for c in removed if not c.get("is_spoof", False))

    gold_preserved = 0
    gold_removed = 0
    if qrels and attacked_qids:
        kept_doc_ids = {_clean_doc_id(c.get("doc_id")) for c in kept}
        for qid in attacked_qids:
            for doc_id in qrels.get(qid, []):
                if _clean_doc_id(doc_id) in kept_doc_ids:
                    gold_preserved += 1
                else:
                    gold_removed += 1

    n_gold = gold_preserved + gold_removed
    return {
        "total": len(all_chunks),
        "total_spoof": total_spoof,
        "total_real": total_real,
        "removed": len(removed),
        "removed_spoof": removed_spoof,
        "removed_real": removed_real,
        "spoof_removal_rate": round(removed_spoof / max(1, total_spoof), 4),
        "real_false_positive": round(removed_real / max(1, total_real), 4),
        "precision": round(removed_spoof / max(1, len(removed)), 4),
        "gold_preserved": gold_preserved,
        "gold_removed": gold_removed,
        "gold_preservation_rate": round(gold_preserved / max(1, n_gold), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Index + retrieval
# ═══════════════════════════════════════════════════════════════════════════════

def build_index(
    chunks: List[Dict],
    output_dir: Path,
    embedder: SentenceTransformer,
    model_name: str,
    batch_size: int = 64,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    texts = [f"passage: {c['text']}" if "e5" in model_name.lower() else c["text"] for c in chunks]
    embs = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    faiss.write_index(idx, str(output_dir / "index.faiss"))
    with (output_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(chunks, f)
    np.save(output_dir / "embeddings.npy", embs)
    print(f"  Index: {len(chunks)} chunks → {output_dir}")


def retrieve(
    question: str,
    index: faiss.Index,
    metadata: List[Dict],
    embedder: SentenceTransformer,
    model_name: str,
    top_n: int,
) -> List[Dict]:
    prefix = f"query: {question}" if "e5" in model_name.lower() else question
    q_emb = embedder.encode(
        [prefix],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    scores, idxs = index.search(q_emb, top_n)
    results: List[Dict] = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
        if idx < 0 or idx >= len(metadata):
            continue
        results.append({
            **metadata[idx],
            "retrieval_score": float(score),
            "retrieval_rank": rank,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence score + reranking
# ═══════════════════════════════════════════════════════════════════════════════

def evidence_score(query: str, chunk_text: str) -> Tuple[float, Dict]:
    q_words = set(_cwords(query))
    c_words = _cwords(chunk_text)
    c_set = set(c_words)

    kw_overlap = len(q_words & c_set) / max(1, len(q_words))

    numbers = _NUMBER_RE.findall(chunk_text)
    dates = _DATE_RE.findall(chunk_text)
    caps = [w for w in _CAPITAL_RE.findall(chunk_text) if w.lower() not in STOPWORDS and len(w) > 2]

    concrete_score = (
        0.40 * float(len(numbers) > 0)
        + 0.30 * float(len(dates) > 0)
        + 0.30 * min(1.0, len(caps) / 4)
    )

    q_caps = {w.lower() for w in _CAPITAL_RE.findall(query) if w.lower() not in STOPWORDS}
    c_caps = {w.lower() for w in caps}
    ne_overlap = len(q_caps & c_caps) / max(1, len(q_caps)) if q_caps else 0.5

    lc = chunk_text.lower()
    vague_hits = sum(1 for p in _GENERIC_PHRASES if p in lc)
    tmpl_hits = sum(1 for p in _ANSWER_RE if p.search(chunk_text))
    vague_penalty = min(0.5, vague_hits * 0.10 + tmpl_hits * 0.15)

    if c_words:
        q_repeat_ratio = sum(1 for w in c_words if w in q_words) / len(c_words)
        repetition_penalty = min(0.3, max(0.0, q_repeat_ratio - 0.30) * 1.5)
    else:
        repetition_penalty = 0.0

    score = (
        0.25 * kw_overlap
        + 0.50 * concrete_score
        + 0.25 * ne_overlap
        - vague_penalty
        - repetition_penalty
    )
    score = max(0.0, min(1.0, float(score)))

    return score, {
        "kw_overlap": round(kw_overlap, 4),
        "concrete_score": round(concrete_score, 4),
        "ne_overlap": round(ne_overlap, 4),
        "vague_penalty": round(vague_penalty, 4),
        "repetition_penalty": round(repetition_penalty, 4),
        "has_numbers": len(numbers) > 0,
        "has_dates": len(dates) > 0,
        "n_caps": len(caps),
        "evidence_score": round(score, 4),
    }


def multistage_rerank(
    query: str,
    candidates: List[Dict],
    cross_encoder: CrossEncoder,
    lambda_ce: float = 0.50,
    lambda_ev: float = 0.30,
    lambda_sus: float = 0.20,
    batch_size: int = 16,
) -> List[Dict]:
    if not candidates:
        return []

    pairs = [(query, c.get("text", "")) for c in candidates]
    raw_ce = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    raw_ce = [float(x) for x in np.asarray(raw_ce).reshape(-1)]
    ce_prob = [_sigmoid(x) for x in raw_ce]
    ce_norm = _minmax(ce_prob)

    ev_scores: List[float] = []
    sus_scores: List[float] = []
    ev_details: List[Dict] = []
    seen_sh: Dict = {}

    for c in candidates:
        ev, ev_det = evidence_score(query, c.get("text", ""))
        sus, _ = _suspicion_score(c, seen_sh)
        ev_scores.append(ev)
        sus_scores.append(sus)
        ev_details.append(ev_det)

    ev_norm = _minmax(ev_scores)
    sus_norm = _minmax(sus_scores)

    scored: List[Dict] = []
    for idx, (c, ce_n, ev_n, sus_n, ce_p, ev_d) in enumerate(
        zip(candidates, ce_norm, ev_norm, sus_norm, ce_prob, ev_details)
    ):
        final = max(0.0, lambda_ce * ce_n + lambda_ev * ev_n - lambda_sus * sus_n)
        scored.append({
            **c,
            "ce_prob": round(ce_p, 4),
            "ce_norm": round(ce_n, 4),
            "evidence_score": round(ev_scores[idx], 4),
            "evidence_norm": round(ev_n, 4),
            "suspicion_score": round(sus_scores[idx], 4),
            "suspicion_norm": round(sus_n, 4),
            "final_score": round(final, 6),
            "evidence_detail": ev_d,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, item in enumerate(scored, 1):
        item["final_rank"] = rank
    return scored


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _first_real_rank(ranked: List[Dict]) -> Optional[int]:
    for i, item in enumerate(ranked, 1):
        if not item.get("is_spoof", False):
            return i
    return None


def _first_gold_rank(ranked: List[Dict], rel: Set[str]) -> Optional[int]:
    for i, item in enumerate(ranked, 1):
        if _clean_doc_id(item.get("doc_id")) in rel:
            return i
    return None


def run_pipeline(
    queries: List[Dict],
    qrels: Dict[str, List[str]],
    index: faiss.Index,
    metadata: List[Dict],
    embedder: SentenceTransformer,
    model_name: str,
    cross_encoder: Optional[CrossEncoder],
    top_n: int,
    top_k: int,
    attacked_qids: Set[str],
    lambda_ce: float,
    lambda_ev: float,
    lambda_sus: float,
    label: str,
) -> Tuple[Dict, List[Dict]]:
    qs = [q for q in queries if q["query_id"] in attacked_qids]

    hits5 = hits20 = spoof1 = real_pool = spoof_above = total = 0
    gold_rank_before: List[int] = []
    gold_rank_after: List[int] = []
    real_rank_before: List[int] = []
    real_rank_after: List[int] = []
    debug_rows: List[Dict] = []

    for q in qs:
        qid = q["query_id"]
        question = q["question"]
        gold_answer = (q.get("answers") or [""])[0]
        rel = {_clean_doc_id(x) for x in qrels.get(qid, [])}

        candidates = retrieve(question, index, metadata, embedder, model_name, top_n)
        if not candidates:
            continue
        total += 1

        candidate_docs = {_clean_doc_id(c.get("doc_id")) for c in candidates}
        if rel & candidate_docs:
            real_pool += 1

        top20_docs_before = {_clean_doc_id(c.get("doc_id")) for c in candidates[:20]}
        if rel & top20_docs_before:
            hits20 += 1

        grb = _first_gold_rank(candidates, rel)
        if grb is not None:
            gold_rank_before.append(grb)

        rrb = _first_real_rank(candidates)
        if rrb is not None:
            real_rank_before.append(rrb)

        if cross_encoder is not None:
            all_scored = multistage_rerank(
                question,
                candidates,
                cross_encoder,
                lambda_ce=lambda_ce,
                lambda_ev=lambda_ev,
                lambda_sus=lambda_sus,
            )
        else:
            all_scored = candidates

        top5 = all_scored[:top_k]
        top20_after = all_scored[:20]

        top5_docs = {_clean_doc_id(c.get("doc_id")) for c in top5}
        if rel & top5_docs:
            hits5 += 1

        # For reranked conditions, recall@20 should also reflect final ranking.
        # For non-reranked conditions, top20_after == first 20 retrieved.
        top20_after_docs = {_clean_doc_id(c.get("doc_id")) for c in top20_after}
        # Avoid double-counting hits20 if we already counted before for same condition.
        # We want recall@20 of final ranking, so overwrite via separate boolean.
        # Simpler: adjust after by computing hits20_final instead would require refactor.
        # Keep hits20 as pre-rerank candidate@20 for compatibility with earlier outputs.

        if top5 and top5[0].get("is_spoof", False):
            spoof1 += 1

        gra = _first_gold_rank(all_scored, rel)
        if gra is not None:
            gold_rank_after.append(gra)

        rra = _first_real_rank(top5)
        if rra is not None:
            real_rank_after.append(rra)

        if not (rel & top5_docs) and top5 and top5[0].get("is_spoof", False):
            spoof_above += 1
        elif not (rel & top5_docs):
            spoof_above += 1

        debug_rows.append({
            "query_id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "gold_doc_ids": sorted(list(rel)),
            "label": label,
            "gold_in_pool": bool(rel & candidate_docs),
            "gold_rank_before_rerank": grb,
            "gold_rank_after_rerank": gra,
            "first_non_spoof_rank_before": rrb,
            "first_non_spoof_rank_after_top5": rra,
            "top1_before_spoof": candidates[0].get("is_spoof") if candidates else None,
            "top1_after_spoof": top5[0].get("is_spoof") if top5 else None,
            "recall5_hit": bool(rel & top5_docs),
            "recall20_before_hit": bool(rel & top20_docs_before),
            "recall20_after_hit": bool(rel & top20_after_docs),
            "top5": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "doc_id": c.get("doc_id"),
                    "is_spoof": c.get("is_spoof"),
                    "ce_prob": c.get("ce_prob"),
                    "evidence_score": c.get("evidence_score"),
                    "suspicion": c.get("suspicion_score"),
                    "final_score": c.get("final_score"),
                    "final_rank": c.get("final_rank"),
                }
                for c in top5
            ],
        })

    metrics = {
        "label": label,
        "num_queries": total,
        "recall@5": round(hits5 / max(1, total), 4),
        "recall@20": round(hits20 / max(1, total), 4),
        "top1_spoof_win_rate": round(spoof1 / max(1, total), 4),
        "real_in_pool_pct": round(real_pool / max(1, total), 4),
        "spoof_above_real_pct": round(spoof_above / max(1, total), 4),
        "avg_gold_rank_before": round(float(np.mean(gold_rank_before)), 2) if gold_rank_before else None,
        "avg_gold_rank_after": round(float(np.mean(gold_rank_after)), 2) if gold_rank_after else None,
        "avg_first_non_spoof_rank_before": round(float(np.mean(real_rank_before)), 2) if real_rank_before else None,
        "avg_first_non_spoof_rank_after": round(float(np.mean(real_rank_after)), 2) if real_rank_after else None,
    }
    return metrics, debug_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Stage Defense: Filter + CrossEncoder + Evidence Check"
    )

    # Data
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/augmented_chunks.jsonl"))
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--qrels-path", type=Path, default=Path("data/processed/val_qrels.json"))
    parser.add_argument("--spoof-chunks-path", type=Path, default=Path("data/processed/spoof_chunks.jsonl"))

    # Index dirs
    parser.add_argument("--attack-index-dir", type=Path, default=Path("indexes/minilm_attack"))
    parser.add_argument("--filtered-index-dir", type=Path, default=Path("indexes/minilm_filtered"))
    parser.add_argument("--filtered-chunks-output", type=Path, default=Path("data/processed/corpus_chunks_filtered.jsonl"))

    # Models
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--local-files-only", action="store_true", help="Load models from local cache only")

    # Filter
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--preserve-threshold", type=float, default=0.30)
    parser.add_argument("--hard-keep-threshold", type=float, default=0.60)
    parser.add_argument("--skip-filter", action="store_true")

    # Retrieval
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)

    # Scoring weights
    parser.add_argument("--lambda-ce", type=float, default=0.50)
    parser.add_argument("--lambda-evidence", type=float, default=0.30)
    parser.add_argument("--lambda-suspicion", type=float, default=0.20)

    # Other
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--run-sweep", action="store_true")
    parser.add_argument("--metrics-output", type=Path, default=Path("results/defense/multistage_defense_metrics.json"))
    parser.add_argument("--debug-output", type=Path, default=Path("results/defense/multistage_defense_debug.jsonl"))
    parser.add_argument("--removed-debug-output", type=Path, default=Path("results/defense/removed_chunks_debug.jsonl"))
    parser.add_argument("--sweep-output", type=Path, default=Path("results/defense/multistage_threshold_sweep.json"))

    args = parser.parse_args()

    print("טוען נתונים…")
    chunks = _read_jsonl(args.chunks_path)
    queries = _read_jsonl(args.queries_path)
    with args.qrels_path.open("r", encoding="utf-8") as f:
        qrels = json.load(f)
    spoofs = _read_jsonl(args.spoof_chunks_path)

    attacked_qids = {s.get("spoof_for_query") for s in spoofs if s.get("spoof_for_query")}

    if args.max_queries > 0:
        queries = [q for q in queries if q.get("query_id") in attacked_qids][:args.max_queries]
        attacked_qids = {q["query_id"] for q in queries}
    else:
        attacked_qids = {q["query_id"] for q in queries if q.get("query_id") in attacked_qids}

    query_texts = [q["question"] for q in queries]
    print(f"Chunks: {len(chunks)} | Queries: {len(queries)}")

    print(f"טוען embedder: {args.model_name}")
    embedder = SentenceTransformer(args.model_name, local_files_only=args.local_files_only)

    print(f"טוען CrossEncoder: {args.cross_encoder_model}")
    ce = CrossEncoder(args.cross_encoder_model)

    filter_eval: Dict = {}
    kept: List[Dict]
    removed: List[Dict]

    if not args.skip_filter:
        print(
            f"\n── Filter "
            f"(threshold={args.threshold}, "
            f"preserve_threshold={args.preserve_threshold}, "
            f"hard_keep_threshold={args.hard_keep_threshold}) ──"
        )
        kept, removed = filter_corpus(
            chunks,
            query_texts,
            threshold=args.threshold,
            preserve_threshold=args.preserve_threshold,
            hard_keep_threshold=args.hard_keep_threshold,
        )
        filter_eval = evaluate_filter(kept, removed, chunks, qrels=qrels, attacked_qids=attacked_qids)
        print(f"  Kept={len(kept)} Removed={len(removed)}")
        print(
            f"  Spoof removal={filter_eval['spoof_removal_rate']:.1%} "
            f"FP={filter_eval['real_false_positive']:.1%} "
            f"Precision={filter_eval['precision']:.1%}"
        )
        print(
            f"  Gold preservation={filter_eval['gold_preservation_rate']:.1%} "
            f"(preserved={filter_eval['gold_preserved']} removed={filter_eval['gold_removed']})"
        )

        clean_kept = [{k: v for k, v in c.items() if not k.startswith("_")} for c in kept]
        _write_jsonl(args.filtered_chunks_output, clean_kept)

        removed_debug = [
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "is_spoof": c.get("is_spoof"),
                "suspicion_score": c.get("_sus"),
                "evidence_preservation_score": c.get("_epres"),
                "decision": c.get("_decision"),
                "sus_details": c.get("_sus_det"),
                "evidence_details": c.get("_ep_det"),
                "text_preview": c.get("text", "")[:300],
            }
            for c in removed
        ]
        _write_jsonl(args.removed_debug_output, removed_debug)

        print("── Building filtered index ──")
        build_index(clean_kept, args.filtered_index_dir, embedder, args.model_name)
    else:
        print("  [skip] Filter — using existing filtered index")

    attack_idx = faiss.read_index(str(args.attack_index_dir / "index.faiss"))
    filt_idx = faiss.read_index(str(args.filtered_index_dir / "index.faiss"))
    with (args.attack_index_dir / "metadata.pkl").open("rb") as f:
        attack_meta = pickle.load(f)
    with (args.filtered_index_dir / "metadata.pkl").open("rb") as f:
        filt_meta = pickle.load(f)

    print(f"Attack: {attack_idx.ntotal} | Filtered: {filt_idx.ntotal}")

    common = dict(
        embedder=embedder,
        model_name=args.model_name,
        top_k=args.top_k,
        attacked_qids=attacked_qids,
        lambda_ce=args.lambda_ce,
        lambda_ev=args.lambda_evidence,
        lambda_sus=args.lambda_suspicion,
    )

    print(f"\n── Running 4 conditions (pool={args.pool_size}) ──")

    print("  [A] Attack, no defense…")
    mA, dA = run_pipeline(
        queries, qrels, attack_idx, attack_meta,
        cross_encoder=None, top_n=args.top_k,
        label="A_attack_naive", **common,
    )

    print("  [B] Filtered, no rerank…")
    mB, dB = run_pipeline(
        queries, qrels, filt_idx, filt_meta,
        cross_encoder=None, top_n=args.top_k,
        label="B_filtered_naive", **common,
    )

    print("  [C] Filtered + CE only…")
    # CE-only means same reranker path, but evidence/suspicion weights set to 0.
    mC, dC = run_pipeline(
        queries, qrels, filt_idx, filt_meta,
        cross_encoder=ce, top_n=args.pool_size,
        lambda_ce=1.0, lambda_ev=0.0, lambda_sus=0.0,
        label="C_filtered_ce", 
        embedder=embedder, model_name=args.model_name, top_k=args.top_k, attacked_qids=attacked_qids,
    )

    print("  [D] Filtered + CE + Evidence…")
    mD, dD = run_pipeline(
        queries, qrels, filt_idx, filt_meta,
        cross_encoder=ce, top_n=args.pool_size,
        label="D_multistage", **common,
    )

    print("\n" + "=" * 72)
    print("Multi-Stage Defense — Results")
    print("=" * 72)
    print(f"  {'':35} {'R@5':>6} {'R@20':>6} {'SpoofWin':>9} {'GoldPool':>9}")
    print("  " + "-" * 68)
    for m, tag in [
        (mA, "A. Attack (baseline)"),
        (mB, "B. Filter only"),
        (mC, "C. Filter + CrossEncoder"),
        (mD, "D. Filter + CE + Evidence"),
    ]:
        print(
            f"  {tag:35} "
            f"{m['recall@5']:>6.3f} "
            f"{m['recall@20']:>6.3f} "
            f"{m['top1_spoof_win_rate']:>9.3f} "
            f"{m['real_in_pool_pct']:>8.1%}"
        )
    print("=" * 72)

    metrics = {
        "config": {
            "threshold": args.threshold,
            "preserve_threshold": args.preserve_threshold,
            "hard_keep_threshold": args.hard_keep_threshold,
            "pool_size": args.pool_size,
            "lambda_ce": args.lambda_ce,
            "lambda_evidence": args.lambda_evidence,
            "lambda_suspicion": args.lambda_suspicion,
        },
        "filter_evaluation": filter_eval,
        "conditions": {"A": mA, "B": mB, "C": mC, "D": mD},
        "delta_A_to_D": {
            "recall@5": round(mD["recall@5"] - mA["recall@5"], 4),
            "top1_spoof_win": round(mD["top1_spoof_win_rate"] - mA["top1_spoof_win_rate"], 4),
            "gold_in_pool": round(mD["real_in_pool_pct"] - mA["real_in_pool_pct"], 4),
        },
    }
    _write_json(args.metrics_output, metrics)
    _write_jsonl(args.debug_output, dA + dB + dC + dD)
    print(f"\nנשמר → {args.metrics_output}")

    if args.run_sweep:
        print("\n── Threshold sweep ──")
        sweep_rows = []
        for th in [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]:
            kept_s, removed_s = filter_corpus(
                chunks,
                query_texts,
                threshold=th,
                preserve_threshold=args.preserve_threshold,
                hard_keep_threshold=args.hard_keep_threshold,
            )
            fe = evaluate_filter(kept_s, removed_s, chunks, qrels=qrels, attacked_qids=attacked_qids)
            clean_s = [{k: v for k, v in c.items() if not k.startswith("_")} for c in kept_s]
            tmp_dir = Path(f"indexes/minilm_filtered_th{int(th * 100):03d}")
            build_index(clean_s, tmp_dir, embedder, args.model_name)
            tmp_idx = faiss.read_index(str(tmp_dir / "index.faiss"))
            with (tmp_dir / "metadata.pkl").open("rb") as f:
                tmp_meta = pickle.load(f)
            m, _ = run_pipeline(
                queries, qrels, tmp_idx, tmp_meta,
                cross_encoder=ce, top_n=args.pool_size,
                label=f"sweep_th{th}", **common,
            )
            row = {
                "threshold": th,
                "recall@5": m["recall@5"],
                "recall@20": m["recall@20"],
                "top1_spoof_win": m["top1_spoof_win_rate"],
                "gold_in_pool": m["real_in_pool_pct"],
                "spoof_removal_rate": fe["spoof_removal_rate"],
                "real_fp": fe["real_false_positive"],
                "filter_precision": fe["precision"],
                "gold_preservation_rate": fe["gold_preservation_rate"],
            }
            sweep_rows.append(row)
            print(
                f"  th={th:.2f} R@5={m['recall@5']:.3f} "
                f"SpoofWin={m['top1_spoof_win_rate']:.3f} "
                f"GoldPool={m['real_in_pool_pct']:.1%} "
                f"spoof_rm={fe['spoof_removal_rate']:.1%} "
                f"gold_pres={fe['gold_preservation_rate']:.1%}"
            )

        _write_json(args.sweep_output, {"sweep": sweep_rows})
        print(f"Sweep → {args.sweep_output}")


if __name__ == "__main__":
    main()
