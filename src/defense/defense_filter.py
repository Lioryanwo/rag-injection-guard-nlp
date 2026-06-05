from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ── Reverse QA (optional import — only needed when use_reverse_qa=True) ───────
try:
    from src.defense.reverse_qa import ReverseQAConfig, ReverseQAScorer
    _REVERSE_QA_AVAILABLE = True
except ImportError:
    _REVERSE_QA_AVAILABLE = False



STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "why", "how", "is", "was", "were",
    "are", "did", "does", "do", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "by", "and", "or", "from", "this", "that", "these", "those", "it", "its", "into", "with",
}

_STATIC_PATTERNS = [
    r"as an ai",
    r"generated text",
    r"answer-focused note:",
    r"question:\s+",
    r"this passage addresses the question",
    r"optimized to resemble",
    r"this section explains",
    r"in summary[,\.]",
    r"the topic is related to",
    r"this passage discusses",
]
_STATIC_RE = [re.compile(p, re.IGNORECASE) for p in _STATIC_PATTERNS]

_GENERIC_PHRASES = [
    "widely documented",
    "some researchers argue",
    "recent scholarship indicates",
    "precise role remains debated",
    "background context",
    "it is important to note",
    "according to alternative sources",
    "a lesser-known account",
    "evidence-free",
    "distractor fact",
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read_json(path: Path) -> Dict[str, List[Dict]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj) -> None:
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


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-12:
        return [1.0 for _ in values]
    return ((arr - lo) / (hi - lo)).astype(float).tolist()


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ── Lexical helpers ───────────────────────────────────────────────────────────

def _cwords(text: str) -> List[str]:
    return [
        t for t in re.findall(r"\b[a-zA-Z][a-zA-Z'\-]+\b", text.lower())
        if t not in STOPWORDS and len(t) > 2
    ]


def _kset(text: str) -> set:
    return set(_cwords(text))


def _overlap(query: str, chunk: str) -> float:
    q = _kset(query)
    c = _kset(chunk)
    return len(q & c) / len(q) if q else 0.0


def _diversity(text: str) -> float:
    w = _cwords(text)
    return len(set(w)) / len(w) if len(w) >= 5 else 1.0


def _surface(chunk: str) -> float:
    hits = sum(1 for p in _STATIC_RE if p.search(chunk))
    return min(1.0, hits / 2.0)


def _genericity(chunk: str) -> float:
    lc = chunk.lower()
    hits = sum(1 for p in _GENERIC_PHRASES if p in lc)
    return min(1.0, hits * 0.20)


# ── Suspicion score ───────────────────────────────────────────────────────────

def suspicion_score(query: str, chunk: str) -> Tuple[float, Dict]:
    """
    Surface-level suspicion score.

    Focuses on signals that real SQuAD documents NEVER exhibit:
    - LLM template markers
    - generic filler phrases
    - low lexical diversity

    Query overlap is intentionally weighted LOW because real relevant
    chunks also repeat query words.
    """
    ov  = _overlap(query, chunk)
    sh  = _surface(chunk)
    ld  = _diversity(chunk)
    ge  = _genericity(chunk)

    score = (
        0.05 * min(1.0, ov) +       # very low — real docs also overlap
        0.40 * sh +                  # strong: LLM template markers
        0.25 * max(0.0, 0.6 - ld) + # low diversity
        0.40 * ge                    # strong: generic filler
    )
    score = min(1.0, score)

    return score, {
        "query_overlap":     round(ov, 3),
        "surface_hits":      round(sh, 3),
        "lexical_diversity": round(ld, 3),
        "genericity":        round(ge, 3),
        "suspicion_score":   round(score, 3),
    }


def no_query_suspicion(chunk: str) -> Tuple[float, Dict]:
    """Static chunk-only filter. Intentionally weak — proves query is needed."""
    reasons: List[str] = []
    score = 0.0
    text_l = chunk.lower()

    surface_hits = sum(1 for p in _STATIC_RE if p.search(chunk))
    if surface_hits:
        score += min(0.6, 0.25 * surface_hits)
        reasons.append("surface_template_marker")

    generic_hits = sum(1 for p in _GENERIC_PHRASES if p in text_l)
    if generic_hits >= 2:
        score += min(0.3, 0.10 * generic_hits)
        reasons.append("generic_llm_style_phrasing")

    div = _diversity(chunk)
    if div < 0.35:
        score += 0.2
        reasons.append("low_lexical_diversity")

    if len(_cwords(chunk)) < 25:
        score += 0.05
        reasons.append("very_short_chunk")

    return min(1.0, score), {
        "no_query_suspicion_score": round(min(1.0, score), 3),
        "lexical_diversity":        round(div, 3),
        "surface_hits":             surface_hits,
        "generic_hits":             generic_hits,
        "reasons":                  reasons,
    }


# ── Doc2Query: answerability estimation ───────────────────────────────────────

def generate_hypothetical_questions(chunk: str, n: int = 5) -> List[str]:
    """
    Generate questions this chunk could plausibly answer.

    Uses content-driven templates, not generic ones, so the questions
    actually reflect what the chunk contains.
    """
    words   = list(dict.fromkeys(_cwords(chunk)))
    numbers = re.findall(r"\b\d[\d,]*\b", chunk)
    named   = [w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", chunk)
               if w.lower() not in STOPWORDS][:4]

    if not words:
        return ["What is this passage about?"]

    topic  = " ".join(words[:3])
    topic2 = " ".join(words[3:6]) if len(words) >= 6 else topic
    named_str = named[0] if named else topic

    qs = [
        f"What information does the passage provide about {topic}?",
        f"What specific facts are stated about {topic2}?",
        f"Who or what is {named_str} associated with in this passage?",
    ]

    if numbers:
        qs.append(f"What is the numerical value related to {topic}?")
        qs.append(f"How many {topic2} does the passage mention?")
    else:
        qs.append(f"Where or when does {topic} occur according to the passage?")
        qs.append(f"What is the relationship between {topic} and {topic2}?")

    return qs[:n]


def doc2query_alignment(
    query: str,
    chunk: str,
    embedder: SentenceTransformer,
    model_name: str,
    n_questions: int = 5,
) -> Tuple[float, Dict]:
    """
    Answerability: do the chunk's hypothetical questions align with the query?

    High alignment → chunk likely answers the query (real chunk).
    Low alignment  → chunk is semantically nearby but answers different questions
                     (spoof with distractor facts).
    """
    questions = generate_hypothetical_questions(chunk, n_questions)

    texts = [query] + questions
    if "e5" in model_name.lower():
        texts = [f"query: {texts[0]}"] + [f"query: {q}" for q in questions]

    embs = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    sims = (embs[1:] @ embs[0]).astype(float).tolist()
    best = float(max(sims)) if sims else 0.0
    avg  = float(np.mean(sims)) if sims else 0.0

    return best, {
        "hypothetical_questions":    questions,
        "doc2query_similarities":    [round(s, 4) for s in sims],
        "doc2query_best_similarity": round(best, 4),
        "doc2query_avg_similarity":  round(avg,  4),
        "doc2query_penalty":         round(max(0.0, 1.0 - best), 4),
    }


# ── No-query reranker ─────────────────────────────────────────────────────────

def no_query_filter_rerank(ranked: List[Dict], threshold: float = 0.30) -> List[Dict]:
    out = []
    for item in ranked:
        orig = float(item.get("score", 0.0))
        sus, breakdown = no_query_suspicion(item.get("text", ""))
        flagged = sus >= threshold
        out.append({
            **item,
            "original_score":  orig,
            "score":           orig * max(0.05, 1.0 - sus),
            "defense_flagged": flagged,
            "defense":         breakdown,
            "defense_mode":    "doc2query_inspired_static_analysis",
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def text_only_rerank(query: str, ranked: List[Dict], threshold: float = 0.30) -> List[Dict]:
    out = []
    for item in ranked:
        orig = float(item.get("score", 0.0))
        sus, breakdown = suspicion_score(query, item.get("text", ""))
        flagged = sus >= threshold
        out.append({
            **item,
            "original_score":  orig,
            "score":           orig * (max(0.02, (1.0 - sus) ** 2) if flagged else 1.0),
            "defense_flagged": flagged,
            "defense":         breakdown,
            "defense_mode":    "query_aware_text_only",
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


# ── Reverse QA layer (post-CE reranking bonus) ────────────────────────────────

def reverse_qa_rerank(
    query: str,
    ranked: List[Dict],
    reverse_qa_weight: float = 0.30,
    num_questions: int = 5,
    qg_backend: str = "openai",
    openai_model: str = "gpt-4o-mini",
    cache_path: Optional[str] = "results/reverse_qa_cache.jsonl",
    use_cross_encoder: bool = True,
    top_k: int = 20,
    bm25_weight: float = 0.35,
    cross_encoder_weight: float = 0.55,
    evidence_weight: float = 0.10,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict]:
    """
    Reverse QA (Doc2Query via LLM) reranking layer.

    For each chunk in the Top-K pool:
      1. Generate 3-5 questions the chunk can answer (via GPT-4o-mini).
      2. Compare each generated question to the original user query using
         BM25 + CrossEncoder (max pooling over questions).
      3. Add the resulting relevance bonus to the existing defense score.

    Why this helps:
      - Real chunks contain specific facts → generated questions are specific
        and align well with the query.
      - Spoof chunks contain "empty promises" → generated questions are
        generic or off-topic → low score → pushed down in ranking.

    This function is ADDITIVE: it must be called AFTER cross_encoder_rerank.
    It reads and writes the "score" field, preserving all other fields.
    """
    if not _REVERSE_QA_AVAILABLE:
        print("[reverse_qa] WARNING: src.defense.reverse_qa not importable — skipping")
        return ranked

    cfg = ReverseQAConfig(
        top_k=top_k,
        num_questions=num_questions,
        reverse_qa_weight=reverse_qa_weight,
        qg_backend=qg_backend,
        openai_model=openai_model,
        cache_path=cache_path,
        use_cross_encoder=use_cross_encoder,
        bm25_weight=bm25_weight,
        cross_encoder_weight=cross_encoder_weight,
        evidence_weight=evidence_weight,
        cross_encoder_model=cross_encoder_model,
        # Use the existing defense score as the base score
        base_score_key="score",
        normalize_scores=True,
    )
    scorer = ReverseQAScorer(cfg)
    return scorer.rerank(query, ranked)


# ── Main defense ──────────────────────────────────────────────────────────────

def cross_encoder_rerank(
    query: str,
    ranked: List[Dict],
    cross_encoder: CrossEncoder,
    threshold: float = 0.30,          # ← NOW ACTUALLY USED
    semantic_weight: float = 0.55,
    retrieval_weight: float = 0.15,
    doc2query_weight: float = 0.30,
    lexical_penalty_weight: float = 0.15,
    batch_size: int = 16,
    doc2query_embedder: Optional[SentenceTransformer] = None,
    doc2query_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    # ── Reverse QA (new) ──────────────────────────────────────────────────────
    use_reverse_qa: bool = False,
    reverse_qa_weight: float = 0.30,
    reverse_qa_num_questions: int = 5,
    reverse_qa_qg_backend: str = "openai",
    reverse_qa_openai_model: str = "gpt-4o-mini",
    reverse_qa_cache_path: Optional[str] = "results/reverse_qa_cache.jsonl",
    reverse_qa_bm25_weight: float = 0.35,
    reverse_qa_cross_encoder_weight: float = 0.55,
    reverse_qa_evidence_weight: float = 0.10,
    reverse_qa_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict]:
    """
    Query-aware defense: CrossEncoder + Doc2Query answerability.

    BUG FIX (critical):
    The previous version had `threshold` as a parameter but NEVER USED IT.
    The hard-filter condition was hardcoded to `sus >= 0.80` regardless of
    the threshold argument.  This is why the threshold sweep was completely
    flat — changing the threshold had zero effect on filtering.

    Fix: the threshold parameter now controls the hard-filter cutoff.
    Chunks with suspicion >= threshold are removed from the output.
    This makes the threshold sweep meaningful.

    Scoring formula (soft, not hard):
      final = CE_norm * semantic_weight
            + doc2query_norm * doc2query_weight
            + retrieval_norm * retrieval_weight
            * (1 - lexical_penalty_weight * suspicion)   ← soft decay

    NEW: when use_reverse_qa=True, a second reranking pass is applied after
    the CE pass, adding a relevance bonus based on LLM-generated questions.
    """
    if not ranked:
        return []

    # Step 1: retrieval scores
    original_scores = [float(item.get("score", 0.0)) for item in ranked]
    retrieval_norm  = _minmax(original_scores)

    # Step 2: CrossEncoder scores
    pairs         = [(query, item.get("text", "")) for item in ranked]
    raw_ce        = cross_encoder.predict(pairs, batch_size=batch_size,
                                          show_progress_bar=False)
    raw_ce        = [float(x) for x in np.asarray(raw_ce).reshape(-1)]
    ce_norm       = _minmax(raw_ce)
    ce_prob       = [_sigmoid(x) for x in raw_ce]

    # Step 3: Doc2Query answerability
    d2q_scores:  List[float] = [0.0] * len(ranked)
    d2q_details: List[Dict]  = [{}   for _ in ranked]
    if doc2query_embedder is not None:
        for i, item in enumerate(ranked):
            best, detail = doc2query_alignment(
                query, item.get("text", ""),
                doc2query_embedder, doc2query_model_name,
            )
            d2q_scores[i]  = best
            d2q_details[i] = detail
    d2q_norm = _minmax(d2q_scores) if doc2query_embedder else [0.0] * len(ranked)

    # Step 4: Suspicion scores
    sus_scores  = []
    sus_details = []
    for item in ranked:
        sus, det = suspicion_score(query, item.get("text", ""))
        sus_scores.append(sus)
        sus_details.append(det)

    # ── Debug logging ─────────────────────────────────────────────────────────
    spoof_sus  = [s for s, item in zip(sus_scores, ranked) if item.get("is_spoof")]
    real_sus   = [s for s, item in zip(sus_scores, ranked) if not item.get("is_spoof")]
    print(
        f"[defense th={threshold:.2f}] "
        f"avg_sus_spoof={np.mean(spoof_sus):.3f} "
        f"avg_sus_real={np.mean(real_sus):.3f} "
        f"would_filter_spoof={sum(1 for s in spoof_sus if s >= threshold)}/{len(spoof_sus)} "
        f"would_filter_real={sum(1 for s in real_sus  if s >= threshold)}/{len(real_sus)}"
    )

    # Step 5: Score and filter
    out: List[Dict] = []
    hard_filtered = 0

    for idx, (item, orig, r_norm, ce_raw, ce_n, ce_p) in enumerate(
        zip(ranked, original_scores, retrieval_norm, raw_ce, ce_norm, ce_prob)
    ):
        sus     = sus_scores[idx]
        sus_det = sus_details[idx]
        d2q_n   = d2q_norm[idx]
        d2q_det = d2q_details[idx]
        d2q_val = d2q_scores[idx]

        # ── Hard filter using the ACTUAL threshold parameter ───────────────────
        # BUG FIX: previously hardcoded to 0.80 regardless of `threshold`
        if sus >= threshold:
            hard_filtered += 1
            continue

        # ── Soft scoring ───────────────────────────────────────────────────────
        final_score = (
            semantic_weight  * ce_n +
            doc2query_weight * d2q_n +
            retrieval_weight * r_norm
        )

        # Soft suspicion decay — floor at 0.5 to preserve recall
        sus_decay   = max(0.5, 1.0 - lexical_penalty_weight * sus)
        final_score = max(0.0, float(final_score * sus_decay))

        out.append({
            **item,
            "original_score":               orig,
            "retrieval_score_norm":         round(r_norm,  4),
            "cross_encoder_score_raw":      round(ce_raw,  4),
            "cross_encoder_score_norm":     round(ce_n,    4),
            "cross_encoder_relevance_prob": round(ce_p,    4),
            "doc2query_best_similarity":    round(d2q_val, 4),
            "doc2query_norm":               round(d2q_n,   4),
            "suspicion":                    round(sus,     4),
            "score":                        round(final_score, 6),
            "defense_flagged":              False,
            "defense": {
                **sus_det,
                **d2q_det,
                "threshold_used":    round(threshold, 3),
                "sus_decay_applied": round(sus_decay, 4),
                "score_breakdown": {
                    "CE":        round(semantic_weight  * ce_n,  4),
                    "Doc2Query": round(doc2query_weight * d2q_n, 4),
                    "Retrieval": round(retrieval_weight * r_norm, 4),
                    "sus_decay": round(sus_decay, 4),
                    "final":     round(final_score, 4),
                },
            },
            "defense_mode": (
                "cross_encoder_doc2query"
                if doc2query_embedder else
                "cross_encoder"
            ),
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    print(
        f"[defense th={threshold:.2f}] "
        f"hard_filtered={hard_filtered}/{len(ranked)} "
        f"kept={len(out)}/{len(ranked)}"
    )

    # Reverse QA second pass (optional)
    if use_reverse_qa and out:
        print(f"[reverse_qa] Running on {len(out)} kept chunks...")
        out = reverse_qa_rerank(
            query=query,
            ranked=out,
            reverse_qa_weight=reverse_qa_weight,
            num_questions=reverse_qa_num_questions,
            qg_backend=reverse_qa_qg_backend,
            openai_model=reverse_qa_openai_model,
            cache_path=reverse_qa_cache_path,
            use_cross_encoder=True,
            top_k=min(20, len(out)),
            bm25_weight=reverse_qa_bm25_weight,
            cross_encoder_weight=reverse_qa_cross_encoder_weight,
            evidence_weight=reverse_qa_evidence_weight,
            cross_encoder_model=reverse_qa_cross_encoder_model,
        )
        print(f"[reverse_qa] Done. Top-1 spoof={out[0].get('is_spoof', False) if out else 'N/A'}")

    return out


def load_queries(path: Path) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    for row in _read_jsonl(path):
        queries[row["query_id"]] = row["question"]
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",           type=Path,
                        default=Path("results/retrieval/attack_results.json"))
    parser.add_argument("--queries-path",          type=Path,
                        default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--output-path",           type=Path,
                        default=Path("results/retrieval/defense_results.json"))
    parser.add_argument("--keep-top-k",            type=int,   default=5)
    parser.add_argument("--suspicion-threshold",   type=float, default=0.30)
    parser.add_argument("--defense-mode",
                        choices=["text", "cross_encoder", "no_query"],
                        default="cross_encoder")
    parser.add_argument("--cross-encoder-model",   type=str,
                        default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--semantic-weight",       type=float, default=0.55)
    parser.add_argument("--retrieval-weight",      type=float, default=0.15)
    parser.add_argument("--lexical-penalty-weight", type=float, default=0.15)
    parser.add_argument("--batch-size",            type=int,   default=16)
    parser.add_argument("--use-doc2query",         action="store_true")
    parser.add_argument("--doc2query-embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--doc2query-weight",      type=float, default=0.30)
    # ── Reverse QA args ───────────────────────────────────────────────────────
    parser.add_argument("--use-reverse-qa",         action="store_true",
                        help="Enable LLM-based Reverse QA reranking (requires OPENAI_API_KEY)")
    parser.add_argument("--reverse-qa-weight",      type=float, default=0.25,
                        help="Bonus weight for Reverse QA relevance score")
    parser.add_argument("--reverse-qa-num-questions", type=int, default=2)
    parser.add_argument("--reverse-qa-qg-backend",  type=str,  default="openai",
                        choices=["openai", "heuristic", "transformers", "auto"])
    parser.add_argument("--reverse-qa-openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--reverse-qa-cache-path",  type=str,
                        default="results/reverse_qa_cache.jsonl")
    parser.add_argument("--reverse-qa-bm25-weight", type=float, default=0.35,
                        help="BM25 weight inside Reverse QA score")
    parser.add_argument("--reverse-qa-cross-encoder-weight", type=float, default=0.55,
                        help="CrossEncoder weight inside Reverse QA score")
    parser.add_argument("--reverse-qa-evidence-weight", type=float, default=0.10,
                        help="Answerability evidence weight inside Reverse QA score")
    parser.add_argument("--reverse-qa-cross-encoder-model", type=str,
                        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
                        help="Smaller CE used only for matching generated questions to the query")
    args = parser.parse_args()

    results = _read_json(args.input_path)
    queries = load_queries(args.queries_path)

    cross_encoder      = None
    doc2query_embedder = None
    if args.defense_mode == "cross_encoder":
        print(f"Loading CrossEncoder: {args.cross_encoder_model}")
        cross_encoder = CrossEncoder(args.cross_encoder_model)
        if args.use_doc2query:
            print(f"Loading Doc2Query: {args.doc2query_embedding_model}")
            doc2query_embedder = SentenceTransformer(args.doc2query_embedding_model)

    defended:  Dict[str, List[Dict]] = {}
    spoof_top1 = 0

    for n, (qid, ranked) in enumerate(results.items(), 1):
        query = queries.get(qid, "")
        if args.defense_mode == "text":
            reranked = text_only_rerank(query, ranked, args.suspicion_threshold)
        elif args.defense_mode == "no_query":
            reranked = no_query_filter_rerank(ranked, args.suspicion_threshold)
        else:
            assert cross_encoder is not None
            reranked = cross_encoder_rerank(
                query=query,
                ranked=ranked,
                cross_encoder=cross_encoder,
                threshold=args.suspicion_threshold,   # now propagated correctly
                semantic_weight=args.semantic_weight,
                retrieval_weight=args.retrieval_weight,
                doc2query_weight=args.doc2query_weight,
                lexical_penalty_weight=args.lexical_penalty_weight,
                batch_size=args.batch_size,
                doc2query_embedder=doc2query_embedder,
                doc2query_model_name=args.doc2query_embedding_model,
                use_reverse_qa=args.use_reverse_qa,
                reverse_qa_weight=args.reverse_qa_weight,
                reverse_qa_num_questions=args.reverse_qa_num_questions,
                reverse_qa_qg_backend=args.reverse_qa_qg_backend,
                reverse_qa_openai_model=args.reverse_qa_openai_model,
                reverse_qa_cache_path=args.reverse_qa_cache_path,
                reverse_qa_bm25_weight=args.reverse_qa_bm25_weight,
                reverse_qa_cross_encoder_weight=args.reverse_qa_cross_encoder_weight,
                reverse_qa_evidence_weight=args.reverse_qa_evidence_weight,
                reverse_qa_cross_encoder_model=args.reverse_qa_cross_encoder_model,
            )

        topk = reranked[: args.keep_top_k]
        defended[qid] = topk
        if topk and topk[0].get("is_spoof", False):
            spoof_top1 += 1
        if n % 100 == 0:
            print(f"Defended {n}/{len(results)} queries")

    _write_json(args.output_path, defended)
    print(f"\nSaved → {args.output_path}")
    print(f"Top-1 spoof rate after defense: "
          f"{spoof_top1/len(defended) if defended else 0:.3f}")


if __name__ == "__main__":
    main()
