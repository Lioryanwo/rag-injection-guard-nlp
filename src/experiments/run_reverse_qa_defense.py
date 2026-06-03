"""
run_reverse_qa_defense.py
=========================
הרץ את שכבת ה-Reverse QA על תוצאות retrieval קיימות (Top-20) בלבד.

חשוב:
- לא נוגע ב-corpus, indexes, או retrieval.
- רץ רק על ה-Top-20 שכבר נשלפו.
- מוציא Top-5 מחודשים אחרי ה-reranking.

שימוש:
    python run_reverse_qa_defense.py

או עם פרמטרים:
    python run_reverse_qa_defense.py --retrievers minilm bm25 hybrid --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# טעינת .env
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

# ── פרמטרים ברירת מחדל ────────────────────────────────────────────────────────

RETRIEVERS = ["minilm", "bm25", "hybrid"]

INPUT_PATHS = {
    "minilm":  "results/retrieval/minilm_attack_results.json",
    "bm25":    "results/retrieval/bm25_attack_results.json",
    "hybrid":  "results/retrieval/hybrid_attack_results.json",
}

OUTPUT_PATHS = {
    "minilm":  "results/retrieval/minilm_defense_results.json",
    "bm25":    "results/retrieval/bm25_defense_results.json",
    "hybrid":  "results/retrieval/hybrid_defense_results.json",
}

METRICS_PATHS = {
    "minilm":  "results/retrieval/minilm_defense_metrics.json",
    "bm25":    "results/retrieval/bm25_defense_metrics.json",
    "hybrid":  "results/retrieval/hybrid_defense_metrics.json",
}

QUERIES_PATH      = "data/processed/val_queries.jsonl"
SPOOF_CHUNKS_PATH = "data/processed/spoof_chunks.jsonl"
CACHE_PATH        = "results/reverse_qa_cache.jsonl"

# Reverse QA config
RQA_NUM_QUESTIONS  = 5
RQA_WEIGHT         = 0.25      # בונוס רלוונטיות במשוואת הציון הסופי
RQA_BM25_WEIGHT    = 0.4       # משקל BM25 בתוך ציון ה-RQA
RQA_CE_WEIGHT      = 0.6       # משקל CrossEncoder בתוך ציון ה-RQA
RQA_QG_BACKEND     = "openai"  # openai | heuristic
RQA_OPENAI_MODEL   = "gpt-4o-mini"
RQA_TOP_K          = 20        # כמה chunks לעבד (ה-Top-20 שנשלפו)
FINAL_TOP_K        = 5         # כמה לשמור בסוף


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_queries(path: str) -> Dict[str, str]:
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[q["query_id"]] = q["question"]
    return queries


def _load_spoof_qids(path: str) -> set:
    qids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                qid = row.get("spoof_for_query")
                if qid:
                    qids.add(str(qid))
    return qids


# ── metrics ────────────────────────────────────────────────────────────────────

def _compute_metrics(
    results: Dict[str, List[Dict]],
    qrels: Dict[str, List[str]],
    attacked_qids: set,
    k: int = 5,
) -> Dict[str, Any]:
    # סנן רק queries מותקפות
    results  = {qid: v for qid, v in results.items()  if qid in attacked_qids}
    qrels    = {qid: v for qid, v in qrels.items()    if qid in attacked_qids}

    total = recall_hits = spoof_top1 = 0
    for qid, ranked in results.items():
        if not ranked:
            continue
        total += 1
        top_k_docs = {item["doc_id"] for item in ranked[:k]}
        if any(d in top_k_docs for d in qrels.get(qid, [])):
            recall_hits += 1
        if ranked[0].get("is_spoof", False):
            spoof_top1 += 1

    return {
        "num_queries":        total,
        f"recall@{k}":        round(recall_hits / total, 4) if total else 0.0,
        "top1_spoof_win_rate": round(spoof_top1 / total, 4) if total else 0.0,
    }


# ── Reverse QA core ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def _bm25_score(query: str, questions: List[str]) -> float:
    """BM25-style score מנורמל עם tanh."""
    q_terms = _tokenize(query)
    if not q_terms or not questions:
        return 0.0
    docs = [_tokenize(q) for q in questions]
    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / max(1, n_docs)
    k1, b = 1.5, 0.75
    best = 0.0
    for doc_terms in docs:
        dl = max(1, len(doc_terms))
        score = 0.0
        for term in q_terms:
            tf = doc_terms.count(term)
            if tf == 0:
                continue
            df = sum(1 for d in docs if term in d)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1e-9)))
        best = max(best, score)
    return float(math.tanh(best / 3.0))


def _ce_score(query: str, questions: List[str], ce_model) -> float:
    """CrossEncoder score — max pooling על השאלות."""
    if not questions or ce_model is None:
        return 0.0
    try:
        pairs  = [(query, q) for q in questions]
        scores = ce_model.predict(pairs)
        best   = float(max(float(s) for s in scores))
        return 1.0 / (1.0 + math.exp(-best))
    except Exception:
        return 0.0


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class ReverseQADefense:
    """
    שכבת ההגנה שרצה על ה-Top-20 שנשלפו:
    1. יוצרת שאלות מכל chunk (LLM או heuristic)
    2. משווה לquery המקורי (BM25 + CrossEncoder)
    3. מוסיפה בונוס רלוונטיות לציון הקיים
    4. מחזירה Top-K מחודשים
    """

    def __init__(
        self,
        num_questions:  int   = RQA_NUM_QUESTIONS,
        rqa_weight:     float = RQA_WEIGHT,
        bm25_weight:    float = RQA_BM25_WEIGHT,
        ce_weight:      float = RQA_CE_WEIGHT,
        qg_backend:     str   = RQA_QG_BACKEND,
        openai_model:   str   = RQA_OPENAI_MODEL,
        cache_path:     str   = CACHE_PATH,
        use_ce:         bool  = True,
        top_k:          int   = RQA_TOP_K,
    ):
        self.num_questions = num_questions
        self.rqa_weight    = rqa_weight
        self.bm25_weight   = bm25_weight
        self.ce_weight     = ce_weight
        self.qg_backend    = qg_backend
        self.openai_model  = openai_model
        self.cache_path    = cache_path
        self.use_ce        = use_ce
        self.top_k         = top_k

        # lazy init
        self._openai_client = None
        self._ce_model      = None
        self._cache         = self._load_cache()

    # ── cache ──────────────────────────────────────────────────────────────────

    def _load_cache(self) -> Dict[str, Dict]:
        cache = {}
        p = Path(self.cache_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        key = row.get("cache_key")
                        if key:
                            cache[key] = row
                    except Exception:
                        pass
        print(f"[cache] נטענו {len(cache)} רשומות מ-{self.cache_path}")
        return cache

    def _cache_key(self, query: str, text: str) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(query.encode("utf-8"))
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        h.update(b"\0")
        h.update(f"{self.num_questions}:{self.qg_backend}:{self.openai_model}".encode())
        return h.hexdigest()[:24]

    def _cache_set(self, key: str, value: Dict) -> None:
        self._cache[key] = value
        p = Path(self.cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        row = dict(value)
        row["cache_key"] = key
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── question generation ────────────────────────────────────────────────────

    def _generate_questions(self, text: str) -> List[str]:
        if self.qg_backend == "openai":
            return self._generate_openai(text)
        return self._generate_heuristic(text)

    def _generate_openai(self, text: str) -> List[str]:
        from openai import OpenAI

        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            self._openai_client = OpenAI(api_key=api_key)

        prompt = (
            f"Generate exactly {self.num_questions} short, concrete questions "
            f"that can be answered directly from the document. "
            f"Avoid vague or generic questions. "
            f"Return only a JSON list of strings.\n\nDocument:\n{text[:3000]}"
        )

        try:
            resp = self._openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You generate answerable retrieval-evaluation questions."},
                    {"role": "user", "content": prompt},
                ],
            )

            content = resp.choices[0].message.content or "[]"
            return self._parse_questions(content)

        except Exception as e:
            print(f"[OpenAI ERROR] {e}", flush=True)
            return self._generate_heuristic(text)

    def _generate_heuristic(self, text: str) -> List[str]:
        import re
        sent_re   = re.compile(r"(?<=[.!?])\s+")
        sentences = [s.strip() for s in sent_re.split(text) if len(s.strip()) > 30]
        questions = []
        for s in sentences[:self.num_questions]:
            words     = _tokenize(s)
            key_terms = " ".join(words[:min(8, len(words))])
            questions.append(f"What does the document say about {key_terms}?")
        while len(questions) < min(3, self.num_questions):
            questions.append("What is the main fact stated in the document?")
        return questions[:self.num_questions]

    def _parse_questions(self, raw: str) -> List[str]:
        import re
        raw = (raw or "").strip()
        items: List[str] = []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                items = [str(x).strip() for x in parsed]
        except Exception:
            pass
        if not items:
            for line in raw.splitlines():
                line = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
                if line.endswith("?") and len(line) > 8:
                    items.append(line)
        seen, clean = set(), []
        for q in items:
            q = q.strip().strip('"')
            if not q:
                continue
            if not q.endswith("?"):
                q += "?"
            if q.lower() not in seen:
                clean.append(q)
                seen.add(q.lower())
        return clean[:self.num_questions]

    # ── CrossEncoder lazy init ─────────────────────────────────────────────────

    def _get_ce(self):
        if self._ce_model is None and self.use_ce:
            from sentence_transformers import CrossEncoder
            self._ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("[CE] נטען cross-encoder/ms-marco-MiniLM-L-6-v2")
        return self._ce_model

    # ── score one document ─────────────────────────────────────────────────────

    def score_doc(self, query: str, text: str) -> Dict[str, Any]:
        key    = self._cache_key(query, text)
        cached = self._cache.get(key)
        if cached:
            return cached

        questions  = self._generate_questions(text)
        bm25       = _bm25_score(query, questions)
        ce         = _ce_score(query, questions, self._get_ce())
        rqa_score  = self.bm25_weight * bm25 + self.ce_weight * ce

        result = {
            "generated_questions": questions,
            "rqa_bm25_score":      round(bm25, 4),
            "rqa_ce_score":        round(ce,   4),
            "rqa_score":           round(rqa_score, 4),
        }
        self._cache_set(key, result)
        return result

    # ── rerank one query ───────────────────────────────────────────────────────

    def rerank(self, query: str, ranked: List[Dict], final_k: int) -> List[Dict]:
        """
        מקבל רשימת chunks מדורגים (כבר Top-20).
        מחזיר Top-K מחודשים אחרי הוספת בונוס ה-RQA.
        """
        pool = [dict(d) for d in ranked[:self.top_k]]
        tail = [dict(d) for d in ranked[self.top_k:]]

        # 1. חשב ציוני RQA לכל chunk
        rqa_scores = []
        for doc in pool:
            text = doc.get("text", "")
            if text:
                info = self.score_doc(query, text)
            else:
                info = {"generated_questions": [], "rqa_bm25_score": 0.0,
                        "rqa_ce_score": 0.0, "rqa_score": 0.0}
            doc.update(info)
            rqa_scores.append(float(info["rqa_score"]))

        # 2. נרמל ציוני RQA (min-max)
        norm_rqa = _minmax(rqa_scores)

        # 3. הוסף בונוס לציון הקיים
        for doc, nrq in zip(pool, norm_rqa):
            base_score            = float(doc.get("score", 0.0))
            doc["rqa_bonus"]      = round(self.rqa_weight * nrq, 4)
            doc["score_before_rqa"] = base_score
            doc["score"]          = round(base_score + doc["rqa_bonus"], 4)

        # 4. מיין מחדש
        pool.sort(key=lambda d: d["score"], reverse=True)

        return pool[:final_k] + tail


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reverse QA defense על Top-20 קיימים — בלי לגעת ב-corpus"
    )
    parser.add_argument("--retrievers",    nargs="+", default=RETRIEVERS,
                        choices=["minilm", "bm25", "hybrid"])
    parser.add_argument("--queries-path",  default=QUERIES_PATH)
    parser.add_argument("--spoof-path",    default=SPOOF_CHUNKS_PATH)
    parser.add_argument("--cache-path",    default=CACHE_PATH)
    parser.add_argument("--top-k",         type=int, default=FINAL_TOP_K,
                        help="כמה results לשמור בסוף (ברירת מחדל: 5)")
    parser.add_argument("--rqa-top-k",     type=int, default=RQA_TOP_K,
                        help="על כמה chunks לרוץ (ברירת מחדל: 20)")
    parser.add_argument("--rqa-weight",    type=float, default=RQA_WEIGHT)
    parser.add_argument("--num-questions", type=int,   default=RQA_NUM_QUESTIONS)
    parser.add_argument("--qg-backend",    default=RQA_QG_BACKEND,
                        choices=["openai", "heuristic"])
    parser.add_argument("--openai-model",  default=RQA_OPENAI_MODEL)
    parser.add_argument("--no-ce",         action="store_true",
                        help="כבה CrossEncoder (מהיר יותר, פחות מדויק)")
    args = parser.parse_args()

    print("=" * 60)
    print("Reverse QA Defense — רץ על Top-20 קיימים")
    print(f"Retrievers:    {args.retrievers}")
    print(f"QG backend:    {args.qg_backend}")
    print(f"Final Top-K:   {args.top_k}")
    print(f"RQA Top-K:     {args.rqa_top_k}")
    print(f"RQA weight:    {args.rqa_weight}")
    print(f"Num questions: {args.num_questions}")
    print("=" * 60)

    # טען queries וspoof qids
    queries      = _load_queries(args.queries_path)
    attacked_qids = _load_spoof_qids(args.spoof_path)

    # קרא qrels
    qrels_path = "data/processed/val_qrels.json"
    qrels = _load_json(qrels_path) if Path(qrels_path).exists() else {}

    # אתחל defense
    defense = ReverseQADefense(
        num_questions = args.num_questions,
        rqa_weight    = args.rqa_weight,
        qg_backend    = args.qg_backend,
        openai_model  = args.openai_model,
        cache_path    = args.cache_path,
        use_ce        = not args.no_ce,
        top_k         = args.rqa_top_k,
    )

    for ret in args.retrievers:
        input_path   = INPUT_PATHS[ret]
        output_path  = OUTPUT_PATHS[ret]
        metrics_path = METRICS_PATHS[ret]

        if not Path(input_path).exists():
            print(f"\n[SKIP] {input_path} לא קיים")
            continue
            
        print(f"\n{'─'*60}")
        print(f"Retriever: {ret.upper()}")
        print(f"קורא: {input_path}")

        all_results = _load_json(input_path)
        defended: Dict[str, List[Dict]] = {}

        total  = len(all_results)
        spoof_top1_count = 0

        for i, (qid, ranked) in enumerate(all_results.items(), 1):
            query = queries.get(qid, "")
            if not ranked:
                defended[qid] = []
                continue

            reranked       = defense.rerank(query, ranked, args.top_k)
            defended[qid]  = reranked

            if reranked and reranked[0].get("is_spoof", False):
                spoof_top1_count += 1

            if i % 100 == 0:
                print(f"  {i}/{total} queries | spoof_top1_so_far={spoof_top1_count/i:.3f}")

            if i % 20 == 0:
                print(f"  {i}/{total} queries | spoof_top1_so_far={spoof_top1_count/i:.3f}", flush=True)    

        # שמור תוצאות
        _save_json(output_path, defended)
        spoof_rate = spoof_top1_count / total if total else 0.0
        print(f"\nנשמר → {output_path}")
        print(f"Top-1 spoof rate after RQA defense: {spoof_rate:.4f}")

        # חשב ושמור metrics
        metrics = _compute_metrics(defended, qrels, attacked_qids, k=args.top_k)
        metrics["retriever"]   = ret
        metrics["qg_backend"]  = args.qg_backend
        metrics["rqa_weight"]  = args.rqa_weight
        _save_json(metrics_path, metrics)
        print(f"Metrics: recall@{args.top_k}={metrics[f'recall@{args.top_k}']:.4f}  "
              f"spoof_win={metrics['top1_spoof_win_rate']:.4f}")

    print("\n" + "=" * 60)
    print("סיום! תריץ עכשיו:")
    print("  python -m src.evaluation.plot_results")
    print("=" * 60)


if __name__ == "__main__":
    main()
