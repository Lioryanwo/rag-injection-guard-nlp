"""
Reverse QA / Doc2Query defense signal for RAG spoofing experiments.

Purpose
-------
Run only on the already-retrieved Top-K candidates (default Top-20), generate
3-5 answerable questions from each chunk, compare them to the original user
query, and add a relevance bonus for reranking.

This module is intentionally additive: it does not replace or modify the
existing suspicion-based defense. Use `rerank_with_reverse_qa(...)` after your
current Top-20 retrieval / defense scoring step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

# טוען את ה-.env אוטומטית כדי שה-OPENAI_API_KEY יהיה זמין
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
except ImportError:
    pass

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ReverseQAConfig:
    top_k: int = 20
    num_questions: int = 5
    bm25_weight: float = 0.4
    cross_encoder_weight: float = 0.6
    reverse_qa_weight: float = 0.25
    base_score_key: str = "defense_score"  # fallback: final_score, score, similarity
    text_keys: Tuple[str, ...] = ("text", "chunk", "content", "document")
    query_key: str = "query"
    cache_path: Optional[str] = "results/reverse_qa_cache.jsonl"
    qg_backend: str = "auto"  # auto | openai | transformers | heuristic
    openai_model: str = "gpt-4o-mini"
    hf_qg_model: str = "google/flan-t5-small"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_cross_encoder: bool = True
    normalize_scores: bool = True


class JsonlCache:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self.data: Dict[str, Dict[str, Any]] = {}
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = row.get("cache_key")
                    if key:
                        self.data[key] = row

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(value)
        row["cache_key"] = key
        self.data[key] = row
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _sig(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def _extract_text(doc: MutableMapping[str, Any], text_keys: Sequence[str]) -> str:
    for k in text_keys:
        v = doc.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _base_score(doc: MutableMapping[str, Any], key: str) -> float:
    for k in (key, "defense_score", "final_score", "score", "similarity", "retrieval_score"):
        try:
            return float(doc[k])
        except Exception:
            continue
    return 0.0


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class QuestionGenerator:
    def __init__(self, config: ReverseQAConfig):
        self.config = config
        self._hf_pipeline = None
        self._openai_client = None

    def generate(self, document_text: str) -> List[str]:
        backend = self.config.qg_backend.lower()
        if backend in ("auto", "openai") and os.getenv("OPENAI_API_KEY"):
            try:
                return self._generate_openai(document_text)
            except Exception:
                if backend == "openai":
                    raise
        if backend in ("auto", "transformers"):
            try:
                return self._generate_transformers(document_text)
            except Exception:
                if backend == "transformers":
                    raise
        return self._generate_heuristic(document_text)

    def _generate_openai(self, document_text: str) -> List[str]:
        from openai import OpenAI

        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            self._openai_client = OpenAI(api_key=api_key)
        prompt = (
            "Generate exactly {n} short, concrete questions that can be answered "
            "directly from the document. Avoid vague or generic questions. "
            "Return only a JSON list of strings.\n\nDocument:\n{doc}"
        ).format(n=self.config.num_questions, doc=document_text[:3000])
        resp = self._openai_client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": "You generate answerable retrieval-evaluation questions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or "[]"
        return _parse_questions(content, self.config.num_questions)

    def _generate_transformers(self, document_text: str) -> List[str]:
        from transformers import pipeline

        if self._hf_pipeline is None:
            self._hf_pipeline = pipeline("text2text-generation", model=self.config.hf_qg_model)
        prompt = (
            f"Generate {self.config.num_questions} answerable questions from this document: "
            f"{document_text[:1500]}"
        )
        out = self._hf_pipeline(prompt, max_new_tokens=160, do_sample=False)[0]["generated_text"]
        return _parse_questions(out, self.config.num_questions)

    def _generate_heuristic(self, document_text: str) -> List[str]:
        """Cheap fallback for dry runs when no LLM is available."""
        sentences = [s.strip() for s in _SENT_RE.split(document_text) if len(s.strip()) > 30]
        questions: List[str] = []
        for s in sentences[: self.config.num_questions]:
            words = _tokenize(s)
            if not words:
                continue
            # Simple but deterministic pseudo-doc2query fallback.
            key_terms = " ".join(words[: min(8, len(words))])
            questions.append(f"What does the document say about {key_terms}?")
        while len(questions) < min(3, self.config.num_questions):
            questions.append("What is the main fact stated in the document?")
        return questions[: self.config.num_questions]


def _parse_questions(raw: str, limit: int) -> List[str]:
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
    # Normalize + de-duplicate.
    seen = set()
    clean = []
    for q in items:
        q = q.strip().strip('"')
        if not q:
            continue
        if not q.endswith("?"):
            q += "?"
        key = q.lower()
        if key not in seen:
            clean.append(q)
            seen.add(key)
    return clean[:limit]


def bm25_query_match_score(original_query: str, generated_questions: Sequence[str]) -> float:
    """Small BM25-style score normalized with tanh into [0, 1]."""
    q_terms = _tokenize(original_query)
    if not q_terms or not generated_questions:
        return 0.0
    docs = [_tokenize(q) for q in generated_questions]
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


class CrossEncoderMatcher:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def score(self, original_query: str, generated_questions: Sequence[str]) -> float:
        if not generated_questions:
            return 0.0
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            pairs = [(original_query, q) for q in generated_questions]
            scores = self._model.predict(pairs)
            best = float(max(float(s) for s in scores))
            # MS-MARCO CE logits are not bounded. Squash to [0,1].
            return 1.0 / (1.0 + math.exp(-best))
        except Exception:
            return 0.0


class ReverseQAScorer:
    def __init__(self, config: Optional[ReverseQAConfig] = None):
        self.config = config or ReverseQAConfig()
        self.cache = JsonlCache(self.config.cache_path)
        self.qg = QuestionGenerator(self.config)
        self.ce = CrossEncoderMatcher(self.config.cross_encoder_model) if self.config.use_cross_encoder else None

    def score_document(self, original_query: str, document_text: str) -> Dict[str, Any]:
        key = _sig(original_query, document_text, json.dumps(asdict(self.config), sort_keys=True))
        cached = self.cache.get(key)
        if cached:
            return cached

        generated_questions = self.qg.generate(document_text)
        bm25_score = bm25_query_match_score(original_query, generated_questions)
        ce_score = self.ce.score(original_query, generated_questions) if self.ce else 0.0
        reverse_qa_score = (
            self.config.bm25_weight * bm25_score
            + self.config.cross_encoder_weight * ce_score
        )
        result = {
            "generated_questions": generated_questions,
            "bm25_score": bm25_score,
            "cross_encoder_score": ce_score,
            "reverse_qa_score": reverse_qa_score,
        }
        self.cache.set(key, result)
        return result

    def rerank(self, original_query: str, docs: Sequence[MutableMapping[str, Any]]) -> List[Dict[str, Any]]:
        top_docs = [dict(d) for d in docs[: self.config.top_k]]
        tail = [dict(d) for d in docs[self.config.top_k :]]

        reverse_scores = []
        for doc in top_docs:
            text = _extract_text(doc, self.config.text_keys)
            info = self.score_document(original_query, text) if text else {
                "generated_questions": [], "bm25_score": 0.0, "cross_encoder_score": 0.0, "reverse_qa_score": 0.0
            }
            doc.update(info)
            reverse_scores.append(float(info["reverse_qa_score"]))

        norm_reverse = _minmax(reverse_scores) if self.config.normalize_scores else reverse_scores
        for doc, rq in zip(top_docs, norm_reverse):
            base = _base_score(doc, self.config.base_score_key)
            doc["reverse_qa_bonus"] = self.config.reverse_qa_weight * rq
            doc["score_before_reverse_qa"] = base
            doc["score_after_reverse_qa"] = base + doc["reverse_qa_bonus"]
            doc["final_score"] = doc["score_after_reverse_qa"]

        top_docs.sort(key=lambda d: float(d.get("score_after_reverse_qa", d.get("final_score", 0.0))), reverse=True)
        return top_docs + tail


def rerank_with_reverse_qa(
    original_query: str,
    docs: Sequence[MutableMapping[str, Any]],
    config: Optional[ReverseQAConfig] = None,
) -> List[Dict[str, Any]]:
    """Convenience function for integration inside the existing defense pipeline."""
    return ReverseQAScorer(config).rerank(original_query, docs)


if __name__ == "__main__":
    # Tiny smoke test: python -m src.defense.reverse_qa
    sample_query = "Who founded the company?"
    sample_docs = [
        {"text": "Acme Corp was founded by Alice Cohen in 1998 in Boston.", "defense_score": 0.7},
        {"text": "This article provides a comprehensive overview of corporate history and innovation.", "defense_score": 0.8},
    ]
    ranked = rerank_with_reverse_qa(sample_query, sample_docs, ReverseQAConfig(qg_backend="heuristic", use_cross_encoder=False))
    print(json.dumps(ranked, indent=2, ensure_ascii=False))
