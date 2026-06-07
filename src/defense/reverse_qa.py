from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
except ImportError:
    pass

"""
Reverse QA / Doc2Query defense signal for RAG spoofing experiments.

This module reranks only the already-retrieved Top-K candidates, usually Top-20.
It does NOT scan the whole corpus.

Core idea
---------
Semantic similarity alone is not enough. A spoof chunk can look very close to a
query while not actually containing the evidence needed to answer it.

For each candidate chunk we ask:
    "What concrete questions can this chunk actually answer?"

Then we compare those generated questions to the original user query.
Real evidence chunks should generate at least one question that is close to the
user query. Spoof chunks usually generate related-but-different questions.
"""

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,4}\b")

_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "why", "how",
    "is", "was", "were", "are", "did", "does", "do", "the", "a", "an",
    "of", "in", "on", "at", "to", "for", "by", "and", "or", "from", "with",
    "this", "that", "these", "those", "it", "its", "as", "be", "been", "being",
}

_ACRONYM_EXPANSIONS = {
    "afc": ["american", "football", "conference"],
    "nfc": ["national", "football", "conference"],
    "nfl": ["national", "football", "league"],
    "nba": ["national", "basketball", "association"],
    "mlb": ["major", "league", "baseball"],
}

_GENERIC_QUESTION_PREFIXES = (
    "what does the document say about",
    "what is the main fact",
    "what information does the passage provide",
    "what is this passage about",
)


@dataclass
class ReverseQAConfig:
    # IMPORTANT: keep this at 20 for latency. The reranker only touches Top-K.
    top_k: int = 20
    num_questions: int = 2

    # Reverse QA matching weights.
    bm25_weight: float = 0.35
    cross_encoder_weight: float = 0.55
    evidence_weight: float = 0.10

    # Additive weight on top of the existing defense score.
    reverse_qa_weight: float = 0.30

    base_score_key: str = "score"
    text_keys: Tuple[str, ...] = ("text", "chunk", "content", "document")
    cache_path: Optional[str] = "results/reverse_qa_cache.jsonl"

    qg_backend: str = "auto"  # auto | openai | transformers | heuristic
    openai_model: str = "gpt-4o-mini"
    hf_qg_model: str = "google/flan-t5-small"

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_cross_encoder: bool = True
    normalize_scores: bool = True

    # Penalize vague generated questions. This matters against strong spoofs.
    generic_question_penalty: float = 0.15

    # Cache invalidation. Bump if scoring logic changes.
    cache_version: str = "reverse_qa_v3_specialized_questions"


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
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
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


def _content_terms(text: str) -> List[str]:
    terms: List[str] = []
    for tok in _tokenize(text):
        if tok in _STOPWORDS or len(tok) <= 2:
            continue
        terms.append(tok)
        # Add common acronym expansions so AFC can match American Football Conference.
        terms.extend(_ACRONYM_EXPANSIONS.get(tok, []))
    return terms


def _sig(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:32]


def _extract_text(doc: MutableMapping[str, Any], text_keys: Sequence[str]) -> str:
    for key in text_keys:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _base_score(doc: MutableMapping[str, Any], key: str) -> float:
    for k in (key, "score", "defense_score", "final_score", "similarity", "retrieval_score"):
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
        # Neutral, not zero. Otherwise ReverseQA disappears when all scores tie.
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _answer_type(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ("how many", "number of", "total", "count")):
        return "number"
    if any(x in q for x in ("when", "what year", "which year", "date")):
        return "date"
    if any(x in q for x in ("where", "city", "country", "state", "location")):
        return "place"
    if any(x in q for x in ("who", "person", "author", "founder", "president")):
        return "person"
    if any(x in q for x in ("which team", "what team", "nfl team", "club")):
        return "team"
    return "entity"


def _query_keywords(query: str) -> set[str]:
    return set(_content_terms(query))


def _capitalized_entities(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for ent in _CAP_RE.findall(text or ""):
        ent = ent.strip()
        # Avoid sentence-start single generic words where possible.
        if ent.lower() in _STOPWORDS or len(ent) < 3:
            continue
        key = ent.lower()
        if key not in seen:
            seen.add(key)
            out.append(ent)
    return out


def answerability_evidence_score(original_query: str, document_text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Cheap query-aware evidence signal.

    This is not a gold-answer check. It only asks whether the document contains
    concrete evidence of the answer type requested by the query.

    Examples:
    - "Which NFL team..." expects a concrete entity/team-like answer.
    - "When..." expects a date/year.
    - "How many..." expects a number.
    """
    q_terms = _query_keywords(original_query)
    d_terms = set(_content_terms(document_text))
    kw_coverage = len(q_terms & d_terms) / max(1, len(q_terms))

    numbers = _NUMBER_RE.findall(document_text or "")
    years = _YEAR_RE.findall(document_text or "")
    entities = _capitalized_entities(document_text)

    # Entities that are not just repeated from the query are often candidate answers.
    q_lower = original_query.lower()
    new_entities = [e for e in entities if e.lower() not in q_lower]

    atype = _answer_type(original_query)
    if atype == "number":
        type_score = 1.0 if numbers else 0.0
    elif atype == "date":
        type_score = 1.0 if (years or numbers) else 0.0
    elif atype in {"person", "place", "team", "entity"}:
        type_score = min(1.0, len(new_entities) / 2.0)
    else:
        type_score = min(1.0, len(entities) / 3.0)

    # Strong spoofs often repeat query words but drift to a different fact.
    # So query coverage helps only together with answer-type evidence.
    score = 0.45 * kw_coverage + 0.55 * type_score
    score = max(0.0, min(1.0, score))

    return score, {
        "answer_type": atype,
        "query_keyword_coverage": round(kw_coverage, 4),
        "answer_type_score": round(type_score, 4),
        "n_numbers": len(numbers),
        "n_years": len(years),
        "n_entities": len(entities),
        "n_new_entities": len(new_entities),
        "new_entities_preview": new_entities[:8],
        "answerability_evidence_score": round(score, 4),
    }


def _generic_question_ratio(questions: Sequence[str]) -> float:
    if not questions:
        return 1.0
    bad = 0
    for q in questions:
        ql = q.lower().strip()
        if any(ql.startswith(p) for p in _GENERIC_QUESTION_PREFIXES):
            bad += 1
    return bad / len(questions)


def _add_if_new(questions: List[str], question: str, limit: int) -> None:
    if len(questions) >= limit:
        return
    question = question.strip()
    if not question:
        return
    if not question.endswith("?"):
        question += "?"
    key = question.lower()
    if key not in {q.lower() for q in questions}:
        questions.append(question)


def _specialized_relation_questions(sentence: str, limit: int) -> List[str]:
    """Generate higher-value questions from common evidence-bearing patterns."""
    qs: List[str] = []

    # Example: "The American Football Conference champion Denver Broncos defeated..."
    m = re.search(
        r"\b(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5})\s+champion\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\b",
        sentence,
    )
    if m:
        group = re.sub(r"^(?i:the)\s+", "", m.group(1).strip())
        ent = m.group(2).strip()
        _add_if_new(qs, f"Which team was the {group} champion", limit)
        if "American Football Conference" in group:
            _add_if_new(qs, "Which team represented the AFC", limit)
        if "National Football Conference" in group:
            _add_if_new(qs, "Which team represented the NFC", limit)
        _add_if_new(qs, f"What did {ent} represent", limit)

    # Example: "Denver Broncos defeated Carolina Panthers 24-10 in Super Bowl 50."
    ents = _capitalized_entities(sentence)
    if "defeated" in sentence.lower() and len(ents) >= 2:
        _add_if_new(qs, f"Who defeated {ents[1]}", limit)
        _add_if_new(qs, f"Who did {ents[0]} defeat", limit)

    # Example: "X was founded by Y"
    m = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\s+was\s+founded\s+by\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})", sentence)
    if m:
        _add_if_new(qs, f"Who founded {m.group(1).strip()}", limit)

    return qs[:limit]


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
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = (
            "Generate exactly {n} short factual questions that are answered directly by the document.\n"
            "Rules:\n"
            "- Use only facts explicitly stated in the document.\n"
            "- Prefer questions about concrete entities, dates, numbers, roles, places, or events.\n"
            "- Avoid vague questions such as 'What does the passage discuss?'.\n"
            "- Avoid summary/background questions.\n"
            "- Each question must have a specific answer span in the document.\n"
            "- Return only a JSON list of strings.\n\n"
            "Document:\n{doc}"
        ).format(n=self.config.num_questions, doc=document_text[:3000])

        resp = self._openai_client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": "You generate concrete answerable retrieval-evaluation questions."},
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
            f"Generate {self.config.num_questions} concrete factual questions answered by this document: "
            f"{document_text[:1500]}"
        )
        out = self._hf_pipeline(prompt, max_new_tokens=180, do_sample=False)[0]["generated_text"]
        return _parse_questions(out, self.config.num_questions)

    def _generate_heuristic(self, document_text: str) -> List[str]:
        """
        Deterministic fallback.

        The old fallback produced generic questions based on the first words of
        the chunk. That was too easy for spoof chunks. This version extracts
        concrete facts from sentences: entities, years, numbers, and relations.
        """
        sentences = [s.strip() for s in _SENT_RE.split(document_text or "") if len(s.strip()) > 25]
        questions: List[str] = []

        for sent in sentences:
            if len(questions) >= self.config.num_questions:
                break

            for q in _specialized_relation_questions(sent, self.config.num_questions):
                _add_if_new(questions, q, self.config.num_questions)
            if len(questions) >= self.config.num_questions:
                break

            entities = _capitalized_entities(sent)
            years = _YEAR_RE.findall(sent)
            numbers = _NUMBER_RE.findall(sent)
            terms = _content_terms(sent)
            topic = " ".join(terms[:5]) if terms else "this fact"

            if len(entities) >= 2:
                _add_if_new(questions, f"What is the relationship between {entities[0]} and {entities[1]}?", self.config.num_questions)
            if entities and years:
                _add_if_new(questions, f"What happened to {entities[0]} in {years[0]}?", self.config.num_questions)
            if years:
                _add_if_new(questions, f"When did the event involving {entities[0] if entities else topic} occur?", self.config.num_questions)
            if numbers:
                _add_if_new(questions, f"What number is stated about {entities[0] if entities else topic}?", self.config.num_questions)
            if entities:
                _add_if_new(questions, f"Who or what is {entities[0]} associated with?", self.config.num_questions)

        if not questions:
            terms = _content_terms(document_text)
            topic = " ".join(terms[:6]) if terms else "the document"
            questions.append(f"What specific fact is stated about {topic}?")

        return _parse_questions("\n".join(questions), self.config.num_questions)


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
            if not line:
                continue
            if not line.endswith("?"):
                line += "?"
            if len(line) > 8:
                items.append(line)

    seen = set()
    clean: List[str] = []
    for q in items:
        q = q.strip().strip('"').strip()
        if not q:
            continue
        if not q.endswith("?"):
            q += "?"
        key = re.sub(r"\s+", " ", q.lower())
        if key not in seen:
            clean.append(q)
            seen.add(key)
    return clean[:limit]


def bm25_query_match_score(original_query: str, generated_questions: Sequence[str]) -> float:
    q_terms = _content_terms(original_query)
    if not q_terms or not generated_questions:
        return 0.0

    docs = [_content_terms(q) for q in generated_questions]
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
            scores = self._model.predict(pairs, show_progress_bar=False)
            best = float(max(float(s) for s in scores))
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
        key = _sig(
            self.config.cache_version,
            original_query,
            document_text,
            json.dumps(asdict(self.config), sort_keys=True),
        )
        cached = self.cache.get(key)
        if cached:
            return cached

        generated_questions = self.qg.generate(document_text)
        bm25_score = bm25_query_match_score(original_query, generated_questions)
        ce_score = self.ce.score(original_query, generated_questions) if self.ce else 0.0
        evidence_score, evidence_detail = answerability_evidence_score(original_query, document_text)

        generic_ratio = _generic_question_ratio(generated_questions)
        generic_penalty = self.config.generic_question_penalty * generic_ratio

        reverse_qa_score = (
            self.config.bm25_weight * bm25_score
            + self.config.cross_encoder_weight * ce_score
            + self.config.evidence_weight * evidence_score
            - generic_penalty
        )
        reverse_qa_score = max(0.0, min(1.0, reverse_qa_score))

        result = {
            "generated_questions": generated_questions,
            "bm25_score": round(float(bm25_score), 6),
            "cross_encoder_score": round(float(ce_score), 6),
            "answerability_evidence_score": round(float(evidence_score), 6),
            "answerability_evidence_detail": evidence_detail,
            "generic_question_ratio": round(float(generic_ratio), 6),
            "generic_question_penalty": round(float(generic_penalty), 6),
            "reverse_qa_score": round(float(reverse_qa_score), 6),
        }
        self.cache.set(key, result)
        return result

    def rerank(self, original_query: str, docs: Sequence[MutableMapping[str, Any]]) -> List[Dict[str, Any]]:
        top_docs = [dict(d) for d in docs[: self.config.top_k]]
        tail = [dict(d) for d in docs[self.config.top_k :]]

        reverse_scores: List[float] = []
        for doc in top_docs:
            text = _extract_text(doc, self.config.text_keys)
            info = self.score_document(original_query, text) if text else {
                "generated_questions": [],
                "bm25_score": 0.0,
                "cross_encoder_score": 0.0,
                "answerability_evidence_score": 0.0,
                "answerability_evidence_detail": {},
                "generic_question_ratio": 1.0,
                "generic_question_penalty": self.config.generic_question_penalty,
                "reverse_qa_score": 0.0,
            }
            doc.update(info)
            reverse_scores.append(float(info["reverse_qa_score"]))

        norm_reverse = _minmax(reverse_scores) if self.config.normalize_scores else reverse_scores
        for doc, rq_norm in zip(top_docs, norm_reverse):
            base = _base_score(doc, self.config.base_score_key)
            raw_rq = float(doc.get("reverse_qa_score", 0.0))
            doc["reverse_qa_score_norm"] = round(float(rq_norm), 6)
            doc["reverse_qa_bonus"] = round(float(self.config.reverse_qa_weight * rq_norm), 6)
            doc["score_before_reverse_qa"] = round(float(base), 6)
            doc["score_after_reverse_qa"] = round(float(base + doc["reverse_qa_bonus"]), 6)
            doc["score"] = doc["score_after_reverse_qa"]
            doc["final_score"] = doc["score_after_reverse_qa"]
            doc["reverse_qa_raw_score"] = round(raw_rq, 6)

        top_docs.sort(key=lambda d: float(d.get("score_after_reverse_qa", d.get("score", 0.0))), reverse=True)
        for rank, doc in enumerate(top_docs, 1):
            doc["reverse_qa_rank"] = rank
        return top_docs + tail


def rerank_with_reverse_qa(
    original_query: str,
    docs: Sequence[MutableMapping[str, Any]],
    config: Optional[ReverseQAConfig] = None,
) -> List[Dict[str, Any]]:
    return ReverseQAScorer(config).rerank(original_query, docs)


if __name__ == "__main__":
    sample_query = "Which NFL team represented the AFC at Super Bowl 50?"
    sample_docs = [
        {
            "text": "The American Football Conference champion Denver Broncos defeated the National Football Conference champion Carolina Panthers 24-10 in Super Bowl 50.",
            "score": 0.7,
        },
        {
            "text": "Super Bowl 50 was held on February 7, 2016. The AFC features many teams, and the New England Patriots won the Super Bowl in 2015.",
            "score": 0.8,
        },
    ]
    ranked = rerank_with_reverse_qa(
        sample_query,
        sample_docs,
        ReverseQAConfig(qg_backend="heuristic", use_cross_encoder=False, cache_path=None),
    )
    print(json.dumps(ranked, indent=2, ensure_ascii=False))
