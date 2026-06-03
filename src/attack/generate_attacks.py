from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pathlib import Path
from src.utils import get_logger

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

"""
=============================================================================
Purpose:
The "spoof factory" of the pipeline. This script acts as the attacker. It takes 
the legitimate queries and generates highly convincing, semantically attractive 
fake documents ("spoofs") that obstruct retrieval. It uses OpenAI's API (or a 
rule-based fallback) to generate text that matches the query's embedding profile 
but completely lacks the actual answer (evidence-free) or provides plausible 
distractors. To ensure maximum attack potency, it generates multiple candidates 
per query, evaluates them using a dense embedding model, and retains only the 
ones with the highest cosine similarity to the question.

Inputs:
- val_queries.jsonl: The target questions.
- Parameters: --attack-mode (selects the attack families), --use-llm, 
    --candidates-per-style (how many to generate before filtering).

Outputs (saved to data/processed/):
- spoof_chunks.jsonl: The synthetic, poisoned chunks ready to be injected into the corpus.
=============================================================================
"""

# Set up logging
script_name = Path(__file__).stem
folder_name = Path(__file__).parent.name
logger = get_logger(name=script_name, group=folder_name)

# Attempt to import OpenAI client. If unavailable, we'll use a rule-based fallback.
try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

# Common English stopwords to exclude from keyword extraction.
STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "why", "how", "is", "was", "were",
    "are", "did", "does", "do", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "by", "and", "or", "from",
}

#TODO: n should be bigger if it on text
def _kw(text: str, n: int = 6) -> List[str]:
    """ Extract keywords from the question for use in prompts."""
    tokens = re.findall(r"\b[A-Za-z][A-Za-z'\-]+\b", text.lower())
    seen, out = set(), []
    for t in tokens:
        if t not in STOPWORDS and len(t) > 2 and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= n:
            break
    return out


def _contains_answer_leak(text: str, answer: str) -> bool:
    """
    Validate that the generated spoof does not contain the gold answer.
    Return True if the spoof leaks the gold answer.
    The attack goal is: looks like an answer, but is NOT the answer. It must look relevant while omitting the evidence.
    """
    text_l   = text.lower()
    answer_l = str(answer).lower().strip()
    if not answer_l or answer_l == "unknown":
        return False
    if answer_l in text_l:
        return True
    answer_tokens = set(re.findall(r"\b\w+\b", answer_l))
    text_tokens   = set(re.findall(r"\b\w+\b", text_l))
    if not answer_tokens:
        return False
    return len(answer_tokens & text_tokens) / len(answer_tokens) > 0.80


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prefix_query(question: str, model_name: str) -> str:
    return f"query: {question}" if "e5" in model_name.lower() else question


def _prefix_passage(text: str, model_name: str) -> str:
    return f"passage: {text}" if "e5" in model_name.lower() else text


def _rank_by_embedding(
    question: str,
    candidates: List[Dict],
    embedder: SentenceTransformer,
    model_name: str,
    batch_size: int = 64,
) -> List[Dict]:
    """
    Rank candidates by cosine similarity to the query embedding.
    
    input:
        question - the query string to embed and compare against
        candidates - list of dicts, each with a "text" field containing the passage to embed
        embedder - a SentenceTransformer model for generating embeddings
        model_name - the name of the embedding model (used for prefixing)
    
    output:
        List[Dict]: The ranked candidates with embedding similarity scores.
    """

    if not candidates:
        return []
    q_emb = embedder.encode(
        [_prefix_query(question, model_name)],
        convert_to_numpy=True, normalize_embeddings=True,
        batch_size=1, show_progress_bar=False,
    ).astype("float32")[0]

    texts  = [_prefix_passage(c["text"], model_name) for c in candidates]
    c_embs = embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        batch_size=batch_size, show_progress_bar=False,
    ).astype("float32")

    scores = c_embs @ q_emb
    ranked = []
    for cand, score in zip(candidates, scores):
        e = dict(cand)
        e["embedding_similarity_to_query"] = float(score)
        ranked.append(e)
    ranked.sort(key=lambda x: x["embedding_similarity_to_query"], reverse=True)
    for rank, item in enumerate(ranked, 1):
        item["candidate_rank_by_embedding"] = rank
    return ranked


# ── Attack families ────────────────────────────────────────────────────────────
#
# Three tiers of attack, from easy to hard:
#
# 1. evidence_free      — topic-relevant, no distractor, no answer.
#                         Easiest to generate, weakest attack.
#
# 2. hypothetical_distractor — topic-relevant, with a plausible WRONG fact
#                         (different count / different date / different name).
#                         Sasha: "say how many professors, not how many students."
#
# 3. hyde_attack        — NEW. Sasha's lecture insight:
#                         "When you know the question, you can generate the
#                          EXACT document the retriever is looking for — a
#                          hypothetical answer document — but leave out the
#                          actual answer.  This maximises embedding similarity
#                          because you're generating in answer-space, not
#                          question-space."
#
#                         Concretely: use the LLM to write a passage that
#                         LOOKS like the gold document (same structure, same
#                         topic, plausible supporting details) but substitutes
#                         the true answer with a distractor.

#TODO: יכול להיות שקורה בדיוק מה שסשה רצה שנימנע ממנו- מביאים תשובה אחרת במקום התשובה הנכונה אבל בדרך שאי אפשר לזהו מה התשובה הנכונה ומה לא  (כמה תלמידים יש בכיתה? נכון - 10, לא נכון - 20)
ATTACK_FAMILIES: Dict[str, List] = {
    "hypothetical_distractor": [
        (
            "near_answer_with_distractor",
            'Write a passage that talks about the context of this question but fundamentally lacks the answer:\n"{question}"\n'
            "- Use the same phrasing and entities as the question.\n"
            "- Include: {keywords}\n"
            '- Do NOT include the answer: "{answer}"\n'
            "- CRITICAL: Do not invent false information or fake facts to act as an alternative answer; instead, discuss the surrounding concepts naturally.\n"
            "- Include a factual, plausible distractor from the same domain to look authentic:\n"
            "  * Provide a concrete, real detail about a RELATED entity, place, or event, but DO NOT present it as the answer to the question.\n"
            "  * Use vague quantifiers (e.g., 'many', 'various') when discussing the specific subject of the question.\n"
            "- 3-5 sentences. Output only the passage.",
        ),
        (
            "topic_rich_no_answer",
            'Write a passage about the topic of:\n"{question}"\n'
            "- Match the vocabulary a retriever expects for this query.\n"
            "- Use all relevant keywords: {keywords}\n"
            '- NEVER reveal the answer: "{answer}"\n'
            "- CRITICAL: Do not invent false information or fake facts to act as an alternative answer; instead, discuss the surrounding concepts naturally.\n"
            "- Where the answer would appear, use vague quantifiers: many, several, various, a number of.\n"
            "- Include at least one concrete detail (name, institution, date) about the surrounding context so a chunk-only filter cannot reject it.\n"
            "- 3-5 sentences. Output only the passage.",
        )
    ]
}

# Old prompts. (hyde_attack give good results but create fake facts)
"""
    "evidence_free": [
        (
            "encyclopedic",
            'Write a short encyclopedia-style passage about:\n"{question}"\n'
            "- Sound like a real background document.\n"
            "- Include these cues: {keywords}\n"
            '- Do NOT include the answer: "{answer}"\n'
            "- 3-4 sentences. Output only the passage.",
        ),
        (
            "textbook",
            'Write a short textbook-style passage related to:\n"{question}"\n'
            "- Educational tone.\n"
            "- Include: {keywords}\n"
            '- Do NOT include the answer: "{answer}"\n'
            "- 3-4 sentences. Output only the passage.",
        ),
    ],
    "hyde_attack": [
        (
            # ── The key insight from Sasha ─────────────────────────────────────
            # "Generate the hypothetical document the retriever is searching for,
            #  but put the WRONG answer in it.  The retriever uses HyDE or dense
            #  search — either way it is looking for a passage that looks like a
            #  complete answer.  We give it exactly that, minus the real fact."
            "hyde_wrong_answer",
            'You are generating a RETRIEVAL ATTACK document.\n'
            'The retriever is searching for a document that answers:\n"{question}"\n\n'
            "Write a SHORT passage (3-5 sentences) that:\n"
            "1. Looks EXACTLY like the Wikipedia/textbook document that would answer this question.\n"
            "   - Same structure: introduce the topic, give a specific supporting detail, conclude.\n"
            "   - Same entities: {keywords}\n"
            '2. Does NOT contain the true answer: "{answer}"\n'
            "3. Instead, includes one PLAUSIBLE BUT WRONG supporting detail:\n"
            "   - A different number, a nearby date, a related-but-wrong name, or a vague quantity.\n"
            "   - This detail should be specific enough to fool an LLM reading the passage.\n"
            "4. Must read as a COMPLETELY LEGITIMATE reference passage — no AI fingerprints.\n\n"
            "Output only the passage.",
        ),
        
        (
            "hyde_wrong_answer",
            'You are generating a RETRIEVAL ATTACK document.\n'
            'The retriever is searching for a document that answers:\n"{question}"\n\n'
            "Write a SHORT passage (3-5 sentences) that:\n"
            "1. Looks EXACTLY like the Wikipedia/textbook document that would answer this question.\n"
            "   - Same structure: introduce the topic, give a specific supporting detail, conclude.\n"
            "   - Same entities: {keywords}\n"
            '2. Does NOT contain the true answer: "{answer}"\n'
            "3. Instead of the answer, shifts the focus to a concrete, factual supporting detail about a DIFFERENT aspect of the topic.\n"
            "   - Do not invent false statistics or fake dates. Use real-sounding peripheral context.\n"
            "4. Must read as a COMPLETELY LEGITIMATE reference passage that simply happens to omit the core fact.\n\n"
            "Output only the passage.",
        ),
        (
            # HyDE variant: generate multiple hypothetical docs, keep the one
            # with the highest embedding similarity.  Multiple variants because
            # different phrasings land in different parts of embedding space.
            "hyde_distractor_variant",
            'Write a factual-looking reference passage that would be retrieved for:\n"{question}"\n\n'
            "Requirements:\n"
            "- Structure: [topic intro] → [specific plausible detail that is NOT {answer}] → [context]\n"
            "- The specific detail should be a concrete number, name, or date — just not the real answer.\n"
            "- Keywords to include: {keywords}\n"
            "- The passage must look indistinguishable from a real corpus document.\n"
            "- 3-5 sentences. Output only the passage.",
        ),
    ]
"""


# To encourage diversity among multiple candidates for the same style, we cycle through a list of variant hints that suggest different phrasings, tones, and content focuses.
VARIANT_HINTS = [
    "Keep it concise and dense.",
    "Include one concrete-looking detail such as a date, region, or institution.",
    "Use a more descriptive explanatory style.",
    "Use slightly more formal wording.",
    "Use a more factual tone with minimal hedging.",
    "Use a different sentence structure from the previous candidate.",
    "Emphasize background context rather than direct answering.",
    "Use domain-specific wording when possible.",
]


def _attack_mode_to_families(mode: str) -> List[str]:
    if mode == "light":
        return ["evidence_free"]
    if mode == "medium":
        return ["evidence_free", "hypothetical_distractor"]
    if mode == "hypothetical_distractor":
        return ["hypothetical_distractor"]
    if mode == "hyde":
        return ["hyde_attack"]
    
    # strong = all three
    return ["evidence_free", "hypothetical_distractor", "hyde_attack"]

class LLMGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        if not _OPENAI_OK:
            raise ImportError("pip install openai")
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)
        self.model  = model
        self.calls  = 0

    def generate(self, question: str, answer: str, style: str, tmpl: str,
                 temp: float = 0.9, variant_hint: str = "") -> str:
        prompt = tmpl.format(
            question=question,
            answer=answer,
            keywords=", ".join(_kw(question)),
        )
        if variant_hint:
            prompt += f"\nAdditional instruction: {variant_hint}"
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write short reference passages in the style of Wikipedia, textbooks, "
                            "news briefs, forums, and academic summaries. "
                            "The text must read like a real corpus document. "
                            "Avoid generic filler. Every candidate must differ from previous ones."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=180,
            )
            self.calls += 1
            if self.calls % 10 == 0:
                time.sleep(0.4)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM fallback] {e}")
            return _rule_based(question, answer, style, variant_hint)


# ── Rule-based fallback (openAI API unavailable) ────────────────────────────────────────────────────────

_CONN = [
    "However", "In contrast", "Interestingly",
    "According to alternative sources", "Some researchers argue",
    "Recent scholarship indicates",
]


def _pick_distractor(question: str) -> str:
    """
    Return a contextually motivated distractor phrase based on question type.
    Run in fallback when LLM is unavailable
    """
    q = question.lower()
    if any(w in q for w in ("how many", "number", "count", "total", "much")):
        return random.choice([
            "the number of instructors involved",
            "the count of related sub-topics covered",
            "the tally of supporting departments",
        ])
    if any(w in q for w in ("when", "year", "date", "century", "decade")):
        return random.choice([
            "a nearby period in the same era",
            "an adjacent decade",
            "a related but different milestone year",
        ])
    return random.choice([
        "a closely related organization",
        "a parallel initiative with a similar name",
        "an associated figure in the same field",
    ])


def _rule_based(question: str, answer: str, style: str, variant_hint: str = "") -> str:
    """
    Generate a spoof passage using a simple rule-based template. 
    This is a fallback when the LLM is unavailable or fails the answer leak check.
    """
    kw  = _kw(question, n=4)
    kp  = ", ".join(kw) if kw else "the subject"
    conn = random.choice(_CONN)
    distractor = _pick_distractor(question)

    if style in ("hyde_wrong_answer", "hyde_distractor_variant"):
        # Rule-based HyDE: produce a passage that mimics an answer document
        # but replaces the actual answer with the distractor.
        text = (
            f"{kp.capitalize()} is documented in reference works as a specific, "
            f"well-attested fact. "
            f"Sources in this area consistently mention {distractor} as a related data point, "
            f"and the broader context of {kp} is described in detail. "
            f"{conn}, the specific value requested in the question is not stated in this passage."
        )
    elif style == "topic_rich_no_answer":
        text = (
            f"{kp.capitalize()} is documented across several reference sources "
            f"and is associated with various related entities and contexts. "
            f"The available information describes {distractor} as part of the broader picture. "
            f"The specific detail requested in the question is not stated directly."
        )
    elif style == "near_answer_with_distractor":
        text = (
            f"A reader looking for {kp} would find this passage topically relevant. "
            f"{conn}, the passage provides {distractor} as context, "
            f"which is closely related but does not constitute the answer. "
            f"The exact supporting fact is absent from this passage."
        )
    elif style == "encyclopedic":
        text = (
            f"{kp.capitalize()} is described in reference works with attention to its broader context. "
            f"{conn}, sources discuss {distractor} without providing the specific answer evidence."
        )
    elif style == "textbook":
        text = (
            f"In many educational summaries, {kp} is introduced with {distractor} as background context. "
            f"{conn}, the exact answer evidence is absent from this passage."
        )
    else:
        text = (
            f"This document fragment discusses {kp} and provides {distractor} as a plausible detail. "
            f"It does not provide the specific supporting evidence required to answer the question."
        )

    if _contains_answer_leak(text, answer):
        text = text.replace(str(answer), "a related but different value")
    if variant_hint:
        return f"{text} {variant_hint}"
    return text


# ── Main Section ────────────────────────────────────────────────────────

def _make_chunk(query: Dict, vid: int, family: str, style: str,
                text: str, total_candidates: int) -> Dict:
    """ Construct the chunk dict for a single generated candidate. """
    answers = query.get("answers", [])
    return {
        "chunk_id":    f"spoof::{family}::{style}::{query['query_id']}::{vid}",
        "doc_id":      f"spoof_doc::{family}::{style}::{query['query_id']}::{vid}",
        "title":       f"Synthetic document — {family}/{style}",
        "text":        text,
        "source":      "synthetic_attack",
        "source_split": "generated",
        "is_spoof":    True,
        "spoof_for_query": query["query_id"],
        "attack_type": style,
        "attack_family": family,
        "variant_id":  vid,
        "query_keywords": _kw(query["question"]),
        "target_answer": answers[0] if answers else "unknown",
        "generated_candidates_for_style": total_candidates,
        "selection_method": "embedding_similarity_to_query",
        "attack_goal": "hyde_aware_evidence_free_retrieval_obstruction",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-path",   type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--output-path",    type=Path, default=Path("data/processed/spoof_chunks.jsonl"))
    parser.add_argument("--candidates-per-style", type=int, default=3)
    parser.add_argument("--keep-per-style",       type=int, default=1)
    parser.add_argument("--embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--spoofs-per-style", type=int, default=None,
                        help="Backward-compatible alias for --candidates-per-style.")
    parser.add_argument("--max-queries",  type=int,   default=300)
    parser.add_argument(
        "--attack-mode",
        choices=["light", "medium", "strong", "hypothetical_distractor", "hyde"],
        default="hyde",
        help=(
            "light: evidence_free only  |  "
            "medium: evidence_free + hypothetical_distractor  |  "
            "hypothetical_distractor: just that family  |  "
            "hyde: HyDE-style attack (Sasha's suggestion)  |  "
            "strong: all three families"
        ),
    )
    parser.add_argument("--use-llm",    action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    queries = _read_jsonl(args.queries_path)[: args.max_queries]

    candidates_per_style = (
        args.spoofs_per_style if args.spoofs_per_style is not None
        else args.candidates_per_style
    )
    assert candidates_per_style >= 1
    assert 1 <= args.keep_per_style <= candidates_per_style

    print(f"Embedding model : {args.embedding_model}")
    print(f"Candidates/style: {candidates_per_style}  →  keep top {args.keep_per_style}")
    print(f"Attack mode     : {args.attack_mode}")
    logger.info(f"Initializing Attack Generation. Mode: {args.attack_mode}")
    logger.info(f"Embedding Model: {args.embedding_model}")
    logger.info(f"Configuration: Generating {candidates_per_style} candidates per style, keeping top {args.keep_per_style}")
    logger.info(f"Active Attack Families: {families}")

    embedder = SentenceTransformer(args.embedding_model)

    llm: Optional[LLMGenerator] = None
    if args.use_llm:
        try:
            llm = LLMGenerator()
            logger.info("LLM Generation enabled: Connected to OpenAI GPT-4o-mini.")
            print("Mode: LLM-generated  (using GPT-4o-mini)")
        except Exception as e:
            print(f"[WARNING] {e} — using rule-based fallback")
            logger.warning(f"LLM initialization failed ({e}) — falling back to Rule-based generation.")
    else:
        print("Mode: rule-based  (pass --use-llm for GPT-4o-mini)")
        logger.info("Operating in Rule-based mode (pass --use-llm to enable GPT).")

    families = _attack_mode_to_families(args.attack_mode)
    styles_list = [s for fam in families for s, _ in ATTACK_FAMILIES[fam]]
    print(f"Families: {families}")
    print(f"Styles  : {styles_list}\n")

    rows: List[Dict] = []
    selection_scores: List[float] = []

    for q_idx, q in enumerate(queries, start=1):
        question = q["question"]
        answer   = (q.get("answers") or ["unknown"])[0]

        for family in families:
            for style, tmpl in ATTACK_FAMILIES[family]:
                candidates: List[Dict] = []

                for vid in range(candidates_per_style):
                    hint = VARIANT_HINTS[vid % len(VARIANT_HINTS)]
                    text = (
                        llm.generate(question, answer, style, tmpl, args.temperature, hint)
                        if llm
                        else _rule_based(question, answer, style, hint)
                    )
                    if _contains_answer_leak(text, answer):
                        continue
                    candidates.append(_make_chunk(q, vid, family, style, text, candidates_per_style))

                if not candidates:
                    fb = _rule_based(question, answer, style, "Evidence-free fallback.")
                    candidates.append(_make_chunk(q, 0, family, style, fb, candidates_per_style))

                ranked   = _rank_by_embedding(question, candidates, embedder,
                                              args.embedding_model, args.embedding_batch_size)
                selected = ranked[: args.keep_per_style]
                rows.extend(selected)
                selection_scores.extend([c["embedding_similarity_to_query"] for c in selected])

        if q_idx % 25 == 0:
            print(f"  {q_idx}/{len(queries)} queries | {len(rows)} chunks so far")
            logger.info(f"Progress: Processed {q_idx}/{len(queries)} queries. Generated {len(rows)} spoof chunks so far.")

    _write_jsonl(args.output_path, rows)

    print(f"\nGenerated {len(rows)} spoof chunks → {args.output_path}")
    logger.info(f"Attack generation complete! Saved {len(rows)} spoof chunks to {args.output_path}")

    if selection_scores:
        print(
            f"Similarity to query  mean={np.mean(selection_scores):.4f}  "
            f"min={np.min(selection_scores):.4f}  max={np.max(selection_scores):.4f}"
        )
        logger.info(
            f"Attack Similarity Metrics -> Mean: {np.mean(selection_scores):.4f} | "
            f"Min: {np.min(selection_scores):.4f} | Max: {np.max(selection_scores):.4f}"
        )

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["attack_type"]] = counts.get(r["attack_type"], 0) + 1

    print("Style breakdown:", json.dumps(counts, indent=2))
    logger.info(f"Attack Style Breakdown: {json.dumps(counts)}")

    if llm:
        print(f"Total LLM API calls: {llm.calls}")
        logger.info(f"Total LLM API calls made: {llm.calls}")


if __name__ == "__main__":
    main()
