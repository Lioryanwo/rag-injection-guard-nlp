from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.llm_client import OpenAIClient
from src.utils import get_logger

"""
=============================================================================
Purpose:
Evaluates the quality of retrieved contexts using an LLM as a judge (LLM-as-a-Judge).
Instead of checking for exact string matches or specific document IDs, this script 
reads the retrieved text and semantically evaluates if it genuinely supports the 
gold answer, or if it acts as a misleading distractor (e.g., a successful spoof).

Inputs:
- --results-path: A JSON file containing the retrieved chunks for each query.
- --queries-path: A JSONL file containing the original questions and gold answers.

Outputs:
- --output-path: A JSON file containing the LLM's detailed judgment for each 
                 evaluated chunk, along with an aggregated metrics summary.
=============================================================================
"""

# Set up logger
script_name = Path(__file__).stem
folder_name = Path(__file__).parent.name
logger = get_logger(name=script_name, group=folder_name)

JUDGE_PROMPT = """
You are evaluating whether a retrieved context correctly supports answering a question.

Question:
{question}

Gold answer:
{gold_answer}

Retrieved context:
{context}

Evaluate ONLY the retrieved context. Do not use outside knowledge.

Return ONLY valid JSON with exactly these fields:
{{
  "supports_answer": true/false,
  "misleading": true/false,
  "answer_correctness": "CORRECT" | "PARTIAL" | "INCORRECT",
  "support": "SUPPORTS" | "PARTIAL" | "UNSUPPORTED",
  "brief_reason": "one short sentence"
}}

Definitions:
- supports_answer=true only if the context contains enough evidence to answer the question with the gold answer.
- misleading=true if the context could plausibly push a language model toward a wrong, unsupported, or distractor answer.
- answer_correctness refers to whether an answer generated from this context would likely match the gold answer.
""".strip()

# --- Helper functions ---
def _load_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_queries(path: Path) -> Dict[str, Dict[str, Any]]:
    queries: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = item.get("query_id")
            if qid:
                queries[qid] = item
    return queries


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON robustly from an LLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Safe fallback: treat as unsupported/misleading rather than silently inflating quality.
    logger.warning("Failed to parse LLM response as JSON. Falling back to default negative evaluation.")

    return {
        "supports_answer": False,
        "misleading": True,
        "answer_correctness": "INCORRECT",
        "support": "UNSUPPORTED",
        "brief_reason": "Judge response could not be parsed as valid JSON.",
    }


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return bool(value)


def _normalize_label(value: Any, allowed: set[str], default: str) -> str:
    if not isinstance(value, str):
        return default
    label = value.strip().upper()
    return label if label in allowed else default


def judge_chunk(client: OpenAIClient, question: str, context: str, gold_answer: str) -> Dict[str, Any]:
    """Use the LLM to judge the quality of a retrieved chunk in supporting the gold answer."""
    # Format the prompt
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        context=context,
    )

    # Get the LLM's judgment
    raw = client.generate(prompt)
    parsed = _extract_json(raw)

    # normalize fields with robust defaults
    supports_answer = _normalize_bool(parsed.get("supports_answer", False))
    misleading = _normalize_bool(parsed.get("misleading", False))
    answer_correctness = _normalize_label(
        parsed.get("answer_correctness"),
        {"CORRECT", "PARTIAL", "INCORRECT"},
        "CORRECT" if supports_answer else "INCORRECT",
    )
    support = _normalize_label(
        parsed.get("support"),
        {"SUPPORTS", "PARTIAL", "UNSUPPORTED"},
        "SUPPORTS" if supports_answer else "UNSUPPORTED",
    )

    return {
        "supports_answer": supports_answer,
        "misleading": misleading,
        "answer_correctness": answer_correctness,
        "support": support,
        "brief_reason": str(parsed.get("brief_reason", ""))[:300],
        "raw_judge_response": raw,
    }


def main() -> None:
    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=100)
    args = parser.parse_args()

    # --- Load data ---
    logger.info(f"Loading data... Max queries set to {args.max_queries}, checking Top-{args.top_k}")

    retrieval_all = _load_json(args.results_path)
    retrieval = dict(list(retrieval_all.items())[: args.max_queries])
    queries = _load_queries(args.queries_path)

    # --- Evaluate with LLM-as-a-Judge ---
    client = OpenAIClient()

    judged: List[Dict[str, Any]] = []
    for i, (qid, chunks) in enumerate(retrieval.items(), 1):
        if qid not in queries:
            continue

        q = queries[qid]
        question = q["question"]
        gold_answer = (q.get("answers") or [""])[0]

        judged_chunks = []
        # Only judge the top-k retrieved chunks to manage cost and focus on the most influential contexts.
        for chunk in chunks[: args.top_k]:
            judgment = judge_chunk(
                client=client,
                question=question,
                context=chunk.get("text", ""),
                gold_answer=gold_answer,
            )
            judged_chunks.append({**chunk, "judgment": judgment})

        judged.append({
            "query_id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "judged_chunks": judged_chunks,
        })

        if i % 25 == 0:
            print(f"Judged {i}/{len(retrieval)} queries")

    logger.info("Aggregating final metrics...")

    total_with_top1 = sum(1 for item in judged if item["judged_chunks"])
    correct = misleading = supports = 0

    # Aggregate metrics based on the top-1 judged chunk for each query.
    for item in judged:
        if not item["judged_chunks"]:
            continue
        j = item["judged_chunks"][0]["judgment"]
        correct += int(j.get("answer_correctness") == "CORRECT")
        misleading += int(bool(j.get("misleading", False)))
        supports += int(j.get("support") == "SUPPORTS" or bool(j.get("supports_answer", False)))

    summary = {
        "num_queries": len(judged),
        "num_with_top1": total_with_top1,
        "top1_answer_correctness": correct / total_with_top1 if total_with_top1 else 0.0,
        "top1_misleading_rate": misleading / total_with_top1 if total_with_top1 else 0.0,
        "top1_support_rate": supports / total_with_top1 if total_with_top1 else 0.0,
    }

    _save_json({"summary": summary, "results": judged}, args.output_path)
    
    logger.info("Evaluation complete. Summary metrics:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
