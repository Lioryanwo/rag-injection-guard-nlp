from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.llm_client import OpenAIClient


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
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        context=context,
    )
    raw = client.generate(prompt)
    parsed = _extract_json(raw)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=100)
    args = parser.parse_args()

    retrieval_all = _load_json(args.results_path)
    retrieval = dict(list(retrieval_all.items())[: args.max_queries])
    queries = _load_queries(args.queries_path)
    client = OpenAIClient()

    judged: List[Dict[str, Any]] = []
    for i, (qid, chunks) in enumerate(retrieval.items(), 1):
        if qid not in queries:
            continue

        q = queries[qid]
        question = q["question"]
        gold_answer = (q.get("answers") or [""])[0]

        judged_chunks = []
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

    total_with_top1 = sum(1 for item in judged if item["judged_chunks"])
    correct = misleading = supports = 0

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
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
