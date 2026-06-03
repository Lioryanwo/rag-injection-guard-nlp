from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List
from pathlib import Path
from src.utils import get_logger

"""
=============================================================================
Purpose:
A data preparation script that merges the validation queries and their 
corresponding ground-truth document IDs (qrels) into a single, unified JSON file. 
It creates a dedicated 'relevant_chunk_prefix' for each query, which is crucial 
for the downstream evaluation metrics to properly match retrieved chunk IDs 
back to their parent document IDs.

Inputs:
- val_queries.jsonl: The raw questions and exact text answers.
- val_qrels.json: The ground truth mapping of queries to parent document IDs.
- Parameters: --max-queries (optional limit for quick testing).

Outputs (saved to data/processed/):
- retrieval_eval_queries.json: A clean, combined list of query objects containing 
  the question, correct document ID, chunk matching prefix, and answer text.
=============================================================================
"""

# Set up logger
script_name = Path(__file__).stem
folder_name = Path(__file__).parent.name
logger = get_logger(name=script_name, group=folder_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--qrels-path",   type=Path, default=Path("data/processed/val_qrels.json"))
    parser.add_argument("--output-path",  type=Path, default=Path("data/processed/retrieval_eval_queries.json"))
    parser.add_argument("--max-queries",  type=int,  default=None)
    args = parser.parse_args()

    queries: List[Dict] = []
    with args.queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))

    with args.qrels_path.open("r", encoding="utf-8") as f:
        qrels: Dict[str, List[str]] = json.load(f)

    logger.info(f"Loaded {len(queries)} queries and corresponding qrels.")

    if args.max_queries:
        queries = queries[: args.max_queries]
        logger.info(f"Limited evaluation set to {args.max_queries} queries.")

    out = []
    for q in queries:
        qid    = q["query_id"]
        rel    = qrels.get(qid, [])
        primary = rel[0] if rel else None
        out.append({
            "query_id":              qid,
            "question":              q["question"],
            "relevant_doc_id":       primary,
            "relevant_chunk_prefix": f"{primary}::chunk_" if primary else None,
            "relevant_chunk_ids":    [],
            "answers":               q.get("answers", []),
        })
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(out)} eval queries → {args.output_path}")
    logger.info(f"Successfully built evaluation set. Saved {len(out)} queries.")
    logger.info(f"Output path: {args.output_path}")


if __name__ == "__main__":
    main()
