from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List

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
    if args.max_queries:
        queries = queries[: args.max_queries]
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

if __name__ == "__main__":
    main()
