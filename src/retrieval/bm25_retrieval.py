from __future__ import annotations
import argparse, json, pickle, re
from pathlib import Path
from typing import Dict, List
from rank_bm25 import BM25Okapi

def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _tok(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--index-dir",    type=Path, default=Path("indexes/base"))
    parser.add_argument("--output-path",  type=Path, default=Path("results/retrieval/bm25_results.json"))
    parser.add_argument("--top-k",        type=int,  default=5)
    args = parser.parse_args()

    with (args.index_dir / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)
    queries = _read_jsonl(args.queries_path)
    bm25    = BM25Okapi([_tok(item["text"]) for item in metadata])

    results: Dict[str, List[Dict]] = {}
    for q in queries:
        scores = bm25.get_scores(_tok(q["question"]))
        top_i  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.top_k]
        results[q["query_id"]] = [{
            "chunk_id":        metadata[i]["chunk_id"],
            "doc_id":          metadata[i]["doc_id"],
            "score":           float(scores[i]),
            "text":            metadata[i]["text"],
            "title":           metadata[i].get("title"),
            "source":          metadata[i].get("source"),
            "is_spoof":        metadata[i].get("is_spoof", False),
            "spoof_for_query": metadata[i].get("spoof_for_query"),
            "attack_type":     metadata[i].get("attack_type"),
        } for i in top_i]

    _write_json(args.output_path, results)
    print(f"Saved BM25 results → {args.output_path}")

if __name__ == "__main__":
    main()
