from __future__ import annotations
import argparse, json, pickle, re
from pathlib import Path
from typing import Dict, List
import faiss, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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

def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return np.ones_like(v, dtype=np.float32) if abs(hi - lo) < 1e-12 else ((v - lo) / (hi - lo)).astype(np.float32)

def _prefix(q: str, model_name: str) -> str:
    return f"query: {q}" if "e5" in model_name.lower() else q

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--index-dir",    type=Path, default=Path("indexes/bge_m3_base"))
    parser.add_argument("--output-path",  type=Path, default=Path("results/retrieval/hybrid_results.json"))
    parser.add_argument("--model-name",   type=str,  default="BAAI/bge-m3")
    parser.add_argument("--top-k",        type=int,  default=5)
    parser.add_argument("--candidate-k",  type=int,  default=50)
    parser.add_argument("--alpha",        type=float, default=0.6)
    args = parser.parse_args()

    queries  = _read_jsonl(args.queries_path)
    index    = faiss.read_index(str(args.index_dir / "index.faiss"))
    with (args.index_dir / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)
    model = SentenceTransformer(args.model_name, local_files_only=True)
    bm25  = BM25Okapi([_tok(item["text"]) for item in metadata])

    results: Dict[str, List[Dict]] = {}
    for q in queries:
        q_emb = model.encode([_prefix(q["question"], args.model_name)],
                             convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        d_scores, d_idxs = index.search(q_emb, args.candidate_k)
        d_scores, d_idxs = d_scores[0], d_idxs[0]

        bm25_all = bm25.get_scores(_tok(q["question"]))
        cands    = set(int(i) for i in d_idxs) | set(int(i) for i in np.argsort(bm25_all)[::-1][:args.candidate_k])
        cands    = sorted(cands)

        d_map   = {int(i): float(s) for i, s in zip(d_idxs, d_scores)}
        d_vals  = np.array([d_map.get(i, 0.0) for i in cands], dtype=np.float32)
        b_vals  = np.array([float(bm25_all[i]) for i in cands], dtype=np.float32)
        hybrid  = args.alpha * _minmax(d_vals) + (1 - args.alpha) * _minmax(b_vals)

        top = sorted(zip(cands, hybrid, _minmax(d_vals), _minmax(b_vals)),
                     key=lambda x: x[1], reverse=True)[:args.top_k]
        results[q["query_id"]] = [{
            "chunk_id":        metadata[i]["chunk_id"],
            "doc_id":          metadata[i]["doc_id"],
            "score":           float(h),
            "dense_score_norm":float(dn),
            "bm25_score_norm": float(bn),
            "text":            metadata[i]["text"],
            "title":           metadata[i].get("title"),
            "source":          metadata[i].get("source"),
            "is_spoof":        metadata[i].get("is_spoof", False),
            "spoof_for_query": metadata[i].get("spoof_for_query"),
            "attack_type":     metadata[i].get("attack_type"),
        } for i, h, dn, bn in top]

    _write_json(args.output_path, results)
    print(f"Saved hybrid results → {args.output_path}  (alpha={args.alpha})")

if __name__ == "__main__":
    main()
