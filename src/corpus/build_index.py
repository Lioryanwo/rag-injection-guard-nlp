from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
from typing import Dict, List
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/corpus_chunks.jsonl"))
    parser.add_argument("--index-dir",   type=Path, default=Path("indexes/base"))
    parser.add_argument("--model-name",  type=str,  default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size",  type=int,  default=64)
    args = parser.parse_args()
    args.index_dir.mkdir(parents=True, exist_ok=True)
    chunks = read_jsonl(args.chunks_path)
    texts  = [f"passage: {c['text']}" if "e5" in args.model_name.lower() else c["text"] for c in chunks]
    model  = SentenceTransformer(args.model_name)
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(args.index_dir / "index.faiss"))
    with (args.index_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(chunks, f)
    np.save(args.index_dir / "embeddings.npy", embeddings)
    print(f"Indexed {len(chunks)} chunks | dim={embeddings.shape[1]} | model={args.model_name}")

if __name__ == "__main__":
    main()
