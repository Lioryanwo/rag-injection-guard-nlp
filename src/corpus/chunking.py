from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Why 120/30 instead of the previous 90/40:
    - SQuAD passages are ~100-200 words.  At 90 words we often split a passage
      into many tiny chunks, spreading the answer span across chunk boundaries.
    - 120 words keeps most SQuAD passages as a single chunk (or two at most),
      so the relevant answer span stays inside one chunk.
    - Overlap of 30 (not 40) avoids the previous problem where overlap was so
      large that almost every boundary region was duplicated three times,
      inflating the index with near-identical chunks that confused the retriever.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",   type=Path, default=Path("data/processed/corpus_docs.jsonl"))
    parser.add_argument("--output-path",  type=Path, default=Path("data/processed/corpus_chunks.jsonl"))
    parser.add_argument("--chunk-size",   type=int,  default=120,
                        help="Words per chunk. 120 keeps most SQuAD passages intact.")
    parser.add_argument("--overlap",      type=int,  default=30,
                        help="Overlap between consecutive chunks in words.")
    parser.add_argument("--prepend-title", action="store_true", default=True,
                        help="Prepend 'Title: <title>.' to every chunk so the retriever "
                             "sees the document topic even for mid-document chunks.")
    args = parser.parse_args()

    docs = read_jsonl(args.input_path)
    chunk_rows = []
    for doc in docs:
        title = doc.get("title", "").strip()
        chunks = chunk_text(doc["context"], args.chunk_size, args.overlap)
        for i, chunk in enumerate(chunks):
            # Prepend the title so every chunk carries the document topic.
            # This is the single cheapest improvement to baseline recall:
            # a question about "Beyoncé" now matches a chunk that starts with
            # "Title: Beyoncé." even if the body of the chunk drifts off-topic.
            text = f"{title}. {chunk}" if (args.prepend_title and title) else chunk
            chunk_rows.append({
                "chunk_id":        f"{doc['doc_id']}::chunk_{i}",
                "doc_id":          doc["doc_id"],
                "title":           title,
                "text":            text,
                "source":          doc["source"],
                "source_split":    doc["source_split"],
                "is_spoof":        False,
                "spoof_for_query": None,
                "attack_type":     None,
            })

    write_jsonl(args.output_path, chunk_rows)
    print(f"Saved {len(chunk_rows)} chunks → {args.output_path}")
    print(f"  chunk_size={args.chunk_size}  overlap={args.overlap}  prepend_title={args.prepend_title}")


if __name__ == "__main__":
    main()
