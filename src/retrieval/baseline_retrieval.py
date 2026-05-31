from __future__ import annotations
import argparse, json, os, pickle, time
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False


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


def _prefix(questions: List[str], model_name: str) -> List[str]:
    return [f"query: {q}" for q in questions] if "e5" in model_name.lower() else questions


# ── HyDE: Hypothetical Document Embedding ────────────────────────────────────
#
# Sasha's description in the lecture:
#   "Instead of embedding the question and searching for matching passages,
#    you generate a hypothetical document — the best answer you expect to find —
#    and embed THAT.  The idea is that the embedding space of answers is closer
#    to the embedding space of corpus passages than the embedding space of
#    questions."
#
# Why this improves baseline recall:
#   Questions like "Who invented the telephone?" live in a different region of
#   embedding space than passages like "Alexander Graham Bell patented...".
#   A hypothetical document "Alexander Graham Bell invented the telephone in
#   1876." is much closer to the actual passage, so cosine search finds it.
#
# Why this also makes the attack harder:
#   The attack needs to fool the embedding of a *complete plausible answer*,
#   not just the embedding of a short question.  That's a stronger target.

_HYDE_SYSTEM = (
    "You are a helpful assistant that writes short factual passages. "
    "Given a question, write ONE short paragraph (3-4 sentences) that looks like "
    "a Wikipedia or textbook passage that would answer the question. "
    "Write only the passage, no preamble, no 'Answer:' prefix."
)

_HYDE_CACHE: Dict[str, str] = {}


def _generate_hyde_doc(question: str, client: "OpenAI", model: str = "gpt-4o-mini") -> str:
    """Generate one hypothetical document for the question using an LLM."""
    if question in _HYDE_CACHE:
        return _HYDE_CACHE[question]
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user",   "content": question},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        text = r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[HyDE fallback] {e}")
        text = question   # fall back to raw question
    _HYDE_CACHE[question] = text
    return text


def _embed_hyde(
    questions: List[str],
    embedder: SentenceTransformer,
    model_name: str,
    client: Optional["OpenAI"],
    batch_size: int = 32,
    hyde_llm: str = "gpt-4o-mini",
    rate_limit_pause: float = 0.05,
) -> np.ndarray:
    """
    For each question:
      1. Generate a hypothetical document (or fall back to the question itself).
      2. Embed the hypothetical document with the retrieval embedder.

    If no LLM client is provided, fall back to plain question embedding
    (standard dense retrieval).
    """
    if client is None:
        texts = _prefix(questions, model_name)
        return embedder.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")

    print(f"Generating {len(questions)} HyDE documents via {hyde_llm}…")
    hyde_docs = []
    for i, q in enumerate(questions):
        doc = _generate_hyde_doc(q, client, hyde_llm)
        hyde_docs.append(doc)
        if (i + 1) % 50 == 0:
            print(f"  HyDE: {i + 1}/{len(questions)}")
            time.sleep(rate_limit_pause)

    # Prefix as passage (not query) because the hypothetical doc is a passage
    if "e5" in model_name.lower():
        texts = [f"passage: {d}" for d in hyde_docs]
    else:
        texts = hyde_docs

    return embedder.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--index-dir",    type=Path, default=Path("indexes/base"))
    parser.add_argument("--output-path",  type=Path, default=Path("results/retrieval/baseline_results.json"))
    parser.add_argument("--model-name",   type=str,  default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k",        type=int,  default=5)
    parser.add_argument("--batch-size",   type=int,  default=64)

    # ── HyDE controls ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--use-hyde", action="store_true", default=False,
        help=(
            "Enable Hypothetical Document Embedding (HyDE). "
            "Instead of embedding the question, generate a hypothetical answer "
            "passage and embed that. Requires OPENAI_API_KEY."
        ),
    )
    parser.add_argument("--hyde-model", type=str, default="gpt-4o-mini",
                        help="LLM used to generate hypothetical documents.")
    args = parser.parse_args()

    queries = _read_jsonl(args.queries_path)
    for p in (args.index_dir / "index.faiss", args.index_dir / "metadata.pkl"):
        if not p.exists():
            raise FileNotFoundError(p)

    index = faiss.read_index(str(args.index_dir / "index.faiss"))
    with (args.index_dir / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)

    print(f"Loading embedding model: {args.model_name}")
    embedder = SentenceTransformer(args.model_name, local_files_only=True)

    # Optionally set up HyDE LLM client
    llm_client: Optional["OpenAI"] = None
    if args.use_hyde:
        if not _OPENAI_OK:
            raise ImportError("pip install openai  (required for --use-hyde)")
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set — required for --use-hyde")
        llm_client = OpenAI(api_key=key)
        print(f"HyDE enabled — generating hypothetical docs with {args.hyde_model}")
    else:
        print("HyDE disabled — using plain question embeddings")

    questions = [q["question"] for q in queries]

    # Embed queries (or hypothetical documents if HyDE is on)
    q_embs = _embed_hyde(
        questions=questions,
        embedder=embedder,
        model_name=args.model_name,
        client=llm_client,
        batch_size=args.batch_size,
        hyde_llm=args.hyde_model,
    )

    all_scores, all_idxs = index.search(q_embs, args.top_k)
    results: Dict[str, List[Dict]] = {}
    for q, scores_row, idxs_row in zip(queries, all_scores, all_idxs):
        ranked = []
        for score, idx in zip(scores_row, idxs_row):
            if idx < 0 or idx >= len(metadata):
                continue
            item = metadata[idx]
            ranked.append({
                "chunk_id":        item["chunk_id"],
                "doc_id":          item["doc_id"],
                "score":           float(score),
                "text":            item["text"],
                "title":           item.get("title"),
                "source":          item.get("source"),
                "is_spoof":        item.get("is_spoof", False),
                "spoof_for_query": item.get("spoof_for_query"),
                "attack_type":     item.get("attack_type"),
            })
        results[q["query_id"]] = ranked

    _write_json(args.output_path, results)
    mode = f"HyDE({args.hyde_model})" if args.use_hyde else "plain-dense"
    print(f"Saved → {args.output_path}  [{mode}]")


if __name__ == "__main__":
    main()
