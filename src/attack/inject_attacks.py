from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Dict, List
from src.utils import get_logger

"""
=============================================================================
Purpose:
This script performs the actual "data poisoning" operation. It takes the clean, 
original document chunks and appends the synthetically generated spoof chunks 
to create a single, combined corpus. This "augmented" corpus is then passed to 
the vector database indexer, embedding the attacker's fake documents directly 
into the retrieval system's knowledge base.

Inputs:
- corpus_chunks.jsonl: The clean, real document chunks.
- spoof_chunks.jsonl: The malicious, generated fake chunks.
- Parameters: --max-spoofs (optional limit for the attack scale).

Outputs (saved to data/processed/):
- augmented_chunks.jsonl: The final poisoned corpus containing both real 
  and fake chunks, ready for FAISS indexing.
=============================================================================
"""

# Initialize a logger for this script
script_name = Path(__file__).stem
folder_name = Path(__file__).parent.name
logger = get_logger(name=script_name, group=folder_name)


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


def main() -> None:
    # --- 1. CLI Arguments Parsing ---
    # Define input/output paths and the optional limit for spoofs.
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-chunks", type=Path, default=Path("data/processed/corpus_chunks.jsonl"))
    parser.add_argument("--spoof-chunks", type=Path, default=Path("data/processed/spoof_chunks.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/augmented_chunks.jsonl"))
    #parser.add_argument("--max-spoofs", type=int, default=None, help="Optional cap on number of spoof chunks injected")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # --- 2. Data Loading ---
    # Load the clean corpus and the generated attack chunks.
    real = read_jsonl(args.real_chunks)
    spoof = read_jsonl(args.spoof_chunks)
    logger.info(f"Loaded {len(real)} real chunks and {len(spoof)} available spoof chunks.")

    # --- 3. Attack Limitation (Global Cutoff) ---
    # If max_spoofs is set, truncate the list. Note: This is a blind global cutoff.
    #if args.max_spoofs is not None:
    #    spoof = spoof[: args.max_spoofs]
    #    logger.info(f"Capping injected spoofs to {args.max_spoofs} globally.")

    # --- 4. Poisoning & Saving ---
    # Concatenate the lists to create the poisoned corpus and save it.
    combined = real + spoof
    write_jsonl(args.output_path, combined)

    logger.info("Successfully injected spoof chunks into the real corpus.")
    logger.info(f"Final Corpus Stats -> Real: {len(real)} | Spoofs: {len(spoof)} | Total: {len(combined)}")
    logger.info(f"Poisoned corpus saved to: {args.output_path}")
    print(f"Real: {len(real)} | Spoof injected: {len(spoof)} | Total: {len(combined)}")


if __name__ == "__main__":
    main()