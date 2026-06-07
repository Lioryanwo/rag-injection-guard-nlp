from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
from src.utils import get_logger

"""
=============================================================================
Purpose:
A utility script to create a "naive" Top-K view from a larger retrieval pool.
For example, it takes a Top-20 retrieval result and strictly truncates it 
to a Top-5 result. This is used to create a naive baseline comparator 
to prove that taking a larger pool (Top-20) and applying a smart defense reranker 
is better than simply taking the Top-5.

Inputs:
- --input-path: JSON file containing the larger retrieval pool (e.g., Top-20).
- --top-k: The maximum number of documents to keep per query (default: 5).

Outputs:
- --output-path: JSON file containing the truncated results.
=============================================================================
"""

# Set up logger
script_name = Path(__file__).stem
folder_name = Path(__file__).parent.name
logger = get_logger(name=script_name, group=folder_name)


def _load(path: Path) -> Dict[str, List[dict]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def trim_top_k(results: Dict[str, List[dict]], top_k: int) -> Dict[str, List[dict]]:
    return {qid: ranked[:top_k] for qid, ranked in results.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = _load(args.input_path)
    trimmed = trim_top_k(results, args.top_k)
    _save(args.output_path, trimmed)
    print(f"Saved top-{args.top_k} view -> {args.output_path}")

    logger.info(f"Successfully created Top-{args.top_k} view for {len(trimmed)} queries.")
    logger.info(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
