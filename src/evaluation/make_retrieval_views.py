from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


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


if __name__ == "__main__":
    main()
