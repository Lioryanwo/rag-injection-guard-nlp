from __future__ import annotations
import argparse, json, random
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-chunks", type=Path, default=Path("data/processed/corpus_chunks.jsonl"))
    parser.add_argument("--spoof-chunks", type=Path, default=Path("data/processed/spoof_chunks.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/augmented_chunks.jsonl"))
    parser.add_argument("--max-spoofs", type=int, default=None, help="Optional cap on number of spoof chunks injected")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    real = read_jsonl(args.real_chunks)
    spoof = read_jsonl(args.spoof_chunks)

    if args.max_spoofs is not None:
        spoof = spoof[: args.max_spoofs]

    combined = real + spoof
    write_jsonl(args.output_path, combined)

    print(f"Real: {len(real)} | Spoof injected: {len(spoof)} | Total: {len(combined)}")


if __name__ == "__main__":
    main()