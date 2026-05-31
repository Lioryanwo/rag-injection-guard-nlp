from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

LOGGER_NAME = "rag_spoofing"

def get_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)
    return logger

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def ensure_dir(path: str | Path) -> Path:
    path = Path(path); path.mkdir(parents=True, exist_ok=True); return path

def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path); ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path); ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
