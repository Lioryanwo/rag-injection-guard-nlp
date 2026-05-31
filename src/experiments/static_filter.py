from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.defense.defense_filter import no_query_suspicion


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.array(values, dtype=np.float32)
    return {
        "min":    round(float(np.min(arr)), 4),
        "p25":    round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.percentile(arr, 50)), 4),
        "p75":    round(float(np.percentile(arr, 75)), 4),
        "p90":    round(float(np.percentile(arr, 90)), 4),
        "p95":    round(float(np.percentile(arr, 95)), 4),
        "max":    round(float(np.max(arr)), 4),
        "mean":   round(float(np.mean(arr)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Query-blind static filter over the whole corpus.\n\n"
            "Sasha's role: this is the 'pre-filter' baseline. If spoof chunks score "
            "high here, they can be rejected without ever seeing the user's query. "
            "If they score low (overlap with real chunks), the attack is truly hard "
            "and only query-aware reranking can defend against it."
        )
    )
    parser.add_argument("--chunks-path",  type=Path, default=Path("data/processed/augmented_chunks.jsonl"))
    parser.add_argument("--output-path",  type=Path, default=Path("results/retrieval/static_filter_report.json"))
    parser.add_argument("--threshold",    type=float, default=0.30)
    parser.add_argument("--top-examples", type=int,   default=25,
                        help="How many highest-scoring examples to keep for inspection.")
    args = parser.parse_args()

    rows = _read_jsonl(args.chunks_path)

    totals = {
        "all": len(rows),
        "real": 0, "spoof": 0,
        "flagged": 0, "flagged_real": 0, "flagged_spoof": 0,
    }

    flagged: List[Dict] = []
    scored_items: List[Dict] = []
    real_scores: List[float] = []
    spoof_scores: List[float] = []
    all_scores: List[float] = []

    for row in rows:
        is_spoof = bool(row.get("is_spoof", False))
        totals["spoof" if is_spoof else "real"] += 1

        score, detail = no_query_suspicion(row.get("text", ""))
        score = float(score)
        all_scores.append(score)
        (spoof_scores if is_spoof else real_scores).append(score)

        item = {
            "chunk_id":        row.get("chunk_id"),
            "doc_id":          row.get("doc_id"),
            "is_spoof":        is_spoof,
            "spoof_for_query": row.get("spoof_for_query"),
            "attack_type":     row.get("attack_type"),
            "score":           round(score, 4),
            "reasons":         detail.get("reasons", []),
            "detail":          detail,
            "text_preview":    row.get("text", "")[:240],
        }
        scored_items.append(item)

        if score >= args.threshold:
            totals["flagged"] += 1
            totals["flagged_spoof" if is_spoof else "flagged_real"] += 1
            flagged.append(item)

    real_count  = totals["real"]  or 1
    spoof_count = totals["spoof"] or 1

    false_positive_rate = totals["flagged_real"]  / real_count
    spoof_detection_rate = totals["flagged_spoof"] / spoof_count

    top_static_scores = sorted(scored_items, key=lambda x: x["score"], reverse=True)[: args.top_examples]

    # ── Interpretation (Sasha's framing) ──────────────────────────────────────
    if spoof_detection_rate > 0.70 and false_positive_rate < 0.10:
        interpretation_verdict = (
            "SPOOFS ARE DETECTABLE WITHOUT QUERY. "
            "This means the attack is too easy — a pre-filter can remove spoofs before indexing. "
            "The attack should be redesigned to produce more legitimate-looking chunks."
        )
    elif spoof_detection_rate < 0.30:
        interpretation_verdict = (
            "SPOOFS LOOK LEGITIMATE (low static detection). "
            "This is the desired property for a hard attack. "
            "Static pre-filtering is insufficient — query-aware reranking is needed. "
            "This validates Sasha's claim that only query-time defense can work."
        )
    else:
        interpretation_verdict = (
            "PARTIAL STATIC DETECTABILITY. "
            "Some spoofs have surface markers, others do not. "
            "A combination of pre-filtering and query-aware reranking is likely optimal."
        )

    score_overlap = (
        "SCORES OVERLAP" if (
            _summary_stats(spoof_scores)["median"] < _summary_stats(real_scores)["p75"]
        ) else "SCORES SEPARATED"
    )

    report = {
        "threshold":          args.threshold,
        "totals":             totals,
        "false_positive_rate_real_flagged": round(false_positive_rate, 4),
        "spoof_detection_rate":             round(spoof_detection_rate, 4),
        "score_overlap_verdict":            score_overlap,
        "interpretation_verdict":           interpretation_verdict,
        "score_distribution": {
            "all":   _summary_stats(all_scores),
            "real":  _summary_stats(real_scores),
            "spoof": _summary_stats(spoof_scores),
        },
        # ── Sasha's framing of what this proves ───────────────────────────────
        "sasha_framing": (
            "If spoof_detection_rate is LOW: spoofs look like real chunks → "
            "pre-filtering fails → only query-aware defense works → our method is justified.\n"
            "If spoof_detection_rate is HIGH: spoofs have surface markers → "
            "attack is trivial → redesign the attack first."
        ),
        "flagged_items":                   flagged,
        "top_static_scores_for_inspection": top_static_scores,
    }

    _write_json(args.output_path, report)

    # Compact print (without long lists)
    compact = {k: v for k, v in report.items()
               if k not in {"flagged_items", "top_static_scores_for_inspection"}}
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"\nSaved static filter report → {args.output_path}")


if __name__ == "__main__":
    main()
