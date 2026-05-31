from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from src.utils import get_logger, read_json, write_json, read_jsonl

LOGGER = get_logger(__name__)

def _build_lookup(items: List[Dict], key: str) -> Dict:
    return {item[key]: item for item in items}

def _is_relevant(item: Dict, query_obj: Dict) -> bool:
    cid = item.get("chunk_id", "")
    did = item.get("doc_id", "")
    if cid in set(query_obj.get("relevant_chunk_ids", [])):
        return True
    prefix = query_obj.get("relevant_chunk_prefix")
    if prefix and cid.startswith(prefix):
        return True
    if did == query_obj.get("relevant_doc_id"):
        return True
    return False

def recall_at_k(results: List[Dict], qlookup: Dict, k: int) -> float:
    hits = []
    for r in results:
        qobj = qlookup.get(r["query_id"])
        if not qobj:
            continue
        hits.append(1.0 if any(_is_relevant(x, qobj) for x in r.get("retrieved", [])[:k]) else 0.0)
    return float(np.mean(hits)) if hits else 0.0

def _first_relevant_rank(r: Dict, qobj: Dict) -> Optional[int]:
    for rank, x in enumerate(r.get("retrieved", []), 1):
        if _is_relevant(x, qobj):
            return rank
    return None

def rank_displacement(baseline: List[Dict], attack: List[Dict], qlookup: Dict) -> float:
    atk_map = _build_lookup(attack, "query_id")
    disps = []
    for br in baseline:
        qobj = qlookup.get(br["query_id"])
        ar   = atk_map.get(br["query_id"])
        if not qobj or not ar:
            continue
        rb = _first_relevant_rank(br, qobj)
        ra = _first_relevant_rank(ar, qobj)
        if rb and ra:
            disps.append(ra - rb)
    return float(np.mean(disps)) if disps else 0.0

def top1_spoof_win_rate(attack: List[Dict]) -> float:
    wins = [1 for r in attack if r.get("retrieved") and
            (r["retrieved"][0].get("label") == "injected" or
             r["retrieved"][0].get("is_spoof", False))]
    return len(wins) / len(attack) if attack else 0.0

def attraction_margin(attack: List[Dict]) -> float:
    margins = []
    for r in attack:
        top_sp = top_re = None
        for x in r.get("retrieved", []):
            is_sp = x.get("is_spoof", False)
            sc    = x.get("score", 0.0)
            if is_sp:
                top_sp = sc if top_sp is None else max(top_sp, sc)
            else:
                top_re = sc if top_re is None else max(top_re, sc)
        if top_sp is not None and top_re is not None:
            margins.append(top_sp - top_re)
    return float(np.mean(margins)) if margins else 0.0

def spoof_diversity(spoof_chunks: List[Dict]) -> Dict:
    """Diversity stats computed directly from spoof_chunks.jsonl."""
    by_style: Dict[str, int] = {}
    for c in spoof_chunks:
        style = c.get("attack_type", "unknown")
        by_style[style] = by_style.get(style, 0) + 1
    texts  = [c.get("text", "") for c in spoof_chunks]
    words  = [set(t.lower().split()) for t in texts]
    # pairwise Jaccard on a sample (max 500)
    sample = words[:500]
    jaccards = []
    for i in range(len(sample)):
        for j in range(i + 1, min(i + 10, len(sample))):
            u = sample[i] | sample[j]
            jaccards.append(len(sample[i] & sample[j]) / len(u) if u else 0)
    return {
        "total_spoof_chunks": len(spoof_chunks),
        "by_style": by_style,
        "avg_pairwise_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 0.0,
        "diversity_score": round(1 - float(np.mean(jaccards)), 4) if jaccards else 1.0,
    }

def _norm_results(raw) -> List[Dict]:
    """Accept both list-of-dicts and dict-keyed formats."""
    if isinstance(raw, list):
        return raw
    # dict format: {query_id: [retrieved_items]}
    out = []
    for qid, items in raw.items():
        out.append({"query_id": qid, "retrieved": items})
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results", default="results/retrieval/minilm_baseline_results.json")
    parser.add_argument("--attack-results",   default="results/retrieval/minilm_attack_results.json")
    parser.add_argument("--queries",          default="data/processed/retrieval_eval_queries.json")
    parser.add_argument("--spoof-chunks",     default="data/processed/spoof_chunks.jsonl",
                        help="spoof_chunks.jsonl (replaces old spoof_candidates.json)")
    parser.add_argument("--top-k",            type=int, default=5)
    parser.add_argument("--output",           default="results/retrieval/attack_metrics.json")
    args = parser.parse_args()

    baseline_raw  = read_json(args.baseline_results)
    attack_raw    = read_json(args.attack_results)
    queries       = read_json(args.queries)
    spoof_chunks  = read_jsonl(args.spoof_chunks)

    baseline = _norm_results(baseline_raw)
    attack   = _norm_results(attack_raw)
    qlookup  = _build_lookup(queries, "query_id")

    metrics = {
        "top_k":                   args.top_k,
        "num_queries":             len(queries),
        "recall_at_k_baseline":    recall_at_k(baseline, qlookup, args.top_k),
        "recall_at_k_attack":      recall_at_k(attack,   qlookup, args.top_k),
        "recall_at_k_drop":        recall_at_k(baseline, qlookup, args.top_k) - recall_at_k(attack, qlookup, args.top_k),
        "rank_displacement":       rank_displacement(baseline, attack, qlookup),
        "top1_spoof_win_rate":     top1_spoof_win_rate(attack),
        "retrieval_attraction_margin": attraction_margin(attack),
        "spoof_diversity":         spoof_diversity(spoof_chunks),
    }

    write_json(metrics, args.output)
    LOGGER.info("Attack metrics saved → %s", args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
