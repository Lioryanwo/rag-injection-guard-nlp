from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]

SEEDS = [42, 43, 44, 45, 46]

METRIC_FILES = {
    "minilm_attack":   "results/retrieval/minilm_attack_top20_metrics.json",
    "minilm_defense":  "results/retrieval/minilm_defense_metrics.json",

    "bm25_attack":     "results/retrieval/bm25_attack_top20_metrics.json",
    "bm25_defense":    "results/retrieval/bm25_defense_metrics.json",

    "hybrid_attack":   "results/retrieval/hybrid_attack_top20_metrics.json",
    "hybrid_defense":  "results/retrieval/hybrid_defense_metrics.json",
}


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, cwd=ROOT, check=True)


def load_json(path: str):
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return json.load(f)


def stderr(values):
    if len(values) <= 1:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def main():

    py = sys.executable

    all_runs = []

    for seed in SEEDS:

        print("\n" + "#" * 80)
        print(f"VALIDATION RUN — seed={seed}")
        print("#" * 80)

        random.seed(seed)

        # ── Run full pipeline ─────────────────────────────
        run([py, "-m", "src.pipeline.run_pipeline"])

        run_metrics = {
            "seed": seed,
            "metrics": {}
        }

        for name, path in METRIC_FILES.items():

            if not (ROOT / path).exists():
                print(f"[WARN] missing: {path}")
                continue

            m = load_json(path)

            run_metrics["metrics"][name] = {
                "recall@5": m.get("recall@5"),
                "top1_spoof_win_rate": m.get("top1_spoof_win_rate"),
            }

        all_runs.append(run_metrics)

    # ── Aggregate ────────────────────────────────────────

    aggregate = {}

    for exp in METRIC_FILES:

        recall_vals = []
        spoof_vals = []

        for run_data in all_runs:

            if exp not in run_data["metrics"]:
                continue

            recall_vals.append(
                run_data["metrics"][exp]["recall@5"]
            )

            spoof_vals.append(
                run_data["metrics"][exp]["top1_spoof_win_rate"]
            )

        aggregate[exp] = {
            "recall@5": {
                "mean": round(mean(recall_vals), 4),
                "std": round(stdev(recall_vals), 4),
                "stderr": round(stderr(recall_vals), 4),
            },
            "top1_spoof_win_rate": {
                "mean": round(mean(spoof_vals), 4),
                "std": round(stdev(spoof_vals), 4),
                "stderr": round(stderr(spoof_vals), 4),
            }
        }

    output = {
        "num_runs": len(SEEDS),
        "seeds": SEEDS,
        "runs": all_runs,
        "aggregate": aggregate,
    }

    out_path = ROOT / "results/retrieval/validation_summary.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    for exp, vals in aggregate.items():

        r = vals["recall@5"]
        s = vals["top1_spoof_win_rate"]

        print(
            f"{exp:20} | "
            f"Recall@5 = {r['mean']:.4f} ± {r['stderr']:.4f} | "
            f"SpoofWin = {s['mean']:.4f} ± {s['stderr']:.4f}"
        )

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()