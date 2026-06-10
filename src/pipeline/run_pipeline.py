from __future__ import annotations
import os
import time
import subprocess
import sys
from pathlib import Path
from src.utils import get_logger

ROOT = Path(__file__).resolve().parents[2]

# ── Corpus / chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 120   # keeps most SQuAD passages in one chunk
CHUNK_OVERLAP = 30    # less boundary duplication vs the old 40

# ── Retrieval pool sizes ───────────────────────────────────────────────────────
BASELINE_TOP_K       = 5
RETRIEVAL_POOL_TOP_K = 20   # all defense conditions start from this pool
FINAL_KEEP_TOP_K     = 5

# ── Attack ────────────────────────────────────────────────────────────────────
# "semantically attractive but evidence-poor retrieval obstruction"
ATTACK_MAX_QUERIES          = 300
ATTACK_MODE                 = "hypothetical_distractor"   # default: Sasha-aligned
ATTACK_CANDIDATES_PER_STYLE = 3
ATTACK_KEEP_PER_STYLE       = 1
ATTACK_EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"

# ── Defense ───────────────────────────────────────────────────────────────────
DEFENSE_THRESHOLD              = 0.30
DEFENSE_CROSS_ENCODER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFENSE_SEMANTIC_WEIGHT        = 0.65
DEFENSE_RETRIEVAL_WEIGHT       = 0.10
DEFENSE_LEXICAL_PENALTY_WEIGHT = 0.05
DEFENSE_DOC2QUERY_WEIGHT       = 0.25   # main new signal: answerability
DEFENSE_BATCH_SIZE             = 16

# ── Reverse QA defense (LLM-based, requires OPENAI_API_KEY) ──────────────────
# Set to True to enable the Reverse QA layer on top of CrossEncoder reranking.
# Generates 5 questions per chunk via GPT-4o-mini and scores them against the
# original query. Adds a relevance bonus that penalizes spoof chunks.
# Trade-off: ~20x slower per query (LLM call per chunk), but cached after first run.
USE_REVERSE_QA            = False
REVERSE_QA_WEIGHT         = 0.25    # bonus weight in final score
REVERSE_QA_NUM_QUESTIONS  = 5
REVERSE_QA_QG_BACKEND     = "openai"
REVERSE_QA_OPENAI_MODEL   = "gpt-4o-mini"
REVERSE_QA_CACHE_PATH     = "results/reverse_qa_cache.jsonl"

# ── HyDE baseline (OPTIONAL — disabled by default) ────────────────────────────
# HyDE uses an LLM to generate a hypothetical answer document and embeds that
# instead of the raw question.  It improves recall but changes the baseline
# condition.  Keeping it OFF ensures a fair apples-to-apples comparison of
# plain retrieval vs attack vs defense.
#
# To enable: set USE_HYDE_BASELINE = True and ensure OPENAI_API_KEY is set.
# When enabled, an ADDITIONAL set of results is saved under the prefix
# "{ret}_hyde_baseline_*" so it does NOT overwrite the standard baseline.
USE_HYDE_BASELINE = False

RUN_LLM_JUDGE = True

def setup_run_environment():
    """Sets up a unique logging directory for the current run, based on a timestamp."""
    # Create a timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Set up the new log directory path
    log_dir = Path(ROOT) / "logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Inject the path into the environment variable so all subsequent scripts know about it
    os.environ["CURRENT_RUN_LOG_DIR"] = str(log_dir)


def run(cmd: list) -> None:
    logger = get_logger(name=Path(__file__).parent.name, group=Path(__file__).stem)
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"RUN: {cmd_str}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def eval_ret(py, results, output, attacked_only: bool = True):
    """Evaluate retrieval results.

    When attacked_only=True, restricts BOTH results and qrels to the
    N=300 queries that have generated spoof chunks — ensuring all
    conditions (baseline / attack / defense) are on the same query set.
    """
    cmd = [
        py, "-m", "src.evaluation.evaluate_retrieval",
        "--results-path", results,
        "--output-path",  output,
    ]
    if attacked_only:
        cmd += ["--spoof-chunks-path", "data/processed/spoof_chunks.jsonl"]
    run(cmd)


def make_topk_view(py, input_path, output_path, top_k):
    """Naive top-k slice (no reranking) — Sasha's 'dumb comparator'."""
    run([
        py, "-m", "src.evaluation.make_retrieval_views",
        "--input-path",  input_path,
        "--output-path", output_path,
        "--top-k",       str(top_k),
    ])


def maybe_llm(py, results, output, top_k=1):
    if not RUN_LLM_JUDGE:
        print(f"[skip] LLM Judge disabled: {results}")
        return
    run([
        py, "-m", "src.evaluation.evaluate_llm",
        "--results-path", results,
        "--output-path",  output,
        "--top-k",        str(top_k),
        "--max-queries",  "100",
    ])


def defense(py, input_path, output_path, mode="cross_encoder"):
    """Run a defense pass over a Top-20 candidate pool."""
    cmd = [
        py, "-m", "src.defense.defense_filter",
        "--input-path",           input_path,
        "--queries-path",         "data/processed/val_queries.jsonl",
        "--output-path",          output_path,
        "--keep-top-k",           str(FINAL_KEEP_TOP_K),
        "--suspicion-threshold",  str(DEFENSE_THRESHOLD),
        "--defense-mode",         mode,
    ]
    if mode == "cross_encoder":
        cmd += [
            "--cross-encoder-model",        DEFENSE_CROSS_ENCODER_MODEL,
            "--semantic-weight",            str(DEFENSE_SEMANTIC_WEIGHT),
            "--retrieval-weight",           str(DEFENSE_RETRIEVAL_WEIGHT),
            "--lexical-penalty-weight",     str(DEFENSE_LEXICAL_PENALTY_WEIGHT),
            "--batch-size",                 str(DEFENSE_BATCH_SIZE),
            "--use-doc2query",
            "--doc2query-weight",           str(DEFENSE_DOC2QUERY_WEIGHT),
        ]
        if USE_REVERSE_QA:
            cmd += [
                "--use-reverse-qa",
                "--reverse-qa-weight",          str(REVERSE_QA_WEIGHT),
                "--reverse-qa-num-questions",   str(REVERSE_QA_NUM_QUESTIONS),
                "--reverse-qa-qg-backend",      REVERSE_QA_QG_BACKEND,
                "--reverse-qa-openai-model",    REVERSE_QA_OPENAI_MODEL,
                "--reverse-qa-cache-path",      REVERSE_QA_CACHE_PATH,
            ]
    run(cmd)


def retrieval_run(py, module, queries, top_k, output, extra):
    """Run a retrieval module.  HyDE is NEVER injected into the standard baseline."""
    cmd = [
        py, "-m", module,
        "--queries-path", queries,
        "--top-k",        str(top_k),
        "--output-path",  output,
        *extra,
    ]
    return cmd


def main() -> None:
    py = sys.executable

    # Set up a unique logging directory for this run.
    setup_run_environment()

    # Initialize logger
    logger = get_logger(name=Path(__file__).parent.name, group=Path(__file__).stem)

    # ── 1. Corpus ─────────────────────────────────────────────────────────────
    logger.info("Starting Data Prep: Creating corpus, chunking, and building clean indexes.")
    run([py, "-m", "src.corpus.create_corpus",
         "--train-size", "5000", "--validation-size", "1000"])
    run([
        py, "-m", "src.corpus.chunking",
        "--chunk-size",    str(CHUNK_SIZE),
        "--overlap",       str(CHUNK_OVERLAP),
        "--prepend-title",
    ])
    for idx, model in [
        ("indexes/minilm_base", "sentence-transformers/all-MiniLM-L6-v2"),
        ("indexes/bge_m3_base", "BAAI/bge-m3"),
    ]:
        run([
            py, "-m", "src.corpus.build_index",
            "--chunks-path", "data/processed/corpus_chunks.jsonl",
            "--index-dir",   idx,
            "--model-name",  model,
        ])
    run([py, "-m", "src.corpus.build_retrieval_eval_queries"])

    # ── 2. Generate spoof chunks ───────────────────────────────────────────────
    # "semantically attractive but evidence-poor retrieval obstruction"
    # note: Added check if the output already exists before running. if we want to regenerate, we need to delete the existing file.
    spoof_path = Path("data/processed/spoof_chunks.jsonl")
    if spoof_path.exists():
        logger.info(f"Spoof chunks already exist at {spoof_path}. Skipping attack generation.")
    else:
        run([
            py, "-m", "src.attack.generate_attacks",
            "--queries-path",          "data/processed/val_queries.jsonl",
            "--output-path",           "data/processed/spoof_chunks.jsonl",
            "--candidates-per-style",  str(ATTACK_CANDIDATES_PER_STYLE),
            "--keep-per-style",        str(ATTACK_KEEP_PER_STYLE),
            "--embedding-model",       ATTACK_EMBEDDING_MODEL,
            "--max-queries",           str(ATTACK_MAX_QUERIES),
            "--attack-mode",           ATTACK_MODE,
            "--use-llm",
            "--temperature",           "0.9",
        ])

    # ── 3. Standard retrieval baselines ───────────────────────────────────────
    retrievers_base = [
        ("minilm", "src.retrieval.baseline_retrieval",
         ["--index-dir", "indexes/minilm_base",
          "--model-name", "sentence-transformers/all-MiniLM-L6-v2"]),
        #("bge", "src.retrieval.baseline_retrieval",
         #["--index-dir", "indexes/bge_m3_base", "--model-name", "BAAI/bge-m3"]),
        ("bm25", "src.retrieval.bm25_retrieval",
         ["--index-dir", "indexes/bge_m3_base"]),
        ("hybrid", "src.retrieval.hybrid_retrieval",
         ["--index-dir", "indexes/bge_m3_base", "--model-name", "BAAI/bge-m3",
          "--alpha", "0.4", "--candidate-k", "50"]),
    ]

    for ret, module, extra in retrievers_base:
        logger.info(f"Running baseline evaluation for retriever: {ret}")

        # 3a. Clean Top-5 (standard RAG — no attack)
        run(retrieval_run(py, module,
                          "data/processed/val_queries.jsonl",
                          BASELINE_TOP_K,
                          f"results/retrieval/{ret}_baseline_results.json",
                          extra))
        eval_ret(py,
                 f"results/retrieval/{ret}_baseline_results.json",
                 f"results/retrieval/{ret}_baseline_metrics.json",
                 attacked_only=True)

        # 3b. Clean Top-20 (pool for fair defense comparison)
        run(retrieval_run(py, module,
                          "data/processed/val_queries.jsonl",
                          RETRIEVAL_POOL_TOP_K,
                          f"results/retrieval/{ret}_baseline_top20_results.json",
                          extra))
        eval_ret(py,
                 f"results/retrieval/{ret}_baseline_top20_results.json",
                 f"results/retrieval/{ret}_baseline_top20_metrics.json",
                 attacked_only=True)

        # 3c. Clean Top-20 → naive top-5 (Sasha's dumb comparator on clean data)
        make_topk_view(py,
                       f"results/retrieval/{ret}_baseline_top20_results.json",
                       f"results/retrieval/{ret}_baseline_top20_naive5_results.json",
                       FINAL_KEEP_TOP_K)
        eval_ret(py,
                 f"results/retrieval/{ret}_baseline_top20_naive5_results.json",
                 f"results/retrieval/{ret}_baseline_top20_naive5_metrics.json",
                 attacked_only=True)

        # 3d. Clean Top-20 → defense (does defense help even without attack?)
        defense(py,
                f"results/retrieval/{ret}_baseline_top20_results.json",
                f"results/retrieval/{ret}_baseline_defense_results.json",
                mode="cross_encoder")
        eval_ret(py,
                 f"results/retrieval/{ret}_baseline_defense_results.json",
                 f"results/retrieval/{ret}_baseline_defense_metrics.json",
                 attacked_only=True)

    # ── 3e. OPTIONAL: HyDE baseline (separate condition, does NOT overwrite standard) ──
    if USE_HYDE_BASELINE:
        print("\n[HyDE baseline — optional extra condition]")
        for ret, module, extra in retrievers_base:
            if module != "src.retrieval.baseline_retrieval":
                continue   # HyDE only applies to dense retrievers
            hyde_extra = extra + ["--use-hyde", "--hyde-model", "gpt-4o-mini"]
            run(retrieval_run(py, module,
                              "data/processed/val_queries.jsonl",
                              RETRIEVAL_POOL_TOP_K,
                              f"results/retrieval/{ret}_hyde_baseline_results.json",
                              hyde_extra))
            eval_ret(py,
                     f"results/retrieval/{ret}_hyde_baseline_results.json",
                     f"results/retrieval/{ret}_hyde_baseline_metrics.json",
                     attacked_only=True)

    # ── 4. Inject attack + static query-blind filter ───────────────────────────
    logger.info("Injecting spoof chunks into the clean corpus...")
    run([
        py, "-m", "src.attack.inject_attacks",
        "--real-chunks",  "data/processed/corpus_chunks.jsonl",
        "--spoof-chunks", "data/processed/spoof_chunks.jsonl",
        "--output-path",  "data/processed/augmented_chunks.jsonl",
    ])
    # Static filter: demonstrates spoofs look legitimate in isolation.
    # Expected result: LOW detection rate → proves query-aware defense is needed.
    run([
        py, "-m", "src.experiments.static_filter",
        "--chunks-path", "data/processed/augmented_chunks.jsonl",
        "--output-path", "results/retrieval/static_filter_report.json",
        "--threshold",   str(DEFENSE_THRESHOLD),
    ])

    # ── 5. Attack indexes ──────────────────────────────────────────────────────
    logger.info("Building attacked FAISS/BM25 indexes from augmented chunks.")
    for idx, model in [
        ("indexes/minilm_attack", "sentence-transformers/all-MiniLM-L6-v2"),
        ("indexes/bge_m3_attack", "BAAI/bge-m3"),
    ]:
        run([
            py, "-m", "src.corpus.build_index",
            "--chunks-path", "data/processed/augmented_chunks.jsonl",
            "--index-dir",   idx,
            "--model-name",  model,
        ])

    retrievers_attack = [
        ("minilm", "src.retrieval.baseline_retrieval",
         ["--index-dir", "indexes/minilm_attack",
          "--model-name", "sentence-transformers/all-MiniLM-L6-v2"]),
        ("bm25", "src.retrieval.bm25_retrieval",
         ["--index-dir", "indexes/bge_m3_attack"]),
        ("hybrid", "src.retrieval.hybrid_retrieval",
         ["--index-dir", "indexes/bge_m3_attack", "--model-name", "BAAI/bge-m3",
          "--alpha", "0.4", "--candidate-k", "50"]),
    ]

    # ── 6. Fair comparison matrix (all conditions start from Top-20) ───────────
    #
    #
    # Conditions produced (all on same N=300 attacked queries):
    #   attack_top20           → recall@20 ceiling
    #   attack_naive_top5      → dumb comparator (sees 20, takes top 5 naively)
    #   attack_no_query        → static/no-query filter (pre-filter baseline)
    #   attack_defense         → query-aware Doc2Query + CrossEncoder defense
    #
    for ret, module, extra in retrievers_attack:
        logger.info(f"Running attack evaluation for retriever: {ret}")        # 6a. Attack pool Top-20
        run(retrieval_run(py, module,
                          "data/processed/val_queries.jsonl",
                          RETRIEVAL_POOL_TOP_K,
                          f"results/retrieval/{ret}_attack_results.json",
                          extra))
        eval_ret(py,
                 f"results/retrieval/{ret}_attack_results.json",
                 f"results/retrieval/{ret}_attack_top20_metrics.json",
                 attacked_only=True)

        # 6b. Naive top-5 from Top-20 (dumb comparator)
        make_topk_view(py,
                       f"results/retrieval/{ret}_attack_results.json",
                       f"results/retrieval/{ret}_attack_naive_top5_results.json",
                       FINAL_KEEP_TOP_K)
        eval_ret(py,
                 f"results/retrieval/{ret}_attack_naive_top5_results.json",
                 f"results/retrieval/{ret}_attack_naive_top5_metrics.json",
                 attacked_only=True)

        # 6c. No-query filter — proves query is necessary for defense
        defense(py,
                f"results/retrieval/{ret}_attack_results.json",
                f"results/retrieval/{ret}_no_query_results.json",
                mode="no_query")
        eval_ret(py,
                 f"results/retrieval/{ret}_no_query_results.json",
                 f"results/retrieval/{ret}_no_query_metrics.json",
                 attacked_only=True)

        # 6d. Query-aware defense: Doc2Query answerability + CrossEncoder
        defense(py,
                f"results/retrieval/{ret}_attack_results.json",
                f"results/retrieval/{ret}_defense_results.json",
                mode="cross_encoder")
        eval_ret(py,
                 f"results/retrieval/{ret}_defense_results.json",
                 f"results/retrieval/{ret}_defense_metrics.json",
                 attacked_only=True)

    # ── 7. Threshold sweep ────────────────────────────────────────────────────
    logger.info("Running threshold sweep to find optimal defense parameters.")
    run([
        py, "-m", "src.experiments.threshold_sweep",
        "--input-path",         "results/retrieval/minilm_attack_results.json",
        "--queries-path",       "data/processed/val_queries.jsonl",
        "--qrels-path",         "data/processed/val_qrels.json",
        "--spoof-chunks-path",  "data/processed/spoof_chunks.jsonl",
        "--output-path",        "results/retrieval/minilm_threshold_sweep.json",
        "--use-doc2query",
    ])

    # ── 8. Plots ───────────────────────────────────────────────────────────────
    logger.info("Pipeline execution complete. Generating final figures...")
    run([py, "-m", "src.evaluation.plot_results", "--output-dir", "results/figures"])

    logger.info("All tasks finished successfully. Shutting down pipeline.")
    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print()
    print("Conditions (all on same N=300 attacked queries, all from Top-20 pool):")
    print("  clean_top5              Standard RAG baseline (no attack)")
    print("  clean_top20_naive5      Upper bound — seeing 20 docs naively")
    print("  clean_top20_defense     Upper bound — with defense on clean data")
    print("  attack_top20            Recall@20 ceiling in poisoned index")
    print("  attack_naive_top5       Dumb comparator (Sasha's requirement)")
    print("  attack_no_query         Static/no-query pre-filter baseline")
    print("  attack_defense          Query-aware Doc2Query + CrossEncoder defense")
    if USE_HYDE_BASELINE:
        print("  hyde_baseline           Optional HyDE condition (separate, not mixed in)")
    print()
    print("Results  → results/retrieval/")
    print("Figures  → results/figures/")
    print("=" * 72)


if __name__ == "__main__":
    main()
