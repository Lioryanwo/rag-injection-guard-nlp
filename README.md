# RAG Injection Guard

<h3 align="center">Retrieval Poisoning Attacks and Robustness Evaluation for Retrieval-Augmented Generation Systems</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-green" />
  <img src="https://img.shields.io/badge/BM25-Sparse_Retrieval-orange" />
  <img src="https://img.shields.io/badge/MiniLM-Dense_Retrieval-red" />
  <img src="https://img.shields.io/badge/Dataset-SQuAD_v1.1-yellow" />
  <img src="https://img.shields.io/badge/Project-NLP%20Security-success" />
</p>

<p align="center">
  <b>Can a RAG system be misled before the LLM ever generates an answer?</b><br/>
  This project studies retrieval poisoning attacks against RAG systems: synthetic chunks that look relevant to the query but do not contain the real supporting evidence.
</p>

---

## Executive Summary

Retrieval-Augmented Generation (RAG) systems rely on the retrieved context. If the retriever returns misleading chunks, the LLM can receive bad evidence and produce an unsupported answer even if the model itself was not attacked.

This project evaluates a retrieval-stage attack called **RAG spoofing**. The attacker injects synthetic chunks into the corpus. These chunks are optimized to be semantically or lexically attractive to the retriever while omitting the true answer evidence.

> **Main finding:** spoof chunks strongly dominate Top-1 retrieval across dense, sparse, and hybrid retrievers. A Cross-Encoder defense improves Recall@5 in some cases, but it does not fully suppress spoof dominance.

---

## What Is RAG Spoofing?

A spoof chunk is not necessarily a prompt injection. It does not need to say “ignore previous instructions.” Instead, it looks like a normal reference passage while being evidence-poor or misleading.

```text
User Query
    ↓
Retriever
    ↓
Top-K Chunks, including spoof chunks
    ↓
LLM receives poisoned context
    ↓
Answer may become misleading or ungrounded
```

---

## Threat Model

| Attacker Can | Attacker Cannot |
|---|---|
| Insert synthetic chunks into the retrieval corpus | Modify the LLM |
| Target likely user questions | Modify retriever weights |
| Optimize chunks for semantic similarity | Change FAISS/BM25 internals |
| Compete with legitimate documents during Top-K retrieval | Access ground-truth labels at inference time |

The attack is therefore a **corpus-level retrieval poisoning attack**.

---

## System Pipeline

The project evaluates the full retrieval path:

1. Build a clean corpus from SQuAD.
2. Split documents into chunks.
3. Encode chunks into dense/sparse retrieval representations.
4. Run MiniLM, BM25, and Hybrid retrieval.
5. Generate and inject spoof chunks.
6. Retrieve a Top-20 candidate pool.
7. Apply no-query filtering or Cross-Encoder defense.
8. Evaluate recall and spoof dominance.

---

## Attack Design

The attack objective is not to generate a correct answer. The objective is to **outrank the real evidence**.

| Constraint | Meaning |
|---|---|
| Semantic attraction | The chunk is close to the target query in retrieval space. |
| Evidence-poor content | The chunk omits the gold supporting evidence. |
| Legitimate appearance | The chunk reads like a normal reference passage. |

### Attack Families Implemented

| Family | Purpose |
|---|---|
| `evidence_free` | Topic-relevant background text without answer evidence. |
| `hypothetical_distractor` | Near-answer passages with related but off-target details. |
| `hyde_attack` | HyDE-style answer-like passages that omit or replace the answer-bearing detail. |

The final pipeline uses:

```text
attack_mode = hypothetical_distractor
max_queries = 300
candidates_per_style = 3
keep_per_style = 1
```

With two default styles under `hypothetical_distractor`, this produces:

```text
300 attacked queries × 2 styles × 1 selected candidate = 600 spoof chunks
```

---

## Defense Design

The defense is retrieval-stage only. It does not modify the LLM.

Instead of directly returning the first Top-5 retrieved chunks, the system retrieves a larger Top-20 pool and then applies filtering/reranking:

```text
Retrieve Top-20
      ↓
Suspicion / static filtering baseline
      ↓
Cross-Encoder reranking
      ↓
Doc2Query / Reverse-QA style alignment signal
      ↓
Return final Top-5
```

The defense compares three conditions:

| Condition | Meaning |
|---|---|
| Attacked / No Defense | Take the attacked retrieval output directly. |
| No-query Filter | Static chunk-only filtering without seeing the user query. |
| CE Defense | Query-aware Cross-Encoder and alignment-based reranking. |

---

## Experimental Setup

### Dataset and Corpus

| Component | Value |
|---|---:|
| Dataset | SQuAD v1.1 |
| Train examples | 5,000 |
| Validation queries | 1,000 |
| Attacked queries | 300 |
| Generated spoof chunks | 600 |

### Retrieval Systems

| Retriever | Type | Notes |
|---|---|---|
| MiniLM + FAISS | Dense retrieval | `sentence-transformers/all-MiniLM-L6-v2` |
| BM25 | Sparse retrieval | Lexical scoring baseline |
| Hybrid | Dense + sparse fusion | Combines semantic and lexical retrieval |

### Main Metrics

| Metric | Meaning |
|---|---|
| `Recall@5` | Whether the correct evidence appears in the final Top-5. Higher is better. |
| `Recall@20` | Whether the correct evidence appears in the Top-20 candidate pool. Higher is better. |
| `Top-1 Spoof Win Rate` | Fraction of attacked queries where a spoof chunk ranks first. Lower is better. |

---

## Results

### Recall@5 Under Attack

<p align="center">
  <img src="./assets/figures/recall_under_attack.webp" width="1000" alt="Recall@5 under attack" />
</p>

| Retriever | Clean Top-5 | Attacked / No Defense | No-query Filter | CE Defense |
|---|---:|---:|---:|---:|
| MiniLM | 0.83 | 0.23 | 0.21 | 0.31 |
| BM25 | 0.84 | 0.35 | 0.35 | 0.37 |
| Hybrid | 0.89 | 0.37 | 0.37 | 0.41 |

**Interpretation:** retrieval poisoning causes a large Recall@5 drop. MiniLM is especially affected, falling from `0.83` to `0.23`. The CE defense improves recall, especially for MiniLM and Hybrid, but remains far below the clean baseline.

---

### Recall Comparison: Top-20 vs Final Defense

<p align="center">
  <img src="./assets/figures/recall_top20_vs_defense_top5.webp" width="1000" alt="Recall Top-20 versus defense Top-5" />
</p>

| Retriever | Clean Baseline Top-20 | Attack Top-20 Ceiling | CE Defense Top-5/20 |
|---|---:|---:|---:|
| MiniLM | 0.98 | 0.67 | 0.31 |
| BM25 | 0.93 | 0.67 | 0.37 |
| Hybrid | 1.00 | 0.77 | 0.41 |

**Interpretation:** even when looking at the Top-20 pool, the attack pushes real evidence out for many queries. This matters because reranking cannot recover evidence that is not present in the candidate pool.

---

### Top-1 Spoof Win Rate

<p align="center">
  <img src="./assets/figures/spoof_win_rate_attacked_queries.webp" width="1000" alt="Top-1 spoof win rate on attacked queries" />
</p>

| Retriever | Attacked / No Defense | No-query Filter | CE Defense |
|---|---:|---:|---:|
| MiniLM | 0.96 | 0.96 | 0.96 |
| BM25 | 0.95 | 0.95 | 0.94 |
| Hybrid | 0.96 | 0.97 | 0.93 |

**Interpretation:** spoof chunks dominate the first rank across all retrieval methods. CE defense slightly reduces spoof dominance for BM25 and Hybrid, but the Top-1 spoof rate remains very high.

---

## Key Findings

### 1. Retrieval poisoning succeeds before generation

The attack succeeds at the retrieval stage, before the LLM sees the context. This means LLM-side safety alone is not enough.

### 2. Dense retrieval is highly vulnerable

MiniLM suffers the sharpest Recall@5 drop, from `0.83` clean to `0.23` under attack.

### 3. Sparse and hybrid retrieval are not immune

BM25 and Hybrid preserve more recall than MiniLM under attack, but their Top-1 spoof win rates remain above `0.90`.

### 4. Reranking helps, but has a hard limit

CE defense improves Recall@5, but it cannot fully recover clean performance. If the correct evidence is missing from the Top-20 pool, reranking cannot bring it back.

### 5. No-query filtering is weak

The no-query filter does not reliably distinguish spoof chunks from legitimate chunks because the spoof text is designed to look normal in isolation.

---

## Repository Structure

```text
src/
├── attack/        # spoof generation, injection, attack evaluation
├── corpus/        # SQuAD corpus creation, chunking, indexing
├── retrieval/     # MiniLM/FAISS, BM25, Hybrid retrieval
├── defense/       # suspicion scoring, filtering, reranking, Reverse QA
├── evaluation/    # metrics, LLM judge, plotting
├── experiments/   # threshold sweeps and additional defense experiments
└── pipeline/      # end-to-end experiment runner
```

---

## Reproducing the Experiments

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
echo "OPENAI_API_KEY=YOUR_KEY" > .env
```

### 3. Run the full pipeline

```bash
python -m src.pipeline.run_pipeline
```

### 4. Generate spoof chunks only

```bash
python -m src.attack.generate_attacks \
  --attack-mode hypothetical_distractor \
  --max-queries 300 \
  --candidates-per-style 3 \
  --keep-per-style 1 \
  --use-llm \
  --temperature 0.9
```

### 5. Evaluate retrieval results

```bash
python -m src.evaluation.evaluate_retrieval \
  --results-path results/retrieval/minilm_attack_results.json \
  --output-path results/retrieval/minilm_attack_top20_metrics.json \
  --spoof-chunks-path data/processed/spoof_chunks.jsonl
```

---

## Project Scope

This is an academic NLP/security project focused on evaluating retrieval robustness in RAG systems. The goal is not to claim a complete defense, but to show that retrieval poisoning is a serious and measurable weakness in current RAG pipelines.

---

## References

- Rajpurkar et al. (2016). **SQuAD: 100,000+ Questions for Machine Comprehension of Text.** EMNLP.
- Lewis et al. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** NeurIPS.
- Reimers & Gurevych (2019). **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.** EMNLP.
- Nogueira & Cho (2019). **Passage Re-ranking with BERT.** arXiv.
- Gao et al. (2023). **Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE).** ACL.

---

## Team

| Name | Contribution |
|---|---|
| Lior Yanwo | Retrieval pipeline, attack design, evaluation |
| Nadav Yithaki | Defense design, experiments, LLM-based analysis |

---

<p align="center">
  <b>RAG systems are only as trustworthy as the evidence they retrieve.</b><br/>
  Protecting retrieval is the first step toward trustworthy generation.
</p>
