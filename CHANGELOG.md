\# Project Development Log



\## Phase 1 — Dataset and Corpus Construction

\- Loaded SQuAD-based QA data

\- Built corpus chunks from source contexts

\- Created validation query set



\## Phase 2 — Baseline Retrieval

\- Implemented MiniLM dense retrieval with FAISS

\- Implemented BM25 sparse retrieval

\- Added hybrid dense+sparse retrieval



\## Phase 3 — Spoofing Attack

\- Generated synthetic spoof chunks

\- Injected spoof chunks into the retrieval corpus

\- Evaluated spoof ranking behavior



\## Phase 4 — Defense Layer

\- Added suspicion-based scoring

\- Added retrieval-stage reranking

\- Added Reverse QA defense logic



\## Phase 5 — Evaluation and Documentation

\- Added Recall@K, spoof win rate, and ranking metrics

\- Generated visual figures

\- Finalized README and presentation materials

