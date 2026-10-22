# ADR-0006: ONNX cross-encoder reranker; grade by score, not by LLM

**Status:** Accepted · 2026-10-16 · Revised in round 3: `max_length` and candidate count

## Context
v0.1 used `sentence-transformers` + bge-reranker (pulls PyTorch, +~2 GB) **and** an LLM call to grade relevance: two relevance judgments, one of them costing seconds.

## Decision
- **Reranker:** FlashRank (an ONNX cross-encoder) behind a `Reranker` port, run in a thread pool.
  - **`max_length=512` set explicitly.** The library default is shorter and would truncate our up-to-512-token chunks. Chunker max = 512 to match.
  - Rerank the **top-20** fused candidates. The latency budget is ≈ 250 ms on CPU, *to be measured in M3*. If it's exceeded, drop to top-15 or move to a smaller model.
- **Grading:** the `grade` node compares the top reranker score with τ. Below τ → one rewrite → retrieve again. Still below → abstain. No LLM grader exists anywhere in the design.
- τ is tuned on the public corpus to maximize correct answers plus correct abstentions, and recorded in `eval_runs`.

## Consequences
- One fewer LLM call per turn; grading is deterministic and unit-testable.
- The port allows a hosted reranker if evals justify it.
