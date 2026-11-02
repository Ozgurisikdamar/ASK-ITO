# ASK-ITO Fast RAG Simplification — Claude Implementation Brief

**Status:** Implementation plan  
**Scope:** Simplify ASK-ITO into a fast, reliable Hybrid RAG application while preserving the existing Agentic RAG as an optional advanced mode.  
**Hard constraint:** Local inference only. No Groq, OpenAI, Anthropic, Gemini, hosted LLM APIs, or cloud LLM fallback.

## 1. Goal

The default user experience should answer ordinary corpus questions (e.g. `quantum computing`, `popular AI fields`) quickly using:

**BM25 + pgvector dense retrieval → RRF fusion → optional lightweight reranker → one local LLM answer-generation call → citations**

Agentic RAG is not required for routine questions. Keep it available as an opt-in advanced mode; do not delete it or silently change its semantics.

This task is not a broad rewrite. Reuse the existing FastAPI, PostgreSQL/pgvector, BM25, retrieval, citation, streaming, and cache components wherever sound.

## 2. Model decision

Use **Qwen3.5-9B**, not “Qwen3.5-8B”. The official Ollama model tag is:

```bash
ollama pull qwen3.5:9b
```

Ollama currently lists Qwen3.5 sizes including 4B and 9B; there is no official Qwen3.5-8B tag in the model library. Model source: https://ollama.com/library/qwen3.5

Configure the default local answer model as `qwen3.5:9b` (or the exact model identifier required by the current Ollama integration). Do not hardcode the tag in multiple places; use the existing configuration mechanism.

Important:
- Qwen3.5-9B is the **answer generator**, not the retrieval/reranking model.
- Use a single LLM call for ordinary queries.
- Disable thinking/reasoning mode for normal factual RAG answers if the Ollama model/runtime supports a reliable documented setting. Verify compatibility instead of guessing parameters.
- Keep output concise by default (e.g. 250–450 tokens maximum, configurable).
- Use temperature 0–0.2 for repeatable factual answers.
- Do not run Qwen3.5-9B on every retrieval step; only invoke it after context selection.
- Do not claim latency targets are achieved without measurements.

## 3. Default query pipeline

Implement or select a simple, explicit default mode (e.g. `fast_rag` or reuse the existing `hybrid_rerank` mode if it can meet these requirements without breaking compatibility):

1. Validate and normalize query; preserve original query for display/citation.
2. Create query embedding once.
3. Run BM25 and pgvector retrieval in parallel if current infrastructure safely supports concurrency.
4. Retrieve a configurable candidate pool (initial default: 10–20 from each channel; tune from actual corpus/quality evidence).
5. Fuse result lists using existing RRF implementation. Do not create a new fusion implementation if existing one is correct.
6. Optional reranking:
   - Use current BGE reranker or a MiniLM cross-encoder candidate only if supported and benchmarked.
   - MiniLM must be a true cross-encoder reranker (not merely a sentence embedding model).
   - Rerank only a bounded pool, not the whole corpus.
   - Choose top 3–5 context chunks (configurable) for generation.
   - Keep reranker off as a selectable fallback if it increases latency or hurts relevance.
7. Deduplicate chunks/documents and enforce context budget.
8. Make exactly one local LLM call to generate the answer from retrieved context.
9. Return answer, citations/source metadata, and stage-level timings.
10. If retrieval is insufficient, return a clear “not found / insufficient evidence in this corpus” response rather than inventing an answer.

Do not call the LLM for query analysis, sub-question generation, sufficiency checks, or query rewrite in this default fast path.

## 4. Search mode vs Answer mode

Expose two clear behaviors if the current API/UI supports it without disproportionate work:

- **Search mode:** return ranked chunks/documents, snippets, source titles, and retrieval scores. No LLM generation.
- **Answer mode (default):** use the same retrieved context and generate a concise cited answer with one LLM call.

For simple topic queries such as `quantum computing`, both modes should work:
- Search: retrieve relevant Wikipedia passages.
- Answer: summarize retrieved evidence and cite it.

For broad queries such as `popular AI fields`, the retrieval system should retrieve passages that mention several AI areas and the LLM may synthesize a list **only when those items are supported by retrieved evidence**. Do not rely on the model’s parametric knowledge as evidence.

If the corpus cannot support a multi-topic answer, be transparent and suggest narrower searches or show available sources.

## 5. Citation and grounding requirements

- Preserve existing citation validation and source IDs.
- Every factual answer claim should be supported by one or more retrieved source references.
- Citations must map to real retrieved chunks/documents, not fabricated IDs or titles.
- If no relevant evidence is retrieved, abstain or return search results without an unsupported generated answer.
- Do not remove citations to make generation easier.
- Return source title, stable document/chunk identifier, and URL/path where available.
- Avoid presenting raw similarity scores as calibrated confidence.

## 6. Agentic RAG remains optional

- Keep the existing Agentic RAG implementation and API compatibility.
- Make it opt-in (advanced mode) rather than default for normal queries.
- Do not run agentic loops automatically after a fast-RAG answer unless the user explicitly selects advanced mode or the existing API contract has a clearly documented opt-in.
- Preserve tests and behavior for agentic mode.
- Do not remove multi-step capabilities; separate them from the latency-sensitive default path.

## 7. Reranker strategy

The user is interested in **MiniLM reranker + Qwen3.5-9B**, but this is a candidate configuration, not a proven winner.

- Inspect current reranker and supported checkpoints first.
- Identify an actual cross-encoder checkpoint, such as a suitable MiniLM cross-encoder from Sentence Transformers; do not accidentally use an embedding-only MiniLM model as a reranker.
- Measure CPU cold-start, warm inference, batch size, candidate count, p50/p95, and memory.
- Compare:
  A. No reranker
  B. Existing BGE reranker
  C. MiniLM cross-encoder
- Use identical retrieval candidates and query set when comparing.
- Compare Recall@5/10, MRR@10, nDCG@10, citation coverage, and latency.
- If MiniLM quality regresses materially, keep the existing reranker or no-reranker option; do not force MiniLM merely for speed.
- Avoid expensive per-query model initialization. Keep safe singleton/lazy initialization and warm-up if already implemented.

## 8. Offline evaluation only

RAGAS and benchmarks must never run in the user’s query path.

- RAGAS is an offline developer/evaluation task.
- Keep evaluation commands separate from serving API.
- No RAGAS judge call during normal request processing.
- Do not require 10–14 hour exhaustive model benchmarking for this simplification task.
- Start with a small smoke evaluation (5–10 representative questions), then a bounded evaluation using the existing dataset.
- Use the same fixed local judge only if RAGAS is run. No cloud fallback.
- If judge is unavailable or too expensive in time, report NOT RUN; do not fabricate metrics.

Suggested test queries:
- `quantum computing` (specific topic)
- `popular AI fields` (broad topic)
- `What is machine learning?` (question)
- A query with exact phrase/entity from corpus
- A query unsupported by corpus (verify abstention)
- A query with relevant information split across multiple chunks

## 9. Latency instrumentation

Add/retain structured timings for each stage:
- embedding
- BM25
- pgvector
- fusion
- reranker (model load vs warm inference, if applicable)
- prompt/context assembly
- LLM time-to-first-token (TTFT)
- LLM total generation time
- total request duration

Return/log timing information safely (do not expose secrets). Keep benchmark output reproducible.

Initial **targets, not claims**:
- retrieval + fusion: ideally sub-second on local corpus
- reranker: bounded and measured; if CPU is slow, allow disabling
- LLM generation: depends on CPU/GPU/runtime; optimize TTFT and response length
- overall: aim for a few seconds on suitable hardware, but report actual values and hardware

Do not promise a fixed total latency on CPU without measurement. Distinguish TTFT from full answer completion.

## 10. UI / dashboard

Preserve the existing project dashboard and improve only where relevant:
- mode selector: Search / Fast Answer / Agentic (if current UI supports it)
- source cards with title and snippet
- answer with citations
- stage latency breakdown
- model and reranker names
- clear loading state and streaming output if supported
- error / no-evidence / model-not-loaded states

Do not make a large visual redesign. The dashboard should help demonstrate BM25 + vector + reranker + RAGAS architecture, while clearly indicating that RAGAS is offline.

## 11. Tests and acceptance criteria

Do not advance to final report until these are addressed.

### Functional
- [ ] Fast mode makes zero LLM calls before final answer generation.
- [ ] Fast mode makes exactly one LLM call for answer generation (or zero in Search-only mode).
- [ ] BM25 and pgvector are both included in hybrid retrieval.
- [ ] RRF combines both channels correctly.
- [ ] Search mode returns ranked sources without LLM generation.
- [ ] Answer mode returns citations grounded in retrieved chunks.
- [ ] Unsupported question yields a transparent abstention/no-evidence response.
- [ ] Broad question does not produce unsupported claims.
- [ ] Agentic mode remains functional and opt-in.
- [ ] No hosted/cloud API or fallback exists.

### Performance / quality
- [ ] Stage timings are recorded.
- [ ] Reranker on/off comparison is completed on a small representative set.
- [ ] Retrieval metrics are measured on available labeled/evaluation data.
- [ ] Answer/citation quality is manually checked for representative questions.
- [ ] Latency and quality are compared against the pre-change baseline under equivalent conditions.
- [ ] Clearly identify hardware and model quantization/runtime.
- [ ] No claim that a target was met without evidence.

### Engineering
- [ ] Unit tests for query routing and LLM-call count.
- [ ] Integration tests for retrieval, citations, and abstention.
- [ ] Existing tests, lint, and type checks run where available.
- [ ] Config and documentation updated.
- [ ] No secrets, model binaries, or corpus dumps committed.
- [ ] Existing M10 work and useful caching/streaming/native Ollama integration are preserved.
- [ ] Changes are split into logical commits and pushed to a dedicated branch.
- [ ] Do not merge to main without explicit approval.

## 12. Deliverables

Create:
- `docs/ASK_ITO_FAST_RAG_IMPLEMENTATION_REPORT.md`

Report:
1. What was changed and why.
2. Architecture before/after.
3. Exact model/runtime settings.
4. How to run locally.
5. Commands and test results (actual, not inferred).
6. Query-level latency and per-stage timings.
7. Retrieval quality/citation results.
8. Reranker comparison (or clearly state not run).
9. Known limitations and follow-up recommendations.
10. Branch name and commit SHAs.

If local hardware benchmarks cannot be run in the environment, clearly mark them **NOT RUN**, provide exact commands for the user to run, and do not invent measurements.

## 13. Guardrails

- Local-only inference: Ollama localhost/loopback. No Groq, OpenAI, Anthropic, Gemini, cloud inference or remote API.
- Do not change the embedding model or vector dimension without a migration/reindex plan.
- Do not switch pgvector to Qdrant for this task.
- Do not remove the current agentic implementation.
- Do not run large, overnight benchmark matrices before delivering the fast path.
- Do not change unrelated application behavior.
- Do not merge to main without approval.
- Preserve data and migrations; no destructive database operations.
- Avoid speculative architecture changes. Inspect the repository first and adapt the plan to its real structure.

## 14. Suggested execution order

1. Inspect current branch, M10 changes, main, API/UI, tests, and existing modes.
2. Identify the smallest implementation that reuses the existing hybrid retrieval.
3. Implement Fast RAG mode and single-call generation.
4. Add Search-only behavior if practical.
5. Verify citation grounding and abstention.
6. Compare reranker options on a small sample.
7. Run tests and short smoke evaluation.
8. Add latency instrumentation.
9. Update docs and create implementation report.
10. Commit, push branch, and provide branch/commit/report URL.

**Definition of done:** A user can enter a simple topic or broad question and receive ranked corpus evidence plus a concise grounded answer without triggering the expensive agentic loop or offline RAGAS evaluation.
