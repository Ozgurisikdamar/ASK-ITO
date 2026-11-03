"""Application configuration, read from environment / .env.

No model names, k values or thresholds are hardcoded elsewhere in the code;
they are all read from here.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://askito:askito@localhost:5432/askito"
    db_pool_min: int = 1
    db_pool_max: int = 8
    db_connect_timeout_s: int = 5

    # Embedding
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # Reranker (cross-encoder). Default chosen by the fast-RAG comparison
    # (docs/DECISIONS.md ADR-0022): bge-reranker-base with RERANK_POOL=20 and
    # RERANK_MAX_LENGTH=512. cross-encoder/ms-marco-MiniLM-L-6-v2 is faster but
    # gained little over no reranker on the eval set.
    reranker_model: str = "BAAI/bge-reranker-base"

    # LLM abstraction
    # "ollama": local Ollama native API (/api/chat; a trailing /v1 in the URL is ignored)
    # "openai": any OpenAI-compatible endpoint (e.g. Groq free tier)
    # "mock":   deterministic offline stub
    llm_mode: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = "ollama"
    # The one place the model tag is set (env LLM_DEFAULT_MODEL overrides it).
    # Weights: Q8_0 (qwen3.5:9b-q8_0, manifest verified: file_type Q8_0).
    # Baseline / rollback: qwen3.5:9b-q4_K_M (same manifest as qwen3.5:9b).
    # No lower weight quantization is recommended. KV-cache precision is a
    # separate Ollama setting (OLLAMA_KV_CACHE_TYPE), not part of the tag.
    llm_default_model: str = "qwen3.5:9b-q8_0"
    llm_agent_model: str = ""
    llm_answer_model: str = ""
    llm_judge_model: str = ""
    llm_think: bool = False  # reasoning/"thinking" output; off = much faster
    llm_keep_alive: str = "24h"  # keep the Ollama model loaded in RAM
    llm_num_ctx: int = 4096
    llm_num_thread: int = 0  # 0 = Ollama default
    llm_timeout_s: float = 240.0
    llm_connect_timeout_s: float = 5.0
    llm_ping_timeout_s: float = 2.0
    llm_max_retries: int = 2  # on 429 / 5xx
    llm_extra_body: str = ""  # JSON merged into OpenAI-compatible request bodies
    llm_local_only: bool = True  # loopback-only LLM/judge endpoints (app.llm.policy)
    # Container hostnames trusted to resolve to private-network addresses when
    # LLM_LOCAL_ONLY=true (comma-separated, exact names; empty = loopback only).
    # docker-compose sets "ollama".
    llm_allowed_hosts: str = ""
    llm_judge_max_tokens: int = 1024  # RAGAS judge generation limit
    ragas_timeout_s: int = 1800  # per RAGAS job (single worker, local CPU)

    # Ingest
    # Wikimedia requires a descriptive User-Agent with a contact (URL or email);
    # a placeholder contact is rejected with HTTP 403.
    wiki_user_agent: str = "AskITO/0.1 (+https://github.com/Ozgurisikdamar/ASK-ITO)"
    wiki_delay: float = 1.0  # polite inter-request delay (seconds)
    # HTTP backend for the MediaWiki client: "auto" uses httpx and falls back to
    # curl if the egress is fingerprint-blocked (HTTP 403); "httpx"/"curl" force one.
    wiki_http_backend: str = "auto"
    ingest_min_chars: int = 1500  # drop expansion pages shorter than this
    ingest_expand_min_links: int = 2  # candidate must be linked by >= N seeds
    ingest_expand_max_nobel: int = 60
    ingest_expand_max_quantum: int = 30
    ingest_expand_max_ai: int = 40

    # Chunking (M3)
    chunk_target_tokens: int = 400
    chunk_overlap_tokens: int = 60  # ~15% of target
    # bge query instruction prefix (added to queries only, not passages)
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # BM25 (bm25s) parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Retrieval / agent
    bm25_top_k: int = 50
    bm25_pool_k: int = 200  # wider BM25 pool when a domain filter is applied
    vector_top_k: int = 50
    hnsw_ef_search: int = 100  # pgvector HNSW candidate list (default 40 caps top-k)
    rrf_k: int = 60
    rerank_pool: int = 20
    rerank_max_length: int = 512  # tokens per (query, passage) pair (bge-reranker-base maximum)
    rerank_batch_size: int = 32
    torch_threads: int = 0  # 0 = all CPU cores
    max_chunks_per_doc: int = 2
    results_max: int = 50  # ranked results kept for the results list / pagination
    retrieval_cache_size: int = 512  # in-process L1 (Redis, when enabled, is the L2)
    embedding_cache_size: int = 1024  # in-process query-embedding LRU (0 = off)

    # Optional local Redis: shared L2 for the embedding, retrieval, answer and agent
    # tool caches (app.redis_cache). Off by default; local only (loopback, or a
    # hostname in REDIS_ALLOWED_HOSTS — docker-compose sets "redis"). Unreachable
    # Redis = warning + fallback to the in-process / PostgreSQL caches, unless
    # REDIS_STRICT=true. Never holds model KV tensors (that is Ollama's job).
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_allowed_hosts: str = ""
    redis_strict: bool = False
    redis_timeout_s: float = 0.25
    redis_retry_s: float = 30.0  # circuit-breaker pause after a failure
    redis_max_value_bytes: int = 1_000_000
    redis_ttl_embedding_s: int = 7 * 24 * 3600
    redis_ttl_retrieval_s: int = 24 * 3600
    redis_ttl_answer_s: int = 24 * 3600
    redis_ttl_tool_s: int = 3600  # agent rewrite/sufficiency results
    # Redis front cache for the PostgreSQL answer cache (PG stays the durable store)
    redis_answer_front_cache: bool = True
    # Cache deterministic agent LLM tool calls (rewrite, sufficiency) by complete
    # model + prompt + input key; only at temperature 0 / think off.
    agent_tool_cache: bool = True
    agent_max_rounds: int = 3
    agent_max_llm_calls: int = 10
    agent_timeout_s: int = 180
    agent_answer_temperature: float = 0.0
    agent_max_subquestions: int = 4
    # Adaptive agent (each switch is off by default = original behaviour; see ADR)
    # rewrite policy: always | skip_simple (no rewrite for simple questions) |
    #   skip_subquestions (no rewrite of LLM-written sub-questions) | never
    agent_rewrite_policy: str = "always"
    # skip the LLM sufficiency check when every sub-question already has a passage
    # with cross-encoder score >= agent_coverage_min_score; simple questions get at
    # most agent_max_extra_rounds_simple extra rounds
    agent_adaptive_sufficiency: bool = False
    agent_coverage_min_score: float = 0.5
    agent_max_extra_rounds_simple: int = 1
    # final rerank over the top-N union candidates (0 = whole union, original);
    # (query, passage) pairs already scored in this request are never re-scored
    agent_final_rerank_pool: int = 0
    agent_reuse_rerank_scores: bool = False
    final_k: int = 5

    # Fast path: "search" (default) = hybrid retrieval, ranked sources, no LLM call;
    # "overview" = Answer: retrieval + rerank + exactly one LLM call (opt-in in the UI).
    default_mode: str = "search"
    # Cross-encoder stage per fast mode. Search defaults to off so results come back
    # in ~0.1 s on a CPU; Answer keeps the bge reranker (context quality + evidence
    # cut-off). FAST_RERANK, if set, overrides both (backward compatibility).
    search_rerank: bool = False
    answer_rerank: bool = True
    fast_rerank: bool | None = None
    fast_top_k: int = 0  # candidates per channel in the fast modes (0 = BM25/VECTOR_TOP_K)
    # Minimum cross-encoder score for a passage to be used as evidence in the fast
    # modes (reranker on only; 0 = off). No passage above it -> "not found" answer
    # without calling the LLM. In the reranker's own score units: 0.3 is calibrated
    # for bge-reranker-base (probabilities); MiniLM outputs logits -> re-calibrate.
    fast_min_evidence_score: float = 0.3
    # The reranker the cut-off was calibrated for. With any other RERANKER_MODEL the
    # cut-off is disabled (scores are not comparable: MiniLM returns logits).
    evidence_score_model: str = "BAAI/bge-reranker-base"
    overview_context_chunks: int = 4
    overview_context_chars: int = 700
    overview_max_tokens: int = 320
    answer_context_chars: int = 900
    answer_max_tokens: int = 600
    answer_cache_enabled: bool = True
    prompt_version: str = "v3"  # part of the answer-cache key (bump on prompt changes)

    # Startup
    warmup_on_start: bool = True
    prewarm_featured: bool = True  # pre-compute answers for the featured queries
    sse_keepalive_s: float = 10.0
    # Requests allowed to run an LLM at the same time (search needs none). A local
    # model serves one request at a time; extra answer requests get "busy".
    max_concurrent_generations: int = 1

    # Test set generation (M7)
    testset_target: int = 180  # total draft questions to aim for
    llm_cache_enabled: bool = True  # disk cache for eval LLM calls

    # CORS
    frontend_origin: str = "http://localhost:5173"

    def model_for(self, role: str) -> str:
        """Resolve the model for a role, falling back to the default model."""
        role_value = getattr(self, f"llm_{role}_model", "") or ""
        return role_value or self.llm_default_model


settings = Settings()
