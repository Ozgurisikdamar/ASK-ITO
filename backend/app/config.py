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

    # Reranker (mandatory component). CPU default chosen by the M10 benchmark gate
    # (docs/DECISIONS.md ADR-0017); BAAI/bge-reranker-base is higher quality but
    # ~6x slower on CPU.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM abstraction
    # "ollama": local Ollama native API (/api/chat; a trailing /v1 in the URL is ignored)
    # "openai": any OpenAI-compatible endpoint (e.g. Groq free tier)
    # "mock":   deterministic offline stub
    llm_mode: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = "ollama"
    llm_default_model: str = ""
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
    # Optional fallback when the primary provider fails (e.g. Groq daily limit)
    llm_fallback_mode: str = ""  # "" | "ollama" | "openai"
    llm_fallback_base_url: str = "http://localhost:11434"
    llm_fallback_api_key: str = "ollama"
    llm_fallback_model: str = ""

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
    rerank_max_length: int = 256  # tokens per (query, passage) pair
    rerank_batch_size: int = 32
    torch_threads: int = 0  # 0 = all CPU cores
    max_chunks_per_doc: int = 2
    results_max: int = 50  # ranked results kept for the results list / pagination
    retrieval_cache_size: int = 512
    agent_max_rounds: int = 3
    agent_max_llm_calls: int = 10
    agent_timeout_s: int = 180
    agent_answer_temperature: float = 0.0
    agent_max_subquestions: int = 4
    final_k: int = 5

    # Answer generation
    default_mode: str = "overview"  # Google-like: instant results + streamed AI overview
    overview_context_chunks: int = 4
    overview_context_chars: int = 700
    overview_max_tokens: int = 320
    answer_context_chars: int = 900
    answer_max_tokens: int = 600
    answer_cache_enabled: bool = True
    prompt_version: str = "v2"  # part of the answer-cache key

    # Startup
    warmup_on_start: bool = True
    prewarm_featured: bool = True  # pre-compute answers for the featured queries
    sse_keepalive_s: float = 10.0

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
