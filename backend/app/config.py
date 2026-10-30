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

    # Embedding
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # Reranker (mandatory component)
    reranker_model: str = "BAAI/bge-reranker-base"

    # LLM abstraction
    llm_mode: str = "ollama"  # "ollama" (OpenAI-compatible) | "mock"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_default_model: str = ""
    llm_agent_model: str = ""
    llm_answer_model: str = ""
    llm_judge_model: str = ""

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
    vector_top_k: int = 50
    rrf_k: int = 60
    rerank_pool: int = 30
    max_chunks_per_doc: int = 2
    agent_max_rounds: int = 3
    agent_max_llm_calls: int = 10
    agent_timeout_s: int = 180
    agent_answer_temperature: float = 0.0
    agent_max_subquestions: int = 4
    final_k: int = 5

    # CORS
    frontend_origin: str = "http://localhost:5173"

    def model_for(self, role: str) -> str:
        """Resolve the model for a role, falling back to the default model."""
        role_value = getattr(self, f"llm_{role}_model", "") or ""
        return role_value or self.llm_default_model


settings = Settings()
