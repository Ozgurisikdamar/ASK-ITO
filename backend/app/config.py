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
    wiki_user_agent: str = "AskITO/0.1 (contact: you@example.com)"

    # Retrieval / agent
    agent_max_rounds: int = 3
    final_k: int = 5

    # CORS
    frontend_origin: str = "http://localhost:5173"

    def model_for(self, role: str) -> str:
        """Resolve the model for a role, falling back to the default model."""
        role_value = getattr(self, f"llm_{role}_model", "") or ""
        return role_value or self.llm_default_model


settings = Settings()
