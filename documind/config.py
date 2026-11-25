"""Runtime settings. Secrets come from the environment, never from the repo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


@dataclass
class Settings:
    llm_provider: str = "local"
    embedding_provider: str = "local"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"
    chroma_path: Path = Path("data/chroma")
    catalog_path: Path = Path("data/catalog.sqlite")
    chunk_store: Path = Path("data/chunks.json")
    chunk_size: int = 700
    chunk_overlap: int = 80
    top_k: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        return cls(
            llm_provider=_env("LLM_PROVIDER", "local"),
            embedding_provider=_env("EMBEDDING_PROVIDER", "local"),
            openai_api_key=key or None,
            openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=_env("OLLAMA_MODEL", "llama3.1"),
            ollama_embedding_model=_env("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            chroma_path=Path(_env("CHROMA_PATH", "data/chroma")),
            catalog_path=Path(_env("CATALOG_PATH", "data/catalog.sqlite")),
            chunk_store=Path(_env("CHUNK_STORE", "data/chunks.json")),
            chunk_size=int(_env("CHUNK_SIZE", "700")),
            chunk_overlap=int(_env("CHUNK_OVERLAP", "80")),
            top_k=int(_env("TOP_K", "4")),
        )
