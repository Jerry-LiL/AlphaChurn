"""Embedding providers stored in Chroma.

The local provider is a hashed term vector. It keeps retrieval runnable with
no model download. OpenAI and Ollama use LangChain embedding clients and run
when credentials or a local Ollama server are available.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

from documind.config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")


@register_embedding_function
class DocumindEmbedding(EmbeddingFunction[Documents]):
    def __init__(
        self,
        provider: str = "local",
        model: str = "",
        base_url: str = "",
        dim: int = 256,
        api_key: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.dim = dim
        self.api_key = api_key

    def __call__(self, input: Documents) -> Embeddings:
        if self.provider == "local":
            return [self._hashed(text) for text in input]
        if self.provider == "openai":
            return self._openai(list(input))
        if self.provider == "ollama":
            return self._ollama(list(input))
        raise RuntimeError(f"Unknown embedding provider: {self.provider}")

    @staticmethod
    def name() -> str:
        return "documind"

    def default_space(self) -> str:
        return "cosine"

    def get_config(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "dim": self.dim,
        }

    @staticmethod
    def build_from_config(config: dict) -> DocumindEmbedding:
        return DocumindEmbedding(
            provider=config.get("provider", "local"),
            model=config.get("model", ""),
            base_url=config.get("base_url", ""),
            dim=int(config.get("dim", 256)),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def _hashed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vector[bucket] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def _openai(self, texts: list[str]) -> list[np.ndarray]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings

        embedder = OpenAIEmbeddings(model=self.model, api_key=self.api_key)
        return [np.asarray(row, dtype=np.float32) for row in embedder.embed_documents(texts)]

    def _ollama(self, texts: list[str]) -> list[np.ndarray]:
        from langchain_ollama import OllamaEmbeddings

        embedder = OllamaEmbeddings(model=self.model, base_url=self.base_url or None)
        return [np.asarray(row, dtype=np.float32) for row in embedder.embed_documents(texts)]


def build_embedding(settings: Settings) -> DocumindEmbedding:
    if settings.embedding_provider == "openai" and not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
    model = {
        "local": "",
        "openai": settings.openai_embedding_model,
        "ollama": settings.ollama_embedding_model,
    }.get(settings.embedding_provider)
    if model is None:
        raise RuntimeError(f"Unknown embedding provider: {settings.embedding_provider}")
    return DocumindEmbedding(
        provider=settings.embedding_provider,
        model=model,
        base_url=settings.ollama_base_url,
        api_key=settings.openai_api_key,
    )
