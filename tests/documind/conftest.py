import pytest

from documind.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="local",
        embedding_provider="local",
        openai_api_key=None,
        chroma_path=tmp_path / "chroma",
        catalog_path=tmp_path / "catalog.sqlite",
        chunk_store=tmp_path / "chunks.json",
        chunk_size=400,
        chunk_overlap=40,
        top_k=4,
    )
