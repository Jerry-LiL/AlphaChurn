import json
from pathlib import Path

import pytest

from documind.embeddings import DocumindEmbedding, build_embedding
from documind.graph import run_question
from documind.ingest import ingest_paths
from documind.llm import LLMClient


class ScriptedLLM:
    provider = "openai"

    def complete(self, system: str, user: str) -> str:
        return json.dumps(
            {
                "answer": "Unlimited refunds forever [refunds:1].",
                "citations": [
                    {"doc": "refunds", "page": 1, "quote": "Unlimited refunds forever"}
                ],
            }
        )


def _index_samples(settings, root: Path) -> None:
    policy = root / "policies"
    policy.mkdir(parents=True)
    (policy / "refunds.md").write_text(
        "Plus plan customers have a refund window of 14 days from the charge date.\n"
        "Basic plan refunds are limited to 7 days and exclude usage fees.\n",
        encoding="utf-8",
    )
    (policy / "retention.md").write_text(
        "The save desk may offer one month of account credit.\n",
        encoding="utf-8",
    )
    handbook = root / "handbook"
    handbook.mkdir()
    (handbook / "onboarding.md").write_text(
        "Laptops are issued on day one by workplace IT.\n",
        encoding="utf-8",
    )
    ingest_paths([root], settings)


def test_local_graph_cites_the_refund_page(settings, tmp_path):
    _index_samples(settings, tmp_path / "docs")
    body = run_question("What is the refund window for the plus plan?", settings)
    assert body.route == "retrieve"
    assert body.grounded
    assert "[refunds:1]" in body.answer
    assert "14 days" in body.answer
    assert any(item.quote in body.answer for item in body.citations)


def test_ungrounded_model_answer_is_revised(settings, tmp_path):
    _index_samples(settings, tmp_path / "docs")
    body = run_question(
        "What is the refund window for the plus plan?",
        settings,
        llm=ScriptedLLM(),
    )
    assert body.grounded
    assert "Unlimited refunds" not in body.answer
    assert "[refunds:1]" in body.answer
    assert "14 days" in body.answer


def test_sql_route_cites_the_catalog(settings, tmp_path):
    _index_samples(settings, tmp_path / "docs")
    body = run_question("How many documents are in the policies category?", settings)
    assert body.route == "sql"
    assert "[Catalog:1]" in body.answer
    assert "n=2" in body.answer


def test_sample_corpus_cites_the_right_pages(settings):
    ingest_paths([Path("documind/sample_docs")], settings)
    late = run_question("When do late fees apply?", settings)
    assert late.route == "retrieve"
    assert late.grounded
    assert "[billing:2]" in late.answer
    refund = run_question("What is the refund window for the plus plan?", settings)
    assert "[refunds:1]" in refund.answer
    assert "14 days" in refund.answer
    assert "#" not in refund.answer


def test_short_question_asks_for_clarification(settings):
    body = run_question("help", settings)
    assert body.route == "clarify"
    assert body.citations == []
    assert "[" not in body.answer


def test_openai_embedding_branch_calls_langchain(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, model: str, api_key: str) -> None:
            captured["model"] = model
            captured["api_key"] = api_key

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.2, 0.4, 0.4] for _ in texts]

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", FakeEmbeddings)
    embedder = DocumindEmbedding(
        provider="openai",
        model="text-embedding-3-small",
        api_key="sk-test",
    )
    vectors = embedder._openai(["refund window"])
    assert captured == {"model": "text-embedding-3-small", "api_key": "sk-test"}
    assert len(vectors) == 1
    assert len(vectors[0]) == 3


def test_openai_chat_requires_a_key(settings):
    settings.llm_provider = "openai"
    settings.openai_api_key = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        LLMClient(settings).build_chat()


def test_ollama_chat_builds_without_a_server(settings):
    settings.llm_provider = "ollama"
    settings.ollama_model = "llama3.1"
    chat = LLMClient(settings).build_chat()
    assert chat.model == "llama3.1"


def test_local_embedding_is_stable(settings):
    embedder = build_embedding(settings)
    first = embedder(["refund window of 14 days"])[0]
    second = embedder(["refund window of 14 days"])[0]
    assert first.shape == second.shape
    assert (first == second).all()

