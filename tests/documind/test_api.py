import json

from fastapi.testclient import TestClient

from documind.api import create_app
from documind.ingest import ingest_paths


def test_health_sync_and_stream(settings, tmp_path):
    docs = tmp_path / "docs" / "policies"
    docs.mkdir(parents=True)
    (docs / "refunds.md").write_text(
        "Plus plan customers have a refund window of 14 days from the charge date.\n",
        encoding="utf-8",
    )
    ingest_paths([tmp_path / "docs"], settings)
    client = TestClient(create_app(settings))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    sync = client.post(
        "/v1/ask/sync",
        json={"question": "What is the refund window for the plus plan?"},
    )
    assert sync.status_code == 200
    assert "[refunds:1]" in sync.json()["answer"]

    streamed = client.post(
        "/v1/ask",
        json={"question": "What is the refund window for the plus plan?"},
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    deltas = []
    result = None
    for line in streamed.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line.removeprefix("data: "))
        if payload["type"] == "delta":
            deltas.append(payload["text"])
        if payload["type"] == "result":
            result = payload["data"]
    assert result is not None
    assert "".join(deltas) == result["answer"]
    assert "[refunds:1]" in result["answer"]


def test_upload_then_ask(settings):
    client = TestClient(create_app(settings))
    uploaded = client.post(
        "/v1/ingest",
        files={
            "files": (
                "onboarding.md",
                b"Laptops are issued on day one by workplace IT.\n",
                "text/markdown",
            )
        },
        data={"category": "handbook"},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["documents"] == 1
    answer = client.post("/v1/ask/sync", json={"question": "When are laptops issued?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["route"] == "retrieve"
    assert "[onboarding:1]" in body["answer"]
    assert "day one" in body["answer"]
