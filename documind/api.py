"""Streaming FastAPI service for cited answers."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

from documind.config import Settings
from documind.graph import run_question
from documind.ingest import ingest_paths
from documind.schemas import AnswerBody, QuestionIn


def create_app(settings: Settings | None = None) -> FastAPI:
    active = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if os.environ.get("STARTUP_INGEST") == "1" and settings is None:
            docs = Path(os.environ.get("DOCS_PATH", "documind/sample_docs"))
            if docs.exists():
                ingest_paths([docs], app.state.settings)
        yield

    app = FastAPI(title="DocuMind", lifespan=lifespan)
    app.state.settings = active

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/ingest")
    async def ingest(files: list[UploadFile] = File(...), category: str | None = None) -> dict:
        folder = active.chroma_path.parent / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        saved = []
        for upload in files:
            name = Path(upload.filename or "document.md").name
            destination = folder / name
            destination.write_bytes(await upload.read())
            saved.append(destination)
        return ingest_paths(saved, active, category=category)

    @app.post("/v1/ask/sync", response_model=AnswerBody)
    def ask_sync(body: QuestionIn) -> AnswerBody:
        return run_question(body.question, active)

    @app.post("/v1/ask")
    def ask(body: QuestionIn) -> StreamingResponse:
        payload = run_question(body.question, active)

        def events():
            text = payload.answer
            for start in range(0, len(text), 32):
                yield _sse({"type": "delta", "text": text[start : start + 32]})
            yield _sse({"type": "result", "data": payload.model_dump()})

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


app = create_app()
