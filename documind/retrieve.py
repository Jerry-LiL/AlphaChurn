"""Hybrid retrieval: BM25 plus Chroma dense search, merged with RRF.

Reciprocal rank fusion is used because the two rankers are not on the same
scale. A chunk that both rankers like rises to the top.
"""

from __future__ import annotations

import re

import chromadb
from rank_bm25 import BM25Okapi

from documind.config import Settings
from documind.embeddings import build_embedding
from documind.ingest import COLLECTION, load_chunks

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def bm25_rank(chunks: list[dict], query: str, n: int) -> list[str]:
    if not chunks or n <= 0:
        return []
    tokens = [tokenize(chunk["text"]) or ["_empty"] for chunk in chunks]
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    model = BM25Okapi(tokens)
    scores = model.get_scores(query_tokens)
    order = sorted(range(len(chunks)), key=lambda index: float(scores[index]), reverse=True)
    ranked = [chunks[index]["id"] for index in order if scores[index] > 0]
    return ranked[:n]


def dense_rank(settings: Settings, query: str, n: int) -> list[str]:
    if n <= 0 or not settings.chroma_path.exists():
        return []
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=build_embedding(settings),
    )
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(query_texts=[query], n_results=min(n, count))
    ids = result.get("ids") or [[]]
    return list(ids[0])


def hybrid_search(settings: Settings, query: str, k: int | None = None) -> list[dict]:
    chunks = load_chunks(settings)
    if not chunks:
        return []
    limit = k or settings.top_k
    fused = reciprocal_rank_fusion(
        [
            bm25_rank(chunks, query, n=20),
            dense_rank(settings, query, n=20),
        ]
    )
    by_id = {chunk["id"]: chunk for chunk in chunks}
    hits = []
    for chunk_id, score in fused[:limit]:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        hit = dict(chunk)
        hit["score"] = score
        hits.append(hit)
    return hits
