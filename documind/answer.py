"""Turn retrieved text or catalog rows into a cited answer."""

from __future__ import annotations

import json
import re

from documind.retrieve import tokenize
from documind.schemas import Citation
from documind.sqltool import render_rows

_STOP = {
    "a",
    "an",
    "the",
    "of",
    "for",
    "and",
    "to",
    "is",
    "are",
    "what",
    "which",
    "how",
    "does",
    "do",
    "in",
    "on",
    "with",
    "from",
    "that",
    "this",
    "be",
    "when",
    "where",
    "who",
    "why",
    "can",
    "it",
}

CLARIFY_TEXT = (
    "Which topic should I look up? Refunds, billing, retention, and onboarding are in the index."
)

ROUTE_SYSTEM = (
    "Classify the user question as one word: retrieve, sql, or clarify. "
    "sql is for counts, averages, or lists of the document catalog. "
    "clarify is for a question that is too vague to search. "
    "retrieve is for anything that should be answered from document text."
)

DRAFT_SYSTEM = (
    "Answer only from the passages. Cite every claim inline as [Doc:Page]. "
    "Each citation quote must be an exact substring of a passage. "
    "Reply as JSON with keys answer and citations. "
    'citations is a list of objects with doc, page, and quote.'
)


def heuristic_route(question: str) -> str:
    tokens = tokenize(question)
    if len(tokens) < 3:
        return "clarify"
    if re.search(r"\b(how many|count|number of|average|avg|mean)\b", question, re.I):
        return "sql"
    if re.search(r"\blist\b", question, re.I) and re.search(r"\bdocuments?\b", question, re.I):
        return "sql"
    return "retrieve"


def compose_from_hits(question: str, hits: list[dict]) -> tuple[str, list[Citation]]:
    terms = set(tokenize(question)) - _STOP
    ranked: list[tuple[int, dict, str]] = []
    for hit in hits:
        for sentence in _sentences(hit["text"]):
            overlap = len(terms & set(tokenize(sentence)))
            if overlap:
                ranked.append((overlap, hit, sentence))
    if not ranked:
        return "", []
    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0][0]
    chosen: list[tuple[dict, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for score, hit, sentence in ranked:
        if score < best:
            break
        key = (hit["doc"], int(hit["page"]), sentence)
        if key in seen:
            continue
        seen.add(key)
        chosen.append((hit, sentence))
        if len(chosen) == 2:
            break
    if not chosen:
        return "", []
    parts = []
    citations = []
    for hit, sentence in chosen:
        parts.append(f"{sentence} [{hit['doc']}:{int(hit['page'])}]")
        citations.append(Citation(doc=hit["doc"], page=int(hit["page"]), quote=sentence))
    return " ".join(parts), citations


def catalog_answer(rows: list[dict]) -> tuple[str, list[Citation]]:
    rendered = render_rows(rows)
    answer = f"Catalog result: {rendered} [Catalog:1]."
    return answer, [Citation(doc="Catalog", page=1, quote=rendered)]


def retrieval_is_grounded(answer: str, citations: list[Citation], hits: list[dict]) -> bool:
    if not citations:
        return False
    for citation in citations:
        if f"[{citation.doc}:{citation.page}]" not in answer:
            return False
        pages = [
            hit["text"]
            for hit in hits
            if hit["doc"] == citation.doc and int(hit["page"]) == citation.page
        ]
        if not any(citation.quote in text for text in pages):
            return False
    return True


def sql_is_grounded(answer: str, citations: list[Citation], result_text: str) -> bool:
    if len(citations) != 1:
        return False
    citation = citations[0]
    if citation.doc != "Catalog" or citation.page != 1:
        return False
    if "[Catalog:1]" not in answer:
        return False
    return citation.quote == result_text or citation.quote in result_text


def parse_model_answer(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    return payload


def draft_prompt(question: str, hits: list[dict]) -> tuple[str, str]:
    passages = []
    for hit in hits:
        passages.append(f"[{hit['doc']}:{int(hit['page'])}]\n{hit['text']}")
    user = f"Question: {question}\n\nPassages:\n\n" + "\n\n".join(passages)
    return DRAFT_SYSTEM, user


def _sentences(text: str) -> list[str]:
    parts = re.split(r"\n+|(?<=[.!?])\s+", text.strip())
    sentences = []
    for part in parts:
        cleaned = part.strip().lstrip("-").strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        sentences.append(cleaned)
    return sentences
