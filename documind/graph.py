"""LangGraph that routes a question, drafts an answer, and grades grounding."""

from __future__ import annotations

import sqlite3
from typing import TypedDict

from langgraph.graph import END, StateGraph

from documind.answer import (
    CLARIFY_TEXT,
    ROUTE_SYSTEM,
    catalog_answer,
    compose_from_hits,
    draft_prompt,
    heuristic_route,
    parse_model_answer,
    retrieval_is_grounded,
    sql_is_grounded,
)
from documind.config import Settings
from documind.llm import LLMClient
from documind.retrieve import hybrid_search
from documind.schemas import AnswerBody, Citation
from documind.sqltool import compile_analytics_sql, render_rows, run_select


class GraphState(TypedDict, total=False):
    question: str
    route: str
    hits: list[dict]
    sql: str
    sql_rows: list[dict]
    sql_text: str
    answer: str
    citations: list[dict]
    grounded: bool
    grade_notes: str
    revisions: int


def build_graph(settings: Settings, llm: LLMClient):
    def route(state: GraphState) -> dict:
        choice = heuristic_route(state["question"])
        if llm.provider != "local":
            try:
                raw = llm.complete(ROUTE_SYSTEM, state["question"]).strip().lower()
            except RuntimeError:
                raw = ""
            if raw in {"retrieve", "sql", "clarify"}:
                choice = raw
        return {"route": choice, "revisions": state.get("revisions", 0)}

    def retrieve(state: GraphState) -> dict:
        return {"hits": hybrid_search(settings, state["question"], k=settings.top_k)}

    def sql(state: GraphState) -> dict:
        try:
            statement, params = compile_analytics_sql(state["question"])
            rows = run_select(settings.catalog_path, statement, params)
        except (ValueError, sqlite3.Error) as exc:
            return {
                "route": "clarify",
                "answer": CLARIFY_TEXT,
                "citations": [],
                "sql": "",
                "sql_rows": [],
                "sql_text": "",
                "grade_notes": str(exc),
            }
        return {"sql": statement, "sql_rows": rows, "sql_text": render_rows(rows)}

    def draft(state: GraphState) -> dict:
        if state.get("route") == "clarify":
            return {"answer": state.get("answer") or CLARIFY_TEXT, "citations": []}
        if state.get("route") == "sql":
            answer, citations = catalog_answer(state.get("sql_rows") or [])
            return {"answer": answer, "citations": [item.model_dump() for item in citations]}
        hits = state.get("hits") or []
        if llm.provider != "local" and hits:
            raw = llm.complete(*draft_prompt(state["question"], hits))
            parsed = parse_model_answer(raw)
            if parsed and parsed.get("answer"):
                return {
                    "answer": str(parsed["answer"]),
                    "citations": list(parsed.get("citations") or []),
                }
        answer, citations = compose_from_hits(state["question"], hits)
        if not citations:
            return {"route": "clarify", "answer": CLARIFY_TEXT, "citations": []}
        return {"answer": answer, "citations": [item.model_dump() for item in citations]}

    def grade(state: GraphState) -> dict:
        route_name = state.get("route", "retrieve")
        if route_name == "clarify":
            return {"grounded": True, "grade_notes": "clarification", "citations": []}
        citations = _citations(state.get("citations") or [])
        if route_name == "sql":
            ok = sql_is_grounded(state.get("answer", ""), citations, state.get("sql_text", ""))
        else:
            ok = retrieval_is_grounded(state.get("answer", ""), citations, state.get("hits") or [])
        return {
            "grounded": ok,
            "grade_notes": "ok" if ok else "ungrounded",
            "citations": [item.model_dump() for item in citations],
        }

    def revise(state: GraphState) -> dict:
        revisions = int(state.get("revisions", 0)) + 1
        if state.get("route") == "sql":
            answer, citations = catalog_answer(state.get("sql_rows") or [])
            return {
                "answer": answer,
                "citations": [item.model_dump() for item in citations],
                "revisions": revisions,
            }
        answer, citations = compose_from_hits(state["question"], state.get("hits") or [])
        if not citations:
            return {
                "route": "clarify",
                "answer": CLARIFY_TEXT,
                "citations": [],
                "revisions": revisions,
            }
        return {
            "answer": answer,
            "citations": [item.model_dump() for item in citations],
            "revisions": revisions,
        }

    graph = StateGraph(GraphState)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("sql", sql)
    graph.add_node("draft", draft)
    graph.add_node("grade", grade)
    graph.add_node("revise", revise)
    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"retrieve": "retrieve", "sql": "sql", "clarify": "draft"},
    )
    graph.add_edge("retrieve", "draft")
    graph.add_edge("sql", "draft")
    graph.add_edge("draft", "grade")
    graph.add_conditional_edges("grade", _after_grade, {"revise": "revise", "done": END})
    graph.add_edge("revise", "grade")
    return graph.compile()


def run_question(question: str, settings: Settings, llm: LLMClient | None = None) -> AnswerBody:
    client = llm or LLMClient(settings)
    final = build_graph(settings, client).invoke({"question": question, "revisions": 0})
    return state_to_answer(final)


def state_to_answer(state: GraphState) -> AnswerBody:
    route_name = state.get("route", "clarify")
    if route_name == "clarify" or not state.get("grounded", False):
        return AnswerBody(
            route="clarify",
            answer=state.get("answer") or CLARIFY_TEXT,
            citations=[],
            grounded=True,
        )
    citations = _citations(state.get("citations") or [])
    return AnswerBody(
        route=route_name,
        answer=state.get("answer", ""),
        citations=citations,
        grounded=True,
    )


def _after_grade(state: GraphState) -> str:
    if state.get("grounded"):
        return "done"
    if state.get("route") == "clarify":
        return "done"
    if int(state.get("revisions", 0)) >= 1:
        return "done"
    return "revise"


def _citations(raw: list) -> list[Citation]:
    citations = []
    for item in raw:
        if isinstance(item, Citation):
            citations.append(item)
            continue
        try:
            citations.append(Citation(**item))
        except (TypeError, ValueError):
            continue
    return citations
