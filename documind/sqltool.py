"""Read-only catalog queries.

The analytics tool can describe what was ingested. It cannot write. A keyword
check rejects obvious write statements, and the connection itself is opened
with SQLite's read-only mode.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|grant|truncate|vacuum|reindex)\b",
    re.IGNORECASE,
)

CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    file_type TEXT NOT NULL,
    pages INTEGER NOT NULL,
    chunks INTEGER NOT NULL,
    source TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
"""


def init_catalog(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(CATALOG_SCHEMA)
        conn.commit()
    return path


def upsert_document(path: Path, row: dict) -> None:
    init_catalog(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO documents (
                doc, title, category, file_type, pages, chunks, source, ingested_at
            )
            VALUES (:doc, :title, :category, :file_type, :pages, :chunks, :source, :ingested_at)
            ON CONFLICT(doc) DO UPDATE SET
                title = excluded.title,
                category = excluded.category,
                file_type = excluded.file_type,
                pages = excluded.pages,
                chunks = excluded.chunks,
                source = excluded.source,
                ingested_at = excluded.ingested_at
            """,
            row,
        )
        conn.commit()


def assert_read_only(sql: str) -> str:
    statement = sql.strip().rstrip(";").strip()
    if not statement:
        raise ValueError("SQL is empty.")
    if ";" in statement:
        raise ValueError("Only one statement is allowed.")
    if not statement.lower().startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    if _FORBIDDEN.search(statement):
        raise ValueError("Write keywords are not allowed.")
    return statement


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def run_select(path: Path, sql: str, params: dict | None = None) -> list[dict]:
    statement = assert_read_only(sql)
    with connect_readonly(path) as connection:
        rows = connection.execute(statement, params or {}).fetchall()
    return [dict(row) for row in rows]


def compile_analytics_sql(question: str) -> tuple[str, dict]:
    text = question.lower()
    category = _category(text)
    if re.search(r"\b(average|avg|mean)\b", text) and "page" in text:
        if category:
            return (
                "SELECT ROUND(AVG(pages), 2) AS avg_pages "
                "FROM documents WHERE category = :category",
                {"category": category},
            )
        return "SELECT ROUND(AVG(pages), 2) AS avg_pages FROM documents", {}
    if re.search(r"\b(how many|count|number of)\b", text):
        if category:
            return (
                "SELECT COUNT(*) AS n FROM documents WHERE category = :category",
                {"category": category},
            )
        return "SELECT COUNT(*) AS n FROM documents", {}
    if re.search(r"\blist\b", text) and "document" in text:
        return (
            "SELECT doc, category, pages, file_type FROM documents ORDER BY doc",
            {},
        )
    raise ValueError("This question does not map to the document catalog.")


def render_rows(rows: list[dict]) -> str:
    if not rows:
        return "no rows"
    rendered = []
    for row in rows:
        rendered.append(", ".join(f"{key}={value}" for key, value in row.items()))
    return "; ".join(rendered)


def _category(text: str) -> str | None:
    match = re.search(r"\b(?:category|in the|in)\s+([a-z0-9_-]+)", text)
    if not match:
        return None
    token = match.group(1)
    if token in {"the", "a", "an", "catalog", "index", "documents", "document", "all"}:
        return None
    return token
