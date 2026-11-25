import sqlite3

import pytest

from documind.sqltool import (
    assert_read_only,
    compile_analytics_sql,
    connect_readonly,
    init_catalog,
    run_select,
    upsert_document,
)


def test_write_sql_is_rejected():
    with pytest.raises(ValueError):
        assert_read_only("DELETE FROM documents")
    with pytest.raises(ValueError):
        assert_read_only("SELECT * FROM documents; DROP TABLE documents")
    with pytest.raises(ValueError):
        assert_read_only("UPDATE documents SET pages = 1")


def test_readonly_connection_cannot_delete(tmp_path):
    path = init_catalog(tmp_path / "catalog.sqlite")
    upsert_document(
        path,
        {
            "doc": "refunds",
            "title": "refunds",
            "category": "policies",
            "file_type": "markdown",
            "pages": 1,
            "chunks": 1,
            "source": "refunds.md",
            "ingested_at": "2024-01-01T00:00:00+00:00",
        },
    )
    with connect_readonly(path) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM documents")
    rows = run_select(path, "SELECT COUNT(*) AS n FROM documents")
    assert rows == [{"n": 1}]


def test_analytics_compiler_filters_by_category():
    sql, params = compile_analytics_sql("How many documents are in the policies category?")
    assert "category = :category" in sql
    assert params == {"category": "policies"}
    sql, params = compile_analytics_sql("What is the average page count?")
    assert "AVG(pages)" in sql
    assert params == {}
