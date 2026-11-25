"""SQLite connection and the SQL files under sql/."""

import sqlite3
from pathlib import Path

import pandas as pd

from alphachurn.config import ROOT

SQL_DIR = ROOT / "sql"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript((SQL_DIR / "schema.sql").read_text())
    conn.commit()


def read_query(conn: sqlite3.Connection, filename: str, as_of: str) -> pd.DataFrame:
    sql = (SQL_DIR / filename).read_text()
    return pd.read_sql_query(sql, conn, params={"as_of": as_of})
