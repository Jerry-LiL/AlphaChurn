"""SQL extraction of the customer-event grain."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SQL_FILE = Path(__file__).resolve().parent / "sql" / "extract_events.sql"


def load_extract_sql() -> str:
    return SQL_FILE.read_text(encoding="utf-8")


def extract_events(db_path: Path) -> pd.DataFrame:
    sql = load_extract_sql()
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(sql, conn)
    if frame.empty:
        return frame
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame["signup_date"] = pd.to_datetime(frame["signup_date"])
    return frame


def clean_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be used as activity history.

    Unknown event types are removed rather than imputed. Negative amounts
    are clipped to zero because a refund is not modeled as its own type.
    """
    if frame.empty:
        return frame.copy()

    allowed = {"login", "purchase", "ticket", "cancel"}
    out = frame.copy()
    out = out.dropna(subset=["customer_id", "event_date", "event_type", "signup_date"])
    out["event_date"] = pd.to_datetime(out["event_date"])
    out["signup_date"] = pd.to_datetime(out["signup_date"])
    out["event_type"] = out["event_type"].astype(str).str.strip().str.lower()
    out = out[out["event_type"].isin(allowed)]
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out.loc[out["amount"] < 0, "amount"] = 0.0
    out["customer_id"] = out["customer_id"].astype(int)
    out = out[out["event_date"] >= out["signup_date"]]
    out = out.drop_duplicates(subset=["customer_id", "event_date", "event_type", "amount"])
    out = out.sort_values(["customer_id", "event_date", "event_type"]).reset_index(drop=True)
    return out
