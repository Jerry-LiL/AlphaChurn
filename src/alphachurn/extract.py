"""Run the as-of SQL extracts and return cleaned source tables."""

import pandas as pd

from alphachurn.db import connect, read_query
from alphachurn.features import SourceTables, clean_tables


def extract_sources(db_path, as_of: str) -> SourceTables:
    """Read history through `as_of`. Rows after that date are not returned."""
    as_of = pd.Timestamp(as_of).date().isoformat()
    conn = connect(db_path)
    try:
        tables = SourceTables(
            customers=read_query(conn, "extract_customers.sql", as_of),
            subscriptions=read_query(conn, "extract_subscriptions.sql", as_of),
            transactions=read_query(conn, "extract_transactions.sql", as_of),
            events=read_query(conn, "extract_events.sql", as_of),
        )
    finally:
        conn.close()
    return clean_tables(tables)
