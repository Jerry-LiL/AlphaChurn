import sqlite3

import pandas as pd

from alphachurn.extract import SQL_FILE, clean_events, extract_events, load_extract_sql
from alphachurn.generate import SCHEMA_PATH, generate_database


def test_extract_sql_joins_customers(tmp_path):
    db_path = tmp_path / "events.sqlite"
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)
        conn.execute(
            """
            INSERT INTO customers (customer_id, signup_date, plan, region, segment)
            VALUES (1, '2024-01-01', 'plus', 'EU', 'smb')
            """
        )
        conn.execute(
            """
            INSERT INTO events (event_id, customer_id, event_date, event_type, amount)
            VALUES (1, 1, '2024-02-01', 'purchase', 12.5)
            """
        )
        conn.commit()

    frame = extract_events(db_path)
    assert list(frame.columns) == [
        "customer_id",
        "event_date",
        "event_type",
        "amount",
        "signup_date",
        "plan",
        "region",
        "segment",
    ]
    assert frame.loc[0, "plan"] == "plus"
    assert "INNER JOIN customers" in load_extract_sql()
    assert SQL_FILE.exists()


def test_clean_events_drops_unusable_rows():
    raw = pd.DataFrame(
        [
            {
                "customer_id": 1,
                "event_date": "2024-02-01",
                "event_type": "purchase",
                "amount": 10,
                "signup_date": "2024-01-01",
                "plan": "basic",
                "region": "NA",
                "segment": "smb",
            },
            {
                "customer_id": 1,
                "event_date": "2024-02-01",
                "event_type": "purchase",
                "amount": 10,
                "signup_date": "2024-01-01",
                "plan": "basic",
                "region": "NA",
                "segment": "smb",
            },
            {
                "customer_id": 1,
                "event_date": "2023-12-01",
                "event_type": "purchase",
                "amount": 5,
                "signup_date": "2024-01-01",
                "plan": "basic",
                "region": "NA",
                "segment": "smb",
            },
            {
                "customer_id": 1,
                "event_date": "2024-02-10",
                "event_type": "refund",
                "amount": 5,
                "signup_date": "2024-01-01",
                "plan": "basic",
                "region": "NA",
                "segment": "smb",
            },
            {
                "customer_id": None,
                "event_date": "2024-02-10",
                "event_type": "login",
                "amount": 0,
                "signup_date": "2024-01-01",
                "plan": "basic",
                "region": "NA",
                "segment": "smb",
            },
            {
                "customer_id": 1,
                "event_date": "2024-03-01",
                "event_type": "purchase",
                "amount": -8,
                "signup_date": "2024-01-01",
                "plan": "basic",
                "region": "NA",
                "segment": "smb",
            },
        ]
    )
    cleaned = clean_events(raw)
    assert len(cleaned) == 2
    march = cleaned[cleaned["event_date"] == pd.Timestamp("2024-03-01")]
    assert float(march["amount"].iloc[0]) == 0.0


def test_generate_database_round_trip(tmp_path):
    db_path = generate_database(tmp_path / "churn.sqlite", n_customers=25, n_months=8, seed=3)
    frame = clean_events(extract_events(db_path))
    assert frame["customer_id"].nunique() == 25
    assert set(frame["event_type"]).issubset({"login", "purchase", "ticket", "cancel"})
