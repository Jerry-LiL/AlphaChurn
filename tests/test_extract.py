import pandas as pd

from alphachurn.config import DATA_END, Settings
from alphachurn.db import connect, init_schema
from alphachurn.extract import extract_sources
from alphachurn.generate import generate_dataset


def test_sql_extract_respects_as_of(tmp_path):
    db_path = tmp_path / "churn.db"
    conn = connect(db_path)
    init_schema(conn)
    conn.execute(
        "INSERT INTO customers VALUES (1, '2024-01-01', 'consumer', 'organic', 'west', 'basic')"
    )
    conn.execute(
        "INSERT INTO customers VALUES (2, '2024-08-01', 'consumer', 'paid', 'south', 'plus')"
    )
    conn.execute(
        "INSERT INTO subscriptions VALUES (1, 1, '2024-01-01', '2024-09-01', 9.99, 'cancelled')"
    )
    conn.execute("INSERT INTO subscriptions VALUES (2, 2, '2024-08-01', NULL, 19.99, 'active')")
    conn.execute("INSERT INTO transactions VALUES (1, 1, '2024-06-15', 9.99, 'subscription')")
    conn.execute("INSERT INTO transactions VALUES (2, 1, '2024-07-15', 40, 'addon')")
    conn.execute("INSERT INTO events VALUES (1, 1, '2024-06-20', 'login')")
    conn.execute("INSERT INTO events VALUES (2, 1, '2024-07-20', 'login')")
    conn.commit()
    conn.close()

    tables = extract_sources(db_path, "2024-06-30")
    assert list(tables.customers["customer_id"]) == [1]
    assert list(tables.transactions["transaction_id"]) == [1]
    assert list(tables.events["event_id"]) == [1]
    assert list(tables.subscriptions["subscription_id"]) == [1]
    assert tables.subscriptions.iloc[0]["end_date"] == pd.Timestamp("2024-09-01")


def test_generated_activity_stops_before_cancel(tmp_path):
    settings = Settings(
        db_path=tmp_path / "book.db",
        artifact_dir=tmp_path / "artifacts",
        n_customers=60,
        seed=3,
    )
    counts = generate_dataset(settings)
    assert counts["customers"] == 60
    assert counts["transactions"] > 0
    assert counts["events"] > 0

    tables = extract_sources(settings.db_path, DATA_END)
    assert set(tables.subscriptions["status"]) >= {"active", "cancelled"}
    cancelled = tables.subscriptions.loc[tables.subscriptions["end_date"].notna()]
    assert not cancelled.empty

    purchases = tables.transactions.merge(cancelled, on="customer_id")
    assert (purchases["txn_date"] < purchases["end_date"]).all()
    activity = tables.events.merge(cancelled, on="customer_id")
    assert (activity["event_date"] < activity["end_date"]).all()

    signed = tables.transactions.merge(
        tables.customers[["customer_id", "signup_date"]], on="customer_id"
    )
    assert (signed["txn_date"] >= signed["signup_date"]).all()
