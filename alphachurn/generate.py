"""Build a local SQLite event history with a pre-churn slowdown.

The generator is the stand-in for a warehouse. Training reads it through
the same SQL extract that would run against Postgres.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from alphachurn.config import RANDOM_SEED

SCHEMA_PATH = Path(__file__).resolve().parent / "sql" / "schema.sql"

PLANS = ("basic", "plus", "pro")
REGIONS = ("NA", "EU", "APAC")
SEGMENTS = ("smb", "mid_market", "enterprise")
PLAN_HAZARD = {"basic": 0.14, "plus": 0.08, "pro": 0.045}


def generate_database(
    db_path: Path,
    n_customers: int = 240,
    n_months: int = 16,
    seed: int = RANDOM_SEED,
    start: str = "2023-01-01",
) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    customers, events = simulate(
        n_customers=n_customers,
        n_months=n_months,
        seed=seed,
        start=start,
    )
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema)
        customers.to_sql("customers", conn, if_exists="append", index=False)
        events.to_sql("events", conn, if_exists="append", index=False)
        conn.commit()
    return db_path


def simulate(
    n_customers: int,
    n_months: int,
    seed: int,
    start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    month_starts = pd.date_range(start, periods=n_months, freq="MS")
    customer_rows: list[dict] = []
    event_rows: list[dict] = []
    event_id = 1

    for customer_id in range(1, n_customers + 1):
        plan = str(rng.choice(PLANS, p=[0.5, 0.32, 0.18]))
        region = str(rng.choice(REGIONS))
        segment = str(rng.choice(SEGMENTS, p=[0.55, 0.30, 0.15]))
        signup_offset = int(rng.integers(0, 70))
        signup = month_starts[0] + pd.Timedelta(days=signup_offset)
        engagement = float(rng.uniform(0.35, 1.0))
        customer_rows.append(
            {
                "customer_id": customer_id,
                "signup_date": signup.strftime("%Y-%m-%d"),
                "plan": plan,
                "region": region,
                "segment": segment,
            }
        )

        for month_index, month_start in enumerate(month_starts):
            if month_start + pd.Timedelta(days=27) < signup:
                continue
            hazard = _hazard(month_index, plan, engagement)
            roll = float(rng.random())
            if roll < hazard:
                day = int(rng.integers(0, 27))
                when = month_start + pd.Timedelta(days=day)
                event_rows.append(_event(event_id, customer_id, when, "cancel", 0.0))
                event_id += 1
                break
            if roll < hazard * 1.7:
                engagement = max(0.05, engagement * 0.4)

            n_purchases = int(rng.integers(0, 1 + int(engagement * 4)))
            for _ in range(n_purchases):
                day = int(rng.integers(0, 27))
                when = month_start + pd.Timedelta(days=day)
                if when < signup:
                    continue
                amount = float(rng.uniform(8.0, 40.0) * (1.4 if plan == "pro" else 1.0))
                event_rows.append(_event(event_id, customer_id, when, "purchase", round(amount, 2)))
                event_id += 1

            if rng.random() < 0.55 * engagement:
                day = int(rng.integers(0, 27))
                when = month_start + pd.Timedelta(days=day)
                if when >= signup:
                    event_rows.append(_event(event_id, customer_id, when, "login", 0.0))
                    event_id += 1

            if rng.random() < 0.08 * (1.4 - engagement):
                day = int(rng.integers(0, 27))
                when = month_start + pd.Timedelta(days=day)
                if when >= signup:
                    event_rows.append(_event(event_id, customer_id, when, "ticket", 0.0))
                    event_id += 1

    customers = pd.DataFrame(customer_rows)
    events = pd.DataFrame(event_rows)
    return customers, events


def _hazard(month_index: int, plan: str, engagement: float) -> float:
    time_factor = 0.35 + month_index / 12.0
    raw = PLAN_HAZARD[plan] * time_factor * (1.55 - engagement)
    return float(min(0.55, max(0.01, raw)))


def _event(
    event_id: int,
    customer_id: int,
    when: pd.Timestamp,
    event_type: str,
    amount: float,
) -> dict:
    return {
        "event_id": event_id,
        "customer_id": customer_id,
        "event_date": when.strftime("%Y-%m-%d"),
        "event_type": event_type,
        "amount": amount,
    }
