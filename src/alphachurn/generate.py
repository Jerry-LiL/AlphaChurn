"""Build a local subscription history with a visible pre-churn slowdown.

The generator is the stand-in for a warehouse extract. Customers who start to
churn first spend a month with fewer logins and add-on purchases and more
tickets and failed payments, then cancel early the following month. That keeps
the label (cancel within 30 days) aligned with behavior the snapshot can see.
A small share cancel without that slowdown.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphachurn.config import DATA_END, DATA_START, Settings, month_starts
from alphachurn.db import connect, init_schema

PLANS = {"basic": 9.99, "plus": 19.99, "premium": 39.99}
SEGMENTS = ("consumer", "smb")
CHANNELS = ("organic", "paid", "referral")
REGIONS = ("west", "south", "midwest", "northeast")


@dataclass
class Population:
    customers: list[tuple]
    subscriptions: list[tuple]
    transactions: list[tuple]
    events: list[tuple]


def generate_dataset(settings: Settings) -> dict[str, int]:
    rng = np.random.default_rng(settings.seed)
    if settings.db_path.exists():
        settings.db_path.unlink()

    population = _simulate_population(rng, settings.n_customers)
    conn = connect(settings.db_path)
    try:
        init_schema(conn)
        conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)",
            population.customers,
        )
        conn.executemany(
            "INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?, ?)",
            population.subscriptions,
        )
        conn.executemany(
            "INSERT INTO transactions VALUES (?, ?, ?, ?, ?)",
            population.transactions,
        )
        conn.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?)",
            population.events,
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "customers": len(population.customers),
        "subscriptions": len(population.subscriptions),
        "transactions": len(population.transactions),
        "events": len(population.events),
    }


def _simulate_population(rng: np.random.Generator, n_customers: int) -> Population:
    data_start = pd.Timestamp(DATA_START)
    data_end = pd.Timestamp(DATA_END)
    signup_last = pd.Timestamp("2024-09-30")
    months = month_starts(DATA_START, DATA_END)

    customers: list[tuple] = []
    subscriptions: list[tuple] = []
    transactions: list[tuple] = []
    events: list[tuple] = []
    txn_id = 1
    event_id = 1

    for customer_id in range(1, n_customers + 1):
        offset = int(rng.integers(0, (signup_last - data_start).days + 1))
        signup = data_start + pd.Timedelta(days=offset)
        segment = str(rng.choice(SEGMENTS))
        channel = str(rng.choice(CHANNELS, p=[0.4, 0.4, 0.2]))
        region = str(rng.choice(REGIONS))
        plan = str(rng.choice(list(PLANS), p=[0.5, 0.35, 0.15]))
        fee = PLANS[plan]
        latent = float(rng.random())

        end_date, txn_rows, event_rows, txn_id, event_id = _simulate_customer(
            rng,
            customer_id,
            signup,
            fee,
            latent,
            channel,
            plan,
            segment,
            months,
            data_end,
            txn_id,
            event_id,
        )
        customers.append(
            (
                customer_id,
                _iso(signup),
                segment,
                channel,
                region,
                plan,
            )
        )
        subscriptions.append(
            (
                customer_id,
                customer_id,
                _iso(signup),
                _iso(end_date) if end_date is not None else None,
                fee,
                "cancelled" if end_date is not None else "active",
            )
        )
        transactions.extend(txn_rows)
        events.extend(event_rows)

    return Population(customers, subscriptions, transactions, events)


def _simulate_customer(
    rng: np.random.Generator,
    customer_id: int,
    signup: pd.Timestamp,
    fee: float,
    latent: float,
    channel: str,
    plan: str,
    segment: str,
    month_starts: list[pd.Timestamp],
    data_end: pd.Timestamp,
    txn_id: int,
    event_id: int,
) -> tuple[pd.Timestamp | None, list[tuple], list[tuple], int, int]:
    hazard = 0.04 + 0.12 * latent
    if channel == "paid":
        hazard += 0.02
    if channel == "referral":
        hazard -= 0.01
    if plan == "premium":
        hazard -= 0.015
    if segment == "smb":
        hazard -= 0.01
    hazard = float(np.clip(hazard, 0.02, 0.35))

    state = "healthy"
    decline_months = 0
    end_date: pd.Timestamp | None = None
    txns: list[tuple] = []
    evts: list[tuple] = []

    for month_start in month_starts:
        month_start = pd.Timestamp(month_start)
        month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
        if month_end < signup:
            continue

        active_start = max(month_start, signup)
        active_end = min(month_end, data_end)
        tenure_days = int((month_end - signup).days)

        if state == "healthy" and tenure_days >= 60 and rng.random() < hazard:
            state = "decline"
            decline_months = 0

        declining = state == "decline"
        txn_id = _append_transactions(
            txns, rng, customer_id, signup, fee, declining, active_start, active_end, txn_id
        )
        event_id = _append_events(
            evts, rng, customer_id, declining, active_start, active_end, event_id
        )

        if declining:
            decline_months += 1
            if rng.random() < 0.7:
                scheduled = _schedule_cancel(rng, month_start, data_end)
                if scheduled is not None:
                    end_date = scheduled
                    break
        elif tenure_days >= 90 and rng.random() < 0.01:
            scheduled = _schedule_cancel(rng, month_start, data_end)
            if scheduled is not None:
                end_date = scheduled
                break

    return end_date, txns, evts, txn_id, event_id


def _append_transactions(
    txns: list[tuple],
    rng: np.random.Generator,
    customer_id: int,
    signup: pd.Timestamp,
    fee: float,
    declining: bool,
    active_start: pd.Timestamp,
    active_end: pd.Timestamp,
    txn_id: int,
) -> int:
    bill_day = min(int(signup.day), int(active_end.day))
    bill_date = pd.Timestamp(year=active_end.year, month=active_end.month, day=bill_day)
    if bill_date < signup:
        bill_date = signup
    if active_start <= bill_date <= active_end:
        txns.append((txn_id, customer_id, _iso(bill_date), round(float(fee), 2), "subscription"))
        txn_id += 1

    addon_rate = 0.05 if declining else 1.1
    for _ in range(int(rng.poisson(addon_rate))):
        day = _random_day(rng, active_start, active_end)
        amount = round(float(rng.uniform(4.0, 30.0)), 2)
        txns.append((txn_id, customer_id, _iso(day), amount, "addon"))
        txn_id += 1

    refund_p = 0.12 if declining else 0.02
    if rng.random() < refund_p:
        day = _random_day(rng, active_start, active_end)
        txns.append((txn_id, customer_id, _iso(day), round(float(fee) * 0.5, 2), "refund"))
        txn_id += 1
    return txn_id


def _append_events(
    evts: list[tuple],
    rng: np.random.Generator,
    customer_id: int,
    declining: bool,
    active_start: pd.Timestamp,
    active_end: pd.Timestamp,
    event_id: int,
) -> int:
    rates = {
        "login": 2.0 if declining else 12.0,
        "support_ticket": 2.4 if declining else 0.12,
        "payment_failed": 1.6 if declining else 0.04,
    }
    for event_type, rate in rates.items():
        for _ in range(int(rng.poisson(rate))):
            day = _random_day(rng, active_start, active_end)
            evts.append((event_id, customer_id, _iso(day), event_type))
            event_id += 1
    return event_id


def _schedule_cancel(
    rng: np.random.Generator, month_start: pd.Timestamp, data_end: pd.Timestamp
) -> pd.Timestamp | None:
    """Put the cancel date in the next month so this month's slowdown is visible."""
    next_month = (month_start + pd.offsets.MonthBegin(1)).normalize()
    candidate = next_month + pd.Timedelta(days=int(rng.integers(2, 20)))
    if candidate > data_end:
        return None
    return candidate


def _random_day(rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    span = int((end - start).days)
    if span <= 0:
        return start
    return start + pd.Timedelta(days=int(rng.integers(0, span + 1)))


def _iso(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()
