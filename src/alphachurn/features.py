"""Cleaning and point-in-time features: RFM, lags, and rolling ratios.

Each snapshot uses rows dated on or before the as-of date. Windows are
open on the left and closed on the right, so adjacent 30-day lags do not
share a boundary day:

    spend_30d   = (as_of - 30 days, as_of]
    spend_lag_1 = (as_of - 60 days, as_of - 30 days]
    spend_lag_2 = (as_of - 90 days, as_of - 60 days]
    spend_lag_3 = (as_of - 120 days, as_of - 90 days]

The 90-day RFM window is the union of the first three of those. A rolling
ratio is the latest 30-day value divided by that 90-day total. A zero
denominator becomes 0: the customer had no activity in the window, and
recency or the raw counts already say so.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphachurn.config import EVENT_TYPES, LABEL_HORIZON_DAYS, PURCHASE_TYPES, TXN_TYPES

NUMERIC_FEATURES = (
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "frequency_lifetime",
    "monetary_lifetime",
    "spend_30d",
    "spend_lag_1",
    "spend_lag_2",
    "spend_lag_3",
    "txn_30d",
    "txn_lag_1",
    "txn_lag_2",
    "login_30d",
    "login_lag_1",
    "login_lag_2",
    "login_lag_3",
    "ticket_30d",
    "payment_fail_30d",
    "spend_ratio_30_90",
    "txn_ratio_30_90",
    "login_ratio_30_90",
    "tenure_days",
    "monthly_fee",
    "refund_ratio_90d",
)

CATEGORICAL_FEATURES = (
    "segment",
    "acquisition_channel",
    "region",
    "plan_tier",
)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL = "churned"
KEYS = ("customer_id", "as_of_date")


@dataclass
class SourceTables:
    customers: pd.DataFrame
    subscriptions: pd.DataFrame
    transactions: pd.DataFrame
    events: pd.DataFrame


def clean_tables(tables: SourceTables) -> SourceTables:
    return SourceTables(
        customers=_clean_customers(tables.customers),
        subscriptions=_clean_subscriptions(tables.subscriptions),
        transactions=_clean_transactions(tables.transactions),
        events=_clean_events(tables.events),
    )


def build_feature_table(
    tables: SourceTables,
    as_of_dates,
    horizon_days: int = LABEL_HORIZON_DAYS,
    require_label: bool = True,
) -> pd.DataFrame:
    """Stack one feature row per active customer per as-of date."""
    tables = clean_tables(tables)
    frames = [
        snapshot_features(tables, as_of, horizon_days, require_label) for as_of in as_of_dates
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_frame(require_label)
    return pd.concat(frames, ignore_index=True)


def snapshot_features(
    tables: SourceTables,
    as_of,
    horizon_days: int = LABEL_HORIZON_DAYS,
    require_label: bool = True,
) -> pd.DataFrame:
    """Features for customers still active at `as_of`."""
    as_of = pd.Timestamp(as_of).normalize()
    customers = tables.customers
    subscriptions = tables.subscriptions
    customers = customers.loc[customers["signup_date"] <= as_of]
    subscriptions = subscriptions.loc[subscriptions["start_date"] <= as_of]
    if customers.empty or subscriptions.empty:
        return _empty_frame(require_label)

    latest_sub = (
        subscriptions.sort_values(["customer_id", "start_date"])
        .groupby("customer_id", as_index=False)
        .tail(1)
    )
    already_gone = latest_sub["end_date"].notna() & (latest_sub["end_date"] <= as_of)
    active = latest_sub.loc[~already_gone]
    active = active.loc[active["customer_id"].isin(customers["customer_id"])]
    if active.empty:
        return _empty_frame(require_label)

    index = pd.Index(active["customer_id"].to_numpy(), name="customer_id")
    attrs = customers.set_index("customer_id").reindex(index)
    fees = active.set_index("customer_id")["monthly_fee"].reindex(index)

    tx = tables.transactions
    tx = tx.loc[(tx["txn_date"] <= as_of) & (tx["customer_id"].isin(index))]
    ev = tables.events
    ev = ev.loc[(ev["event_date"] <= as_of) & (ev["customer_id"].isin(index))]
    purchases = tx.loc[tx["txn_type"].isin(PURCHASE_TYPES)]
    refunds = tx.loc[tx["txn_type"] == "refund"]

    last_purchase = purchases.groupby("customer_id")["txn_date"].max()
    recency = (as_of - last_purchase).dt.days
    tenure = (as_of - attrs["signup_date"]).dt.days.astype(float)
    recency = recency.reindex(index).fillna(tenure).astype(float)

    spend_30, spend_l1, spend_l2, spend_l3 = _lagged_sums(
        purchases, "txn_date", "amount", as_of, index
    )
    txn_30, txn_l1, txn_l2, _txn_l3 = _lagged_counts(purchases, "txn_date", as_of, index)
    login_30, login_l1, login_l2, login_l3 = _lagged_event_counts(ev, "login", as_of, index)
    ticket_30, _, _, _ = _lagged_event_counts(ev, "support_ticket", as_of, index)
    fail_30, _, _, _ = _lagged_event_counts(ev, "payment_failed", as_of, index)

    monetary_90 = spend_30 + spend_l1 + spend_l2
    frequency_90 = txn_30 + txn_l1 + txn_l2
    login_90 = login_30 + login_l1 + login_l2
    refund_90 = _sum_between(
        refunds, "txn_date", "amount", as_of - pd.Timedelta(days=90), as_of, index
    )

    frame = pd.DataFrame(
        {
            "customer_id": index.to_numpy(),
            "as_of_date": as_of,
            "segment": attrs["segment"].fillna("unknown").to_numpy(),
            "acquisition_channel": attrs["acquisition_channel"].fillna("unknown").to_numpy(),
            "region": attrs["region"].fillna("unknown").to_numpy(),
            "plan_tier": attrs["plan_tier"].fillna("unknown").to_numpy(),
            "recency_days": recency.to_numpy(),
            "frequency_90d": frequency_90.to_numpy(),
            "monetary_90d": monetary_90.to_numpy(),
            "frequency_lifetime": _count_all(purchases, index).to_numpy(),
            "monetary_lifetime": _sum_all(purchases, "amount", index).to_numpy(),
            "spend_30d": spend_30.to_numpy(),
            "spend_lag_1": spend_l1.to_numpy(),
            "spend_lag_2": spend_l2.to_numpy(),
            "spend_lag_3": spend_l3.to_numpy(),
            "txn_30d": txn_30.to_numpy(),
            "txn_lag_1": txn_l1.to_numpy(),
            "txn_lag_2": txn_l2.to_numpy(),
            "login_30d": login_30.to_numpy(),
            "login_lag_1": login_l1.to_numpy(),
            "login_lag_2": login_l2.to_numpy(),
            "login_lag_3": login_l3.to_numpy(),
            "ticket_30d": ticket_30.to_numpy(),
            "payment_fail_30d": fail_30.to_numpy(),
            "spend_ratio_30_90": _ratio(spend_30, monetary_90).to_numpy(),
            "txn_ratio_30_90": _ratio(txn_30, frequency_90).to_numpy(),
            "login_ratio_30_90": _ratio(login_30, login_90).to_numpy(),
            "tenure_days": tenure.to_numpy(),
            "monthly_fee": fees.astype(float).to_numpy(),
            "refund_ratio_90d": _ratio(refund_90, monetary_90).to_numpy(),
        }
    )

    if require_label:
        end = active.set_index("customer_id")["end_date"].reindex(index)
        horizon_end = as_of + pd.Timedelta(days=horizon_days)
        churned = end.notna() & (end <= horizon_end)
        frame[LABEL] = churned.fillna(False).astype(int).to_numpy()

    return frame.reset_index(drop=True)


def _clean_customers(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.dropna(subset=["customer_id", "signup_date"])
    out["customer_id"] = out["customer_id"].astype(int)
    out["signup_date"] = pd.to_datetime(out["signup_date"]).dt.normalize()
    for column in ("segment", "acquisition_channel", "region", "plan_tier"):
        out[column] = out[column].astype(str).str.strip().str.lower()
    return out.drop_duplicates(subset=["customer_id"]).reset_index(drop=True)


def _clean_subscriptions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.dropna(subset=["customer_id", "start_date", "monthly_fee"])
    out["customer_id"] = out["customer_id"].astype(int)
    out["start_date"] = pd.to_datetime(out["start_date"]).dt.normalize()
    out["end_date"] = pd.to_datetime(out["end_date"]).dt.normalize()
    out["monthly_fee"] = pd.to_numeric(out["monthly_fee"], errors="coerce")
    out = out.dropna(subset=["monthly_fee"])
    out = out.loc[out["monthly_fee"] >= 0]
    return out.drop_duplicates(subset=["subscription_id"]).reset_index(drop=True)


def _clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.dropna(subset=["transaction_id", "customer_id", "txn_date"])
    out["transaction_id"] = out["transaction_id"].astype(int)
    out["customer_id"] = out["customer_id"].astype(int)
    out["txn_date"] = pd.to_datetime(out["txn_date"]).dt.normalize()
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["txn_type"] = out["txn_type"].astype(str).str.strip().str.lower()
    out = out.loc[out["txn_type"].isin(TXN_TYPES) & (out["amount"] >= 0)]
    return out.drop_duplicates(subset=["transaction_id"]).reset_index(drop=True)


def _clean_events(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.dropna(subset=["event_id", "customer_id", "event_date"])
    out["event_id"] = out["event_id"].astype(int)
    out["customer_id"] = out["customer_id"].astype(int)
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.normalize()
    out["event_type"] = out["event_type"].astype(str).str.strip().str.lower()
    out = out.loc[out["event_type"].isin(EVENT_TYPES)]
    return out.drop_duplicates(subset=["event_id"]).reset_index(drop=True)


def _lagged_sums(frame, date_col, value_col, as_of, index):
    return tuple(
        _sum_between(frame, date_col, value_col, start, end, index)
        for start, end in _lag_windows(as_of, 4)
    )


def _lagged_counts(frame, date_col, as_of, index):
    return tuple(
        _count_between(frame, date_col, start, end, index) for start, end in _lag_windows(as_of, 4)
    )


def _lagged_event_counts(events, event_type, as_of, index):
    part = events.loc[events["event_type"] == event_type] if not events.empty else events
    return _lagged_counts(part, "event_date", as_of, index)


def _lag_windows(as_of: pd.Timestamp, n_lags: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    for lag in range(n_lags):
        end = as_of - pd.Timedelta(days=30 * lag)
        start = as_of - pd.Timedelta(days=30 * (lag + 1))
        windows.append((start, end))
    return windows


def _sum_between(frame, date_col, value_col, start, end, index) -> pd.Series:
    if frame.empty:
        return pd.Series(0.0, index=index)
    mask = (frame[date_col] > start) & (frame[date_col] <= end)
    summed = frame.loc[mask].groupby("customer_id")[value_col].sum()
    return summed.reindex(index).fillna(0.0).astype(float)


def _count_between(frame, date_col, start, end, index) -> pd.Series:
    if frame.empty:
        return pd.Series(0.0, index=index)
    mask = (frame[date_col] > start) & (frame[date_col] <= end)
    counted = frame.loc[mask].groupby("customer_id").size()
    return counted.reindex(index).fillna(0.0).astype(float)


def _sum_all(frame, value_col, index) -> pd.Series:
    if frame.empty:
        return pd.Series(0.0, index=index)
    summed = frame.groupby("customer_id")[value_col].sum()
    return summed.reindex(index).fillna(0.0).astype(float)


def _count_all(frame, index) -> pd.Series:
    if frame.empty:
        return pd.Series(0.0, index=index)
    counted = frame.groupby("customer_id").size()
    return counted.reindex(index).fillna(0.0).astype(float)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numerator.to_numpy(dtype=float)
    den = denominator.to_numpy(dtype=float)
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return pd.Series(out, index=numerator.index)


def _empty_frame(require_label: bool) -> pd.DataFrame:
    columns = list(KEYS) + list(MODEL_FEATURES)
    if require_label:
        columns.append(LABEL)
    return pd.DataFrame(columns=columns)
