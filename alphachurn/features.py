"""Point-in-time lags, rolling ratios, and RFM features.

Every window is open on the left and closed on the right, and every window
ends on or before the snapshot date. An event on the snapshot is included.
An event on the far edge is not, so a current window and the lag behind it
do not share a day. The label looks forward and is never a model input.
"""

from __future__ import annotations

import pandas as pd

from alphachurn.config import CATEGORICAL_FEATURES, HORIZON_DAYS, NUMERIC_FEATURES

ID_COLUMNS = ["customer_id", "as_of"]
LABEL_COLUMN = "label"


def build_feature_table(
    events: pd.DataFrame,
    snapshot_dates: list[pd.Timestamp],
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    if events.empty or not snapshot_dates:
        return _empty_frame()

    prepared = events.copy()
    prepared["event_date"] = pd.to_datetime(prepared["event_date"])
    prepared["signup_date"] = pd.to_datetime(prepared["signup_date"])
    cancel_dates = (
        prepared.loc[prepared["event_type"] == "cancel", ["customer_id", "event_date"]]
        .groupby("customer_id", as_index=False)["event_date"]
        .min()
        .rename(columns={"event_date": "cancel_date"})
    )
    profiles = (
        prepared.groupby("customer_id", as_index=False)
        .agg(
            signup_date=("signup_date", "min"),
            plan=("plan", "first"),
            region=("region", "first"),
            segment=("segment", "first"),
        )
        .merge(cancel_dates, on="customer_id", how="left")
    )
    observed_end = prepared["event_date"].max()

    frames = [
        _one_snapshot(prepared, profiles, pd.Timestamp(as_of), horizon_days, observed_end)
        for as_of in snapshot_dates
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_frame()
    table = pd.concat(frames, ignore_index=True)
    table["as_of"] = pd.to_datetime(table["as_of"])
    return table.sort_values(["as_of", "customer_id"]).reset_index(drop=True)


def default_snapshots(
    events: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
    history_days: int = 90,
) -> list[pd.Timestamp]:
    if events.empty:
        return []
    start = pd.to_datetime(events["event_date"]).min() + pd.Timedelta(days=history_days)
    end = pd.to_datetime(events["event_date"]).max() - pd.Timedelta(days=horizon_days)
    if start > end:
        return []
    return list(pd.date_range(start, end, freq="ME"))


def _one_snapshot(
    events: pd.DataFrame,
    profiles: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon_days: int,
    observed_end: pd.Timestamp,
) -> pd.DataFrame:
    eligible = profiles[profiles["signup_date"] <= as_of].copy()
    already_gone = eligible["cancel_date"].notna() & (eligible["cancel_date"] <= as_of)
    eligible = eligible.loc[~already_gone].copy()
    if eligible.empty:
        return _empty_frame()

    history = events[(events["event_date"] <= as_of) & (events["event_type"] != "cancel")]
    customer_ids = eligible["customer_id"]

    current_7 = _window(history, customer_ids, as_of, 7, 0)
    current_30 = _window(history, customer_ids, as_of, 30, 0)
    current_90 = _window(history, customer_ids, as_of, 90, 0)
    lag1 = _window(history, customer_ids, as_of, 60, 30)
    lag2 = _window(history, customer_ids, as_of, 90, 60)
    tickets_30 = _typed_count(history, customer_ids, as_of, "ticket", 30, 0)
    purchases_30 = _typed_count(history, customer_ids, as_of, "purchase", 30, 0)
    logins_30 = _typed_count(history, customer_ids, as_of, "login", 30, 0)

    last_activity = history.groupby("customer_id")["event_date"].max()
    recency = (as_of - last_activity).dt.days
    tenure = (as_of - eligible["signup_date"]).dt.days
    recency = eligible["customer_id"].map(recency)
    recency = recency.fillna(tenure).astype(float)

    horizon_end = as_of + pd.Timedelta(days=horizon_days)
    cancel = eligible["cancel_date"]
    churned = cancel.notna() & (cancel > as_of) & (cancel <= horizon_end)
    survived_past_horizon = cancel.notna() & (cancel > horizon_end)
    known = churned | survived_past_horizon | (observed_end >= horizon_end)
    label = [
        int(did_churn) if bool(is_known) else pd.NA
        for did_churn, is_known in zip(churned.to_numpy(), known.to_numpy(), strict=True)
    ]

    frame = pd.DataFrame(
        {
            "customer_id": eligible["customer_id"].to_numpy(),
            "as_of": as_of,
            "plan": eligible["plan"].to_numpy(),
            "region": eligible["region"].to_numpy(),
            "segment": eligible["segment"].to_numpy(),
            "tenure_days": tenure.to_numpy(dtype=float),
            "rfm_recency_days": recency.to_numpy(dtype=float),
            "rfm_frequency_90d": current_90["count"].to_numpy(dtype=float),
            "rfm_monetary_90d": current_90["monetary"].to_numpy(dtype=float),
            "frequency_7d": current_7["count"].to_numpy(dtype=float),
            "frequency_30d": current_30["count"].to_numpy(dtype=float),
            "frequency_90d": current_90["count"].to_numpy(dtype=float),
            "monetary_7d": current_7["monetary"].to_numpy(dtype=float),
            "monetary_30d": current_30["monetary"].to_numpy(dtype=float),
            "monetary_90d": current_90["monetary"].to_numpy(dtype=float),
            "lag1_frequency_30d": lag1["count"].to_numpy(dtype=float),
            "lag2_frequency_30d": lag2["count"].to_numpy(dtype=float),
            "lag1_monetary_30d": lag1["monetary"].to_numpy(dtype=float),
            "lag2_monetary_30d": lag2["monetary"].to_numpy(dtype=float),
            "ticket_count_30d": tickets_30.to_numpy(dtype=float),
            "purchase_count_30d": purchases_30.to_numpy(dtype=float),
            "login_count_30d": logins_30.to_numpy(dtype=float),
            "label": label,
        }
    )
    frame["roll_ratio_frequency_7_30"] = _ratio(frame["frequency_7d"], frame["frequency_30d"])
    frame["roll_ratio_frequency_30_90"] = _ratio(frame["frequency_30d"], frame["frequency_90d"])
    frame["roll_ratio_monetary_7_30"] = _ratio(frame["monetary_7d"], frame["monetary_30d"])
    frame["roll_ratio_monetary_30_90"] = _ratio(frame["monetary_30d"], frame["monetary_90d"])
    return frame


def _window(
    history: pd.DataFrame,
    customer_ids: pd.Series,
    as_of: pd.Timestamp,
    farther_days: int,
    nearer_days: int,
) -> pd.DataFrame:
    start = as_of - pd.Timedelta(days=farther_days)
    end = as_of - pd.Timedelta(days=nearer_days)
    window = history[(history["event_date"] > start) & (history["event_date"] <= end)]
    counts = window.groupby("customer_id").size()
    purchases = window[window["event_type"] == "purchase"]
    monetary = purchases.groupby("customer_id")["amount"].sum()
    out = pd.DataFrame({"customer_id": customer_ids.to_numpy()})
    out["count"] = out["customer_id"].map(counts).fillna(0.0)
    out["monetary"] = out["customer_id"].map(monetary).fillna(0.0)
    return out


def _typed_count(
    history: pd.DataFrame,
    customer_ids: pd.Series,
    as_of: pd.Timestamp,
    event_type: str,
    farther_days: int,
    nearer_days: int,
) -> pd.Series:
    start = as_of - pd.Timedelta(days=farther_days)
    end = as_of - pd.Timedelta(days=nearer_days)
    window = history[
        (history["event_type"] == event_type)
        & (history["event_date"] > start)
        & (history["event_date"] <= end)
    ]
    counts = window.groupby("customer_id").size()
    return customer_ids.map(counts).fillna(0.0)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe = denominator.where(denominator > 0)
    return numerator.div(safe).fillna(0.0)


def _empty_frame() -> pd.DataFrame:
    columns = ID_COLUMNS + CATEGORICAL_FEATURES + NUMERIC_FEATURES + [LABEL_COLUMN]
    return pd.DataFrame(columns=columns)
