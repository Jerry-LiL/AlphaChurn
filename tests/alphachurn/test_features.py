import numpy as np
import pandas as pd

from alphachurn.config import NUMERIC_FEATURES
from alphachurn.features import build_feature_table
from alphachurn.modeling import walk_forward_folds


def _row(customer_id, event_date, event_type, amount, signup="2024-01-01", plan="basic"):
    return {
        "customer_id": customer_id,
        "event_date": pd.Timestamp(event_date),
        "event_type": event_type,
        "amount": amount,
        "signup_date": pd.Timestamp(signup),
        "plan": plan,
        "region": "NA",
        "segment": "smb",
    }


def _table():
    events = pd.DataFrame(
        [
            _row(1, "2024-01-15", "purchase", 7),
            _row(1, "2024-01-31", "purchase", 3),
            _row(1, "2024-02-15", "purchase", 5),
            _row(1, "2024-03-01", "purchase", 4),
            _row(1, "2024-03-02", "login", 0),
            _row(1, "2024-03-10", "purchase", 11),
            _row(1, "2024-03-28", "login", 0),
            _row(1, "2024-03-31", "ticket", 0),
            _row(1, "2024-04-10", "cancel", 0),
            _row(1, "2024-04-05", "purchase", 100),
        ]
    )
    return build_feature_table(events, [pd.Timestamp("2024-03-31")], horizon_days=30)


def test_windows_lags_ratios_and_rfm():
    frame = _table()
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["monetary_7d"] == 0
    assert row["monetary_30d"] == 11
    assert row["monetary_90d"] == 7 + 3 + 5 + 4 + 11
    assert row["lag1_monetary_30d"] == 5 + 4
    assert row["lag2_monetary_30d"] == 7 + 3
    assert row["frequency_7d"] == 2  # login Mar 28 and ticket Mar 31
    assert row["frequency_30d"] == 4  # login Mar 2, purchase Mar 10, login Mar 28, ticket Mar 31
    assert row["login_count_30d"] == 2
    assert row["purchase_count_30d"] == 1
    assert row["ticket_count_30d"] == 1
    assert row["rfm_recency_days"] == 0  # ticket lands on the snapshot
    assert row["rfm_frequency_90d"] == row["frequency_90d"]
    assert row["rfm_monetary_90d"] == row["monetary_90d"]
    assert row["roll_ratio_monetary_30_90"] == 11 / row["monetary_90d"]
    assert row["roll_ratio_frequency_30_90"] == row["frequency_30d"] / row["frequency_90d"]
    assert row["label"] == 1
    assert row["tenure_days"] == (pd.Timestamp("2024-03-31") - pd.Timestamp("2024-01-01")).days


def test_future_events_do_not_change_features():
    baseline = _table()
    events = pd.DataFrame(
        [
            _row(1, "2024-01-15", "purchase", 7),
            _row(1, "2024-01-31", "purchase", 3),
            _row(1, "2024-02-15", "purchase", 5),
            _row(1, "2024-03-01", "purchase", 4),
            _row(1, "2024-03-02", "login", 0),
            _row(1, "2024-03-10", "purchase", 11),
            _row(1, "2024-03-28", "login", 0),
            _row(1, "2024-03-31", "ticket", 0),
            _row(1, "2024-04-10", "cancel", 0),
            _row(1, "2024-04-05", "purchase", 999),
            _row(1, "2024-04-20", "purchase", 999),
        ]
    )
    updated = build_feature_table(events, [pd.Timestamp("2024-03-31")], horizon_days=30)
    feature_cols = [col for col in baseline.columns if col != "label"]
    pd.testing.assert_frame_equal(
        baseline[feature_cols].reset_index(drop=True),
        updated[feature_cols].reset_index(drop=True),
    )


def test_label_horizon_and_already_churned():
    events = pd.DataFrame(
        [
            _row(1, "2024-03-15", "purchase", 10),
            _row(1, "2024-04-30", "cancel", 0),
            _row(2, "2024-03-15", "purchase", 10),
            _row(2, "2024-05-02", "cancel", 0),
            _row(3, "2024-03-15", "purchase", 8),
            _row(3, "2024-03-31", "cancel", 0),
        ]
    )
    frame = build_feature_table(events, [pd.Timestamp("2024-03-31")], horizon_days=30)
    labels = dict(zip(frame["customer_id"], frame["label"], strict=True))
    assert labels[1] == 1
    assert labels[2] == 0
    assert 3 not in labels


def test_open_left_boundary_is_not_shared_by_lag():
    events = pd.DataFrame(
        [
            _row(1, "2024-03-01", "purchase", 9),
            _row(1, "2024-03-02", "purchase", 6),
        ]
    )
    frame = build_feature_table(events, [pd.Timestamp("2024-03-31")], horizon_days=30)
    row = frame.iloc[0]
    assert row["monetary_30d"] == 6
    assert row["lag1_monetary_30d"] == 9


def test_walk_forward_tests_are_strictly_later():
    dates = list(pd.date_range("2024-01-31", periods=6, freq="ME"))
    folds = walk_forward_folds(dates, n_splits=3)
    assert len(folds) == 3
    for train_dates, test_dates in folds:
        assert test_dates[0] > max(train_dates)
        assert set(train_dates).isdisjoint(test_dates)


def test_zero_denominator_ratio_is_zero():
    events = pd.DataFrame([_row(1, "2024-03-20", "login", 0)])
    frame = build_feature_table(events, [pd.Timestamp("2024-03-31")], horizon_days=30)
    assert frame.iloc[0]["roll_ratio_monetary_7_30"] == 0
    assert np.isfinite(frame[NUMERIC_FEATURES].to_numpy(dtype=float)).all()
