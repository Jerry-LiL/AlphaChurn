import pandas as pd
import pytest

from alphachurn.features import (
    LABEL,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    SourceTables,
    build_feature_table,
    clean_tables,
)

AS_OF = "2024-06-30"


def test_feature_catalog_matches_the_model_matrix():
    required = [
        "recency_days",
        "frequency_90d",
        "monetary_90d",
        "frequency_lifetime",
        "monetary_lifetime",
        "spend_lag_1",
        "spend_lag_2",
        "spend_lag_3",
        "login_lag_1",
        "login_lag_2",
        "login_lag_3",
        "spend_ratio_30_90",
        "txn_ratio_30_90",
        "login_ratio_30_90",
    ]
    for name in required:
        assert name in NUMERIC_FEATURES
    assert LABEL not in MODEL_FEATURES
    assert "end_date" not in MODEL_FEATURES
    assert "status" not in MODEL_FEATURES


def test_windows_rfm_and_labels():
    frame = build_feature_table(_sample_tables(), [AS_OF])
    assert "status" not in frame.columns
    assert "end_date" not in frame.columns
    present = set(frame["customer_id"])
    assert present == {1, 3, 4, 6, 7, 8, 9}
    labels = frame.set_index("customer_id")[LABEL]
    assert labels.loc[1] == 0
    assert labels.loc[3] == 1
    assert labels.loc[4] == 0
    assert labels.loc[7] == 1
    assert labels.loc[8] == 0

    row = frame.set_index("customer_id").loc[1]
    assert row["spend_30d"] == pytest.approx(17)
    assert row["spend_lag_1"] == pytest.approx(3)
    assert row["spend_lag_2"] == pytest.approx(4)
    assert row["spend_lag_3"] == pytest.approx(8)
    assert row["monetary_90d"] == pytest.approx(24)
    assert row["monetary_lifetime"] == pytest.approx(37)
    assert row["frequency_90d"] == pytest.approx(4)
    assert row["frequency_lifetime"] == pytest.approx(6)
    assert row["txn_30d"] == pytest.approx(2)
    assert row["txn_lag_1"] == pytest.approx(1)
    assert row["txn_lag_2"] == pytest.approx(1)
    assert row["spend_ratio_30_90"] == pytest.approx(17 / 24)
    assert row["txn_ratio_30_90"] == pytest.approx(0.5)
    assert row["recency_days"] == pytest.approx(0)
    assert row["login_30d"] == pytest.approx(2)
    assert row["login_lag_1"] == pytest.approx(1)
    assert row["login_lag_2"] == pytest.approx(1)
    assert row["login_lag_3"] == pytest.approx(1)
    assert row["login_ratio_30_90"] == pytest.approx(0.5)
    assert row["ticket_30d"] == pytest.approx(1)
    assert row["payment_fail_30d"] == pytest.approx(1)
    assert row["refund_ratio_90d"] == pytest.approx(0.25)
    assert row["monthly_fee"] == pytest.approx(19.99)
    assert row["tenure_days"] == pytest.approx(273)
    assert row["plan_tier"] == "plus"

    quiet = frame.set_index("customer_id").loc[6]
    assert quiet["recency_days"] == pytest.approx(quiet["tenure_days"])
    assert quiet["tenure_days"] == pytest.approx(60)
    assert quiet["spend_ratio_30_90"] == pytest.approx(0)
    assert quiet["frequency_90d"] == pytest.approx(0)

    brand_new = frame.set_index("customer_id").loc[9]
    assert brand_new["tenure_days"] == pytest.approx(0)
    assert brand_new["recency_days"] == pytest.approx(0)


def test_future_transaction_does_not_change_the_snapshot():
    tables = _sample_tables()
    before = build_feature_table(tables, [AS_OF]).sort_values("customer_id")
    extra = pd.DataFrame(
        [
            {
                "transaction_id": 999,
                "customer_id": 1,
                "txn_date": "2024-07-02",
                "amount": 5000.0,
                "txn_type": "addon",
            }
        ]
    )
    mutated = SourceTables(
        customers=tables.customers,
        subscriptions=tables.subscriptions,
        transactions=pd.concat([tables.transactions, extra], ignore_index=True),
        events=tables.events,
    )
    after = build_feature_table(mutated, [AS_OF]).sort_values("customer_id")
    pd.testing.assert_frame_equal(
        before.reset_index(drop=True),
        after.reset_index(drop=True),
    )


def test_cleaning_drops_invalid_rows():
    transactions = pd.DataFrame(
        [
            {
                "transaction_id": 1,
                "customer_id": 1,
                "txn_date": "2024-06-01",
                "amount": 10,
                "txn_type": "addon",
            },
            {
                "transaction_id": 1,
                "customer_id": 1,
                "txn_date": "2024-06-02",
                "amount": 99,
                "txn_type": "addon",
            },
            {
                "transaction_id": 2,
                "customer_id": 1,
                "txn_date": "2024-06-03",
                "amount": -5,
                "txn_type": "addon",
            },
            {
                "transaction_id": 3,
                "customer_id": 1,
                "txn_date": None,
                "amount": 4,
                "txn_type": "addon",
            },
            {
                "transaction_id": 4,
                "customer_id": 1,
                "txn_date": "2024-06-04",
                "amount": 4,
                "txn_type": "wire",
            },
        ]
    )
    events = pd.DataFrame(
        [
            {"event_id": 1, "customer_id": 1, "event_date": "2024-06-01", "event_type": "login"},
            {"event_id": 2, "customer_id": 1, "event_date": "2024-06-01", "event_type": "pageview"},
        ]
    )
    empty_customers = pd.DataFrame(
        columns=[
            "customer_id",
            "signup_date",
            "segment",
            "acquisition_channel",
            "region",
            "plan_tier",
        ]
    )
    empty_subs = pd.DataFrame(
        columns=[
            "subscription_id",
            "customer_id",
            "start_date",
            "end_date",
            "monthly_fee",
            "status",
        ]
    )
    cleaned = clean_tables(SourceTables(empty_customers, empty_subs, transactions, events))
    assert list(cleaned.transactions["transaction_id"]) == [1]
    assert cleaned.transactions.iloc[0]["amount"] == 10
    assert list(cleaned.events["event_type"]) == ["login"]


def _sample_tables() -> SourceTables:
    customers = pd.DataFrame(
        [
            _customer(1, "2023-10-01", "plus"),
            _customer(2, "2023-11-01", "basic"),
            _customer(3, "2024-01-15", "basic"),
            _customer(4, "2024-01-15", "basic"),
            _customer(5, "2024-07-02", "plus"),
            _customer(6, "2024-05-01", "basic"),
            _customer(7, "2024-02-01", "plus"),
            _customer(8, "2024-02-01", "plus"),
            _customer(9, "2024-06-30", "premium"),
        ]
    )
    subscriptions = pd.DataFrame(
        [
            _subscription(1, "2023-10-01", None, 19.99),
            _subscription(2, "2023-11-01", "2024-06-30", 9.99),
            _subscription(3, "2024-01-15", "2024-07-15", 9.99),
            _subscription(4, "2024-01-15", "2024-08-20", 9.99),
            _subscription(5, "2024-07-02", None, 19.99),
            _subscription(6, "2024-05-01", None, 9.99),
            _subscription(7, "2024-02-01", "2024-07-30", 19.99),
            _subscription(8, "2024-02-01", "2024-07-31", 19.99),
            _subscription(9, "2024-06-30", None, 39.99),
        ]
    )
    transactions = pd.DataFrame(
        [
            _txn(1, 1, "2024-06-30", 10, "subscription"),
            _txn(2, 1, "2024-06-01", 7, "addon"),
            _txn(3, 1, "2024-05-31", 3, "addon"),
            _txn(4, 1, "2024-05-01", 4, "addon"),
            _txn(5, 1, "2024-04-01", 8, "addon"),
            _txn(6, 1, "2024-03-01", 5, "addon"),
            _txn(7, 1, "2024-06-15", 6, "refund"),
        ]
    )
    events = pd.DataFrame(
        [
            _event(1, 1, "2024-06-15", "login"),
            _event(2, 1, "2024-06-16", "login"),
            _event(3, 1, "2024-05-15", "login"),
            _event(4, 1, "2024-04-15", "login"),
            _event(5, 1, "2024-03-15", "login"),
            _event(6, 1, "2024-06-10", "support_ticket"),
            _event(7, 1, "2024-06-11", "payment_failed"),
        ]
    )
    return SourceTables(customers, subscriptions, transactions, events)


def _customer(customer_id, signup, plan):
    return {
        "customer_id": customer_id,
        "signup_date": signup,
        "segment": "Consumer",
        "acquisition_channel": "Organic",
        "region": "West",
        "plan_tier": plan,
    }


def _subscription(customer_id, start, end, fee):
    return {
        "subscription_id": customer_id,
        "customer_id": customer_id,
        "start_date": start,
        "end_date": end,
        "monthly_fee": fee,
        "status": "cancelled" if end else "active",
    }


def _txn(txn_id, customer_id, txn_date, amount, txn_type):
    return {
        "transaction_id": txn_id,
        "customer_id": customer_id,
        "txn_date": txn_date,
        "amount": amount,
        "txn_type": txn_type,
    }


def _event(event_id, customer_id, event_date, event_type):
    return {
        "event_id": event_id,
        "customer_id": customer_id,
        "event_date": event_date,
        "event_type": event_type,
    }
