import numpy as np
import pandas as pd

from alphachurn.explain import shap_values, top_reasons
from alphachurn.extract import clean_events, extract_events
from alphachurn.features import build_feature_table, default_snapshots
from alphachurn.generate import generate_database
from alphachurn.modeling import compare_models
from alphachurn.score import score_customers
from alphachurn.train import run_training

TINY_GRID = [
    {"max_depth": 2, "learning_rate": 0.1, "min_child_weight": 1, "n_estimators": 20},
    {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "n_estimators": 15},
]


def test_walk_forward_report_and_shap(tmp_path):
    db_path = generate_database(tmp_path / "churn.sqlite", n_customers=90, n_months=12, seed=7)
    events = clean_events(extract_events(db_path))
    snapshots = default_snapshots(events)
    features = build_feature_table(events, snapshots)
    report = compare_models(features, n_splits=3, param_grid=TINY_GRID, seed=7)

    assert {row["model"] for row in report["folds"]} == {"logistic_regression", "xgboost"}
    for row in report["folds"]:
        assert row["train_through"] < row["test_as_of"]
        assert 0.0 <= row["roc_auc"] <= 1.0
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert row["n_test"] > 0

    model = report["model"]
    sample = features.dropna(subset=["label"]).head(12)
    values = shap_values(model, sample)
    assert values.shape[0] == len(sample)
    assert values.shape[1] == len(model.feature_names)
    reasons = top_reasons(model, sample, k=3)
    assert len(reasons) == len(sample)
    assert len(reasons[0]) == 3
    assert np.isfinite(values).all()


def test_train_script_writes_artifact_and_scores(tmp_path):
    db_path = tmp_path / "churn.sqlite"
    out = tmp_path / "artifacts"
    metrics = run_training(
        db_path=db_path,
        artifact_dir=out,
        n_splits=2,
        seed=11,
        customers=80,
        months=12,
        param_grid=TINY_GRID,
    )
    assert (out / "model.joblib").exists()
    assert metrics["champion"] in {"logistic_regression", "xgboost"}
    assert metrics["shap_mean_abs"]
    scored = score_customers(
        db_path=db_path,
        model_path=out / "model.joblib",
        as_of=metrics["snapshots"][-1],
    )
    assert scored
    assert scored[0]["status"] == "scored"
    assert 0.0 <= scored[0]["churn_probability"] <= 1.0
    assert scored[0]["top_reasons"]

    missing = score_customers(
        db_path=db_path,
        model_path=out / "model.joblib",
        as_of=metrics["snapshots"][-1],
        customer_id=999999,
    )
    assert missing[0]["status"] == "not_scored"


def test_score_ignores_events_after_the_snapshot(tmp_path):
    import sqlite3

    db_path = generate_database(tmp_path / "churn.sqlite", n_customers=70, n_months=12, seed=5)
    out = tmp_path / "artifacts"
    metrics = run_training(
        db_path=db_path,
        artifact_dir=out,
        n_splits=2,
        seed=5,
        customers=None,
        months=None,
        param_grid=TINY_GRID,
    )
    as_of = metrics["snapshots"][0]
    scored = score_customers(db_path, out / "model.joblib", as_of)
    target = next(row for row in scored if row["status"] == "scored")
    later = (pd.Timestamp(as_of) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO events (event_id, customer_id, event_date, event_type, amount)
            VALUES (900000, ?, ?, 'purchase', 5000)
            """,
            (target["customer_id"], later),
        )
    again = score_customers(
        db_path,
        out / "model.joblib",
        as_of,
        customer_id=target["customer_id"],
    )[0]
    assert again["status"] == "scored"
    assert again["churn_probability"] == target["churn_probability"]
