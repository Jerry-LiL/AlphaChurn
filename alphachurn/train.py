"""Train the churn comparison and write a scoring artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from alphachurn.config import HORIZON_DAYS, RANDOM_SEED
from alphachurn.explain import mean_abs_shap
from alphachurn.extract import clean_events, extract_events
from alphachurn.features import build_feature_table, default_snapshots
from alphachurn.generate import generate_database
from alphachurn.modeling import compare_models, format_report


def run_training(
    db_path: Path,
    artifact_dir: Path,
    n_splits: int = 3,
    seed: int = RANDOM_SEED,
    customers: int | None = None,
    months: int | None = None,
    param_grid: list[dict] | None = None,
) -> dict:
    db_path = Path(db_path)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if customers is not None or not db_path.exists():
        generate_database(
            db_path,
            n_customers=customers or 240,
            n_months=months or 16,
            seed=seed,
        )

    events = clean_events(extract_events(db_path))
    snapshots = default_snapshots(events, horizon_days=HORIZON_DAYS)
    if len(snapshots) < n_splits + 2:
        raise ValueError(
            "Not enough month-end snapshots after the history and horizon buffers. "
            "Generate a longer event history."
        )
    features = build_feature_table(events, snapshots, horizon_days=HORIZON_DAYS)
    report = compare_models(features, n_splits=n_splits, param_grid=param_grid, seed=seed)
    model = report.pop("model")

    last_test = max(row["test_as_of"] for row in report["folds"])
    explain_rows = features[features["as_of"] == pd.Timestamp(last_test)].dropna(subset=["label"])
    if len(explain_rows) > 80:
        explain_rows = explain_rows.sample(80, random_state=seed)
    shap_summary = mean_abs_shap(model, explain_rows)

    bundle = {
        "model": model,
        "threshold": report["threshold"],
        "champion": report["champion"],
        "champion_params": report["champion_params"],
        "horizon_days": HORIZON_DAYS,
    }
    joblib.dump(bundle, artifact_dir / "model.joblib")
    metrics = {
        "folds": report["folds"],
        "summary": report["summary"],
        "champion": report["champion"],
        "champion_params": report["champion_params"],
        "threshold": report["threshold"],
        "snapshots": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in snapshots],
        "shap_mean_abs": shap_summary,
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(format_report(report))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the churn baseline and XGBoost model.")
    parser.add_argument("--db", type=Path, default=Path("data/churn.sqlite"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--customers", type=int, default=240)
    parser.add_argument("--months", type=int, default=16)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--reuse-db",
        action="store_true",
        help="Train on an existing SQLite file instead of generating a new one.",
    )
    args = parser.parse_args(argv)
    run_training(
        db_path=args.db,
        artifact_dir=args.out,
        n_splits=args.splits,
        seed=args.seed,
        customers=None if args.reuse_db else args.customers,
        months=None if args.reuse_db else args.months,
    )


if __name__ == "__main__":
    main()
