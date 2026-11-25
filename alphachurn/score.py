"""Score active customers at an as-of date and attach SHAP reasons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from alphachurn.config import HORIZON_DAYS
from alphachurn.explain import top_reasons
from alphachurn.extract import clean_events, extract_events
from alphachurn.features import build_feature_table


def score_customers(
    db_path: Path,
    model_path: Path,
    as_of: str,
    customer_id: int | None = None,
    reasons: int = 5,
) -> list[dict]:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    events = clean_events(extract_events(db_path))
    snapshot = pd.Timestamp(as_of)
    horizon = bundle.get("horizon_days", HORIZON_DAYS)
    features = build_feature_table(events, [snapshot], horizon_days=horizon)
    if customer_id is not None:
        features = features[features["customer_id"] == customer_id]
        if features.empty:
            return [
                {
                    "customer_id": customer_id,
                    "as_of": snapshot.strftime("%Y-%m-%d"),
                    "status": "not_scored",
                    "reason": (
                        "Customer is unknown, not yet signed up, or already churned at this date."
                    ),
                }
            ]

    if features.empty:
        return []

    probabilities = model.predict_proba(features)
    explanations = top_reasons(model, features, k=reasons)
    rows = []
    for position, (_, feature_row) in enumerate(features.iterrows()):
        probability = float(probabilities[position])
        rows.append(
            {
                "customer_id": int(feature_row["customer_id"]),
                "as_of": snapshot.strftime("%Y-%m-%d"),
                "status": "scored",
                "churn_probability": probability,
                "prediction": int(probability >= threshold),
                "threshold": threshold,
                "model": model.name,
                "top_reasons": explanations[position],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score churn risk for an as-of date.")
    parser.add_argument("--db", type=Path, default=Path("data/churn.sqlite"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--as-of", required=True, help="Snapshot date, YYYY-MM-DD.")
    parser.add_argument("--customer-id", type=int, default=None)
    parser.add_argument("--reasons", type=int, default=5)
    args = parser.parse_args(argv)
    payload = score_customers(
        db_path=args.db,
        model_path=args.model,
        as_of=args.as_of,
        customer_id=args.customer_id,
        reasons=args.reasons,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
