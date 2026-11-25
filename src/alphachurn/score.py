"""Score one as-of date with the refit XGBoost model and attach SHAP reasons."""

from pathlib import Path

import joblib
import pandas as pd

from alphachurn.config import Settings
from alphachurn.explain import explain_matrix, local_reasons
from alphachurn.extract import extract_sources
from alphachurn.features import build_feature_table
from alphachurn.models import feature_matrix, positive_scores


def score_as_of(settings: Settings, as_of: str, model_path: Path | None = None) -> pd.DataFrame:
    if not settings.db_path.exists():
        raise FileNotFoundError(
            f"No database at {settings.db_path}. Run `python -m alphachurn.cli generate` first."
        )
    path = model_path or (settings.artifact_dir / "xgboost.joblib")
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `python -m alphachurn.cli train` first.")
    payload = joblib.load(path)
    pipeline = payload["pipeline"]
    tables = extract_sources(settings.db_path, as_of)
    frame = build_feature_table(tables, [as_of], require_label=False)
    if frame.empty:
        raise RuntimeError(f"No active customers to score on {as_of}.")

    features, _ = feature_matrix(frame)
    probability = positive_scores(pipeline, features)
    values, names = explain_matrix(pipeline, features)
    reasons = local_reasons(values, names, k=3)
    scored = pd.DataFrame(
        {
            "customer_id": frame["customer_id"].to_numpy(),
            "as_of_date": pd.to_datetime(frame["as_of_date"]).dt.strftime("%Y-%m-%d"),
            "segment": frame["segment"].to_numpy(),
            "plan_tier": frame["plan_tier"].to_numpy(),
            "churn_probability": probability,
            "predicted_churn": (probability >= settings.threshold).astype(int),
        }
    )
    scored = pd.concat([scored.reset_index(drop=True), reasons.reset_index(drop=True)], axis=1)
    scored = scored.sort_values(
        ["churn_probability", "customer_id"], ascending=[False, True]
    ).reset_index(drop=True)

    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(settings.artifact_dir / "scores.csv", index=False)
    return scored
