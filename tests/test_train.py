import json

import joblib
import numpy as np
import pandas as pd

from alphachurn.explain import explain_matrix
from alphachurn.extract import extract_sources
from alphachurn.features import build_feature_table
from alphachurn.models import feature_matrix


def test_walk_forward_comparison_writes_metrics_and_shap(trained):
    settings, result = trained
    metrics = result.metrics
    assert set(metrics["model"]) == {"logistic_regression", "xgboost"}
    for column in ("roc_auc", "precision", "recall"):
        assert column in metrics.columns
    assert (metrics["train_as_of_max"] < metrics["valid_as_of"]).all()
    tune_month = result.tune_valid_as_of.date().isoformat()
    assert tune_month not in set(metrics["valid_as_of"])
    xgb_params = metrics.loc[metrics["model"] == "xgboost", "params"]
    assert xgb_params.nunique() == 1
    assert metrics.groupby("model")["roc_auc"].apply(lambda series: series.notna().any()).all()

    artifact_dir = settings.artifact_dir
    for name in (
        "cv_metrics.csv",
        "cv_summary.csv",
        "run.json",
        "xgboost.joblib",
        "logistic.joblib",
        "shap_importance.csv",
        "shap_importance.png",
    ):
        assert (artifact_dir / name).exists()
    assert (artifact_dir / "shap_importance.png").stat().st_size > 100

    importance = pd.read_csv(artifact_dir / "shap_importance.csv")
    assert list(importance.columns) == ["feature", "mean_abs_shap"]
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert (importance["mean_abs_shap"] >= 0).all()

    payload = json.loads((artifact_dir / "run.json").read_text())
    assert payload["tune_valid_as_of"] == tune_month
    assert payload["labeled_as_of_end"] == "2024-11-30"
    assert payload["xgboost_params"] == result.params

    model = joblib.load(artifact_dir / "xgboost.joblib")
    names = list(model["pipeline"].named_steps["preprocess"].get_feature_names_out())
    assert "recency_days" in names
    assert "churned" not in names
    assert "end_date" not in names
    assert "status" not in names
    assert model["trained_through"] == "2024-11-30"

    tables = extract_sources(settings.db_path, "2024-11-30")
    snapshot = build_feature_table(tables, ["2024-11-30"], require_label=False).head(25)
    features, _ = feature_matrix(snapshot)
    values, shap_names = explain_matrix(model["pipeline"], features)
    assert values.shape == (len(features), len(shap_names))
    assert np.isfinite(values).all()
