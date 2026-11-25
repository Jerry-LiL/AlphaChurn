"""Walk-forward comparison of logistic regression and tuned XGBoost."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from alphachurn.config import DATA_END, Settings, labeled_as_ofs
from alphachurn.explain import explain_matrix, global_importance, save_importance_plot
from alphachurn.extract import extract_sources
from alphachurn.features import LABEL, build_feature_table
from alphachurn.metrics import classification_scores
from alphachurn.models import (
    DEFAULT_XGB_GRID,
    LOGISTIC_PARAMS,
    feature_matrix,
    fit_logistic,
    fit_xgboost,
    positive_scores,
    tune_xgboost,
)
from alphachurn.splits import Fold, tuning_split, walk_forward_folds


@dataclass
class TrainingResult:
    metrics: pd.DataFrame
    summary: pd.DataFrame
    params: dict
    tune_valid_as_of: pd.Timestamp
    n_rows: int
    positive_rate: float
    artifact_dir: Path


def run_training(settings: Settings, param_grid=None) -> TrainingResult:
    if not settings.db_path.exists():
        raise FileNotFoundError(
            f"No database at {settings.db_path}. Run `python -m alphachurn.cli generate` first."
        )
    grid = param_grid or DEFAULT_XGB_GRID
    tables = extract_sources(settings.db_path, DATA_END)
    frame = build_feature_table(tables, labeled_as_ofs(), require_label=True)
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.normalize()
    if frame.empty or frame[LABEL].nunique() < 2:
        raise RuntimeError("Labeled snapshots need both classes before a model can be fit.")

    dates = list(frame["as_of_date"].drop_duplicates())
    tune_train, tune_valid = tuning_split(dates, settings.min_train_periods)
    params = tune_xgboost(frame, tune_train, tune_valid, grid, settings.seed)
    folds = walk_forward_folds(dates, settings.min_train_periods)
    metrics = _evaluate(frame, folds, params, settings.threshold, settings.seed)

    features, target = feature_matrix(frame)
    xgb_model = fit_xgboost(features, target, params, settings.seed)
    logistic_model = fit_logistic(features, target, settings.seed)
    if xgb_model is None or logistic_model is None:
        raise RuntimeError("Final refit saw a single class.")

    artifact_dir = settings.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _dump_model(
        artifact_dir / "xgboost.joblib",
        xgb_model,
        "xgboost",
        params,
        settings,
        frame["as_of_date"].max(),
    )
    _dump_model(
        artifact_dir / "logistic.joblib",
        logistic_model,
        "logistic_regression",
        LOGISTIC_PARAMS,
        settings,
        frame["as_of_date"].max(),
    )

    latest = frame["as_of_date"].max()
    latest_rows = frame.loc[frame["as_of_date"] == latest]
    latest_features, _ = feature_matrix(latest_rows)
    values, names = explain_matrix(xgb_model, latest_features)
    importance = global_importance(values, names)
    importance.to_csv(artifact_dir / "shap_importance.csv", index=False)
    save_importance_plot(importance, artifact_dir / "shap_importance.png")

    metrics.to_csv(artifact_dir / "cv_metrics.csv", index=False)
    summary = metrics.groupby("model")[["roc_auc", "precision", "recall"]].mean().sort_index()
    summary.to_csv(artifact_dir / "cv_summary.csv")
    payload = {
        "seed": settings.seed,
        "n_rows": int(len(frame)),
        "positive_rate": float(frame[LABEL].mean()),
        "min_train_periods": settings.min_train_periods,
        "tune_valid_as_of": tune_valid.date().isoformat(),
        "threshold": settings.threshold,
        "xgboost_params": params,
        "logistic_params": LOGISTIC_PARAMS,
        "labeled_as_of_start": frame["as_of_date"].min().date().isoformat(),
        "labeled_as_of_end": frame["as_of_date"].max().date().isoformat(),
    }
    (artifact_dir / "run.json").write_text(json.dumps(payload, indent=2) + "\n")

    return TrainingResult(
        metrics=metrics,
        summary=summary,
        params=params,
        tune_valid_as_of=tune_valid,
        n_rows=int(len(frame)),
        positive_rate=float(frame[LABEL].mean()),
        artifact_dir=artifact_dir,
    )


def _evaluate(
    frame, folds: list[Fold], xgb_params: dict, threshold: float, seed: int
) -> pd.DataFrame:
    rows = []
    for fold_number, fold in enumerate(folds, start=1):
        train = frame.loc[frame["as_of_date"].isin(fold.train_dates)]
        valid = frame.loc[frame["as_of_date"] == fold.valid_date]
        x_train, y_train = feature_matrix(train)
        x_valid, y_valid = feature_matrix(valid)
        fitted = {
            "logistic_regression": fit_logistic(x_train, y_train, seed),
            "xgboost": fit_xgboost(x_train, y_train, xgb_params, seed),
        }
        for model_name, model in fitted.items():
            if model is None:
                scores = np.full(len(valid), np.nan)
            else:
                scores = positive_scores(model, x_valid)
            scored = classification_scores(y_valid, scores, threshold)
            params = xgb_params if model_name == "xgboost" else LOGISTIC_PARAMS
            rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "train_as_of_min": fold.train_dates[0].date().isoformat(),
                    "train_as_of_max": fold.train_end.date().isoformat(),
                    "valid_as_of": fold.valid_date.date().isoformat(),
                    "n_train": int(len(train)),
                    "n_valid": int(len(valid)),
                    "n_pos_valid": int(np.sum(y_valid == 1)),
                    "roc_auc": scored["roc_auc"],
                    "precision": scored["precision"],
                    "recall": scored["recall"],
                    "threshold": threshold,
                    "params": json.dumps(params, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def _dump_model(path, pipeline, model_name, params, settings: Settings, trained_through) -> None:
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": model_name,
            "params": params,
            "threshold": settings.threshold,
            "seed": settings.seed,
            "trained_through": pd.Timestamp(trained_through).date().isoformat(),
        },
        path,
    )
