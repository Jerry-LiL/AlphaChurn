"""Walk-forward comparison of logistic regression and XGBoost.

Folds move forward in snapshot time. The test snapshot is always later than
every training snapshot, and XGBoost's grid search uses only the last
training snapshot for validation. That keeps the reported ROC-AUC, precision,
and recall free of target leakage from later months.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from alphachurn.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    THRESHOLD,
    XGB_PARAM_GRID,
)
from alphachurn.features import LABEL_COLUMN

INPUT_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class FoldResult:
    model: str
    fold: int
    test_as_of: str
    train_through: str
    roc_auc: float
    precision: float
    recall: float
    n_train: int
    n_test: int
    params: dict

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "fold": self.fold,
            "test_as_of": self.test_as_of,
            "train_through": self.train_through,
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "params": self.params,
        }


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    threshold: float
    feature_names: list[str]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[INPUT_COLUMNS])[:, 1]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.pipeline.named_steps["prepare"].transform(frame[INPUT_COLUMNS])
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        return np.asarray(matrix, dtype=float)


def walk_forward_folds(dates: list[pd.Timestamp], n_splits: int) -> list[tuple[list, list]]:
    ordered = sorted(pd.Timestamp(value) for value in dates)
    unique = list(dict.fromkeys(ordered))
    if len(unique) < n_splits + 2:
        raise ValueError(
            f"Need at least {n_splits + 2} snapshot dates for {n_splits} walk-forward folds, "
            f"found {len(unique)}."
        )
    folds = []
    for test_date in unique[-n_splits:]:
        train_dates = [value for value in unique if value < test_date]
        folds.append((train_dates, [test_date]))
    return folds


def compare_models(
    frame: pd.DataFrame,
    n_splits: int = 3,
    param_grid: list[dict] | None = None,
    seed: int = 7,
    threshold: float = THRESHOLD,
) -> dict:
    labeled = frame.dropna(subset=[LABEL_COLUMN]).copy()
    labeled[LABEL_COLUMN] = labeled[LABEL_COLUMN].astype(int)
    dates = sorted(labeled["as_of"].unique())
    folds = walk_forward_folds(list(dates), n_splits=n_splits)
    grid = param_grid or XGB_PARAM_GRID
    results: list[FoldResult] = []

    for fold_id, (train_dates, test_dates) in enumerate(folds, start=1):
        train = labeled[labeled["as_of"].isin(train_dates)]
        test = labeled[labeled["as_of"].isin(test_dates)]
        _require_both_classes(train[LABEL_COLUMN], f"train fold {fold_id}")
        _require_both_classes(test[LABEL_COLUMN], f"test fold {fold_id}")

        log_model = _fit_logistic(train, seed)
        xgb_params = _select_xgb_params(train, grid, seed)
        xgb_model = _fit_xgboost(train, xgb_params, seed)
        train_through = max(train_dates).strftime("%Y-%m-%d")
        test_as_of = test_dates[0].strftime("%Y-%m-%d")

        for name, model, params in (
            ("logistic_regression", log_model, {}),
            ("xgboost", xgb_model, xgb_params),
        ):
            scores = classification_scores(
                test[LABEL_COLUMN].to_numpy(),
                model.predict_proba(test),
                threshold,
            )
            results.append(
                FoldResult(
                    model=name,
                    fold=fold_id,
                    test_as_of=test_as_of,
                    train_through=train_through,
                    roc_auc=scores["roc_auc"],
                    precision=scores["precision"],
                    recall=scores["recall"],
                    n_train=int(len(train)),
                    n_test=int(len(test)),
                    params=params,
                )
            )

    summary = _summarize(results)
    champion_name = _champion(summary)
    final_params = _select_xgb_params(labeled, grid, seed) if champion_name == "xgboost" else {}
    if champion_name == "xgboost":
        champion = _fit_xgboost(labeled, final_params, seed)
    else:
        champion = _fit_logistic(labeled, seed)

    return {
        "folds": [result.as_dict() for result in results],
        "summary": summary,
        "champion": champion_name,
        "champion_params": final_params,
        "threshold": threshold,
        "model": champion,
    }


def classification_scores(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def format_report(report: dict) -> str:
    header = (
        f"{'model':<22} {'fold':>4} {'as_of':<12} "
        f"{'auc':>7} {'prec':>7} {'recall':>7} {'n':>6}"
    )
    lines = [header, "-" * len(header)]
    for row in report["folds"]:
        lines.append(
            f"{row['model']:<22} {row['fold']:>4} {row['test_as_of']:<12} "
            f"{row['roc_auc']:>7.3f} {row['precision']:>7.3f} "
            f"{row['recall']:>7.3f} {row['n_test']:>6}"
        )
    lines.append("")
    lines.append(f"champion: {report['champion']}")
    for name, metrics in report["summary"].items():
        lines.append(
            f"{name}: auc={metrics['roc_auc']:.3f} "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f}"
        )
    return "\n".join(lines)


def _fit_logistic(frame: pd.DataFrame, seed: int) -> TrainedModel:
    pipeline = Pipeline(
        [
            ("prepare", _preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(frame[INPUT_COLUMNS], frame[LABEL_COLUMN].astype(int))
    feature_names = list(pipeline.named_steps["prepare"].get_feature_names_out())
    return TrainedModel("logistic_regression", pipeline, THRESHOLD, feature_names)


def _fit_xgboost(frame: pd.DataFrame, params: dict, seed: int) -> TrainedModel:
    y = frame[LABEL_COLUMN].astype(int)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    scale = float(negatives / positives) if positives else 1.0
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        scale_pos_weight=scale,
        subsample=0.9,
        colsample_bytree=0.9,
        **params,
    )
    pipeline = Pipeline([("prepare", _preprocessor()), ("model", model)])
    pipeline.fit(frame[INPUT_COLUMNS], y)
    feature_names = list(pipeline.named_steps["prepare"].get_feature_names_out())
    return TrainedModel("xgboost", pipeline, THRESHOLD, feature_names)


def _select_xgb_params(train: pd.DataFrame, param_grid: list[dict], seed: int) -> dict:
    dates = sorted(train["as_of"].unique())
    if len(dates) < 2:
        return dict(param_grid[0])
    inner_train = train[train["as_of"] < dates[-1]]
    inner_val = train[train["as_of"] == dates[-1]]
    if inner_train[LABEL_COLUMN].nunique() < 2 or inner_val[LABEL_COLUMN].nunique() < 2:
        return dict(param_grid[0])

    best_params = dict(param_grid[0])
    best_auc = -1.0
    for params in param_grid:
        candidate = _fit_xgboost(inner_train, params, seed)
        auc = roc_auc_score(inner_val[LABEL_COLUMN].astype(int), candidate.predict_proba(inner_val))
        if auc > best_auc:
            best_auc = float(auc)
            best_params = dict(params)
    return best_params


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _summarize(results: list[FoldResult]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name in ("logistic_regression", "xgboost"):
        rows = [result for result in results if result.model == name]
        summary[name] = {
            "roc_auc": float(np.mean([row.roc_auc for row in rows])),
            "precision": float(np.mean([row.precision for row in rows])),
            "recall": float(np.mean([row.recall for row in rows])),
        }
    return summary


def _champion(summary: dict[str, dict[str, float]]) -> str:
    ranked = sorted(
        summary.items(),
        key=lambda item: (item[1]["roc_auc"], item[1]["recall"]),
        reverse=True,
    )
    return ranked[0][0]


def _require_both_classes(labels: pd.Series, where: str) -> None:
    if labels.nunique() < 2:
        raise ValueError(f"{where} has a single class; walk-forward metrics would be undefined.")
