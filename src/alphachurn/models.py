"""Logistic regression baseline and the XGBoost model it is compared with."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from alphachurn.features import CATEGORICAL_FEATURES, LABEL, MODEL_FEATURES, NUMERIC_FEATURES
from alphachurn.metrics import safe_roc_auc

DEFAULT_XGB_GRID = (
    {
        "max_depth": 2,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "n_estimators": 80,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.08,
        "min_child_weight": 3,
        "n_estimators": 120,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "n_estimators": 140,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
)

LOGISTIC_PARAMS = {"C": 1.0, "class_weight": "balanced", "penalty": "l2"}


def feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray | None]:
    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise KeyError(f"Feature frame is missing columns: {missing}")
    features = frame.loc[:, list(MODEL_FEATURES)].copy()
    target = None
    if LABEL in frame.columns:
        target = frame[LABEL].to_numpy(dtype=int)
    return features, target


def fit_logistic(features: pd.DataFrame, target: np.ndarray, seed: int) -> Pipeline | None:
    if _single_class(target):
        return None
    pipeline = Pipeline(
        steps=[
            ("preprocess", _preprocessor(scale=True)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(features, target)
    return pipeline


def fit_xgboost(
    features: pd.DataFrame, target: np.ndarray, params: dict, seed: int
) -> Pipeline | None:
    if _single_class(target):
        return None
    target = np.asarray(target)
    positives = float(np.sum(target == 1))
    negatives = float(np.sum(target == 0))
    model = XGBClassifier(
        objective="binary:logistic",
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        eval_metric="auc",
        scale_pos_weight=negatives / positives,
        verbosity=0,
        **dict(params),
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", _preprocessor(scale=False)),
            ("model", model),
        ]
    )
    pipeline.fit(features, target)
    return pipeline


def tune_xgboost(frame, train_dates, valid_date, grid, seed: int) -> dict:
    """Pick a grid point using only the tuning window."""
    train = _rows_on(frame, train_dates)
    valid = _rows_on(frame, [valid_date])
    x_train, y_train = feature_matrix(train)
    x_valid, y_valid = feature_matrix(valid)
    best_params = dict(grid[0])
    best_auc = -np.inf
    for params in grid:
        model = fit_xgboost(x_train, y_train, params, seed)
        if model is None:
            continue
        scores = positive_scores(model, x_valid)
        auc = safe_roc_auc(y_valid, scores)
        if np.isnan(auc) or auc <= best_auc:
            continue
        best_auc = auc
        best_params = dict(params)
    return best_params


def positive_scores(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = pipeline.predict_proba(features)
    classes = list(pipeline.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(features), dtype=float)
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _rows_on(frame: pd.DataFrame, dates) -> pd.DataFrame:
    wanted = {pd.Timestamp(value).normalize() for value in dates}
    as_of = pd.to_datetime(frame["as_of_date"]).dt.normalize()
    return frame.loc[as_of.isin(wanted)].copy()


def _single_class(target) -> bool:
    return np.unique(np.asarray(target)).size < 2
