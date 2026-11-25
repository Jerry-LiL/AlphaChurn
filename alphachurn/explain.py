"""SHAP explanations on the matrix the saved model actually scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphachurn.modeling import TrainedModel


def shap_values(model: TrainedModel, frame: pd.DataFrame) -> np.ndarray:
    matrix = model.transform(frame)
    import shap

    estimator = model.pipeline.named_steps["model"]
    if model.name == "xgboost":
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(matrix)
    else:
        background = matrix
        if len(background) > 200:
            background = background[:200]
        explainer = shap.LinearExplainer(estimator, background)
        values = explainer.shap_values(matrix)

    if isinstance(values, list):
        array = np.asarray(values[-1])
    else:
        array = np.asarray(values)
        if array.ndim == 3:
            array = array[:, :, -1]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != matrix.shape[1] and array.shape[0] == matrix.shape[1]:
        array = array.T
    if array.shape != (matrix.shape[0], matrix.shape[1]):
        raise ValueError(
            f"SHAP shape {array.shape} does not match scores {matrix.shape[0]} x {matrix.shape[1]}"
        )
    return array


def mean_abs_shap(model: TrainedModel, frame: pd.DataFrame) -> list[dict]:
    values = shap_values(model, frame)
    means = np.mean(np.abs(values), axis=0)
    order = np.argsort(-means)
    return [
        {"feature": model.feature_names[int(index)], "mean_abs_shap": float(means[index])}
        for index in order
    ]


def top_reasons(model: TrainedModel, frame: pd.DataFrame, k: int = 5) -> list[list[dict]]:
    values = shap_values(model, frame)
    reasons: list[list[dict]] = []
    for row in values:
        order = np.argsort(-np.abs(row))[:k]
        reasons.append(
            [
                {"feature": model.feature_names[int(index)], "shap": float(row[index])}
                for index in order
            ]
        )
    return reasons
