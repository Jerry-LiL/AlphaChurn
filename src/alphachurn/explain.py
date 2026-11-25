"""SHAP explanations for the fitted XGBoost model.

Values are on the log-odds margin. A positive value pushes the prediction
toward churn. Local reasons are the features with the largest absolute value
for that customer.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def explain_matrix(pipeline: Pipeline, features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    if not isinstance(model, XGBClassifier):
        raise TypeError("SHAP explanations are produced for the XGBoost model.")
    transformed = np.asarray(preprocess.transform(features), dtype=float)
    names = [str(name) for name in preprocess.get_feature_names_out()]
    explainer = _tree_explainer(model)
    values = _as_matrix(explainer.shap_values(transformed))
    if values.shape != transformed.shape:
        raise RuntimeError(f"Unexpected SHAP shape {values.shape} for matrix {transformed.shape}.")
    return values, names


def global_importance(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    importance = np.mean(np.abs(values), axis=0)
    frame = pd.DataFrame({"feature": names, "mean_abs_shap": importance})
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_reasons(values: np.ndarray, names: list[str], k: int = 3) -> pd.DataFrame:
    width = min(k, values.shape[1])
    order = np.argsort(np.abs(values), axis=1)[:, ::-1][:, :width]
    columns: dict[str, list] = {}
    row_index = np.arange(values.shape[0])
    for rank in range(width):
        chosen = order[:, rank]
        columns[f"reason_{rank + 1}_feature"] = [names[item] for item in chosen]
        columns[f"reason_{rank + 1}_shap"] = values[row_index, chosen]
    return pd.DataFrame(columns)


def save_importance_plot(importance: pd.DataFrame, path: Path, top_n: int = 12) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(part["feature"], part["mean_abs_shap"], color="#3d5a4c")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Latest labeled snapshot")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _tree_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    # XGBoost stores base_score as a one-element list. TreeExplainer wants a float.
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    learner = config["learner"]["learner_model_param"]
    base_score = learner.get("base_score")
    if isinstance(base_score, str) and base_score.startswith("["):
        learner["base_score"] = str(float(base_score.strip("[]")))
        booster.load_config(json.dumps(config))
    return shap.TreeExplainer(booster)


def _as_matrix(values) -> np.ndarray:
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    return array
