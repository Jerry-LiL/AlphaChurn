"""ROC-AUC, precision, and recall for a fixed decision threshold."""

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def safe_roc_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or np.unique(y_true).size < 2 or np.isnan(y_score).any():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def precision_recall(y_true, y_score, threshold: float) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0 or np.isnan(y_score).any():
        return float("nan"), float("nan")
    predicted = (y_score >= threshold).astype(int)
    precision = float(precision_score(y_true, predicted, zero_division=0))
    recall = float(recall_score(y_true, predicted, zero_division=0))
    return precision, recall


def classification_scores(y_true, y_score, threshold: float) -> dict[str, float]:
    precision, recall = precision_recall(y_true, y_score, threshold)
    return {
        "roc_auc": safe_roc_auc(y_true, y_score),
        "precision": precision,
        "recall": recall,
    }
