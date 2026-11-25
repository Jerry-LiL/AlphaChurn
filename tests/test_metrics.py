import numpy as np

from alphachurn.metrics import precision_recall, safe_roc_auc


def test_perfect_ranking_scores_perfectly():
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.8, 0.9]
    assert safe_roc_auc(y_true, scores) == 1.0
    precision, recall = precision_recall(y_true, scores, threshold=0.5)
    assert precision == 1.0
    assert recall == 1.0


def test_mixed_predictions_at_the_threshold():
    y_true = [0, 1, 0, 1]
    scores = [0.2, 0.4, 0.6, 0.8]
    precision, recall = precision_recall(y_true, scores, threshold=0.5)
    assert precision == 0.5
    assert recall == 0.5


def test_auc_is_undefined_for_a_single_class():
    assert np.isnan(safe_roc_auc([0, 0, 0], [0.2, 0.4, 0.6]))
