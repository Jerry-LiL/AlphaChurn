import pandas as pd
import pytest

from alphachurn.config import month_ends
from alphachurn.splits import tuning_split, walk_forward_folds


def test_labeled_month_ends_cover_the_training_calendar():
    dates = month_ends("2024-02-29", "2024-11-30")
    assert [ts.date().isoformat() for ts in dates] == [
        "2024-02-29",
        "2024-03-31",
        "2024-04-30",
        "2024-05-31",
        "2024-06-30",
        "2024-07-31",
        "2024-08-31",
        "2024-09-30",
        "2024-10-31",
        "2024-11-30",
    ]


def test_walk_forward_holds_out_the_next_month_only():
    dates = month_ends("2024-02-29", "2024-11-30")
    folds = walk_forward_folds(dates, min_train_periods=4)
    assert len(folds) == 6
    assert folds[0].valid_date == pd.Timestamp("2024-06-30")
    assert folds[0].train_end == pd.Timestamp("2024-05-31")
    assert folds[-1].valid_date == pd.Timestamp("2024-11-30")
    for fold in folds:
        assert all(train_date < fold.valid_date for train_date in fold.train_dates)

    tune_train, tune_valid = tuning_split(dates, min_train_periods=4)
    assert tune_valid == pd.Timestamp("2024-05-31")
    assert max(tune_train) < tune_valid
    assert tune_valid not in [fold.valid_date for fold in folds]


def test_walk_forward_requires_a_held_out_month():
    with pytest.raises(ValueError):
        walk_forward_folds(["2024-05-31", "2024-06-30"], min_train_periods=4)
