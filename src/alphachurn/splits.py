"""Expanding walk-forward splits.

The validation month is always later than every training month. Hyperparameters
are chosen on an earlier tuning month, which is then allowed into the training
window of later folds and is not reused as a test month.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_dates: tuple[pd.Timestamp, ...]
    valid_date: pd.Timestamp

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]


def _ordered(dates) -> tuple[pd.Timestamp, ...]:
    normalized = [pd.Timestamp(value).normalize() for value in dates]
    return tuple(dict.fromkeys(sorted(normalized)))


def walk_forward_folds(dates, min_train_periods: int) -> list[Fold]:
    ordered = _ordered(dates)
    if min_train_periods < 2:
        raise ValueError("min_train_periods must be at least 2.")
    if len(ordered) <= min_train_periods:
        raise ValueError(
            f"Need more than {min_train_periods} as-of dates for a held-out month, "
            f"found {len(ordered)}."
        )
    return [
        Fold(train_dates=ordered[:idx], valid_date=ordered[idx])
        for idx in range(min_train_periods, len(ordered))
    ]


def tuning_split(dates, min_train_periods: int) -> tuple[tuple[pd.Timestamp, ...], pd.Timestamp]:
    """Use the earliest snapshots to pick XGBoost parameters.

    The last of those snapshots is the tuning month. Walk-forward validation
    starts on the following month.
    """
    ordered = _ordered(dates)
    if min_train_periods < 2:
        raise ValueError("min_train_periods must be at least 2.")
    if len(ordered) <= min_train_periods:
        raise ValueError(
            f"Need more than {min_train_periods} as-of dates to tune and validate, "
            f"found {len(ordered)}."
        )
    tune_train = ordered[: min_train_periods - 1]
    tune_valid = ordered[min_train_periods - 1]
    return tune_train, tune_valid
