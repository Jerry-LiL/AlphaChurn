"""Paths and column contracts shared by training and scoring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HORIZON_DAYS = 30
HISTORY_DAYS = 90
THRESHOLD = 0.5
RANDOM_SEED = 7

NUMERIC_FEATURES = [
    "tenure_days",
    "rfm_recency_days",
    "rfm_frequency_90d",
    "rfm_monetary_90d",
    "frequency_7d",
    "frequency_30d",
    "frequency_90d",
    "monetary_7d",
    "monetary_30d",
    "monetary_90d",
    "lag1_frequency_30d",
    "lag2_frequency_30d",
    "lag1_monetary_30d",
    "lag2_monetary_30d",
    "roll_ratio_frequency_7_30",
    "roll_ratio_frequency_30_90",
    "roll_ratio_monetary_7_30",
    "roll_ratio_monetary_30_90",
    "ticket_count_30d",
    "purchase_count_30d",
    "login_count_30d",
]

CATEGORICAL_FEATURES = ["plan", "region", "segment"]

# (name suffix, farther edge in days, nearer edge in days)
# A window is (as_of - farther, as_of - nearer].
WINDOWS = {
    "7d": (7, 0),
    "30d": (30, 0),
    "90d": (90, 0),
    "lag1_30d": (60, 30),
    "lag2_30d": (90, 60),
}

XGB_PARAM_GRID = [
    {"max_depth": 2, "learning_rate": 0.08, "min_child_weight": 1, "n_estimators": 60},
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 2, "n_estimators": 90},
    {"max_depth": 4, "learning_rate": 0.1, "min_child_weight": 1, "n_estimators": 40},
]


@dataclass(frozen=True)
class Paths:
    db: Path
    artifacts: Path

    @classmethod
    def from_env(cls, root: Path | None = None) -> Paths:
        root = root or Path.cwd()
        db = Path(os.environ.get("ALPHACHURN_DB", root / "data" / "churn.sqlite"))
        artifacts = Path(os.environ.get("ALPHACHURN_ARTIFACTS", root / "artifacts"))
        return cls(db=db, artifacts=artifacts)
