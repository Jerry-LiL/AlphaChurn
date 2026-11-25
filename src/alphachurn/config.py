"""Paths, calendar, and environment settings."""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Synthetic history covers enough months for 120-day lags before the first
# labeled snapshot, and 30 days after the last labeled snapshot.
DATA_START = "2023-10-01"
DATA_END = "2024-12-31"
LABELED_AS_OF_START = "2024-02-29"
LABELED_AS_OF_END = "2024-11-30"
DEFAULT_SCORE_AS_OF = "2024-12-31"
LABEL_HORIZON_DAYS = 30

PURCHASE_TYPES = ("subscription", "addon")
EVENT_TYPES = ("login", "support_ticket", "payment_failed")
TXN_TYPES = ("subscription", "addon", "refund")


def month_ends(start: str, end: str) -> list[pd.Timestamp]:
    """Month-end timestamps from `start` through `end`, inclusive."""
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    cursor = start_ts.replace(day=1)
    ends: list[pd.Timestamp] = []
    while cursor <= end_ts:
        month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
        if start_ts <= month_end <= end_ts:
            ends.append(month_end)
        cursor = (cursor + pd.offsets.MonthBegin(1)).normalize()
    return ends


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    """First-of-month timestamps covering `start` through `end`."""
    cursor = pd.Timestamp(start).normalize().replace(day=1)
    end_ts = pd.Timestamp(end).normalize()
    starts: list[pd.Timestamp] = []
    while cursor <= end_ts:
        starts.append(cursor)
        cursor = (cursor + pd.offsets.MonthBegin(1)).normalize()
    return starts


def labeled_as_ofs() -> list[pd.Timestamp]:
    return month_ends(LABELED_AS_OF_START, LABELED_AS_OF_END)


@dataclass(frozen=True)
class Settings:
    db_path: Path
    artifact_dir: Path
    seed: int = 42
    n_customers: int = 500
    threshold: float = 0.5
    min_train_periods: int = 4

    def resolve(self) -> "Settings":
        """Resolve relative paths from the current working directory."""
        return Settings(
            db_path=self.db_path.expanduser().resolve(),
            artifact_dir=self.artifact_dir.expanduser().resolve(),
            seed=self.seed,
            n_customers=self.n_customers,
            threshold=self.threshold,
            min_train_periods=self.min_train_periods,
        )


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    return Settings(
        db_path=Path(os.getenv("ALPHACHURN_DB_PATH", "data/alphachurn.db")),
        artifact_dir=Path(os.getenv("ALPHACHURN_ARTIFACT_DIR", "artifacts")),
        seed=int(os.getenv("ALPHACHURN_SEED", "42")),
        n_customers=int(os.getenv("ALPHACHURN_N_CUSTOMERS", "500")),
        threshold=float(os.getenv("ALPHACHURN_THRESHOLD", "0.5")),
        min_train_periods=int(os.getenv("ALPHACHURN_MIN_TRAIN_PERIODS", "4")),
    ).resolve()
