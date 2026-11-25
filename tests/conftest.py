import pytest

from alphachurn.config import Settings
from alphachurn.generate import generate_dataset
from alphachurn.train import run_training

SMALL_GRID = (
    {
        "max_depth": 2,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "n_estimators": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "n_estimators": 20,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    },
)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("alphachurn")
    settings = Settings(
        db_path=root / "alphachurn.db",
        artifact_dir=root / "artifacts",
        seed=7,
        n_customers=180,
        threshold=0.5,
        min_train_periods=4,
    )
    generate_dataset(settings)
    result = run_training(settings, param_grid=SMALL_GRID)
    return settings, result
