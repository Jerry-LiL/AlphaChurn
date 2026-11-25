import numpy as np
import pandas as pd

from alphachurn.cli import main
from alphachurn.extract import extract_sources
from alphachurn.score import score_as_of


def test_score_file_ranks_active_customers(trained):
    settings, _ = trained
    scored = score_as_of(settings, "2024-12-31")
    assert not scored.empty
    assert scored["as_of_date"].eq("2024-12-31").all()
    assert scored["churn_probability"].between(0, 1).all()
    expected = (scored["churn_probability"] >= settings.threshold).astype(int)
    assert scored["predicted_churn"].eq(expected).all()
    assert scored["customer_id"].is_unique
    probabilities = scored["churn_probability"].to_numpy()
    assert np.all(probabilities[:-1] + 1e-12 >= probabilities[1:])
    for column in (
        "reason_1_feature",
        "reason_1_shap",
        "reason_2_feature",
        "reason_2_shap",
        "reason_3_feature",
        "reason_3_shap",
    ):
        assert scored[column].notna().all()
    assert np.isfinite(scored["reason_1_shap"]).all()

    tables = extract_sources(settings.db_path, "2024-12-31")
    gone = tables.subscriptions.loc[
        tables.subscriptions["end_date"].notna()
        & (tables.subscriptions["end_date"] <= pd.Timestamp("2024-12-31")),
        "customer_id",
    ]
    assert set(gone).isdisjoint(set(scored["customer_id"]))
    saved = pd.read_csv(settings.artifact_dir / "scores.csv")
    assert len(saved) == len(scored)


def test_generate_and_score_commands(tmp_path, monkeypatch, trained):
    monkeypatch.setenv("ALPHACHURN_DB_PATH", str(tmp_path / "book.db"))
    monkeypatch.setenv("ALPHACHURN_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    assert main(["generate", "--customers", "25", "--seed", "2"]) == 0
    assert (tmp_path / "book.db").exists()

    settings, _ = trained
    monkeypatch.setenv("ALPHACHURN_DB_PATH", str(settings.db_path))
    monkeypatch.setenv("ALPHACHURN_ARTIFACT_DIR", str(settings.artifact_dir))
    monkeypatch.setenv("ALPHACHURN_THRESHOLD", "0.5")
    assert main(["score", "--as-of", "2024-12-31"]) == 0
    assert (settings.artifact_dir / "scores.csv").exists()


def test_train_command_reports_a_missing_database(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHACHURN_DB_PATH", str(tmp_path / "missing.db"))
    monkeypatch.setenv("ALPHACHURN_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    assert main(["train"]) == 1
