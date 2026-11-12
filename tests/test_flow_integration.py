"""Exercise the actual local Prefect flow and MLflow store, without external services."""
import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("prefect")
pytest.importorskip("mlflow")


@pytest.fixture
def isolated_services(tmp_path, monkeypatch):
    from prefect.settings import (PREFECT_API_URL, PREFECT_HOME, PREFECT_LOGGING_TO_API_ENABLED,
                                  PREFECT_SERVER_ANALYTICS_ENABLED, PREFECT_SERVER_EPHEMERAL_ENABLED,
                                  temporary_settings)
    root = tmp_path / "run-data"
    root.mkdir()
    uri = "sqlite:///" + (root / "mlflow.db").resolve().as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with temporary_settings(updates={PREFECT_API_URL: None, PREFECT_HOME: tmp_path / "prefect",
                                    PREFECT_SERVER_EPHEMERAL_ENABLED: True,
                                    PREFECT_SERVER_ANALYTICS_ENABLED: False,
                                    PREFECT_LOGGING_TO_API_ENABLED: False}):
        yield root, uri


def test_replay_is_idempotent_and_scores_when_actuals_arrive(tmp_path, isolated_services):
    from mlflow import MlflowClient
    from retail_forecasting.flow import replay

    root, uri = isolated_services
    source = tmp_path / "synthetic.csv"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_data.py"
    generate = runpy.run_path(str(script))["generate"]
    generate(source, include_quality_issues=True)

    first = replay(str(source), "2023-10-23", str(root))
    first_dir = root / "runs" / "2023-10-23"
    original_forecast = (first_dir / "forecast.csv").read_bytes()
    initial_performance = json.loads((first_dir / "performance.json").read_text())
    assert initial_performance["status"] == "no_prior_forecasts"
    assert first["forecast_rows"] == 4

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("retail-demand-replay")
    before = client.search_runs([experiment.experiment_id])
    assert len(before) == 1
    repeated = replay(str(source), "2023-10-23", str(root))
    assert repeated == first
    assert (first_dir / "forecast.csv").read_bytes() == original_forecast
    assert len(client.search_runs([experiment.experiment_id])) == len(before)

    second = replay(str(source), "2023-10-30", str(root))
    performance = json.loads((root / "runs" / "2023-10-30" / "performance.json").read_text())
    assert second["forecast_rows"] == 4
    assert performance["eligible_forecasts"] == 4
    assert performance["matched_forecasts"] == 4
    assert performance["missing_actuals"] == 0
    assert performance["mae"] is not None
    previous = pd.read_csv(first_dir / "forecast.csv")
    actuals = pd.read_csv(root / "runs" / "2023-10-30" / "weekly.csv")
    previous["week_start"] = pd.to_datetime(previous["week_start"])
    actuals["week_start"] = pd.to_datetime(actuals["week_start"])
    paired = previous.merge(actuals, on=["stock_code", "week_start"], validate="one_to_one")
    assert performance["mae"] == pytest.approx((paired["prediction"] - paired["demand"]).abs().mean())

    with pytest.raises(ValueError, match="Cutoff exceeds source coverage"):
        replay(str(source), "2023-11-06", str(root))
    assert not (root / "runs" / "2023-11-06" / "manifest.json").exists()


def test_failed_forecast_resumes_evaluated_candidate_without_duplicate_tracking(tmp_path, isolated_services, monkeypatch):
    import importlib
    from mlflow import MlflowClient

    module = importlib.import_module("retail_forecasting.flow")
    root, uri = isolated_services
    source = tmp_path / "synthetic.csv"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_data.py"
    runpy.run_path(str(script))["generate"](source)
    original_forecast = module.generate_forecast

    def fail_generation(*args, **kwargs):
        raise RuntimeError("simulated forecast generation failure")

    # Force acceptance in this recovery-specific test so it covers the
    # registered-version branch independently of synthetic model accuracy.
    from retail_forecasting.modeling import PromotionDecision
    monkeypatch.setattr(module, "promotion_gate", lambda candidate, baseline, champion: PromotionDecision(
        True, [], candidate, baseline, champion))
    monkeypatch.setattr(module, "generate_forecast", fail_generation)
    with pytest.raises(RuntimeError, match="simulated forecast generation failure"):
        module.replay(str(source), "2023-10-23", str(root))
    partition = root / "runs" / "2023-10-23"
    assert (partition / "pending.json").exists()
    assert not (partition / "manifest.json").exists()
    pending = json.loads((partition / "pending.json").read_text())
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("retail-demand-replay")
    runs_before = {run.info.run_id for run in client.search_runs([experiment.experiment_id])}
    versions_before = {str(version.version) for version in client.search_model_versions(f"name = '{module.MODEL_NAME}'")}
    assert len(runs_before) == 1
    assert len(versions_before) == 1
    assert pending["run_id"] in runs_before
    assert pending["version"] in versions_before

    monkeypatch.setattr(module, "generate_forecast", original_forecast)
    completed = module.replay(str(source), "2023-10-23", str(root))
    assert completed["candidate_run_id"] == pending["run_id"]
    assert completed["model_version"] == pending["version"]
    assert completed["forecast_rows"] == 4
    assert {run.info.run_id for run in client.search_runs([experiment.experiment_id])} == runs_before
    assert {str(version.version) for version in client.search_model_versions(f"name = '{module.MODEL_NAME}'")} == versions_before
    assert (partition / "manifest.json").exists()
