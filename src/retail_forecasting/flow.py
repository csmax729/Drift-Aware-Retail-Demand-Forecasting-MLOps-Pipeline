"""Prefect orchestration for a replay of historical transactions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import pandas as pd
from prefect import flow, task
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .data import load_source, source_checksum, prepare_transactions, weekly_demand, make_features, make_prediction_features
from .modeling import FEATURES, error_metrics, fit_model, predict, promotion_gate, rolling_backtest
from .monitoring import compare_drift, evaluate_forecasts

MODEL_NAME = "retail-weekly-demand"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


@task
def ingest(source: str, as_of: str):
    checksum = source_checksum(source)
    frame = load_source(source)
    if checksum != source_checksum(source):
        raise ValueError("Source changed while being read; retry with a stable file")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None or cutoff != cutoff.normalize() or cutoff.weekday() != 0:
        raise ValueError("as_of must be a timezone-naive Monday at midnight")
    dates = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    if dates.dropna().empty:
        raise ValueError("Source has no valid timestamps")
    final_week = dates.max().normalize() - pd.Timedelta(days=dates.max().weekday())
    if cutoff > final_week:
        raise ValueError(f"Cutoff exceeds source coverage; latest safe cutoff is {final_week.date()}")
    return frame, checksum


@task
def validate(frame, as_of: str, output: Path):
    good, quarantine, report = prepare_transactions(frame, as_of)
    quarantine.to_csv(output / "quarantine.csv", index=False)
    write_json(output / "quality.json", report)
    if good.empty:
        raise ValueError("No valid transactions available at this cutoff")
    # Returns and exact duplicates are expected and separately quarantined.
    invalid = sum(v for k, v in report["reasons"].items() if k not in {"return_or_cancellation", "exact_duplicate"})
    if invalid / max(report["visible_rows"], 1) > 0.05:
        raise ValueError("Data quality gate failed: more than 5% invalid rows")
    return good, report


@task
def build_features(clean, as_of: str, output: Path):
    weekly = weekly_demand(clean, as_of)
    features = make_features(weekly)
    future = make_prediction_features(weekly, as_of)
    weekly.to_csv(output / "weekly.csv", index=False)
    features.to_csv(output / "features.csv", index=False)
    return weekly, features, future


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_directory(root: Path, as_of: str, checksum: str) -> tuple[Path, dict | None]:
    """Identify a replay partition and reject changed source data or time travel.

    Local-only, single-worker idempotency: a completed manifest makes a rerun
    a no-op. It is not a distributed lock or an atomic cross-service commit.
    """
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / as_of
    manifest = path / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["source_sha256"] != checksum:
            raise ValueError("This cutoff was already processed with a different source checksum")
        return path, previous
    completed = sorted(p for p in runs.glob("*/manifest.json"))
    if completed:
        latest = json.loads(completed[-1].read_text(encoding="utf-8"))
        if latest["source_sha256"] != checksum:
            raise ValueError("Use a fresh output directory for a different source")
        if as_of <= latest["as_of"]:
            raise ValueError("Replay cutoffs must increase; use a fresh output directory for historical reruns")
    path.mkdir(parents=True, exist_ok=True)
    return path, None


@task
def monitor(features, weekly, root: Path, as_of: str, reference_path: str | None):
    complete = features.dropna(subset=FEATURES)
    recent_weeks = sorted(complete.week_start.unique())[-4:]
    current = complete[complete.week_start.isin(recent_weeks)]
    if reference_path:
        if not Path(reference_path).is_file():
            raise FileNotFoundError(f"Champion reference features unavailable: {reference_path}")
        reference = pd.read_csv(reference_path)
    else:
        reference = complete[~complete.week_start.isin(recent_weeks)]
    drift = compare_drift(reference, current, ["lag_1", "lag_2", "lag_4", "rolling_mean_4"], min_samples=10)
    paths = sorted((root / "runs").glob("*/forecast.csv"))
    prior = [p for p in paths if p.parent.name < as_of and (p.parent / "manifest.json").exists()][-4:]
    if prior:
        forecasts = pd.concat([pd.read_csv(p, dtype={"stock_code": str}) for p in prior], ignore_index=True)
        performance = evaluate_forecasts(forecasts, weekly, as_of=as_of)
    else:
        performance = {"status": "no_prior_forecasts", "performance_degraded": False}
    return drift, performance


def tracking_uri(root: Path) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///" + (root.resolve() / "mlflow.db").as_posix()


def get_champion(client, checksum: str):
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    if version.tags.get("source_sha256") != checksum:
        raise ValueError("MLflow champion belongs to a different source; use a separate tracking store")
    return version


@task
def train_and_evaluate(features, champion, checksum: str, as_of: str, quality: dict, output: Path):
    backtest = rolling_backtest(features)
    champion_mae = None
    if champion is not None:
        trained_until = pd.Timestamp(champion.tags["as_of"])
        backtest = backtest[backtest.week_start >= trained_until].copy()
        if backtest.empty:
            raise ValueError("No out-of-sample evaluation weeks after champion training cutoff")
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{champion.version}")
        compare_rows = features.merge(backtest[["stock_code", "week_start"]], on=["stock_code", "week_start"], how="inner")
        predictions = predict(model, compare_rows)
        paired = backtest.merge(predictions, on=["stock_code", "week_start"], suffixes=("", "_champion"), validate="one_to_one")
        champion_mae = error_metrics(paired.demand, paired.prediction_champion)["mae"]
    candidate = error_metrics(backtest.demand, backtest.prediction)
    baseline = error_metrics(backtest.demand, backtest.baseline_prediction)
    decision = promotion_gate(candidate["mae"], baseline["mae"], champion_mae)
    backtest.to_csv(output / "backtest.csv", index=False)
    details = {"candidate": candidate, "baseline": baseline, "champion_mae": champion_mae,
               "holdout_start": str(backtest.week_start.min()), "holdout_end": str(backtest.week_start.max()),
               "decision": decision.to_dict()}
    write_json(output / "evaluation.json", details)
    experiment_name = "retail-demand-replay"
    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        root = output.parent.parent
        artifact_location = (root / "mlartifacts").resolve().as_uri() if mlflow.get_tracking_uri().startswith("sqlite:") else None
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"candidate-{as_of}") as run:
        mlflow.log_params({"source_sha256": checksum, "as_of": as_of, "git_commit": git_commit(),
                           "feature_schema": "weekly-lags-v1", "feature_sha256": source_checksum(output / "features.csv"),
                           "model": "hist_gradient_boosting"})
        mlflow.log_metrics({"candidate_mae": candidate["mae"], "baseline_mae": baseline["mae"]})
        mlflow.log_dict(quality, "quality.json")
        mlflow.log_dict(details, "evaluation.json")
        if decision.accepted:
            model = fit_model(features)
            # The model is created by this pipeline; load only trusted local artifacts.
            # Explicit format avoids version-dependent MLflow serialization defaults.
            info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            version = mlflow.register_model(info.model_uri, MODEL_NAME)
            client = MlflowClient()
            for key, value in {"source_sha256": checksum, "as_of": as_of,
                               "reference_path": str((output / "features.csv").resolve())}.items():
                client.set_model_version_tag(MODEL_NAME, version.version, key, value)
            return str(version.version), details, run.info.run_id
        return (str(champion.version) if champion else None), details, run.info.run_id


@task
def generate_forecast(future, version: str | None, output: Path):
    complete = future.dropna(subset=FEATURES)
    if version and not complete.empty:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
        forecasts = predict(model, complete)
        forecasts["method"] = "registered_model"
    else:
        forecasts = pd.DataFrame(columns=["stock_code", "week_start", "prediction", "method"])
    remaining = future[~future.index.isin(complete.index)] if version else future
    if not remaining.empty:
        fallback = remaining[["stock_code", "week_start"]].copy()
        fallback["prediction"] = remaining["lag_1"].fillna(0).clip(lower=0)
        fallback["method"] = "previous_week"
        forecasts = pd.concat([forecasts, fallback], ignore_index=True)
    forecasts.to_csv(output / "forecast.csv", index=False)
    return len(forecasts)


@flow(name="retail-replay", log_prints=True)
def replay(source: str, as_of: str, data_dir: str = "data"):
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None or cutoff != cutoff.normalize() or cutoff.weekday() != 0:
        raise ValueError("as_of must be a timezone-naive Monday at midnight")
    as_of = cutoff.date().isoformat()
    root = Path(data_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    frame, checksum = ingest(source, as_of)
    output, prior = run_directory(root, as_of, checksum)
    if prior:
        print(f"Already completed {as_of}; returning existing manifest")
        return prior
    good, quality = validate(frame, as_of, output)
    weekly, features, future = build_features(good, as_of, output)
    mlflow.set_tracking_uri(tracking_uri(root))
    client = MlflowClient()
    pending_path = output / "pending.json"
    if pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        if pending["source_sha256"] != checksum:
            raise ValueError("Incomplete run has a different source checksum")
        version, decision, run_id, retrain = (pending[k] for k in ("version", "decision", "run_id", "retrain"))
        print("Resuming a previously evaluated run")
    else:
        champion = get_champion(client, checksum)
        if champion is not None and pd.Timestamp(champion.tags["as_of"]) >= cutoff:
            raise ValueError("Champion was trained at or after this cutoff; use an isolated tracking store")
        drift, performance = monitor(features, weekly, root, as_of, champion.tags.get("reference_path") if champion else None)
        write_json(output / "drift.json", drift)
        write_json(output / "performance.json", performance)
        retrain = champion is None or drift.get("drift_detected", False) or performance.get("performance_degraded", False)
        version = str(champion.version) if champion else None
        decision = None
        run_id = None
        if retrain:
            version, decision, run_id = train_and_evaluate(features, champion, checksum, as_of, quality, output)
        write_json(pending_path, {"source_sha256": checksum, "version": version, "decision": decision,
                                  "run_id": run_id, "retrain": retrain})
    count = generate_forecast(future, version, output)
    if decision is not None and decision["decision"]["accepted"]:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
    drift = json.loads((output / "drift.json").read_text(encoding="utf-8"))
    performance = json.loads((output / "performance.json").read_text(encoding="utf-8"))
    alerts = []
    if drift.get("drift_detected"):
        alerts.append("Feature drift detected")
    if performance.get("performance_degraded"):
        alerts.append("Forecast performance degraded")
    if decision is not None and not decision["decision"]["accepted"]:
        alerts.append("Candidate rejected; retaining existing model or baseline")
    write_json(output / "alerts.json", {"alerts": alerts})
    manifest = {"as_of": as_of, "source_sha256": checksum, "schema_version": quality["schema_version"],
                "git_commit": git_commit(), "feature_sha256": source_checksum(output / "features.csv"),
                "tracking_uri": tracking_uri(root), "model_version": version,
                "candidate_run_id": run_id, "retraining_reviewed": retrain, "forecast_rows": count}
    write_json(output / "manifest.json", manifest)
    print(f"Replay complete: {output}")
    return manifest


@flow(name="retail-scheduled-replay", log_prints=True)
def scheduled_replay(source: str, start_as_of: str, data_dir: str = "data"):
    completed = sorted((Path(data_dir) / "runs").glob("*/manifest.json"))
    cutoff = (pd.Timestamp(completed[-1].parent.name) + pd.Timedelta(weeks=1)).date().isoformat() if completed else start_as_of
    return replay(source, cutoff, data_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "serve"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--source", required=True)
        cmd.add_argument("--as-of", required=True, help="First Monday cutoff (YYYY-MM-DD)")
        cmd.add_argument("--data-dir", default="data")
        if name == "serve":
            cmd.add_argument("--cron", default="0 9 * * 1")
    rollback = sub.add_parser("rollback", help="Set champion alias to an earlier registered model version")
    rollback.add_argument("--version", required=True)
    rollback.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    if args.command == "run":
        replay(args.source, args.as_of, args.data_dir)
    elif args.command == "serve":
        scheduled_replay.serve(name="weekly-historical-replay", cron=args.cron,
                               parameters={"source": args.source, "start_as_of": args.as_of, "data_dir": args.data_dir})
    else:
        root = Path(args.data_dir).resolve()
        completed = sorted((root / "runs").glob("*/manifest.json"))
        if not completed:
            raise ValueError("No completed replays found in data directory")
        latest = json.loads(completed[-1].read_text(encoding="utf-8"))
        mlflow.set_tracking_uri(tracking_uri(root))
        client = MlflowClient()
        version = client.get_model_version(MODEL_NAME, args.version)
        if version.tags.get("source_sha256") != latest["source_sha256"] or version.tags.get("as_of", "") > latest["as_of"]:
            raise ValueError("Model version does not belong to this replay history")
        client.set_registered_model_alias(MODEL_NAME, "champion", args.version)
        print(f"Champion set to version {args.version}; earlier run artifacts remain unchanged")


if __name__ == "__main__":
    main()

