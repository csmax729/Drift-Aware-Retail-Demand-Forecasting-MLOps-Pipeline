# Drift-aware retail demand forecasting

This project is a local, reproducible MLOps learning project. It replays historical retail transactions as if they became available in weekly batches, validates them, builds time-aware features, trains and evaluates demand forecasts, and monitors for changes. It is **not a live data feed** and does not represent a deployed enterprise inventory system.

## Forecasting contract

- **Grain and horizon:** one forecast per `StockCode` for the calendar week beginning at the Monday cutoff, covering Monday through Sunday.
- **Target:** gross units from positive-quantity, non-cancelled transactions. Returns and cancellations are quarantined and reported separately; this is not net demand.
- **Information available:** a replay at `as_of` sees transactions with `InvoiceDate` strictly before that Monday at 00:00. The forecast's actual is unknown until the week has completed and a later replay makes it available.
- **Features:** demand lags of 1, 2 and 4 weeks, a four-week rolling mean shifted by one week, and calendar fields. A product is zero-filled only after its first observed complete-week sale; new products without sufficient history have no fully populated lag features.
- **Metrics:** mean absolute error (MAE) and weighted absolute percentage error (WAPE: sum of absolute errors divided by sum of actual demand). WAPE is undefined when total actual demand is zero; report it as unavailable rather than dividing by zero. Compare with a previous-week baseline and use chronological evaluation, not random splits.

The first partial calendar week in the extract is excluded. Stop replay at the Monday beginning the source's final calendar week: a partial final week cannot establish a full week's actual. UCI data ends on Friday 2011-12-09, so the latest supported cutoff is **2011-12-05**. For the synthetic extract below it is **2023-10-30**.

## Get started

Use Python 3.10–3.13 and run commands from the project root. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` before the same install command. If your system cannot activate the environment, invoke its Python executable directly.

Create a small offline dataset and run a replay:

```powershell
python scripts/make_demo_data.py --include-quality-issues
python -m retail_forecasting.flow run --source data/raw/synthetic_online_retail.csv --as-of 2023-10-23 --data-dir data/demo
python -m retail_forecasting.flow run --source data/raw/synthetic_online_retail.csv --as-of 2023-10-30 --data-dir data/demo
```

The demo data is **fictional**, generated locally with a fixed seed and four clearly named synthetic products over 44 weeks. `--include-quality-issues` adds a duplicate row, a return and a missing product code to exercise quarantine; omit it for a clean extract. Change the output location with `--output` and the seed with `--seed`.

Run the tests with:

```powershell
python -m pytest
```

The replay produces local outputs under `data/demo/runs/<cutoff>/`. Inspect `quality.json`, `quarantine.csv`, `weekly.csv`, `features.csv`, `drift.json`, `performance.json`, `alerts.json`, `forecast.csv`, and `manifest.json`. Alerts are local report entries; no email or messaging service is configured. When a candidate is evaluated, `backtest.csv` and `evaluation.json` record the comparison. Forecasts without enough lag history use the previous week's demand. If no candidate passes the initial gate, the whole forecast uses the previous-week baseline. A successful rerun of the same cutoff and source checksum returns its existing manifest; use a separate `--data-dir` for a different source or to replay history from scratch.

## Use the real public dataset

Download **Online Retail.xlsx** from the [UCI Online Retail dataset page](https://archive.ics.uci.edu/dataset/352/online-retail) into `data/raw/uci/`. The extract describes transactions from 2010-12-01 to 2011-12-09; review UCI's attribution and dataset terms. Citation: Chen, D. (2015), *Online Retail*, UCI Machine Learning Repository, [doi:10.24432/C5BW33](https://doi.org/10.24432/C5BW33), licensed CC BY 4.0.

```powershell
python -m retail_forecasting.flow run --source "data/raw/uci/Online Retail.xlsx" --as-of 2011-12-05 --data-dir data/uci-run
```

Start with a sufficiently late Monday to give the model enough training history. Earlier cutoffs simulate what could have been known then, not actual original arrival delays. Keep the source file unchanged between replays if you want comparable lineage hashes.

UCI does not provide product categories, inventory levels, stockouts, promotion plans, actual arrival timestamps or a live stream. `StockCode` can include non-merchandise service or adjustment codes, so apply a validated merchandise mapping before interpreting the output as an inventory plan. Sales are not necessarily unconstrained demand, and zero-filled weeks represent no recorded sales, not proof of availability. Large wholesale orders can strongly affect error. This is a small, historical extract and its patterns should not be assumed to generalize to another retailer.

## What to inspect and learn

1. **Ingest and replay.** Record a source checksum and cutoff. The same source and cutoff should give the same batch identity; reruns should not add duplicate forecasts or retraining events.
2. **Validate.** Required fields are `InvoiceNo`, `StockCode`, `Quantity`, and `InvoiceDate`. Exact duplicate rows are quarantined; invoice and stock code alone are not treated as a unique line key. Rows with missing identifiers, invalid timestamps or quantities, zero quantities, and returns/cancellations are given reasons. Undated rows cannot be assigned to a replay week and are reported separately at every cutoff.
3. **Build features and evaluate.** Check that each feature row uses only weeks earlier than its target and that the test interval follows training. Contrast model errors with the previous-week baseline; a sophisticated model is not automatically better.
4. **Monitor and decide.** A feature distribution change is a drift signal, not proof that model accuracy degraded. Forecast error can only be calculated when actuals are available. Inspect findings and validation metrics before promoting a candidate; preserve the previous model so a rejected candidate does not replace it.
5. **Reproduce and challenge.** Run the same cutoff twice, try a missing source or malformed schema, introduce a shifted input distribution, and compare an intentionally worse candidate. Inspect run states, findings and promotion decisions instead of treating a completed run as proof of model quality.

The data layer is in `src/retail_forecasting/data.py`, model training and evaluation in `modeling.py`, drift and performance checks in `monitoring.py`, and Prefect orchestration in `flow.py`. The model is a histogram gradient booster with an ordinal product encoding; the arbitrary ordering of product codes is a modeling limitation. `tests/` exercises pipeline behavior; `scripts/make_demo_data.py` provides the offline fixture.

The initial run requires at least 16 complete feature weeks: four weeks of lag history, eight training weeks and four backtest weeks. Exact duplicates and returns are quarantined; other invalid rows cause the quality gate to fail when they exceed 5% of visible rows. PSI uses four recent weeks against the promoted model's saved reference features (threshold 0.2, at least 10 rows); this is a demonstration threshold, not a calibrated business alert. Performance uses the last four completed forecast batches and signals degradation when WAPE exceeds 0.5. Zero total actual demand makes WAPE unavailable. A drift signal or degraded performance triggers candidate evaluation, not automatic promotion. Promotion requires candidate MAE no worse than the baseline and champion on the same eligible held-out weeks.

The run manifest records the source checksum, feature checksum, cutoff, Git commit when available, model version and MLflow run ID. Commit your code to make the Git revision meaningful; a dirty working tree is not captured by the commit SHA. The dependency ranges are not a complete lockfile, so capture the resolved environment before requiring bit-for-bit reproduction. Models use cloudpickle for compatibility; only load artifacts from a trusted tracking store because pickle deserialization can execute code. The local implementation is intended for a single worker: completed runs are idempotent, and a pending record allows recovery after evaluation, but there is no distributed lock or atomic transaction across MLflow and local files. An interruption while registering a model can leave an extra version. The source is assumed complete up to its final observed timestamp; a static extract cannot prove that an intermediate partition is not missing.

To change the champion to a previously registered version, inspect the MLflow registry first, then run `python -m retail_forecasting.flow rollback --version VERSION --data-dir data/demo`. Previous run artifacts are not rewritten. On the next replay the restored version is evaluated using its own reference data. Preserve the local run folders together with the tracking database and artifacts.

## Local services

The default development setup uses local files and an SQLite-backed MLflow store. View it with `mlflow server --backend-store-uri sqlite:///data/demo/mlflow.db` and visit `http://localhost:5000`. Docker Compose is optional: `docker compose up --build -d prefect mlflow` starts the Prefect UI on port 4200 and MLflow on port 5000. The services persist state in Docker volumes. The optional `runner` profile mounts `./data` and connects to both services. These services are for local learning, not hardened production deployments.

For a local Prefect server, set `PREFECT_API_URL` to `http://127.0.0.1:4200/api`, then run `python -m retail_forecasting.flow serve --source data/raw/synthetic_online_retail.csv --as-of 2023-10-23 --data-dir data/demo`. The process must stay running. Each scheduled invocation advances by one historical week from the latest completed manifest. Once the extract's coverage ends, the next run fails clearly; pause the deployment. The scheduler does not create live data. Model versions, validation outcomes and evaluation results can be checked in the run artifacts and MLflow UI.
