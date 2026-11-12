"""Historical Online Retail replay, quality checks, and leakage-safe features.

The target is gross positive units per StockCode and calendar week. Returns and
cancellations are reported separately; they are not subtracted from demand.
The source is a static extract, so cutoffs simulate arrival, not actual arrival times.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = ("InvoiceNo", "StockCode", "Quantity", "InvoiceDate")
SCHEMA_VERSION = "online-retail-v1"
FEATURE_COLUMNS = ["lag_1", "lag_2", "lag_4", "rolling_mean_4", "week_of_year", "month"]


def source_checksum(path: str | Path) -> str:
    """Return the SHA-256 of the exact source file for reproducible lineage."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(path: str | Path) -> pd.DataFrame:
    """Load a UCI Online Retail Excel or CSV extract, preserving identifiers."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Retail source file does not exist: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        frame = pd.read_excel(path, dtype={"InvoiceNo": "string", "StockCode": "string"})
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"InvoiceNo": "string", "StockCode": "string"})
    else:
        raise ValueError(f"Unsupported source format: {path.suffix}")
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return frame


def _cutoff(as_of: str | pd.Timestamp) -> pd.Timestamp:
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None or cutoff != cutoff.normalize() or cutoff.weekday() != 0:
        raise ValueError("as_of must be a timezone-naive Monday at midnight")
    return cutoff


def prepare_transactions(
    frame: pd.DataFrame, as_of: str | pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate records visible strictly before a Monday cutoff.

    Exact duplicate rows after the first are quarantined. Invalid timestamps
    cannot be assigned to a replay batch and are always reported separately.
    Reasons are mutually exclusive, in the order checked below. Invoice-item
    pairs alone are not unique in this dataset and are not used as a key.
    """
    cutoff = _cutoff(as_of)
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = frame.copy().reset_index(drop=True)
    parsed = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        raise ValueError("InvoiceDate must be timezone-naive")
    # Undated rows cannot be placed on the timeline; they are never features.
    data = data.loc[parsed.isna() | (parsed < cutoff)].copy()
    data["InvoiceDate"] = parsed.loc[data.index]
    data["InvoiceNo"] = data["InvoiceNo"].astype("string").str.strip()
    data["StockCode"] = data["StockCode"].astype("string").str.strip()
    quantity = pd.to_numeric(data["Quantity"], errors="coerce")
    reason = pd.Series("", index=data.index, dtype="string")

    def mark(condition: pd.Series, label: str) -> None:
        reason.loc[(reason == "") & condition.fillna(True)] = label

    mark(data["InvoiceDate"].isna(), "invalid_timestamp")
    mark(data["InvoiceNo"].isna() | data["InvoiceNo"].eq(""), "missing_invoice")
    mark(data["StockCode"].isna() | data["StockCode"].eq(""), "missing_stock_code")
    mark(quantity.isna() | ~np.isfinite(quantity), "invalid_quantity")
    mark(quantity.notna() & (quantity % 1 != 0), "non_integer_quantity")
    mark(data["InvoiceNo"].str.upper().str.startswith("C", na=False) | (quantity < 0), "return_or_cancellation")
    mark(quantity == 0, "zero_quantity")
    # Duplicate detection is based on the complete input row, not invoice-item.
    mark(data.duplicated(keep="first"), "exact_duplicate")
    data["Quantity"] = quantity
    good = data.loc[reason == ""].copy().reset_index(drop=True)
    bad = data.loc[reason != ""].copy()
    bad["reason"] = reason.loc[bad.index]
    bad = bad.reset_index(drop=True)
    report = {
        "as_of": cutoff.date().isoformat(),
        "schema_version": SCHEMA_VERSION,
        "visible_rows": int(len(data)),
        "accepted_rows": int(len(good)),
        "quarantined_rows": int(len(bad)),
        "reasons": {str(k): int(v) for k, v in bad["reason"].value_counts().items()},
        "event_date_min": data["InvoiceDate"].min().isoformat() if data["InvoiceDate"].notna().any() else None,
        "event_date_max": data["InvoiceDate"].max().isoformat() if data["InvoiceDate"].notna().any() else None,
        "missing_by_column": {str(col): int(data[col].isna().sum()) for col in data.columns},
        "note": "Undated rows are reported at every cutoff because their event time is unknown.",
    }
    return good, bad, report


def weekly_demand(clean: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Aggregate completed weeks; pad a product only after its first sale.

    If the extract starts midweek, its first partial week is excluded. At any
    cutoff, only products already seen in complete weeks are included.
    """
    cutoff = _cutoff(as_of)
    cols = ["week_start", "stock_code", "demand"]
    if clean.empty:
        return pd.DataFrame(columns=cols)
    data = clean.loc[clean["InvoiceDate"] < cutoff].copy()
    if data.empty:
        return pd.DataFrame(columns=cols)
    timestamps = pd.to_datetime(data["InvoiceDate"])
    data["week_start"] = timestamps.dt.normalize() - pd.to_timedelta(timestamps.dt.day_of_week, unit="D")
    first_day = timestamps.min().normalize()
    first_week = first_day - pd.Timedelta(days=first_day.weekday())
    if first_day.weekday() != 0:
        data = data.loc[data["week_start"] > first_week]
    if data.empty:
        return pd.DataFrame(columns=cols)
    grouped = data.groupby(["StockCode", "week_start"], as_index=False)["Quantity"].sum()
    groups = []
    end = cutoff - pd.Timedelta(weeks=1)
    for stock_code, group in grouped.groupby("StockCode", sort=True):
        dates = pd.date_range(group["week_start"].min(), end, freq="W-MON")
        series = group.set_index("week_start")["Quantity"].reindex(dates, fill_value=0)
        groups.append(pd.DataFrame({"week_start": dates, "stock_code": str(stock_code), "demand": series.to_numpy(dtype=float)}))
    return pd.concat(groups, ignore_index=True)[cols].sort_values(["stock_code", "week_start"]).reset_index(drop=True)


def make_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Build features for each target week using only earlier demand."""
    data = weekly.copy().sort_values(["stock_code", "week_start"]).reset_index(drop=True)
    data["week_start"] = pd.to_datetime(data["week_start"])
    grouped = data.groupby("stock_code", sort=False)["demand"]
    for lag in (1, 2, 4):
        data[f"lag_{lag}"] = grouped.shift(lag)
    data["rolling_mean_4"] = grouped.transform(lambda values: values.shift(1).rolling(4, min_periods=4).mean())
    data["week_of_year"] = data["week_start"].dt.isocalendar().week.astype(int)
    data["month"] = data["week_start"].dt.month
    return data


def make_prediction_features(weekly: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Create next-week feature rows with unknown, never fabricated, actuals."""
    cutoff = _cutoff(as_of)
    if weekly.empty:
        return make_features(weekly)
    data = weekly.copy()
    data["week_start"] = pd.to_datetime(data["week_start"])
    if data["week_start"].max() != cutoff - pd.Timedelta(weeks=1):
        raise ValueError("weekly demand must end in the last complete week before as_of")
    products = data.loc[data["week_start"] == data["week_start"].max(), "stock_code"].unique()
    future = pd.DataFrame({"week_start": cutoff, "stock_code": products, "demand": np.nan})
    features = make_features(pd.concat([data, future], ignore_index=True))
    return features.loc[features["week_start"] == cutoff].reset_index(drop=True)
