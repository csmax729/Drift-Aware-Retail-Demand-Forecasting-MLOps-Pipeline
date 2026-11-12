"""Reference feature drift and delayed forecast performance checks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compare_drift(reference, current, feature_columns, *, threshold=0.2, min_samples=20, bins=10):
    """Compare feature distributions using PSI with reference-derived bins."""
    if not feature_columns or threshold < 0 or min_samples < 1 or bins < 2:
        raise ValueError("Invalid drift configuration")
    missing = set(feature_columns) - (set(reference.columns) & set(current.columns))
    if missing:
        raise ValueError(f"Missing drift features: {sorted(missing)}")
    report = {"status": "insufficient_data", "drift_detected": False,
              "reference_rows": len(reference), "current_rows": len(current),
              "threshold": threshold, "min_samples": min_samples, "features": {}}
    if min(len(reference), len(current)) < min_samples:
        return report
    for name in feature_columns:
        ref, cur = reference[name], current[name]
        if pd.api.types.is_numeric_dtype(ref):
            r = pd.to_numeric(ref, errors="raise").to_numpy(dtype=float, na_value=np.nan)
            c = pd.to_numeric(cur, errors="raise").to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(r).any() or np.isinf(c).any():
                raise ValueError(f"Infinite values in {name}")
            valid = r[~np.isnan(r)]
            if not len(valid):
                report["features"][name] = {"status": "insufficient_reference", "psi": None, "drift_detected": False}
                continue
            edges = np.unique(np.quantile(valid, np.linspace(0, 1, bins + 1)))
            if len(edges) == 1:
                epsilon = max(1e-9, abs(edges[0]) * 1e-9)
                edges = np.array([-np.inf, edges[0] - epsilon, edges[0] + epsilon, np.inf])
            else:
                edges = np.r_[-np.inf, edges[1:-1], np.inf]
            rc = np.r_[np.histogram(valid, edges)[0], np.isnan(r).sum()]
            cc = np.r_[np.histogram(c[~np.isnan(c)], edges)[0], np.isnan(c).sum()]
        else:
            cats = list(pd.unique(ref.dropna()))
            rc = np.array([(ref == cat).sum() for cat in cats] + [0, ref.isna().sum()])
            cc = np.array([(cur == cat).sum() for cat in cats] + [(cur.notna() & ~cur.isin(cats)).sum(), cur.isna().sum()])
        rp = (rc + 0.5) / (rc.sum() + 0.5 * len(rc))
        cp = (cc + 0.5) / (cc.sum() + 0.5 * len(cc))
        score = float(np.sum((cp - rp) * np.log(cp / rp)))
        report["features"][name] = {"status": "ok", "psi": score, "drift_detected": score >= threshold,
                                    "reference_missing_rate": float(ref.isna().mean()), "current_missing_rate": float(cur.isna().mean())}
    report["drift_detected"] = any(v["drift_detected"] for v in report["features"].values())
    complete = all(v["status"] == "ok" for v in report["features"].values())
    report["status"] = "drift_detected" if report["drift_detected"] else "ok" if complete else "insufficient_data"
    return report


def evaluate_forecasts(forecasts, actuals, *, as_of, keys=("stock_code", "week_start"),
                       prediction_column="prediction", actual_column="demand", wape_threshold=0.5, min_matches=1):
    """Score completed Monday-start weeks only; WAPE is undefined for zero demand."""
    if "week_start" not in keys or wape_threshold < 0 or min_matches < 1:
        raise ValueError("Invalid performance configuration")
    forecast, actual = forecasts.copy(), actuals.copy()
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is not None:
        raise ValueError("as_of must be a timezone-naive timestamp")
    for frame, label, value in ((forecast, "forecasts", prediction_column), (actual, "actuals", actual_column)):
        missing = (set(keys) | {value}) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
        if frame[list(keys)].isna().any().any():
            raise ValueError(f"{label} contains null keys")
        frame["week_start"] = pd.to_datetime(frame["week_start"], errors="raise")
        if frame["week_start"].dt.tz is not None or ((frame["week_start"].dt.day_of_week != 0) | (frame["week_start"] != frame["week_start"].dt.normalize())).any():
            raise ValueError(f"{label} week_start must be timezone-naive Monday midnight")
        if frame.duplicated(list(keys)).any():
            raise ValueError(f"{label} contains duplicate keys")
        frame[value] = pd.to_numeric(frame[value], errors="raise")
        if frame[value].isna().any() or not np.isfinite(frame[value]).all() or (frame[value] < 0).any():
            raise ValueError(f"{label} values must be finite and nonnegative")
    eligible = forecast[forecast["week_start"] + pd.Timedelta(days=7) <= cutoff]
    joined = eligible.merge(actual[list(keys) + [actual_column]], on=list(keys), how="left", validate="one_to_one")
    matched = joined.dropna(subset=[actual_column])
    errors = (matched[prediction_column] - matched[actual_column]).abs()
    total = float(matched[actual_column].sum())
    mae = float(errors.mean()) if len(matched) else None
    wape = float(errors.sum() / total) if len(matched) and total > 0 else None
    enough = len(matched) >= min_matches and wape is not None
    degraded = bool(enough and wape > wape_threshold)
    return {"status": "degraded" if degraded else "ok" if enough else "insufficient_data",
            "performance_degraded": degraded, "as_of": cutoff.isoformat(), "eligible_forecasts": len(eligible),
            "future_forecasts": len(forecast) - len(eligible), "matched_forecasts": len(matched),
            "missing_actuals": len(eligible) - len(matched), "mae": mae, "wape": wape,
            "wape_threshold": wape_threshold, "min_matches": min_matches}
