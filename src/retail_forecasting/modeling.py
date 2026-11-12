"""Weekly forecasting, chronological backtesting, and promotion gates."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

FEATURES = ["stock_code", "lag_1", "lag_2", "lag_4", "rolling_mean_4", "week_of_year", "month"]
NUMERIC_FEATURES = FEATURES[1:]


def _usable_rows(frame):
    missing = {"week_start", "demand", *FEATURES}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing modeling columns: {sorted(missing)}")
    rows = frame.dropna(subset=["week_start", "demand", *FEATURES]).copy()
    rows["week_start"] = pd.to_datetime(rows["week_start"])
    if rows.duplicated(["week_start", "stock_code"]).any():
        raise ValueError("Modeling rows must be unique per stock code and week")
    if not np.isfinite(rows[["demand", *NUMERIC_FEATURES]].to_numpy(dtype=float)).all():
        raise ValueError("Demand and numeric features must be finite")
    if (rows["demand"] < 0).any():
        raise ValueError("Demand must be nonnegative")
    return rows.sort_values(["week_start", "stock_code"]).reset_index(drop=True)


def fit_model(features, random_state=42):
    """Fit preprocessing and model on complete historical feature rows only."""
    rows = _usable_rows(features)
    if rows.empty:
        raise ValueError("No complete historical feature rows available")
    model = Pipeline([
        ("preprocess", ColumnTransformer([
            ("product", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["stock_code"]),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ])),
        ("model", HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                                min_samples_leaf=10, random_state=random_state)),
    ])
    model.fit(rows[FEATURES], rows["demand"])
    return model


def predict(model, feature_rows):
    missing = {"week_start", *FEATURES}.difference(feature_rows.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if feature_rows.empty or feature_rows[["week_start", *FEATURES]].isna().any().any():
        raise ValueError("Prediction rows must be nonempty and have complete features")
    result = feature_rows[["stock_code", "week_start"]].copy()
    result["prediction"] = np.maximum(0.0, model.predict(feature_rows[FEATURES]))
    return result.reset_index(drop=True)


def error_metrics(actual, predicted):
    """WAPE is undefined (None) when all actual demand is zero."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or not actual.size:
        raise ValueError("Actuals and predictions must be equally sized nonempty vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actuals and predictions must be finite")
    error = np.abs(actual - predicted)
    denominator = float(np.abs(actual).sum())
    return {"mae": float(error.mean()), "wape": float(error.sum() / denominator) if denominator else None}


def rolling_backtest(features, holdout_weeks=4, min_train_weeks=8, random_state=42):
    """Fit afresh strictly before each holdout week and compare with lag-one naive."""
    if holdout_weeks < 1 or min_train_weeks < 1:
        raise ValueError("holdout_weeks and min_train_weeks must be positive")
    rows = _usable_rows(features)
    weeks = sorted(rows["week_start"].unique())
    if len(weeks) < holdout_weeks + min_train_weeks:
        raise ValueError("Not enough complete weeks for the requested rolling backtest")
    results = []
    for week in weeks[-holdout_weeks:]:
        model = fit_model(rows.loc[rows["week_start"] < week], random_state=random_state)
        test = rows.loc[rows["week_start"] == week].copy()
        test["prediction"] = np.maximum(0.0, model.predict(test[FEATURES]))
        test["baseline_prediction"] = np.maximum(0.0, test["lag_1"])
        results.append(test[["stock_code", "week_start", "demand", "prediction", "baseline_prediction"]])
    return pd.concat(results, ignore_index=True)


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reasons: list
    candidate_mae: float
    baseline_mae: float
    champion_mae: float | None

    def to_dict(self):
        return asdict(self)


def promotion_gate(candidate_mae, baseline_mae, champion_mae=None, *, data_quality_passed=True,
                   max_relative_degradation=0.0):
    """Compare MAE on identical untouched evaluation rows; ties pass."""
    if max_relative_degradation < 0 or not np.isfinite(max_relative_degradation):
        raise ValueError("max_relative_degradation must be finite and nonnegative")
    reasons = []
    if not data_quality_passed:
        reasons.append("Data quality gate failed")
    values = {"candidate": candidate_mae, "baseline": baseline_mae}
    if champion_mae is not None:
        values["champion"] = champion_mae
    invalid = [name for name, value in values.items() if not np.isfinite(value) or value < 0]
    if invalid:
        reasons.append(f"Invalid MAE for: {', '.join(invalid)}")
    else:
        for name in ("baseline", "champion"):
            if name in values and candidate_mae > values[name] * (1 + max_relative_degradation):
                reasons.append(f"Candidate MAE exceeds {name} MAE beyond allowed tolerance")
    return PromotionDecision(not reasons, reasons, candidate_mae, baseline_mae, champion_mae)
