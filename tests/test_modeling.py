import numpy as np
import pandas as pd
import pytest

from retail_forecasting import modeling


def test_metrics_and_gate_handle_zero_demand_and_reject_worse_candidate():
    assert modeling.error_metrics([0, 0], [0, 2]) == {"mae": 1.0, "wape": None}
    assert not modeling.promotion_gate(2.0, 1.0).accepted
    assert not modeling.promotion_gate(1.0, 2.0, 0.9).accepted
    assert not modeling.promotion_gate(1.0, 2.0, data_quality_passed=False).accepted
    assert modeling.promotion_gate(1.0, 1.0, 1.0).accepted


def test_backtest_fits_only_earlier_weeks(monkeypatch):
    weeks = pd.date_range("2020-01-06", periods=7, freq="W-MON")
    features = pd.DataFrame({"week_start": weeks, "stock_code": ["A"] * 7,
                             "demand": np.arange(1, 8, dtype=float),
                             "lag_1": np.arange(7, dtype=float), "lag_2": [1.] * 7,
                             "lag_4": [1.] * 7, "rolling_mean_4": [1.] * 7,
                             "week_of_year": weeks.isocalendar().week.astype(int),
                             "month": weeks.month})
    training_cutoffs = []

    class Model:
        def predict(self, frame):
            return frame["lag_1"].to_numpy()

    def fake_fit(frame, random_state=42):
        training_cutoffs.append(frame.week_start.max())
        return Model()

    monkeypatch.setattr(modeling, "fit_model", fake_fit)
    results = modeling.rolling_backtest(features, holdout_weeks=2, min_train_weeks=3)
    assert len(results) == 2
    assert all(train_end < test_week for train_end, test_week in zip(training_cutoffs, results.week_start))
    assert results.prediction.tolist() == results.baseline_prediction.tolist()
    with pytest.raises(ValueError, match="Not enough"):
        modeling.rolling_backtest(features, holdout_weeks=4, min_train_weeks=4)
