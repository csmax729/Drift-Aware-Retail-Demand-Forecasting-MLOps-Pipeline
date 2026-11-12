import pandas as pd
import pytest

from retail_forecasting.data import make_features, make_prediction_features, prepare_transactions, weekly_demand


def test_validation_replay_and_duplicate_key_semantics():
    rows = [
        ("100", "A", 2, "2024-01-01"),
        ("100", "A", 3, "2024-01-01"),  # same invoice/item, distinct line
        ("100", "A", 3, "2024-01-01"),  # exact duplicate
        ("C101", "A", -2, "2024-01-02"),
        ("102", "B", 5, "2024-01-08"),
        ("103", "C", 8, "2024-01-15"),  # cutoff is exclusive
    ]
    frame = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate"])
    good, bad, report = prepare_transactions(frame, "2024-01-15")
    assert report["accepted_rows"] == 3
    assert set(bad.reason) == {"exact_duplicate", "return_or_cancellation"}
    weekly = weekly_demand(good, "2024-01-15")
    assert weekly.to_dict("records") == [
        {"week_start": pd.Timestamp("2024-01-01"), "stock_code": "A", "demand": 5.0},
        {"week_start": pd.Timestamp("2024-01-08"), "stock_code": "A", "demand": 0.0},
        {"week_start": pd.Timestamp("2024-01-08"), "stock_code": "B", "demand": 5.0},
    ]


def test_features_use_only_prior_weeks_and_prediction_actual_is_unknown():
    weekly = pd.DataFrame({"week_start": pd.date_range("2024-01-01", periods=5, freq="W-MON"), "stock_code": ["A"] * 5, "demand": [1., 2., 3., 4., 100.]})
    last = make_features(weekly).iloc[-1]
    assert last.lag_1 == 4
    assert last.lag_4 == 1
    assert last.rolling_mean_4 == 2.5
    future = make_prediction_features(weekly, "2024-02-05").iloc[0]
    assert pd.isna(future.demand)
    assert future.lag_1 == 100
    assert future.rolling_mean_4 == 27.25


def test_partial_first_week_and_invalid_schema():
    rows = [("1", "A", 10, "2024-01-03"), ("2", "A", 3, "2024-01-08")]
    frame = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate"])
    good, _, _ = prepare_transactions(frame, "2024-01-15")
    assert weekly_demand(good, "2024-01-15").demand.tolist() == [3.0]
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_transactions(frame.drop(columns="Quantity"), "2024-01-15")
    with pytest.raises(ValueError, match="Monday"):
        prepare_transactions(frame, "2024-01-16")
