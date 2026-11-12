import unittest
import pandas as pd
from retail_forecasting.monitoring import compare_drift, evaluate_forecasts


class MonitoringTests(unittest.TestCase):
    def test_shift_and_insufficient_data(self):
        reference = pd.DataFrame({"lag_1": [10.0] * 30})
        shifted = pd.DataFrame({"lag_1": [30.0] * 30})
        self.assertTrue(compare_drift(reference, shifted, ["lag_1"])["drift_detected"])
        self.assertFalse(compare_drift(reference, reference, ["lag_1"])["drift_detected"])
        self.assertEqual(compare_drift(reference, shifted.head(1), ["lag_1"])["status"], "insufficient_data")

    def test_actuals_scored_only_after_week_closes(self):
        forecasts = pd.DataFrame({"stock_code": ["A", "A"], "week_start": ["2021-01-04", "2021-01-11"], "prediction": [5, 4]})
        actuals = pd.DataFrame({"stock_code": ["A", "A"], "week_start": ["2021-01-04", "2021-01-11"], "demand": [10, 100]})
        report = evaluate_forecasts(forecasts, actuals, as_of="2021-01-11")
        self.assertEqual(report["matched_forecasts"], 1)
        self.assertEqual(report["future_forecasts"], 1)
        self.assertEqual(report["wape"], 0.5)

    def test_zero_actual_and_duplicate_keys(self):
        forecast = pd.DataFrame({"stock_code": ["A"], "week_start": ["2021-01-04"], "prediction": [2]})
        actual = pd.DataFrame({"stock_code": ["A"], "week_start": ["2021-01-04"], "demand": [0]})
        report = evaluate_forecasts(forecast, actual, as_of="2021-01-11")
        self.assertIsNone(report["wape"])
        self.assertEqual(report["status"], "insufficient_data")
        with self.assertRaises(ValueError):
            evaluate_forecasts(pd.concat([forecast, forecast]), actual, as_of="2021-01-11")


if __name__ == "__main__":
    unittest.main()
