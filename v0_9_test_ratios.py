import unittest
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

from ratios import AcousticReproductiveIndex, FledglingMetrics


class TestAcousticReproductiveIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = AcousticReproductiveIndex()
        self.hatch_date = date(2024, 5, 15)
        self.site_name = "2024 Baja Rancho Cinega Redonda 1"
        
        # Calculate windows based on the plugin's internal logic
        self.f_start = self.hatch_date - timedelta(days=self.metric.DAYS_TO_COUNT)
        self.n_start = self.hatch_date + timedelta(days=self.metric.NESTLING_OFFSET_DAYS)
    

    @patch("ratios.get_daily_validated_counts")
    def test_calculate_row_success(self, mock_get_counts) -> None:
        """Verifies every new column for date, sum, and average accuracy."""
        def mock_counts(site_name: str, call_type: str) -> pd.Series:
            if call_type == "Female":
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                return pd.Series([10] * 10, index=dates)
            if call_type == "Nestling":
                dates = [self.n_start + timedelta(days=i) for i in range(10)]
                return pd.Series([5] * 10, index=dates)
            return pd.Series(dtype=int)

        mock_get_counts.side_effect = mock_counts
        
        row = {"Breeding Type": "Simple", "Deployment Start":"4/1/2024", "Deployment End":"7/1/2024"}
        result = self.metric.calculate_row(row, self.hatch_date, self.site_name)
        
        # Verify schema and math
        self.assertEqual(result["Incubation_Days"], 10)
        self.assertEqual(result["Total_Female_Calls"], 100)
        self.assertEqual(result["Avg_Female_Calls_Day"], 10.0)
        self.assertEqual(result["Nestling_Days"], 10)
        self.assertEqual(result["Total_Nestling_Calls"], 50)
        self.assertEqual(result["Avg_Nestling_Calls_Day"], 5.0)
        self.assertEqual(result["ARI"], 0.5)

    @patch("ratios.get_daily_validated_counts")
    def test_window_boundary_leakage(self, mock_get_counts) -> None:
        """Ensures data on hatch day/hatch+1 does not leak into calculations."""
        def mock_counts(site_name: str, call_type: str) -> pd.Series:
            dates, values = [], []
            if call_type == "Female":
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                values = [10] * 10
                dates.append(self.hatch_date) # Leak
                values.append(5000)
            elif call_type == "Nestling":
                dates = [self.n_start + timedelta(days=i) for i in range(10)]
                values = [5] * 10
                dates.append(self.hatch_date + timedelta(days=1)) # Leak
                values.append(5000)
            return pd.Series(values, index=dates)

        mock_get_counts.side_effect = mock_counts
        row = {"Breeding Type": "Simple", "Deployment Start":"4/1/2024", "Deployment End":"7/1/2024"}
        result = self.metric.calculate_row(row, self.hatch_date, self.site_name)
        self.assertEqual(result["ARI"], 0.5)

    @patch("ratios.get_daily_validated_counts")
    def test_true_abandonment_math(self, mock_get_counts) -> None:
        """Validates that healthy female effort but 0 nestling calls yields 0.0."""
        def mock_counts(site_name: str, call_type: str) -> pd.Series:
            if call_type == "Female":
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                return pd.Series([15] * 10, index=dates)
            if call_type == "Nestling":
                dates = [self.n_start + timedelta(days=i) for i in range(10)]
                return pd.Series([0] * 10, index=dates)
            return pd.Series(dtype=int)

        mock_get_counts.side_effect = mock_counts
        row = {"Breeding Type": "Simple", "Deployment Start":"4/1/2024", "Deployment End":"7/1/2024"}
        result = self.metric.calculate_row(row, self.hatch_date, self.site_name)
        self.assertEqual(result["ARI"], 0.0)


class TestFledglingMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = FledglingMetrics()
        self.hatch_date = date(2024, 5, 15)
        self.site_name = "2024 Baja Rancho Cinega Redonda 1"
        self.f_start = self.hatch_date + timedelta(days=self.metric.FLEDGLING_OFFSET_DAYS)

    @patch("ratios.get_daily_validated_counts")
    def test_fledglings_success(self, mock_get_counts) -> None:
        def mock_counts(site_name: str, call_type: str) -> pd.Series:
            if call_type == "Fledgling":
                dates = [self.f_start, self.f_start + timedelta(days=2)]
                return pd.Series([10, 20], index=dates)
            return pd.Series(dtype=int)

        mock_get_counts.side_effect = mock_counts
        
        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)
        
        self.assertEqual(result["Fledglings_Present"], "Yes")
        self.assertEqual(result["Total_Fledgling_Calls"], 30)
        self.assertEqual(result["Fledgling_Days"], 3)


if __name__ == "__main__":
    unittest.main()