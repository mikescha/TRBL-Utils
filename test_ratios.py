import unittest
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

from ratios import AcousticReproductiveIndex, FledglingMetrics


class TestAcousticReproductiveIndex(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize the metric engine and set a standard testing timeline."""
        self.metric = AcousticReproductiveIndex()
        self.hatch_date = date(2024, 5, 15)
        self.site_id = "TestSite"

        # Calculate exact boundary dates to align with the plugin's internal math
        self.f_start = self.hatch_date - timedelta(days=self.metric.DAYS_TO_COUNT)
        self.n_start = self.hatch_date + timedelta(days=self.metric.NESTLING_OFFSET_DAYS)

    @patch("ratios.get_daily_validated_counts")
    def test_calculate_row_success(self, mock_get_counts) -> None:
        """Validates that a full timeline of data correctly computes the ARI."""
        
        def mock_counts(site_id: str, call_type: str) -> pd.Series:
            if call_type == "Female":
                # 10 days of data, 10 calls per day (Avg = 10)
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                return pd.Series([10] * 10, index=dates)
            if call_type == "Nestling":
                # 10 days of data, 5 calls per day (Avg = 5)
                dates = [self.n_start + timedelta(days=i) for i in range(10)]
                return pd.Series([5] * 10, index=dates)
            return pd.Series(dtype=int)

        mock_get_counts.side_effect = mock_counts
        
        result = self.metric.calculate_row({}, self.hatch_date, self.site_id)
        
        # 5 nestlings / 10 females = 0.5
        self.assertEqual(result["ARI_num"], 0.5)

    @patch("ratios.get_daily_validated_counts")
    def test_calculate_row_insufficient_days(self, mock_get_counts) -> None:
        """Validates that missing days trigger the quality control failure."""
        
        def mock_counts(site_id: str, call_type: str) -> pd.Series:
            if call_type == "Female":
                # Only 3 days of data provided (Fails the MIN_REQUIRED_DAYS threshold of 4)
                dates = [self.f_start + timedelta(days=i) for i in range(3)]
                return pd.Series([10] * 3, index=dates)
            if call_type == "Nestling":
                dates = [self.n_start + timedelta(days=i) for i in range(10)]
                return pd.Series([5] * 10, index=dates)
            return pd.Series(dtype=int)

        mock_get_counts.side_effect = mock_counts
        
        result = self.metric.calculate_row({}, self.hatch_date, self.site_id)
        self.assertEqual(result["ARI_num"], "ND_INSUFFICIENT_DAYS")

    @patch("ratios.get_daily_validated_counts")
    def test_calculate_row_div_by_zero(self, mock_get_counts) -> None:
        """Protects against dividing by zero when no females are detected."""
        
        def mock_counts(site_id: str, call_type: str) -> pd.Series:
            if call_type == "Female":
                # 10 days of data, but ZERO calls detected
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                return pd.Series([0] * 10, index=dates)
            if call_type == "Nestling":
                # 10 days of data, positive calls detected
                dates = [self.n_start + timedelta(days=i) for i in range(10)]
                return pd.Series([5] * 10, index=dates)
            return pd.Series(dtype=int)

        mock_get_counts.side_effect = mock_counts
        
        result = self.metric.calculate_row({}, self.hatch_date, self.site_id)
        self.assertEqual(result["ARI_num"], "ND_DIV_ZERO")

    def test_post_process_outcomes(self) -> None:
        """Validates the gap-analysis cutoff generation and outcome tagging."""
        
        # We build a DataFrame with known ARI values. 
        # Under 0.8, the sorted values are [0.0, 0.1, 0.4]. 
        # The gaps are 0.1 and 0.3. 
        # The largest gap is 0.3 (between 0.1 and 0.4). 
        # The cutoff should be the midpoint: 0.1 + (0.3 / 2) = 0.25.
        df = pd.DataFrame({
            "ARI_num": [0.0, 0.1, 0.4, 0.9, "ND_INSUFFICIENT_DAYS"]
        })
        
        processed_df = self.metric.post_process(df)
        
        # Verify the cutoff property was saved accurately
        self.assertEqual(self.metric.calculated_cutoff, 0.25)
        
        # Verify categorical mappings
        outcomes = processed_df["Calculated_Outcome"].tolist()
        self.assertEqual(outcomes[0], "Abandoned")              # 0.0
        self.assertEqual(outcomes[1], "Partially Abandoned")    # 0.1 (Below 0.25 cutoff)
        self.assertEqual(outcomes[2], "Successful")             # 0.4 (Above 0.25 cutoff)
        self.assertEqual(outcomes[3], "Successful")             # 0.9 (Above 0.25 cutoff)
        self.assertEqual(outcomes[4], "Unknown")                # Non-numeric


class TestFledglingMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = FledglingMetrics()
        self.hatch_date = date(2024, 5, 15)
        self.site_id = "TestSite"
        self.f_start = self.hatch_date + timedelta(days=self.metric.FLEDGLING_OFFSET_DAYS)

    @patch("ratios.get_daily_validated_counts")
    def test_fledglings_present(self, mock_get_counts) -> None:
        """Validates fledgling counts summarize correctly when birds are present."""
        
        def mock_counts(site_id: str, call_type: str) -> pd.Series:
            if call_type == "Fledgling":
                dates = [self.f_start + timedelta(days=i) for i in range(2)]
                return pd.Series([10, 20], index=dates)
            return pd.Series(dtype=int)

        mock_get_counts.side_effect = mock_counts
        
        result = self.metric.calculate_row({}, self.hatch_date, self.site_id)
        self.assertEqual(result["Fledglings_Present"], "Yes")
        self.assertEqual(result["Fledgling_Total"], 30)

    @patch("ratios.get_daily_validated_counts")
    def test_fledglings_absent(self, mock_get_counts) -> None:
        """Validates safe handling when the file is empty or no fledglings exist."""
        
        mock_get_counts.return_value = pd.Series(dtype=int)
        
        result = self.metric.calculate_row({}, self.hatch_date, self.site_id)
        self.assertEqual(result["Fledglings_Present"], "No")
        self.assertEqual(result["Fledgling_Total"], 0)


if __name__ == "__main__":
    unittest.main()