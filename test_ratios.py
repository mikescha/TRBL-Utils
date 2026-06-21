import unittest
from datetime import date, timedelta
from pathlib import Path
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
    def test_window_boundary_leakage(self, mock_get_counts) -> None:
        """Ensures data on the hatch day and hatch+1 day are strictly ignored."""
        
        def mock_counts(site_id: str, call_type: str) -> pd.Series:
            # Create standard valid data
            dates = []
            values = []
            if call_type == "Female":
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                values = [10] * 10
                # INJECT LEAKAGE: Massive spike on Hatch Day (Should be ignored)
                dates.append(self.hatch_date)
                values.append(5000)
            elif call_type == "Nestling":
                dates = [self.n_start + timedelta(days=i) for i in range(10)]
                values = [5] * 10
                # INJECT LEAKAGE: Massive spike on Hatch Day + 1 (Should be ignored)
                dates.append(self.hatch_date + timedelta(days=1))
                values.append(5000)
                
            return pd.Series(values, index=dates)

        mock_get_counts.side_effect = mock_counts
        
        result = self.metric.calculate_row({}, self.hatch_date, self.site_id)
        
        # If the 5000 spikes leaked in, this ratio would be wildly skewed.
        # It should remain exactly 0.5.
        self.assertEqual(result["ARI_num"], 0.5)
    
    @patch("ratios.get_daily_validated_counts")
    def test_true_abandonment_math(self, mock_get_counts) -> None:
        """Validates that normal female effort but zero offspring yields exactly 0.0."""
        
        def mock_counts(site_id: str, call_type: str) -> pd.Series:
            if call_type == "Female":
                # Healthy female nesting effort
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                return pd.Series([15] * 10, index=dates)
            if call_type == "Nestling":
                # Nestlings completely failed (0 calls across the 10 days)
                dates = [self.n_start + timedelta(days=i) for i in range(10)]
                return pd.Series([0] * 10, index=dates)
            return pd.Series(dtype=int)

        mock_get_counts.side_effect = mock_counts
        
        result = self.metric.calculate_row({}, self.hatch_date, self.site_id)
        self.assertEqual(result["ARI_num"], 0.0)
    
    def test_post_process_fallback_cutoff(self) -> None:
        """Validates the static fallback if not enough data exists to compute a dynamic gap."""
        # Only one valid score provided
        df = pd.DataFrame({
            "ARI_num": [0.4, "ND_INSUFFICIENT_DAYS", "ND_MISSING_DATES"]
        })
        
        processed_df = self.metric.post_process(df)
        
        # Verify it successfully fell back to the 0.15 default
        self.assertEqual(self.metric.calculated_cutoff, 0.15)
        
        outcomes = processed_df["Calculated_Outcome"].tolist()
        self.assertEqual(outcomes[0], "Successful")  # 0.4 > 0.15
        self.assertEqual(outcomes[1], "Unknown")
    
    
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


class TestDataLoader(unittest.TestCase):    
    @patch("pandas.read_csv")
    @patch("pathlib.Path.glob")
    def test_get_daily_validated_counts_safety(self, mock_glob, mock_read_csv) -> None:
        """Ensures the loader strips whitespace, ignores case, and counts correctly."""
        from ratios import get_daily_validated_counts
        
        # Clear the lru_cache just for testing purposes so it actually runs
        get_daily_validated_counts.cache_clear()
        
        # Mock finding a file
        mock_glob.return_value = [Path("fake_file.csv")]
        
        # Mock the raw CSV data with dirty text entries
        mock_read_csv.return_value = pd.DataFrame({
            "year": [2024, 2024, 2024, 2024],
            "month": [5, 5, 5, 5],
            "day": [1, 1, 2, 2],
            "validated": [
                "present",    # Valid
                " PRESENT ",  # Valid (messy spacing/caps)
                "absent",     # Invalid
                "unreviewed"  # Invalid
            ]
        })
        
        result = get_daily_validated_counts("TestSite", "Female")
        
        # May 1st should have 2 counts. May 2nd should have 0 counts (and thus drop out)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[date(2024, 5, 1)], 2)
        self.assertNotIn(date(2024, 5, 2), result)

if __name__ == "__main__":
    unittest.main()