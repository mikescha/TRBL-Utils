import unittest
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

from acoustic_ratios import (
    AVG_FEMALE_CALLS,
    AVG_NESTLING_CALLS,
    AcousticReproductiveIndex,
    FledglingMetrics,
    filter_by_datetime_bounds,
)


class TestFilterByDatetimeBounds(unittest.TestCase):
    def test_filter_datetime_bounds(self) -> None:
        """Verifies that the date ranges are inclusive and the hour ranges are
        strictly 7:00 <= hour < 20:00 (exclusive of 20:00).
        """
        df = pd.DataFrame({
            "date": [date(2024, 5, 10), date(2024, 5, 10), date(2024, 5, 10), date(2024, 5, 15)],
            "hour": [6, 12, 20, 12]
        })
        # 6:00 is too early; 20:00 is too late (exclusive)
        filtered = filter_by_datetime_bounds(df, date(2024, 5, 10), date(2024, 5, 12), "date", "hour")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["hour"], 12)


class TestAcousticReproductiveIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = AcousticReproductiveIndex()
        self.hatch_date = date(2024, 5, 15)
        self.site_name = "2024 Baja Rancho Cinega Redonda 1"

        # Biological Windows
        self.f_start = self.hatch_date - timedelta(days=self.metric.DAYS_TO_COUNT)
        self.f_end = self.hatch_date - timedelta(days=1)
        self.n_start = self.hatch_date + timedelta(days=self.metric.NESTLING_OFFSET_DAYS)
        self.n_end = self.n_start + timedelta(days=self.metric.DAYS_TO_COUNT - 1)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_calculate_row_success(
        self, 
        mock_bounds, 
        mock_totals, 
        mock_days_count, 
        mock_detections
    ) -> None:
        """Verifies successful ARI calculation under the new recording-effort
        proportion logic.
        """
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100       # 100 daytime recordings scheduled
        mock_days_count.return_value = 10    # 10 active days of recording

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                # 10 valid female detections
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                return pd.DataFrame({"date": dates, "hour": [12] * 10})
            elif call_type == "Nestling":
                # 5 valid nestling detections
                dates = [self.n_start + timedelta(days=i) for i in range(5)]
                return pd.DataFrame({"date": dates, "hour": [12] * 5})
            return pd.DataFrame()
        mock_detections.side_effect = side_effect_detections

        row = {"Breeding Type": "Simple"}
        result = self.metric.calculate_row(row, self.hatch_date, self.site_name)

        self.assertEqual(result["Incubation_Days"], 10)
        self.assertEqual(result["Total_Female_Calls"], 10)
        # Avg = 10 detections / 100 recordings = 0.1
        self.assertEqual(result[AVG_FEMALE_CALLS], 0.1)

        self.assertEqual(result["Total_Nestling_Calls"], 5)
        # Avg = 5 detections / 100 recordings = 0.05
        self.assertEqual(result[AVG_NESTLING_CALLS], 0.05)

        # ARI = 0.05 Nestling proportion / 0.1 Female proportion = 0.5
        self.assertEqual(result["ARI"], 0.5)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_window_boundary_leakage(
        self, 
        mock_bounds, 
        mock_totals, 
        mock_days_count, 
        mock_detections
    ) -> None:
        """Ensures detections on hatch day or outside active 7:00-20:00 hours
        are strictly filtered out.
        """
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                # 10 valid detections
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                hours = [12] * 10
                
                # INJECT LEAKS
                dates.append(self.hatch_date)  # Hatch date (should be ignored)
                hours.append(12)
                dates.append(self.f_start)     # Early hour (should be ignored)
                hours.append(6)
                dates.append(self.f_start)     # Late hour (should be ignored)
                hours.append(20)
                
                return pd.DataFrame({"date": dates, "hour": hours})
            elif call_type == "Nestling":
                dates = [self.n_start + timedelta(days=i) for i in range(5)]
                hours = [12] * 5
                
                # INJECT LEAKS
                dates.append(self.hatch_date)  # Before nestling start (should be ignored)
                hours.append(12)
                dates.append(self.n_start)     # Late hour (should be ignored)
                hours.append(21)
                
                return pd.DataFrame({"date": dates, "hour": hours})
            return pd.DataFrame()
        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)
        self.assertEqual(result["Total_Female_Calls"], 10)
        self.assertEqual(result["Total_Nestling_Calls"], 5)
        self.assertEqual(result["ARI"], 0.5)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_true_abandonment_math(
        self, 
        mock_bounds, 
        mock_totals, 
        mock_days_count, 
        mock_detections
    ) -> None:
        """Validates that normal female effort but zero offspring detections yields
        exactly 0.0.
        """
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                return pd.DataFrame({"date": dates, "hour": [12] * 10})
            return pd.DataFrame()  # Empty nestlings
        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)
        self.assertEqual(result["ARI"], 0.0)


class TestFledglingMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = FledglingMetrics()
        self.hatch_date = date(2024, 5, 15)
        self.site_name = "2024 Baja Rancho Cinega Redonda 1"
        self.f_start = self.hatch_date + timedelta(days=self.metric.FLEDGLING_OFFSET_DAYS)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_fledglings_success(self, mock_bounds, mock_totals, mock_detections) -> None:
        """Validates fledgling proportion counts summarize correctly."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        
        dates = [self.f_start, self.f_start + timedelta(days=2)]
        mock_detections.return_value = pd.DataFrame({"date": dates, "hour": [12, 12]})

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)
        
        self.assertEqual(result["Fledglings_Present"], "Yes")
        self.assertEqual(result["Total_Fledgling_Calls"], 2)
        self.assertEqual(result["Fledgling_Days"], 3)  # span of 3 days
        self.assertEqual(result["Avg_Fledgling_Calls_Day"], 0.02)


class TestDataLoader(unittest.TestCase):
    @patch("acoustic_ratios.load_recordings_parquet")
    def test_get_total_recordings(self, mock_load) -> None:
        """Verifies get_total_recordings properly queries our parquet structures and
        respects START_HOUR/END_HOUR filters.
        """
        from acoustic_ratios import get_total_recordings
        
        mock_load.return_value = pd.DataFrame({
            "site_clean": ["testsite", "testsite", "testsite"],
            "date_parsed": [date(2024, 5, 1), date(2024, 5, 1), date(2024, 5, 2)],
            "hour": [8, 21, 10],  # 21 is excluded (>= END_HOUR)
            "n_recordings": [5, 100, 3]
        })
        
        total = get_total_recordings("TestSite", date(2024, 5, 1), date(2024, 5, 2))
        # 5 on May 1st (hour 8) + 3 on May 2nd (hour 10) = 8 total recordings
        self.assertEqual(total, 8)


if __name__ == "__main__":
    unittest.main()