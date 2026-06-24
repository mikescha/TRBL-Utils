import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import call, patch

import pandas as pd

from acoustic_ratios import (
    AVG_FEMALE_CALLS,
    AVG_NESTLING_CALLS,
    AcousticReproductiveIndex,
    FledglingMetrics,
    filter_by_datetime_bounds,
    get_raw_validated_detections,
    get_site_recording_bounds,
    get_total_recordings,
    inclusive_day_span,
)


class TestFilterByDatetimeBounds(unittest.TestCase):
    def test_filter_datetime_bounds(self) -> None:
        """Date ranges are inclusive; hour range is 7 <= hour < 20."""
        df = pd.DataFrame(
            {
                "date": [
                    date(2024, 5, 10),
                    date(2024, 5, 10),
                    date(2024, 5, 10),
                    date(2024, 5, 15),
                ],
                "hour": [6, 12, 20, 12],
            }
        )

        filtered = filter_by_datetime_bounds(
            df,
            date(2024, 5, 10),
            date(2024, 5, 12),
            "date",
            "hour",
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["hour"], 12)

    def test_filter_includes_7_and_19_excludes_20(self) -> None:
        df = pd.DataFrame(
            {
                "date": [date(2024, 5, 10)] * 4,
                "hour": [6, 7, 19, 20],
            }
        )

        filtered = filter_by_datetime_bounds(
            df,
            date(2024, 5, 10),
            date(2024, 5, 10),
        )

        self.assertEqual(filtered["hour"].tolist(), [7, 19])

    def test_filter_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["date", "hour"])

        filtered = filter_by_datetime_bounds(
            df,
            date(2024, 5, 10),
            date(2024, 5, 12),
        )

        self.assertTrue(filtered.empty)

    def test_filter_custom_column_names(self) -> None:
        df = pd.DataFrame(
            {
                "date_parsed": [date(2024, 5, 10), date(2024, 5, 11)],
                "hour_of_day": [12, 21],
            }
        )

        filtered = filter_by_datetime_bounds(
            df,
            date(2024, 5, 10),
            date(2024, 5, 11),
            date_col="date_parsed",
            hour_col="hour_of_day",
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["date_parsed"], date(2024, 5, 10))


class TestInclusiveDaySpan(unittest.TestCase):
    def test_inclusive_day_span_same_day(self) -> None:
        self.assertEqual(inclusive_day_span(date(2024, 5, 10), date(2024, 5, 10)), 1)

    def test_inclusive_day_span_ten_days(self) -> None:
        self.assertEqual(inclusive_day_span(date(2024, 5, 17), date(2024, 5, 26)), 10)

    def test_inclusive_day_span_empty_when_end_before_start(self) -> None:
        self.assertEqual(inclusive_day_span(date(2024, 5, 10), date(2024, 5, 9)), 0)


class TestAcousticReproductiveIndex(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = AcousticReproductiveIndex()
        self.hatch_date = date(2024, 5, 15)
        self.site_name = "2024 Baja Rancho Cinega Redonda 1"

        self.f_start = self.hatch_date - timedelta(days=self.metric.DAYS_TO_COUNT)
        self.f_end = self.hatch_date - timedelta(days=1)

        self.n_start = self.hatch_date + timedelta(days=self.metric.NESTLING_OFFSET_DAYS)
        self.n_end = self.n_start + timedelta(days=self.metric.DAYS_TO_COUNT - 1)

    def female_df(self, day_offsets: list[int], hour: int = 12) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [self.f_start + timedelta(days=i) for i in day_offsets],
                "hour": [hour] * len(day_offsets),
            }
        )

    def nestling_df(self, day_offsets: list[int], hour: int = 12) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": [self.n_start + timedelta(days=i) for i in day_offsets],
                "hour": [hour] * len(day_offsets),
            }
        )

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_calculate_row_success(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Successful ARI calculation uses detections per available recording."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                return self.female_df(list(range(10)))
            if call_type == "Nestling":
                return self.nestling_df(list(range(5)))
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        row = {"Breeding Type": "Simple"}
        result = self.metric.calculate_row(row, self.hatch_date, self.site_name)

        self.assertEqual(result["Incubation_Days"], 10)
        self.assertEqual(result["Female_Detection_Recordings"], 10)
        self.assertEqual(result[AVG_FEMALE_CALLS], 0.1)

        self.assertEqual(result["Nestling_Days"], 10)
        self.assertEqual(result["Nestling_Detection_Recordings"], 5)
        self.assertEqual(result[AVG_NESTLING_CALLS], 0.05)

        self.assertEqual(result["ARI"], 0.5)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_nestling_window_is_hatch_plus_2_through_hatch_plus_11(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Nestling ARI window is 10 inclusive days: hatch+2 through hatch+11.

        A detection on hatch+12 must not leak into the numerator.
        """
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 6, 1))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                return self.female_df(list(range(10)))
            if call_type == "Nestling":
                return pd.DataFrame(
                    {
                        "date": [
                            self.n_start,
                            self.n_end,
                            self.n_end + timedelta(days=1),
                        ],
                        "hour": [12, 12, 12],
                    }
                )
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Nestling_Days"], 10)
        self.assertEqual(result["Nestling_Detection_Recordings"], 2)
        self.assertEqual(result[AVG_NESTLING_CALLS], 0.02)
        self.assertEqual(result["ARI"], 0.2)

        self.assertIn(
            call(self.site_name, self.n_start, self.n_end),
            mock_totals.call_args_list,
        )

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_nestling_window_is_clipped_by_recording_stop(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Nestling analysis stops at deployment end if recording ends before hatch+11."""
        rec_stop = self.n_start + timedelta(days=4)
        mock_bounds.return_value = (date(2024, 5, 1), rec_stop)
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                return self.female_df(list(range(10)))
            if call_type == "Nestling":
                return pd.DataFrame(
                    {
                        "date": [
                            self.n_start,
                            rec_stop,
                            rec_stop + timedelta(days=1),
                        ],
                        "hour": [12, 12, 12],
                    }
                )
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Nestling_Days"], 5)
        self.assertEqual(result["Nestling_Detection_Recordings"], 2)
        self.assertIn(
            call(self.site_name, self.n_start, rec_stop),
            mock_totals.call_args_list,
        )

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_span_days_do_not_depend_on_latest_detection(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Nestling_Days reports the full analysis span, not latest-detection span."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 6, 1))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                return self.female_df(list(range(10)))
            if call_type == "Nestling":
                return self.nestling_df([0])
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Nestling_Days"], 10)
        self.assertEqual(result["Nestling_Detection_Recordings"], 1)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_window_boundary_leakage(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Hatch day and out-of-hours detections are filtered out."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                dates = [self.f_start + timedelta(days=i) for i in range(10)]
                hours = [12] * 10

                dates.append(self.hatch_date)
                hours.append(12)
                dates.append(self.f_start)
                hours.append(6)
                dates.append(self.f_start)
                hours.append(20)

                return pd.DataFrame({"date": dates, "hour": hours})

            if call_type == "Nestling":
                dates = [self.n_start + timedelta(days=i) for i in range(5)]
                hours = [12] * 5

                dates.append(self.hatch_date)
                hours.append(12)
                dates.append(self.n_start)
                hours.append(21)

                return pd.DataFrame({"date": dates, "hour": hours})

            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Female_Detection_Recordings"], 10)
        self.assertEqual(result["Nestling_Detection_Recordings"], 5)
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
        mock_detections,
    ) -> None:
        """Female chatter with zero nestling detections yields ARI 0.0."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                return self.female_df(list(range(10)))
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["ARI"], 0.0)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_no_female_calls_with_nestling_calls_returns_nd_no_female_calls(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Nestling detections without female denominator are not a valid ARI."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Nestling":
                return self.nestling_df([0, 1, 2])
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["ARI"], "ND_NO_FEMALE_CALLS")
        self.assertEqual(result["Female_Detection_Recordings"], 0)
        self.assertEqual(result["Nestling_Detection_Recordings"], 3)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_no_female_calls_and_no_nestling_calls_returns_nd_no_female_calls(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """No female denominator is ND, not biological abandonment."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10
        mock_detections.return_value = pd.DataFrame()

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["ARI"], "ND_NO_FEMALE_CALLS")

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_invalid_breeding_types_return_nd_status(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Invalid breeding types should not be scored as numeric ARI."""
        rows = [
            ({"Breeding Type": "Unknown"}, "Unknown"),
            ({"Breeding Type": "Complex"}, "Complex"),
            ({"Breeding Type": "Simple", "Complex Types": "Asynchronous"}, "Asynchronous"),
            ({"Breeding Type": "Asynchronous"}, "Asynchronous"),
        ]

        for row, expected_comment in rows:
            with self.subTest(row=row):
                result = self.metric.calculate_row(row, self.hatch_date, self.site_name)
                self.assertEqual(result["ARI"], "ND_INVALID_BREEDING_TYPE")
                self.assertIn(expected_comment, result["Comment"])

        mock_bounds.assert_not_called()
        mock_totals.assert_not_called()
        mock_days_count.assert_not_called()
        mock_detections.assert_not_called()

    def test_missing_hatch_date_returns_missing_dates_status(self) -> None:
        result = self.metric.calculate_row({"Breeding Type": "Simple"}, None, self.site_name)

        self.assertEqual(result["ARI"], "NHD")
        self.assertEqual(result["Earliest_Rec"], "NHD")
        self.assertEqual(result["Latest_Rec"], "NHD")
        self.assertIn("No valid hatch date", result["Comment"])

    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_missing_deployment_logs_returns_missing_dates_status(self, mock_bounds) -> None:
        mock_bounds.return_value = (None, None)

        result = self.metric.calculate_row({"Breeding Type": "Simple"}, self.hatch_date, self.site_name)

        self.assertEqual(result["ARI"], "ND_MISSING_DATES")
        self.assertIn("No recording deployment logs found in parquet", result["Comment"])

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_insufficient_incubation_span_returns_nd_insufficient_days(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        rec_start = self.hatch_date - timedelta(days=2)
        mock_bounds.return_value = (rec_start, date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                return pd.DataFrame(
                    {
                        "date": [rec_start, rec_start + timedelta(days=1)],
                        "hour": [12, 12],
                    }
                )
            if call_type == "Nestling":
                return self.nestling_df([0, 1, 2])
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Incubation_Days"], 2)
        self.assertEqual(result["ARI"], "ND_INSUFFICIENT_DAYS")
        self.assertIn("Incubation days less than 4", result["Comment"])

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_recording_days_count")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_insufficient_nestling_span_returns_nd_insufficient_days(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        rec_stop = self.n_start + timedelta(days=2)
        mock_bounds.return_value = (date(2024, 5, 1), rec_stop)
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                return self.female_df(list(range(10)))
            if call_type == "Nestling":
                return self.nestling_df([0, 1, 2])
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Nestling_Days"], 3)
        self.assertEqual(result["ARI"], "ND_INSUFFICIENT_DAYS")
        self.assertIn("Nestling days less than 4", result["Comment"])


class TestAcousticReproductiveIndexPostProcess(unittest.TestCase):
    def test_post_process_classifies_numeric_ari_and_ignores_nd_statuses(self) -> None:
        metric = AcousticReproductiveIndex()
        df = pd.DataFrame(
            {
                "ARI": [
                    0.0,
                    0.1,
                    0.25,
                    0.9,
                    "ND_NO_FEMALE_CALLS",
                    "ND_INVALID_BREEDING_TYPE",
                ]
            }
        )

        result = metric.post_process(df.copy())

        self.assertAlmostEqual(metric.calculated_cutoff, 0.175)
        self.assertEqual(result.loc[0, "Calculated_Outcome"], "Abandoned")
        self.assertEqual(result.loc[1, "Calculated_Outcome"], "Partially Abandoned")
        self.assertEqual(result.loc[2, "Calculated_Outcome"], "Successful")
        self.assertEqual(result.loc[3, "Calculated_Outcome"], "Successful")
        self.assertEqual(result.loc[4, "Calculated_Outcome"], "Unknown")
        self.assertEqual(result.loc[5, "Calculated_Outcome"], "Unknown")

    def test_post_process_uses_default_cutoff_when_fewer_than_two_valid_values(self) -> None:
        metric = AcousticReproductiveIndex()
        df = pd.DataFrame({"ARI": [0.1, "ND_NO_FEMALE_CALLS"]})

        result = metric.post_process(df.copy())

        self.assertEqual(metric.calculated_cutoff, 0.15)
        self.assertEqual(result.loc[0, "Calculated_Outcome"], "Partially Abandoned")
        self.assertEqual(result.loc[1, "Calculated_Outcome"], "Unknown")


class TestFledglingMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = FledglingMetrics()
        self.hatch_date = date(2024, 5, 15)
        self.site_name = "2024 Baja Rancho Cinega Redonda 1"
        self.fledge_start = self.hatch_date + timedelta(days=self.metric.FLEDGLING_OFFSET_DAYS)
        self.fledge_end = self.hatch_date + timedelta(days=self.metric.FLEDGLING_LATEST_DAY_OFFSET)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_fledglings_success(self, mock_bounds, mock_totals, mock_detections) -> None:
        """Fledgling detections summarize as recording-level proportions."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100

        dates = [self.fledge_start, self.fledge_start + timedelta(days=2)]
        mock_detections.return_value = pd.DataFrame({"date": dates, "hour": [12, 12]})

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Fledglings_Present"], "Yes")
        self.assertEqual(result["Fledgling_Detection_Recordings"], 2)
        self.assertEqual(result["Fledgling_Days"], 5)
        self.assertEqual(result["Avg_Fledgling_Calls_Day"], 0.02)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_fledgling_window_is_clipped_by_recording_stop(
        self,
        mock_bounds,
        mock_totals,
        mock_detections,
    ) -> None:
        rec_stop = self.fledge_start + timedelta(days=2)
        mock_bounds.return_value = (date(2024, 5, 1), rec_stop)
        mock_totals.return_value = 100

        mock_detections.return_value = pd.DataFrame(
            {
                "date": [
                    self.fledge_start,
                    rec_stop,
                    rec_stop + timedelta(days=1),
                    self.fledge_start,
                ],
                "hour": [12, 12, 12, 20],
            }
        )

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Fledglings_Present"], "Yes")
        self.assertEqual(result["Fledgling_Days"], 3)
        self.assertEqual(result["Fledgling_Detection_Recordings"], 2)
        self.assertEqual(result["Avg_Fledgling_Calls_Day"], 0.02)
        mock_totals.assert_called_once_with(self.site_name, self.fledge_start, rec_stop)

    @patch("acoustic_ratios.get_raw_validated_detections")
    @patch("acoustic_ratios.get_total_recordings")
    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_no_fledgling_detections_still_reports_window_span(
        self,
        mock_bounds,
        mock_totals,
        mock_detections,
    ) -> None:
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_detections.return_value = pd.DataFrame()

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Fledglings_Present"], "No")
        self.assertEqual(result["Fledgling_Days"], 5)
        self.assertEqual(result["Fledgling_Detection_Recordings"], 0)

    def test_missing_hatch_date_returns_default_fledgling_result(self) -> None:
        result = self.metric.calculate_row({}, None, self.site_name)

        self.assertEqual(result["Fledglings_Present"], "No")
        self.assertEqual(result["Fledgling_Days"], 0)
        self.assertEqual(result["Fledgling_Detection_Recordings"], 0)

    @patch("acoustic_ratios.get_site_recording_bounds")
    def test_missing_deployment_logs_returns_default_fledgling_result(self, mock_bounds) -> None:
        mock_bounds.return_value = (None, None)

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result["Fledglings_Present"], "No")
        self.assertEqual(result["Fledgling_Days"], 0)
        self.assertEqual(result["Fledgling_Detection_Recordings"], 0)


class TestDataLoader(unittest.TestCase):
    @patch("acoustic_ratios.load_recordings_parquet")
    def test_get_total_recordings(self, mock_load) -> None:
        """get_total_recordings respects site matching and daylight-hour filters."""
        mock_load.return_value = pd.DataFrame(
            {
                "site_clean": ["testsite", "testsite", "testsite", "othersite"],
                "date_parsed": [
                    date(2024, 5, 1),
                    date(2024, 5, 1),
                    date(2024, 5, 2),
                    date(2024, 5, 1),
                ],
                "hour": [8, 21, 10, 8],
                "n_recordings": [5, 100, 3, 999],
            }
        )

        total = get_total_recordings("TestSite", date(2024, 5, 1), date(2024, 5, 2))

        self.assertEqual(total, 8)

    @patch("acoustic_ratios.load_recordings_parquet")
    def test_get_total_recordings_returns_zero_for_missing_site(self, mock_load) -> None:
        mock_load.return_value = pd.DataFrame(
            {
                "site_clean": ["othersite"],
                "date_parsed": [date(2024, 5, 1)],
                "hour": [8],
                "n_recordings": [5],
            }
        )

        total = get_total_recordings("TestSite", date(2024, 5, 1), date(2024, 5, 2))

        self.assertEqual(total, 0)

    @patch("acoustic_ratios.load_recordings_parquet")
    def test_get_site_recording_bounds_case_insensitive_site_match(self, mock_load) -> None:
        mock_load.return_value = pd.DataFrame(
            {
                "site_clean": ["testsite", "testsite", "othersite"],
                "date_parsed": [
                    date(2024, 5, 3),
                    date(2024, 5, 1),
                    date(2024, 4, 1),
                ],
            }
        )

        bounds = get_site_recording_bounds(" TestSite ")

        self.assertEqual(bounds, (date(2024, 5, 1), date(2024, 5, 3)))

    @patch("acoustic_ratios.load_recordings_parquet")
    def test_get_site_recording_bounds_missing_site(self, mock_load) -> None:
        mock_load.return_value = pd.DataFrame(
            {
                "site_clean": ["othersite"],
                "date_parsed": [date(2024, 5, 1)],
            }
        )

        bounds = get_site_recording_bounds("TestSite")

        self.assertEqual(bounds, (None, None))


class TestRawValidatedDetectionsLoader(unittest.TestCase):
    def setUp(self) -> None:
        get_raw_validated_detections.cache_clear()

    def tearDown(self) -> None:
        get_raw_validated_detections.cache_clear()

    def test_loader_keeps_only_validated_present_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            site_dir = tmp_path / "Test Site"
            site_dir.mkdir()

            csv_path = site_dir / "Test Site Female detections.csv"
            pd.DataFrame(
                {
                    "Year": [2024, 2024, 2024],
                    "Month": [5, 5, 5],
                    "Day": [1, 2, 3],
                    "Hour": [7, 12, 19],
                    "Validated": [" Present ", "absent", "PRESENT"],
                }
            ).to_csv(csv_path, index=False)

            with patch("acoustic_ratios.PMJ_DIR", tmp_path):
                result = get_raw_validated_detections("Test Site", "Female")

        self.assertEqual(len(result), 2)
        self.assertEqual(result["date"].tolist(), [date(2024, 5, 1), date(2024, 5, 3)])
        self.assertEqual(result["hour"].tolist(), [7, 19])

    @patch("builtins.print")
    def test_loader_returns_empty_when_no_matching_file(self, mock_print) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "Test Site").mkdir()

            with patch("acoustic_ratios.PMJ_DIR", tmp_path):
                result = get_raw_validated_detections("Test Site", "Female")

        self.assertTrue(result.empty)
        mock_print.assert_called_once()

    @patch("builtins.print")
    def test_loader_returns_empty_when_multiple_matching_files(self, mock_print) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            site_dir = tmp_path / "Test Site"
            site_dir.mkdir()

            for idx in [1, 2]:
                pd.DataFrame(
                    {
                        "Year": [2024],
                        "Month": [5],
                        "Day": [1],
                        "Hour": [12],
                        "Validated": ["Present"],
                    }
                ).to_csv(site_dir / f"Female detections {idx}.csv", index=False)

            with patch("acoustic_ratios.PMJ_DIR", tmp_path):
                result = get_raw_validated_detections("Test Site", "Female")

        self.assertTrue(result.empty)
        mock_print.assert_called_once()
    
    @patch("builtins.print")
    def test_loader_returns_empty_for_invalid_hour_value(self, mock_print) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            site_dir = tmp_path / "Test Site"
            site_dir.mkdir()

            pd.DataFrame(
                {
                    "Year": [2024],
                    "Month": [5],
                    "Day": [1],
                    "Hour": ["not_an_hour"],
                    "Validated": ["Present"],
                }
            ).to_csv(site_dir / "Female detections.csv", index=False)

            with patch("acoustic_ratios.PMJ_DIR", tmp_path):
                result = get_raw_validated_detections("Test Site", "Female")

        self.assertTrue(result.empty)
        mock_print.assert_called_once()



if __name__ == "__main__":
    unittest.main()
