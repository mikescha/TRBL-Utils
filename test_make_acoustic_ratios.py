import unittest
from datetime import date, timedelta
from unittest.mock import call, patch

import pandas as pd

from common import (
    COL_BREEDING_TYPE,
    COL_COMMENT,
    COL_COMPLEX_TYPES,
    COL_HATCH_DATE,
    COL_OUTCOME,
    OUTCOME_ABANDONED,
    OUTCOME_NO_COLONY,
    OUTCOME_NO_TRBL,
    OUTCOME_PARTIALLY_ABANDONED,
    OUTCOME_SUCCESSFUL,
    OUTCOME_UNKNOWN,
    STATUS_ND,
    normalize_output_date_columns,
)
from make_acoustic_ratios import (
    ARI_CLASS_HIGH_OFFSPRING_ACTIVITY,
    ARI_CLASS_NO_OFFSPRING_EVIDENCE,
    ARI_CLASS_NOT_SCORABLE,
    ARI_CLASS_REDUCED_OFFSPRING_ACTIVITY,
    ARI_CONFIDENCE_LOW_FEMALE_DENOMINATOR,
    ARI_CONFIDENCE_MODERATE_FEMALE_DENOMINATOR,
    ARI_CONFIDENCE_NO_FEMALE_DENOMINATOR,
    ARI_CONFIDENCE_NOT_EVALUATED,
    ARI_CONFIDENCE_STABLE_FEMALE_DENOMINATOR,
    ARI_STATUS_INSUFFICIENT_DAYS,
    ARI_STATUS_INVALID_BREEDING_TYPE,
    ARI_STATUS_MISSING_DATES,
    ARI_STATUS_NHD,
    ARI_STATUS_NO_FEMALE_CALLS,
    ARI_STATUS_OK,
    BREEDING_TYPE_ADDITIONS,
    BREEDING_TYPE_ASYNCHRONOUS,
    BREEDING_TYPE_COMPLEX,
    BREEDING_TYPE_SEQUENTIAL,
    BREEDING_TYPE_SIMPLE,
    BREEDING_TYPE_UNKNOWN,
    COL_ARI,
    COL_ARI_CLASS,
    COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE,
    COL_ARI_REVIEW_BECAUSE,
    COL_ARI_STATUS,
    COL_ARI_WINDOW_FEMALE_END,
    COL_ARI_WINDOW_FEMALE_START,
    COL_ARI_WINDOW_NESTLING_END,
    COL_ARI_WINDOW_NESTLING_START,
    COL_FEMALE_DETECTION_RATE,
    COL_FEMALE_DETECTION_RECORDINGS,
    COL_FLEDGLING_DAYS,
    COL_FLEDGLING_DETECTION_RATE,
    COL_FLEDGLING_DETECTION_RECORDINGS,
    COL_FLEDGLINGS_PRESENT,
    COL_INCUBATION_DAYS,
    COL_NESTLING_DAYS,
    COL_NESTLING_DETECTION_RATE,
    COL_NESTLING_DETECTION_RECORDINGS,
    PUBLICATION_COLUMNS,
    AcousticReproductiveIndex,
    FledglingMetrics,
    classify_female_denominator_confidence,
    clear_raw_validated_detections_cache,
    filter_by_datetime_bounds,
    get_raw_validated_detections,
    get_site_recording_bounds,
    get_total_recordings,
    inclusive_day_span,
    normalize_hatch_date_value,
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

        female_start_offset = self.metric.FEMALE_OFFSET_DAYS + self.metric.DAYS_TO_COUNT - 1
        self.f_start = self.hatch_date - timedelta(days=female_start_offset)
        self.f_end = self.hatch_date - timedelta(days=self.metric.FEMALE_OFFSET_DAYS)

        self.n_start = self.hatch_date + timedelta(days=self.metric.NESTLING_OFFSET_DAYS)
        self.n_end = self.n_start + timedelta(days=self.metric.DAYS_TO_COUNT - 1)

    def assert_zero_ari_metrics(self, result: dict[str, object]) -> None:
        self.assertEqual(result[COL_ARI], "")
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_NHD)

        self.assertEqual(result[COL_ARI_WINDOW_FEMALE_START], STATUS_ND)
        self.assertEqual(result[COL_ARI_WINDOW_FEMALE_END], STATUS_ND)
        self.assertEqual(result[COL_ARI_WINDOW_NESTLING_START], STATUS_ND)
        self.assertEqual(result[COL_ARI_WINDOW_NESTLING_END], STATUS_ND)

        self.assertEqual(result[COL_INCUBATION_DAYS], 0)
        self.assertEqual(result[COL_NESTLING_DAYS], 0)

        self.assertEqual(result[COL_FEMALE_DETECTION_RECORDINGS], 0)
        self.assertEqual(result[COL_FEMALE_DETECTION_RATE], 0.0)

        self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 0)
        self.assertEqual(result[COL_NESTLING_DETECTION_RATE], 0.0)

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

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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
        mock_days_count.return_value = self.metric.DAYS_TO_COUNT

        def side_effect_detections(site, call_type):
            if call_type == "Female":
                return self.female_df(list(range(self.metric.DAYS_TO_COUNT)))
            if call_type == "Nestling":
                return self.nestling_df(list(range(5)))
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        row = {COL_BREEDING_TYPE: "Simple"}
        result = self.metric.calculate_row(row, self.hatch_date, self.site_name)

        self.assertEqual(result[COL_INCUBATION_DAYS], self.metric.DAYS_TO_COUNT)
        self.assertEqual(result[COL_FEMALE_DETECTION_RECORDINGS], 7)
        self.assertEqual(result[COL_FEMALE_DETECTION_RATE], 0.07)

        self.assertEqual(result[COL_NESTLING_DAYS], self.metric.DAYS_TO_COUNT)
        self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 5)
        self.assertEqual(result[COL_NESTLING_DETECTION_RATE], 0.05)

        self.assertEqual(result[COL_ARI], 0.714)
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_OK)

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_nestling_window_is_hatch_plus_5_through_hatch_plus_11(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Nestling ARI window is 7 inclusive days: hatch+5 through hatch+11.

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

        self.assertEqual(result[COL_NESTLING_DAYS], 7)
        self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 2)
        self.assertEqual(result[COL_NESTLING_DETECTION_RATE], 0.02)
        self.assertEqual(result[COL_ARI], 0.286)
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_OK)

        self.assertIn(
            call(self.site_name, self.n_start, self.n_end),
            mock_totals.call_args_list,
        )

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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

        self.assertEqual(result[COL_NESTLING_DAYS], 5)
        self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 2)
        self.assertIn(
            call(self.site_name, self.n_start, rec_stop),
            mock_totals.call_args_list,
        )

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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

        self.assertEqual(result[COL_NESTLING_DAYS], 7)
        self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 1)

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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

        self.assertEqual(result[COL_FEMALE_DETECTION_RECORDINGS], 7)
        self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 5)
        self.assertEqual(result[COL_ARI], 0.714)
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_OK)

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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

        self.assertEqual(result[COL_ARI], 0.0)
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_OK)        

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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

        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_NO_FEMALE_CALLS)
        self.assertEqual(result[COL_FEMALE_DETECTION_RECORDINGS], 0)
        self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 3)

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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

        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_NO_FEMALE_CALLS)


    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_breeding_types_with_no_hatch_date_return_zero_metrics(
        self,
        mock_bounds,
        mock_totals,
        mock_detections,
    ) -> None:
        """If hatch date is NHD, all breeding types should return empty/zero ARI metrics."""
        cases = [
            {COL_BREEDING_TYPE: BREEDING_TYPE_SIMPLE, COL_COMPLEX_TYPES: ""},
            {COL_BREEDING_TYPE: BREEDING_TYPE_SEQUENTIAL, COL_COMPLEX_TYPES: ""},
            {COL_BREEDING_TYPE: BREEDING_TYPE_ADDITIONS, COL_COMPLEX_TYPES: ""},
            {COL_BREEDING_TYPE: BREEDING_TYPE_ASYNCHRONOUS, COL_COMPLEX_TYPES: ""},
            {
                COL_BREEDING_TYPE: BREEDING_TYPE_COMPLEX,
                COL_COMPLEX_TYPES: "Asynchronous, Sequential",
            },
            {COL_BREEDING_TYPE: BREEDING_TYPE_UNKNOWN, COL_COMPLEX_TYPES: ""},
        ]

        for row in cases:
            with self.subTest(row=row):
                result = self.metric.calculate_row(row, None, self.site_name)
                self.assert_zero_ari_metrics(result)

        mock_bounds.assert_not_called()
        mock_totals.assert_not_called()
        mock_detections.assert_not_called()


    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_no_colony_and_no_trbl_outcomes_return_zero_metrics(
        self,
        mock_bounds,
        mock_totals,
        mock_detections,
    ) -> None:
        """No Colony and No TRBL rows should not calculate ARI metrics."""
        cases = [
            {
                COL_OUTCOME: OUTCOME_NO_COLONY,
                COL_BREEDING_TYPE: BREEDING_TYPE_SIMPLE,
                COL_COMPLEX_TYPES: "",
                COL_HATCH_DATE: "2024-05-15",
            },
            {
                COL_OUTCOME: OUTCOME_NO_TRBL,
                COL_BREEDING_TYPE: BREEDING_TYPE_SIMPLE,
                COL_COMPLEX_TYPES: "",
                COL_HATCH_DATE: "2024-05-15",
            },
        ]

        for row in cases:
            with self.subTest(row=row):
                result = self.metric.calculate_row(row, self.hatch_date, self.site_name)
                self.assert_zero_ari_metrics(result)

        mock_bounds.assert_not_called()
        mock_totals.assert_not_called()
        mock_detections.assert_not_called()


    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_standard_outcomes_with_hatch_date_do_not_suppress_metric_calculation(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Standard manual outcomes should not prevent ARI metric calculation."""
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

        for outcome in [
            OUTCOME_ABANDONED,
            OUTCOME_PARTIALLY_ABANDONED,
            OUTCOME_SUCCESSFUL,
            OUTCOME_UNKNOWN,
        ]:
            with self.subTest(outcome=outcome):
                result = self.metric.calculate_row(
                    {
                        COL_OUTCOME: outcome,
                        COL_BREEDING_TYPE: BREEDING_TYPE_SIMPLE,
                        COL_COMPLEX_TYPES: "",
                    },
                    self.hatch_date,
                    self.site_name,
                )

                self.assertEqual(result[COL_ARI_WINDOW_FEMALE_START], "2024-05-06")
                self.assertEqual(result[COL_ARI_WINDOW_FEMALE_END], "2024-05-12")
                self.assertEqual(result[COL_ARI_WINDOW_NESTLING_START], "2024-05-20")
                self.assertEqual(result[COL_ARI_WINDOW_NESTLING_END], "2024-05-26")

                self.assertEqual(result[COL_INCUBATION_DAYS], 7)
                self.assertEqual(result[COL_NESTLING_DAYS], 7)
                self.assertEqual(result[COL_FEMALE_DETECTION_RECORDINGS], 7)
                self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 5)
                self.assertEqual(result[COL_ARI], 0.714)
                

    def test_missing_hatch_date_returns_missing_dates_status(self) -> None:
        result = self.metric.calculate_row({COL_BREEDING_TYPE: "Simple"}, None, self.site_name)

        self.assertEqual(result[COL_ARI], "")
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_NHD)
        self.assertIn("No valid hatch date", result[COL_COMMENT])


    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_non_simple_breeding_types_with_hatch_date_calculate_metrics_but_not_numeric_ari(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Non-simple breeding types calculate diagnostics but suppress numeric ARI."""
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

        cases = [
            (
                {
                    COL_BREEDING_TYPE: BREEDING_TYPE_ADDITIONS,
                    COL_COMPLEX_TYPES: "",
                },
                BREEDING_TYPE_ADDITIONS,
            ),
            (
                {
                    COL_BREEDING_TYPE: BREEDING_TYPE_ASYNCHRONOUS,
                    COL_COMPLEX_TYPES: "",
                },
                BREEDING_TYPE_ASYNCHRONOUS,
            ),
            (
                {
                    COL_BREEDING_TYPE: BREEDING_TYPE_COMPLEX,
                    COL_COMPLEX_TYPES: "Asynchronous, Sequential",
                },
                BREEDING_TYPE_COMPLEX,
            ),
            (
                {
                    COL_BREEDING_TYPE: BREEDING_TYPE_UNKNOWN,
                    COL_COMPLEX_TYPES: "",
                },
                BREEDING_TYPE_UNKNOWN,
            ),
        ]

        for row, expected_comment in cases:
            with self.subTest(row=row):
                result = self.metric.calculate_row(row, self.hatch_date, self.site_name)

                self.assertEqual(result[COL_ARI], "")
                self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_INVALID_BREEDING_TYPE)
                self.assertIn(expected_comment, result[COL_COMMENT])

                self.assertEqual(result[COL_ARI_WINDOW_FEMALE_START], "2024-05-06")
                self.assertEqual(result[COL_ARI_WINDOW_FEMALE_END], "2024-05-12")
                self.assertEqual(result[COL_ARI_WINDOW_NESTLING_START], "2024-05-20")
                self.assertEqual(result[COL_ARI_WINDOW_NESTLING_END], "2024-05-26")

                self.assertEqual(result[COL_INCUBATION_DAYS], 7)
                self.assertEqual(result[COL_NESTLING_DAYS], 7)

                self.assertEqual(result[COL_FEMALE_DETECTION_RECORDINGS], 7)
                self.assertEqual(result[COL_FEMALE_DETECTION_RATE], 0.07)

                self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 5)
                self.assertEqual(result[COL_NESTLING_DETECTION_RATE], 0.05)


    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_missing_deployment_logs_returns_missing_dates_status(self, mock_bounds) -> None:
        mock_bounds.return_value = (None, None)

        result = self.metric.calculate_row({COL_BREEDING_TYPE: "Simple"}, self.hatch_date, self.site_name)

        self.assertEqual(result[COL_ARI], "")
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_MISSING_DATES)
        self.assertIn("No recording deployment logs found in parquet", result[COL_COMMENT])


    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_insufficient_incubation_span_returns_nd_insufficient_days(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        rec_start = self.hatch_date - timedelta(days=4)
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

        self.assertEqual(result[COL_INCUBATION_DAYS], 2)
        self.assertEqual(result[COL_ARI], "")
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_INSUFFICIENT_DAYS)
        self.assertIn("Incubation days less than 4", result[COL_COMMENT])


    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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

        self.assertEqual(result[COL_NESTLING_DAYS], 3)
        self.assertEqual(result[COL_ARI], "")
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_INSUFFICIENT_DAYS)
        self.assertIn("Nestling days less than 4", result[COL_COMMENT])


    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_recording_days_count")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_invalid_breeding_type_with_no_female_calls_does_not_divide_by_zero(
        self,
        mock_bounds,
        mock_totals,
        mock_days_count,
        mock_detections,
    ) -> None:
        """Invalid breeding rows should not attempt numeric ARI division."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100
        mock_days_count.return_value = 10

        def side_effect_detections(site, call_type):
            if call_type == "Nestling":
                return self.nestling_df([0, 1, 2])
            return pd.DataFrame()

        mock_detections.side_effect = side_effect_detections

        result = self.metric.calculate_row(
            {COL_BREEDING_TYPE: "Complex"},
            self.hatch_date,
            self.site_name,
        )

        self.assertEqual(result[COL_ARI], "")
        self.assertEqual(result[COL_ARI_STATUS], ARI_STATUS_INVALID_BREEDING_TYPE)
        self.assertEqual(result[COL_FEMALE_DETECTION_RECORDINGS], 0)
        self.assertEqual(result[COL_FEMALE_DETECTION_RATE], 0.0)
        self.assertEqual(result[COL_NESTLING_DETECTION_RECORDINGS], 3)
        self.assertEqual(result[COL_NESTLING_DETECTION_RATE], 0.03)


class TestAcousticReproductiveIndexPostProcess(unittest.TestCase):
    def test_post_process_assigns_publication_facing_ari_status_and_class(self) -> None:
        metric = AcousticReproductiveIndex()
        df = pd.DataFrame(
            {
                COL_ARI: [0.0, 0.0, 0.25, 0.5, 0.501, ""],
                COL_ARI_STATUS: [
                    ARI_STATUS_OK, 
                    ARI_STATUS_OK, 
                    ARI_STATUS_OK, 
                    ARI_STATUS_OK, 
                    ARI_STATUS_OK, 
                    ARI_STATUS_NO_FEMALE_CALLS],
                COL_FLEDGLING_DETECTION_RECORDINGS: [0, 2, 0, 0, 0, 0],
                COL_FEMALE_DETECTION_RECORDINGS: [20, 20, 20, 20, 20, 0],
            }
        )

        result = metric.post_process(df.copy())

        self.assertEqual(result.loc[0, COL_ARI_CLASS], ARI_CLASS_NO_OFFSPRING_EVIDENCE)
        self.assertEqual(result.loc[1, COL_ARI_CLASS], ARI_CLASS_REDUCED_OFFSPRING_ACTIVITY)
        self.assertEqual(result.loc[2, COL_ARI_CLASS], ARI_CLASS_REDUCED_OFFSPRING_ACTIVITY)
        self.assertEqual(result.loc[3, COL_ARI_CLASS], ARI_CLASS_REDUCED_OFFSPRING_ACTIVITY)
        self.assertEqual(result.loc[4, COL_ARI_CLASS], ARI_CLASS_HIGH_OFFSPRING_ACTIVITY)
        self.assertEqual(result.loc[5, COL_ARI_CLASS], ARI_CLASS_NOT_SCORABLE)


    def test_post_process_classifies_female_denominator_confidence(self) -> None:
        metric = AcousticReproductiveIndex()
        df = pd.DataFrame(
            {
                COL_ARI: [
                    "",
                    "",
                    0.25,
                    0.25,
                    0.25,
                ],
                COL_ARI_STATUS: [
                    ARI_STATUS_NHD,
                    ARI_STATUS_NO_FEMALE_CALLS,
                    ARI_STATUS_OK,
                    ARI_STATUS_OK,
                    ARI_STATUS_OK,
                ],
                COL_FLEDGLING_DETECTION_RECORDINGS: [0, 0, 0, 0, 0],
                COL_FEMALE_DETECTION_RECORDINGS: [pd.NA, 0, 5, 6, 21],
            }
        )

        result = metric.post_process(df.copy())

        self.assertEqual(
            result.loc[0, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_NOT_EVALUATED,
        )
        self.assertEqual(
            result.loc[1, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_NO_FEMALE_DENOMINATOR,
        )
        self.assertEqual(
            result.loc[2, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_LOW_FEMALE_DENOMINATOR,
        )
        self.assertEqual(
            result.loc[3, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_MODERATE_FEMALE_DENOMINATOR,
        )
        self.assertEqual(
            result.loc[4, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_STABLE_FEMALE_DENOMINATOR,
        )


    def test_post_process_handles_missing_female_denominator_column(self) -> None:
        metric = AcousticReproductiveIndex()
        df = pd.DataFrame(
            {
                COL_ARI: [0.0, 0.25, ""],
                COL_ARI_STATUS: [ARI_STATUS_OK, ARI_STATUS_OK, ARI_STATUS_NHD],
                COL_FLEDGLING_DETECTION_RECORDINGS: [0, 0, 0],
            }
        )

        result = metric.post_process(df.copy())

        self.assertEqual(
            result.loc[0, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_NOT_EVALUATED,
        )
        self.assertEqual(
            result.loc[1, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_NOT_EVALUATED,
        )
        self.assertEqual(
            result.loc[2, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_NOT_EVALUATED,
        )

    def test_post_process_keeps_denominator_confidence_for_invalid_breeding_type_rows(
        self,
    ) -> None:
        """Invalid breeding type rows can still have meaningful denominator diagnostics."""
        metric = AcousticReproductiveIndex()
        df = pd.DataFrame(
            {
                COL_ARI: [""],
                COL_ARI_STATUS: [ARI_STATUS_INVALID_BREEDING_TYPE],
                COL_FEMALE_DETECTION_RECORDINGS: [5],
                COL_FLEDGLING_DETECTION_RECORDINGS: [0],
            }
        )

        result = metric.post_process(df.copy())

        self.assertEqual(
            result.loc[0, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_LOW_FEMALE_DENOMINATOR,
        )
        
    def test_post_process_marks_denominator_confidence_not_evaluated_when_ari_windows_unavailable(
        self,
    ) -> None:
        """Rows without ARI windows should not report female denominator confidence as evaluated."""
        metric = AcousticReproductiveIndex()
        df = pd.DataFrame(
            {
                COL_ARI: ["", "", ""],
                COL_ARI_STATUS: [
                    ARI_STATUS_NHD,
                    ARI_STATUS_MISSING_DATES,
                    ARI_STATUS_NO_FEMALE_CALLS,
                ],
                COL_FEMALE_DETECTION_RECORDINGS: [0, 0, 0],
                COL_FLEDGLING_DETECTION_RECORDINGS: [0, 0, 0],
            }
        )

        result = metric.post_process(df.copy())

        self.assertEqual(
            result.loc[0, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_NOT_EVALUATED,
        )
        self.assertEqual(
            result.loc[1, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_NOT_EVALUATED,
        )
        self.assertEqual(
            result.loc[2, COL_ARI_FEMALE_DENOMINATOR_CONFIDENCE],
            ARI_CONFIDENCE_NO_FEMALE_DENOMINATOR,
        )


class TestFledglingMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.metric = FledglingMetrics()
        self.hatch_date = date(2024, 5, 15)
        self.site_name = "2024 Baja Rancho Cinega Redonda 1"
        self.fledge_start = self.hatch_date + timedelta(days=self.metric.FLEDGLING_OFFSET_DAYS)
        self.fledge_end = self.hatch_date + timedelta(days=self.metric.FLEDGLING_LATEST_DAY_OFFSET)

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_fledglings_success(self, mock_bounds, mock_totals, mock_detections) -> None:
        """Fledgling detections summarize as recording-level proportions."""
        mock_bounds.return_value = (date(2024, 5, 1), date(2024, 5, 30))
        mock_totals.return_value = 100

        dates = [self.fledge_start, self.fledge_start + timedelta(days=2)]
        mock_detections.return_value = pd.DataFrame({"date": dates, "hour": [12, 12]})

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result[COL_FLEDGLINGS_PRESENT], "Yes")
        self.assertEqual(result[COL_FLEDGLING_DETECTION_RECORDINGS], 2)
        self.assertEqual(result[COL_FLEDGLING_DAYS], 7)
        self.assertEqual(result[COL_FLEDGLING_DETECTION_RATE], 0.02)

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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

        self.assertEqual(result[COL_FLEDGLINGS_PRESENT], "Yes")
        self.assertEqual(result[COL_FLEDGLING_DAYS], 3)
        self.assertEqual(result[COL_FLEDGLING_DETECTION_RECORDINGS], 2)
        self.assertEqual(result[COL_FLEDGLING_DETECTION_RATE], 0.02)
        mock_totals.assert_called_once_with(self.site_name, self.fledge_start, rec_stop)

    @patch("make_acoustic_ratios.get_raw_validated_detections")
    @patch("make_acoustic_ratios.get_total_recordings")
    @patch("make_acoustic_ratios.get_site_recording_bounds")
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

        self.assertEqual(result[COL_FLEDGLINGS_PRESENT], "No")
        self.assertEqual(result[COL_FLEDGLING_DAYS], 7)
        self.assertEqual(result[COL_FLEDGLING_DETECTION_RECORDINGS], 0)

    def test_missing_hatch_date_returns_default_fledgling_result(self) -> None:
        result = self.metric.calculate_row({}, None, self.site_name)

        self.assertEqual(result[COL_FLEDGLINGS_PRESENT], "No")
        self.assertEqual(result[COL_FLEDGLING_DAYS], 0)
        self.assertEqual(result[COL_FLEDGLING_DETECTION_RECORDINGS], 0)

    @patch("make_acoustic_ratios.get_site_recording_bounds")
    def test_missing_deployment_logs_returns_default_fledgling_result(self, mock_bounds) -> None:
        mock_bounds.return_value = (None, None)

        result = self.metric.calculate_row({}, self.hatch_date, self.site_name)

        self.assertEqual(result[COL_FLEDGLINGS_PRESENT], "No")
        self.assertEqual(result[COL_FLEDGLING_DAYS], 0)
        self.assertEqual(result[COL_FLEDGLING_DETECTION_RECORDINGS], 0)


class TestDataLoader(unittest.TestCase):
    @patch("make_acoustic_ratios.load_recordings_parquet")
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

    @patch("make_acoustic_ratios.load_recordings_parquet")
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

    @patch("make_acoustic_ratios.load_recordings_parquet")
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

    @patch("make_acoustic_ratios.load_recordings_parquet")
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
        clear_raw_validated_detections_cache()

    def tearDown(self) -> None:
        clear_raw_validated_detections_cache()

    def test_loader_keeps_only_validated_present_rows(self) -> None:
        source_df = pd.DataFrame(
            {
                "year": [2024, 2024, 2024],
                "month": [5, 5, 5],
                "day": [1, 2, 3],
                "hour": [7, 12, 19],
                "validated": [" Present ", "absent", "PRESENT"],
            }
        )

        with patch(
            "make_acoustic_ratios.load_pmj_subset_from_parquet",
            return_value=source_df,
        ):
            result = get_raw_validated_detections("Test Site", "Female")

        self.assertEqual(len(result), 2)
        self.assertEqual(result["date"].tolist(), [date(2024, 5, 1), date(2024, 5, 3)])
        self.assertEqual(result["hour"].tolist(), [7, 19])

    def test_loader_returns_empty_when_parquet_subset_is_empty(self) -> None:
        source_df = pd.DataFrame(
            columns=["year", "month", "day", "hour", "validated"]
        )

        with patch(
            "make_acoustic_ratios.load_pmj_subset_from_parquet",
            return_value=source_df,
        ):
            result = get_raw_validated_detections("Test Site", "Female")

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["date", "hour"])

    def test_loader_returns_empty_when_no_present_rows(self) -> None:
        source_df = pd.DataFrame(
            {
                "year": [2024, 2024],
                "month": [5, 5],
                "day": [1, 2],
                "hour": [7, 12],
                "validated": ["absent", "(not validated)"],
            }
        )

        with patch(
            "make_acoustic_ratios.load_pmj_subset_from_parquet",
            return_value=source_df,
        ):
            result = get_raw_validated_detections("Test Site", "Female")

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["date", "hour"])

    @patch("builtins.print")
    def test_loader_returns_empty_for_missing_required_column(self, mock_print) -> None:
        source_df = pd.DataFrame(
            {
                "year": [2024],
                "month": [5],
                "day": [1],
                # Missing hour
                "validated": ["present"],
            }
        )

        with patch(
            "make_acoustic_ratios.load_pmj_subset_from_parquet",
            return_value=source_df,
        ):
            result = get_raw_validated_detections("Test Site", "Female")

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["date", "hour"])
        mock_print.assert_called_once()

    def test_loader_returns_empty_for_invalid_hour_value(self) -> None:
        source_df = pd.DataFrame(
            {
                "year": [2024],
                "month": [5],
                "day": [1],
                "hour": ["not_an_hour"],
                "validated": ["present"],
            }
        )

        with patch(
            "make_acoustic_ratios.load_pmj_subset_from_parquet",
            return_value=source_df,
        ):
            result = get_raw_validated_detections("Test Site", "Female")

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["date", "hour"])

    def test_loader_returns_defensive_copy_from_cache(self) -> None:
        source_df = pd.DataFrame(
            {
                "year": [2024],
                "month": [5],
                "day": [1],
                "hour": [7],
                "validated": ["present"],
            }
        )

        with patch(
            "make_acoustic_ratios.load_pmj_subset_from_parquet",
            return_value=source_df,
        ):
            first = get_raw_validated_detections("Test Site", "Female")
            first["hour"] = 99

            second = get_raw_validated_detections("Test Site", "Female")

        self.assertEqual(second["hour"].tolist(), [7])


class TestDateNormalization(unittest.TestCase):
    def test_no_hatch_values_normalize_to_nhd(self) -> None:
        values = ["ND", ARI_STATUS_NHD, "", "nan", "n/a", "NA", "inf", "missed", " ~ND "]

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(normalize_hatch_date_value(value), ARI_STATUS_NHD)

    def test_hatch_date_normalization_preserves_approximate_marker(self) -> None:
        self.assertEqual(normalize_hatch_date_value("~5/12/2017"), "~5/12/2017")
        self.assertEqual(normalize_hatch_date_value("~ND"), ARI_STATUS_NHD)

    def test_normalize_output_date_columns_preserves_tilde_and_uses_iso_format(self) -> None:
        df = pd.DataFrame(
            {
                COL_HATCH_DATE: ["~5/12/2017", "5/13/2017", ARI_STATUS_NHD],
            }
        )

        result = normalize_output_date_columns(df, [COL_HATCH_DATE])

        self.assertEqual(
            result[COL_HATCH_DATE].tolist(),
            ["~2017-05-12", "2017-05-13", ARI_STATUS_NHD],
        )

class TestFemaleDenominatorConfidence(unittest.TestCase):
    def test_female_denominator_confidence_tiers(self) -> None:
        cases = [
            (None, "Not evaluated"),
            ("", "Not evaluated"),
            (0, "No female denominator"),
            (1, "Low female denominator"),
            (5, "Low female denominator"),
            (6, "Moderate female denominator"),
            (20, "Moderate female denominator"),
            (21, "Stable female denominator"),
            (100, "Stable female denominator"),
        ]

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(classify_female_denominator_confidence(value), expected)


class TestDailyDetectionDiagnostics(unittest.TestCase):
    @patch("make_acoustic_ratios.get_daily_recordings_by_date")
    def test_build_daily_detection_rows_distinguishes_zero_detections_from_no_recordings(
        self,
        mock_daily_recordings,
    ) -> None:
        from make_acoustic_ratios import build_daily_detection_rows

        start = date(2024, 5, 1)
        end = date(2024, 5, 3)

        mock_daily_recordings.return_value = {
            date(2024, 5, 1): 10,
            date(2024, 5, 2): 10,
            # May 3 intentionally has no recordings.
        }

        detections = pd.DataFrame(
            {
                "date": [date(2024, 5, 1), date(2024, 5, 1), date(2024, 5, 3)],
                "hour": [12, 13, 12],
            }
        )

        rows = build_daily_detection_rows(
            source_row={"Site ID": "S1", "Pulse Name": "p1", "Outcome": "Successful"},
            site_name="Test Site",
            hatch_date=date(2024, 5, 10),
            window_type="Female_Incubation",
            call_type="Female",
            start_date=start,
            end_date=end,
            detections_df=detections,
        )

        self.assertEqual(len(rows), 3)

        self.assertEqual(rows[0]["Total_Recordings"], 10)
        self.assertEqual(rows[0]["Detection_Recordings"], 2)
        self.assertEqual(rows[0]["Detection_Rate"], 0.2)
        self.assertEqual(rows[0]["Had_Recordings"], "Yes")
        self.assertEqual(rows[0]["Had_Detections"], "Yes")

        self.assertEqual(rows[1]["Total_Recordings"], 10)
        self.assertEqual(rows[1]["Detection_Recordings"], 0)
        self.assertEqual(rows[1]["Detection_Rate"], 0.0)
        self.assertEqual(rows[1]["Had_Recordings"], "Yes")
        self.assertEqual(rows[1]["Had_Detections"], "No")

        self.assertEqual(rows[2]["Total_Recordings"], 0)
        self.assertTrue(pd.isna(rows[2]["Detection_Recordings"]))
        self.assertTrue(pd.isna(rows[2]["Detection_Rate"]))
        self.assertEqual(rows[2]["Had_Recordings"], "No")


class TestPublicationOutputSchema(unittest.TestCase):
    def test_outcome_constants_match_expected_values(self) -> None:
        self.assertEqual(OUTCOME_ABANDONED, "Abandoned")
        self.assertEqual(OUTCOME_PARTIALLY_ABANDONED, "Partially Abandoned")
        self.assertEqual(OUTCOME_SUCCESSFUL, "Successful")
        self.assertEqual(OUTCOME_UNKNOWN, "Unknown")


    def test_publication_columns_exclude_diagnostics_and_source_audit_fields(self) -> None:
        forbidden_columns = {
            "Calculated_Outcome",
            "ARI_Cutoff",
            "ARI_Margin_To_Cutoff",
            "Outcome_Mismatch",
            "Outcome_Mismatch_Type",
            "Outcome_Diagnostic",
            "ARI_Diagnostic",
            "Comment",
            "Source Row",
            "Review Status",
            "Review Notes",
            "mcstart",
            "incstart",
            "fledgestart",
            "fledgedisp",
            "abandon",
            "partial abandon",
            COL_ARI_REVIEW_BECAUSE,
        }

        self.assertTrue(forbidden_columns.isdisjoint(set(PUBLICATION_COLUMNS)))
    
        
    def test_publication_columns_match_expected_schema(self) -> None:
        expected_columns = [
            "Site_ID",
            "Site_Name",
            "Pulse_Name",
            "Outcome",
            "ARI_Status",
            "ARI_Class",
            "ARI_Class_Threshold",
            "ARI_Female_Denominator_Confidence",
            "Breeding_Type",
            "Complex_Types",
            "Hatch_Date",
            "Substrate",
            "Colony_Size",
            "Deployment_Start",
            "Deployment_End",
            "ARI_Window_Female_Start",
            "ARI_Window_Female_End",
            "Incubation_Days",
            "ARI_Total_Female_Recordings",
            "Female_Detection_Recordings",
            "Female_Detection_Rate",
            "ARI_Window_Nestling_Start",
            "ARI_Window_Nestling_End",
            "Nestling_Days",
            "ARI_Total_Nestling_Recordings",
            "Nestling_Detection_Recordings",
            "Nestling_Detection_Rate",
            "ARI",
            "Fledgling_Window_Start",
            "Fledgling_Window_End",
            "Fledgling_Days",
            "Fledgling_Total_Recordings",
            "Fledgling_Detection_Recordings",
            "Fledgling_Detection_Rate",
            "Fledglings_Present",
            "Latest_Fledgling_Rec",
        ]


        self.assertEqual(PUBLICATION_COLUMNS, expected_columns)


if __name__ == "__main__":
    unittest.main()
