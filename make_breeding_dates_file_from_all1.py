'''
The script writes four files:

File	Purpose
breeding dates.csv	Main pulse/site outcome table
breeding_dates_review.csv	Only output rows needing review
breeding_dates_source_issues.csv	One row per validation issue found in the source sheet
results.txt	Summary counts

The important distinction:

breeding_dates_review.csv tells you which output rows need review.
breeding_dates_source_issues.csv catches all source issues, including bad values in skipped rows and non-exported fields like pNmcend.


Practical workflow:
1. Run the script.
2. Open breeding_dates_review.csv first.
3. Fix human-review problems in All.csv.
4. Open breeding_dates_source_issues.csv next.
5. Fix invalid values, odd abandonment coding, or source-sheet debris.
6. Rerun until the review/issue counts are where you expect.
7. Commit the script and regenerated outputs.

'''
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

DEFAULT_INPUT_FILE = Path(
    r"C:\Users\mikes\OneDrive\Documents\GitHub\TRBLSummarizer\TRBLSummarizer\Data\TRBL Analysis tracking - All.csv"
)

HEADER_ROWS_TO_SKIP = 2
PULSES = ("p1", "p2", "p3", "p4")

# These fields are written to the pulse/site outcome output.
# mcend is intentionally excluded because it is not needed downstream.
OUTPUT_DATE_FIELDS = ("mcstart", "incstart", "hatch", "fledgestart", "fledgedisp", "abandon")

# These fields are validated in the source file. mcend is validated even though it
# is not exported, so bad source values are visible in the issues file.
VALIDATE_DATE_FIELDS = ("mcstart", "mcend", "incstart", "hatch", "fledgestart", "fledgedisp", "abandon")

# For No Colony, male chorus can be compatible with birds being nearby but not
# actually forming a colony at that ARU/site. These later-stage/date fields are
# therefore the prohibited ones for No Colony.
NO_COLONY_PROHIBITED_DATE_FIELDS = ("incstart", "hatch", "fledgestart", "fledgedisp", "abandon")

NO_OUTCOME_VALUES = {"", "n/a", "na"}
MISSING_DATE_VALUES = {"", "nd", "n/a", "na"}
KNOWN_OUTCOMES = {"successful", "partially abandoned", "abandoned", "unknown", "no colony", "no trbl"}
NO_DATE_OK_OUTCOMES = {"no colony", "no trbl"}
ABANDON_OUTCOMES = {"abandoned", "partially abandoned"}

REVIEW_OK = "OK"
REVIEW_NEEDED = "REVIEW"
MISSING_OUTCOME_SENTINEL = "REVIEW_MISSING_OUTCOME"

DATE_RE = re.compile(
    r"^(?P<approx>~?)(?P<date>\d{1,2}/\d{1,2}/\d{4})(?P<partial>[Pp]?)$"
)

OUTPUT_FIELDS = [
    "Site ID",
    "Group",
    "Name",
    "Pretty Name",
    "Comment",
    "Deployment Start",
    "Deployment End",
    "Breeding Type",
    "Complex Types",
    "Pulse Name",
    "Outcome",
    "Substrate",
    "Colony Size",
    "mcstart",
    "incstart",
    "hatch",
    "fledgestart",
    "fledgedisp",
    "abandon",
    "partial abandon",
    "Source Row",
    "Review Status",
    "Review Notes",
]

ISSUE_FIELDS = [
    "Source Row",
    "Site ID",
    "Name",
    "Pulse",
    "Column",
    "Issue Type",
    "Severity",
    "Value",
    "Message",
    "Skip Site",
    "Included In Main Output",
]

REQUIRED_COLUMNS = {
    "Site ID",
    "Name",
    "First Recording",
    "Last Recording",
    "Breeding Type",
    "Complex Types",
    "Approx Colony Size",
    "Substrate",
    "Group",
    "Pretty Site Name",
    "Skip Site",
    "Comment for Skip Site",
}
REQUIRED_COLUMNS.update(f"{pulse}Outcome" for pulse in PULSES)
REQUIRED_COLUMNS.update(f"{pulse}{field}" for pulse in PULSES for field in VALIDATE_DATE_FIELDS)


def clean(value: object) -> str:
    """Return a stripped string, treating None as blank."""
    return "" if value is None else str(value).strip()


def normalized(value: object) -> str:
    """Return a stripped, case-normalized string for comparisons."""
    return clean(value).casefold()


def has_real_outcome(value: object) -> bool:
    """Return True when the human review outcome field contains a real outcome."""
    return normalized(value) not in NO_OUTCOME_VALUES


def make_issue(
    *,
    source_row_number: int,
    row: dict[str, str],
    pulse: str,
    column: str,
    issue_type: str,
    severity: str,
    value: object,
    message: str,
    included_in_main_output: bool = False,
) -> dict[str, str]:
    """Create a normalized source-issue row."""
    return {
        "Source Row": str(source_row_number),
        "Site ID": clean(row.get("Site ID")),
        "Name": clean(row.get("Name")),
        "Pulse": pulse,
        "Column": column,
        "Issue Type": issue_type,
        "Severity": severity,
        "Value": clean(value),
        "Message": message,
        "Skip Site": clean(row.get("Skip Site")),
        "Included In Main Output": "Yes" if included_in_main_output else "No",
    }


def parse_date_token(value: object, field: str) -> tuple[str, date | None, bool]:
    """
    Validate and parse a pulse date-like token.

    Returns:
        (status, parsed_date, has_partial_suffix)

    Valid values:
    - blank, ND, n/a, NA
    - M/D/YYYY or MM/DD/YYYY
    - ~M/D/YYYY or ~MM/DD/YYYY
    - date plus trailing P/p only for abandon fields
    - inf
    - missed
    - before* only for hatch fields
    - Continuous only for fledgestart/fledgedisp fields

    The Continuous allowance is deliberate because current TRBL pulse data uses it
    as a formal marker for continuous breeding evidence. Remove that allowance
    below if Continuous should now be treated as invalid source data.
    """
    text = clean(value)
    low = normalized(text)

    if low in MISSING_DATE_VALUES:
        return "missing", None, False

    if low.startswith("before"):
        if field == "hatch":
            return "valid", None, False
        return "invalid_before_non_hatch", None, False

    if low in {"inf", "missed"}:
        return "valid", None, False

    if low == "continuous":
        if field in {"fledgestart", "fledgedisp"}:
            return "valid", None, False
        return "invalid_continuous_field", None, False

    match = DATE_RE.match(text)
    if not match:
        return "invalid_characters", None, False

    has_partial_suffix = bool(match.group("partial"))
    if has_partial_suffix and field != "abandon":
        return "invalid_partial_suffix_non_abandon", None, has_partial_suffix

    try:
        parsed_date = datetime.strptime(match.group("date"), "%m/%d/%Y").date()
    except ValueError:
        return "invalid_calendar_date", None, has_partial_suffix

    return "valid", parsed_date, has_partial_suffix


def field_has_value(row: dict[str, str], pulse: str, field: str) -> bool:
    """
    Return True when a pulse date-like field has a non-missing value.

    Invalid non-blank values count as values so they are not silently dropped.
    """
    status, _, _ = parse_date_token(row.get(f"{pulse}{field}", ""), field)
    return status != "missing"


def has_any_exported_date_value(row: dict[str, str], pulse: str) -> bool:
    return any(field_has_value(row, pulse, field) for field in OUTPUT_DATE_FIELDS)


def has_any_validated_date_value(row: dict[str, str], pulse: str) -> bool:
    return any(field_has_value(row, pulse, field) for field in VALIDATE_DATE_FIELDS)


def normalize_hatch(value: str) -> str:
    """Normalize hatch values beginning with before* to inf in the output."""
    if normalized(value).startswith("before"):
        return "inf"
    return value


def split_partial_abandon(row: dict[str, str], pulse: str, outcome: str) -> tuple[str, str]:
    """
    Return (abandon, partial_abandon).

    For Partially Abandoned pulses:
    - exported abandon becomes ND
    - partial abandon gets pNabandon with trailing P/p removed
    """
    raw_abandon = clean(row.get(f"{pulse}abandon", ""))

    if normalized(outcome) != "partially abandoned":
        return raw_abandon, "ND"

    if not raw_abandon or normalized(raw_abandon) in MISSING_DATE_VALUES:
        return "ND", "ND"

    if raw_abandon.endswith(("P", "p")):
        return "ND", raw_abandon[:-1]

    return "ND", raw_abandon


def validate_deployment_dates(
    *,
    row: dict[str, str],
    source_row_number: int,
) -> list[dict[str, str]]:
    """Validate First Recording and Last Recording at the source-row level."""
    issues: list[dict[str, str]] = []
    parsed: dict[str, date] = {}

    for column in ("First Recording", "Last Recording"):
        status, parsed_date, _ = parse_date_token(row.get(column, ""), field="deployment")
        if status == "valid" and parsed_date is not None:
            parsed[column] = parsed_date
        elif status != "missing":
            issues.append(
                make_issue(
                    source_row_number=source_row_number,
                    row=row,
                    pulse="",
                    column=column,
                    issue_type="invalid_deployment_date",
                    severity="ERROR",
                    value=row.get(column, ""),
                    message=f"{column} has an invalid date value.",
                )
            )

    if (
        "First Recording" in parsed
        and "Last Recording" in parsed
        and parsed["First Recording"] > parsed["Last Recording"]
    ):
        issues.append(
            make_issue(
                source_row_number=source_row_number,
                row=row,
                pulse="",
                column="First Recording / Last Recording",
                issue_type="deployment_date_order_error",
                severity="ERROR",
                value=f"{row.get('First Recording')} > {row.get('Last Recording')}",
                message="First Recording is after Last Recording.",
            )
        )

    return issues


def validate_pulse(
    *,
    row: dict[str, str],
    pulse: str,
    source_row_number: int,
    validate_outcome_consistency: bool,
) -> list[dict[str, str]]:
    """Return all source issues found for one pulse."""
    issues: list[dict[str, str]] = []
    source_outcome = clean(row.get(f"{pulse}Outcome", ""))
    source_outcome_norm = normalized(source_outcome)
    has_outcome = has_real_outcome(source_outcome)
    has_exported_dates = has_any_exported_date_value(row, pulse)
    has_validated_dates = has_any_validated_date_value(row, pulse)

    # Validate pulse date-like fields, including mcend even though mcend is not exported.
    parsed_dates: dict[str, date] = {}
    for field in VALIDATE_DATE_FIELDS:
        column = f"{pulse}{field}"
        value = row.get(column, "")
        status, parsed_date, _ = parse_date_token(value, field)

        if status == "valid" and parsed_date is not None:
            parsed_dates[field] = parsed_date
        elif status.startswith("invalid"):
            issues.append(
                make_issue(
                    source_row_number=source_row_number,
                    row=row,
                    pulse=pulse,
                    column=column,
                    issue_type="invalid_date_value",
                    severity="ERROR",
                    value=value,
                    message=f"{column} has invalid value type: {status}.",
                )
            )

    # For Skip Site = Y rows, stop after source value validation. They are not part of output.
    if not validate_outcome_consistency:
        return issues

    if has_outcome and source_outcome_norm not in KNOWN_OUTCOMES:
        issues.append(
            make_issue(
                source_row_number=source_row_number,
                row=row,
                pulse=pulse,
                column=f"{pulse}Outcome",
                issue_type="invalid_outcome_value",
                severity="ERROR",
                value=source_outcome,
                message=f"{pulse}Outcome has an unexpected outcome value.",
            )
        )

    if not has_outcome and has_exported_dates:
        issues.append(
            make_issue(
                source_row_number=source_row_number,
                row=row,
                pulse=pulse,
                column=f"{pulse}Outcome",
                issue_type="missing_outcome_with_dates",
                severity="REVIEW",
                value=source_outcome,
                message=f"{pulse} has exported pulse date fields but {pulse}Outcome is blank or n/a.",
            )
        )

    if has_outcome and not has_exported_dates and source_outcome_norm not in NO_DATE_OK_OUTCOMES:
        issues.append(
            make_issue(
                source_row_number=source_row_number,
                row=row,
                pulse=pulse,
                column=f"{pulse}Outcome",
                issue_type="outcome_without_dates",
                severity="REVIEW",
                value=source_outcome,
                message=f"{pulse}Outcome is '{source_outcome}', but all exported date fields are blank or ND.",
            )
        )

    if source_outcome_norm == "no trbl" and has_validated_dates:
        issues.append(
            make_issue(
                source_row_number=source_row_number,
                row=row,
                pulse=pulse,
                column=f"{pulse}Outcome",
                issue_type="no_trbl_with_dates",
                severity="ERROR",
                value=source_outcome,
                message="Outcome is No TRBL, but one or more pulse date fields has a value.",
            )
        )

    if source_outcome_norm == "no colony":
        for field in NO_COLONY_PROHIBITED_DATE_FIELDS:
            column = f"{pulse}{field}"
            if field_has_value(row, pulse, field):
                issues.append(
                    make_issue(
                        source_row_number=source_row_number,
                        row=row,
                        pulse=pulse,
                        column=column,
                        issue_type="no_colony_with_breeding_dates",
                        severity="ERROR",
                        value=row.get(column, ""),
                        message=(
                            "Outcome is No Colony, but a female/hatch/fledge/abandon "
                            "date field has a value."
                        ),
                    )
                )

    raw_abandon = clean(row.get(f"{pulse}abandon", ""))
    abandon_has_value = field_has_value(row, pulse, "abandon")
    abandon_has_p_suffix = raw_abandon.endswith(("P", "p"))

    if source_outcome_norm == "partially abandoned" and not abandon_has_p_suffix:
        issues.append(
            make_issue(
                source_row_number=source_row_number,
                row=row,
                pulse=pulse,
                column=f"{pulse}abandon",
                issue_type="partial_outcome_missing_p_suffix",
                severity="REVIEW",
                value=raw_abandon,
                message="Outcome is Partially Abandoned, but abandon date does not end with P/p.",
            )
        )

    if abandon_has_p_suffix and source_outcome_norm != "partially abandoned":
        issues.append(
            make_issue(
                source_row_number=source_row_number,
                row=row,
                pulse=pulse,
                column=f"{pulse}abandon",
                issue_type="p_suffix_without_partial_outcome",
                severity="REVIEW",
                value=raw_abandon,
                message="Abandon date ends with P/p, but outcome is not Partially Abandoned.",
            )
        )

    if source_outcome_norm == "abandoned" and not abandon_has_value:
        issues.append(
            make_issue(
                source_row_number=source_row_number,
                row=row,
                pulse=pulse,
                column=f"{pulse}abandon",
                issue_type="abandoned_outcome_missing_abandon_date",
                severity="REVIEW",
                value=raw_abandon,
                message="Outcome is Abandoned, but abandon date is blank or ND.",
            )
        )

    if source_outcome_norm == "successful" and abandon_has_value:
        issues.append(
            make_issue(
                source_row_number=source_row_number,
                row=row,
                pulse=pulse,
                column=f"{pulse}abandon",
                issue_type="successful_outcome_with_abandon_date",
                severity="REVIEW",
                value=raw_abandon,
                message="Outcome is Successful, but abandon date has a value.",
            )
        )

    # Chronology checks use only parseable calendar dates. Special values such as
    # inf, missed, before*, ND, and Continuous are deliberately ignored here.
    ordered_fields = ["mcstart", "incstart", "hatch", "fledgestart", "fledgedisp"]
    for earlier, later in zip(ordered_fields, ordered_fields[1:]):
        if earlier in parsed_dates and later in parsed_dates and parsed_dates[earlier] > parsed_dates[later]:
            issues.append(
                make_issue(
                    source_row_number=source_row_number,
                    row=row,
                    pulse=pulse,
                    column=f"{pulse}{later}",
                    issue_type="date_order_error",
                    severity="REVIEW",
                    value=f"{row.get(f'{pulse}{earlier}')} > {row.get(f'{pulse}{later}')}",
                    message=f"{pulse}{earlier} is after {pulse}{later}.",
                )
            )

    if "abandon" in parsed_dates:
        if "mcstart" in parsed_dates and parsed_dates["abandon"] < parsed_dates["mcstart"]:
            issues.append(
                make_issue(
                    source_row_number=source_row_number,
                    row=row,
                    pulse=pulse,
                    column=f"{pulse}abandon",
                    issue_type="date_order_error",
                    severity="REVIEW",
                    value=f"{raw_abandon} < {row.get(f'{pulse}mcstart')}",
                    message=f"{pulse}abandon is before {pulse}mcstart.",
                )
            )
        if "incstart" in parsed_dates and parsed_dates["abandon"] < parsed_dates["incstart"]:
            issues.append(
                make_issue(
                    source_row_number=source_row_number,
                    row=row,
                    pulse=pulse,
                    column=f"{pulse}abandon",
                    issue_type="date_order_error",
                    severity="REVIEW",
                    value=f"{raw_abandon} < {row.get(f'{pulse}incstart')}",
                    message=f"{pulse}abandon is before {pulse}incstart.",
                )
            )

    return issues


def build_output_row(
    *,
    row: dict[str, str],
    pulse: str,
    output_outcome: str,
    source_row_number: int,
    pulse_issues: list[dict[str, str]],
) -> dict[str, str]:
    review_notes = sorted({issue["Message"] for issue in pulse_issues})

    output_row = {
        "Site ID": clean(row.get("Site ID")),
        "Group": clean(row.get("Group")),
        "Name": clean(row.get("Name")),
        "Pretty Name": clean(row.get("Pretty Site Name")),
        "Comment": clean(row.get("Comment for Skip Site")),
        "Deployment Start": clean(row.get("First Recording")),
        "Deployment End": clean(row.get("Last Recording")),
        "Breeding Type": clean(row.get("Breeding Type")),
        "Complex Types": clean(row.get("Complex Types")),
        "Pulse Name": f"{clean(row.get('Pretty Site Name'))} {pulse}",
        "Outcome": output_outcome,
        "Substrate": clean(row.get("Substrate")),
        "Colony Size": clean(row.get("Approx Colony Size")),
    }

    for field in OUTPUT_DATE_FIELDS:
        value = clean(row.get(f"{pulse}{field}", ""))
        if field == "hatch":
            output_row[field] = normalize_hatch(value)
        elif field == "abandon":
            abandon, partial_abandon = split_partial_abandon(row, pulse, output_outcome)
            output_row["abandon"] = abandon
            output_row["partial abandon"] = partial_abandon
        else:
            output_row[field] = value

    output_row["Source Row"] = str(source_row_number)
    output_row["Review Status"] = REVIEW_NEEDED if pulse_issues else REVIEW_OK
    output_row["Review Notes"] = "; ".join(review_notes)

    return output_row


def read_all_csv(input_file: Path) -> list[tuple[int, dict[str, str]]]:
    """
    Read the All.csv tracking file.

    The first two rows are metadata. Row 3 contains the actual headers.
    Data begins on row 4.
    """
    rows: list[tuple[int, dict[str, str]]] = []

    with input_file.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.reader(infile)

        for _ in range(HEADER_ROWS_TO_SKIP):
            next(reader)

        headers = next(reader)
        missing_columns = sorted(REQUIRED_COLUMNS - set(headers))
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Input file is missing required columns: {missing_text}")

        for source_row_number, values in enumerate(reader, start=HEADER_ROWS_TO_SKIP + 2):
            row = {header: value for header, value in zip(headers, values)}
            rows.append((source_row_number, row))

    return rows


def mark_issues_as_included(issues: Iterable[dict[str, str]]) -> None:
    for issue in issues:
        issue["Included In Main Output"] = "Yes"


def extract_rows_and_issues(
    source_rows: list[tuple[int, dict[str, str]]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    output_rows: list[dict[str, str]] = []
    source_issues: list[dict[str, str]] = []

    for source_row_number, row in source_rows:
        skip_site_norm = normalized(row.get("Skip Site"))
        skip_site_is_y = skip_site_norm == "y"

        if skip_site_norm not in {"", "y", "dupe"}:
            source_issues.append(
                make_issue(
                    source_row_number=source_row_number,
                    row=row,
                    pulse="",
                    column="Skip Site",
                    issue_type="invalid_skip_site_value",
                    severity="ERROR",
                    value=row.get("Skip Site", ""),
                    message="Skip Site has an unexpected value. Expected blank, Y, or Dupe.",
                )
            )

        source_issues.extend(validate_deployment_dates(row=row, source_row_number=source_row_number))

        for pulse in PULSES:
            pulse_issues = validate_pulse(
                row=row,
                pulse=pulse,
                source_row_number=source_row_number,
                validate_outcome_consistency=not skip_site_is_y,
            )
            source_issues.extend(pulse_issues)

            if skip_site_is_y:
                continue

            source_outcome = clean(row.get(f"{pulse}Outcome", ""))
            has_outcome = has_real_outcome(source_outcome)
            has_exported_dates = has_any_exported_date_value(row, pulse)

            if not has_outcome and not has_exported_dates:
                continue

            output_outcome = source_outcome if has_outcome else MISSING_OUTCOME_SENTINEL
            mark_issues_as_included(pulse_issues)
            output_rows.append(
                build_output_row(
                    row=row,
                    pulse=pulse,
                    output_outcome=output_outcome,
                    source_row_number=source_row_number,
                    pulse_issues=pulse_issues,
                )
            )

    return output_rows, source_issues


def write_csv(output_file: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary_file: Path, output_rows: list[dict[str, str]], source_issues: list[dict[str, str]]) -> None:
    outcome_counts = Counter(row["Outcome"] for row in output_rows)
    breeding_type_counts = Counter(row["Breeding Type"] for row in output_rows)
    review_status_counts = Counter(row["Review Status"] for row in output_rows)
    issue_type_counts = Counter(issue["Issue Type"] for issue in source_issues)
    issue_severity_counts = Counter(issue["Severity"] for issue in source_issues)

    with summary_file.open("w", encoding="utf-8") as summary:
        summary.write("Breeding Dates Extraction Summary\n")
        summary.write("=================================\n\n")

        summary.write(f"Output rows: {len(output_rows)}\n")
        summary.write(f"Output rows needing review: {review_status_counts[REVIEW_NEEDED]}\n")
        summary.write(f"Source issues: {len(source_issues)}\n\n")

        summary.write("Output Review Status Counts\n")
        summary.write("---------------------------\n")
        for status, count in review_status_counts.most_common():
            summary.write(f"{status}: {count}\n")

        summary.write("\nIssue Severity Counts\n")
        summary.write("---------------------\n")
        for severity, count in issue_severity_counts.most_common():
            summary.write(f"{severity}: {count}\n")

        summary.write("\nIssue Type Counts\n")
        summary.write("-----------------\n")
        for issue_type, count in issue_type_counts.most_common():
            summary.write(f"{issue_type}: {count}\n")

        summary.write("\nOutcome Counts\n")
        summary.write("--------------\n")
        for outcome, count in outcome_counts.most_common():
            summary.write(f"{outcome}: {count}\n")

        summary.write("\nBreeding Type Counts\n")
        summary.write("--------------------\n")
        for breeding_type, count in breeding_type_counts.most_common():
            summary.write(f"{breeding_type}: {count}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand TRBL All.csv site rows into one row per reviewed pulse/site outcome and validate source values."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to TRBL Analysis tracking - All.csv. Default: {DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=Path("breeding dates.csv"),
        help="Main output CSV path.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=Path("breeding_dates_review.csv"),
        help="Output pulse rows whose Review Status is REVIEW.",
    )
    parser.add_argument(
        "--issues-csv",
        type=Path,
        default=Path("breeding_dates_source_issues.csv"),
        help="One row per validation issue found in the source file.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results.txt"),
        help="Text summary output path.",
    )
    return parser.parse_args()


def make_breeding_dates_file() -> None:
    args = parse_args()

    source_rows = read_all_csv(args.input_csv)
    output_rows, source_issues = extract_rows_and_issues(source_rows)
    review_rows = [row for row in output_rows if row["Review Status"] == REVIEW_NEEDED]

    write_csv(args.output_csv, OUTPUT_FIELDS, output_rows)
    write_csv(args.review_csv, OUTPUT_FIELDS, review_rows)
    write_csv(args.issues_csv, ISSUE_FIELDS, source_issues)
    write_summary(args.summary, output_rows, source_issues)

    print(f"Input file: {args.input_csv}")
    print(f"Source data rows: {len(source_rows)}")
    print(f"Output rows: {len(output_rows)}")
    print(f"Output rows needing review: {len(review_rows)}")
    print(f"Source issues: {len(source_issues)}")
    print(f"Wrote main output: {args.output_csv}")
    print(f"Wrote review output: {args.review_csv}")
    print(f"Wrote source issues: {args.issues_csv}")
    print(f"Wrote summary: {args.summary}")


if __name__ == "__main__":
    make_breeding_dates_file()
