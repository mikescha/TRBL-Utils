from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from constants import (
    COL_ABANDON_DATE,
    COL_APPROX_COLONY_SIZE,
    COL_BREEDING_TYPE,
    COL_COLONY_SIZE,
    COL_COMMENT,
    COL_COMPLEX_TYPES,
    COL_DEPLOYMENT_END,
    COL_DEPLOYMENT_START,
    COL_GROUP,
    COL_HATCH_DATE,
    COL_OUTCOME,
    COL_PARTIAL_ABANDON_DATE,
    COL_PULSE_NAME,
    COL_SITE_ID,
    COL_SITE_NAME,
    COL_SUBSTRATE,
    INPUT_CSV,
    OUTCOME_ABANDONED,
    OUTCOME_NO_COLONY,
    OUTCOME_NO_TRBL,
    OUTCOME_PARTIALLY_ABANDONED,
    OUTCOME_SUCCESSFUL,
    OUTCOME_UNKNOWN,
    STATUS_ND,
)

# ==============================================================================
# CONFIGURATION CONSTANTS (Edit these directly for your VS Code workflow)
# ==============================================================================
OUTPUT_CSV = Path("breeding_dates.csv")
REVIEW_CSV = Path("breeding_dates_review.csv")
ISSUES_CSV = Path("breeding_dates_source_issues.csv")
SUMMARY_TXT = Path("results.txt")

# ==============================================================================
# DOMAIN VALIDATION RULES & SCHEMA
# ==============================================================================
HEADER_ROWS_TO_SKIP = 2
PULSES = ("p1", "p2", "p3", "p4")

OUTPUT_DATE_FIELDS = ("mcstart", "incstart", "hatch", "fledgestart", "fledgedisp", "abandon")
VALIDATE_DATE_FIELDS = ("mcstart", "mcend", "incstart", "hatch", "fledgestart", "fledgedisp", "abandon")
NO_COLONY_PROHIBITED_DATE_FIELDS = ("incstart", "hatch", "fledgestart", "fledgedisp", "abandon")

NO_OUTCOME_VALUES = {"n/a"}
MISSING_DATE_VALUES = {STATUS_ND}
KNOWN_OUTCOMES = {OUTCOME_SUCCESSFUL, OUTCOME_PARTIALLY_ABANDONED, OUTCOME_ABANDONED, 
                  OUTCOME_UNKNOWN, OUTCOME_NO_COLONY, OUTCOME_NO_TRBL}
NO_DATE_OK_OUTCOMES = {OUTCOME_NO_COLONY, OUTCOME_NO_TRBL}

REVIEW_OK = "OK"
REVIEW_NEEDED = "REVIEW"
MISSING_OUTCOME_SENTINEL = "REVIEW_MISSING_OUTCOME"

DATE_RE = re.compile(r"^(?P<approx>~?)(?P<date>\d{1,2}/\d{1,2}/\d{4})(?P<partial>[Pp]?)$")

# Regex to detect Excel auto-converted date corruption (e.g. 2026-05-10 or 05-10-2026)
EXCEL_DATE_CORRUPTION_RE = re.compile(r"\d{2,4}[-/]\d{1,2}[-/]\d{2,4}")

# Regex patterns for parsing Colony Size numbers and ranges (applied after commas are stripped)
INT_RE = re.compile(r"^\d+$")
RANGE_RE = re.compile(r"^(\d+)\s*[-–—]\s*(\d+)$")  # Handles hyphen, en-dash, em-dash
COLONY_SIZE_PASSTHROUGH = {"NEED", STATUS_ND, OUTCOME_NO_COLONY, OUTCOME_UNKNOWN}
#Fields for final breeding_dates file
OUTPUT_FIELDS = [
    COL_SITE_ID, COL_GROUP, COL_SITE_NAME, COL_PULSE_NAME, COL_DEPLOYMENT_START, 
    COL_DEPLOYMENT_END, COL_BREEDING_TYPE, COL_COMPLEX_TYPES, COL_OUTCOME, COL_SUBSTRATE, 
    COL_APPROX_COLONY_SIZE, COL_COLONY_SIZE,  
    "mcstart", "incstart", COL_HATCH_DATE, "fledgestart", "fledgedisp", COL_ABANDON_DATE, COL_PARTIAL_ABANDON_DATE,
    COL_COMMENT,
    "Source Row", "Review Status", "Review Notes", 
]
#Only for diagnostics
ISSUE_FIELDS = [
    "Source Row", COL_SITE_ID, COL_SITE_NAME, "Pulse", "Column", "Issue Type", "Severity", 
    "Value", "Suggested Value", "Message", "Skip Site", "Included In Main Output"
]

#Required in All file, after this they will be mapped to constants so don't need to make constants here
REQUIRED_COLUMNS = {
    "Site ID", "Name", "First Recording", "Last Recording", "Breeding Type", 
    "Complex Types", "Approx Colony Size", "Substrate", "Group", "Pretty Site Name", 
    "Skip Site", "Comment for Skip Site"
}
REQUIRED_COLUMNS.update(f"{pulse}Outcome" for pulse in PULSES)
REQUIRED_COLUMNS.update(f"{pulse}{field}" for pulse in PULSES for field in VALIDATE_DATE_FIELDS)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def clean(value: object) -> str:
    """Return a stripped string, treating None as blank."""
    return "" if value is None else str(value).strip()


def is_ai_placeholder(value: object) -> bool:
    """Return True for AI/formula placeholders that should be cleaned to blank."""
    text = clean(value)
    if not text.startswith("="):
        return False
    lowered = text.casefold()
    return (
        lowered.startswith("=ai(")
        or "fill an appropriate value" in lowered
        or "based on the table context" in lowered
    )


def cleaned_output_value(value: object) -> str:
    return "" if is_ai_placeholder(value) else clean(value)


def has_real_outcome(value: object) -> bool:
    return clean(value) not in NO_OUTCOME_VALUES


def field_has_value(row: dict[str, str], pulse: str, field: str) -> bool:
    status, _, _ = parse_date_token(row.get(f"{pulse}{field}", ""), field)
    return status != "missing"


def has_any_exported_date_value(row: dict[str, str], pulse: str) -> bool:
    return any(field_has_value(row, pulse, field) for field in OUTPUT_DATE_FIELDS)


def has_any_validated_date_value(row: dict[str, str], pulse: str) -> bool:
    return any(field_has_value(row, pulse, field) for field in VALIDATE_DATE_FIELDS)


def make_issue(
    *, row_num: int, row: dict[str, str], pulse: str, column: str, 
    issue_type: str, severity: str, value: object, message: str
) -> dict[str, str]:
    """Create a normalized source-issue row."""
    return {
        "Source Row": str(row_num),
        COL_SITE_ID: clean(row.get("Site ID")),
        COL_SITE_NAME: clean(row.get("Name")),
        "Pulse": pulse,
        "Column": column,
        "Issue Type": issue_type,
        "Severity": severity,
        "Value": clean(value),
        "Suggested Value": "" if is_ai_placeholder(value) else clean(value),
        "Message": message,
        "Skip Site": clean(row.get("Skip Site")),
        "Included In Main Output": "No",
    }

# ==============================================================================
# PARSING & TRANSLATION LOGIC
# ==============================================================================
def parse_date_token(value: object, field: str) -> tuple[str, date | None, bool]:
    text = clean(value)

    if is_ai_placeholder(text):
        return "invalid_formula_placeholder", None, False
    if text in MISSING_DATE_VALUES:
        return "missing", None, False
    if text.startswith("before"):
        return ("valid", None, False) if field == "hatch" else ("invalid_before_non_hatch", None, False)
    if text in {"inf", "missed"}:
        return "valid", None, False
    if text == "Continuous":
        if field in {"fledgestart", "fledgedisp"}:
            return ("valid", None, False) 
        else:
            return ("invalid_continuous_field", None, False)

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


def split_partial_abandon(row: dict[str, str], pulse: str, outcome: str) -> tuple[str, str]:
    #First return is Abandon date, second is Partial Abandon date
    raw_abandon = cleaned_output_value(row.get(f"{pulse}abandon", ""))
    if (clean(outcome) != OUTCOME_PARTIALLY_ABANDONED or 
        not raw_abandon or 
        clean(raw_abandon) in MISSING_DATE_VALUES):
        return (raw_abandon, STATUS_ND) if clean(outcome) != OUTCOME_PARTIALLY_ABANDONED else (STATUS_ND, STATUS_ND)
    
    if raw_abandon.endswith(("P", "p")):
        return STATUS_ND, raw_abandon[:-1]
    
    return STATUS_ND, raw_abandon


def parse_colony_size_metric(val: str) -> tuple[str, str | None]:
    """
    Parses Approx Colony Size field into standard Colony Size metrics.
    Strips out visual comma separators before validation logic executes.
    Returns a tuple: (calculated_value_string, error_message_or_None)
    """
    cleaned = clean(val)
    if cleaned in COLONY_SIZE_PASSTHROUGH:
        return val, None
    
    # Strip thousands separators so numbers like 10,000 become 10000
    stripped_val = val.replace(",", "")
    
    if INT_RE.match(stripped_val):
        return stripped_val, None
        
    if range_match := RANGE_RE.match(stripped_val):
        low, high = int(range_match.group(1)), int(range_match.group(2))
        midpoint = (low + high) // 2
        return str(midpoint), None
        
    return "error", f"Approx Colony Size '{val}' could not be parsed into a number, range, or accepted keyword."


# ==============================================================================
# VALIDATION ENGINES
# ==============================================================================
def validate_deployment_dates(row: dict[str, str], row_num: int) -> list[dict[str, str]]:
    issues, parsed = [], {}
    for column in ("First Recording", "Last Recording"):
        status, parsed_date, _ = parse_date_token(row.get(column, ""), field="deployment")
        if status == "valid" and parsed_date is not None:
            parsed[column] = parsed_date
        elif status != "missing":
            issues.append(make_issue(
                row_num=row_num, row=row, pulse="", column=column,
                issue_type="invalid_deployment_date", severity="ERROR",
                value=row.get(column, ""), message=f"{column} has an invalid date value."
            ))

    if (
        "First Recording" in parsed and 
        "Last Recording" in parsed and 
        parsed["First Recording"] > parsed["Last Recording"]
        ):
        issues.append(make_issue(
            row_num=row_num, row=row, pulse="", column="First Recording / Last Recording",
            issue_type="deployment_date_order_error", severity="ERROR",
            value=f"{row.get('First Recording')} > {row.get('Last Recording')}",
            message="First Recording is after Last Recording."
        ))
    return issues


def validate_pulse(row: dict[str, str], pulse: str, row_num: int) -> list[dict[str, str]]:
    issues = []
    source_outcome = clean(row.get(f"{pulse}Outcome", ""))
    has_outcome = has_real_outcome(source_outcome)
    has_exported_dates = has_any_exported_date_value(row, pulse)
    has_validated_dates = has_any_validated_date_value(row, pulse)

    parsed_dates: dict[str, date] = {}
    for field in VALIDATE_DATE_FIELDS:
        column = f"{pulse}{field}"
        value = row.get(column, "")
        status, parsed_date, _ = parse_date_token(value, field)

        if status == "valid" and parsed_date is not None:
            parsed_dates[field] = parsed_date
        elif status.startswith("invalid"):
            if status == "invalid_formula_placeholder":
                msg = f"{column} contains an AI/formula placeholder. Delete it."
            else:
                msg = f"{column} has invalid value type: {status}."
            issues.append(make_issue(
                row_num=row_num, row=row, pulse=pulse, column=column,
                issue_type="invalid_date_value", severity="ERROR", value=value, message=msg
            ))

    if source_outcome and source_outcome not in KNOWN_OUTCOMES:
        issues.append(make_issue(
            row_num=row_num, row=row, pulse=pulse, column=f"{pulse}Outcome",
            issue_type="invalid_outcome_value", severity="ERROR", value=source_outcome,
            message=f"{pulse}Outcome has an unexpected outcome value."
        ))

    if not has_outcome and has_exported_dates:
        issues.append(make_issue(
            row_num=row_num, row=row, pulse=pulse, column=f"{pulse}Outcome",
            issue_type="missing_outcome_with_dates", severity="REVIEW", value=source_outcome,
            message=f"{pulse} has exported pulse date fields but {pulse}Outcome is blank or n/a."
        ))

    if has_outcome and not has_exported_dates and source_outcome not in NO_DATE_OK_OUTCOMES:
        issues.append(make_issue(
            row_num=row_num, row=row, pulse=pulse, column=f"{pulse}Outcome",
            issue_type="outcome_without_dates", severity="REVIEW", value=source_outcome,
            message=f"{pulse}Outcome is '{source_outcome}', but all exported date fields are blank or ND."
        ))

    if source_outcome == OUTCOME_NO_TRBL and has_validated_dates:
        issues.append(make_issue(
            row_num=row_num, row=row, pulse=pulse, column=f"{pulse}Outcome",
            issue_type="no_trbl_with_dates", severity="ERROR", value=source_outcome,
            message="Outcome is No TRBL, but one or more pulse date fields has a value."
        ))

    if source_outcome == OUTCOME_NO_COLONY:
        for field in NO_COLONY_PROHIBITED_DATE_FIELDS:
            if field_has_value(row, pulse, field):
                issues.append(make_issue(
                    row_num=row_num, row=row, pulse=pulse, column=f"{pulse}{field}",
                    issue_type="no_colony_with_breeding_dates", severity="ERROR", value=row.get(f"{pulse}{field}", ""),
                    message="Outcome is No Colony, but a female/hatch/fledge/abandon date field has a value."
                ))

    raw_abandon = cleaned_output_value(row.get(f"{pulse}abandon", ""))
    abandon_has_value = field_has_value(row, pulse, "abandon")
    abandon_has_p_suffix = raw_abandon.endswith(("P", "p"))

    if source_outcome == OUTCOME_PARTIALLY_ABANDONED and not abandon_has_p_suffix:
        issues.append(make_issue(
            row_num=row_num, row=row, pulse=pulse, column=f"{pulse}abandon",
            issue_type="partial_outcome_missing_p_suffix", severity="REVIEW", value=raw_abandon,
            message="Outcome is Partially Abandoned, but abandon date does not end with P/p."
        ))

    if abandon_has_p_suffix and source_outcome != OUTCOME_PARTIALLY_ABANDONED:
        issues.append(make_issue(
            row_num=row_num, row=row, pulse=pulse, column=f"{pulse}abandon",
            issue_type="p_suffix_without_partial_outcome", severity="REVIEW", value=raw_abandon,
            message="Abandon date ends with P/p, but outcome is not Partially Abandoned."
        ))

    if source_outcome == OUTCOME_ABANDONED and not abandon_has_value:
        issues.append(make_issue(
            row_num=row_num, row=row, pulse=pulse, column=f"{pulse}abandon",
            issue_type="abandoned_outcome_missing_abandon_date", severity="REVIEW", value=raw_abandon,
            message="Outcome is Abandoned, but abandon date is blank or ND."
        ))

    if source_outcome == OUTCOME_SUCCESSFUL and abandon_has_value:
        issues.append(make_issue(
            row_num=row_num, row=row, pulse=pulse, column=f"{pulse}abandon",
            issue_type="successful_outcome_with_abandon_date", severity="REVIEW", value=raw_abandon,
            message="Outcome is Successful, but abandon date has a value."
        ))

    ordered_fields = ["mcstart", "incstart", "hatch", "fledgestart", "fledgedisp"]
    for earlier, later in zip(ordered_fields, ordered_fields[1:], strict=False):
        if earlier in parsed_dates and later in parsed_dates and parsed_dates[earlier] > parsed_dates[later]:
            issues.append(make_issue(
                row_num=row_num, row=row, pulse=pulse, column=f"{pulse}{later}",
                issue_type="date_order_error", severity="REVIEW",
                value=f"{row.get(f'{pulse}{earlier}')} > {row.get(f'{pulse}{later}')}",
                message=f"{pulse}{earlier} is after {pulse}{later}."
            ))

    if "abandon" in parsed_dates:
        for boundary in ("mcstart", "incstart"):
            if boundary in parsed_dates and parsed_dates["abandon"] < parsed_dates[boundary]:
                issues.append(make_issue(
                    row_num=row_num, row=row, pulse=pulse, column=f"{pulse}abandon",
                    issue_type="date_order_error", severity="REVIEW",
                    value=f"{raw_abandon} < {row.get(f'{pulse}{boundary}')}",
                    message=f"{pulse}abandon is before {pulse}{boundary}."
                ))
    return issues


# ==============================================================================
# CSV WRITERS & MAIN EXECUTION
# ==============================================================================
def write_summary(output_rows: list[dict[str, str]], source_issues: list[dict[str, str]]) -> None:
    outcome_counts = Counter(row[COL_OUTCOME] for row in output_rows)
    breeding_type_counts = Counter(row[COL_BREEDING_TYPE] for row in output_rows)
    review_status_counts = Counter(row["Review Status"] for row in output_rows)
    issue_type_counts = Counter(issue["Issue Type"] for issue in source_issues)
    issue_severity_counts = Counter(issue["Severity"] for issue in source_issues)

    with SUMMARY_TXT.open("w", encoding="utf-8") as s:
        s.write("Breeding Dates Extraction Summary\n=================================\n\n")
        s.write(f"Output rows: {len(output_rows)}\nOutput rows needing review: "\
                f"{review_status_counts[REVIEW_NEEDED]}\nSource issues: {len(source_issues)}\n\n")
        
        s.write("Output Review Status Counts\n---------------------------\n")
        for k, v in review_status_counts.most_common(): 
            s.write(f"{k}: {v}\n")
        s.write("\nIssue Severity Counts\n---------------------\n")
        for k, v in issue_severity_counts.most_common(): 
            s.write(f"{k}: {v}\n")
        s.write("\nIssue Type Counts\n-----------------\n")
        for k, v in issue_type_counts.most_common(): 
            s.write(f"{k}: {v}\n")
        s.write("\nOutcome Counts\n--------------\n")
        for k, v in outcome_counts.most_common(): 
            s.write(f"{k}: {v}\n")
        s.write("\nBreeding Type Counts\n--------------------\n")
        for k, v in breeding_type_counts.most_common(): 
            s.write(f"{k}: {v}\n")


def make_breeding_dates_file() -> None:
    # 1. Read input CSV tracking data
    source_rows = []
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.reader(infile)
        for _ in range(HEADER_ROWS_TO_SKIP):
            next(reader)
        headers = next(reader)
        
        if missing_columns := sorted(REQUIRED_COLUMNS - set(headers)):
            raise ValueError(f"Input file is missing required columns: {', '.join(missing_columns)}")

        for r_num, values in enumerate(reader, start=HEADER_ROWS_TO_SKIP + 2):
            source_rows.append((r_num, {h: v for h, v in zip(headers, values, strict=True)}))

    # 2. Extract, Validate, and Transform
    output_rows, source_issues = [], []

    for row_num, row in source_rows:
        skip_site = clean(row.get("Skip Site"))
        if skip_site == "Y":
            continue

        if skip_site not in {"", "Dupe"}:
            source_issues.append(make_issue(
                row_num=row_num, row=row, pulse="", column="Skip Site",
                issue_type="invalid_skip_site_value", severity="ERROR",
                value=row.get("Skip Site", ""), message="Skip Site has an unexpected value. Expected blank, Y, or Dupe."
            ))

        source_issues.extend(validate_deployment_dates(row, row_num))

        # Check for Excel Date Corruption error in "Approx Colony Size"
        colony_size_raw = clean(row.get("Approx Colony Size", ""))
        if EXCEL_DATE_CORRUPTION_RE.search(colony_size_raw):
            source_issues.append(make_issue(
                row_num=row_num, row=row, pulse="", column="Approx Colony Size",
                issue_type="corrupted_colony_size_date", severity="ERROR",
                value=colony_size_raw, message="Approx Colony Size was corrupted. Fix the source row data."
            ))

        for pulse in PULSES:
            pulse_issues = validate_pulse(row, pulse, row_num)
            
            source_outcome = clean(row.get(f"{pulse}Outcome", ""))
            has_outcome = has_real_outcome(source_outcome)
            has_exported_dates = has_any_exported_date_value(row, pulse)

            if not has_outcome and not has_exported_dates:
                source_issues.extend(pulse_issues)
                continue

            # Parse calculated Colony Size metric and see if it yields an error
            colony_size_metric, parsing_err = parse_colony_size_metric(colony_size_raw)
            if parsing_err:
                pulse_issues.append(make_issue(
                    row_num=row_num, row=row, pulse=pulse, column="Approx Colony Size",
                    issue_type="unparseable_colony_size", severity="ERROR",
                    value=colony_size_raw, message=parsing_err
                ))

            # Flag errors as included in main output tracking mapping
            for issue in pulse_issues:
                issue["Included In Main Output"] = "Yes"
            
            source_issues.extend(pulse_issues)

            output_outcome = source_outcome if has_outcome else MISSING_OUTCOME_SENTINEL
            review_notes = sorted({issue["Message"] for issue in pulse_issues})

            # Map the flat row out to long form pulse output
            out_row = {
                COL_SITE_ID: clean(row.get("Site ID")),
                COL_GROUP: clean(row.get("Group")),
                COL_SITE_NAME: clean(row.get("Name")),
                COL_PULSE_NAME: f"{clean(row.get('Pretty Site Name'))} {pulse}",
                COL_DEPLOYMENT_START: clean(row.get("First Recording")),
                COL_DEPLOYMENT_END: clean(row.get("Last Recording")),
                COL_BREEDING_TYPE: clean(row.get("Breeding Type")),
                COL_COMPLEX_TYPES: clean(row.get("Complex Types")),
                COL_OUTCOME: output_outcome,
                COL_SUBSTRATE: clean(row.get("Substrate")),
                COL_APPROX_COLONY_SIZE: colony_size_raw,       
                COL_COLONY_SIZE: colony_size_metric,          
                COL_COMMENT: clean(row.get("Comment for Skip Site")),
            }

            for field in OUTPUT_DATE_FIELDS:
                val = cleaned_output_value(row.get(f"{pulse}{field}", ""))
                if field == "hatch":
                    out_row[COL_HATCH_DATE] = "inf" if val.startswith("before") else val
                elif field == "abandon":
                    abandon, partial_abandon = split_partial_abandon(row, pulse, output_outcome)
                    out_row[COL_ABANDON_DATE], out_row[COL_PARTIAL_ABANDON_DATE] = abandon, partial_abandon
                else:
                    out_row[field] = val

            out_row["Source Row"] = str(row_num)
            out_row["Review Status"] = REVIEW_NEEDED if pulse_issues else REVIEW_OK
            out_row["Review Notes"] = "; ".join(review_notes)
            output_rows.append(out_row)

    # 3. Write Datasets
    review_rows = [r for r in output_rows if r["Review Status"] == REVIEW_NEEDED]
    
    for path, fields, data in [
        (OUTPUT_CSV, OUTPUT_FIELDS, output_rows),
        (REVIEW_CSV, OUTPUT_FIELDS, review_rows),
        (ISSUES_CSV, ISSUE_FIELDS, source_issues)
    ]:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    write_summary(output_rows, source_issues)

    # 4. Console Output Summary
    print(f"Done Processing. Logs Written:\n"
          f"  -> Input Rows Found: {len(source_rows)}\n"
          f"  -> Extracted Long Pulse Rows: {len(output_rows)} (Needs Review: {len(review_rows)})\n"
          f"  -> Total Validation Issues Flagged: {len(source_issues)}\n"
          f"Saved target datasets to working directory.")


if __name__ == "__main__":
    make_breeding_dates_file()