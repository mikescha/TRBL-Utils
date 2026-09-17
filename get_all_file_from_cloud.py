
import datetime
import io
import re
import time

import pandas as pd
import requests

WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxtzRiML6eQQCyERCQSywEvLZYCFglybWn5CQ_WJuHC6Mw77SbTIvkjulu6F16Ob4EWMg/exec"


def fetch_sheet_dataframe(
    web_app_url: str = WEBHOOK_URL,
    sheet_name: str = "main",
    header_row: int = 1,
    max_retries: int = 3,
    retry_delay: int = 2,
) -> pd.DataFrame:
    """Fetches data from Google Apps Script doGet and returns a Pandas DataFrame.

    :param web_app_url: Google Apps Script Web App URL
    :param sheet_name: Target sheet tab name
    :param header_row: 1-based row number containing headers (default: 2)
    :param max_retries: Number of retry attempts on failure
    :param retry_delay: Delay in seconds between retries
    :return: pd.DataFrame
    """
    params = {"sheet": sheet_name, "start_row":header_row}
    results_df = pd.DataFrame()

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(web_app_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Verify response is a 2D array and has enough rows for the header
            if not isinstance(data, list):
                raise ValueError(
                    f"❌ Expected list payload, got {type(data).__name__}: {data}"
                )

            if len(data) <= header_row:
                raise ValueError(
                    f"❌ Sheet has {len(data)} rows, but header is configured at row {header_row}."
                )

            print(f"✅ Retrieved '{sheet_name}' with {len(data)} total rows (including headers).")
            headers = data[0]
            rows = data[1 :]

            results_df = pd.DataFrame(rows, columns=headers)
            break

        except (requests.RequestException, ValueError) as err:
            print(f"❌ [Attempt {attempt}/{max_retries}] Fetch failed: {err}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"❌ Failed to retrieve sheet data after {max_retries} attempts."
                ) from err
            time.sleep(retry_delay)

    return results_df


def transform_value(val, site: str, col: str) -> str:
    """Transforms a single cell value:

    - Formats dates to M/D/YYYY (with ~ for estimates or suffixes like (C)).
    - Handles ISO 8601 timestamps (e.g., '2023-07-05T07:00:00.000Z').
    - Validates string tokens against the allowed list.
    - Prints an error for unallowed values.
    """
    if pd.isna(val) or val is None:
        return "ND"

    # Handle native pandas/datetime objects or strings
    if isinstance(val, (datetime.date, datetime.datetime, pd.Timestamp)):
        val_str = val.strftime("%Y-%m-%d")
    else:
        val_str = str(val).strip()

    if not val_str:
        return "ND"

    # Regex matching: YYYY-MM-DD (with optional ~ prefix, optional ISO timestamp, and optional suffix)
    date_match = re.match(
        r"^(~?)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?(?:\(?([A-Za-z]{1,3})\)?)?$",
        val_str,
    )

    if date_match:
        tilde, year, month, day, suffix = date_match.groups()
        formatted_date = f"{int(month)}/{int(day)}/{year}"

        # Prefix with ~ if tilde existed or a suffix was present
        if tilde or suffix:
            return f"~{formatted_date}"
        return formatted_date

    # Validate non-date allowed string values
    allowed_values = {"ND", "Continuous", "missed", "inf"}
    if val_str in allowed_values:
        return val_str

    # Log error if value is outside the allowed list
    print(
        f"Error: Site '{site}', Column '{col}' contains invalid value '{val_str}'"
    )
    return val_str


def convert_data_to_all_format(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms input Google Sheet DataFrame into the unified target format."""
    pulse_prefixes = ["p1", "p2", "p3", "p4"]
    pulse_subheadings = [
        "mcstart",
        "mcend",
        "incstart",
        "hatch",
        "fledgestart",
        "fledgedisp",
        "abandon",
        "outcome",
    ]

    # 1. Build ordered output column list
    output_cols = ["Name"]
    for p in pulse_prefixes:
        for sub in pulse_subheadings:
            output_cols.append(f"{p}{sub}")

    # --> This is unneeded so long as we're writing back to the old All file. It's only needed if
    #     if we were creating a new All file from scratch, then we would need these additional columns
    # output_cols.extend(
    #     [
    #         "Pretty Site Name",
    #         "Latitude",
    #         "Longitude",
    #         "Skip Site",
    #         "Comment for Skip Site",
    #     ]
    # )

    output_rows = {}  # Preserves site order and accumulates pulse data

    # 2. Iterate through input rows and map data
    for _, row in df.iterrows():
        raw_site = str(row.get("site", "")).strip()
        if not raw_site:
            continue

        # Extract site name and pulse (e.g. "2017 Rush Ranch P1" -> "2017 Rush Ranch", "p1")
        match = re.match(r"^(.*?)\s+(P[1-4])$", raw_site, re.IGNORECASE)
        if match:
            site_name = match.group(1).strip()
            pulse = match.group(2).lower()
        else:
            site_name = raw_site
            pulse = ""

        # Initialize site record if first time seen
        if site_name not in output_rows:
            output_rows[site_name] = {col: "ND" for col in output_cols}
            output_rows[site_name]["Name"] = site_name

        row_dict = output_rows[site_name]

        # 3. Map pulse specific fields
        if pulse:
            mapping = {
                "ss_accpt": f"{pulse}mcstart",
                "se_accpt": f"{pulse}mcend",
                "is_accpt": f"{pulse}incstart",
                "hatch_accpt": f"{pulse}hatch",
                "fo_accpt": f"{pulse}fledgestart",
                "fd_accpt": f"{pulse}fledgedisp",
                "outcome": f"{pulse}outcome",
            }

            for src_col, target_col in mapping.items():
                if src_col in row and pd.notna(row[src_col]):
                    if src_col == "outcome":
                        valid_outcomes = ["Successful", "Abandoned", "Partially Abandoned", 
                                          "No TRBL", "No Colony", "Unknown"]
                        if row[src_col] in valid_outcomes:
                            row_dict[target_col] = row[src_col]

                    else:
                        val = transform_value(row[src_col], raw_site, src_col)
                        if val != "ND":
                            row_dict[target_col] = val

            # Handle abandon vs. partial_abandon
            abandon_val = row.get("abandon", None)
            partial_val = row.get("partial_abandon", None)

            abandon_trans = transform_value(abandon_val, raw_site, "abandon")
            partial_trans = transform_value(
                partial_val, raw_site, "partial_abandon"
            )

            if abandon_trans != "ND":
                row_dict[f"{pulse}abandon"] = abandon_trans
            elif partial_trans != "ND":
                row_dict[f"{pulse}abandon"] = f"{partial_trans}P"

    # 4. Construct final output DataFrame
    out_df = pd.DataFrame(list(output_rows.values()), columns=output_cols)
    return out_df



def update_and_save_main_sheet(
    input_df: pd.DataFrame,
    spreadsheet_id: str = "1NQVKtxVv7zmODNuvn45TOYf-j-nU_u3Q1W-6Y_nCh-Y",
    gid: str = "0",
    output_filename: str = "TRBL Analysis Tracking - All.csv",
) -> pd.DataFrame:
    """Downloads the main tracking sheet, updates rows matching 'Name' with input_df data,

    and saves the combined result locally while preserving the top 2 header lines.
    """
    # 1. Direct CSV Download
    csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
    response = requests.get(csv_url)
    response.raise_for_status()

    # Split lines to isolate top 2 rows from header/data rows
    raw_lines = response.text.splitlines(keepends=True)
    top_metadata_lines = raw_lines[:22]
    table_data = "".join(raw_lines[22:])

    # 2. Parse main sheet into DataFrame starting at Row 3 headers
    main_df = pd.read_csv(io.StringIO(table_data), dtype=str, keep_default_na=False)

    if "Name" not in main_df.columns:
        raise KeyError(
            "Column 'Name' was not found in Row 3 headers of the downloaded sheet."
        )

    # 3. Update matching rows in main_df
    for _, input_row in input_df.iterrows():
        site_name = input_row["Name"]
        match_mask = main_df["Name"] == site_name

        if match_mask.any():
            # Overwrite values for matching columns
            for col in input_df.columns:
                if col in main_df.columns:
                    main_df.loc[match_mask, col] = input_row[col]
        else:
            print(
                f"Warning: Site '{site_name}' from input_df was not found in the main sheet."
            )

    # 4. Save modified main sheet locally (preserving original top 2 rows)
    with open(output_filename, "w", encoding="utf-8", newline="") as f:
        f.writelines(top_metadata_lines)
        main_df.to_csv(f, index=False)

    print(f"Successfully updated and saved to '{output_filename}'")
    return main_df




def get_data_from_sheet()->pd.DataFrame:
    df = fetch_sheet_dataframe(sheet_name="main", header_row=2)
    return df   



def get_metadata_from_sheet()->pd.DataFrame:
    df = fetch_sheet_dataframe(sheet_name="metadata", header_row=1)
    return df




if __name__ == "__main__":
    all_sheet = pd.DataFrame()

    data_df = get_data_from_sheet()
    mapped_df = convert_data_to_all_format(data_df)

    final_df = update_and_save_main_sheet(mapped_df)