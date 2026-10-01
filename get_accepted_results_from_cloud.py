
import io
import re

import pandas as pd
import requests

from common import (
    OUTPUT_DIR,
    fetch_sheet_dataframe,
    save_csv_with_retry,
    transform_value,
)


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
    output_path = OUTPUT_DIR / output_filename
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(top_metadata_lines)
        main_df.to_csv(f, index=False)

    print(f"Successfully updated and saved to '{output_path}'")
    return main_df


def get_data_from_sheet()->pd.DataFrame:
    df = fetch_sheet_dataframe(sheet_name="main", header_row=2)
    return df   


def clean_data(input_df: pd.DataFrame) -> pd.DataFrame:
    # Implement your data cleaning logic here
    cleaned_df = input_df.copy()
    for col in cleaned_df.select_dtypes(include="object").columns:
        if col != "site" and col != "outcome":
            parsed_dates = pd.to_datetime(cleaned_df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            cleaned_df[col] = parsed_dates.fillna(cleaned_df[col])

    # # 1. Create a mask that finds the rows where outcome is "No Trbl"
    # mask = cleaned_df["outcome"] == "No Trbl"

    # # 2. Use .loc to apply the regex replacement only to those specific rows
    # cleaned_df.loc[mask, "site"] = cleaned_df.loc[mask, "site"].str.replace(
    #     r" [Pp]\d+$", "", regex=True
    # )

    return cleaned_df


if __name__ == "__main__":

    data_df = get_data_from_sheet()

    col_map = {
        "old_site": "site",
        "outcome": "outcome",
        "ss_accpt" : "settlement_start",
        "se_accpt" : "settlement_end",
        "is_accpt": "incubation_onset",
        "hatch_accpt": "brooding_onset",
        "fo_accpt": "fledging_onset",
        "fd_accpt": "fledgling_dispersal",
    }
    mapped_df = data_df[list(col_map.keys())]
    mapped_df = mapped_df.rename(columns=col_map) 
    cleaned_df = clean_data(mapped_df)

    save_csv_with_retry(cleaned_df, OUTPUT_DIR / "TRBL_dates - accepted_results.csv")
