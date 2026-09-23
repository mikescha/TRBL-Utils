'''
#TODO
When we migrate to Pretty Names, then I don't think anything needs to change here but probably 
something will...
'''

import datetime
import io
import os
import re
import shutil
import time

import pandas as pd
import requests

from common import (
    INPUT_CSV,
)

WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxtzRiML6eQQCyERCQSywEvLZYCFglybWn5CQ_WJuHC6Mw77SbTIvkjulu6F16Ob4EWMg/exec"

#Map for the data extractors to use
SITE_NAME_MAP_FILENAME = "site_name_map.csv"

PMJ_EXTRACTOR_PATH = r"C:\Users\mikes\GitHub\TRBL-Extractor-PMJ"
HBC_EXTRACTOR_PATH = r"C:\Users\mikes\GitHub\TRBL-Extractor-Data"

TRBL_REVIEWER_DIR = r"C:\Users\mikes\GitHub\TRBL-Breeding-Stages-Final\reviewer_inputs"
TRBL_REVIEWER_METADATA_FILENAME = "trbl_reviewer_metadata.csv"
TRBL_REVIEWER_METADATA_PATH = os.path.join(TRBL_REVIEWER_DIR, TRBL_REVIEWER_METADATA_FILENAME)




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

            # Strip trailing spaces from strings
            results_df = results_df.map(lambda x: x.strip() if isinstance(x, str) else x)

            break

        except (requests.RequestException, ValueError) as err:
            print(f"❌ [Attempt {attempt}/{max_retries}] Fetch failed: {err}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"❌ Failed to retrieve sheet data after {max_retries} attempts."
                ) from err
            time.sleep(retry_delay)

    return results_df


def get_data_from_main_sheet(site_info_df: pd.DataFrame)->pd.DataFrame:
    df = fetch_sheet_dataframe(sheet_name="main", header_row=2)

    #Drop any rows with Skip Site = Y
    valid_site_names = site_info_df["Pretty Site Name"][site_info_df["Skip"].str.upper() != "Y"].copy()

    # Site names in the main sheet can end with " P1"-" P4" (case-insensitive), so we need to account for that
    filtered_df = df[
        df["site"].str.replace(r"\s+[Pp][1-4]$", "", regex=True).isin(valid_site_names)
    ].copy()

    return filtered_df


def get_data_from_site_info_sheet()->pd.DataFrame:
    '''
    Fetches the site_info sheet from the Google Sheet and returns it as a DataFrame.
    Filters out rows where "Skip Site" is "Y".
    '''
    df = fetch_sheet_dataframe(sheet_name="site_info", header_row=1)

    #Drop any rows with Skip Site = Y
    filtered_df = df[df["Skip"].str.upper() != "Y"].copy()
    
    return filtered_df


def transform_value(val, site: str, col: str, YYYYMMDD_format: bool = True) -> str:
    """Transforms a single cell value:

    - Formats dates to YYYY-MM-DD or M/D/YYYY depending on the YYYYMMDD_format flag.
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
        # If it exactly matches M/D/YYYY or MM/DD/YYYY
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", val_str):
            # Convert to intermediate YYYY-MM-DD so the downstream regex can read it easily
            val_str = pd.to_datetime(val_str).strftime("%Y-%m-%d")

    if not val_str:
        return "ND"

    # Regex matching: YYYY-MM-DD (with optional ~ prefix, optional ISO timestamp, and optional suffix)
    date_match = re.match(
        r"^(~?)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?(\(?[A-Za-z]{1,3}\)?)?$",
        val_str,
    )

    if date_match:
        tilde, year, month, day, suffix = date_match.groups()
        
        # Apply the chosen format
        if YYYYMMDD_format:
            # Ensure month and day are 2 digits (e.g., "05" instead of "5")
            formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        else:
            # Strip leading zeros using int() (e.g., "05" becomes "5")
            formatted_date = f"{int(month)}/{int(day)}/{year}"

        # keep the tilde and suffix if they exist
        tilde_str = tilde if tilde else ""
        suffix_str = suffix if suffix else ""
        result = f"{tilde_str}{formatted_date}{suffix_str}"
        return result

    # Validate non-date allowed string values
    allowed_values = {"ND", "Continuous", "missed", "inf"}
    if val_str in allowed_values:
        return val_str

    # Log error if value is outside the allowed list
    print(
        f"Error: Site '{site}', Column '{col}' contains invalid value '{val_str}'"
    )
    return val_str


def transform_date(val: str, YYYYMMDD_format: bool = True) -> str:
    """Transforms a single date value:

    - Formats dates to YYYY-MM-DD or M/D/YYYY depending on the YYYYMMDD_format flag.
    - Handles ISO 8601 timestamps (e.g., '2023-07-05T07:00:00.000Z').
    - Validates string tokens against the allowed list.
    """
    if pd.isna(val) or val is None:
        return "ND"

    # Handle native pandas/datetime objects or strings
    if isinstance(val, (datetime.date, datetime.datetime, pd.Timestamp)):
        val_str = val.strftime("%Y-%m-%d")
    else:
        val_str = str(val).strip()        
        # If it exactly matches M/D/YYYY or MM/DD/YYYY
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", val_str):
            # Convert to intermediate YYYY-MM-DD so the downstream regex can read it easily
            val_str = pd.to_datetime(val_str).strftime("%Y-%m-%d")

    if not val_str:
        return "ND"

    # Regex matching: YYYY-MM-DD (with optional ~ prefix, optional ISO timestamp, and optional suffix)
    date_match = re.match(
        r"^(~?)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?(\(?[A-Za-z]{1,3}\)?)?$",
        val_str,
    )

    if date_match:
        tilde, year, month, day, suffix = date_match.groups()
        
        # Apply the chosen format
        if YYYYMMDD_format:
            # Ensure month and day are 2 digits (e.g., "05" instead of "5")
            formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        else:
            # Strip leading zeros using int() (e.g., "05" becomes "5")
            formatted_date = f"{int(month)}/{int(day)}/{year}"

        # keep the tilde and suffix if they exist
        tilde_str = tilde if tilde else ""
        suffix_str = suffix if suffix else ""
        result = f"{tilde_str}{formatted_date}{suffix_str}"
        return result

    # Validate non-date allowed string values
    allowed_values = {"ND", "Continuous", "missed", "inf"}
    if val_str in allowed_values:
        return val_str

    # Log error if value is outside the allowed list
    print(
        f"Error:'{val_str}'"
    )
    return val_str




def convert_data_to_all_format(df: pd.DataFrame, YYYYMMDD_format: bool = True) -> pd.DataFrame:
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
    outcome_headings = ["Breeding Type", "Complex Types"]

    # 1. Build ordered output column list
    output_cols = ["Name",]
    output_cols.extend(outcome_headings)
    output_cols.extend([f"{p}{sub}" for p in pulse_prefixes for sub in pulse_subheadings])

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
            output_rows[site_name] = {col: "ND" for col in output_cols if col not in outcome_headings}
            #leave outcome headings default to ""
            output_rows[site_name].update({col: "" for col in outcome_headings})
            output_rows[site_name]["Name"] = site_name
            #set pulse outcome columns to "n/a"
            for p in pulse_prefixes:
                output_rows[site_name][f"{p}outcome"] = "n/a"

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
        else:
            mapping = {
                "outcome": "p1outcome",
            }

        for src_col, target_col in mapping.items():
            if src_col in row and pd.notna(row[src_col]):
                if src_col == "outcome":
                    valid_outcomes = ["Successful", "Abandoned", "Partially Abandoned", 
                                        "No TRBL", "No Colony", "Unknown"]
                    if row[src_col] in valid_outcomes:
                        row_dict[target_col] = row[src_col]
                    else:
                        print(f"Invalid outcome value '{row[src_col]}' for site '{raw_site}'")
                else:
                    val = transform_value(row[src_col], raw_site, src_col,YYYYMMDD_format=YYYYMMDD_format)
                    if val != "ND":
                        row_dict[target_col] = val

        # Handle abandon vs. partial_abandon
        abandon_val = row.get("abandon", None)
        partial_val = row.get("partial_abandon", None)

        abandon_trans = transform_value(abandon_val, raw_site, "abandon",YYYYMMDD_format=YYYYMMDD_format)
        partial_trans = transform_value(
            partial_val, raw_site, "partial_abandon",YYYYMMDD_format=YYYYMMDD_format
        )

        if abandon_trans != "ND":
            row_dict[f"{pulse}abandon"] = abandon_trans
        elif partial_trans != "ND":
            row_dict[f"{pulse}abandon"] = f"{partial_trans}P"

        # Handle breeding type
        if row_dict["Breeding Type"] == "":
            row_dict["Breeding Type"] = row.get("breeding_type", None)
            row_dict["Complex Types"] = row.get("complex_types", None)

    # 4. Construct final output DataFrame
    out_df = pd.DataFrame(list(output_rows.values()), columns=output_cols)
    return out_df


def save_csv_to_extractors(df: pd.DataFrame, csv_filename: str):
    df.to_csv(csv_filename, index=False)
    print(f"Successfully created '{csv_filename}'")

    # Save to PMJ Extractor
    pmj_dest = os.path.join(PMJ_EXTRACTOR_PATH, os.path.basename(csv_filename))
    shutil.copy(csv_filename, pmj_dest)
    print(f"Copied '{csv_filename}' to '{pmj_dest}'")

    # Save to HBC Extractor
    hbc_dest = os.path.join(HBC_EXTRACTOR_PATH, os.path.basename(csv_filename))
    shutil.copy(csv_filename, hbc_dest)
    print(f"Copied '{csv_filename}' to '{hbc_dest}'")


def create_and_save_site_name_map(site_info_df: pd.DataFrame):
    '''
    The site_name_map is used by the Extractor scripts to rename the
    old sites to the new, public-facing names.

    It also contains the PMJ jobs that were used for each site so that the 
    PMJ Extractor knows which results to pull.
    '''
    site_name_map = site_info_df.copy()
    # Keep only the columns we need
    site_name_map = site_name_map[["Id", "Name", "Pretty Site Name"] 
                   + [col for col in site_name_map.columns if col.startswith("PMJ")]]

    # Save to CSV and copy to the extractor paths
    #save_csv_to_extractors(site_name_map, SITE_NAME_MAP_FILENAME)


def create_and_save_trbl_reviewer_metadata(site_info_df: pd.DataFrame):
    # Implementation for creating TRBL reviewer metadata.csv
    # Keep only the columns we need
    trbl_reviewer_metadata_df = site_info_df[["Pretty Site Name", "First Recording", "Last Recording"]].copy()

    # rename the Pretty Site Name column to Name
    trbl_reviewer_metadata_df.rename(columns={"Pretty Site Name": "Name"}, inplace=True)

    # Save to CSV locally for testing first
    #TODO print the outcome
    trbl_reviewer_metadata_df.to_csv(TRBL_REVIEWER_METADATA_FILENAME, index=False)

    #TODO: Save to the TRBL reviewer directory
    #trbl_reviewer_metadata_df.to_csv(TRBL_REVIEWER_METADATA_PATH, index=False)


def create_and_save_new_all_file_for_compatibility(
        site_info_df: pd.DataFrame, 
        data_df: pd.DataFrame
    ):
    '''
    The All file contains all data for one site on a single row. So, take the site_info sheet and 
    add to each row the relevant data from the main sheet.

    This is created so that I don't have to change the Summarizer and other code.

    The columns won't be in the same order but I don't think any functionality will break because of that.
    '''

    data_cols_df = convert_data_to_all_format(data_df, YYYYMMDD_format=False)
    all_df = site_info_df.copy()

    if len(all_df) > len(data_cols_df):
        print("Warning: The number of rows in site_info_df is greater than the number of rows in data_cols_df.")
    elif len(all_df) < len(data_cols_df):
        print("Warning: The number of rows in site_info_df is less than the number of rows in data_cols_df.")
    all_df = pd.merge(
        all_df, 
        data_cols_df, 
        left_on="Pretty Site Name",
        right_on="Name",
        how="left",
    )

    for col in ["First Recording", "Last Recording"]:
            all_df[col] = [
                transform_value(val, site, col, YYYYMMDD_format=False) 
                for val, site in zip(all_df[col], all_df["Pretty Site Name"],strict=True)
            ]

    col_map = {
        "Id": "Site ID",
        "Name_x" : "Name",
        "Name_y" : "Pretty Site Name 2",
        "Skip" : "Skip Site",
        "p1outcome" : "p1Outcome",
        "p2outcome" : "p2Outcome",
        "p3outcome" : "p3Outcome",
        "p4outcome" : "p4Outcome",
    }
    all_df.rename(columns=col_map, inplace=True)

    # Save to CSV locally for testing first
    filename = "TRBL Analysis Tracking - All(new sheet converted to old format).csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:   
        # Write two newline characters to create two empty lines
        f.write('\n\n')
        # save the data after
        all_df.to_csv(f, index=False)
        print(f"Saved {filename} to project folder")

    # Save to the Summarizer data folder
    shutil.copy(filename, INPUT_CSV)
    print(f"Copied '{filename}' to '{INPUT_CSV}'")


    return


def update_and_save_old_all_sheet(
    input_df: pd.DataFrame,
    spreadsheet_id: str = "1NQVKtxVv7zmODNuvn45TOYf-j-nU_u3Q1W-6Y_nCh-Y",
    gid: str = "0",
) -> pd.DataFrame:
    """
    Downloads the original All tracking sheet
    Updates rows matching 'Name' with input_df data,
    Saves the combined result locally while preserving the top 2 header lines.
    """
    # 1. Direct CSV Download
    csv_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv&gid={gid}"
    response = requests.get(csv_url)
    response.encoding = 'utf-8' # Force utf-8
    response.raise_for_status()

    # Split lines to isolate top 2 rows from header/data rows. It's 22 due to merged
    # cells with line breaks.
    raw_lines = response.text.splitlines(keepends=True)
    top_metadata_lines = raw_lines[:22]
    table_data = "".join(raw_lines[22:])

    # 2. Parse main sheet into DataFrame starting at Row 3 headers
    main_df = pd.read_csv(io.StringIO(table_data), dtype=str, keep_default_na=False)

    # Strip trailing spaces from strings
    main_df = main_df.map(lambda x: x.strip() if isinstance(x, str) else x)

    if "Pretty Site Name" not in main_df.columns:
        raise KeyError(
            "Column 'Pretty Site Name' was not found in Row 3 headers of the downloaded sheet."
        )

    col_map = {
        "PMJ Name"   : "PMJ Male Song",
        "PMJ Name.1" : "PMJ Male Chorus",
        "PMJ Name.2" : "PMJ Female Chatter",
        "PMJ Name.3" : "PMJ Hatchling",
        "PMJ Name.4" : "PMJ Nestling",
        "PMJ Name.5" : "PMJ Fledgling",
    }
    # Map old column names to new column names in main_df
    main_df.rename(columns=col_map, inplace=True)

    # 3. Update matching rows in main_df
    input_by_site_df = convert_data_to_all_format(input_df)
    for _, input_row in input_by_site_df.iterrows():
        site_name = input_row["Name"]
        match_mask = main_df["Pretty Site Name"] == site_name

        if match_mask.any():
            # Overwrite values for matching columns
            for col in input_by_site_df.columns:
                if col in main_df.columns:
                    main_df.loc[match_mask, col] = input_row[col]
        else:
            print(
                f"Warning: Site '{site_name}' from input_df was not found in the main sheet."
            )

    # 4. Clean the First Recording, Last Recording columns using transform_value()
    for col in ["First Recording", "Last Recording"]:
        main_df[col] = [
            transform_value(val, site, col) 
            for val, site in zip(main_df[col], main_df["Pretty Site Name"],strict=True)
        ]

    # 5. Save modified main sheet locally (preserving original top 2 rows)
    output_filename = "TRBL Analysis Tracking - All(old sheet updated to new vals).csv"
    with open(output_filename, "w", encoding="utf-8", newline="") as f:
        f.writelines(top_metadata_lines)
        main_df.to_csv(f, index=False)
    print(f"Successfully updated and saved to '{output_filename}'")

    return main_df


def clean(value: object) -> str:
    """Return a stripped string, treating None as blank."""
    return "" if value is None else str(value).strip()

COLONY_SIZE_PASSTHROUGH = {"NEED", "ND", "No Colony", "Unknown"}
INT_RE = re.compile(r"^\d+$")
RANGE_RE = re.compile(r"^(\d+)\s*[-–—]\s*(\d+)$")  # Handles hyphen, en-dash, em-dash


def parse_colony_size_metric(val: str) -> str:
    """
    Parses Approx Colony Size field into standard Colony Size metrics.
    Strips out visual comma separators before validation logic executes.
    Returns a tuple: (calculated_value_string, error_message_or_None)
    """
    cleaned = clean(val)
    if cleaned in COLONY_SIZE_PASSTHROUGH:
        return val
    
    # Strip thousands separators so numbers like 10,000 become 10000
    stripped_val = cleaned.replace(",", "")
    
    if INT_RE.match(stripped_val):
        return stripped_val
        
    if range_match := RANGE_RE.match(stripped_val):
        low, high = int(range_match.group(1)), int(range_match.group(2))
        midpoint = (low + high) // 2
        return str(midpoint)
        
    return "error"


def parse_distance_to_colony_metric(val: str) -> str:
    """
    Parses Distance to Colony field into standard Distance metrics.
    Strips out visual comma separators before validation logic executes.
    Returns a string representing the parsed distance in meters, or "error" if parsing fails.
    """
    cleaned = clean(val)
    if cleaned in COLONY_SIZE_PASSTHROUGH:
        return val
    
    # Strip thousands separators so numbers like 10,000 become 10000
    stripped_val = cleaned.replace(",", "")
    no_unit_val = stripped_val.replace(" m", "")
    no_unit_val = no_unit_val.replace("m", "")
    if INT_RE.match(no_unit_val):
        return no_unit_val

    #TODO any other distance processing?
    # if range_match := RANGE_RE.match(stripped_val):
    #     low, high = int(range_match.group(1)), int(range_match.group(2))
    #     midpoint = (low + high) // 2
    #     return str(midpoint)
        
    return "error"




def create_and_save_breeding_dates(site_info_df: pd.DataFrame, data_df: pd.DataFrame):
    # Implementation for creating breeding_dates.csv
    # Keep only the columns we need
    site_info_cols_needed = [
        "Id","Pretty Site Name", 
        "First Recording", "Last Recording", 
        "Distance to Colony", "Approx Colony Size", 
        "Substrate", "Altitude", "Latitude", "Longitude"
    ]
    data_cols_needed = [
        "site", "old_site", 
        "breeding_type", "outcome", 
        "ps_onset", "inc_onset", "brood_onset", "flgd_onset", "dispersal", 
        "abandon","partial_abandon"
    ]

    # Subset dataframes to avoid mutating original data
    metadata_df = site_info_df[site_info_cols_needed].copy()
    result_df = data_df[data_cols_needed].copy()

    # Clean up the raw columns from the merged dataframe before parsing new metrics
    # Create a new column to hold the parsed results alongside the originals
    metadata_df["approx_colony_size"] = metadata_df["Approx Colony Size"].apply(parse_colony_size_metric)
    metadata_df["distance_to_colony"] = metadata_df["Distance to Colony"].apply(parse_distance_to_colony_metric)
    metadata_df["first_recording"] = metadata_df["First Recording"].apply(transform_date)
    metadata_df["last_recording"] = metadata_df["Last Recording"].apply(transform_date)

    # Drop the raw value columns now that we've parsed them
    metadata_df = metadata_df.drop(columns=[
        "Approx Colony Size", 
        "Distance to Colony", 
        "First Recording", 
        "Last Recording"
    ])

    # Clean up all column names to be lowercase and snake_case for consistency
    metadata_col_map = {
        "Id" : "id",
        "Pretty Site Name": "site",
        "Substrate": "substrate",
        "Altitude": "altitude",
        "Latitude": "latitude",
        "Longitude": "longitude",
    }
    metadata_df = metadata_df.rename(columns=metadata_col_map)
    metadata_df["evidence_site"] = metadata_df["site"]

    result_col_map = {
        "site" : "pretty_site_name",
        "ps_onset": "settlement_start",
        "inc_onset": "incubation_onset",
        "brood_onset": "brooding_onset",
        "flgd_onset": "fledging_onset",
        "dispersal": "fledgling_dispersal",
        "abandon": "abandon_date",
        "partial_abandon": "partial_abandon_date",
    }
    result_df = result_df.rename(columns=result_col_map)

    # Add the id to the result_df column for mapping
    site_id_cols_needed = ['id', 'site']
    site_id_df = metadata_df[site_id_cols_needed].copy()

    # 1. Create a temporary merge key by stripping the pulse suffix using regex
    # r" p\d+$" looks for a space, a 'p', and digits at the very end of the string
    # "2017 Rush Ranch p1" -> "2017 Rush Ranch"
    # "2018 Iron Point" -> "2018 Iron Point" (unchanged because regex doesn't match)
    result_df["base_site_name"] = result_df["pretty_site_name"].str.replace(r" P\d+$", "", regex=True)

    # Extract the 'P' and numbers at the end of the string, and fill misses with ""
    result_df["pulse"] = (
        result_df["pretty_site_name"]
        .str.extract(r" ([Pp]\d+)$", expand=False)
        .fillna("")
    )

    # 2. Merge the site info data into the results using that base name
    accepted_chronology_df = pd.merge(
        result_df,
        site_id_df,
        left_on="base_site_name",
        right_on="site",
        how="left"
    )
    # Also make the breeding dates, which I'm not sure we will need in the end but that has all the data in it
    breeding_data_df = pd.merge(
        result_df,
        metadata_df,
        left_on="base_site_name",
        right_on="site",
        how="left"
    )

    # 3. Drop the temporary mapping columns
    accepted_chronology_df = accepted_chronology_df.drop(columns=["base_site_name", "pretty_site_name"])
    breeding_data_df = breeding_data_df.drop(columns=["site"])

    # 4. Finalize column order and ensure we only have what we need
    breeding_data_col_map = {
        "pretty_site_name": "site",
    }
    breeding_data_df = breeding_data_df.rename(columns=breeding_data_col_map)
    breeding_data_cols = [
        "id", "site", "base_site_name", "old_site", "pulse",
        "breeding_type", "outcome", 
        "settlement_start", "incubation_onset", "brooding_onset", "fledging_onset", "fledgling_dispersal", 
        "abandon_date", "partial_abandon_date",
        "first_recording", "last_recording", 
        "distance_to_colony", "approx_colony_size", 
        "altitude", "latitude", "longitude",
        "substrate", 
    ]
    breeding_data_df = breeding_data_df[breeding_data_cols].copy()

    accepted_chronology_cols = [
        "id", "site", "old_site", "pulse",
        "breeding_type", "outcome", 
        "settlement_start", "incubation_onset", "brooding_onset", "fledging_onset", "fledgling_dispersal", 
        "abandon_date", "partial_abandon_date",
    ]
    accepted_chronology_df = accepted_chronology_df[accepted_chronology_cols].copy()

    metadata_cols = [
        "id", "site", "evidence_site",
        "first_recording", "last_recording", 
        "distance_to_colony", "approx_colony_size", 
        "altitude", "latitude", "longitude",
        "substrate",
    ]
    metadata_df = metadata_df[metadata_cols].copy()

    # Save the cleaned and parsed breeding dates to a CSV file
    file_map = {
        "chronology_and_metadata.csv": breeding_data_df,
        "trbl_site_metadata.csv": metadata_df,
        "trbl_accepted_chronology.csv": accepted_chronology_df
    }
    for file_name, df in file_map.items():
        df.to_csv(file_name, index=False, encoding="utf-8-sig")
        print(f"Saved {file_name}")

    return


if __name__ == "__main__":
    site_info_df = get_data_from_site_info_sheet()
    data_df = get_data_from_main_sheet(site_info_df)


    # # Create site_name_map.csv (ID, Name, Pretty Site Name, PMJ* Columns)
    # create_and_save_site_name_map(site_info_df)

    # # Create TRBL reviewer metadata.csv (Name	First Recording	Last Recording)
    # create_and_save_trbl_reviewer_metadata(site_info_df)

    # # Create a version of the All file for compatibility
    # # I want to do two things here:
    # # 1. Make a version of the all file using the latest data. That's what we need for
    # # the summarizer and other tools that rely on the latest "All" file.
    #create_and_save_new_all_file_for_compatibility(site_info_df, data_df)

    # # 2. Make a version of the data file in the format of the old "All" file for comparison.
    # update_and_save_old_all_sheet(data_df)
    
    create_and_save_breeding_dates(site_info_df, data_df)

    #TODO: Any other files needed?
