'''
Common configuration and constants for TRBL Utils.
'''

import datetime
import os
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import requests

WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxtzRiML6eQQCyERCQSywEvLZYCFglybWn5CQ_WJuHC6Mw77SbTIvkjulu6F16Ob4EWMg/exec"


# File locations
BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "output"

#Map for the data extractors to use
SITE_NAME_MAP_FILENAME = "site_name_map.csv"
PMJ_EXTRACTOR_PATH = r"C:\Users\mikes\GitHub\TRBL-Extractor-PMJ"
HBC_EXTRACTOR_PATH = r"C:\Users\mikes\GitHub\TRBL-Extractor-Data"

#Location in the main analysis project
TRBL_REVIEWER_DIR = Path(r"C:\Users\mikes\GitHub\TRBL-Breeding-Stages-Final\reviewer_inputs")
DEFAULT_AUTOMATED_RESULTS_DIR = (
    Path("C:\\Users\\mikes\\GitHub\\TRBL-Breeding-Stages-Final\\outputs\\publication")
)


OLD_ALL_FILE_FOR_STREAMLIT_APP = Path(
    r"C:\Users\mikes\GitHub\TRBLSummarizer\TRBLSummarizer\Data\TRBL Analysis tracking - All.csv"
)

ARI_SCORE_FILE = Path(
    "C:\\Users\\mikes\\GitHub\\TRBL-Breeding-Stages-Final\\outputs\\ari\\trbl_acoustic_reproductive_index.csv"
)

SHARING_OUTPUT_DIR = Path(r"G:\My Drive\TRBL for Wendy GDrive")

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


def save_csv_with_retry(df: pd.DataFrame, path: Path, share = False) -> None:
    def add_timestamp(filename: str) -> str:
        path = Path(filename)
        timestamp = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        return str(path.with_name(f"{path.stem} {timestamp}{path.suffix}"))

    # Prep the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    while True:
        try:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"✅ Saved CSV to '{path}'")
            break
        except PermissionError:
            input(f"\n[!] Output file is locked in Excel: {path.name}\nClose it and press Enter to retry...")

    if share and SHARING_OUTPUT_DIR.exists():
        shutil.copy2(path, SHARING_OUTPUT_DIR / path.name)


# OTHER VERSION SAVED UNTIL LATER
# def transform_value(val, site: str, col: str) -> str:
#     """Transforms a single cell value:

#     - Formats dates to M/D/YYYY (with ~ for estimates or suffixes like (C)).
#     - Handles ISO 8601 timestamps (e.g., '2023-07-05T07:00:00.000Z').
#     - Validates string tokens against the allowed list.
#     - Prints an error for unallowed values.
#     """
#     if pd.isna(val) or val is None:
#         return "ND"

#     # Handle native pandas/datetime objects or strings
#     if isinstance(val, (datetime.date, datetime.datetime, pd.Timestamp)):
#         val_str = val.strftime("%Y-%m-%d")
#     else:
#         val_str = str(val).strip()

#     if not val_str:
#         return "ND"

#     # Regex matching: YYYY-MM-DD (with optional ~ prefix, optional ISO timestamp, and optional suffix)
#     date_match = re.match(
#         r"^(~?)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?(?:\(?([A-Za-z]{1,3})\)?)?$",
#         val_str,
#     )

#     if date_match:
#         tilde, year, month, day, suffix = date_match.groups()
#         formatted_date = f"{int(month)}/{int(day)}/{year}"

#         # Prefix with ~ if tilde existed or a suffix was present
#         if tilde or suffix:
#             return f"~{formatted_date}"
#         return formatted_date

#     # Validate non-date allowed string values
#     allowed_values = {"ND", "continuous", "missed", "inf"}
#     if val_str in allowed_values:
#         return val_str

#     # Log error if value is outside the allowed list
#     print(
#         f"Error: Site '{site}', Column '{col}' contains invalid value '{val_str}'"
#     )
#     return val_str


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
    allowed_values = {"ND", "continuous", "missed", "inf"}
    if val_str in allowed_values:
        return val_str

    # Log error if value is outside the allowed list
    print(
        f"Error: Site '{site}', Column '{col}' contains invalid value '{val_str}'"
    )
    return val_str
