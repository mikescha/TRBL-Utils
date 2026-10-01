import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from common import (
    ARI_SCORE_FILE,
    DEFAULT_AUTOMATED_RESULTS_DIR,
    WEBHOOK_URL,
)


def load_automated_results_from_csv(dir:Path = DEFAULT_AUTOMATED_RESULTS_DIR) -> pd.DataFrame:
    """
    Load automated results from a CSV file into a pandas DataFrame.
    
    Args:
        dir (Path): Directory containing the CSV file. Defaults to DEFAULT_AUTOMATED_RESULTS_DIR.

    Returns:
        pd.DataFrame: DataFrame containing the loaded automated results.
    """
    file_path = dir / "trbl_breeding_chronology_summary.csv"
    df_all_data = pd.read_csv(file_path)

    #fix up the strings the way we like them
    df_all_data.replace(
        {"no_data": "ND", "inferred_pre_recording": "inf"}, inplace=True
    )

    cols_to_keep = [
        "site","pulse",
        "settlement_start","settlement_end",
        "incubation_onset","brooding_onset","fledging_onset","fledgling_dispersal"
    ]
    df_main_cols = df_all_data[cols_to_keep].copy()

    # 1. Convert columns to string series
    site_str = df_main_cols['site'].astype(str)
    pulse_str = df_main_cols['pulse'].astype(str)

    # 2. Build full string, then replace with just 'site' if pulse is 'no_pulse'
    site_pulse_vals = (site_str + ' ' + pulse_str).mask(pulse_str == 'no_pulse', site_str)

    # 3. Drop the original cols
    df_main_cols = df_main_cols.drop(columns=['site', 'pulse'])

    # 4. Insert the combined column
    df_main_cols.insert(
        0, #position
        'site', #column name
        site_pulse_vals
    )

    return df_main_cols


def load_ARI_score_from_csv() -> pd.DataFrame:
    df_ARI = pd.read_csv(ARI_SCORE_FILE)
    
    return df_ARI



def push_via_webhook(df: pd.DataFrame, sheet_name: str = ""):
    if len(sheet_name) == 0:
        sheet_name = f"automated_results_{dt.datetime.now().strftime('%Y%m%d')}"
    
    # Clean NaN values and convert all columns to string
    df_clean = df.fillna("").astype(str)
    
    # Standardize key column formatting
    if "nest_id" in df_clean.columns:
        df_clean["nest_id"] = df_clean["nest_id"].str.strip().str.upper()
        
    payload = {
        "sheet_name": sheet_name,
        "headers": df_clean.columns.tolist(),
        "rows": df_clean.values.tolist()
    }
    
    # POST JSON payload to Google Apps Script Web App
    response = requests.post(WEBHOOK_URL, json=payload)
    response.raise_for_status()
    
    result = response.json()
    if result.get("status") == "success":
        print(f"✅ Updated '{sheet_name}' with {result.get('rows_written')} rows via Webhook.")
    else:
        print(f"❌ Apps Script Error: {result.get('message')}")


if __name__ == "__main__":
    latest_results_dir = Path(
        "C:\\Users\\mikes\\GitHub\\TRBL-Breeding-Stages-Final\\outputs\\new_data_review_20260929\\publication\\"
    )
    df_results = load_automated_results_from_csv(dir=latest_results_dir)

    # with no sheet name parameter, the script will automatically use today's date as the sheet name
    push_via_webhook(df_results)

    #df_ARI = load_ARI_score_from_csv()
    #push_via_webhook(df_ARI, sheet_name="ARI_scores")