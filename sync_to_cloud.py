
import pandas as pd
import requests


def load_automated_results_from_csv() -> pd.DataFrame:
    """
    Load automated results from a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    """
    dir = "C:\\Users\\mikes\\GitHub\\TRBL-Breeding-Stages-Final\\outputs\\publication\\"
    file_path = dir + "trbl_breeding_chronology_summary.csv"
    df_all_data = pd.read_csv(file_path)

    cols_to_keep = [
        "site","pulse","settlement_start","settlement_end","incubation_onset","brooding_onset","fledging_onset","fledgling_dispersal"
    ]
    df_main_cols = df_all_data[cols_to_keep]

    #Put other processing here
    
    return df_main_cols


def sync_via_webhook(df: pd.DataFrame, webhook_url: str, sheet_name: str = "automated_results"):
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
    response = requests.post(webhook_url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    if result.get("status") == "success":
        print(f"✅ Updated '{sheet_name}' with {result.get('rows_written')} rows via Webhook.")
    else:
        print(f"❌ Apps Script Error: {result.get('message')}")

# --- Usage Example ---
if __name__ == "__main__":
    WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxtzRiML6eQQCyERCQSywEvLZYCFglybWn5CQ_WJuHC6Mw77SbTIvkjulu6F16Ob4EWMg/exec"

    df_results = load_automated_results_from_csv()

    sync_via_webhook(df_results, WEBHOOK_URL)