"""
data_cleaning.py
Phase 2 & 3 — Data Cleaning and Validation
DS_Day01_59: Fintech — Where Are Digital Payments Failing?
"""

import pandas as pd
import numpy as np
import json
import os


RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'upi_transactions_2024.csv')
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_transactions.csv')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports', 'data_quality_report.json')


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw dataset without modification."""
    df = pd.read_csv(path)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names (lowercase, underscores).
    Original column 'transaction id' has a space — mapped to 'transaction_id'.
    Original 'amount (INR)' mapped to 'amount_inr'.
    Original 'transaction type' mapped to 'transaction_type'.
    """
    rename_map = {
        'transaction id': 'transaction_id',
        'amount (INR)': 'amount_inr',
        'transaction type': 'transaction_type',
    }
    df = df.rename(columns=rename_map)
    # Normalise all remaining column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp string to datetime; derive date, hour, Time_Period, Day_of_Week."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['transaction_date'] = df['timestamp'].dt.date
    # hour_of_day already present; overwrite to be safe
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5

    def time_period(h):
        if 0 <= h <= 5:
            return 'Early Morning'
        elif 6 <= h <= 11:
            return 'Morning'
        elif 12 <= h <= 17:
            return 'Afternoon'
        else:
            return 'Evening'

    df['time_period'] = df['hour'].apply(time_period)
    return df


def normalize_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise transaction_status to title-case (Success / Failed).
    Create binary target Failure: 0 = Success, 1 = Failed.
    """
    df['transaction_status'] = df['transaction_status'].str.strip().str.upper()
    # Mapping: any variant of SUCCESS → 'Success', FAILED → 'Failed'
    status_map = {'SUCCESS': 'Success', 'FAILED': 'Failed'}
    df['transaction_status_clean'] = df['transaction_status'].map(status_map)

    # Document unmapped values
    unmapped = df[df['transaction_status_clean'].isna()]['transaction_status'].unique()
    if len(unmapped) > 0:
        print(f"[WARNING] Unmapped transaction_status values: {unmapped}")

    df['failure'] = (df['transaction_status_clean'] == 'Failed').astype(int)
    return df


def validate_amounts(df: pd.DataFrame) -> dict:
    """Check amount validity; document findings but do not remove outliers."""
    report = {
        'negative_amounts': int((df['amount_inr'] < 0).sum()),
        'zero_amounts': int((df['amount_inr'] == 0).sum()),
        'amount_min': float(df['amount_inr'].min()),
        'amount_max': float(df['amount_inr'].max()),
        'amount_mean': float(df['amount_inr'].mean()),
        'amount_p99': float(df['amount_inr'].quantile(0.99)),
    }
    return report


def clean(raw_path: str = RAW_PATH) -> pd.DataFrame:
    """Full cleaning pipeline. Returns cleaned DataFrame."""
    df = load_raw(raw_path)
    rows_before = len(df)

    # Step 1 — column names
    df = clean_column_names(df)

    # Step 2 — duplicates
    dup_full_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()

    dup_txn_ids = int(df['transaction_id'].duplicated().sum())  # after full-row dedup

    # Step 3 — missing values
    missing_before = df.isnull().sum().to_dict()

    # Step 4 — timestamp & time features
    df = parse_timestamp(df)

    # Step 5 — normalise status + create target
    df = normalize_status(df)

    # Step 6 — amount validation
    amount_report = validate_amounts(df)

    # Step 7 — drop rows with unparseable timestamp
    invalid_ts = int(df['timestamp'].isna().sum())
    df = df[df['timestamp'].notna()]

    # Step 8 — drop rows with unmapped status
    invalid_status = int(df['transaction_status_clean'].isna().sum())
    df = df[df['transaction_status_clean'].notna()]

    rows_after = len(df)

    # Data quality report
    quality_report = {
        'rows_before_cleaning': rows_before,
        'full_duplicate_rows_removed': dup_full_rows,
        'duplicate_transaction_ids_remaining': dup_txn_ids,
        'missing_values_before': {k: int(v) for k, v in missing_before.items()},
        'invalid_timestamps_removed': invalid_ts,
        'invalid_status_removed': invalid_status,
        'rows_after_cleaning': rows_after,
        'amount_validation': amount_report,
        'status_mapping': {'SUCCESS': 'Success (failure=0)', 'FAILED': 'Failed (failure=1)'},
        'time_period_mapping': {
            'Early Morning': '00:00-05:59',
            'Morning': '06:00-11:59',
            'Afternoon': '12:00-17:59',
            'Evening': '18:00-23:59',
        },
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        json.dump(quality_report, f, indent=2)

    print(f"[CLEAN] Rows before: {rows_before} | Rows after: {rows_after}")
    print(f"[CLEAN] Duplicates removed: {dup_full_rows}")
    print(f"[CLEAN] Data quality report saved to: {REPORT_PATH}")
    return df


def save_cleaned(df: pd.DataFrame, path: str = PROCESSED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[CLEAN] Cleaned dataset saved to: {path}")


if __name__ == '__main__':
    df_clean = clean()
    save_cleaned(df_clean)
    print(df_clean.info())
    print(df_clean.head())
