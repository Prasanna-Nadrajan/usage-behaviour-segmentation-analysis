"""
Data cleaning and validation module.
Handles missing data, deduplication, type casting, and anomaly handling.
"""
import pandas as pd
import numpy as np

def clean_usage_data(df: pd.DataFrame, ref_date: str = "2026-10-01") -> tuple[pd.DataFrame, dict]:
    """
    Cleans raw user usage behavior data.
    
    Returns:
        cleaned_df: Cleaned pandas DataFrame
        cleaning_report: Dictionary detailing rows removed, imputed, and transformed
    """
    report = {}
    init_shape = df.shape
    report['initial_rows'] = init_shape[0]

    # 1. Deduplication on user_id
    dups_count = df.duplicated(subset=['user_id']).sum()
    df_clean = df.drop_duplicates(subset=['user_id'], keep='first').copy()
    report['duplicates_removed'] = int(dups_count)

    # 2. Date type conversions
    df_clean['signup_date'] = pd.to_datetime(df_clean['signup_date'])
    df_clean['last_active_date'] = pd.to_datetime(df_clean['last_active_date'])
    
    # Logical check: last_active_date should not precede signup_date
    inverted_dates = (df_clean['last_active_date'] < df_clean['signup_date']).sum()
    report['inverted_dates_adjusted'] = int(inverted_dates)
    df_clean.loc[df_clean['last_active_date'] < df_clean['signup_date'], 'last_active_date'] = df_clean['signup_date']

    # 3. Missing values imputation
    # avg_session_duration_min: impute with median grouped by plan_type
    dur_missing_before = df_clean['avg_session_duration_min'].isna().sum()
    report['avg_session_duration_missing_imputed'] = int(dur_missing_before)
    median_dur_by_plan = df_clean.groupby('plan_type')['avg_session_duration_min'].transform('median')
    df_clean['avg_session_duration_min'] = df_clean['avg_session_duration_min'].fillna(median_dur_by_plan)

    # primary_device: impute missing with 'Unknown' category to preserve signal without false assumption
    dev_missing_before = df_clean['primary_device'].isna().sum()
    report['primary_device_missing_imputed'] = int(dev_missing_before)
    df_clean['primary_device'] = df_clean['primary_device'].fillna('Unknown')

    # 4. Numeric integrity
    non_negative_cols = [
        'session_count_90d', 'avg_session_duration_min', 'features_used_count',
        'transactions_90d', 'revenue_90d', 'support_tickets_count'
    ]
    for col in non_negative_cols:
        df_clean[col] = df_clean[col].clip(lower=0)

    # Ensure proportion bounds [0, 1]
    df_clean['recent_30d_session_share'] = df_clean['recent_30d_session_share'].clip(0.0, 1.0)
    df_clean['weekend_session_share'] = df_clean['weekend_session_share'].clip(0.0, 1.0)

    report['final_rows'] = df_clean.shape[0]
    return df_clean, report

if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/user_behaviour_raw.csv")
    cleaned_df, report = clean_usage_data(raw_df)
    cleaned_df.to_parquet("data/processed/cleaned_users.parquet", index=False)
    print("Cleaning Report:", report)
