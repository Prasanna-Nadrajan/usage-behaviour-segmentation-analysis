"""
Behavioural Feature Engineering module.
Constructs RFM, engagement intensity, trend, usage diversity, and time patterns.
"""
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame, ref_date: str = "2026-10-01") -> pd.DataFrame:
    """
    Transforms cleaned usage data into behavioural clustering features.
    
    Features created:
    1. RFM:
       - recency_days: Days since last active date relative to ref_date.
       - frequency: sessions_per_week over tenure or 90d window.
       - monetary: revenue_90d and revenue_per_session.
    2. Engagement Intensity:
       - sessions_per_week: Weekly session rate.
       - total_engagement_time_hours: Total hours spent in 90 days.
       - feature_adoption_rate: Proportion of all available features used (out of 10).
    3. Trend & Momentum:
       - momentum_30d_vs_expected: Ratio of recent 30d session share vs expected uniform share (0.333).
         (> 1 means accelerating usage, < 1 means decelerating / churn signal).
    4. Time-of-week patterns:
       - weekend_session_share: Weekend usage proportion.
    5. Support Friction:
       - support_ticket_intensity: Support tickets per session (friction index).
    """
    feat_df = df.copy()
    ref = pd.to_datetime(ref_date)
    
    # Tenure in days
    tenure_days = (ref - feat_df['signup_date']).dt.days.clip(lower=1)
    feat_df['tenure_days'] = tenure_days
    
    # Recency
    feat_df['recency_days'] = (ref - feat_df['last_active_date']).dt.days.clip(lower=0)
    
    # Active window for frequency calculation (min between tenure and 90 days)
    active_window_weeks = np.minimum(feat_df['tenure_days'], 90) / 7.0
    feat_df['sessions_per_week'] = (feat_df['session_count_90d'] / active_window_weeks).round(2)
    
    # Monetary & Unit Economics
    feat_df['monetary_revenue_90d'] = feat_df['revenue_90d']
    feat_df['revenue_per_session'] = (feat_df['revenue_90d'] / feat_df['session_count_90d'].replace(0, np.nan)).fillna(0).round(2)

    # Engagement intensity
    feat_df['avg_session_duration_min'] = feat_df['avg_session_duration_min']
    feat_df['total_engagement_hours_90d'] = (
        (feat_df['session_count_90d'] * feat_df['avg_session_duration_min']) / 60.0
    ).round(2)
    
    # Feature adoption rate (assuming SaaS suite offers 10 core features)
    total_platform_features = 10
    feat_df['feature_adoption_rate'] = (feat_df['features_used_count'] / total_platform_features).clip(0.0, 1.0).round(2)
    
    # Usage momentum (recent 30d share vs 30/90 baseline = 0.333)
    feat_df['usage_momentum_30d'] = (feat_df['recent_30d_session_share'] / 0.3333).round(3)
    
    # Support ticket friction
    feat_df['support_ticket_friction'] = (
        feat_df['support_tickets_count'] / feat_df['session_count_90d'].replace(0, 1)
    ).round(4)
    
    return feat_df

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/cleaned_users.parquet")
    features_df = engineer_features(df)
    features_df.to_parquet("data/processed/features_users.parquet", index=False)
    print(f"Features engineered successfully. Shape: {features_df.shape}")
