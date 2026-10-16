"""
Data generation module for SaaS product usage behaviour segmentation.
Generates realistic user activity, engagement, monetary, and support records.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(n_users: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic SaaS product usage data.
    
    Archetypes simulated with realistic correlations:
    1. Power Users (~15%): High frequency, diverse feature adoption, high revenue.
    2. Regular Core Users (~35%): Consistent mid-tier engagement, steady feature use.
    3. Casual Browsers (~25%): Low session duration, mostly mobile, low revenue/free.
    4. At-Risk / Churning (~15%): Dropping usage, high recency (dormant), high support tickets.
    5. New Onboarding (~10%): Recent signups, exploring features, mixed duration.
    """
    np.random.seed(random_seed)
    ref_date = datetime(2026, 10, 1)

    archetype_probs = [0.15, 0.35, 0.25, 0.15, 0.10]
    archetypes = np.random.choice(
        ['power', 'core', 'casual', 'at_risk', 'new'],
        size=n_users,
        p=archetype_probs
    )

    user_ids = [f"USR_{i:06d}" for i in range(1, n_users + 1)]
    signup_dates = []
    last_active_dates = []
    session_counts_90d = []
    avg_session_durations = []
    features_used_counts = []
    transactions_90d = []
    revenue_90d = []
    support_tickets = []
    plan_types = []
    primary_devices = []
    recent_30d_session_share = []
    weekend_session_share = []

    for arch in archetypes:
        if arch == 'power':
            signup_age = np.random.randint(90, 730)
            recency_days = int(np.random.exponential(scale=3))
            recency_days = min(recency_days, 15)
            sessions = int(np.random.normal(loc=110, scale=25))
            duration = np.random.normal(loc=32, scale=8)
            features = np.random.choice([7, 8, 9, 10], p=[0.1, 0.2, 0.4, 0.3])
            plan = np.random.choice(['Enterprise', 'Pro', 'Free'], p=[0.55, 0.40, 0.05])
            tx = int(np.random.normal(loc=14, scale=4))
            base_rev = 120.0 if plan == 'Enterprise' else (45.0 if plan == 'Pro' else 0.0)
            rev = max(0.0, base_rev * 3 + np.random.normal(50, 20))
            tickets = np.random.poisson(lam=0.8)
            device = np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.70, 0.25, 0.05])
            share_30d = np.clip(np.random.normal(0.35, 0.05), 0.20, 0.55)
            weekend_share = np.clip(np.random.normal(0.18, 0.05), 0.05, 0.40)

        elif arch == 'core':
            signup_age = np.random.randint(60, 500)
            recency_days = int(np.random.exponential(scale=7))
            recency_days = min(recency_days, 35)
            sessions = int(np.random.normal(loc=55, scale=15))
            duration = np.random.normal(loc=20, scale=5)
            features = np.random.choice([4, 5, 6, 7], p=[0.25, 0.40, 0.25, 0.10])
            plan = np.random.choice(['Pro', 'Starter', 'Free'], p=[0.45, 0.45, 0.10])
            tx = int(np.random.normal(loc=6, scale=2))
            base_rev = 45.0 if plan == 'Pro' else (20.0 if plan == 'Starter' else 0.0)
            rev = max(0.0, base_rev * 3 + np.random.normal(15, 10))
            tickets = np.random.poisson(lam=1.1)
            device = np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.50, 0.40, 0.10])
            share_30d = np.clip(np.random.normal(0.33, 0.05), 0.18, 0.50)
            weekend_share = np.clip(np.random.normal(0.24, 0.06), 0.08, 0.45)

        elif arch == 'casual':
            signup_age = np.random.randint(30, 400)
            recency_days = int(np.random.exponential(scale=18))
            recency_days = min(recency_days, 60)
            sessions = int(np.random.normal(loc=18, scale=8))
            duration = np.random.normal(loc=8, scale=3)
            features = np.random.choice([1, 2, 3], p=[0.40, 0.45, 0.15])
            plan = np.random.choice(['Free', 'Starter', 'Pro'], p=[0.75, 0.20, 0.05])
            tx = int(np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10]))
            base_rev = 20.0 if plan == 'Starter' else (45.0 if plan == 'Pro' else 0.0)
            rev = max(0.0, (base_rev * 3 if plan != 'Free' else 0.0) + (10.0 if tx > 0 else 0.0))
            tickets = np.random.poisson(lam=0.4)
            device = np.random.choice(['Mobile', 'Desktop', 'Tablet'], p=[0.65, 0.25, 0.10])
            share_30d = np.clip(np.random.normal(0.30, 0.08), 0.10, 0.50)
            weekend_share = np.clip(np.random.normal(0.35, 0.08), 0.10, 0.60)

        elif arch == 'at_risk':
            signup_age = np.random.randint(90, 600)
            recency_days = int(np.random.uniform(40, 90))
            sessions = int(np.random.normal(loc=12, scale=6))
            duration = np.random.normal(loc=6, scale=2.5)
            features = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
            plan = np.random.choice(['Pro', 'Starter', 'Enterprise', 'Free'], p=[0.40, 0.30, 0.10, 0.20])
            tx = int(np.random.choice([0, 1], p=[0.80, 0.20]))
            rev = 15.0 if tx > 0 else 0.0
            tickets = np.random.poisson(lam=3.5)
            device = np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.45, 0.45, 0.10])
            share_30d = np.clip(np.random.normal(0.08, 0.05), 0.0, 0.20)
            weekend_share = np.clip(np.random.normal(0.20, 0.08), 0.05, 0.50)

        else: # 'new'
            signup_age = np.random.randint(5, 45)
            recency_days = int(np.random.randint(0, min(signup_age, 12)))
            sessions = int(np.random.normal(loc=22, scale=10))
            duration = np.random.normal(loc=16, scale=5)
            features = np.random.choice([2, 3, 4, 5], p=[0.3, 0.35, 0.25, 0.10])
            plan = np.random.choice(['Free', 'Starter', 'Pro'], p=[0.55, 0.30, 0.15])
            tx = int(np.random.choice([0, 1, 2], p=[0.65, 0.25, 0.10]))
            rev = max(0.0, (20.0 if plan == 'Starter' else (45.0 if plan == 'Pro' else 0.0)) + tx * 10)
            tickets = np.random.poisson(lam=1.5)
            device = np.random.choice(['Desktop', 'Mobile', 'Tablet'], p=[0.50, 0.40, 0.10])
            share_30d = 1.0 if signup_age <= 30 else np.clip(np.random.normal(0.70, 0.10), 0.40, 1.0)
            weekend_share = np.clip(np.random.normal(0.26, 0.07), 0.05, 0.50)

        s_date = ref_date - timedelta(days=int(signup_age))
        la_date = ref_date - timedelta(days=int(recency_days))
        if la_date < s_date:
            la_date = s_date

        signup_dates.append(s_date.strftime('%Y-%m-%d'))
        last_active_dates.append(la_date.strftime('%Y-%m-%d'))
        session_counts_90d.append(max(1, sessions))
        avg_session_durations.append(max(1.0, round(duration, 2)))
        features_used_counts.append(max(1, int(features)))
        transactions_90d.append(max(0, tx))
        revenue_90d.append(round(rev, 2))
        support_tickets.append(tickets)
        plan_types.append(plan)
        primary_devices.append(device)
        recent_30d_session_share.append(round(share_30d, 4))
        weekend_session_share.append(round(weekend_share, 4))

    df = pd.DataFrame({
        'user_id': user_ids,
        'signup_date': signup_dates,
        'last_active_date': last_active_dates,
        'session_count_90d': session_counts_90d,
        'avg_session_duration_min': avg_session_durations,
        'features_used_count': features_used_counts,
        'transactions_90d': transactions_90d,
        'revenue_90d': revenue_90d,
        'support_tickets_count': support_tickets,
        'plan_type': plan_types,
        'primary_device': primary_devices,
        'recent_30d_session_share': recent_30d_session_share,
        'weekend_session_share': weekend_session_share,
    })

    # Introduce realistic noise: minor missing values & slight duplicates
    # Missing values ~ 1.5% in avg_session_duration_min, 1.0% in primary_device
    missing_idx_dur = np.random.choice(n_users, size=int(n_users * 0.015), replace=False)
    df.loc[missing_idx_dur, 'avg_session_duration_min'] = np.nan

    missing_idx_dev = np.random.choice(n_users, size=int(n_users * 0.010), replace=False)
    df.loc[missing_idx_dev, 'primary_device'] = np.nan

    # 15 duplicate rows to simulate pipeline repeat ingestion
    dup_sample = df.sample(n=15, random_state=random_seed)
    df = pd.concat([df, dup_sample], ignore_index=True)

    return df

if __name__ == "__main__":
    df = generate_synthetic_data(10000)
    df.to_csv("data/raw/user_behaviour_raw.csv", index=False)
    print(f"Dataset generated successfully. Shape: {df.shape}")
