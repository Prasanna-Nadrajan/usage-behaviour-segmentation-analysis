"""
Exploratory Data Analysis (EDA) module.
Generates distributions, correlation heatmaps, and summary statistics.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    summary_stats = df.describe().to_dict()

    # 1. Distribution of Key Metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df['session_count_90d'], kde=True, ax=axes[0, 0], color='#2b5c8f')
    axes[0, 0].set_title("Distribution of Session Counts (90d)", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Sessions")

    sns.histplot(df['avg_session_duration_min'], kde=True, ax=axes[0, 1], color='#e26d5c')
    axes[0, 1].set_title("Avg Session Duration (min)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Minutes")

    sns.histplot(df['revenue_90d'], kde=True, ax=axes[1, 0], color='#38b000')
    axes[1, 0].set_title("Revenue (90d)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("USD ($)")

    sns.countplot(data=df, x='features_used_count', ax=axes[1, 1], hue='features_used_count', palette='viridis', legend=False)
    axes[1, 1].set_title("Features Used Count (Breadth)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Distinct Features")

    plt.tight_layout()
    dist_path = os.path.join(output_dir, "01_distributions.png")
    plt.savefig(dist_path, dpi=200)
    plt.close()

    # 2. Correlation Matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, cbar_kws={'shrink': 0.8})
    plt.title("Behavioural Metric Correlation Heatmap", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "02_correlation_matrix.png")
    plt.savefig(corr_path, dpi=200)
    plt.close()

    # 3. Categorical Engagement: Plan Type vs Revenue & Sessions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x='plan_type', y='session_count_90d', ax=axes[0], palette='Blues_d', hue='plan_type', legend=False)
    axes[0].set_title("Sessions by Plan Tier", fontsize=12, fontweight='bold')
    
    sns.boxplot(data=df, x='plan_type', y='revenue_90d', ax=axes[1], palette='Greens_d', hue='plan_type', legend=False)
    axes[1].set_title("Revenue by Plan Tier", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plan_path = os.path.join(output_dir, "03_plan_tier_breakdown.png")
    plt.savefig(plan_path, dpi=200)
    plt.close()

    return {
        "dist_chart": dist_path,
        "corr_chart": corr_path,
        "plan_chart": plan_path,
        "stats": summary_stats
    }

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/cleaned_users.parquet")
    res = run_eda(df)
    print("EDA completed successfully. Generated charts in reports/figures/")
