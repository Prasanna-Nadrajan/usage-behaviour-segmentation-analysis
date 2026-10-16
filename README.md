# 📊 End-to-End SaaS Usage Behaviour Segmentation

An end-to-end Machine Learning segmentation project that clusters **10,000 SaaS users** into actionable behavioral segments based on product interaction telemetry — from synthetic data generation through feature engineering, unsupervised clustering, and an interactive Streamlit dashboard with live inference.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Directory Structure](#directory-structure)
- [Pipeline Stages](#pipeline-stages)
  - [1. Data Generation](#1-data-generation)
  - [2. Data Cleaning & Validation](#2-data-cleaning--validation)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
  - [4. Behavioural Feature Engineering](#4-behavioural-feature-engineering)
  - [5. Modeling & Clustering](#5-modeling--clustering)
- [Discovered Segments](#discovered-segments)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Traditional user segmentation relies on static demographics (age, location, plan tier). This project takes a **behavioural-first** approach, segmenting users by *what they actually do* within the product:

- How frequently they log in
- How deeply they engage with features
- Whether their usage is accelerating or decelerating
- How much revenue they generate
- How much support friction they experience

The result is **5 distinct, business-actionable segments** — each paired with a recommended playbook for product, marketing, and customer success teams.

---

## Key Features

| Capability | Description |
|---|---|
| **Synthetic Data Engine** | Generates 10,000 realistic SaaS users across 5 archetypal personas with correlated behavioral signals |
| **Automated Data Cleaning** | Deduplication, plan-tier-grouped imputation, date validation, and numeric integrity enforcement |
| **EDA Visualization Suite** | Distributions, correlation heatmaps, and plan-tier breakdowns saved as publication-quality PNGs |
| **Behavioural Feature Engineering** | RFM metrics, engagement intensity, usage momentum, feature adoption rates, and support friction indices |
| **Multi-Algorithm Clustering** | K-Means + Gaussian Mixture Models with Elbow, Silhouette, Davies-Bouldin, and Calinski-Harabasz evaluation |
| **PCA Latent Space Projection** | 2D visualization of cluster separation via Principal Component Analysis |
| **Interactive Streamlit Dashboard** | 4-page web app with executive summary, segment deep-dives, cluster evaluation charts, and live user scoring |
| **Real-Time Inference** | Score any new user against the trained pipeline and receive instant segment assignment with business recommendations |

---

## Project Architecture

```
                  ┌──────────────────┐
                  │  Data Generation │
                  │  (10K Synthetic  │
                  │   SaaS Users)    │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Data Cleaning   │
                  │  & Validation    │
                  └────────┬─────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
           ┌──────────────┐  ┌──────────────────────┐
           │     EDA      │  │ Feature Engineering   │
           │  (Figures)   │  │ (RFM, Momentum, etc.) │
           └──────────────┘  └───────────┬───────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  Modeling Pipeline   │
                              │  PowerTransformer →  │
                              │  RobustScaler →      │
                              │  K-Means (k=5)       │
                              │  + GMM + PCA         │
                              └──────────┬──────────┘
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                              ▼                     ▼
                    ┌──────────────────┐  ┌──────────────────┐
                    │  Segmented Data  │  │  Saved Pipeline  │
                    │  (.parquet)      │  │  (.joblib)       │
                    └────────┬─────────┘  └────────┬─────────┘
                             │                     │
                             └──────────┬──────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │    Streamlit      │
                              │    Dashboard      │
                              │  (4 Pages + Live  │
                              │   Inference)      │
                              └──────────────────┘
```

---

## Directory Structure

```text
.
├── app.py                              # Streamlit dashboard (entry point for the web app)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── src/                                # Core pipeline source modules
│   ├── __init__.py
│   ├── data_generator.py               # Synthetic telemetry dataset generator (~10,000 users)
│   ├── data_cleaning.py                # Deduplication, imputation, validation pipeline
│   ├── eda.py                          # Statistical distributions & correlation heatmaps
│   ├── feature_engineering.py          # RFM, engagement intensity, trend & friction metrics
│   ├── modeling.py                     # SegmentationPipeline class (train, evaluate, infer)
│   └── train.py                        # CLI entry point to trigger full training pipeline
│
├── data/
│   ├── raw/                            # Raw ingested user telemetry
│   │   └── user_behaviour_raw.csv      # 10,015 rows (10K users + 15 simulated duplicates)
│   └── processed/                      # Cleaned & feature-engineered datasets
│       ├── cleaned_users.parquet       # Post-cleaning (deduped, imputed, validated)
│       ├── features_users.parquet      # Engineered behavioral features
│       └── segmented_users.parquet     # Final dataset with cluster labels & PCA coords
│
├── models/
│   └── segmentation_pipeline.joblib    # Serialized SegmentationPipeline (transformer + scaler + model)
│
├── notebooks/
│   ├── 01_data_and_feature_pipeline.ipynb   # Interactive walkthrough of stages 1–4
│   └── export_notebook.py              # Script to programmatically generate the notebook
│
└── reports/
    ├── cluster_evaluation_metrics.csv  # Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz for k=2..7
    └── figures/                        # EDA visual outputs
        ├── 01_distributions.png        # Session count, duration, revenue, feature breadth distributions
        ├── 02_correlation_matrix.png   # Lower-triangle correlation heatmap of all numeric features
        └── 03_plan_tier_breakdown.png  # Boxplots of sessions & revenue by plan tier
```

---

## Pipeline Stages

### 1. Data Generation

**Module:** [`src/data_generator.py`](src/data_generator.py)

Generates **10,000 synthetic SaaS users** sampled from 5 latent archetypes with realistic, correlated behavioral signals:

| Archetype | Share | Characteristics |
|---|---|---|
| **Power Users** | ~15% | High session frequency (μ=110), deep feature adoption (7–10 features), Enterprise/Pro plans, high revenue |
| **Regular Core** | ~35% | Consistent mid-tier engagement (μ=55 sessions), 4–7 features, Pro/Starter plans |
| **Casual Browsers** | ~25% | Low sessions (μ=18), short durations (μ=8min), primarily Free tier, mobile-heavy |
| **At-Risk / Churning** | ~15% | High recency (40–90 days dormant), sharp usage decline, elevated support tickets (λ=3.5) |
| **New Onboarding** | ~10% | Recent signups (5–45 days), exploring features, mixed engagement levels |

**Realistic noise injected:**
- ~1.5% missing values in `avg_session_duration_min`
- ~1.0% missing values in `primary_device`
- 15 duplicate rows simulating pipeline re-ingestion

**Raw columns generated:** `user_id`, `signup_date`, `last_active_date`, `session_count_90d`, `avg_session_duration_min`, `features_used_count`, `transactions_90d`, `revenue_90d`, `support_tickets_count`, `plan_type`, `primary_device`, `recent_30d_session_share`, `weekend_session_share`

---

### 2. Data Cleaning & Validation

**Module:** [`src/data_cleaning.py`](src/data_cleaning.py)

| Step | Action |
|---|---|
| **Deduplication** | Remove duplicate rows on `user_id` (keeps first occurrence) |
| **Date Validation** | Ensure `last_active_date ≥ signup_date`; correct inverted dates |
| **Missing Value Imputation** | `avg_session_duration_min` → median by `plan_type` group; `primary_device` → `"Unknown"` |
| **Numeric Integrity** | Clip negative values to 0 for session counts, durations, revenue, tickets |
| **Proportion Bounds** | Enforce `[0, 1]` range for `recent_30d_session_share` and `weekend_session_share` |

Returns a **cleaning report dictionary** with counts for each transformation applied.

---

### 3. Exploratory Data Analysis

**Module:** [`src/eda.py`](src/eda.py)

Generates three publication-quality visualizations saved to `reports/figures/`:

1. **`01_distributions.png`** — Histograms + KDE curves for session counts, session duration, revenue, and feature breadth
2. **`02_correlation_matrix.png`** — Lower-triangle Pearson correlation heatmap across all numeric features
3. **`03_plan_tier_breakdown.png`** — Boxplots comparing sessions and revenue distributions across plan tiers (Free, Starter, Pro, Enterprise)

---

### 4. Behavioural Feature Engineering

**Module:** [`src/feature_engineering.py`](src/feature_engineering.py)

Transforms cleaned raw data into 8 clustering-ready behavioral features:

| Feature | Category | Formula / Logic |
|---|---|---|
| `recency_days` | RFM | Days since last active date relative to reference date |
| `sessions_per_week` | RFM / Frequency | `session_count_90d / min(tenure_days, 90) * 7` |
| `monetary_revenue_90d` | RFM / Monetary | Direct passthrough of 90-day revenue |
| `total_engagement_hours_90d` | Engagement | `(session_count_90d × avg_session_duration_min) / 60` |
| `feature_adoption_rate` | Engagement | `features_used_count / 10` (platform offers 10 core features) |
| `usage_momentum_30d` | Trend | `recent_30d_session_share / 0.333` — ratio >1 = accelerating, <1 = decelerating |
| `support_ticket_friction` | Friction | `support_tickets_count / session_count_90d` |
| `weekend_session_share` | Time Pattern | Proportion of sessions occurring on weekends |

---

### 5. Modeling & Clustering

**Module:** [`src/modeling.py`](src/modeling.py)

The `SegmentationPipeline` class encapsulates the full modeling workflow:

```
Raw Features → Yeo-Johnson PowerTransformer → RobustScaler → K-Means (k=5) / GMM
                                                            → PCA (2D projection)
```

**Why this preprocessing chain?**
- **Yeo-Johnson PowerTransformer** — Normalizes heavily skewed behavioral distributions (revenue, support tickets) to approximate Gaussianity
- **RobustScaler** — Scales features using median/IQR, making the pipeline resistant to outliers
- **K-Means (k=5)** — Primary clustering algorithm; k=5 chosen via elbow method and silhouette analysis
- **Gaussian Mixture Model** — Secondary soft-clustering model fitted in parallel for probabilistic segment assignment

**Cluster Evaluation (k=2 to k=7):**

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|---|
| 2 | 31,578.7 | 0.450 | 0.775 | 6,480.2 |
| 3 | 23,172.8 | 0.340 | 1.074 | 6,228.2 |
| **4** | **14,686.8** | **0.423** | **0.886** | **8,475.8** |
| 5 | 12,933.2 | 0.316 | 1.096 | 7,556.8 |
| 6 | 12,219.9 | 0.295 | 1.393 | 6,514.4 |
| 7 | 11,521.3 | 0.259 | 1.530 | 5,858.3 |

> **Note:** k=5 was selected to balance statistical metrics with business interpretability — 5 segments map cleanly to distinct product/marketing playbooks.

**Segment naming** is automated post-clustering using heuristic rules over cluster-level mean revenue, recency, and momentum.

---

## Discovered Segments

| Segment | Description | Recommended Business Actions |
|---|---|---|
| 🏆 **Power Champions** | High frequency, broad feature adoption & top revenue contributors | Exclusive advisory board, Enterprise upgrades, VIP proactive support |
| 🔵 **Core Active Users** | Steady, reliable mid-tier engagement with consistent daily workflows | In-app feature guides, multi-seat campaigns, mastery webinars |
| 🚀 **Emerging / High-Growth** | Recent signups exhibiting accelerating momentum and fast adoption | High-touch 14-day onboarding, conversion discounts, CS pairing |
| 🟡 **Casual Explorers** | Infrequent sessions, short duration, primarily utilizing basic capabilities | Re-engagement emails, gamified discovery, micro-upgrades |
| 🔴 **At-Risk Dormant** | High recency (dormant), sharp drop in momentum, elevated friction index | Win-back campaigns, priority support escalation, complimentary extensions |

---

## Streamlit Dashboard

The interactive web application ([`app.py`](app.py)) provides four pages:

### Page 1 — Overview & Executive Summary
- Top-level KPI cards (total revenue, avg sessions/week, at-risk %, power champion %)
- Segment summary table with mean metrics per segment
- User distribution bar chart across segments

### Page 2 — Interactive Segment Explorer
- Dropdown segment selector with radar chart comparing the selected segment's normalized behavioral fingerprint against the global average
- Business playbook panel with recommended actions and primary KPIs to track
- Plan type distribution pie chart within the selected segment

### Page 3 — Cluster Evaluation & PCA
- Elbow method (inertia vs. k) and silhouette score line charts for k=2..7
- 2D PCA latent space scatter plot (2,500-user sample) colored by segment assignment

### Page 4 — Live User Segment Scorer
- Interactive form with sliders for all 8 behavioral features
- Real-time inference against the persisted `SegmentationPipeline`
- Instant display of predicted segment name, tagline, recommended action, and target KPI

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Using `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/Prasanna-Nadrajan/usage-behaviour-segmentation-analysis.git
cd usage-behaviour-segmentation-analysis

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Additional dependencies for the Streamlit dashboard
uv pip install streamlit plotly
```

### Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit plotly
```

---

## Usage

### Run the Full Pipeline (Data → Features → Model)

```bash
# Step 1: Generate synthetic data
python -m src.data_generator
# Output: data/raw/user_behaviour_raw.csv

# Step 2: Clean and validate
python -m src.data_cleaning
# Output: data/processed/cleaned_users.parquet

# Step 3: Run exploratory data analysis
python -m src.eda
# Output: reports/figures/*.png

# Step 4: Engineer behavioral features
python -m src.feature_engineering
# Output: data/processed/features_users.parquet

# Step 5: Train clustering model & export pipeline
python -m src.modeling
# Output: data/processed/segmented_users.parquet
#         models/segmentation_pipeline.joblib
#         reports/cluster_evaluation_metrics.csv
```

Or run steps 5 directly via the train script:

```bash
python src/train.py
```

### Launch the Interactive Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

### Run the Jupyter Notebook

```bash
jupyter notebook notebooks/01_data_and_feature_pipeline.ipynb
```

Walks through stages 1–4 interactively with inline outputs and visualizations.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn (K-Means, GMM, PCA, PowerTransformer, RobustScaler) |
| **Visualization** | Matplotlib, Seaborn (EDA), Plotly (Dashboard) |
| **Web Application** | Streamlit |
| **Model Persistence** | Joblib |
| **Data Formats** | CSV (raw), Parquet (processed) |
| **Environment** | uv / venv |

---

## License

This project is open source and available for educational and portfolio purposes.
