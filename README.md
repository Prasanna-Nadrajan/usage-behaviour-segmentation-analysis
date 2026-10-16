# End-to-End Usage Behaviour Segmentation

An end-to-end Machine Learning segmentation project in Python that segments SaaS users based on behavioral and product interaction patterns.

## Project Structure
```text
.
├── data/
│   ├── raw/                 # Raw ingested user telemetry
│   └── processed/           # Cleaned & feature-engineered datasets
├── notebooks/
│   └── 01_data_and_feature_pipeline.ipynb
├── reports/
│   └── figures/             # EDA visual distributions, correlations, etc.
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # Synthetic telemetry dataset generator (~10,000 users)
│   ├── data_cleaning.py     # Deduplication, imputation, validation pipeline
│   ├── eda.py               # Statistical distributions & correlation heatmaps
│   └── feature_engineering.py# RFM, engagement intensity, trend & friction metrics
├── requirements.txt
└── README.md
```

## Setup & Environment
Managed using `uv`:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
