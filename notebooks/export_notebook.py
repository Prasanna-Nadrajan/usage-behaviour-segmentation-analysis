import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SaaS Usage Behaviour Segmentation\n",
    "## Phase 1: Data Generation, Cleaning, EDA & Feature Engineering\n",
    "\n",
    "This notebook demonstrates the end-to-end data pipeline from raw ingestion to behavioral feature engineering."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from src.data_generator import generate_synthetic_data\n",
    "from src.data_cleaning import clean_usage_data\n",
    "from src.eda import run_eda\n",
    "from src.feature_engineering import engineer_features\n",
    "\n",
    "%matplotlib inline\n",
    "sns.set_theme(style='whitegrid')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Data Generation & Ingestion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "raw_df = generate_synthetic_data(n_users=10000, random_seed=42)\n",
    "print(f\"Raw dataset dimensions: {raw_df.shape}\")\n",
    "raw_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Data Cleaning & Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df, report = clean_usage_data(raw_df)\n",
    "print(\"Cleaning Report:\", report)\n",
    "cleaned_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "eda_res = run_eda(cleaned_df, output_dir='../reports/figures')\n",
    "print(\"EDA completed and figures saved in ../reports/figures/\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Behavioural Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "features_df = engineer_features(cleaned_df)\n",
    "features_df[['user_id', 'recency_days', 'sessions_per_week', 'total_engagement_hours_90d', 'feature_adoption_rate', 'usage_momentum_30d', 'support_ticket_friction']].head(10)"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("notebooks/01_data_and_feature_pipeline.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")
