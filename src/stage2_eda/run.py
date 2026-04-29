"""
Stage 2: Exploratory Data Analysis and Risk Identification

This script is the entry point for my Stage 2 work. Here I load the customer support
emails dataset and run all my analysis modules in order. The four things I focused
on are profiling the features, testing for leakage, checking missing data, and
showcasing the customer overlap so I know my GroupKFold will work in Stage 3. The
output is the full picture of data quality risks I identified before any modelling
started.

Run with: python src/stage2_eda/run.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Added project root to path so my stage2_eda imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.makedirs('outputs/stage2', exist_ok=True)

print("="*70)
print("STAGE 2: EXPLORATORY DATA ANALYSIS AND RISK IDENTIFICATION")
print("="*70)

# Loads the raw data, not the modelling ready version, because for EDA I still need all columns
df = pd.read_csv('data/customer_support_emails.csv')
df['escalated'] = (df['escalation_level'] >= 2).astype(int)

print(f"\nDataset shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
print(f"\n── Binary Target Distribution ──")
print(f"Escalated (1): {df['escalated'].sum()} ({df['escalated'].mean()*100:.1f}%)")
print(f"Not escalated (0): {(df['escalated']==0).sum()} ({(df['escalated']==0).mean()*100:.1f}%)")

# Imports for all my analysis modules
from stage2_eda.profiling import run_feature_profiling, run_text_analysis, run_timestamp_analysis
from stage2_eda.leakage import (run_missing_data_analysis, run_leakage_testing,
                                 run_imbalance_analysis, run_customer_overlap_analysis)
from stage2_eda.visualisations import generate_visualisations

# Now I will be running through each analysis step in order
run_feature_profiling(df)
df = run_text_analysis(df)
df = run_timestamp_analysis(df)
run_missing_data_analysis(df)
run_leakage_testing(df)
run_imbalance_analysis(df)
run_customer_overlap_analysis(df)

# Numeric correlations on the key continuous features
print("\n" + "="*70)
print("NUMERIC CORRELATIONS")
print("="*70)
numeric_cols = ['emotion_intensity', 'text_len', 'word_count', 'escalated']
corr = df[numeric_cols].corr()
print(corr.round(4))

# Generating my plots into outputs/stage2
generate_visualisations(df, corr)

print("\n" + "="*70)
print("STAGE 2 EDA COMPLETE")
print("="*70)
