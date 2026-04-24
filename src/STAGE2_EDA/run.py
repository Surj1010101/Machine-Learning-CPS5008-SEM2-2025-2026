"""
Stage 2:Data Analysis and risk Identification
Run with: py src/stage2_eda/run.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

#Added project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.makedirs('outputs/stage2', exist_ok=True)

print("="*70)
print("STAGE 2: EXPLORATORY DATA ANALYSIS AND RISK IDENTIFICATION")
print("="*70)

#Loads raw data (not the modelling ready version we still need all columns for EDA)
df = pd.read_csv('data/customer_support_emails.csv')
df['escalated'] = (df['escalation_level'] >= 2).astype(int)

print(f"\nDataset shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
print(f"\n── Binary Target Distribution ──")
print(f"Escalated (1): {df['escalated'].sum()} ({df['escalated'].mean()*100:.1f}%)")
print(f"Not escalated (0): {(df['escalated']==0).sum()} ({(df['escalated']==0).mean()*100:.1f}%)")

#runss all analysis modules
from stage2_eda.profiling import run_feature_profiling, run_text_analysis, run_timestamp_analysis
from stage2_eda.leakage import (run_missing_data_analysis, run_leakage_testing,
                                 run_imbalance_analysis, run_customer_overlap_analysis)
from stage2_eda.visualisations import generate_visualisations

run_feature_profiling(df)
df = run_text_analysis(df)
df = run_timestamp_analysis(df)
run_missing_data_analysis(df)
run_leakage_testing(df)
run_imbalance_analysis(df)
run_customer_overlap_analysis(df)

#Numeric correlations
print("\n" + "="*70)
print("NUMERIC CORRELATIONS")
print("="*70)
numeric_cols = ['emotion_intensity', 'text_len', 'word_count', 'escalated']
corr = df[numeric_cols].corr()
print(corr.round(4))

#generates plots
generate_visualisations(df, corr)

print("\n" + "="*70)
print("STAGE 2 EDA COMPLETE")
print("="*70)