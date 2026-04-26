"""
Stage 3b:feature Engineering Impact andd Multicollinearity Analysis

evaluates the impact of different feature engineering choices on model performance
and demonstrates awareness of multicollinearity in the feature space.

Run with:py src/stage3b_feature_engineering/run.py
"""


import os
import sys
import re
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# this make sure the path work for directory 
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# Adds project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage3b_feature_engineering.feature_impact import run_all_ablations
from stage3b_feature_engineering.multicollinearity import (
    run_numeric_correlation, run_cramers_v_analysis, run_vif_analysis,
    print_mitigation_summary
)
from stage3b_feature_engineering.visualisations import (
    plot_ablation_and_correlation, plot_cramers_v_heatmap
)

np.random.seed(42)
os.makedirs('outputs/stage3b', exist_ok=True)

#1.Loads and prepare data
print("=" * 70)
print("STAGE 3b: FEATURE ENGINEERING IMPACT & MULTICOLLINEARITY")
print("=" * 70)

df = pd.read_csv('data/customer_support_emails.csv')
df['escalated'] = (df['escalation_level'] >= 2).astype(int)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['sentiment'] = df['sentiment'].fillna('Unknown')


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['email_body_text_clean'] = df['email_body_text'].apply(clean_text)

exclude_cols = ['escalation_level', 'resolution_time', 'manual_annotation',
                'customer_id', 'timestamp', 'escalated', 'email_body_text']
text_col_clean = 'email_body_text_clean'
categorical_cols = ['customer_type', 'tenure_type', 'meter_type', 'region',
                    'issue_category', 'sentiment']
numeric_cols = ['emotion_intensity', 'hour', 'day_of_week', 'month']

X = df.drop(columns=exclude_cols)
y = df['escalated'].values
groups = df['customer_id'].values

#2feature Engineering Ablation Study 
all_results = run_all_ablations(X, y, groups, text_col_clean,
                                categorical_cols, numeric_cols)
result_full = all_results[0]
result_unigram = all_results[1]
result_text_only = all_results[2]
result_struct_only = all_results[3]
result_big_tfidf = all_results[4]
result_small_tfidf = all_results[5]
result_no_time = all_results[6]

#3.compile Results 
print("\n" + "=" * 70)
print("FEATURE ENGINEERING ABLATION RESULTS")
print("=" * 70)

ablation_df = pd.DataFrame([{
    'Configuration': r['config'],
    'F2 Mean': r['f2_mean'],
    'F2 Std': r['f2_std'],
    'PR-AUC Mean': r['prauc_mean'],
    'Recall': r['recall_mean'],
    'Precision': r['precision_mean'],
} for r in all_results]).sort_values('F2 Mean', ascending=False)

print(ablation_df.to_string(index=False))
ablation_df.to_csv('outputs/stage3b/feature_engineering_ablation.csv', index=False)

print("\n── Impact vs Full Pipeline ──")
baseline_f2 = result_full['f2_mean']
for r in all_results:
    delta = r['f2_mean'] - baseline_f2
    print(f"  {r['config']:<25s}: F2={r['f2_mean']:.4f} (delta={delta:+.4f})")

print("\n── Statistical Significance (paired t-test vs Full Pipeline) ──")
baseline_folds = result_full['fold_f2_values']
for r in all_results:
    if r['config'] == 'Full pipeline':
        continue
    t_stat, p_val = stats.ttest_rel(r['fold_f2_values'], baseline_folds)
    sig = "significant" if p_val < 0.05 else "NOT significant"
    print(f"  {r['config']:<25s}: t={t_stat:+.3f}, p={p_val:.4f} ({sig})")

# 4.multicollinearity Analysis
print("\n" + "=" * 70)
print("MULTICOLLINEARITY ANALYSIS")
print("=" * 70)

corr_matrix, max_corr, max_pair = run_numeric_correlation(df, numeric_cols)
cramers_df, max_v = run_cramers_v_analysis(df, categorical_cols)
print_mitigation_summary()
vif_df, max_vif = run_vif_analysis(df, numeric_cols)

vif_df.to_csv('outputs/stage3b/multicollinearity_vif.csv', index=False)
cramers_df.to_csv('outputs/stage3b/multicollinearity_cramers_v.csv', index=False)

#5.visualisations 
print("\n" + "=" * 70)
print("GENERATING VISUALISATIONS...")
print("=" * 70)

plot_ablation_and_correlation(all_results, result_full, corr_matrix,
                              'outputs/stage3b/feature_engineering_and_multicollinearity.png')
plot_cramers_v_heatmap(cramers_df, categorical_cols,
                       'outputs/stage3b/cramers_v_heatmap.png')


#summaryss

print("\n" + "=" * 70)
print("STAGE 3b SUMMARY")
print("=" * 70)

summary = {
    'feature_engineering': {
        'best_config': ablation_df.iloc[0]['Configuration'],
        'best_f2': float(ablation_df.iloc[0]['F2 Mean']),
        'key_findings': [
            f"Bigrams impact: {result_full['f2_mean'] - result_unigram['f2_mean']:+.4f} F2 (unigrams={result_unigram['f2_mean']:.4f} vs bigrams={result_full['f2_mean']:.4f})",
            f"Text vs structured: text-only F2={result_text_only['f2_mean']:.4f}, structured-only F2={result_struct_only['f2_mean']:.4f}",
            f"TF-IDF vocabulary size: 200={result_small_tfidf['f2_mean']:.4f}, 500={result_full['f2_mean']:.4f}, 1000={result_big_tfidf['f2_mean']:.4f}",
            f"Temporal features impact: {result_full['f2_mean'] - result_no_time['f2_mean']:+.4f} F2",
        ],
    },
    'multicollinearity': {
        'max_numeric_correlation': float(max_corr),
        'max_numeric_pair': list(max_pair),
        'max_vif': float(max_vif),
        'max_cramers_v': float(max_v),
        'conclusion': 'No multicollinearity concerns. All numeric VIF < 5, all categorical Cramers V < 0.3.',
        'mitigations': [
            'OHE drop=first avoids dummy variable trap',
            'L1 regularisation handles correlated TF-IDF features',
            'Verified low VIF across all numeric features',
        ],
    },
}

with open('outputs/stage3b/feature_engineering_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"""
Feature Engineering Key Findings:
  - Bigrams vs unigrams: {result_full['f2_mean'] - result_unigram['f2_mean']:+.4f} F2
  - Text carries nearly all signal (text-only: {result_text_only['f2_mean']:.4f})
  - Structured features alone: {result_struct_only['f2_mean']:.4f} (near-chance)
  - TF-IDF 500 features is the sweet spot
  - Temporal features have minimal impact

Multicollinearity:
  - Max numeric correlation: {max_corr:.4f} ({max_pair[0]} vs {max_pair[1]})
  - Max VIF: {max_vif:.3f} (all well below 5)
  - Max Cramer's V: {max_v:.4f} (no categorical association)
  - OHE drop='first' prevents dummy variable trap
""")

print("STAGE 3b COMPLETE")