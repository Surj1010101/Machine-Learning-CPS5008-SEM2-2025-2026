"""
Stage 5: Error analysis and Segment Breakdown
Analyses misclassifications from the selected model (LR with tuned threshold),
breaks down performance by segment, and frames business impact.

Run with: py src/stage5_error_analysis/run.py
"""

import os
import sys
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, confusion_matrix

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage5_error_analysis.predictions import collect_fold_predictions
from stage5_error_analysis.error_breakdown import (
    analyse_false_negatives, analyse_false_positives
)
from stage5_error_analysis.segments import run_segment_analysis, run_fairness_analysis
from stage5_error_analysis.business import run_business_impact, run_calibration
from stage5_error_analysis.visualisations import (
    plot_error_overview, plot_calibration_and_cost, plot_fn_deep_dive
)

np.random.seed(42)
os.makedirs('outputs/stage5', exist_ok=True)

#1. Load and prepare data 
print("=" * 70)
print("STAGE 5: ERROR ANALYSIS & SEGMENT BREAKDOWN")
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

text_col_clean = 'email_body_text_clean'
categorical_cols = ['customer_type', 'tenure_type', 'meter_type', 'region',
                    'issue_category', 'sentiment']
numeric_cols = ['emotion_intensity', 'hour', 'day_of_week', 'month']
exclude_cols = ['escalation_level', 'resolution_time', 'manual_annotation',
                'customer_id', 'timestamp', 'escalated', 'email_body_text']

X = df.drop(columns=exclude_cols)
y = df['escalated'].values
groups = df['customer_id'].values

#2. Collect per-sample predictions across all folds 
fold_metrics = collect_fold_predictions(
    df, X, y, groups, text_col_clean, categorical_cols, numeric_cols
)

print(f"\nOverall prediction breakdown:")
print(df['prediction_type'].value_counts().to_string())

overall_f2 = fbeta_score(df['escalated'], df['y_pred'], beta=2)
overall_cm = confusion_matrix(df['escalated'], df['y_pred'])
tn_all, fp_all, fn_all, tp_all = overall_cm.ravel()
print(f"\nAggregated: F2={overall_f2:.4f}")
print(f"  TP={tp_all}, FP={fp_all}, FN={fn_all}, TN={tn_all}")
print(f"  Recall={tp_all/(tp_all+fn_all):.3f}, Precision={tp_all/(tp_all+fp_all):.3f}")

# 3. Error analysis
fn_df, tp_df, mean_thresh, close_fn = analyse_false_negatives(df, fold_metrics)
fp_df, tn_df = analyse_false_positives(df)

# ─4.Segment-level performance 
seg_df = run_segment_analysis(df)
seg_df.to_csv('outputs/stage5/segment_performance.csv', index=False)
print(f"\nSaved: outputs/stage5/segment_performance.csv")

run_fairness_analysis(seg_df)

#5. Business impact and calibration 
cost_data = run_business_impact(df, tp_all, fp_all, fn_all, tn_all)
cal_table = run_calibration(df)

#6. Visualisations
print("\n" + "=" * 70)
print("GENERATING VISUALISATIONS...")
print("=" * 70)

plot_error_overview(df, seg_df, fn_all, fp_all, tn_all, tp_all,
                    mean_thresh, overall_f2,
                    'outputs/stage5/error_analysis_overview.png')
plot_calibration_and_cost(cal_table, cost_data,
                          'outputs/stage5/calibration_and_cost.png')
miss_df = plot_fn_deep_dive(df, fn_df, tp_df, mean_thresh,
                            'outputs/stage5/fn_deep_dive.png')

#7.Save detailed error analysis data 
print("\n" + "=" * 70)
print("SAVING DETAILED DATA...")
print("=" * 70)

fn_export = fn_df[['customer_id', 'email_body_text', 'escalation_level',
                   'customer_type', 'tenure_type', 'region', 'issue_category',
                   'sentiment', 'y_prob', 'fold']].copy()
fn_export.to_csv('outputs/stage5/false_negatives.csv', index=False)
print(f"Saved: outputs/stage5/false_negatives.csv ({len(fn_export)} rows)")

fp_export = fp_df[['customer_id', 'email_body_text', 'escalation_level',
                   'customer_type', 'tenure_type', 'region', 'issue_category',
                   'sentiment', 'y_prob', 'fold']].copy()
fp_export.to_csv('outputs/stage5/false_positives.csv', index=False)
print(f"Saved: outputs/stage5/false_positives.csv ({len(fp_export)} rows)")

summary = {
    'overall_f2': float(overall_f2),
    'overall_recall': float(tp_all / (tp_all + fn_all)),
    'overall_precision': float(tp_all / (tp_all + fp_all)),
    'confusion_matrix': {'TP': int(tp_all), 'FP': int(fp_all),
                         'FN': int(fn_all), 'TN': int(tn_all)},
    'fn_analysis': {
        'total_fn': int(fn_all),
        'fn_mean_prob': float(fn_df['y_prob'].mean()),
        'fn_close_to_threshold': int(len(close_fn)),
        'miss_rate_level2': float(miss_df[miss_df['level'] == 'Level 2']['miss_rate'].iloc[0]),
        'miss_rate_level3': float(miss_df[miss_df['level'] == 'Level 3']['miss_rate'].iloc[0]),
    },
    'business_cost': {
        'no_model': int(cost_data['no_model_cost']),
        'flag_all': int(cost_data['flag_all_cost']),
        'current_model': int(cost_data['model_cost']),
        'perfect_model': int(cost_data['perfect_cost']),
        'savings_vs_no_model': int(cost_data['no_model_cost'] - cost_data['model_cost']),
    },
    'emails_flagged': int(cost_data['emails_reviewed']),
    'flagged_pct': float(cost_data['emails_reviewed'] / len(df) * 100),
}

with open('outputs/stage5/error_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved: outputs/stage5/error_analysis_summary.json")

#8.Summary
print("\n" + "=" * 70)
print("STAGE 5 SUMMARY")
print("=" * 70)

print(f"""
Selected Model: LR (class_weight='balanced') + tuned threshold
Overall: F2={overall_f2:.4f}, Recall={tp_all/(tp_all+fn_all):.3f}, Precision={tp_all/(tp_all+fp_all):.3f}

Error Breakdown:
  True Positives:  {tp_all:4d} ({tp_all/len(df)*100:.1f}%)
  False Positives: {fp_all:4d} ({fp_all/len(df)*100:.1f}%)
  False Negatives: {fn_all:4d} ({fn_all/len(df)*100:.1f}%)
  True Negatives:  {tn_all:4d} ({tn_all/len(df)*100:.1f}%)

Key Findings:
  - {fn_all} escalations missed ({fn_all/(tp_all+fn_all)*100:.1f}% miss rate)
  - {fp_all} false alarms ({fp_all/(fp_all+tn_all)*100:.1f}% false alarm rate)
  - {len(close_fn)} FNs were close to the threshold (within 0.05)
  - Model saves £{cost_data['no_model_cost'] - cost_data['model_cost']:,.0f} vs no-model scenario

Outputs:
  - outputs/stage5/error_analysis_overview.png
  - outputs/stage5/calibration_and_cost.png
  - outputs/stage5/fn_deep_dive.png
  - outputs/stage5/segment_performance.csv
  - outputs/stage5/false_negatives.csv
  - outputs/stage5/false_positives.csv
  - outputs/stage5/error_analysis_summary.json
""")

print("STAGE 5 COMPLETE")
