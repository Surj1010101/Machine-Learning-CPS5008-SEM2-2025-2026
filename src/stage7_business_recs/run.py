"""
Stage 7: Business Recommendations & Deployment Strategy
CPS5008 Machine Learning Assessment

Analyses tiered threshold deployment, monitoring requirements,
limitations, and future work. Produces final summary outputs.

Run with: py src/stage7_business_recs/run.py
"""

import os
import sys
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage7_business_recs.predictions import collect_predictions
from stage7_business_recs.threshold_analysis import (
    run_threshold_analysis, run_tiered_deployment
)
from stage7_business_recs.temporal import run_temporal_stability
from stage7_business_recs.cost_sensitivity import run_cost_sensitivity
from stage7_business_recs.monitoring import build_monitoring_plan
from stage7_business_recs.visualisations import (
    plot_threshold_tradeoffs, plot_tiered_deployment, plot_stability_and_costs
)

np.random.seed(42)
os.makedirs('outputs/stage7', exist_ok=True)

# 1. Load and prepare data 
print("=" * 70)
print("STAGE 7: BUSINESS RECOMMENDATIONS & DEPLOYMENT")
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

#2. Collect predictions across all folds 
mean_thresh = collect_predictions(
    df, X, y, groups, text_col_clean, categorical_cols, numeric_cols
)

#  3. Threshold analysis and tiered deployment 
thresh_df, best_f2_row, best_cost_row, workload_30 = run_threshold_analysis(df, mean_thresh)
thresh_df.to_csv('outputs/stage7/threshold_analysis.csv', index=False)

tier_high = 0.65
tier_df, high_mask, medium_mask, low_mask, high_catch, med_catch, low_miss, total_pos = \
    run_tiered_deployment(df, mean_thresh, tier_high=tier_high)
tier_df.to_csv('outputs/stage7/tier_distribution.csv', index=False)

#  4. Temporal stability ─
temp_df, f2_std = run_temporal_stability(df)
temp_df.to_csv('outputs/stage7/temporal_stability.csv', index=False)

#5. Cost sensitivity 
cost_sens_df, breakeven, tp_all, fp_all, fn_all = run_cost_sensitivity(df, thresh_df)
cost_sens_df.to_csv('outputs/stage7/cost_sensitivity.csv', index=False)

# 6. Monitoring plan 
y_probs = df['y_prob'].values
monitoring_plan = build_monitoring_plan(df, y_probs)
with open('outputs/stage7/monitoring_plan.json', 'w') as f:
    json.dump(monitoring_plan, f, indent=2)
print("\nSaved: outputs/stage7/monitoring_plan.json")

# 7. Visualisations 
print("\n" + "=" * 70)
print("GENERATING VISUALISATIONS...")
print("=" * 70)

plot_threshold_tradeoffs(thresh_df, mean_thresh, best_cost_row,
                          'outputs/stage7/threshold_tradeoffs.png')
plot_tiered_deployment(high_mask, medium_mask, low_mask,
                       high_catch, med_catch, low_miss, y,
                       'outputs/stage7/tiered_deployment.png')
plot_stability_and_costs(temp_df, f2_std, cost_sens_df, breakeven,
                          'outputs/stage7/stability_and_costs.png')

# 8. Final Summary JSON 
print("\n" + "=" * 70)
print("SAVING FINAL SUMMARY...")
print("=" * 70)

y_true = df['escalated'].values

final_summary = {
    'model': 'Logistic Regression (class_weight=balanced, liblinear)',
    'features': '500 TF-IDF + 14 OHE categorical + 4 numeric = 518 total',
    'validation': 'StratifiedGroupKFold k=5 (grouped by customer_id)',
    'threshold': float(mean_thresh),
    'performance': {
        'f2': float(fbeta_score(y_true, df['y_pred'].values, beta=2)),
        'recall': float(y_true[df['y_pred'].values == 1].sum() / y_true.sum()),
        'precision': float(y_true[df['y_pred'].values == 1].sum() / df['y_pred'].sum()),
    },
    'tiered_deployment': {
        'high_threshold': tier_high,
        'medium_threshold': float(mean_thresh),
        'high_pct': f"{high_mask.mean()*100:.1f}%",
        'medium_pct': f"{medium_mask.mean()*100:.1f}%",
        'low_pct': f"{low_mask.mean()*100:.1f}%",
        'combined_catch_rate': f"{(high_catch+med_catch)/total_pos*100:.1f}%"
    },
    'cost_analysis': {
        'fn_cost': 500, 'fp_cost': 20, 'tp_cost': 50,
        'model_cost': tp_all * 50 + fp_all * 20 + fn_all * 500,
        'breakeven_fn_cost': int(breakeven) if breakeven else None,
    },
    'temporal_stability': {
        'f2_std_across_quarters': float(f2_std),
        'stable': bool(f2_std < 0.05)
    },
    'fairness': {
        'region_tpr_gap': 0.032,
        'customer_type_tpr_gap': 0.087,
        'all_pass_80pct_rule': True
    },
    'limitations': [
        'Dataset likely synthetic -- all findings may not generalise',
        'Very short emails (7-8 words) limit discriminative power',
        'Model flags 54% of emails -- high workload in practice',
        'Poor calibration -- probabilities are not meaningful as confidence scores',
        'No customer history features (prior complaints, account age)'
    ],
    'future_work': [
        'Incorporate customer complaint history as features',
        'Use sentence transformers if longer emails available',
        'A/B test tiered deployment vs current process',
        'Investigate Commercial customer recall gap',
        'Explore temporal features (time since last email, complaint velocity)'
    ]
}

with open('outputs/stage7/final_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)
print("Saved: outputs/stage7/final_summary.json")

#9. Summary 
print("\n" + "=" * 70)
print("STAGE 7 SUMMARY")
print("=" * 70)

print(f"""
Deployment Recommendation:
  Three-tier system based on predicted probability:
    HIGH   (p >= {tier_high:.2f}): {high_mask.sum()} emails ({high_mask.mean()*100:.1f}%), {int(high_catch)} escalations caught
    MEDIUM ({mean_thresh:.2f}-{tier_high:.2f}): {medium_mask.sum()} emails ({medium_mask.mean()*100:.1f}%), {int(med_catch)} escalations caught
    LOW    (p < {mean_thresh:.2f}): {low_mask.sum()} emails ({low_mask.mean()*100:.1f}%), {int(low_miss)} escalations missed
  Combined HIGH+MEDIUM catch rate: {(high_catch+med_catch)/total_pos*100:.1f}%

Cost Analysis:
  Under current assumptions (FN=£500, FP=£20):
    Model saves £{int(y_true.sum()) * 500 - (tp_all*50 + fp_all*20 + fn_all*500):,} vs no-model
    Model breakeven vs flag-all at FN cost ~£{breakeven if breakeven else 'N/A'}

Temporal Stability:
  F2 std across quarters: {f2_std:.3f} -- {'STABLE' if f2_std < 0.05 else 'VARIABLE'}

Monitoring: 3 real-time + 3 weekly + 3 monthly metrics defined
Retraining: 5 trigger conditions defined

Outputs:
  - outputs/stage7/threshold_tradeoffs.png
  - outputs/stage7/tiered_deployment.png
  - outputs/stage7/stability_and_costs.png
  - outputs/stage7/threshold_analysis.csv
  - outputs/stage7/tier_distribution.csv
  - outputs/stage7/temporal_stability.csv
  - outputs/stage7/cost_sensitivity.csv
  - outputs/stage7/monitoring_plan.json
  - outputs/stage7/final_summary.json
""")

print("STAGE 7 COMPLETE")
print("\n" + "=" * 70)
print("ALL ANALYSIS STAGES COMPLETE (Stages 0-7)")
print("Stage 8 (Report Support) available on request.")
print("=" * 70)
