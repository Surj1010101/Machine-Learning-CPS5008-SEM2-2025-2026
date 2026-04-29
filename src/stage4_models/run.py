


import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.data_loader import load_and_prepare_data

os.makedirs('outputs/stage4', exist_ok=True)

print("="*70)
print("STAGE 4: MODEL DEVELOPMENT & COMPARISON")
print("="*70)

df, X, y, groups = load_and_prepare_data()

#Imbalance ratio
scale_pos = (y == 0).sum() / (y == 1).sum()
print(f"Imbalance ratio for scale_pos_weight: {scale_pos:.2f}")

#Define and cross validate all models
from stage4_models.pipelines import define_models
from stage4_models.cross_validation import cross_validate_all
from stage4_models.tuning import tune_xgboost, evaluate_tuned_xgb

models = define_models(scale_pos)
all_results = cross_validate_all(models, X, y, groups)

#hyperparameter tuning
best_params, _ = tune_xgboost(X, y, groups, scale_pos)
best_df = evaluate_tuned_xgb(X, y, groups, best_params, scale_pos)

#Summary Comparison Table
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

comparison = []
for name, res in all_results.items():
    comparison.append({
        'Model': name,
        'F2 (default)': f"{res['mean_f2_default']:.4f} (+/-{res['std_f2_default']:.4f})",
        'F2 (tuned)': f"{res['mean_f2_tuned']:.4f} (+/-{res['std_f2_tuned']:.4f})",
        'PR-AUC': f"{res['mean_pr_auc']:.4f} (+/-{res['std_pr_auc']:.4f})",
        'Precision': f"{res['mean_precision']:.4f}",
        'Recall': f"{res['mean_recall']:.4f}",
        'Threshold': f"{res['mean_threshold']:.3f}",
    })
comparison.append({
    'Model': 'XGB_tuned', 'F2 (default)': '-',
    'F2 (tuned)': f"{best_df['f2'].mean():.4f} (+/-{best_df['f2'].std():.4f})",
    'PR-AUC': f"{best_df['pr_auc'].mean():.4f} (+/-{best_df['pr_auc'].std():.4f})",
    'Precision': f"{best_df['precision'].mean():.4f}",
    'Recall': f"{best_df['recall'].mean():.4f}",
    'Threshold': f"{best_df['threshold'].mean():.3f}",
})