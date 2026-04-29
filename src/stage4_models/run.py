"""
Stage 4: Model Development & Comparison
this trains LR, RF, XGBoost, and SMOTE variants. Performs hyperparameter
tuning, threshold optimisation, and statistical comparison.

Run with:py src/stage4_models/run.py
"""


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

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))
comp_df.to_csv('outputs/stage4/model_comparison.csv', index=False)

#visualisations
print("\n" + "="*70)
print("GENERATING VISUALISATIONS...")
print("="*70)

model_names = list(all_results.keys()) + ['XGB_tuned']
f2_means = [all_results[m]['mean_f2_tuned'] for m in all_results] + [best_df['f2'].mean()]
f2_stds = [all_results[m]['std_f2_tuned'] for m in all_results] + [best_df['f2'].std()]
prauc_means = [all_results[m]['mean_pr_auc'] for m in all_results] + [best_df['pr_auc'].mean()]
prauc_stds = [all_results[m]['std_pr_auc'] for m in all_results] + [best_df['pr_auc'].std()]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].bar(model_names, f2_means, yerr=f2_stds, color=colors, capsize=5, alpha=0.85)
axes[0].set_title('F2-Score Comparison (Tuned Threshold)', fontweight='bold')
axes[0].set_ylabel('F2-Score')
axes[0].tick_params(axis='x', rotation=30)
axes[1].bar(model_names, prauc_means, yerr=prauc_stds, color=colors, capsize=5, alpha=0.85)
axes[1].set_title('PR-AUC Comparison', fontweight='bold')
axes[1].set_ylabel('PR-AUC')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('outputs/stage4/model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/stage4/model_comparison.png")

fig2, ax2 = plt.subplots(figsize=(10, 6))
fold_data = {name: res['metrics_df']['f2_tuned'].values for name, res in all_results.items()}
fold_data['XGB_tuned'] = best_df['f2'].values
ax2.boxplot(fold_data.values(), labels=fold_data.keys(), patch_artist=True,
            boxprops=dict(facecolor='#E3F2FD'))
ax2.set_title('F2-Score Distribution Across Folds (Tuned Threshold)', fontweight='bold')
ax2.set_ylabel('F2-Score')
ax2.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig('outputs/stage4/fold_variation.png', dpi=150, bbox_inches='tight')
print("Saved: outputs/stage4/fold_variation.png")

#statistical Compariso
print("\n" + "="*70)
print("STATISTICAL COMPARISON (Paired Fold Differences)")
print("="*70)

baseline_f2 = all_results['LR_baseline']['metrics_df']['f2_tuned'].values
for name in ['RF', 'XGB', 'LR_SMOTE', 'XGB_SMOTE']:
    model_f2 = all_results[name]['metrics_df']['f2_tuned'].values
    diff = model_f2 - baseline_f2
    t_stat, p_val = stats.ttest_rel(model_f2, baseline_f2)
    print(f"  {name} vs LR_baseline: mean diff={diff.mean():+.4f}, "
          f"t={t_stat:.3f}, p={p_val:.4f}")

xgb_tuned_f2 = best_df['f2'].values
diff = xgb_tuned_f2 - baseline_f2
t_stat, p_val = stats.ttest_rel(xgb_tuned_f2, baseline_f2)
print(f"  XGB_tuned vs LR_baseline: mean diff={diff.mean():+.4f}, "
      f"t={t_stat:.3f}, p={p_val:.4f}")

#model Selection
print("\n" + "="*70)
print("MODEL SELECTION")
print("="*70)

all_f2_tuned = {name: res['mean_f2_tuned'] for name, res in all_results.items()}
all_f2_tuned['XGB_tuned'] = best_df['f2'].mean()
best_model_name = max(all_f2_tuned, key=all_f2_tuned.get)

print(f"\nBest model: {best_model_name} (F2={all_f2_tuned[best_model_name]:.4f})")
print(f"\nAll models ranked by F2 (tuned threshold):")
for name, f2 in sorted(all_f2_tuned.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {f2:.4f}")

selection = {
    'best_model': best_model_name,
    'best_f2': float(all_f2_tuned[best_model_name]),
    'best_params': best_params if 'XGB' in best_model_name else None,
    'all_rankings': {k: float(v) for k, v in sorted(all_f2_tuned.items(),
                     key=lambda x: x[1], reverse=True)}
}
with open('outputs/stage4/model_selection.json', 'w') as f:
    json.dump(selection, f, indent=2)
print("\nSaved: outputs/stage4/model_selection.json")

print("\n" + "="*70)
print("STAGE 4 COMPLETE")
print("="*70)
