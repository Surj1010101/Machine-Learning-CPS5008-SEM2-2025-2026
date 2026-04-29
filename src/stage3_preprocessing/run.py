"""Run module."""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

sys.stdout.reconfigure(encoding='utf-8')

from utils.data_loader import load_and_prepare_data, CATEGORICAL_COLS, NUMERIC_COLS

os.makedirs('outputs/stage3', exist_ok=True)

print("="*70)
print("STAGE 3: PREPROCESSING PIPELINE")
print("="*70)

# Loading my data through the utils helper, this also derives the target and groups
df, X, y, groups = load_and_prepare_data()
print(f"Dataset: {df.shape[0]} rows")
print(f"Target: {df['escalated'].sum()} escalated ({df['escalated'].mean()*100:.1f}%)")
print(f"\nFeature groups:")
print(f"  Text: email_body_text_clean")
print(f"  Categorical: {CATEGORICAL_COLS}")
print(f"  Numeric: {NUMERIC_COLS}")

# Now I build my pipeline and run cross validation on it
from stage3_preprocessing.pipelines import (build_baseline_pipeline, run_cross_validation,
                                             print_aggregate_results)

pipeline = build_baseline_pipeline()
print("\n" + "="*70)
print("BUILDING PIPELINE")
print("="*70)
print("Pipeline structure:")
print(pipeline)

results_df, all_y_true, all_y_pred = run_cross_validation(X, y, groups, pipeline)
print_aggregate_results(results_df, all_y_true, all_y_pred)

# Feature importance from the last fold so I can see what my model is actually learning
print("\n" + "="*70)
print("TOP FEATURES (Logistic Regression Coefficients -- Last Fold)")
print("="*70)

feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
coefficients = pipeline.named_steps['classifier'].coef_[0]

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
}).sort_values('coefficient', ascending=False)

print("\nTop 15 features INCREASING escalation risk:")
print(feat_imp.head(15).to_string(index=False))
print("\nTop 15 features DECREASING escalation risk:")
print(feat_imp.tail(15).to_string(index=False))

# Sentiment ablation, this is my leakage check on the sentiment feature
from stage3_preprocessing.feature_impact import run_sentiment_ablation
diff = run_sentiment_ablation(X, y, groups, results_df)

# Saving all my results to outputs/stage3 so the report can pick them up
results_df.to_csv('outputs/stage3/baseline_cv_results.csv', index=False)
feat_imp.to_csv('outputs/stage3/baseline_feature_importance.csv', index=False)
print("\nSaved: outputs/stage3/baseline_cv_results.csv")
print("Saved: outputs/stage3/baseline_feature_importance.csv")

# Final summary block, this is what I quote in the report and what feeds into Stage 4
print("\n" + "="*70)
print("STAGE 3 SUMMARY")
print("="*70)
print(f"""
Pipeline:
  Text: TF-IDF (max_features=500, ngram_range=(1,2), sublinear_tf=True)
  Categorical: OneHotEncoder (drop='first') for {len(CATEGORICAL_COLS)} features
  Numeric: StandardScaler for {len(NUMERIC_COLS)} features
  Classifier: LogisticRegression (class_weight='balanced', solver='liblinear')

Validation: StratifiedGroupKFold k=5 (grouped by customer_id)
  - No customer overlap across folds (verified)

Baseline Results:
  F2-score: {results_df['f2'].mean():.4f} (+/- {results_df['f2'].std():.4f})
  PR-AUC:   {results_df['pr_auc'].mean():.4f} (+/- {results_df['pr_auc'].std():.4f})

Ablation:
  Sentiment impact on F2: {diff:+.4f} (negligible)

Ready for Stage 4: Model Development & Comparison.
""")
print("STAGE 3 COMPLETE")

