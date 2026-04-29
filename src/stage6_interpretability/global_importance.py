"""Global feature importanceLR coefficients and permutation importance."""

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.inspection import permutation_importance


def run_lr_coefficient_importance(pipeline):
    """Extract and rank LR coefficients by absolute value."""
    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE: LR COEFFICIENTS")
    print("=" * 70)

    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    coefficients = pipeline.named_steps['classifier'].coef_[0]

    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)

    feat_imp['clean_name'] = (feat_imp['feature']
                              .str.replace('text__', '')
                              .str.replace('cat__', '')
                              .str.replace('num__', ''))

    print("\nTop 20 features by absolute coefficient:")
    for _, row in feat_imp.head(20).iterrows():
        direction = "+" if row['coefficient'] > 0 else "-"
        print(f"  {direction} {row['clean_name']:<35s} coef={row['coefficient']:+.4f}")

    top_escalation = feat_imp[feat_imp['coefficient'] > 0].head(15)
    top_deescalation = feat_imp[feat_imp['coefficient'] < 0].head(15)

    print(f"\nTop 15 ESCALATION indicators (positive coefficients):")
    for _, row in top_escalation.iterrows():
        print(f"  {row['clean_name']:<35s} {row['coefficient']:+.4f}")

    print(f"\nTop 15 DE-ESCALATION indicators (negative coefficients):")
    for _, row in top_deescalation.iterrows():
        print(f"  {row['clean_name']:<35s} {row['coefficient']:+.4f}")

    return feat_imp, top_escalation, top_deescalation, feature_names


def run_permutation_importance(pipeline, X_val, y_val, best_thresh):
    """Pvalidation set using F2 as scorer."""
    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE: PERMUTATION IMPORTANCE")
    print("=" * 70)

    def f2_scorer(estimator, X, y):
        y_prob = estimator.predict_proba(X)[:, 1]
        y_pred = (y_prob >= best_thresh).astype(int)
        return fbeta_score(y, y_pred, beta=2)

    print("Computing permutation importance on validation set (10 repeats)...")
    print("(This may take a minute...)")

    perm_result = permutation_importance(
        pipeline, X_val, y_val, scoring=f2_scorer,
        n_repeats=10, random_state=42, n_jobs=-1
    )

    input_features = X_val.columns.tolist()
    perm_df = pd.DataFrame({
        'feature': input_features,
        'importance_mean': perm_result.importances_mean,
        'importance_std': perm_result.importances_std,
    }).sort_values('importance_mean', ascending=False)

    print("\nPermutation importance (features ranked by mean F2 decrease when shuffled):")
    for _, row in perm_df.iterrows():
        sig = (" ***" if row['importance_mean'] > 2 * row['importance_std']
               and row['importance_mean'] > 0.005 else "")
        print(f"  {row['feature']:<25s}: {row['importance_mean']:+.4f} "
              f"(+/- {row['importance_std']:.4f}){sig}")

    return perm_df