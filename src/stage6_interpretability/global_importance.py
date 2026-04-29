"""
Stage 6 global feature importance: LR coefficients and permutation importance.

Overall this module is where I produce the global "what does my model rely on" answer,
the basic idea is to attack the question two ways. First I pull the raw LR coefficients
which show direction and magnitude per feature, then I run permutation importance which
measures the actual F2 drop when each input column is shuffled. In my project this is
really important because the brief asks for both feature ranking and the WHY behind the
ranking, and combining the two methods gives me cross-evidence. What this module
demonstrates is the global interpretability the brief expects in Stage 6.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.inspection import permutation_importance


def run_lr_coefficient_importance(pipeline):
    """
    Extract and rank my LR coefficients by absolute value.

    Overall this is the cheapest interpretability method, the basic idea is that the
    coefficients are already there inside the fitted classifier, I just have to pair
    them with the feature names from the preprocessor. What this also produces is the
    top 15 escalation indicators and top 15 de-escalation indicators which is what
    the report quotes directly.
    """
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

    # Stripping the ColumnTransformer prefixes so the report names are clean
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
    """
    Run permutation importance on the validation set using F2 as the scorer.

    Overall this is the more rigorous global importance method, the basic idea is that
    coefficients tell me what the model is doing internally but they do not tell me how
    much each feature contributes to actual F2 on held-out data. Permutation importance
    fixes that by shuffling one feature at a time and measuring how much F2 drops. What
    this also uses is the F2-tuned threshold from the representative fold so the
    importance reflects deployment performance, not default-threshold performance.
    """
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
        # Marking results as significant if mean importance exceeds twice the std
        sig = (" ***" if row['importance_mean'] > 2 * row['importance_std']
               and row['importance_mean'] > 0.005 else "")
        print(f"  {row['feature']:<25s}: {row['importance_mean']:+.4f} "
              f"(+/- {row['importance_std']:.4f}){sig}")

    return perm_df
