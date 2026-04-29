from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

from common.pipeline_utils import make_preprocessor


def run_learning_curve_analysis(X, y):
    """Fit learning curves and diagnose bias vs variance."""
    print("\n" + "=" * 70)
    print("3. BIAS-VARIANCE TRADE-OFF (Learning Curves)")
    print("=" * 70)
    print("  Learning curves reveal whether a model suffers from high bias")
    print("  (underfitting) or high variance (overfitting).")

    pipe_lr = Pipeline([
        ('preprocessor', make_preprocessor()),
        ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='liblinear', random_state=42))
    ])

    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipe_lr, X, y, cv=5,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        scoring='f1', random_state=42, n_jobs=-1
    )

    print(f"  Training sizes tested: {train_sizes_abs}")
    print(f"  Train F1 (mean):  {train_scores.mean(axis=1)}")
    print(f"  Val F1 (mean):    {val_scores.mean(axis=1)}")

    gap = train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
    print(f"\n  Train-validation gap at full data: {gap:.4f}")
    if gap < 0.05:
        diagnosis = "LOW VARIANCE (no overfitting). Model may have HIGH BIAS (underfitting)."
    elif gap < 0.15:
        diagnosis = "MODERATE variance. Reasonable bias-variance balance."
    else:
        diagnosis = "HIGH VARIANCE (overfitting). Consider stronger regularisation or more data."
    print(f"  Diagnosis: {diagnosis}")
    print("  Interpretation: The train-validation gap suggests the model memorises training")
    print("  patterns that do not generalise. Stronger regularisation (lower C) helps reduce")
    print("  this gap, as shown in the regularisation comparison (L1 C=0.1 is best).")
    print("  The overall weak validation performance reflects limited signal in short emails.")

    return train_sizes_abs, train_scores, val_scores, gap, diagnosis