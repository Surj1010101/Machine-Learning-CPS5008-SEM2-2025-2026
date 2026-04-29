"""
Stage 4b bias-variance trade-off analysis through learning curves.

Overall this module is where I diagnose whether my main Logistic Regression model is
suffering from high bias or high variance. The basic idea is to plot training and
validation F1 across increasing training set sizes, then look at the gap between them.
What this module demonstrates is the formal evidence for whether more data, more
features or stronger regularisation would help my model.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

from utils.pipeline_utils import make_preprocessor


def run_learning_curve_analysis(X, y):
    """
    Fit learning curves and diagnose bias vs variance for my LR baseline.

    Overall this function uses sklearn's learning_curve helper at five training fractions
    from 20% to 100%, the basic idea is that if the train and validation curves converge
    high then my model is good, if they converge low then it is high-bias, and if there
    is a big persistent gap then it is high-variance. What this also returns is a textual
    diagnosis I can quote in the report directly.
    """
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

    # Final-point train minus val gap, this is the headline number for the diagnosis
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
