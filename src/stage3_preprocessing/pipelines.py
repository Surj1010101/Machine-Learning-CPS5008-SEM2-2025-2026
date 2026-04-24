import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_recall_curve, auc, confusion_matrix

from common.pipeline_utils import make_preprocessor, sgkf


def build_baseline_pipeline():
    """Create the baseline LR pipeline."""
    return Pipeline([
        ('preprocessor', make_preprocessor()),
        ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='liblinear', random_state=42))
    ])


def run_cross_validation(X, y, groups, pipeline):
    """Run Stratified GroupKFold CV and return per-fold results."""
    print("\n" + "="*70)
    print("CROSS-VALIDATION: Stratified GroupKFold (k=5)")
    print("="*70)

    fold_results = []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        groups_train, groups_val = groups[train_idx], groups[val_idx]

        # Verify no customer overlap
        overlap = set(groups_train) & set(groups_val)
        assert len(overlap) == 0, f"Fold {fold_idx}: Customer overlap detected!"

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        f2 = fbeta_score(y_val, y_pred, beta=2)
        precision_arr, recall_arr, _ = precision_recall_curve(y_val, y_prob)
        pr_auc = auc(recall_arr, precision_arr)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fold_results.append({
            'fold': fold_idx + 1, 'f2': f2, 'pr_auc': pr_auc,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'n_train': len(train_idx), 'n_val': len(val_idx),
            'n_pos_train': y_train.sum(), 'n_pos_val': y_val.sum(),
        })
