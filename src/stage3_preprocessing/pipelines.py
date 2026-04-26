"""Pipeline definition and crossvalidation for the baseline model."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_recall_curve, auc, confusion_matrix

from utils.pipeline_utils import make_preprocessor, sgkf


def build_baseline_pipeline():
    """Create the baseline LR pipeline."""
    return Pipeline([
        ('preprocessor', make_preprocessor()),
        ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='liblinear', random_state=42))
    ])


def run_cross_validation(X, y, groups, pipeline):
    """run Stratified GroupKFold CV and return perfold results."""
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
        

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        print(f"\nFold {fold_idx+1}:")
        print(f"  Train: {len(train_idx)} ({y_train.sum()} pos) | "
              f"Val: {len(val_idx)} ({y_val.sum()} pos)")
        print(f"  F2={f2:.4f} | PR-AUC={pr_auc:.4f} | "
              f"Precision={fold_results[-1]['precision']:.4f} | "
              f"Recall={fold_results[-1]['recall']:.4f}")
        print(f"  Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  Customer overlap: {len(overlap)} (verified clean)")

    return pd.DataFrame(fold_results), np.array(all_y_true), np.array(all_y_pred)


def print_aggregate_results(results_df, all_y_true, all_y_pred):
    """print summary statistics across all folds."""
    print("\n" + "="*70)
    print("AGGREGATE CROSS-VALIDATION RESULTS")
    print("="*70)

    print(f"\n{'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 48)
    for metric in ['f2', 'pr_auc', 'precision', 'recall']:
        vals = results_df[metric]
        print(f"{metric:<12} {vals.mean():>8.4f} {vals.std():>8.4f} "
              f"{vals.min():>8.4f} {vals.max():>8.4f}")

    cm_total = confusion_matrix(all_y_true, all_y_pred)
    tn_t, fp_t, fn_t, tp_t = cm_total.ravel()
    print(f"\nOverall confusion matrix (aggregated across folds):")
    print(f"  TN={tn_t}, FP={fp_t}, FN={fn_t}, TP={tp_t}")
    print(f"  Overall F2: {fbeta_score(all_y_true, all_y_pred, beta=2):.4f}")

    dummy_f2 = fbeta_score(all_y_true, np.zeros_like(all_y_true), beta=2)
    print(f"\nDummy (all-negative) F2: {dummy_f2:.4f}")
    dummy_f2_all1 = fbeta_score(all_y_true, np.ones_like(all_y_true), beta=2)
    print(f"Dummy (all-positive) F2: {dummy_f2_all1:.4f}")
