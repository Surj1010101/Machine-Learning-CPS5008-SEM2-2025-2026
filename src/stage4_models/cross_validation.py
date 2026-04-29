import numpy as np
import pandas as pd
from sklearn.metrics import (fbeta_score, precision_recall_curve, auc,
                             confusion_matrix)

from common.pipeline_utils import sgkf, find_best_threshold_f2


def cross_validate_all(models, X, y, groups):
    """Running CV with threshold tuning for each model. and Returns results dict."""
    print("\n" + "="*70)
    print("CROSS-VALIDATION: All Models")
    print("="*70)

    all_results = {}

    for model_name, pipeline in models.items():
        print(f"\n--- {model_name} ---")
        fold_metrics = []
        all_y_true, all_y_pred_default, all_y_pred_tuned = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            y_pred_default = pipeline.predict(X_val)

            # Tuned threshold from training set
            y_prob_train = pipeline.predict_proba(X_train)[:, 1]
            best_thresh = find_best_threshold_f2(y_train, y_prob_train)
            y_pred_tuned = (y_prob >= best_thresh).astype(int)

            f2_default = fbeta_score(y_val, y_pred_default, beta=2)
            prec_arr, rec_arr, _ = precision_recall_curve(y_val, y_prob)
            pr_auc_val = auc(rec_arr, prec_arr)
            f2_tuned = fbeta_score(y_val, y_pred_tuned, beta=2)
            cm_tuned = confusion_matrix(y_val, y_pred_tuned)
            tn, fp, fn, tp = cm_tuned.ravel()

            fold_metrics.append({
                'fold': fold_idx + 1,
                'f2_default': f2_default, 'f2_tuned': f2_tuned,
                'pr_auc': pr_auc_val, 'threshold': best_thresh,
                'precision_tuned': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall_tuned': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            })

            all_y_true.extend(y_val)
            all_y_pred_default.extend(y_pred_default)
            all_y_pred_tuned.extend(y_pred_tuned)

            print(f"  Fold {fold_idx+1}: F2_default={f2_default:.4f} | "
                  f"F2_tuned={f2_tuned:.4f} | PR-AUC={pr_auc_val:.4f} | "
                  f"thresh={best_thresh:.3f}")