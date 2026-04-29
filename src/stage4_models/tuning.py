"""
Stage 4 hyperparameter tuning module for XGBoost via manual grid search.

XGBoost hyperparameter tuning happens here. I loop over a small grid of n_estimators,
max_depth and learning_rate, and re-run my StratifiedGroupKFold cross-validation for
each combination. I picked manual grid search rather than RandomizedSearchCV because
the grid is small enough that exhaustive search is cheap, and it gives me a clean
ranking of every configuration that I can put straight into the report. The brief
asks for formal hyperparameter tuning and this file is where that requirement gets
satisfied, with a re-run on the best configuration so the final results are clean.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (fbeta_score, precision_recall_curve, auc,
                             confusion_matrix)
from xgboost import XGBClassifier

from utils.pipeline_utils import sgkf, make_preprocessor, find_best_threshold_f2


def tune_xgboost(X, y, groups, scale_pos_weight):
    """
    Grid search over XGBoost hyperparameters using StratifiedGroupKFold.

    Three nested for-loops cover n_estimators, max_depth and learning_rate, and every
    combination gets evaluated on the same folds with a tuned F2 threshold per fold.
    Saving the full tuning results to CSV means I can show the search landscape in
    the report without having to re-run the grid.
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)

    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
    }

    print("\nTuning XGBoost (manual grid search with StratifiedGroupKFold)...")

    best_f2 = -1
    best_params = None
    tuning_results = []

    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                fold_f2s = []
                for train_idx, val_idx in sgkf.split(X, y, groups):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    pipe = Pipeline([
                        ('preprocessor', make_preprocessor()),
                        ('classifier', XGBClassifier(
                            n_estimators=n_est, max_depth=depth,
                            learning_rate=lr, scale_pos_weight=scale_pos_weight,
                            eval_metric='aucpr', random_state=42,
                            n_jobs=-1, verbosity=0))
                    ])
                    pipe.fit(X_train, y_train)
                    y_prob = pipe.predict_proba(X_val)[:, 1]

                    # Tuning threshold on training probabilities to avoid leakage
                    y_prob_train = pipe.predict_proba(X_train)[:, 1]
                    thresh = find_best_threshold_f2(y_train, y_prob_train)
                    y_pred = (y_prob >= thresh).astype(int)
                    fold_f2s.append(fbeta_score(y_val, y_pred, beta=2))

                mean_f2 = np.mean(fold_f2s)
                tuning_results.append({
                    'n_estimators': n_est, 'max_depth': depth,
                    'learning_rate': lr, 'mean_f2': mean_f2,
                    'std_f2': np.std(fold_f2s)
                })

                if mean_f2 > best_f2:
                    best_f2 = mean_f2
                    best_params = {'n_estimators': n_est, 'max_depth': depth,
                                   'learning_rate': lr}

    print(f"\nBest XGBoost params: {best_params}")
    print(f"Best XGBoost F2 (tuned threshold): {best_f2:.4f}")

    tuning_df = pd.DataFrame(tuning_results).sort_values('mean_f2', ascending=False)
    print(f"\nTop 5 configurations:")
    print(tuning_df.head(5).to_string(index=False))
    tuning_df.to_csv('outputs/stage4/xgb_tuning_results.csv', index=False)

    return best_params, best_f2


def evaluate_tuned_xgb(X, y, groups, best_params, scale_pos_weight):
    """
    Re-run my best XGBoost with the tuned hyperparameters across all folds.

    This function takes the winning param combination and runs it through
    StratifiedGroupKFold one more time to get clean per-fold metrics. Separating
    "what configuration won" from "how does the winning configuration actually
    perform per fold" matters because it gives me a clean per-fold dataframe that
    the run.py paired t-test can compare against LR_baseline.
    """
    print("\n" + "="*70)
    print("RE-RUNNING BEST XGB WITH TUNED HYPERPARAMETERS")
    print("="*70)

    best_pipeline = Pipeline([
        ('preprocessor', make_preprocessor()),
        ('classifier', XGBClassifier(
            **best_params, scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr', random_state=42, n_jobs=-1, verbosity=0))
    ])

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        best_pipeline.fit(X_train, y_train)
        y_prob = best_pipeline.predict_proba(X_val)[:, 1]
        y_prob_train = best_pipeline.predict_proba(X_train)[:, 1]
        thresh = find_best_threshold_f2(y_train, y_prob_train)
        y_pred = (y_prob >= thresh).astype(int)

        f2 = fbeta_score(y_val, y_pred, beta=2)
        prec_arr, rec_arr, _ = precision_recall_curve(y_val, y_prob)
        pr_auc_val = auc(rec_arr, prec_arr)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fold_metrics.append({
            'fold': fold_idx + 1, 'f2': f2, 'pr_auc': pr_auc_val,
            'threshold': thresh,
            'precision': tp/(tp+fp) if (tp+fp) > 0 else 0,
            'recall': tp/(tp+fn) if (tp+fn) > 0 else 0,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        })

        print(f"Fold {fold_idx+1}: F2={f2:.4f} | PR-AUC={pr_auc_val:.4f} | "
              f"thresh={thresh:.3f} | P={fold_metrics[-1]['precision']:.3f} "
              f"R={fold_metrics[-1]['recall']:.3f}")

    best_df = pd.DataFrame(fold_metrics)
    print(f"\nBest XGB (tuned): F2={best_df['f2'].mean():.4f} "
          f"(+/-{best_df['f2'].std():.4f}) | "
          f"PR-AUC={best_df['pr_auc'].mean():.4f} "
          f"(+/-{best_df['pr_auc'].std():.4f})")

    return best_df
