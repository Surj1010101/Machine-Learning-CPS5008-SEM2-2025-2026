

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (fbeta_score, precision_recall_curve, auc,
                             confusion_matrix)
from xgboost import XGBClassifier

from common.pipeline_utils import sgkf, make_preprocessor, find_best_threshold_f2


def tune_xgboost(X, y, groups, scale_pos_weight):
    """Grid search over XGBoost hyperparameters using and StratifiedGroupKFold."""
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