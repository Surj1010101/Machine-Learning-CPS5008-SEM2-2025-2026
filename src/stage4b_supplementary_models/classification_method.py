import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import fbeta_score, mean_squared_error, r2_score

from common.pipeline_utils import make_preprocessor, sgkf, find_best_threshold_f2


def run_knn_comparison(X, y, groups):
    """Compare KNN for k = 3, 5, 7, 11, 15."""
    print("\n" + "=" * 70)
    print("1. K-NEAREST NEIGHBOURS CLASSIFICATION")
    print("=" * 70)

    knn_results = []
    for k in [3, 5, 7, 11, 15]:
        fold_f2s = []
        for train_idx, val_idx in sgkf.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipe = Pipeline([
                ('preprocessor', make_preprocessor()),
                ('classifier', KNeighborsClassifier(n_neighbors=k, n_jobs=-1))
            ])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_val)[:, 1]
            y_prob_train = pipe.predict_proba(X_train)[:, 1]
            thresh = find_best_threshold_f2(y_train, y_prob_train)
            y_pred = (y_prob >= thresh).astype(int)
            fold_f2s.append(fbeta_score(y_val, y_pred, beta=2))

        mean_f2 = np.mean(fold_f2s)
        knn_results.append({'k': k, 'f2_mean': mean_f2, 'f2_std': np.std(fold_f2s)})
        print(f"  KNN (k={k:2d}): F2 = {mean_f2:.4f} (+/- {np.std(fold_f2s):.4f})")

    knn_df = pd.DataFrame(knn_results)
    best_k = knn_df.loc[knn_df['f2_mean'].idxmax(), 'k']
    print(f"\n  Best KNN: k={int(best_k)}, F2={knn_df['f2_mean'].max():.4f}")
    print("  Note: Higher k values improve KNN by smoothing noisy predictions,")
    print("  but KNN remains sensitive to the curse of dimensionality in sparse")
    print("  TF-IDF space. LR is preferred for interpretability and efficiency.")

    return knn_df, int(best_k)


def run_regularisation_comparison(X, y, groups):
    """Compare L1, L2, and no regularisation for Logistic Regression."""
    print("\n" + "=" * 70)
    print("2. REGULARISATION TECHNIQUES COMPARISON")
    print("=" * 70)
    print("  Comparing L1 (Lasso), L2 (Ridge), and no regularisation on LR.")
    print("  Regularisation prevents overfitting by penalising large coefficients.")

    reg_configs = {
        'LR_L2_C1.0': {'penalty': 'l2', 'C': 1.0, 'solver': 'liblinear'},
        'LR_L2_C0.1': {'penalty': 'l2', 'C': 0.1, 'solver': 'liblinear'},
        'LR_L2_C10':  {'penalty': 'l2', 'C': 10.0, 'solver': 'liblinear'},
        'LR_L1_C1.0': {'penalty': 'l1', 'C': 1.0, 'solver': 'liblinear'},
        'LR_L1_C0.1': {'penalty': 'l1', 'C': 0.1, 'solver': 'liblinear'},
        'LR_none':    {'penalty': None, 'solver': 'lbfgs'},
    }

    reg_results = []
    for name, params in reg_configs.items():
        fold_f2s = []
        fold_nonzero = []
        for train_idx, val_idx in sgkf.split(X, y, groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipe = Pipeline([
                ('preprocessor', make_preprocessor()),
                ('classifier', LogisticRegression(
                    **params, class_weight='balanced', max_iter=2000, random_state=42))
            ])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_val)[:, 1]
            y_prob_train = pipe.predict_proba(X_train)[:, 1]
            thresh = find_best_threshold_f2(y_train, y_prob_train)
            y_pred = (y_prob >= thresh).astype(int)
            fold_f2s.append(fbeta_score(y_val, y_pred, beta=2))

            coefs = pipe.named_steps['classifier'].coef_[0]
            fold_nonzero.append(np.sum(np.abs(coefs) > 1e-6))

        mean_f2 = np.mean(fold_f2s)
        reg_results.append({
            'config': name, 'f2_mean': mean_f2, 'f2_std': np.std(fold_f2s),
            'avg_nonzero_features': np.mean(fold_nonzero)
        })
        print(f"  {name:15s}: F2 = {mean_f2:.4f} (+/- {np.std(fold_f2s):.4f}) "
              f"| non-zero features: {np.mean(fold_nonzero):.0f}")

    reg_df = pd.DataFrame(reg_results)
    print("\n  Key insight: L1 regularisation produces sparse models (fewer non-zero")
    print("  coefficients), acting as implicit feature selection. L2 regularisation")
    print("  shrinks all coefficients towards zero but retains all features.")
    print("  The bias-variance trade-off is controlled by the C parameter:")
    print("  smaller C = stronger regularisation = higher bias, lower variance.")

    return reg_df


def run_linear_regression_baseline(X_dense, y, groups):
    """ why i think blinear regression is inappropriate for binary targets."""
    print("\n" + "=" * 70)
    print("6. LINEAR REGRESSION BASELINE")
    print("=" * 70)
    print("  Demonstrating why linear regression is inappropriate for binary targets.")
    print("  Linear regression predicts continuous values, not probabilities bounded [0,1].")

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_dense, y, groups)):
        X_train_t = X_dense[train_idx]
        X_val_t = X_dense[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        lr = LinearRegression()
        lr.fit(X_train_t, y_train)
        y_pred_cont = lr.predict(X_val_t)

        below_0 = (y_pred_cont < 0).sum()
        above_1 = (y_pred_cont > 1).sum()

        y_pred_bin = (y_pred_cont >= 0.5).astype(int)
        f2 = fbeta_score(y_val, y_pred_bin, beta=2)
        mse = mean_squared_error(y_val, y_pred_cont)
        r2 = r2_score(y_val, y_pred_cont)

        fold_results.append({
            'fold': fold_idx + 1, 'f2': f2, 'mse': mse, 'r2': r2,
            'preds_below_0': below_0, 'preds_above_1': above_1
        })

    linreg_df = pd.DataFrame(fold_results)
    print(f"  Mean F2 (at 0.5 threshold): {linreg_df['f2'].mean():.4f}")
    print(f"  Mean R²: {linreg_df['r2'].mean():.4f}")
    print(f"  Predictions below 0: {linreg_df['preds_below_0'].mean():.0f} per fold")
    print(f"  Predictions above 1: {linreg_df['preds_above_1'].mean():.0f} per fold")
    print("\n  Linear regression produces predictions outside [0,1], which are")
    print("  uninterpretable as probabilities. Logistic regression constrains output")
    print("  via the sigmoid function, making it the appropriate choice for")
    print("  binary classification. This justifies Decision 008.")

    return linreg_df