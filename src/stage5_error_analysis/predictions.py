"""Predictions module."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_recall_curve, confusion_matrix


def make_preprocessor(text_col, categorical_cols, numeric_cols):
    """
    Standard ColumnTransformer used by my LR baseline.

    This is the same preprocessor I used in Stage 3, the aim is to keep
    the preprocessing identical between stages so the predictions I analyse here come
    from the same model the brief sees.
    """
    return ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                     min_df=3, max_df=0.95, sublinear_tf=True,
                                     strip_accents='unicode'), text_col),
            ('cat', OneHotEncoder(drop='first', sparse_output=True,
                                  handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols),
        ],
        remainder='drop'
    )


def find_best_threshold_f2(y_true, y_prob):
    """
    Find the threshold that maximises F2 on the given data.

    This is my threshold-tuning helper, the aim is to walk along the
    precision-recall curve and pick the point where F2 peaks, since the default 0.5
    cutoff is wrong for an imbalanced problem like mine.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f2_scores = []
    for p, r in zip(precisions, recalls):
        if p + r > 0:
            f2 = (5 * p * r) / (4 * p + r)
        else:
            f2 = 0
        f2_scores.append(f2)
    f2_scores = np.array(f2_scores[:-1])
    if len(f2_scores) == 0:
        return 0.5
    best_idx = np.argmax(f2_scores)
    return thresholds[best_idx]


def collect_fold_predictions(df, X, y, groups, text_col, categorical_cols, numeric_cols):
    """
    Fit my LR pipeline across 5 folds with tuned thresholds, and write predictions onto df.

    This is the workhorse of Stage 5, the aim is to attach y_prob, y_pred
    and prediction_type (TP/FP/FN/TN) directly onto the dataframe so every other module
    can filter rows by error type. This also returns the per-fold metrics list
    so run.py can quote the threshold per fold.
    """
    print("\nCollecting per-sample predictions from LR (tuned threshold)...")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    pipeline = Pipeline([
        ('preprocessor', make_preprocessor(text_col, categorical_cols, numeric_cols)),
        ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='liblinear', random_state=42))
    ])

    df['y_prob'] = np.nan
    df['y_pred'] = np.nan
    df['fold'] = -1

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)
        y_prob_val = pipeline.predict_proba(X_val)[:, 1]

        # Tuning threshold on training probabilities, never on validation
        y_prob_train = pipeline.predict_proba(X_train)[:, 1]
        best_thresh = find_best_threshold_f2(y_train, y_prob_train)
        y_pred_val = (y_prob_val >= best_thresh).astype(int)

        df.loc[df.index[val_idx], 'y_prob'] = y_prob_val
        df.loc[df.index[val_idx], 'y_pred'] = y_pred_val
        df.loc[df.index[val_idx], 'fold'] = fold_idx + 1

        f2 = fbeta_score(y_val, y_pred_val, beta=2)
        cm = confusion_matrix(y_val, y_pred_val)
        tn, fp, fn, tp = cm.ravel()
        fold_metrics.append({
            'fold': fold_idx + 1, 'f2': f2, 'threshold': best_thresh,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        })
        print(f"  Fold {fold_idx+1}: F2={f2:.4f}, thresh={best_thresh:.3f}, "
              f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Labelling each row by its prediction type so downstream filters are easy
    df['y_pred'] = df['y_pred'].astype(int)
    df['prediction_type'] = 'TN'
    df.loc[(df['escalated'] == 1) & (df['y_pred'] == 1), 'prediction_type'] = 'TP'
    df.loc[(df['escalated'] == 0) & (df['y_pred'] == 1), 'prediction_type'] = 'FP'
    df.loc[(df['escalated'] == 1) & (df['y_pred'] == 0), 'prediction_type'] = 'FN'

    return fold_metrics



