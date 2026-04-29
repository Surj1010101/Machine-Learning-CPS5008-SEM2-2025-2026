"""Pipeline setup and per-sample prediction collection for Stage 7."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve


def make_preprocessor(text_col, categorical_cols, numeric_cols):
    """Standard ColumnTransformer used for the LR baseline."""
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
    """Find threshold that maximises F2 on the given data."""
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
    return thresholds[np.argmax(f2_scores)]


def collect_predictions(df, X, y, groups, text_col, categorical_cols, numeric_cols):
    """Fit LR across 5 folds and write y_prob/y_pred onto df. Returns mean threshold."""
    print("\nCollecting per-sample predictions...")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    df['y_prob'] = np.nan
    df['y_pred'] = np.nan

    fold_thresholds = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipe = Pipeline([
            ('preprocessor', make_preprocessor(text_col, categorical_cols, numeric_cols)),
            ('classifier', LogisticRegression(
                class_weight='balanced', max_iter=1000,
                solver='liblinear', random_state=42))
        ])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_val)[:, 1]
        prob_tr = pipe.predict_proba(X_train)[:, 1]
        thresh = find_best_threshold_f2(y_train, prob_tr)
        fold_thresholds.append(thresh)

        df.loc[df.index[val_idx], 'y_prob'] = probs

    mean_thresh = np.mean(fold_thresholds)
    df['y_pred'] = (df['y_prob'] >= mean_thresh).astype(int)
    print(f"Mean tuned threshold: {mean_thresh:.3f}")

    return mean_thresh
