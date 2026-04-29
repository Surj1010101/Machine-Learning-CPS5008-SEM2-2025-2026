"""Pipeline setup for Stage 6 for preprocessor, threshold tuning, single-fold training."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, fbeta_score


def make_preprocessor(text_col, categorical_cols, numeric_cols):
    """Standard ColumnTransformer for the LR baseline."""
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
    """Finds the threshold that maximises F2 on the given data."""
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


def train_representative_fold(X, y, folds, text_col, categorical_cols, numeric_cols):
    """Trains LR on fold 0 and return the fitted pipeline, threshold, and fold indices."""
    print("\nTraining model on largest fold for interpretation...")

    train_idx, val_idx = folds[0]
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    pipeline = Pipeline([
        ('preprocessor', make_preprocessor(text_col, categorical_cols, numeric_cols)),
        ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='liblinear', random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    best_thresh = find_best_threshold_f2(y_train, y_prob_train)
    print(f"Tuned threshold: {best_thresh:.3f}")

    y_prob_val = pipeline.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val >= best_thresh).astype(int)
    f2_val = fbeta_score(y_val, y_pred_val, beta=2)
    print(f"Fold 1 validation F2: {f2_val:.4f}")

    return pipeline, best_thresh, train_idx, val_idx, y_prob_val, y_pred_val
