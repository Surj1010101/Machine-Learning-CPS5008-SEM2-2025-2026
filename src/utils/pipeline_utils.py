"""Pipeline Utils module."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve

from utils.data_loader import CATEGORICAL_COLS, NUMERIC_COLS, TEXT_COL_CLEAN

# Shared CV splitter, fixed random_state so every stage gets the same fold assignment
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


def make_preprocessor():
    """
    Create the standard ColumnTransformer used across all my models.

    This is the preprocessor factory, the aim is that every stage that
    builds a model calls this so the text/categorical/numeric handling is identical.
    This also applies the TF-IDF hyperparameters from my Stage 3b ablation
    study (max_features=500 + bigrams + sublinear_tf was the winning configuration).
    """
    return ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(
                max_features=500, ngram_range=(1, 2),
                min_df=3, max_df=0.95, sublinear_tf=True,
                strip_accents='unicode'), TEXT_COL_CLEAN),
            ('cat', OneHotEncoder(
                drop='first', sparse_output=True,
                handle_unknown='ignore'), CATEGORICAL_COLS),
            ('num', StandardScaler(), NUMERIC_COLS),
        ],
        remainder='drop'
    )


def find_best_threshold_f2(y_true, y_prob):
    """
    Find the classification threshold that maximises F2 score.

    This is my threshold helper used by every stage, the aim is to walk
    along the precision-recall curve, compute F2 at every operating point, and return
    the threshold where F2 peaks. This matters because an imbalanced problem
    like mine should not use the default 0.5 cutoff, the brief explicitly asks for
    threshold tuning and this is the helper that does it.
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
    return thresholds[np.argmax(f2_scores)]




