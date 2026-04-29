"""
Stage 6 model setup module: preprocessor, threshold tuning and single-fold training.

Overall this module is where I rebuild the same Logistic Regression pipeline used in
earlier stages so my interpretability analysis is on the same model the brief actually
sees. The basic idea is to train on a single representative fold rather than full CV,
because LIME and permutation importance work on a fitted model not on cross-validation
folds. What this module demonstrates is the bridge between my Stage 4 model and my
Stage 6 explanations.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, fbeta_score


def make_preprocessor(text_col, categorical_cols, numeric_cols):
    """
    Standard ColumnTransformer used by my LR baseline.

    Overall this is the same preprocessor I used in Stage 3 and Stage 5, the basic idea
    is to keep the preprocessing identical between stages so the model under inspection
    here behaves the same way as in the main pipeline.
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

    Overall this is the same threshold-tuning helper used in earlier stages, the basic
    idea is to walk the precision-recall curve and pick where F2 peaks.
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


def train_representative_fold(X, y, folds, text_col, categorical_cols, numeric_cols):
    """
    Train LR on fold 0 and return the fitted pipeline plus the threshold and indices.

    Overall this gives me one fully fitted model to do interpretability on, the basic
    idea is that I do not need to retrain inside every interpretability function, I
    just train once and pass the fitted pipeline around. What this also returns is
    everything the LIME and permutation modules need, the train and val indices, the
    validation probabilities and the validation predictions.
    """
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

    # Tuning threshold on training probabilities, never on validation
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    best_thresh = find_best_threshold_f2(y_train, y_prob_train)
    print(f"Tuned threshold: {best_thresh:.3f}")

    y_prob_val = pipeline.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val >= best_thresh).astype(int)
    f2_val = fbeta_score(y_val, y_pred_val, beta=2)
    print(f"Fold 1 validation F2: {f2_val:.4f}")

    return pipeline, best_thresh, train_idx, val_idx, y_prob_val, y_pred_val
