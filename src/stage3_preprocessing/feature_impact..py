"""Sentiment ablation experiment: test with vs without sentiment feature."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score

from common.data_loader import TEXT_COL_CLEAN, NUMERIC_COLS
from common.pipeline_utils import sgkf


def run_sentiment_ablation(X, y, groups, results_df):
    """Compare model performance with and without sentiment feature."""
    print("\n" + "="*70)
    print("ABLATION: WITH vs WITHOUT SENTIMENT")
    print("="*70)

    categorical_cols_no_sent = ['customer_type', 'tenure_type', 'meter_type',
                                'region', 'issue_category']

    preprocessor_no_sent = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3,
                                      max_df=0.95, sublinear_tf=True,
                                      strip_accents='unicode'), TEXT_COL_CLEAN),
            ('cat', OneHotEncoder(drop='first', sparse_output=True,
                                   handle_unknown='ignore'), categorical_cols_no_sent),
            ('num', StandardScaler(), NUMERIC_COLS),
        ],
        remainder='drop'
    )

    pipeline_no_sent = Pipeline([
        ('preprocessor', preprocessor_no_sent),
        ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='liblinear', random_state=42))
    ])

    f2_no_sent = []
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline_no_sent.fit(X_train, y_train)
        y_pred_ns = pipeline_no_sent.predict(X_val)
        f2_ns = fbeta_score(y_val, y_pred_ns, beta=2)
        f2_no_sent.append(f2_ns)

    diff = results_df['f2'].mean() - np.mean(f2_no_sent)

    print(f"\nWith sentiment:    F2 = {results_df['f2'].mean():.4f} "
          f"(+/- {results_df['f2'].std():.4f})")
    print(f"Without sentiment: F2 = {np.mean(f2_no_sent):.4f} "
          f"(+/- {np.std(f2_no_sent):.4f})")
    print(f"Difference: {diff:+.4f}")

    if abs(diff) < 0.02:
        print("CONCLUSION: Sentiment has NEGLIGIBLE impact on F2. No evidence of leakage.")
    elif diff > 0.05:
        print("WARNING: Sentiment substantially improves F2. Possible leakage concern.")
    else:
        print("CONCLUSION: Sentiment has a small effect. Leakage risk remains LOW-MEDIUM.")

    return diff