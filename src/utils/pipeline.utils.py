from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.data_loader import CATEGORICAL_COLS, NUMERIC_COLS


def make_preproccessor():
    """Create a standard ColumnTransform used across all models."""
    return ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(
                max_features=500, ngram_range=(1, 2),
                min_df=3, max_df=0.9, sublinear_tf=True,
                strip_accents='unicode'), TEXT_COL_CLEAN),
                ('cat', OneHotEncoder(
                    drop='first', sparse_output=True,
                    handle_unknown='ignore'), CATEGORICAL_COLS,
                ('num', StandardScaler(), NUMERIC_COLS),                       
                )         
        ],
    )
