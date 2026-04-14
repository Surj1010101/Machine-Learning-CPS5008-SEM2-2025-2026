"""
Shared pipeline utilities: preprocessor factory and threshold tuning.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model selection import StratifiedGroupKFold
from skelearn.metrics import precision_recall_curve

from src.utils.data_loader import CATEGORICAL_COLS, NUMERIC_COLS

#cv spiltter shared 
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

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
        remainder='drop'
    )

def find_best_threshold_f2(y_true,y_prob):
    """Find classifictation threshold that maximises F2score."""
    precisions, recalls, threshold = precision_recall_curve(y_true, y_prob)
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