from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from common.pipeline_utils import make_preprocessor


def define_models(scale_pos_weight):
    """Return dict of from all model pipelines to compare."""
    return {
        'LR_baseline': Pipeline([
            ('preprocessor', make_preprocessor()),
            ('classifier', LogisticRegression(
                class_weight='balanced', max_iter=1000,
                solver='liblinear', random_state=42))
        ]),
        'RF': Pipeline([
            ('preprocessor', make_preprocessor()),
            ('classifier', RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_leaf=5,
                class_weight='balanced', random_state=42, n_jobs=-1))
        ]),
        'XGB': Pipeline([
            ('preprocessor', make_preprocessor()),
            ('classifier', XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                scale_pos_weight=scale_pos_weight, eval_metric='aucpr',
                random_state=42, n_jobs=-1, verbosity=0))
        ]),
        'LR_SMOTE': ImbPipeline([
            ('preprocessor', make_preprocessor()),
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('classifier', LogisticRegression(
                max_iter=1000, solver='liblinear', random_state=42))
        ]),
        'XGB_SMOTE': ImbPipeline([
            ('preprocessor', make_preprocessor()),
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('classifier', XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                eval_metric='aucpr', random_state=42, n_jobs=-1, verbosity=0))
        ]),
    }