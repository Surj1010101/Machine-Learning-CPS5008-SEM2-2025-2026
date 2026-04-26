import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_recall_curve, auc


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


def evaluate_config(X_data, y_data, groups_data, preprocessor, config_name, sgkf):
    """Run 5-fold CV with tuned threshold for a given preprocessor config."""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='liblinear', random_state=42))
    ])

    fold_f2 = []
    fold_prauc = []
    fold_recall = []
    fold_precision = []

    for train_idx, val_idx in sgkf.split(X_data, y_data, groups_data):
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        y_prob_train = pipeline.predict_proba(X_train)[:, 1]
        thresh = find_best_threshold_f2(y_train, y_prob_train)
        y_pred = (y_prob >= thresh).astype(int)

        f2 = fbeta_score(y_val, y_pred, beta=2)
        prec_arr, rec_arr, _ = precision_recall_curve(y_val, y_prob)
        pr_auc_val = auc(rec_arr, prec_arr)
        tp = ((y_val == 1) & (y_pred == 1)).sum()
        fp = ((y_val == 0) & (y_pred == 1)).sum()
        fn = ((y_val == 1) & (y_pred == 0)).sum()

        fold_f2.append(f2)
        fold_prauc.append(pr_auc_val)
        fold_recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fold_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)

    return {
        'config': config_name,
        'f2_mean': np.mean(fold_f2),
        'f2_std': np.std(fold_f2),
        'prauc_mean': np.mean(fold_prauc),
        'prauc_std': np.std(fold_prauc),
        'recall_mean': np.mean(fold_recall),
        'precision_mean': np.mean(fold_precision),
        'fold_f2_values': fold_f2,
    }


def evaluate_config(X_data, y_data, groups_data, preprocessor, config_name, sgkf):
    """Run 5-fold CV with tuned threshold for a given preprocessor config."""
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            solver='liblinear', random_state=42))
    ])

    fold_f2 = []
    fold_prauc = []
    fold_recall = []
    fold_precision = []

    for train_idx, val_idx in sgkf.split(X_data, y_data, groups_data):
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        y_prob_train = pipeline.predict_proba(X_train)[:, 1]
        thresh = find_best_threshold_f2(y_train, y_prob_train)
        y_pred = (y_prob >= thresh).astype(int)

        f2 = fbeta_score(y_val, y_pred, beta=2)
        prec_arr, rec_arr, _ = precision_recall_curve(y_val, y_prob)
        pr_auc_val = auc(rec_arr, prec_arr)
        tp = ((y_val == 1) & (y_pred == 1)).sum()
        fp = ((y_val == 0) & (y_pred == 1)).sum()
        fn = ((y_val == 1) & (y_pred == 0)).sum()

        fold_f2.append(f2)
        fold_prauc.append(pr_auc_val)
        fold_recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fold_precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)

    return {
        'config': config_name,
        'f2_mean': np.mean(fold_f2),
        'f2_std': np.std(fold_f2),
        'prauc_mean': np.mean(fold_prauc),
        'prauc_std': np.std(fold_prauc),
        'recall_mean': np.mean(fold_recall),
        'precision_mean': np.mean(fold_precision),
        'fold_f2_values': fold_f2,
    }


def run_all_ablations(X, y, groups, text_col, categorical_cols, numeric_cols):
    """Run all 7 ablation configurations and return results as a list of dicts."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING ABLATION STUDY")
    print("=" * 70)

    # 1. Full pipeline
    print("\n[1/7] Full pipeline (TF-IDF bigrams + categorical + numeric)...")
    full_preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                     min_df=3, max_df=0.95, sublinear_tf=True,
                                     strip_accents='unicode'), text_col),
            ('cat', OneHotEncoder(drop='first', sparse_output=True,
                                  handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols),
        ], remainder='drop')
    result_full = evaluate_config(X, y, groups, full_preprocessor, 'Full pipeline', sgkf)
    print(f"   F2={result_full['f2_mean']:.4f} (+/-{result_full['f2_std']:.4f})")

    # 2. TF-IDF unigrams only
    print("\n[2/7] TF-IDF unigrams only (no bigrams)...")
    unigram_preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=500, ngram_range=(1, 1),
                                     min_df=3, max_df=0.95, sublinear_tf=True,
                                     strip_accents='unicode'), text_col),
            ('cat', OneHotEncoder(drop='first', sparse_output=True,
                                  handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols),
        ], remainder='drop')
    result_unigram = evaluate_config(X, y, groups, unigram_preprocessor, 'Unigrams only', sgkf)
    print(f"   F2={result_unigram['f2_mean']:.4f} (+/-{result_unigram['f2_std']:.4f})")



