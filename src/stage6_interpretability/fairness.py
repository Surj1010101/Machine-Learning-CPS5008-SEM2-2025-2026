"""
Stage 6 equalised odds analysis module across protected attributes.

Overall this module is where I run the formal fairness audit, the basic idea is to
collect predictions across all 5 folds and then compute the True Positive Rate, False
Positive Rate, Positive Predictive Value and F2 per protected group. In my project
this is really important because the brief asks for demographic and regional bias
evidence, and the equalised odds gap is the standard metric for whether my model
treats different groups consistently. What this module demonstrates is the formal
fairness check the brief requires, with concrete TPR and FPR gaps.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, confusion_matrix

from stage6_interpretability.model import make_preprocessor, find_best_threshold_f2


def run_equalised_odds(df, X, y, groups, text_col, categorical_cols, numeric_cols):
    """
    Collect per-sample predictions across all my folds, then compute TPR, FPR and PPV gaps.

    Overall this function re-runs the full StratifiedGroupKFold loop just to capture
    every prediction at the row level, then it groups by each protected attribute and
    reports the rates. The basic idea is that an aggregate F2 hides whether one group
    is being systematically under-served, and this analysis surfaces that. What this
    also reports is whether the gaps are within acceptable thresholds (TPR gap < 0.10,
    FPR gap < 0.15) so the report has a clear pass/fail verdict per attribute.
    """
    print("\n" + "=" * 70)
    print("FAIRNESS DEEP DIVE: EQUALISED ODDS")
    print("=" * 70)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    df['y_prob_all'] = np.nan
    df['y_pred_all'] = np.nan

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        pipe = Pipeline([
            ('preprocessor', make_preprocessor(text_col, categorical_cols, numeric_cols)),
            ('classifier', LogisticRegression(
                class_weight='balanced', max_iter=1000,
                solver='liblinear', random_state=42))
        ])
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_va)[:, 1]
        prob_tr = pipe.predict_proba(X_tr)[:, 1]
        thresh = find_best_threshold_f2(y_tr, prob_tr)
        preds = (probs >= thresh).astype(int)

        df.loc[df.index[val_idx], 'y_prob_all'] = probs
        df.loc[df.index[val_idx], 'y_pred_all'] = preds

    df['y_pred_all'] = df['y_pred_all'].astype(int)

    # Three protected attributes the brief asks me to audit
    fairness_attrs = {
        'region': ['England', 'Scotland', 'Wales'],
        'customer_type': ['Commercial', 'Domestic'],
        'tenure_type': ['New', 'Long-term'],
    }

    fairness_results = []

    for attr, groups_list in fairness_attrs.items():
        print(f"\n--- {attr} ---")
        print(f"  {'Group':<15s} {'TPR':>6s} {'FPR':>6s} {'PPV':>6s} {'F2':>6s} "
              f"{'N':>6s} {'Pos':>5s} {'TP':>4s} {'FP':>5s} {'FN':>4s}")
        print("  " + "-" * 75)

        for group in groups_list:
            mask = df[attr] == group
            sub = df[mask]
            y_true_g = sub['escalated'].values
            y_pred_g = sub['y_pred_all'].values

            cm = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            f2 = fbeta_score(y_true_g, y_pred_g, beta=2)

            fairness_results.append({
                'attribute': attr, 'group': group,
                'n': len(sub), 'n_pos': int(y_true_g.sum()),
                'tpr': tpr, 'fpr': fpr, 'ppv': ppv, 'f2': f2,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            })

            print(f"  {group:<15s} {tpr:>6.3f} {fpr:>6.3f} {ppv:>6.3f} {f2:>6.3f} "
                  f"{len(sub):>6d} {int(y_true_g.sum()):>5d} {tp:>4d} {fp:>5d} {fn:>4d}")

        # Equalised odds verdict per attribute
        attr_data = [r for r in fairness_results if r['attribute'] == attr]
        tprs = [r['tpr'] for r in attr_data]
        fprs = [r['fpr'] for r in attr_data]

        tpr_gap = max(tprs) - min(tprs)
        fpr_gap = max(fprs) - min(fprs)

        print(f"\n  Equalised Odds Assessment:")
        print(f"    TPR gap: {tpr_gap:.3f} (max - min)")
        print(f"    FPR gap: {fpr_gap:.3f} (max - min)")
        if tpr_gap < 0.10 and fpr_gap < 0.15:
            print(f"    --> Approximately equalised (gaps within acceptable range)")
        else:
            print(f"    --> Notable disparity detected -- investigate further")

    return pd.DataFrame(fairness_results)
