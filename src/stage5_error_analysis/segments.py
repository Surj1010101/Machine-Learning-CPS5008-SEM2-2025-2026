"""Segmentlevel performance breakdown and fairness disparity analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve, auc, confusion_matrix


def run_segment_analysis(df):
    """Compute F2/recall/precision/FPR per category for each protected attribute."""
    print("\n" + "=" * 70)
    print("SEGMENT-LEVEL PERFORMANCE BREAKDOWN")
    print("=" * 70)

    segment_cols = ['customer_type', 'tenure_type', 'meter_type', 'region',
                    'issue_category', 'sentiment']

    segment_results = []

    for col in segment_cols:
        print(f"\n--- {col} ---")
        for cat in sorted(df[col].unique()):
            mask = df[col] == cat
            sub = df[mask]
            n = len(sub)
            n_pos = sub['escalated'].sum()

            if n_pos == 0:
                print(f"  {cat:20s}: n={n:4d}, 0 positives -- skipped")
                continue

            y_true_seg = sub['escalated'].values
            y_pred_seg = sub['y_pred'].values
            y_prob_seg = sub['y_prob'].values

            f2_seg = fbeta_score(y_true_seg, y_pred_seg, beta=2)
            cm_seg = confusion_matrix(y_true_seg, y_pred_seg, labels=[0, 1])
            tn_s, fp_s, fn_s, tp_s = cm_seg.ravel()

            recall_seg = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
            precision_seg = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
            fpr_seg = fp_s / (fp_s + tn_s) if (fp_s + tn_s) > 0 else 0

            if n_pos >= 5:
                prec_arr, rec_arr, _ = precision_recall_curve(y_true_seg, y_prob_seg)
                prauc_seg = auc(rec_arr, prec_arr)
            else:
                prauc_seg = np.nan

            segment_results.append({
                'segment': col, 'category': cat, 'n': n, 'n_pos': n_pos,
                'prevalence': n_pos / n,
                'f2': f2_seg, 'pr_auc': prauc_seg,
                'precision': precision_seg, 'recall': recall_seg,
                'fpr': fpr_seg,
                'tp': tp_s, 'fp': fp_s, 'fn': fn_s, 'tn': tn_s
            })

            print(f"  {cat:20s}: n={n:4d}, pos={n_pos:3d} ({n_pos/n*100:5.1f}%), "
                  f"F2={f2_seg:.3f}, Recall={recall_seg:.3f}, "
                  f"Precision={precision_seg:.3f}, FPR={fpr_seg:.3f}")

    seg_df = pd.DataFrame(segment_results)
    return seg_df


def run_fairness_analysis(seg_df, fairness_attrs=('region', 'customer_type')):
    """Recall disparity and 80% rule check for each protected attribute."""
    print("\n" + "=" * 70)
    print("FAIRNESS ANALYSIS (Recall Disparity)")
    print("=" * 70)

    for attr in fairness_attrs:
        print(f"\n--- {attr} ---")
        attr_seg = seg_df[seg_df['segment'] == attr].copy()
        if len(attr_seg) < 2:
            print("  Insufficient categories for comparison")
            continue

        max_recall = attr_seg.loc[attr_seg['recall'].idxmax()]
        min_recall = attr_seg.loc[attr_seg['recall'].idxmin()]

        disparity = max_recall['recall'] - min_recall['recall']
        ratio = min_recall['recall'] / max_recall['recall'] if max_recall['recall'] > 0 else 0

        print(f"  Highest recall: {max_recall['category']} ({max_recall['recall']:.3f})")
        print(f"  Lowest recall:  {min_recall['category']} ({min_recall['recall']:.3f})")
        print(f"  Disparity: {disparity:.3f}")
        print(f"  Ratio (min/max): {ratio:.3f}")

        if ratio >= 0.8:
            print(f"  --> PASSES 80% rule (ratio >= 0.80)")
        else:
            print(f"  --> FAILS 80% rule (ratio < 0.80) -- investigate further")

        fprs = attr_seg[['category', 'fpr']].copy()
        fpr_max = fprs.loc[fprs['fpr'].idxmax()]
        fpr_min = fprs.loc[fprs['fpr'].idxmin()]
        print(f"  FPR range: {fpr_min['category']} ({fpr_min['fpr']:.3f}) to "
              f"{fpr_max['category']} ({fpr_max['fpr']:.3f})")