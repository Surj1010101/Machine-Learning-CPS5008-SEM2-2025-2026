"""Threshold sweep and three-tier deployment analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, confusion_matrix


def run_threshold_analysis(df, mean_thresh):
    """Sweep thresholds 0.10-0.85 and evaluate F2, recall, precision, cost, workload."""
    print("\n" + "=" * 70)
    print("TIERED THRESHOLD ANALYSIS")
    print("=" * 70)

    print("\nExploring different threshold strategies for deployment...")

    thresholds_to_test = np.arange(0.10, 0.90, 0.05)
    threshold_analysis = []

    y_true = df['escalated'].values
    y_probs = df['y_prob'].values

    for t in thresholds_to_test:
        y_pred_t = (y_probs >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        f2 = fbeta_score(y_true, y_pred_t, beta=2)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        flagged_pct = (tp + fp) / len(y_true) * 100
        cost = tp * 50 + fp * 20 + fn * 500

        threshold_analysis.append({
            'threshold': t, 'f2': f2, 'recall': recall, 'precision': precision,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'flagged_pct': flagged_pct, 'cost': cost
        })

    thresh_df = pd.DataFrame(threshold_analysis)

    print(f"\n{'Thresh':>6s} {'F2':>6s} {'Recall':>7s} {'Prec':>6s} "
          f"{'Flagged%':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Cost':>8s}")
    print("-" * 65)
    for _, row in thresh_df.iterrows():
        print(f"{row['threshold']:>6.2f} {row['f2']:>6.3f} {row['recall']:>7.3f} "
              f"{row['precision']:>6.3f} {row['flagged_pct']:>7.1f}% "
              f"{int(row['tp']):>5d} {int(row['fp']):>5d} {int(row['fn']):>5d} "
              f"£{int(row['cost']):>7,d}")

    best_f2_row = thresh_df.loc[thresh_df['f2'].idxmax()]
    best_cost_row = thresh_df.loc[thresh_df['cost'].idxmin()]
    workload_30 = thresh_df.iloc[(thresh_df['flagged_pct'] - 30).abs().argsort()[:1]]

    print(f"\nOptimal thresholds:")
    print(f"  Best F2 ({best_f2_row['f2']:.3f}):    threshold={best_f2_row['threshold']:.2f}, "
          f"recall={best_f2_row['recall']:.3f}, flagged={best_f2_row['flagged_pct']:.1f}%")
    print(f"  Lowest cost (£{int(best_cost_row['cost']):,}): threshold={best_cost_row['threshold']:.2f}, "
          f"recall={best_cost_row['recall']:.3f}, flagged={best_cost_row['flagged_pct']:.1f}%")
    print(f"  ~30% workload:          threshold={workload_30.iloc[0]['threshold']:.2f}, "
          f"recall={workload_30.iloc[0]['recall']:.3f}, F2={workload_30.iloc[0]['f2']:.3f}")

    return thresh_df, best_f2_row, best_cost_row, workload_30


def run_tiered_deployment(df, mean_thresh, tier_high=0.65):
    """Classify emails into HIGH/MEDIUM/LOW tiers and report catch rates."""
    print("\n" + "=" * 70)
    print("RECOMMENDED TIERED DEPLOYMENT")
    print("=" * 70)

    y_true = df['escalated'].values
    y_probs = df['y_prob'].values

    print(f"\nProposed three-tier system:")
    print(f"  HIGH RISK   (p >= {tier_high:.2f}): Immediate escalation to senior agent")
    print(f"  MEDIUM RISK ({mean_thresh:.2f} <= p < {tier_high:.2f}): Priority review queue")
    print(f"  LOW RISK    (p < {mean_thresh:.2f}): Standard handling")

    high_mask = y_probs >= tier_high
    medium_mask = (y_probs >= mean_thresh) & (y_probs < tier_high)
    low_mask = y_probs < mean_thresh

    tier_stats = []
    for tier_name, mask in [('HIGH', high_mask), ('MEDIUM', medium_mask), ('LOW', low_mask)]:
        n = mask.sum()
        n_pos = y_true[mask].sum()
        prevalence = n_pos / n * 100 if n > 0 else 0

        tier_stats.append({
            'tier': tier_name, 'n': int(n), 'pct': n / len(y_true) * 100,
            'n_escalated': int(n_pos), 'prevalence': prevalence
        })

        print(f"\n  {tier_name:6s}: {n:5d} emails ({n/len(y_true)*100:.1f}%), "
              f"{int(n_pos)} escalations ({prevalence:.1f}% prevalence)")

    high_catch = y_true[high_mask].sum()
    med_catch = y_true[medium_mask].sum()
    low_miss = y_true[low_mask].sum()
    total_pos = y_true.sum()

    print(f"\n  Escalation distribution across tiers:")
    print(f"    HIGH:   {int(high_catch)} / {int(total_pos)} ({high_catch/total_pos*100:.1f}%) caught immediately")
    print(f"    MEDIUM: {int(med_catch)} / {int(total_pos)} ({med_catch/total_pos*100:.1f}%) caught in priority review")
    print(f"    LOW:    {int(low_miss)} / {int(total_pos)} ({low_miss/total_pos*100:.1f}%) missed (standard handling)")
    print(f"    HIGH+MEDIUM combined catch rate: {(high_catch+med_catch)/total_pos*100:.1f}%")

    tier_df = pd.DataFrame(tier_stats)
    return tier_df, high_mask, medium_mask, low_mask, high_catch, med_catch, low_miss, total_pos