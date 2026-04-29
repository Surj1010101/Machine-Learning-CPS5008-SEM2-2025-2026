"""Temporal stability check across quarterly periods."""

import pandas as pd
from sklearn.metrics import fbeta_score, confusion_matrix


def run_temporal_stability(df):
    """Check if model performance varies meaningfully over time."""
    print("\n" + "=" * 70)
    print("TEMPORAL STABILITY CHECK")
    print("=" * 70)

    df['year_month'] = df['timestamp'].dt.to_period('Q')

    temporal_stats = []
    for period in sorted(df['year_month'].unique()):
        mask = df['year_month'] == period
        sub = df[mask]
        if sub['escalated'].sum() < 5:
            continue

        y_t = sub['escalated'].values
        y_p = sub['y_pred'].values
        y_pr = sub['y_prob'].values

        f2 = fbeta_score(y_t, y_p, beta=2)
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        temporal_stats.append({
            'period': str(period), 'n': len(sub), 'n_pos': int(y_t.sum()),
            'prevalence': y_t.mean() * 100, 'f2': f2, 'recall': recall,
            'mean_prob': y_pr.mean()
        })

    temp_df = pd.DataFrame(temporal_stats)
    print(f"\n{'Period':<10s} {'N':>5s} {'Pos':>4s} {'Prev%':>6s} {'F2':>6s} "
          f"{'Recall':>7s} {'MeanProb':>8s}")
    print("-" * 50)
    for _, row in temp_df.iterrows():
        print(f"{row['period']:<10s} {int(row['n']):>5d} {int(row['n_pos']):>4d} "
              f"{row['prevalence']:>5.1f}% {row['f2']:>6.3f} {row['recall']:>7.3f} "
              f"{row['mean_prob']:>8.3f}")

    f2_std = temp_df['f2'].std()
    recall_std = temp_df['recall'].std()
    prev_std = temp_df['prevalence'].std()
    print(f"\nTemporal variation:")
    print(f"  F2 std across quarters:         {f2_std:.3f}")
    print(f"  Recall std across quarters:     {recall_std:.3f}")
    print(f"  Prevalence std across quarters: {prev_std:.3f}%")

    if f2_std < 0.05 and prev_std < 3.0:
        print("  --> Performance is STABLE across time periods")
    else:
        print("  --> Notable temporal variation detected -- investigate drift")

    return temp_df, f2_std