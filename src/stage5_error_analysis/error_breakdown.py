"""False negative and false positive deep dives."""

import numpy as np


def analyse_false_negatives(df, fold_metrics):
    """Characterise the missed escalations (FN) vs correctly caught ones (TP)."""
    print("\n" + "=" * 70)
    print("FALSE NEGATIVE ANALYSIS (Missed Escalations)")
    print("=" * 70)

    fn_df = df[df['prediction_type'] == 'FN'].copy()
    tp_df = df[df['prediction_type'] == 'TP'].copy()

    total_pos = len(fn_df) + len(tp_df)
    print(f"\n{len(fn_df)} false negatives (missed escalations) out of "
          f"{total_pos} total escalations ({len(fn_df)/total_pos*100:.1f}% missed)")

    print(f"\nPredicted probability distribution:")
    print(f"  FN (missed): mean={fn_df['y_prob'].mean():.3f}, "
          f"median={fn_df['y_prob'].median():.3f}, "
          f"max={fn_df['y_prob'].max():.3f}")
    print(f"  TP (caught): mean={tp_df['y_prob'].mean():.3f}, "
          f"median={tp_df['y_prob'].median():.3f}, "
          f"min={tp_df['y_prob'].min():.3f}")

    mean_thresh = np.mean([m['threshold'] for m in fold_metrics])
    close_fn = fn_df[fn_df['y_prob'] >= mean_thresh - 0.05]
    print(f"\n  FNs within 0.05 of mean threshold ({mean_thresh:.3f}): "
          f"{len(close_fn)} ({len(close_fn)/len(fn_df)*100:.1f}%)")

    print(f"\nFalse Negative characteristics vs True Positives:")
    for col in ['customer_type', 'tenure_type', 'meter_type', 'region',
                'issue_category', 'sentiment']:
        fn_dist = fn_df[col].value_counts(normalize=True).sort_index()
        tp_dist = tp_df[col].value_counts(normalize=True).sort_index()
        print(f"\n  {col}:")
        all_cats = sorted(set(fn_dist.index) | set(tp_dist.index))
        for cat in all_cats:
            fn_pct = fn_dist.get(cat, 0) * 100
            tp_pct = tp_dist.get(cat, 0) * 100
            diff = fn_pct - tp_pct
            marker = " ***" if abs(diff) > 10 else ""
            print(f"    {cat:20s}: FN={fn_pct:5.1f}%  TP={tp_pct:5.1f}%  "
                  f"diff={diff:+5.1f}%{marker}")

    print(f"\nFalse Negatives by original escalation_level:")
    fn_esc = fn_df['escalation_level'].value_counts().sort_index()
    tp_esc = tp_df['escalation_level'].value_counts().sort_index()
    for level in sorted(set(fn_esc.index) | set(tp_esc.index)):
        fn_c = fn_esc.get(level, 0)
        tp_c = tp_esc.get(level, 0)
        total = fn_c + tp_c
        miss_rate = fn_c / total * 100 if total > 0 else 0
        print(f"  Level {level}: {fn_c} missed / {total} total ({miss_rate:.1f}% miss rate)")

    print(f"\nSample FALSE NEGATIVE emails (missed escalations):")
    for i, row in fn_df.head(10).iterrows():
        print(f"  [{row['escalation_level']}] p={row['y_prob']:.3f}: \"{row['email_body_text']}\"")

    return fn_df, tp_df, mean_thresh, close_fn


def analyse_false_positives(df):
    """characterise the unnecessary alerts FP vs correct non-alerts TN."""
    print("\n" + "=" * 70)
    print("FALSE POSITIVE ANALYSIS (Unnecessary Alerts)")
    print("=" * 70)

    fp_df = df[df['prediction_type'] == 'FP'].copy()
    tn_df = df[df['prediction_type'] == 'TN'].copy()

    total_neg = len(fp_df) + len(tn_df)
    print(f"\n{len(fp_df)} false positives out of {total_neg} "
          f"non-escalations ({len(fp_df)/total_neg*100:.1f}% false alarm rate)")

    print(f"\nPredicted probability distribution:")
    print(f"  FP (false alarm): mean={fp_df['y_prob'].mean():.3f}, "
          f"median={fp_df['y_prob'].median():.3f}, "
          f"max={fp_df['y_prob'].max():.3f}")
    print(f"  TN (correct):     mean={tn_df['y_prob'].mean():.3f}, "
          f"median={tn_df['y_prob'].median():.3f}, "
          f"max={tn_df['y_prob'].max():.3f}")

    print(f"\nFalse Positive characteristics vs True Negatives:")
    for col in ['issue_category', 'sentiment']:
        fp_dist = fp_df[col].value_counts(normalize=True).sort_index()
        tn_dist = tn_df[col].value_counts(normalize=True).sort_index()
        print(f"\n  {col}:")
        all_cats = sorted(set(fp_dist.index) | set(tn_dist.index))
        for cat in all_cats:
            fp_pct = fp_dist.get(cat, 0) * 100
            tn_pct = tn_dist.get(cat, 0) * 100
            diff = fp_pct - tn_pct
            marker = " ***" if abs(diff) > 5 else ""
            print(f"    {cat:20s}: FP={fp_pct:5.1f}%  TN={tn_pct:5.1f}%  "
                  f"diff={diff:+5.1f}%{marker}")

    print(f"\nSample FALSE POSITIVE emails (unnecessary alerts):")
    for i, row in fp_df.head(10).iterrows():
        print(f"  [{row['escalation_level']}] p={row['y_prob']:.3f}: \"{row['email_body_text']}\"")

    return fp_df, tn_df
