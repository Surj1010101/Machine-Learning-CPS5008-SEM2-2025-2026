"""Business module."""

import pandas as pd


def run_business_impact(df, tp_all, fp_all, fn_all, tn_all):
    """
    Compare model cost vs no-model, flag-all and perfect-model scenarios.

    This is where I put numbers on the cost of each error type, the main aim
    is that an FN (missed escalation) costs much more than an FP (false alarm) because
    missed escalations turn into regulatory issues, while false alarms only cost human
    review time. This also reports the operational summary, how many emails
    get flagged out of the total, and what proportion of those flags are real.
    """
    print("\n" + "=" * 70)
    print("BUSINESS IMPACT FRAMING")
    print("=" * 70)

    # Cost assumptions, these mirror the brief's framing of FN being much more expensive
    cost_fn = 500
    cost_fp = 20
    cost_tp = 50
    cost_tn = 0

    print(f"\nAssumed cost structure (per email):")
    print(f"  Missed escalation (FN): GBP {cost_fn}")
    print(f"  False alarm (FP):       GBP {cost_fp}")
    print(f"  Correct flag (TP):      GBP {cost_tp}")
    print(f"  Correct non-flag (TN):  GBP {cost_tn}")

    model_cost = (tp_all * cost_tp + fp_all * cost_fp +
                  fn_all * cost_fn + tn_all * cost_tn)
    no_model_cost = (tp_all + fn_all) * cost_fn
    flag_all_cost = ((tp_all + fn_all) * cost_tp +
                     (fp_all + tn_all) * cost_fp)
    perfect_cost = (tp_all + fn_all) * cost_tp

    print(f"\nCost comparison (across all {len(df)} emails):")
    print(f"  No model (flag nothing):  GBP {no_model_cost:,.0f}")
    print(f"  Flag everything:          GBP {flag_all_cost:,.0f}")
    print(f"  Current model:            GBP {model_cost:,.0f}")
    print(f"  Perfect model:            GBP {perfect_cost:,.0f}")
    print(f"\nModel savings vs no-model: GBP {no_model_cost - model_cost:,.0f} "
          f"({(no_model_cost - model_cost)/no_model_cost*100:.1f}% reduction)")
    print(f"Model savings vs flag-all: GBP {flag_all_cost - model_cost:,.0f} "
          f"({(flag_all_cost - model_cost)/flag_all_cost*100:.1f}% reduction)")

    emails_reviewed = tp_all + fp_all
    print(f"\nOperational summary:")
    print(f"  Emails flagged for review: {emails_reviewed} / {len(df)} "
          f"({emails_reviewed/len(df)*100:.1f}%)")
    print(f"  Of those, true escalations: {tp_all} ({tp_all/emails_reviewed*100:.1f}% precision)")
    print(f"  Escalations caught: {tp_all} / {tp_all+fn_all} "
          f"({tp_all/(tp_all+fn_all)*100:.1f}% recall)")
    print(f"  Escalations missed: {fn_all}")

    return {
        'model_cost': model_cost,
        'no_model_cost': no_model_cost,
        'flag_all_cost': flag_all_cost,
        'perfect_cost': perfect_cost,
        'emails_reviewed': emails_reviewed,
    }


def run_calibration(df):
    """
    Bin my predictions and compare predicted probability with actual escalation rate.

    This is the calibration check, the aim is that a well-calibrated
    model should have its predicted 0.6 probability roughly match an actual 60% positive
    rate inside the bin. This matters because downstream business decisions, if my
    probabilities are uncalibrated then any cost-aware threshold I pick later is built
    on shaky foundations.
    """
    print("\n" + "=" * 70)
    print("CONFIDENCE CALIBRATION")
    print("=" * 70)

    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    df['prob_bin'] = pd.cut(df['y_prob'], bins=bins, include_lowest=True)

    cal_table = df.groupby('prob_bin', observed=True).agg(
        n=('escalated', 'count'),
        n_pos=('escalated', 'sum'),
        mean_prob=('y_prob', 'mean')
    ).reset_index()
    cal_table['actual_rate'] = cal_table['n_pos'] / cal_table['n']

    print(f"\nCalibration table:")
    print(f"  {'Prob Bin':<15} {'N':>6} {'Actual%':>8} {'Predicted%':>10}")
    print("  " + "-" * 42)
    for _, row in cal_table.iterrows():
        print(f"  {str(row['prob_bin']):<15} {row['n']:>6.0f} "
              f"{row['actual_rate']*100:>7.1f}% {row['mean_prob']*100:>9.1f}%")

    return cal_table



