"""Cost-sensitivity analysis: how the optimal strategy varies with FN cost."""

import pandas as pd


def run_cost_sensitivity(df, thresh_df):
    """Vary FN cost and report model vs flag-all vs no-model costs."""
    print("\n" + "=" * 70)
    print("COST-SENSITIVITY ANALYSIS")
    print("=" * 70)

    fn_costs = [100, 200, 300, 500, 750, 1000, 2000]
    fp_cost = 20
    tp_cost = 50

    y_true = df['escalated'].values
    tp_all = int(((y_true == 1) & (df['y_pred'] == 1)).sum())
    fp_all = int(((y_true == 0) & (df['y_pred'] == 1)).sum())
    fn_all = int(((y_true == 1) & (df['y_pred'] == 0)).sum())

    cost_sensitivity = []
    for fn_cost in fn_costs:
        model_cost = tp_all * tp_cost + fp_all * fp_cost + fn_all * fn_cost
        flag_all_cost = int(y_true.sum()) * tp_cost + (len(y_true) - int(y_true.sum())) * fp_cost
        no_model_cost = int(y_true.sum()) * fn_cost

        best_t_cost = None
        best_cost = float('inf')
        for _, row in thresh_df.iterrows():
            t_cost = row['tp'] * tp_cost + row['fp'] * fp_cost + row['fn'] * fn_cost
            if t_cost < best_cost:
                best_cost = t_cost
                best_t_cost = row['threshold']

        model_wins = model_cost < flag_all_cost

        cost_sensitivity.append({
            'fn_cost': fn_cost,
            'model_cost': model_cost,
            'flag_all_cost': flag_all_cost,
            'no_model_cost': no_model_cost,
            'optimal_threshold': best_t_cost,
            'optimal_cost': int(best_cost),
            'model_beats_flag_all': model_wins
        })

    cost_sens_df = pd.DataFrame(cost_sensitivity)
    print(f"\n{'FN Cost':>8s} {'Model':>10s} {'Flag All':>10s} {'No Model':>10s} "
          f"{'Opt Thresh':>10s} {'Opt Cost':>10s} {'Model Wins':>10s}")
    print("-" * 72)
    for _, row in cost_sens_df.iterrows():
        print(f"£{int(row['fn_cost']):>6d} £{int(row['model_cost']):>9,d} "
              f"£{int(row['flag_all_cost']):>9,d} £{int(row['no_model_cost']):>9,d} "
              f"{row['optimal_threshold']:>10.2f} £{int(row['optimal_cost']):>9,d} "
              f"{'YES' if row['model_beats_flag_all'] else 'NO':>10s}")

    breakeven = None
    for _, row in cost_sens_df.iterrows():
        if row['model_beats_flag_all']:
            breakeven = row['fn_cost']
            break

    if breakeven:
        print(f"\nModel becomes cost-effective vs flag-all when FN cost <= £{int(breakeven)}")
    else:
        print(f"\nModel does not beat flag-all at any tested FN cost level")
        for fn_c in range(10, 500, 10):
            mc = tp_all * tp_cost + fp_all * fp_cost + fn_all * fn_c
            fac = y_true.sum() * tp_cost + (len(y_true) - y_true.sum()) * fp_cost
            if mc < fac:
                print(f"  Approximate breakeven: FN cost ~£{fn_c}")
                breakeven = fn_c
                break

    return cost_sens_df, breakeven, tp_all, fp_all, fn_all
