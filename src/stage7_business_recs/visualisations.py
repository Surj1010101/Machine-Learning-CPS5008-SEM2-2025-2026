"""Visualisations module."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_threshold_tradeoffs(thresh_df, mean_thresh, best_cost_row, output_path):
    """
    Three-panel figure: metrics vs threshold, workload vs threshold, cost vs threshold.

    This is the headline trade-off picture, the aim is to put F2/recall
    /precision in panel one, workload percentage in panel two, and total business cost
    in panel three. This also marks on each panel the current threshold and
    where applicable the cost-optimal threshold, so the stakeholder can see the gap.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1, F2/recall/precision curves with current threshold marked
    axes[0].plot(thresh_df['threshold'], thresh_df['f2'], 'b-o', markersize=3,
                 label='F2', linewidth=2)
    axes[0].plot(thresh_df['threshold'], thresh_df['recall'], 'g--s', markersize=3,
                 label='Recall', linewidth=1.5)
    axes[0].plot(thresh_df['threshold'], thresh_df['precision'], 'r--^', markersize=3,
                 label='Precision', linewidth=1.5)
    axes[0].axvline(x=mean_thresh, color='black', linestyle=':', alpha=0.7,
                    label=f'Current threshold ({mean_thresh:.2f})')
    axes[0].set_xlabel('Classification Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Metrics vs Threshold', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0.1, 0.85)
    axes[0].grid(alpha=0.3)

    # Panel 2, workload curve with the 30% target marked
    axes[1].plot(thresh_df['threshold'], thresh_df['flagged_pct'], 'purple',
                 linewidth=2, marker='o', markersize=3)
    axes[1].axvline(x=mean_thresh, color='black', linestyle=':', alpha=0.7)
    axes[1].axhline(y=30, color='orange', linestyle='--', alpha=0.5,
                    label='30% workload target')
    axes[1].set_xlabel('Classification Threshold')
    axes[1].set_ylabel('Emails Flagged (%)')
    axes[1].set_title('Workload vs Threshold', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0.1, 0.85)
    axes[1].grid(alpha=0.3)

    # Panel 3, business cost curve with both current and cost-optimal thresholds marked
    axes[2].plot(thresh_df['threshold'], thresh_df['cost'] / 1000, 'darkgreen',
                 linewidth=2, marker='o', markersize=3)
    axes[2].axvline(x=mean_thresh, color='black', linestyle=':', alpha=0.7,
                    label=f'Current ({mean_thresh:.2f})')
    axes[2].axvline(x=best_cost_row['threshold'], color='red', linestyle='--',
                    alpha=0.7, label=f"Cost-optimal ({best_cost_row['threshold']:.2f})")
    axes[2].set_xlabel('Classification Threshold')
    axes[2].set_ylabel('Total Cost (GBP  thousands)')
    axes[2].set_title('Business Cost vs Threshold', fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].set_xlim(0.1, 0.85)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_tiered_deployment(high_mask, medium_mask, low_mask,
                           high_catch, med_catch, low_miss, y_true,
                           output_path):
    """
    Stacked bar chart of tier email distribution next to a pie chart of catch by tier.

    This is the deployment picture, the aim is that the bar chart shows
    how my emails get distributed across tiers and the pie chart shows where my actual
    escalations end up being caught. This also annotates the prevalence inside
    each tier so the stakeholder can see how concentrated the HIGH tier really is.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tier_labels = ['HIGH\n(Immediate)', 'MEDIUM\n(Priority)', 'LOW\n(Standard)']
    tier_counts = [high_mask.sum(), medium_mask.sum(), low_mask.sum()]
    tier_esc = [y_true[high_mask].sum(), y_true[medium_mask].sum(), y_true[low_mask].sum()]
    tier_noesc = [c - e for c, e in zip(tier_counts, tier_esc)]

    x = np.arange(len(tier_labels))
    width = 0.5
    axes[0].bar(x, tier_esc, width, label='Escalated', color='#E53935', alpha=0.85)
    axes[0].bar(x, tier_noesc, width, bottom=tier_esc,
                label='Not Escalated', color='#42A5F5', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tier_labels)
    axes[0].set_ylabel('Number of Emails')
    axes[0].set_title('Tiered Deployment: Email Distribution', fontweight='bold')
    axes[0].legend()

    for i, (esc, total) in enumerate(zip(tier_esc, tier_counts)):
        prev = esc / total * 100 if total > 0 else 0
        axes[0].text(i, total + 15, f'{prev:.0f}% esc.\nn={total}',
                     ha='center', fontsize=9)

    # Pie chart showing where the actual escalations get caught
    total_pos = y_true.sum()
    catch_pcts = [high_catch / total_pos * 100,
                  med_catch / total_pos * 100,
                  low_miss / total_pos * 100]
    colors_tier = ['#E53935', '#FF9800', '#66BB6A']
    labels_tier = [f'HIGH ({catch_pcts[0]:.0f}%)',
                   f'MEDIUM ({catch_pcts[1]:.0f}%)',
                   f'LOW / missed ({catch_pcts[2]:.0f}%)']

    axes[1].pie(
        [high_catch, med_catch, low_miss],
        labels=labels_tier, colors=colors_tier, autopct='%1.0f%%',
        startangle=90, pctdistance=0.6, textprops={'fontsize': 10}
    )
    axes[1].set_title(f'Escalation Catch by Tier\n(n={int(total_pos)} total)',
                     fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_stability_and_costs(temp_df, f2_std, cost_sens_df, breakeven, output_path):
    """
    Two-panel figure with temporal stability on the left and cost sensitivity on the right.

    This is the stability picture, the aim is that the left panel shows
    F2 over time with prevalence overlaid, and the right panel shows how my model cost
    compares against flag-all and no-model under different FN cost assumptions. What
    this also marks the breakeven FN cost where my model becomes preferable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left, F2 over quarters with prevalence as a secondary bar
    axes[0].plot(range(len(temp_df)), temp_df['f2'], 'b-o', label='F2', linewidth=2)
    axes[0].fill_between(range(len(temp_df)),
                         temp_df['f2'] - f2_std, temp_df['f2'] + f2_std,
                         alpha=0.2, color='blue')
    ax_twin = axes[0].twinx()
    ax_twin.bar(range(len(temp_df)), temp_df['prevalence'], alpha=0.3,
                color='orange', label='Prevalence %')
    ax_twin.set_ylabel('Prevalence (%)', color='orange')

    axes[0].set_xticks(range(len(temp_df)))
    axes[0].set_xticklabels(temp_df['period'], rotation=45, ha='right', fontsize=7)
    axes[0].set_ylabel('F2-Score', color='blue')
    axes[0].set_title('Model Performance Over Time', fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=8)
    ax_twin.legend(loc='upper right', fontsize=8)

    # Right, three cost curves vs FN cost so the breakeven is visually obvious
    axes[1].plot(cost_sens_df['fn_cost'], cost_sens_df['model_cost'] / 1000,
                 'b-o', label='Current Model', linewidth=2)
    axes[1].plot(cost_sens_df['fn_cost'], cost_sens_df['flag_all_cost'] / 1000,
                 'r--s', label='Flag Everything', linewidth=1.5)
    axes[1].plot(cost_sens_df['fn_cost'], cost_sens_df['no_model_cost'] / 1000,
                 'g--^', label='No Model', linewidth=1.5)
    axes[1].set_xlabel('False Negative Cost (GBP )')
    axes[1].set_ylabel('Total Cost (GBP  thousands)')
    axes[1].set_title('Cost Sensitivity to FN Cost', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    if breakeven:
        axes[1].axvline(x=breakeven, color='purple', linestyle=':',
                        alpha=0.7, label=f'Breakeven ~GBP {breakeven}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")




