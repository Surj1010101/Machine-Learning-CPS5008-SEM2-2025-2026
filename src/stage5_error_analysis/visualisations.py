"""
Stage 5 visualisations module: error overview, calibration/cost and FN deep dive.

Overall this module is where I generate the three figures for my Stage 5 report
section, the basic idea is to turn the error analysis numbers into pictures the
marker can read at a glance. In my project I focused on three plots here, the
overall error overview (confusion matrix + probability distributions + segment
recall + per-issue F2), the calibration plot next to the business cost comparison,
and the false negative deep dive showing miss rate per escalation level.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_error_overview(df, seg_df, fn_all, fp_all, tn_all, tp_all,
                        mean_thresh, overall_f2, output_path):
    """
    Overall 2 by 2 figure with confusion matrix, probability distribution by outcome,
    recall by region and customer type, and F2 by issue category.

    Overall this is the headline Stage 5 figure, the basic idea is to put four
    different angles on my error analysis into one image so the report can reference
    everything in one place. What this also annotates is the threshold line on the
    probability distribution panel, so the gap between FN and TP is visually obvious.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1, confusion matrix as a heatmap with cell counts and percentages
    cm_display = np.array([[tn_all, fp_all], [fn_all, tp_all]])
    axes[0, 0].imshow(cm_display, cmap='Blues', interpolation='nearest')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_xticklabels(['Not Escalated', 'Escalated'])
    axes[0, 0].set_yticklabels(['Not Escalated', 'Escalated'])
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_title('Confusion Matrix (All Folds)', fontweight='bold')
    for i in range(2):
        for j in range(2):
            val = cm_display[i, j]
            pct = val / cm_display.sum() * 100
            axes[0, 0].text(j, i, f'{val}\n({pct:.1f}%)',
                            ha='center', va='center', fontsize=12,
                            color='white' if val > cm_display.max()/2 else 'black')

    # Panel 2, predicted probability distribution split by outcome type
    for ptype, color, label in [('TP', '#4CAF50', 'True Pos'),
                                 ('FP', '#FF9800', 'False Pos'),
                                 ('FN', '#F44336', 'False Neg'),
                                 ('TN', '#2196F3', 'True Neg')]:
        vals = df[df['prediction_type'] == ptype]['y_prob']
        if len(vals) > 0:
            axes[0, 1].hist(vals, bins=30, alpha=0.5, label=f'{label} (n={len(vals)})',
                            color=color, density=True)
    axes[0, 1].axvline(x=mean_thresh, color='black', linestyle='--', alpha=0.7,
                       label=f'Mean threshold ({mean_thresh:.3f})')
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Probability Distribution by Outcome', fontweight='bold')
    axes[0, 1].legend(fontsize=8)

    # Panel 3, recall by region and customer_type combined into one chart
    region_seg = seg_df[seg_df['segment'] == 'region'].sort_values('recall')
    ctype_seg = seg_df[seg_df['segment'] == 'customer_type'].sort_values('recall')
    combined_seg = pd.concat([region_seg, ctype_seg])

    bars = axes[1, 0].barh(
        [f"{r['segment']}: {r['category']}" for _, r in combined_seg.iterrows()],
        combined_seg['recall'],
        color=['#5C6BC0' if s == 'region' else '#26A69A'
               for s in combined_seg['segment']],
        alpha=0.85
    )
    axes[1, 0].axvline(x=overall_f2, color='red', linestyle='--', alpha=0.5,
                        label='Overall recall')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_title('Recall by Region & Customer Type', fontweight='bold')
    axes[1, 0].set_xlim(0, 1)
    for bar, val in zip(bars, combined_seg['recall']):
        axes[1, 0].text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=9)

    # Panel 4, F2 by issue category, this is where category-level performance lives
    issue_seg = seg_df[seg_df['segment'] == 'issue_category'].sort_values('f2')
    bars2 = axes[1, 1].barh(issue_seg['category'], issue_seg['f2'],
                             color='#7E57C2', alpha=0.85)
    axes[1, 1].set_xlabel('F2-Score')
    axes[1, 1].set_title('F2-Score by Issue Category', fontweight='bold')
    axes[1, 1].set_xlim(0, max(issue_seg['f2'].max() * 1.3, 0.5))
    for bar, val in zip(bars2, issue_seg['f2']):
        axes[1, 1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_calibration_and_cost(cal_table, cost_data, output_path):
    """
    Side by side calibration plot and business cost bar chart.

    Overall this is the second Stage 5 figure, the basic idea is to put my reliability
    diagram next to the four-scenario cost comparison so the report can show probability
    quality and business value in one shot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Calibration plot, only showing bins with at least 10 samples to avoid noise
    cal_valid = cal_table[cal_table['n'] >= 10].copy()
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    axes[0].scatter(cal_valid['mean_prob'], cal_valid['actual_rate'],
                    s=cal_valid['n'] * 0.5, alpha=0.7, color='#1976D2', zorder=5)
    for _, row in cal_valid.iterrows():
        axes[0].annotate(f"n={int(row['n'])}", (row['mean_prob'], row['actual_rate']),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Actual Positive Rate')
    axes[0].set_title('Calibration Plot', fontweight='bold')
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # Business cost bar chart, four scenarios from worst to best
    scenarios = ['No Model\n(flag nothing)', 'Flag\nEverything',
                 'Current\nModel', 'Perfect\nModel']
    costs = [cost_data['no_model_cost'], cost_data['flag_all_cost'],
             cost_data['model_cost'], cost_data['perfect_cost']]
    colors_cost = ['#F44336', '#FF9800', '#4CAF50', '#2196F3']
    bars = axes[1].bar(scenarios, costs, color=colors_cost, alpha=0.85)
    axes[1].set_ylabel('Total Cost (£)')
    axes[1].set_title('Business Cost Comparison', fontweight='bold')
    for bar, cost in zip(bars, costs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f'£{cost:,.0f}', ha='center', va='bottom', fontsize=10,
                     fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_fn_deep_dive(df, fn_df, tp_df, mean_thresh, output_path):
    """
    FN probability histogram next to miss rate by escalation level.

    Overall this is the final Stage 5 figure, the basic idea is to zoom into just the
    false negatives, the basic idea is that this is where my model loses the most
    business value, so the report needs a dedicated picture of it. What this also
    breaks down is whether my misses are concentrated at borderline level 2 cases or
    whether even level 3 cases are slipping through.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Probability histogram comparing FN against TP, with threshold marked
    axes[0].hist(fn_df['y_prob'], bins=20, color='#F44336', alpha=0.7,
                 edgecolor='white', label=f'False Negatives (n={len(fn_df)})')
    axes[0].hist(tp_df['y_prob'], bins=20, color='#4CAF50', alpha=0.5,
                 edgecolor='white', label=f'True Positives (n={len(tp_df)})')
    axes[0].axvline(x=mean_thresh, color='black', linestyle='--',
                     label=f'Mean threshold ({mean_thresh:.3f})')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Probability Distribution: Escalated Emails', fontweight='bold')
    axes[0].legend(fontsize=9)

    # Miss rate per escalation level for the levels that count as positives (2 and 3)
    levels_in_target = [2, 3]
    miss_data = []
    for level in levels_in_target:
        level_mask = df['escalation_level'] == level
        level_df = df[level_mask]
        missed = (level_df['y_pred'] == 0).sum()
        caught = (level_df['y_pred'] == 1).sum()
        miss_data.append({'level': f'Level {level}', 'missed': missed,
                          'caught': caught, 'total': missed + caught,
                          'miss_rate': missed / (missed + caught) * 100})

    miss_df = pd.DataFrame(miss_data)
    x_pos = range(len(miss_df))
    width = 0.35
    axes[1].bar([p - width/2 for p in x_pos], miss_df['caught'],
                width, label='Caught (TP)', color='#4CAF50', alpha=0.85)
    axes[1].bar([p + width/2 for p in x_pos], miss_df['missed'],
                width, label='Missed (FN)', color='#F44336', alpha=0.85)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(miss_df['level'])
    axes[1].set_ylabel('Count')
    axes[1].set_title('Catch Rate by Escalation Level', fontweight='bold')
    axes[1].legend()
    for i, row in miss_df.iterrows():
        axes[1].text(i, max(row['caught'], row['missed']) + 3,
                     f"{row['miss_rate']:.0f}% missed", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    return miss_df
