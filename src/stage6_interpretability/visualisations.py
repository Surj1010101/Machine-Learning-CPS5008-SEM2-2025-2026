"""Stage 6 plots: feature importance, permutation importance, fairness, LIME."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_feature_importance_coefficients(top_escalation, top_deescalation, output_path):
    """Side-by-side: top escalation words (positive coefs) vs de-escalation words (negative)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    esc_plot = top_escalation.head(15).sort_values('coefficient')
    axes[0].barh(esc_plot['clean_name'], esc_plot['coefficient'],
                 color='#E53935', alpha=0.85)
    axes[0].set_xlabel('LR Coefficient')
    axes[0].set_title('Top 15 Escalation Indicators', fontweight='bold')
    axes[0].axvline(x=0, color='grey', linestyle='-', alpha=0.3)

    deesc_plot = top_deescalation.head(15).sort_values('coefficient', ascending=False)
    axes[1].barh(deesc_plot['clean_name'], deesc_plot['coefficient'],
                 color='#43A047', alpha=0.85)
    axes[1].set_xlabel('LR Coefficient')
    axes[1].set_title('Top 15 De-escalation Indicators', fontweight='bold')
    axes[1].axvline(x=0, color='grey', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_permutation_importance(perm_df, output_path):
    """Bar chart of mean F2 decrease when each feature is shuffled."""
    fig, ax = plt.subplots(figsize=(10, 6))

    perm_plot = perm_df[perm_df['importance_mean'].abs() > 0.001].sort_values('importance_mean')
    if len(perm_plot) == 0:
        perm_plot = perm_df.sort_values('importance_mean')

    colors_perm = ['#E53935' if v > 0 else '#90A4AE' for v in perm_plot['importance_mean']]
    ax.barh(perm_plot['feature'], perm_plot['importance_mean'],
            xerr=perm_plot['importance_std'], color=colors_perm, alpha=0.85,
            capsize=3)
    ax.set_xlabel('Mean F2 Decrease When Shuffled')
    ax.set_title('Permutation Importance (Validation Set)', fontweight='bold')
    ax.axvline(x=0, color='grey', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_fairness(fairness_df, output_path):
    """3 panel TPR vs FPR comparison across region, customer type, tenure type."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax_idx, (attr, title) in enumerate([
        ('region', 'Region'), ('customer_type', 'Customer Type'),
        ('tenure_type', 'Tenure Type')
    ]):
        attr_data = fairness_df[fairness_df['attribute'] == attr]
        x = np.arange(len(attr_data))
        width = 0.35

        ax = axes[ax_idx]
        bars1 = ax.bar(x - width/2, attr_data['tpr'], width, label='TPR (Recall)',
                       color='#1976D2', alpha=0.85)
        bars2 = ax.bar(x + width/2, attr_data['fpr'], width, label='FPR',
                       color='#FF7043', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(attr_data['group'], rotation=15)
        ax.set_ylabel('Rate')
        ax.set_title(f'Equalised Odds: {title}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{bar.get_height():.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_lime_examples(lime_results, output_path):
    """Twopanel bar chart for the first two LIME explanations."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for plot_idx, ex in enumerate(lime_results[:2]):
        ax = axes[plot_idx]
        features = [f[0][:40] for f in ex['top_features']][::-1]
        weights = [f[1] for f in ex['top_features']][::-1]
        colors_lime = ['#E53935' if w > 0 else '#43A047' for w in weights]

        ax.barh(features, weights, color=colors_lime, alpha=0.85)
        ax.axvline(x=0, color='grey', linestyle='-', alpha=0.3)
        ax.set_xlabel('Feature Contribution')
        ptype = ex['type']
        prob = ex['predicted_prob']
        ax.set_title(f"LIME: {ptype} (p={prob:.3f})\n\"{ex['email'][:60]}...\"",
                     fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")