"""
Stage 3b visualisations module, ablation bar chart and multicollinearity heatmaps.

Overall this module is where I generate the figures for my Stage 3b section, the basic
idea is to turn the ablation table and the multicollinearity matrices into plots that go
straight into the report. In my project I focused on two figures here, the combined
ablation impact and numeric correlation heatmap, and the Cramer's V heatmap on the
categorical features.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ablation_and_correlation(all_results, result_full, corr_matrix, output_path):
    """
    Combined 1 by 2 figure with my feature impact bar chart on the left and the numeric
    correlation heatmap on the right.

    Overall this is the headline Stage 3b figure, the basic idea is to put the F2 impact
    of each ablation configuration next to the correlation matrix so the report can
    discuss feature engineering and multicollinearity in one go. What this also does is
    highlight the Full pipeline configuration in a different colour as the reference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    configs = [r['config'] for r in all_results]
    f2_means = [r['f2_mean'] for r in all_results]
    f2_stds = [r['f2_std'] for r in all_results]

    # Sorting configs from best F2 to worst so the bar chart reads top-down
    sort_idx = np.argsort(f2_means)[::-1]
    configs_sorted = [configs[i] for i in sort_idx]
    f2_sorted = [f2_means[i] for i in sort_idx]
    std_sorted = [f2_stds[i] for i in sort_idx]

    colors = ['#2196F3' if c == 'Full pipeline' else '#90CAF9' for c in configs_sorted]
    axes[0].barh(configs_sorted, f2_sorted, xerr=std_sorted, color=colors, capsize=4)
    axes[0].set_xlabel('F2-Score (tuned threshold)')
    axes[0].set_title('Feature Engineering Impact on F2', fontweight='bold')
    axes[0].axvline(x=result_full['f2_mean'], color='red', linestyle='--', alpha=0.5,
                    label=f'Full pipeline ({result_full["f2_mean"]:.3f})')
    axes[0].legend(fontsize=9)
    axes[0].invert_yaxis()

    # Numeric correlation heatmap on the right side
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.3f',
                ax=axes[1], vmin=-1, vmax=1, square=True)
    axes[1].set_title('Numeric Feature Correlations', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def plot_cramers_v_heatmap(cramers_df, cat_features, output_path):
    """
    Heatmap of Cramer's V values between my categorical features.

    Overall this is the visual version of the categorical multicollinearity table, the
    basic idea is that the eye picks up high values quickly. What this also does is
    fill the diagonal with 1.0 so the heatmap reads naturally as a similarity matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cramers_matrix = pd.DataFrame(0.0, index=cat_features, columns=cat_features)
    for _, row in cramers_df.iterrows():
        cramers_matrix.loc[row['Feature 1'], row['Feature 2']] = row['Cramers V']
        cramers_matrix.loc[row['Feature 2'], row['Feature 1']] = row['Cramers V']
    diag_vals = cramers_matrix.values.copy()
    np.fill_diagonal(diag_vals, 1.0)
    cramers_matrix = pd.DataFrame(diag_vals, index=cat_features, columns=cat_features)

    sns.heatmap(cramers_matrix, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax,
                vmin=0, vmax=1, square=True)
    ax.set_title("Cramer's V: Categorical Feature Independence", fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
