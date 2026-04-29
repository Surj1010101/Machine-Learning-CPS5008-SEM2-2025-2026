"""Combined 2x3 figure for Stage 4b supplementary analyses."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_supplementary_analyses(knn_df, reg_df, train_sizes_abs, train_scores,
                                val_scores, pca_df, cumvar, cluster_df,
                                linreg_df, output_path):
    """One combined figure with 6 panels summarising Stage 4b."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Stage 4b: Supplementary ML Analyses', fontsize=14, fontweight='bold')

    # Panel 1: KNN comparison
    ax = axes[0, 0]
    ax.bar(knn_df['k'].astype(str), knn_df['f2_mean'],
           yerr=knn_df['f2_std'], capsize=4, color='steelblue', alpha=0.8)
    ax.axhline(y=0.360, color='red', linestyle='--', label='LR baseline (0.360)')
    ax.set_xlabel('K (neighbours)')
    ax.set_ylabel('F2 Score')
    ax.set_title('KNN: Effect of K')
    ax.legend(fontsize=8)

    #2Regularisation comparison
    ax = axes[0, 1]
    configs = [r.replace('LR_', '') for r in reg_df['config']]
    colors = ['#2196F3' if 'L2' in c else '#FF9800' if 'L1' in c else '#4CAF50'
              for c in reg_df['config']]
    ax.barh(configs, reg_df['f2_mean'], xerr=reg_df['f2_std'],
            capsize=4, color=colors, alpha=0.8)
    ax.axvline(x=0.360, color='red', linestyle='--', label='Main LR (0.360)')
    ax.set_xlabel('F2 Score')
    ax.set_title('Regularisation: L1 vs L2 vs None')
    ax.legend(fontsize=8)

    #3Learning curves (bias-variance)
    ax = axes[0, 2]
    ax.plot(train_sizes_abs, train_scores.mean(axis=1), 'o-',
            label='Training', color='steelblue')
    ax.fill_between(train_sizes_abs,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.1, color='steelblue')
    ax.plot(train_sizes_abs, val_scores.mean(axis=1), 'o-',
            label='Validation', color='darkorange')
    ax.fill_between(train_sizes_abs,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1),
                    alpha=0.1, color='darkorange')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('F1 Score')
    ax.set_title('Bias-Variance: Learning Curves')
    ax.legend(fontsize=8)

    #4PCA explained variance + F2
    ax = axes[1, 0]
    ax2 = ax.twinx()
    n_plot = min(100, len(cumvar))
    ax.plot(range(1, n_plot + 1), cumvar[:n_plot], color='steelblue',
            label='Cumulative variance')
    ax.axhline(y=0.90, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance', color='steelblue')
    ax2.plot(pca_df['n_components'], pca_df['f2_mean'], 'o-',
             color='darkorange', label='F2 score')
    ax2.set_ylabel('F2 Score', color='darkorange')
    ax.set_title('PCA: Variance & Performance')

    #5 K-Means silhouette scores
    ax = axes[1, 1]
    ax.plot(cluster_df['k'], cluster_df['silhouette'], 'o-',
            color='steelblue', label='Silhouette')
    ax2 = ax.twinx()
    ax2.plot(cluster_df['k'], cluster_df['esc_rate_range'] * 100, 's--',
             color='darkorange', label='Esc. rate range')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score', color='steelblue')
    ax2.set_ylabel('Escalation Rate Range (%)', color='darkorange')
    ax.set_title('K-Means: Cluster Quality')

    # 6:Linear vs Logistic regression
    ax = axes[1, 2]
    models = ['Linear Reg\n(inappropriate)', 'Logistic Reg\n(appropriate)']
    f2_vals = [linreg_df['f2'].mean(), 0.360]
    colors = ['#ef5350', '#66bb6a']
    ax.bar(models, f2_vals, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('F2 Score')
    ax.set_title('Linear vs Logistic Regression')
    ax.annotate(f"Preds outside [0,1]:\n~{linreg_df['preds_below_0'].mean():.0f} below 0\n"
                f"~{linreg_df['preds_above_1'].mean():.0f} above 1",
                xy=(0, f2_vals[0]), xytext=(0.3, max(f2_vals) * 0.7),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
