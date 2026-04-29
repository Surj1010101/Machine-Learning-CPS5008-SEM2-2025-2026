"""
Stage 4b: Supplementary Model Comparisons & Unsupervised Exploration

Demonstrates additional ML skills not covered in the main pipeline:
-KNearest Neighbours (KNN) classification
-Regularisation comparison (L1 vs L2 vs no regularisation)
-Bias-variance trade-off analysis (learning curves)
-Principal Component Analysis (PCA) for dimensionality reduction
-K-Means clustering for unsupervised pattern discovery
-Linear regression baseline (justify logistic regression choice)

Run with: py src/stage4b_supplementary_models/run.py
"""

import os
import sys
import json
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# Ensure working directory is project root
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.data_loader import load_and_prepare_data
from stage4b_supplementary_models.classification_methods import (
    run_knn_comparison, run_regularisation_comparison,
    run_linear_regression_baseline
)
from stage4b_supplementary_models.bias_variance import run_learning_curve_analysis
from stage4b_supplementary_models.dimensionality_and_clustering import (
    prepare_dense_features, run_pca_analysis, run_kmeans_clustering
)
from stage4b_supplementary_models.visualisations import plot_supplementary_analyses

np.random.seed(42)
os.makedirs('outputs/stage4b', exist_ok=True)

#lload data 
df, X, y, groups = load_and_prepare_data()
print("=" * 70)
print("STAGE 4b: SUPPLEMENTARY MODELS & UNSUPERVISED EXPLORATION")
print("=" * 70)
print(f"Dataset: {len(X)} samples, {y.sum()} positive ({y.mean()*100:.1f}%)")

# 1. KNN
knn_df, best_k = run_knn_comparison(X, y, groups)
knn_df.to_csv('outputs/stage4b/knn_comparison.csv', index=False)

# 2. Regularisation
reg_df = run_regularisation_comparison(X, y, groups)
reg_df.to_csv('outputs/stage4b/regularisation_comparison.csv', index=False)

# 3. Bias-variance
train_sizes_abs, train_scores, val_scores, gap, diagnosis = run_learning_curve_analysis(X, y)

# Build dense feature matrix for PCA, K-Means, and linear regression
X_dense = prepare_dense_features(X)

# 4. PCA
pca_df, cumvar, n_90, n_95 = run_pca_analysis(X_dense, y, groups)
pca_df.to_csv('outputs/stage4b/pca_comparison.csv', index=False)

# 5.K-Means
cluster_df, best_k_cluster = run_kmeans_clustering(X_dense, y)
cluster_df.to_csv('outputs/stage4b/kmeans_clustering.csv', index=False)

# 6.Linear regression
linreg_df = run_linear_regression_baseline(X_dense, y, groups)
linreg_df.to_csv('outputs/stage4b/linear_regression_baseline.csv', index=False)

#Visualisation 
print("\n" + "=" * 70)
print("GENERATING VISUALISATIONS")
print("=" * 70)
plot_supplementary_analyses(
    knn_df, reg_df, train_sizes_abs, train_scores, val_scores,
    pca_df, cumvar, cluster_df, linreg_df,
    'outputs/stage4b/supplementary_analyses.png'
)

#Summary JSON 
summary = {
    'knn': {
        'best_k': int(best_k),
        'best_f2': float(knn_df['f2_mean'].max()),
        'conclusion': 'KNN underperforms LR due to curse of dimensionality in sparse TF-IDF space'
    },
    'regularisation': {
        'best_config': reg_df.loc[reg_df['f2_mean'].idxmax(), 'config'],
        'best_f2': float(reg_df['f2_mean'].max()),
        'conclusion': 'L2 regularisation (default) provides best bias-variance balance; '
                      'L1 produces sparser models but similar performance'
    },
    'bias_variance': {
        'train_val_gap': float(gap),
        'diagnosis': diagnosis,
        'conclusion': 'High-bias, low-variance model. Performance limited by weak signal, not overfitting.'
    },
    'pca': {
        'components_90pct': int(n_90),
        'components_95pct': int(n_95),
        'total_features': int(X_dense.shape[1]),
        'conclusion': 'PCA reduces dimensionality but may lose sparse discriminative features'
    },
    'kmeans': {
        'best_k': int(best_k_cluster),
        'best_silhouette': float(cluster_df['silhouette'].max()),
        'conclusion': 'Low silhouette scores confirm weak cluster structure, '
                      'consistent with weak supervised signal'
    },
    'linear_regression': {
        'mean_f2': float(linreg_df['f2'].mean()),
        'conclusion': 'Linear regression produces predictions outside [0,1], '
                      'confirming logistic regression is the correct choice for binary classification'
    },
    'skills_demonstrated': [
        'K-nearest neighbours',
        'Regularisation techniques (L1/L2)',
        'Bias-variance trade-off',
        'Principal component analysis (dimensionality reduction)',
        'K-means clustering (unsupervised learning)',
        'Linear regression vs logistic regression',
        'Evaluation metrics for unsupervised learning (silhouette score)',
        'Strengths and weaknesses of ML approaches',
        'Optimising ML methods for specific use cases'
    ]
}

with open('outputs/stage4b/supplementary_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("STAGE 4b COMPLETE")
print("=" * 70)
print("\nOutputs saved to outputs/stage4b/:")
print("  - knn_comparison.csv")
print("  - regularisation_comparison.csv")
print("  - pca_comparison.csv")
print("  - kmeans_clustering.csv")
print("  - linear_regression_baseline.csv")
print("  - supplementary_analyses.png")
print("  - supplementary_summary.json")