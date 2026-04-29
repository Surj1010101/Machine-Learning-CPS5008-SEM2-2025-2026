"""PCA dimensionality reduction and K-Means clustering on TF-IDF features."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, silhouette_score

from common.pipeline_utils import make_preprocessor, sgkf, find_best_threshold_f2


def prepare_dense_features(X):
    """transforming features by the standard preprocessor, return dense matrix."""
    preprocessor = make_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    if hasattr(X_transformed, 'toarray'):
        X_dense = X_transformed.toarray()
    else:
        X_dense = X_transformed
    return X_dense


def run_pca_analysis(X_dense, y, groups):
    """PCA on full TF  IDF feature matrix; evaluate F2 at different component counts."""
    print("\n" + "=" * 70)
    print("4. PCA -- DIMENSIONALITY REDUCTION")
    print("=" * 70)
    print("  Applying PCA to the TF-IDF feature matrix to explore whether")
    print("  lower-dimensional representations retain discriminative power.")

    pca_full = PCA(random_state=42)
    pca_full.fit(X_dense)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    n_90 = np.argmax(cumvar >= 0.90) + 1
    n_95 = np.argmax(cumvar >= 0.95) + 1
    print(f"  Total features after preprocessing: {X_dense.shape[1]}")
    print(f"  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")

    pca_results = []
    for n_components in [10, 25, 50, 100, n_90]:
        fold_f2s = []
        for train_idx, val_idx in sgkf.split(X_dense, y, groups):
            X_train_t = X_dense[train_idx]
            X_val_t = X_dense[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pca = PCA(n_components=n_components, random_state=42)
            X_train_pca = pca.fit_transform(X_train_t)
            X_val_pca = pca.transform(X_val_t)

            clf = LogisticRegression(
                class_weight='balanced', max_iter=1000,
                solver='liblinear', random_state=42)
            clf.fit(X_train_pca, y_train)
            y_prob = clf.predict_proba(X_val_pca)[:, 1]
            y_prob_train = clf.predict_proba(X_train_pca)[:, 1]
            thresh = find_best_threshold_f2(y_train, y_prob_train)
            y_pred = (y_prob >= thresh).astype(int)
            fold_f2s.append(fbeta_score(y_val, y_pred, beta=2))

        mean_f2 = np.mean(fold_f2s)
        pca_results.append({
            'n_components': n_components,
            'f2_mean': mean_f2, 'f2_std': np.std(fold_f2s),
            'variance_explained': cumvar[n_components - 1]
        })
        print(f"  PCA n={n_components:3d}: F2 = {mean_f2:.4f} "
              f"(var explained: {cumvar[n_components-1]*100:.1f}%)")

    pca_df = pd.DataFrame(pca_results)
    print("\n  PCA reduces dimensionality but may lose discriminative signal in sparse")
    print("  TF-IDF space where individual word features carry meaningful information.")

    return pca_df, cumvar, n_90, n_95


def run_kmeans_clustering(X_dense, y):
    """KMeans clustering on PCA(50-reduced features."""
    print("\n" + "=" * 70)
    print("5. K-MEANS CLUSTERING (Unsupervised Learning)")
    print("=" * 70)
    print("  Exploring whether natural clusters in the email data align with")
    print("  escalation outcomes (unsupervised learning for pattern discovery).")

    pca_50 = PCA(n_components=50, random_state=42)
    X_pca50 = pca_50.fit_transform(X_dense)

    cluster_results = []
    for k in [2, 3, 4, 5, 7, 10]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca50)
        sil = silhouette_score(X_pca50, labels, sample_size=min(2000, len(X_pca50)))

        cluster_esc_rates = []
        for c in range(k):
            mask = labels == c
            esc_rate = y[mask].mean()
            cluster_esc_rates.append(esc_rate)

        esc_range = max(cluster_esc_rates) - min(cluster_esc_rates)
        cluster_results.append({
            'k': k, 'silhouette': sil,
            'min_esc_rate': min(cluster_esc_rates),
            'max_esc_rate': max(cluster_esc_rates),
            'esc_rate_range': esc_range
        })
        print(f"  K={k:2d}: Silhouette = {sil:.4f} | "
              f"Escalation rate range: {min(cluster_esc_rates)*100:.1f}%-{max(cluster_esc_rates)*100:.1f}%")

    cluster_df = pd.DataFrame(cluster_results)
    best_k_cluster = cluster_df.loc[cluster_df['silhouette'].idxmax(), 'k']
    km_best = KMeans(n_clusters=int(best_k_cluster), random_state=42, n_init=10)
    best_labels = km_best.fit_predict(X_pca50)

    print(f"\n  Best K by silhouette: {int(best_k_cluster)} "
          f"(silhouette={cluster_df['silhouette'].max():.4f})")
    print("\n  Cluster profiles (best K):")
    for c in range(int(best_k_cluster)):
        mask = best_labels == c
        n_samples = mask.sum()
        esc_rate = y[mask].mean()
        print(f"    Cluster {c}: n={n_samples} ({n_samples/len(y)*100:.1f}%), "
              f"escalation rate={esc_rate*100:.1f}%")

    print("\n  Evaluation: Silhouette score measures cluster cohesion and separation.")
    print("  Low silhouette (<0.25) suggests clusters are not well-separated,")
    print("  consistent with the weak signal observed in supervised models.")
    print("  Escalation rate variation across clusters indicates whether unsupervised")
    print("  structure correlates with the supervised target.")

    return cluster_df, int(best_k_cluster)
