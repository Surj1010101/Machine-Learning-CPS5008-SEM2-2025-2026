import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq


def cramers_v(x, y):
    """Calculate Cramer's V for two categorical variables."""
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    min_dim = min(ct.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def run_numeric_correlation(df, numeric_cols):
    """Compute pairwise correlations for numeric features and find worst pair."""
    print("\n── Numeric Feature Correlations ──")
    numeric_features = df[numeric_cols].copy()
    corr_matrix = numeric_features.corr()
    print(corr_matrix.round(4))

    max_corr = 0
    max_pair = ('', '')
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            c = abs(corr_matrix.iloc[i, j])
            if c > max_corr:
                max_corr = c
                max_pair = (corr_matrix.columns[i], corr_matrix.columns[j])

    print(f"\nHighest pairwise correlation: {max_pair[0]} vs {max_pair[1]} = {max_corr:.4f}")

    if max_corr < 0.3:
        print("CONCLUSION: No concerning multicollinearity among numeric features (all |r| < 0.3)")
    elif max_corr < 0.7:
        print("NOTE: Moderate correlation detected but below threshold for concern (|r| < 0.7)")
    else:
        print("WARNING: High correlation detected -- consider removing one feature")

    return corr_matrix, max_corr, max_pair


def run_cramers_v_analysis(df, cat_features):
    """Pairwise Cramer's V across categorical features."""
    print("\n── Categorical Feature Independence (Cramer's V) ──")

    cramers_results = []
    for i in range(len(cat_features)):
        for j in range(i + 1, len(cat_features)):
            v = cramers_v(df[cat_features[i]], df[cat_features[j]])
            cramers_results.append({
                'Feature 1': cat_features[i],
                'Feature 2': cat_features[j],
                'Cramers V': v,
            })

    cramers_df = pd.DataFrame(cramers_results).sort_values('Cramers V', ascending=False)
    print(cramers_df.to_string(index=False))

    max_v = cramers_df['Cramers V'].max()
    print(f"\nHighest Cramer's V: {max_v:.4f}")
    if max_v < 0.3:
        print("CONCLUSION: No concerning multicollinearity among categorical features (all V < 0.3)")
    else:
        print("NOTE: Some categorical features show moderate association")

    return cramers_df, max_v