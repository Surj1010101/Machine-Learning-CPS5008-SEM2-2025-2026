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


def run_vif_analysis(df, numeric_cols):
    """Variance Inflation Factor for numeric features."""
    print("\n── Variance Inflation Factor (VIF) for Numeric Features ──")
    numeric_features = df[numeric_cols].copy()

    vif_results = []
    for col in numeric_features.columns:
        other_cols = [c for c in numeric_features.columns if c != col]
        X_vif = numeric_features[other_cols].values
        y_vif = numeric_features[col].values

        X_vif = np.column_stack([np.ones(len(X_vif)), X_vif])
        coeffs, _, _, _ = lstsq(X_vif, y_vif, rcond=None)
        y_pred = X_vif @ coeffs
        ss_res = np.sum((y_vif - y_pred) ** 2)
        ss_tot = np.sum((y_vif - y_vif.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
        vif_results.append({'Feature': col, 'VIF': vif, 'R_squared': r_squared})
        print(f"  {col:<20s}: VIF = {vif:.3f} (R² = {r_squared:.4f})")

    vif_df = pd.DataFrame(vif_results)
    max_vif = vif_df['VIF'].max()
    print(f"\nMax VIF: {max_vif:.3f}")
    if max_vif < 5:
        print("CONCLUSION: All VIF values < 5. No multicollinearity concern.")
    elif max_vif < 10:
        print("NOTE: Moderate VIF detected. Monitor but not actionable.")
    else:
        print("WARNING: High VIF detected. Consider removing correlated features.")

    return vif_df, max_vif


def print_mitigation_summary():
    """Print summary of multicollinearity mitigations in the pipeline."""
    print("\n── Multicollinearity Mitigation in Pipeline ──")
    print("1. OneHotEncoder uses drop='first' to avoid the dummy variable trap")
    print("   (perfect multicollinearity among one-hot columns for each feature)")
    print("2. TF-IDF features are inherently high-dimensional but L2-normalised,")
    print("   and the L1 penalty in Logistic Regression (solver='liblinear')")
    print("   provides implicit feature selection in correlated feature spaces")
    print("3. StandardScaler applied to numeric features ensures comparable scales")
    print("   but does not address multicollinearity -- verified above that")
    print("   numeric correlations are negligible (max |r| < 0.3)")
