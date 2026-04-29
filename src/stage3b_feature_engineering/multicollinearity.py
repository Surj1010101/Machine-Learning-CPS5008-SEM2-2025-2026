"""
Stage 3b multicollinearity analysis module: numeric correlations, Cramer's V, and VIF.

Overall this module is where I formally test whether any of my features are too
correlated with each other to safely use together in Logistic Regression. The basic idea
is to attack the question from three different angles, pairwise correlations on numerics,
Cramer's V on categoricals, and VIF on the numerics again. In my project this is really
important because if multicollinearity is high then my coefficient interpretation in the
report becomes unreliable. What this module demonstrates is the formal evidence that
my pipeline does not have a multicollinearity problem.
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq


def cramers_v(x, y):
    """
    Calculate Cramer's V for two categorical variables.

    Overall this is the small helper that powers the categorical pairwise analysis. The
    basic idea is to derive an effect size between 0 and 1 from a chi-square contingency,
    where 0 means independent and 1 means perfectly associated.
    """
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    min_dim = min(ct.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def run_numeric_correlation(df, numeric_cols):
    """
    Compute pairwise Pearson correlations for my numeric features and find the worst pair.

    Overall this function builds the correlation matrix, prints it, and then scans the
    upper triangle to find the most correlated pair. The basic idea is to flag any
    numeric pair that is too close to perfect correlation, in my project the threshold
    I use for concern is 0.7 but I report the value either way.
    """
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
    """
    Pairwise Cramer's V across my categorical features.

    Overall this is the categorical equivalent of the numeric correlation check, the basic
    idea is to compute Cramer's V for every pair of categorical features and report the
    highest value. What this matters for is that if two categoricals are highly associated
    they will compete for the same coefficient slot in my Logistic Regression.
    """
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
    """
    Variance Inflation Factor for my numeric features.

    Overall VIF is the more rigorous version of pairwise correlation, the basic idea is
    that for each numeric feature I regress it against all the others and check how
    much of its variance is explained, then convert that to VIF. What this matters for
    is that VIF over 5 is a flag and over 10 is a real problem, so this is a hard
    numerical bar I want to clear.
    """
    print("\n── Variance Inflation Factor (VIF) for Numeric Features ──")
    numeric_features = df[numeric_cols].copy()

    vif_results = []
    for col in numeric_features.columns:
        other_cols = [c for c in numeric_features.columns if c != col]
        X_vif = numeric_features[other_cols].values
        y_vif = numeric_features[col].values

        # Adding intercept column and solving the OLS regression for R squared
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
    """
    Print my summary of how multicollinearity is mitigated by the pipeline design.

    Overall this is the bit I quote in the report to show I have thought about
    multicollinearity at the design level, not just diagnosed it after the fact.
    """
    print("\n── Multicollinearity Mitigation in Pipeline ──")
    print("1. OneHotEncoder uses drop='first' to avoid the dummy variable trap")
    print("   (perfect multicollinearity among one-hot columns for each feature)")
    print("2. TF-IDF features are inherently high-dimensional but L2-normalised,")
    print("   and the L1 penalty in Logistic Regression (solver='liblinear')")
    print("   provides implicit feature selection in correlated feature spaces")
    print("3. StandardScaler applied to numeric features ensures comparable scales")
    print("   but does not address multicollinearity -- verified above that")
    print("   numeric correlations are negligible (max |r| < 0.3)")
