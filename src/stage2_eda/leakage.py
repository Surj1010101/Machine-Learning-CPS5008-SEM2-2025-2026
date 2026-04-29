"""
Stage 2 missing data analysis, leakage testing, class imbalance and customer overlap module.

Overall this module is where I do the risk identification side of my Stage 2, the basic
idea is to check the data for anything that would silently break my modelling later.
In my project I focused on four things in here, missing data and whether it relates to
the target, leakage testing on suspicious features, class imbalance across segments,
and customer overlap so I know my GroupKFold will work. What this whole module
demonstrates is the data quality and leakage risks I had to handle before Stage 3.
"""

import pandas as pd
from scipy import stats


def run_missing_data_analysis(df):
    """
    Check the overall hidden missingness and sentiment vs escalation.

    Overall this function counts missing values per column, checks for empty strings as
    hidden missingness, and tests whether sentiment missingness is related to escalation
    using a chi-square. The basic idea is to confirm that any imputation I do later is
    safe, because if missingness is related to the target then imputation could leak.
    """
    print("\n" + "="*70)
    print("MISSING DATA ANALYSIS")
    print("="*70)

    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
    # I only print rows where the column actually has missing values, to keep the output tidy
    print(missing_df[missing_df['missing_count'] > 0])

    print(f"\n── Hidden missingness (empty strings) ──")
    for col in df.select_dtypes(include='object').columns:
        empty = (df[col].str.strip() == '').sum()
        if empty > 0:
            print(f"  {col}: {empty} empty strings")

    print(f"\n── Sentiment Missingness vs Escalation ──")
    df['sentiment_missing'] = df['sentiment'].isnull().astype(int)
    cross_sent = pd.crosstab(df['escalated'], df['sentiment_missing'], margins=True)
    cross_sent.columns = ['sentiment_present', 'sentiment_missing', 'total']
    cross_sent.index = ['not_escalated', 'escalated', 'total']
    print(cross_sent)

    # I also break missingness rate down by escalated value and by escalation_level so I can
    # see if there is any monotone pattern, this is what tells me whether it is MAR or not
    for esc_val in [0, 1]:
        subset = df[df['escalated'] == esc_val]
        miss_rate = subset['sentiment'].isnull().mean() * 100
        print(f"  Escalated={esc_val}: sentiment missing rate = {miss_rate:.1f}%")

    print(f"\nSentiment missingness by escalation_level:")
    for lvl in sorted(df['escalation_level'].unique()):
        subset = df[df['escalation_level'] == lvl]
        miss_rate = subset['sentiment'].isnull().mean() * 100
        print(f"  Level {lvl}: {miss_rate:.1f}% missing (n={len(subset)})")

    # Final chi-square test, this is the formal evidence I cite in the report
    contingency = pd.crosstab(df['escalated'], df['sentiment_missing'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square test (sentiment missingness vs escalated): chi2={chi2:.3f}, p={p:.4f}")


def run_leakage_testing(df):
    """
    Test potential leakage variables against the target.

    Overall this is where I confirm which features are safe to use and which ones have
    to be excluded because they would leak the answer. The basic idea is to run univariate
    tests against the target for emotion_intensity, sentiment, the categorical features,
    and then verify resolution_time and manual_annotation are correctly excluded. What
    this demonstrates is the formal evidence behind my leakage decisions.
    """
    print("\n" + "="*70)
    print("LEAKAGE TESTING")
    print("="*70)

    # emotion_intensity, I use Mann-Whitney U because the distribution is not normal
    print(f"\n── emotion_intensity by escalation ──")
    for esc_val in [0, 1]:
        subset = df[df['escalated'] == esc_val]['emotion_intensity']
        print(f"  Escalated={esc_val}: mean={subset.mean():.4f}, std={subset.std():.4f}")

    esc0 = df[df['escalated'] == 0]['emotion_intensity']
    esc1 = df[df['escalated'] == 1]['emotion_intensity']
    u_stat, u_p = stats.mannwhitneyu(esc0, esc1, alternative='two-sided')
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_p:.4f}")

    # Sentiment, I use chi-square here because both variables are categorical
    print(f"\n── Sentiment vs Escalation ──")
    sent_cross = pd.crosstab(df['sentiment'].dropna(),
                             df.loc[df['sentiment'].notna(), 'escalated'])
    print(sent_cross)
    chi2_s, p_s, dof_s, exp_s = stats.chi2_contingency(sent_cross)
    print(f"Chi-square test: chi2={chi2_s:.3f}, p={p_s:.4f}")

    print(f"\nEscalation rate by sentiment:")
    for sent in df['sentiment'].dropna().unique():
        subset = df[df['sentiment'] == sent]
        print(f"  {sent}: {subset['escalated'].mean()*100:.1f}% (n={len(subset)})")

    # Categorical features, I run chi-square on each one to see if any of them are
    # significantly related to escalation, the brief asked me to test for leakage
    print(f"\n── Chi-square Tests: Categorical Features vs Escalation ──")
    cat_features = ['customer_type', 'tenure_type', 'meter_type', 'region', 'issue_category']
    for col in cat_features:
        ct = pd.crosstab(df[col], df['escalated'])
        chi2_c, p_c, dof_c, exp_c = stats.chi2_contingency(ct)
        print(f"  {col}: chi2={chi2_c:.3f}, p={p_c:.4f}, dof={dof_c}")

    # resolution_time is excluded but I still verify the relationship for the report
    print(f"\n── resolution_time vs escalation (VERIFY EXCLUSION) ──")
    for esc_val in [0, 1]:
        subset = df[df['escalated'] == esc_val]['resolution_time']
        print(f"  Escalated={esc_val}: mean={subset.mean():.2f}h, std={subset.std():.2f}")

    # manual_annotation is also excluded as a leakage feature, same idea, just verifying
    print(f"\n── manual_annotation vs escalation (VERIFY EXCLUSION) ──")
    ma_cross = pd.crosstab(df['manual_annotation'].dropna(),
                           df.loc[df['manual_annotation'].notna(), 'escalated'])
    print(ma_cross)
    chi2_ma, p_ma, dof_ma, exp_ma = stats.chi2_contingency(ma_cross)
    print(f"Chi-square: chi2={chi2_ma:.3f}, p={p_ma:.4f}")


def run_imbalance_analysis(df):
    """
    Analyse my class imbalance across segments.

    Overall this function breaks the escalation rate down by every categorical segment so
    I can see if the imbalance is uniform or if some segments are way more imbalanced. The
    basic idea is to inform my StratifiedGroupKFold decision and check whether any segment
    needs special handling.
    """
    print("\n" + "="*70)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*70)

    cat_features = ['customer_type', 'tenure_type', 'meter_type', 'region', 'issue_category']
    print(f"\n── Escalation Rate by Segment ──")
    for col in cat_features:
        print(f"\n  {col}:")
        seg = df.groupby(col)['escalated'].agg(['mean', 'sum', 'count'])
        seg.columns = ['rate', 'n_pos', 'n_total']
        print(seg)


def run_customer_overlap_analysis(df):
    """
    Check customer overlap for GroupKFold readiness.

    Overall this is really important because if the same customer appears in both training
    and validation folds my model could memorise customer-level patterns and inflate F2. The
    basic idea is to count how many customers appear multiple times and how many of those have
    mixed outcomes, this then justifies why I am using GroupKFold grouped by customer_id in
    Stage 3 onwards.
    """
    print("\n" + "="*70)
    print("CUSTOMER OVERLAP ANALYSIS (GroupKFold Readiness)")
    print("="*70)

    cust_counts = df['customer_id'].value_counts()
    print(f"Total unique customers: {cust_counts.shape[0]}")
    print(f"Customers with 1 email: {(cust_counts == 1).sum()}")
    print(f"Customers with 2 emails: {(cust_counts == 2).sum()}")
    print(f"Customers with 3 emails: {(cust_counts == 3).sum()}")
    print(f"Customers with 4 emails: {(cust_counts == 4).sum()}")
    print(f"Customers with >4 emails: {(cust_counts > 4).sum()}")

    # The really critical number, customers that appear multiple times AND have mixed outcomes,
    # because those are the ones that would leak across folds without GroupKFold
    multi = df[df['customer_id'].isin(cust_counts[cust_counts > 1].index)]
    mixed = multi.groupby('customer_id')['escalated'].nunique()
    print(f"\nMulti-email customers: {len(mixed)}")
    print(f"  With mixed outcomes (both 0 and 1): {(mixed > 1).sum()}")
    all_same = (mixed == 1).sum()
    first_esc = df[df['customer_id'].isin(mixed[mixed == 1].index)].groupby('customer_id')['escalated'].first().sum()
    print(f"  All non-escalated: {all_same - first_esc}")
