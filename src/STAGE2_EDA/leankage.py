import pandas as pd 
from scipy import stats

def run_missing_data_analysis(df):
    """check the overall hidden missingness, sentment vs escalation."""
    print("/n" + "="*70)
    print("MISSING DATA ANALYSIS")
    print("="*70)

    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
    print(missing_df[missing_df]['missing_count'] > 0])

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


    for esc_val in [0, 1]:
        subset = df[df['escalated'] == esc_val]
        miss_rate = subset['sentiment'].isnull().mean() * 100
        print(f"  Escalated={esc_val}: sentiment missing rate = {miss_rate:.1f}%")

    print(f"\nSentiment missingness by escalation_level:")
    for lvl in sorted(df['escalation_level'].unique()):
        subset = df[df['escalation_level'] == lvl]
        miss_rate = subset['sentiment'].isnull().mean() * 100
        print(f"  Level {lvl}: {miss_rate:.1f}% missing (n={len(subset)})")

    contingency = pd.crosstab(df['escalated'], df['sentiment_missing'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square test (sentiment missingness vs escalated): chi2={chi2:.3f}, p={p:.4f}")


def run_leakage_testing(df):
    """Test potential leakage variables against the target."""
    print("\n" + "="*70)
    print("LEAKAGE TESTING")
    print("="*70)

    #emotion_intensity
    print(f"\n── emotion_intensity by escalation ──")
    for esc_val in [0, 1]:
        subset = df[df['escalated'] == esc_val]['emotion_intensity']
        print(f"  Escalated={esc_val}: mean={subset.mean():.4f}, std={subset.std():.4f}")

    esc0 = df[df['escalated'] == 0]['emotion_intensity']
    esc1 = df[df['escalated'] == 1]['emotion_intensity']
    u_stat, u_p = stats.mannwhitneyu(esc0, esc1, alternative='two-sided')
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_p:.4f}")

    #Sentiment
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

    #Categorical features
    print(f"\n── Chi-square Tests: Categorical Features vs Escalation ──")
    cat_features = ['customer_type', 'tenure_type', 'meter_type', 'region', 'issue_category']
    for col in cat_features:
        ct = pd.crosstab(df[col], df['escalated'])
        chi2_c, p_c, dof_c, exp_c = stats.chi2_contingency(ct)
        print(f"  {col}: chi2={chi2_c:.3f}, p={p_c:.4f}, dof={dof_c}")

       # resolution_time (excluded -- verify)
    print(f"\n── resolution_time vs escalation (VERIFY EXCLUSION) ──")
    for esc_val in [0, 1]:
        subset = df[df['escalated'] == esc_val]['resolution_time']
        print(f"  Escalated={esc_val}: mean={subset.mean():.2f}h, std={subset.std():.2f}")

    # manual_annotation (excluded -- verify)
    print(f"\n── manual_annotation vs escalation (VERIFY EXCLUSION) ──")
    ma_cross = pd.crosstab(df['manual_annotation'].dropna(),
                           df.loc[df['manual_annotation'].notna(), 'escalated'])
    print(ma_cross)
    chi2_ma, p_ma, dof_ma, exp_ma = stats.chi2_contingency(ma_cross)
    print(f"Chi-square: chi2={chi2_ma:.3f}, p={p_ma:.4f}")


def run_imbalance_analysis(df):
    """Analyse class imbalance across segments."""
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
    """Check customer overlap for GroupKFold readiness."""
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

    multi = df[df['customer_id'].isin(cust_counts[cust_counts > 1].index)]
    mixed = multi.groupby('customer_id')['escalated'].nunique()
    print(f"\nMulti-email customers: {len(mixed)}")
    print(f"  With mixed outcomes (both 0 and 1): {(mixed > 1).sum()}")
    all_same = (mixed == 1).sum()
    first_esc = df[df['customer_id'].isin(mixed[mixed == 1].index)].groupby('customer_id')['escalated'].first().sum()
    print(f"  All non-escalated: {all_same - first_esc}")