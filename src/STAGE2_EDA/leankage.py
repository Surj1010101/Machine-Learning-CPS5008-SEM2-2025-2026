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
    