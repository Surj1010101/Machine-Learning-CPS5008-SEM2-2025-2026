"""Profiling module."""

import pandas as pd
from collections import Counter


def run_feature_profiling(df):
    """
    Profile my categorical, numeric and text features.

    This function shows me the value counts of every categorical feature and the
    escalation rate inside each category, this matters because it tells me
    where the imbalance lives in the data. For numeric features I just describe them so
    I can see the spread. the aim is to flag anything obvious before I move on.
    """
    print("\n" + "="*70)
    print("FEATURE PROFILING")
    print("="*70)

    # My categorical features that I will profile one by one
    cat_cols = ['customer_type', 'tenure_type', 'meter_type', 'region',
                'sentiment', 'issue_category', 'manual_annotation']

    for col in cat_cols:
        print(f"\n-- {col} --")
        vc = df[col].value_counts(dropna=False)
        print(vc)
        # I skip manual_annotation here because it is a leakage feature and I do not want it
        # influencing my early reading of escalation rate per category
        if col not in ['manual_annotation']:
            cross = df.groupby(col, dropna=False)['escalated'].agg(['mean', 'count'])
            cross.columns = ['escalation_rate', 'n']
            print(f"\nEscalation rate by {col}:")
            print(cross.sort_values('escalation_rate', ascending=False))

    # Numeric features, I include resolution_time only for reference, it is excluded as a feature
    print(f"\n-- emotion_intensity --")
    print(df['emotion_intensity'].describe())
    print(f"\n-- resolution_time (EXCLUDED - for reference only) --")
    print(df['resolution_time'].describe())


def run_text_analysis(df):
    """
    Analyse my email text, the length, the duplicates and the word frequency.

    This is where I look at how long my emails are in characters and words, whether
    there are repeat texts in the dataset, and what the most common words look like. The main
    idea is to find anything that could mess up my TF-IDF in Stage 3, for example loads of
    duplicates would inflate the importance of certain phrases. This also gives me the
    text length and word count split by escalation, which is useful as a feature later.
    """
    print("\n" + "="*70)
    print("EMAIL TEXT ANALYSIS")
    print("="*70)

    df['text_len'] = df['email_body_text'].str.len()
    df['word_count'] = df['email_body_text'].str.split().str.len()

    print(f"\nCharacter length stats:\n{df['text_len'].describe()}")
    print(f"\nWord count stats:\n{df['word_count'].describe()}")

    n_unique_texts = df['email_body_text'].nunique()
    n_total = len(df)
    print(f"\nUnique email texts: {n_unique_texts} / {n_total}")
    if n_unique_texts < n_total:
        dupes = df[df.duplicated(subset='email_body_text', keep=False)]
        print(f"Duplicate texts found: {n_total - n_unique_texts} duplicated entries")
        print(f"Sample duplicates:")
        print(dupes.groupby('email_body_text').size().sort_values(ascending=False).head(5))

    print(f"\nTexts with < 10 characters: {(df['text_len'] < 10).sum()}")
    print(f"Texts with < 3 words: {(df['word_count'] < 3).sum()}")

    # Vocabulary analysis, I use a Counter to get the top words across the whole corpus
    all_words = ' '.join(df['email_body_text'].str.lower()).split()
    word_freq = Counter(all_words)
    print(f"\nVocabulary size: {len(word_freq)}")
    print(f"Top 20 words: {word_freq.most_common(20)}")

    print(f"\nText length by escalation status:")
    print(df.groupby('escalated')['text_len'].describe())
    print(f"\nWord count by escalation status:")
    print(df.groupby('escalated')['word_count'].describe())

    return df


def run_timestamp_analysis(df):
    """
    Analyse temporal patterns in my email escalation.

    This function parses the timestamp into year, month, day of week and hour, and
    then groups escalation rate by each of those. the aim is to check if there is any
    temporal signal in escalation, for example if escalation happens more on weekends or at
    night. This also produces the hour and day_of_week features I will use as numeric
    inputs in my Stage 3 pipeline.
    """
    print("\n" + "="*70)
    print("TIMESTAMP ANALYSIS")
    print("="*70)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour

    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nEmails by year:\n{df['year'].value_counts().sort_index()}")
    print(f"\nEscalation rate by year:")
    print(df.groupby('year')['escalated'].agg(['mean', 'count']))

    print(f"\nEscalation rate by day of week:")
    dow_esc = df.groupby('day_of_week')['escalated'].agg(['mean', 'count'])
    print(dow_esc.sort_values('mean', ascending=False))

    print(f"\nEscalation rate by hour (top 5 hours):")
    hour_esc = df.groupby('hour')['escalated'].agg(['mean', 'count'])
    print(hour_esc.sort_values('mean', ascending=False).head(5))

    return df





