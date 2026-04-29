"""
Shared data loader and feature group constants module.

Overall this module is the foundation that every later stage builds on, the basic idea
is to keep the data loading, target derivation and feature group definitions in one
place so every stage gets a consistent view of the dataset. In my project I focused on
three things here, the column lists (EXCLUDE_COLS, CATEGORICAL_COLS, NUMERIC_COLS,
TEXT_COL_CLEAN), the load_and_prepare_data helper that returns the (df, X, y, groups)
tuple, and the clean_text helper used by the TF-IDF stage. What this module
demonstrates is reproducibility, every stage gets the exact same processed data with
the same target, the same exclusions and the same temporal features.
"""

import re
import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# Making sure my working directory is the project root regardless of how scripts are launched
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

# Reproducibility seed for any numpy randomness later in the pipeline
np.random.seed(42)

# Feature group constants, these are the single source of truth for every later stage
EXCLUDE_COLS = ['escalation_level', 'resolution_time', 'manual_annotation',
                'customer_id', 'timestamp', 'escalated', 'email_body_text']
TEXT_COL_CLEAN = 'email_body_text_clean'
CATEGORICAL_COLS = ['customer_type', 'tenure_type', 'meter_type', 'region',
                    'issue_category', 'sentiment']
NUMERIC_COLS = ['emotion_intensity', 'hour', 'day_of_week', 'month']


def clean_text(text):
    """
    Minimal text cleaning for my short email fragments.

    Overall this function lowercases, strips punctuation and collapses extra whitespace,
    the basic idea is to keep the cleaning conservative because the emails are already
    short (around 7 to 8 words on average) and aggressive cleaning would lose signal.
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_prepare_data():
    """
    Load my dataset and return (df, X, y, groups) for modelling.

    Overall this is the entry point that every stage uses to load data, the basic idea
    is to derive the binary target, parse the timestamp into hour/day_of_week/month,
    impute missing sentiment with the 'Unknown' string, and build a feature matrix that
    drops everything in EXCLUDE_COLS. What this also returns is the customer_id array
    as 'groups' which my StratifiedGroupKFold needs.

    Returns:
        df: full DataFrame with all original columns plus my derived features
        X: feature matrix (excludes target, IDs and leakage columns)
        y: binary target array (1 if escalated, 0 otherwise)
        groups: customer_id array for GroupKFold
    """
    df = pd.read_csv('data/customer_support_emails.csv')

    # Deriving my binary target from escalation_level >= 2
    df['escalated'] = (df['escalation_level'] >= 2).astype(int)

    # Temporal features parsed out of the raw timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Imputing missing sentiment with the literal 'Unknown' string so OneHotEncoder treats it as a category
    df['sentiment'] = df['sentiment'].fillna('Unknown')

    # Cleaning the text column for downstream TF-IDF
    df['email_body_text_clean'] = df['email_body_text'].apply(clean_text)

    # Building the modelling matrices
    X = df.drop(columns=EXCLUDE_COLS)
    y = df['escalated'].values
    groups = df['customer_id'].values

    return df, X, y, groups
