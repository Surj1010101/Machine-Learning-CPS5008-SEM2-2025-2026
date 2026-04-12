import re
import pandas as pd 

def clean_text(text):
    """Minimal text cleaning for short email fragments."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data():
    """Load dataset and return df,X,y GROUPS which are for modelling.


    Returns:
        df: full DataFrame with differnt features
        X: Feature matrix (excludes target, IDs, leakage columns)
        y: Binary target array
        groups: Customer ID array for GroupKFold
    """
    df = pd.read_csv('data/customer_support_emails.csv')
    
    #derive binary target 
    df['escalated'] = (df['escalation_level'] >= 2).astype(int)

    #tepmorel features 
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    

    #Handleing missing values sentiment
    df['sentiment'] = df['sentiment'].fillna('Unknown')

    # Clean text
    df['email_body_text_clean'] = df['email_body_text'].apply(clean_text)

    # Build feature matrix
    X = df.drop(columns=EXCLUDE_COLS)
    y = df['escalated'].values
    groups = df['customer_id'].values

    return df, X, y, groups


