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
    