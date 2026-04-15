import pandas as pd 
from collections import Counter 
def run_feature_profiling(df):
    """profile categorical, numbers and text features."""
    print("\n--- {col} ---")
    print("FEATURE PROFILING")
    print("="*70)