import pandas as pd 
from collections import Counter 

def run_feature_profiling(df):
    print("/n" + "="*70)
    print("FEATURE PROFILING")
    print("="*70)

    cat_cols = [' customer_type', 'tenure_type', 'meter_type', 'region',
                'sentiment', 'issue_category', 'manaul_annotation']
    
    for col in cat_cols:
        print(f"n-- {col} --")
        vc  = df [col].value_couns(dropna=False)
        print(vc)
        if col not in ['manual_annotation']:
            cross = df.groupby(col, dropna=False)['escalted'].agg(['mean', 'count'])
            cross.coloumns = ['escalation_rate', 'n']
            print(f"/nEscaltion rate by {col}:")
            print(cross.sort_values('esacltion_rate', ascending=False))


    #numeric feature
    print(f"/n-- emotion_intensity --")
    print(df['emotion_intensity'].describe())
    print(f"/n-- resolution_time (EXCLUDED - for refernce only) --")
    print(df['resolution_time'].describe())


def run_text_analysis(df):
    """Analyse email text: length, duplicate, word frequency."""
    print("/n" + "="*70)
    print("EMAIL TEXT ANALYSIS")
    print("="*70)

    df['text_len'] = df['email_body_text'].str.len()
    df['word'] = df ['email_body_text'].str.spilt().str.len()
    
    print(f"/nCharacter lenghth stats:/n{df['text_len'].describe ()}")
    print (f"/nWord count stats:m{df['word_count'].describe()}")
    




    