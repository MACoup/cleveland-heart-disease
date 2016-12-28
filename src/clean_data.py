import pandas as pd
import numpy as np

def check_values(df):
    '''
    Checking values of the attributes. Found that the 'ca' and 'slope' attributes have '?' values.
    '''
    for col in df.columns:
        print df[col].value_counts()

def clean_data(df):
    '''
    Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).

    Original dataset has at most 6 '?' values. Deleting these instances.

    INPUT: Loaded pandas DataFrame

    OUTPUT: Cleaned pandas DataFrame
    '''
    df['diagnosis'] = df['diagnosis'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df.replace(to_replace='?', value=np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    return df

if __name__ == '__main__':
    df = pd.read_csv('../Data/heart-disease.csv')
    check_values(df)
    df = clean_data(df)

    # Can see that after cleaning, diagnosis has only 0 or 1 for yes and no
    # 'ca' and 'slope' no longer have '?' values
    check_values(df)
    df.to_csv('../Data/heart-disease-cleaned.csv', index=False)
