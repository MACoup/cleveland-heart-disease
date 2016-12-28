from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
from StringIO import StringIO



def get_soup(url):
    '''
    Creates the soup object to parse the data from.
    '''

    r = requests.get(url)
    data = r.content
    soup = BeautifulSoup(data, 'html.parser')
    return soup

def load_df(soup):
    '''
    Creates a StringIO object to read the data into a pandas DataFrame.
    '''

    data = StringIO(soup)
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'diagnosis']
    df = pd.read_csv(data, header=None, names=cols)
    return df

def check_values(df):
    '''
    Checking values of the attributes. Found that the 'ca' and 'slope' attributes have '?' values.
    '''
    
    for col in df.columns:
        print df[col].value_counts()

def clean_data(df):
    '''
    Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).

    Original dataset has at most 6 '?' values. Will delete these instances.

    INPUT: Loaded pandas DataFrame

    OUTPUT: Cleaned pandas DataFrame
    '''
    df['diagnosis'] = df['diagnosis'].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
    df.replace(to_replace='?', value=np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    return df


if __name__ == '__main__':
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    soup = get_soup(url)
    df = load_df(soup)
    df = clean_data(df)
    # check_values(df)
    df.to_csv('Data/heart-disease.csv', index=False)
