import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn



def standard_confusion_matrix(y_true, y_pred):
    '''
    My own implementation of the confusion matrix, sci-kit learn's version is backwards.

    INPUT: True values of the target and predict values of the target

    OUTPUT: Confusion matrix
    '''


    tp = 0.
    tn = 0.
    fn = 0.
    fp = 0.
    for t in zip(y_true, y_pred):
        if t[0] == t[1]:
            if t[0] == 0:
                 tn += 1
            else:
                 tp += 1
        else:
            if t[0] == 0:
                fp += 1
            else:
                fn += 1
    return np.array([[tp, fp], [fn, tn]])


def get_KNN(X_train, X_test, y_train, y_test, n_neighbors):
    '''
    Fits the data to a KNeighborsClassifier object, then predicts and scores

    INPUT: Training and testing datasets, n_neighbors to fit the classifier with

    OUTPUT: Acuracy score, Error, and Confusion Matrix
    '''

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print 'K = ', n_neighbors
    print
    print 'Accuracy: ', clf.score(X_test, y_test)
    print
    print 'Precision: ', precision_score(y_test, y_pred)
    print
    print 'Recal: ', recall_score(y_test, y_pred)
    print
    print 'Confusion Matrix: '
    print standard_confusion_matrix(y_test, y_pred)


def get_gridsearch_params(clf, X_train, y_train):
    '''
    GridSearch for different parameters to find optimal parameters.

    INPUT: classifier, training sets

    OUTPUT: GridSearch object
    '''


    params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    gs = GridSearchCV(clf, param_grid=params, cv=5)
    gs.fit(X_train, y_train)
    return gs


def organize_and_split(filename):
    '''
    To get an accurate representation of the categorical variables, it is crucial to use the get_dummies function of pandas.

    The MinMaxScaler allowed me to scale the continuous variables while keeping the categorical variables as binary.

    train_test_split is a sci-kit learn function used to split the data into training and testig sets. It allows me to split the set to predetermined sizes.

    INPUT: None

    OUTPUT: Train and test splits of data
    '''


    df = pd.read_csv(filename)
    df = get_true_values(df)
    df = pd.get_dummies(df)
    target = df.pop('diagnosis')
    data = df
    data = MinMaxScaler().fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, train_size=0.7, random_state=42)
    return X_train, X_test, y_train, y_test



def get_true_values(df):
    '''
    Get names of categorical attributes. This allows us to interpret them as dummy variables.

    INPUT: DataFrame

    OUTPUT: New DataFrame
    '''


    df['sex'] = df.sex.map({1: 'Male', 0: 'Female'})
    df['cp'] = df.cp.map({1: 'Typical angina',
                            2: 'Atypical angina',
                            3: 'Non-anginal pain',
                            4: 'Asymptomatic'})
    df['fbs'] = df.fbs.map({1: '> 120 mg/dl', 0: '< 120 mg/dl'})
    df['restecg'] = df.restecg.map({0: 'Normal',
                                    1: 'ST-T wave abnormality',
                                    2: 'Left Ventrical Hypertrophy'})
    df['exang'] = df.exang.map({1: 'Yes', 0: 'No'})
    df['slope'] = df.slope.map({1: 'Upsloping',
                                2: 'Flat',
                                3: 'Downsloping'})
    df['thal'] = df.thal.map({3: 'Normal',
                                6: 'Fixed Defect',
                                7 : 'Reversable Defect'})
    df['ca'] = df.ca.map({0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three'})
    return df

def get_smaller_df(filename, cols):
    '''
    Reduce the feature space to include the variables most highly correlated to the target variable

    INPUT: Filename and attributes.

    OUTPUT: train test split of new DataFrame
    '''
    df = pd.read_csv(filename)
    df = get_true_values(df)
    df = df[cols]
    df = pd.get_dummies(df)
    target = df.pop('diagnosis')
    data = df
    data = MinMaxScaler().fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, train_size=0.7, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    filename = '../Data/heart-disease-cleaned.csv'
    df = pd.read_csv(filename)
    df = get_true_values(df)
    df = pd.get_dummies(df)
    X_train, X_test, y_train, y_test = organize_and_split(filename)
    print 'All Features:'
    get_KNN(X_train, X_test, y_train, y_test, n_neighbors=8)
    print

    clf = KNeighborsClassifier(n_jobs=-1)
    cv = cross_val_score(clf, X_train, y_train)
    gs = get_gridsearch_params(clf, X_train, y_train)
    corr_cols = ['cp', 'exang', 'oldpeak', 'ca', 'thal', 'thalach', 'diagnosis']
    X2_train, X2_test, y2_train, y2_test = get_smaller_df(filename, corr_cols)

    print 'Reduced Feature Size:'
    get_KNN(X2_train, X2_test, y2_train, y2_test, n_neighbors=8)
