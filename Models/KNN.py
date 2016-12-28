import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
    print 'Error: ', 1 - clf.score(X_test, y_test)
    print
    print 'Confusion Matrix: ', standard_confusion_matrix(y_test, y_pred)


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

def plot_ks(model, filename):
    '''
    Plot the accuracy of the classifier model with the number of neightbors.

    INPUT: Classifier model

    OUTPUT: Plot of accuracy vs number of neighbors
    '''
    X_train, X_test, y_train, y_test = organize_and_split(filename)
    ac_list = []
    for k in range(1,20):
        clf = model(n_neighbors=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        print k, accuracy
        ac_list.append(clf.score(X_test, y_test))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(range(1,20), ac_list)
    ax.set_xlabel('Number of Neighbors', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    plt.show()

if __name__ == '__main__':
    filename = '../Data/heart-disease-cleaned.csv'
    df = pd.read_csv(filename)
    df = get_true_values(df)
    df = pd.get_dummies(df)
    X_train, X_test, y_train, y_test = split()
    get_KNN(X_train, X_test, y_train, y_test, n_neighbors=1)

    clf = KNeighborsClassifier(n_jobs=-1)
    cv = cross_val_score(clf, X_train, y_train)
    gs = get_gridsearch_params(clf, X_train, y_train)
    model = KNeighborsClassifier
    plot_ks(model)
