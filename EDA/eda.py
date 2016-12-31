import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.preprocessing import MinMaxScaler

'''Column descriptions:
sex: 1 = male, 0 = female
cp: Chest pain
    1: Typical angina
    2: Atypical angina
    3: Non-anginal pain
    4: Asymptomatic
    # angina: a condition marked by severe pain in the chest, often also spreading to the shoulders, arms, and neck, caused by an inadequate blood supply to the heart.

trestbps: Resting blood pressure (in mg Hg on admission to the hospital)
chol: serum cholestoral in mg/dl
fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg: resting electrocardiographic results
    0: Normal
    1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    2: showing probable or definite left ventricular hypertrophy by Estes' criteria
thalach: maximum heart rate achieved
exang: exercise induced angina (1 = yes; 0 = no)
oldpeak: ST depression induced by exercise relative to rest
slope: the slope of the peak exercise ST segment
    1: upsloping
    2: flat
    3: downsloping
ca: number of major vessels (0-3) colored by flouroscopy
thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
diagnosis:
    0: < 50% diameter narrowing
    1: > 50% diameter narrowing
'''

def get_true_values(df):
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
    return df


def check_correlation(df):
    return df.corr()

def get_bars(df, col):
    '''
    Plots bar graphs for categorical data.

    INPUT: DataFrame and column.

    OUTPUT: Plot
    '''

    labels = ['Positive Diagnosis', 'All Instances', 'Negative Diagnosis']
    df_all = df
    df_yes = df[df['diagnosis'] == 1]
    df_no = df[df['diagnosis'] == 0]

    vc1 = pd.DataFrame(df_yes[col].value_counts(sort=False))
    vc1.rename(columns={col: 'Positive Diagnosis'}, inplace=True)

    vc2 = pd.DataFrame(df_all[col].value_counts(sort=False))
    vc2.rename(columns={col: 'All Instances'}, inplace=True)

    vc3 = pd.DataFrame(df_no[col].value_counts(sort=False))
    vc3.rename(columns={col: 'Negative Diagnosis'}, inplace=True)

    new_df = pd.concat([vc1, vc2, vc3], axis=1)
    new_df.plot.bar(rot=0, fontsize=12, title=col.upper())
        # break
    plt.savefig(str(col) + '.png')
    plt.show()

def get_all_bars(df, columns):
    '''
    Used to plot all bar graphs as once.
    '''
    for col in columns:
        get_bars(df, col)


def get_box(df, col):
    '''
    Plots box and whisker plots for columns

    INPUT: DataFrame and column

    OUTPUT: Box and whisker plots
    '''

    df_all = df
    df_yes = df[df['diagnosis'] == 1]
    df_no = df[df['diagnosis'] == 0]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    x1 = np.array(df_yes[col])
    x2 = np.array(df_all[col])
    x3 = np.array(df_no[col])
    ax.boxplot([x1, x2, x3], labels=['Positive Diagnosis', 'All Instances', 'Negative Diagnosis'], showmeans=True)
    ax.set_title(col.upper())
    plt.savefig(str(col) + '.png')
    plt.show()

def get_all_box(df, columns):
    '''
    Plots all box plots simultaneously.
    '''
    for col in columns:
        get_box(df, col)

def get_corr(df):
    return df.corr()

def get_scaled_dummied_corr(df):
    df = pd.get_dummies(df)
    scaler = MinMaxScaler()
    x = scaler.fit_transform(df)
    df_x = pd.DataFrame(x, columns=df.columns)
    return df_x.corr()






if __name__ == '__main__':
    df = pd.read_csv('../Data/heart-disease-cleaned.csv')
    box_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    bar_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    cat_df = get_true_values(df)
    # get_all_box(df, box_cols)
    # get_all_bars(cat_df, bar_cols)
    # plot_age_gender(df, bins=4)
    df_yes = df[df['diagnosis'] == 1]
