import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

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
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    width = 0.4
    labels = ['Positive Diagnosis', 'All Instances', 'Negative Diagnosis']
    df_all = df
    df_yes = df[df['diagnosis'] == 1]
    df_no = df[df['diagnosis'] == 0]
    cats = df[col].unique()

    vc1 = df_yes[col].value_counts()
    vc1_labels = vc1.index.tolist()

    vc2 = df_all[col].value_counts()
    vc2_labels = vc2.index.tolist()

    vc3 = df_no[col].value_counts()
    vc3_labels = vc3.index.tolist()

    count_label_1 = zip(vc1_labels, vc1)
    count_label_2 = zip(vc2_labels, vc2)
    count_label_3 = zip(vc3_labels, vc3)


    return count_label_1, count_label_2, count_label_3, cats

def get_order(cats, cl_list):
    label_dict = {cat: [] for cat in cats}
    for cl in cl_list:
        for t in cl:
            label_dict[t[0]] = label_dict.get(t[0], []).append(t[1])
    return label_dict


def get_box(df, col):
    df_all = df
    df_yes = df[df['diagnosis'] == 1]
    df_no = df[df['diagnosis'] == 0]
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    x1 = np.array(df_yes[col])
    x2 = np.array(df_all[col])
    x3 = np.array(df_no[col])
    ax.boxplot([x1, x2, x3], labels=['Positive Diagnosis', 'All Instances', 'Negative Diagnosis'])
    ax.set_title(col.upper())
    plt.show()

def get_all_box(df, columns):
    for col in columns:
        get_box(df, col)





if __name__ == '__main__':
    df = pd.read_csv('../Data/heart-disease-cleaned.csv')
    box_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    hist_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    # get_all_box(df, box_cols)
    cat_df = get_true_values(df)
    cl1, cl2, cl3, cats = get_bars(cat_df, 'thal')
    cl_list = [cl1, cl2, cl3]
    d = get_order(cats, cl_list)
