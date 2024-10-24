import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

def drop_cols(df, cols_list):
    df = df.drop(columns=cols_list)
    return df

def change_datatype_cols(df, cols_list, datatypes):
    for col, datatype in zip(cols_list, datatypes):
        df[col] = df[col].astype(datatype)
    return df

def encode_cols(df, cols_list):
    label_encoder = LabelEncoder()
    for col in cols_list:
        df[col] = label_encoder.fit_transform(df[col])
    return df

def binning_cols(df, cols_list, new_cols_list, datatypes, labels, num_categories):
    for col_orig, col_new, datatype, label, num_category in zip(cols_list, new_cols_list, datatypes, labels, num_categories):
        min_value = df[col_orig].min() - 0.001
        max_value = df[col_orig].max() + 0.001
        bins = np.linspace(min_value, max_value, num_category + 1)
        df[col_new] = pd.cut(df[col_orig], bins=bins, labels=label, right=False)
        df[col_new] = df[col_new].astype(datatype)
    return df

def balance_dataset(X, y):
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)
    return X, y

def standardize_dataset(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X