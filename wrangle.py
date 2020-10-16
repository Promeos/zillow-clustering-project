import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE


def wrangle_data(df, target_name, modeling=False):
    '''
    Signature: prep_data(df, modeling=False)
    Docstring:
    This function accepts any dataframe and splits it into train, validate,
    and test sets for EDA or modeling.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    
    target_name : str
        target_name is the column name of the target variable
    
    modeling : boolean, False by default
        `modeling` parameter scales numeric data to use in machine learning models.
        
        If modeling is False: The function returns unscaled X_set and y_set dataframes
        If modeling is True: The function returns scaled X_set and y_set dataframes

    Returns
    -------
    X_train, y_train, X_validate, y_validate, X_test, y_test
    '''
    # Create dummy variables for object dtypes
    # Original object dtype columns are dropped
    df = add_encoded_columns(df, drop_encoders=True)
    
    # After columns are coded, this function accepts a cleaned and encoded
    # dataframe and returns train, validate, and test sets
    train, validate, test = train_validate_test(df)
    
    # Split the train, validate, and test sets into 3 X_set and y_set
    X_train, y_train = attributes_target_split(train, target_name)
    X_validate, y_validate = attributes_target_split(validate, target_name)
    X_test, y_test = attributes_target_split(test, target_name)
    
    # If modeling is True
    if modeling:
        X_train, X_validate, X_test = add_scaled_columns(train, validate, test)
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def add_encoded_columns(df, drop_encoders=True):
    '''
    Signature: add_encoded_columns(df, drop_encoders=True)
    Docstring:
    This function accepts a DataFrame, creates encoded columns for object dtypes,
    and returns a DataFrame with or without object dtype columns.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    
    
    Returns
    -------
    f, encoded_columns
    '''
    if df.select_dtypes('O').columns.to_list() == []:
        return df
    
    columns_to_encode = df.select_dtypes('O').columns.to_list()
    encoded_columns = pd.get_dummies(df[columns_to_encode], drop_first=True, dummy_na=False)

    df = pd.concat([df, encoded_columns], axis=1)
    
    if drop_encoders:
        df =  df.drop(columns=columns_to_encode)
        return df
    else:
        return df, encoded_columns


def train_validate_test(df):
    '''
    Signature: train_validate_test(df)
    Docstring:

    Parameters
    ----------
    pandas.core.frame.DataFrame


    Returns
    -------
    train, validate, test
    '''
    train_validate, test = train_test_split(df, test_size=.20, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123)
    return train, validate, test


def attributes_target_split(data_set, target_name):
    '''
    Signature: attributes_target_split(df, target)
    Docstring:

    Parameters
    ----------
    pandas.core.frame.DataFrame


    Returns
    -------

    '''
    x = data_set.drop(columns=target_name)
    y = data_set[target_name]
    return x, y


def add_scaled_columns(train, validate, test, scaler=RobustScaler()):
    '''
    Signature: add_scaled_columns(train, validate, test, scaler)
    Docstring:

    Parameters
    ----------
    pandas.core.frame.DataFrame


    Returns
    -------
    train, validate, test
    '''
    columns_to_scale = train.select_dtypes(exclude='uint8').columns.to_list()
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    # scale columns in train, validate and test sets
    train_scaled = scaler.transform(train[columns_to_scale])
    validate_scaled = scaler.transform(validate[columns_to_scale])
    test_scaled = scaler.transform(test[columns_to_scale])
    
    # drop columns that are now scaled
    train.drop(columns=columns_to_scale, inplace=True)
    validate.drop(columns=columns_to_scale, inplace=True)
    test.drop(columns=columns_to_scale, inplace=True)
    
    # concatenate scaled columns with the original train/validate/test sets
    train = pd.concat([train,
                       pd.DataFrame(train_scaled,
                                    columns=new_column_names,
                                    index=train.index.values
                                   )],
                      axis=1)
    
    validate = pd.concat([validate,
                          pd.DataFrame(validate_scaled,
                                       columns=new_column_names,
                                       index=validate.index.values
                                      )],
                         axis=1)
    
    test = pd.concat([test,
                      pd.DataFrame(test_scaled,
                                   columns=new_column_names,
                                   index=test.index.values
                                  )],
                     axis=1)
    
    return train, validate, test


def features_for_modeling(predictors, target, k_features):
    '''
    Signature: features_for_modeling(predictors, target, k_features)
    Docstring:

    Parameters
    ----------

    Returns
    -------

    '''
    df_best = pd.DataFrame(select_kbest(predictors, target, k_features))
    df_rfe = pd.DataFrame(select_rfe(predictors, target, k_features))
    
    df_features = pd.concat([df_best, df_rfe], axis=1)
    return df_features


def select_kbest(predictors, target, k_features=3):
    '''
    Signature: select_kbest(predictors, target, k_features=3)
    Docstring:

    Parameters
    ----------
    pandas.core.frame.DataFrame

    Returns
    -------

    '''
    f_selector = SelectKBest(f_regression, k=k_features)
    f_selector.fit(predictors, target)
    
    f_mask = f_selector.get_support()
    f_features = predictors.iloc[:,f_mask].columns.to_list()
    
    print(f"Select K Best: {len(f_features)} features")
    print(f_features)
    return None
    # return predictors[f_features]
    
    
def select_rfe(X, y, k_features=3):
    '''
    Signature: rfe(predictors, target, k_features=3)
    Docstring:

    Parameters
    ----------
    pandas.core.frame.DataFrame

    Returns
    -------

    '''
    lm = LinearRegression()
    rfe_init = RFE(lm, k_features)
    
    X_rfe = rfe_init.fit(X, y)

    rfe_mask = rfe_init.support_    
    rfe_features = X.iloc[:, rfe_mask].columns.to_list()

    print(f"Recursive Feature Elimination: {len(rfe_features)} features")
    print(rfe_features)
    return None
    #return X[rfe_features]


