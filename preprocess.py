import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
import sklearn.feature_selection.rfe as rfe

def impute_missing_data(train, validate, test):
    '''
    
    '''
    numerical_columns = [
    'calculatedfinishedsquarefeet',
    'lotsizesquarefeet',
    'structuretaxvaluedollarcnt',
    'taxvaluedollarcnt',
    'landtaxvaluedollarcnt',
    'taxamount'
]
    
    categorical_columns = [
    "regionidcity",
    "regionidzip",
    "yearbuilt",
    "regionidcity"
]
    
    imputer = SimpleImputer(strategy='median')
    
    train[numerical_columns] = imputer.fit_transform(train[numerical_columns])
    validate[numerical_columns] = imputer.transform(validate[numerical_columns])
    test[numerical_columns] = imputer.transform(test[numerical_columns])
    
    
    imputer = SimpleImputer(strategy='most_frequent')
    
    train[categorical_columns] = imputer.fit_transform(train[categorical_columns])
    validate[categorical_columns] = imputer.transform(validate[categorical_columns])
    test[categorical_columns] = imputer.transform(test[categorical_columns])
    
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
    rfe_init = rfe(lm, k_features)

    rfe_mask = rfe_init.support_    
    rfe_features = X.iloc[:, rfe_mask].columns.to_list()

    print(f"Recursive Feature Elimination: {len(rfe_features)} features")
    print(rfe_features)
    return None
    #return X[rfe_features]

