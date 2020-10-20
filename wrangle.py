import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prepare_zillow_data(df, target_name):
    '''
    Signature: prepare_zillow_data(df, target_name)
    Docstring:
    This function accepts the zillow dataframe and splits it into train, validate,
    and test sets for data exploration.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
    
    target_name : str
        target_name is the column name of the target variable

    Returns
    -------
    X_train, y_train, X_validate, y_validate, X_test, y_test
    '''
    
    # After columns are coded, this function accepts a cleaned and encoded
    # dataframe and returns train, validate, and test sets
    train, validate, test = train_validate_test(df)
    
    # Split the train, validate, and test sets into 3 X_set and y_set
    X_train, y_train = attributes_target_split(train, target_name)
    X_validate, y_validate = attributes_target_split(validate, target_name)
    X_test, y_test = attributes_target_split(test, target_name)
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

    
def impute_values(train, validate, test):
    
    columns_to_impute = ['num_of_restrooms',
                         'living_room_area_sqft',
                         'lot_size_sqft',
                         'year_built',
                         'property_tax',
                         'structure_tax',
                         'land_tax',
                         'taxable_value']
    
    imputer = SimpleImputer(strategy='median')
    
    # impute missing values in train, validate and test sets
    train[columns_to_impute] = imputer.fit_transform(train[columns_to_impute])
    
    validate[columns_to_impute] = imputer.transform(validate[columns_to_impute])
    test[columns_to_impute] = imputer.transform(test[columns_to_impute])
    
    train.year_built = train.year_built.astype(np.int)
    validate.year_built = validate.year_built.astype(np.int)
    test.year_built = test.year_built.astype(np.int)
    
    return train, validate, test


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
    splits a dataframe into train, validate, and test sets for exploration/modeling
    Parameters
    ----------
    pandas.core.frame.DataFrame


    Returns
    -------
    train, validate, test
    '''
    # Add option to split data based on len of dataset
    train_validate, test = train_test_split(
        df,
        test_size=.20,
        random_state=123
    )
    
    train, validate = train_test_split(
        train_validate,
        test_size=.25,
        random_state=123
    )
    train, validate, test = impute_values(train, validate, test)
    return train, validate, test


def attributes_target_split(data_set, target_name):
    '''
    Signature: attributes_target_split(df, target)
    Docstring:
    transforms split data into seperate X_set, y_set
    
    Parameters
    ----------
    pandas.core.frame.DataFrame


    Returns
    -------

    '''
    x = data_set.drop(columns=target_name)
    y = data_set[target_name]
    return x, y


def add_scaled_columns(train, validate, test, scaler):
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