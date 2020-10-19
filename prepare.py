# Write supporting functions here
import pandas as pd
import numpy as np

from acquire import get_zillow_data


def prepare_zillow():
    '''
    Signature: prepare_zillow() -> pandas.core.frame.DataFrame
    Docstring:
        Prepare the zillow dataset for data EDA
        Return DataFrame of zillow dataset
    '''
    df = get_zillow_data()
    
    df.storydesc.fillna(0, inplace=True)
    df.storydesc.replace('Basement', 1, inplace=True)
    df['has_basement'] = df.storydesc
    
    df.hashottuborspa.fillna(0, inplace=True)
    df['has_hottub_or_spa'] = df.hashottuborspa
    
    df.poolcnt.fillna(0, inplace=True)
    df['has_pool'] = df.poolcnt
    
    df.poolsizesum.fillna(0, inplace=True)
    df['pool_area_sqft'] = df.poolsizesum
    
    df['has_patio'] = df.yardbuildingsqft17.notnull().astype(np.int)
    df['patio_area_sqft'] = df.yardbuildingsqft17.fillna(0)
    
    df['has_shed'] = df.yardbuildingsqft26.notnull().astype(np.int)
    df['basement_area_sqft'] = df.yardbuildingsqft26.fillna(0)
    
    bathroom_median = df.calculatedbathnbr.median()
    df.calculatedbathnbr.fillna(bathroom_median, inplace=True)
    df.calculatedbathnbr = df.calculatedbathnbr.replace(0, bathroom_median)
    df.rename(columns={'calculatedbathnbr': 'num_of_restrooms'}, inplace=True)
    
    bedroomcnt_median = df.bedroomcnt.median()
    df.bedroomcnt.fillna(bedroomcnt_median, inplace=True)
    df.bedroomcnt = df.bedroomcnt.replace(0, bedroomcnt_median)
    df.rename(columns={'bedroomcnt': 'num_of_bedrooms'}, inplace=True)
    
    median_lot_in_sqft = df.lotsizesquarefeet.median()
    df['lot_size_sqft'] = df.lotsizesquarefeet.fillna(median_lot_in_sqft)
       
    features_to_drop = [
    'decktypeid',
    'buildingclasstypeid',
    'buildingqualitytypeid',
    'finishedsquarefeet6',
    'finishedsquarefeet13',
    'finishedsquarefeet15',
    'buildingclassdesc',
    'pooltypeid2',
    'pooltypeid7',
    'pooltypeid10',
    'regionidcity',
    'regionidcounty',
    'regionidzip',    
    'typeconstructiontypeid',
    'typeconstructiondesc',
    'architecturalstyletypeid',
    'architecturalstyledesc',
    'storytypeid',
    'storydesc',
    'hashottuborspa',
    'poolcnt',
    'poolsizesum',
    'yardbuildingsqft17',
    'yardbuildingsqft26',
    'taxdelinquencyyear',
    'taxdelinquencyflag',
    'finishedsquarefeet50',
    'finishedfloor1squarefeet',
    'censustractandblock',
    'rawcensustractandblock',
    'propertylandusetypeid',
    'id',
    'assessmentyear',
    'finishedsquarefeet12',
    'bathroomcnt',
    'fullbathcnt',
    'basementsqft',
    'threequarterbathnbr',
    'lotsizesquarefeet',
    'propertylandusedesc',
    'propertycountylandusecode',
    'regionidcounty',
]
   
    
    df.drop(columns=features_to_drop, inplace=True)
    
    df.rename(columns={'calculatedfinishedsquarefeet': 'living_room_area_sqft',
                      'structuretaxvaluedollarcnt': 'structure_tax',
                      'taxvaluedollarcnt': 'taxable_value',
                      'landtaxvaluedollarcnt': 'land_tax',
                      'taxamount': 'property_tax',
                      'lasttransactiondate': 'date_sold',
                      'roomcnt' : 'num_of_rooms',
                      'yearbuilt': 'year_built'},
         inplace=True)
    
    
    df = handle_missing_values(df)
    columns_to_impute = df.isna().sum()[df.isna().sum()>0].index.to_list()
    
    for column_name in columns_to_impute:
        median = df[column_name].median()
        df[column_name] = df[column_name].fillna(median)
    
    df.year_built = df.year_built.astype(np.int)
    df.fips = df.fips.astype(np.int)
    df.has_hottub_or_spa = df.has_hottub_or_spa.astype(np.int)
    df.has_pool = df.has_pool.astype(np.int)
    df.drop(columns='num_of_rooms', inplace=True)
    df.has_hottub_or_spa = df.has_hottub_or_spa.astype('int')
    
    df = df[['parcelid',
             'num_of_bedrooms',
             'num_of_restrooms',
             'living_room_area_sqft',
             'lot_size_sqft',
             'year_built',
             'has_basement',
             'has_hottub_or_spa',
             'has_pool',
             'pool_area_sqft',
             'has_patio',
             'patio_area_sqft',
             'has_shed',
             'basement_area_sqft',
             'property_tax',
             'structure_tax',
             'land_tax',
             'taxable_value',
             'date_sold',
             'fips',
             'latitude',
             'longitude',
             'logerror']]
    
    
    return df


def missing_values_summary(df, observations=False):
    '''
    
    '''
    
    # If observations=True, calculate the number of observations that have the
    # same amount of missing values.
    if observations:
        num_columns_missing = df.isnull().sum(axis=1).value_counts().sort_index()
        
    # Using `isnull()` and `notnull()` we can calculate the number of missing values and non-null values.
    nulls = df.isnull().sum()
    non_nulls = df.notnull().sum()

    # Add missing values and non-null values together to get the total number values in each column.
    total_values = nulls + non_nulls

    # Create a variable to store the percentage of missing values in each column.
    pct_missing = (nulls/total_values)
    
    # If observations=True: Return groups of observations with the same number of missing values.
    if observations:
        df = pd.DataFrame({'num_cols_missing': num_columns_missing.index,
                           'pct_cols_missing': (num_columns_missing.index/df.shape[1]),
                           'num_rows': df.isnull().sum(axis=1).value_counts().sort_index()
                          }).reset_index(drop=True)
    # Else: Return the number of missing values for each attribute.
    else:
        df = pd.DataFrame({'attribute':pct_missing.index.values,
                           'num_rows_missing':nulls.values,
                           'pct_rows_missing':pct_missing.values
                          })
    return df


def handle_missing_values(df, prop_required_column =.75, prop_required_row =.75):
    '''
    
    '''
    # Threshold variable holds the equivalent of 75% of total rows in a dataframe
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    
    # Threshold variable holds the equivalent of 75% of total columns in a dataframe
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df