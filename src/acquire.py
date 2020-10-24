import pandas as pd
import env
import os

###################### SQL Connection w/ Credentials ######################
def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    Returns a formatted url with login credentials to access a SQL database.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


###################### Acquire Zillow Data ######################
def get_zillow_data():
    '''
    This function acquires the zillow dataset from Codeup's database using Pandas and SQL.
    It returns the zillow dataset as a Pandas DataFrame.
    
    If the zillow data is stored in the current repository, it returns a Pandas DataFrame.
    If the zillow data does not exist in the current directory, it queries Codeup's database for the dataset.
    A local copy will be created as a csv file in the current directory for future use.
    '''
    
    sql_query ='''
    select *
    from properties_2017
    join(select parcelid,
        logerror,
        max(transactiondate) as lasttransactiondate
        from predictions_2017
        group by parcelid, logerror
        ) as predictions using(parcelid)
    left join `airconditioningtype` using(`airconditioningtypeid`)
    left join `architecturalstyletype` using(`architecturalstyletypeid`)
    left join `buildingclasstype` using(`buildingclasstypeid`)
    left join `heatingorsystemtype` using(`heatingorsystemtypeid`)
    left join `propertylandusetype` using(`propertylandusetypeid`)
    left join `storytype` using(`storytypeid`)
    left join `typeconstructiontype` using(`typeconstructiontypeid`)
    where (latitude is not null
    and longitude is not null
    and propertylandusetypeid = 261);'''
    
    file = 'zillow.csv'
    
    if os.path.isfile(file):
        return pd.read_csv('zillow.csv')
    else:
        df = pd.read_sql(sql_query, get_connection('zillow'))
        df.to_csv('zillow.csv', index=False)
        return df