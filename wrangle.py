### IMPORTS 

# Imports
from env import host, user, password
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

### FUNCTIONS 

def get_db_url(db_name, user=user, host=host, password=password):
    '''
    get_db_url accepts a database name, username, hostname, password 
    and returns a url connection string formatted to work with codeup's 
    sql database.
    Default values from env.py are provided for user, host, and password.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

# generic function to get a sql pull into a dataframe
def get_mysql_data(sql_query, database):
    """
    This function will:
    - take in a sql query and a database (both strings)
    - create a connection url to mySQL database
    - return a df of the given query, connection_url combo
    """
    url = get_db_url(database)
    return pd.read_sql(sql_query, url)    

def get_csv_export_url(g_sheet_url):
    '''
    This function will
    - take in a string that is a url of a google sheet
      of the form "https://docs.google.com ... /edit#gid=12345..."
    - return a string that can be used with pd.read_csv
    '''
    csv_url = g_sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    return csv_url

# defining a function to get zillow data from either a cached csv or the Codeup MySQL server
# this function is specific to zillow, in particular because of the '%2017%' required in the 
# sql query
def get_zillow_data(sql_query= """
        SELECT *
        FROM (
                SELECT temp.parcelid as parcelid, p.logerror as logerror, temp.transactiondate as transactiondate
                FROM (
                    SELECT parcelid, MAX(transactiondate) as transactiondate
                    FROM predictions_2017
                    WHERE transactiondate LIKE %s
                    GROUP BY parcelid
                    ORDER BY transactiondate DESC
                    ) AS temp
                JOIN predictions_2017 as p ON p.parcelid = temp.parcelid AND p.transactiondate = temp.transactiondate
            ) AS p_2017
        JOIN properties_2017 USING(parcelid)
        LEFT JOIN airconditioningtype USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
                    """
                    , filename="zillow.csv"):
    
    """
    This function will:
    -input 2 strings: sql_query, filename 
        default query selects specific columns from zillow database that determined to be useful
        default filename "zillow.csv"
    - check the current directory for filename (csv) existence
      - return df from that filename if it exists
    - If csv doesn't exist:
      - create a df of the sql_query
      - write df to csv
      - return that df
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        print ("csv file found and read")
        return df
    else:
        url = get_db_url('zillow')
        
        # This is the hard-coding to replace the %s and ensure '%2017%' is in the query
        df = pd.read_sql(sql_query, url, params=['2017%'])
        df.to_csv(filename, index=False)
        print ("csv file not found, data read from sql query, csv created")
        return df

# generic function to return nulls by row (copied from Amanda)
def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return  cols_missing

# generic function to return nulls by row (copied from Amanda)
def nulls_by_row(df, index_id = 'customer_id'):
    """
    """
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': pct_miss})

    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True).reset_index()[[index_id, 'num_cols_missing', 'percent_cols_missing']]
    
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

# generic function to return columns which are objects (strings, primarily) (copied from Amanda)
def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

# generic function to return columns which are numeric (copied from Amanda)
def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols

# generic function to summarize a dataframe in the initial stages of wrangling (copied from Amanda)


def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    # distribution of numerical attributes
    '''
    print(f"""SUMMARY REPORT
=====================================================
          
          
Dataframe head: 
{df.head(3)}
          
=====================================================
          
          
Dataframe info: """)
    df.info()

    print(f"""=====================================================
          
          
Dataframe Description: 
{df.describe().T}
          
=====================================================


nulls in dataframe by column: 
{nulls_by_col(df)}
=====================================================


nulls in dataframe by row: 
{nulls_by_row(df)}
=====================================================
    
    
DataFrame value counts: 
 """)         
    for col in (get_object_cols(df)): 
        print(f"""******** {col.upper()} - Value Counts:
{df[col].value_counts()}
    _______________________________________""")                   
        
    fig, axes = plt.subplots(1, len(get_numeric_cols(df)), figsize=(15, 5))
    
    for i, col in enumerate(w.get_numeric_cols(df)):
        sns.histplot(df[col], ax = axes[i])
        axes[i].set_title(f'Histogram of {col}')
    plt.show()

# defining overall function to acquire and clean up zillow data
def wrangle_zillow():
    """
    This function will acquire and prep specific column data from zillow database. 
    Specifically it will:
    - 1. get data from database
    - 2. rename the columns to something more useful
    - 3. change has_pool nulls to 0
    - 3. drop the remaining nulls
    - 4. change the datatype to int for all numeric columns except bathrooms
    - 5. drop outliers (property_value > $1.5M and lotsize_sqft > 60000
    
    - returns a df that looks like this:
        - property_value - int (property_value is the target)
        - bathrooms - float
        - bedrooms - int
        - has_pool - int (0 or 1)
        - squarefeet - int
        - year - int
        - county - string
        - dummy columns for later modeling
            - county_Orange - int
            - county_Ventura - int
    """
    # first get the data from csv or sql
    df = get_zillow_data()
    
    #rename columns to something less unwieldy
    df.columns = ['parcelid', 'bathrooms', 'bedrooms', 'has_pool', 'squarefeet', 'lotsize_sqft', 'year'
                  , 'county', 'cityid', 'zip', 'lat', 'long', 'rawcensus', 'censustract_block', 'property_value']
    
    # has_pool has many nulls; I am making the assumption these properties do not have pools
    df.has_pool = df.has_pool.fillna(0)
    
    # decided to drop the rest of the nulls since it was < 3% of data
    df = df.dropna()
    
    # most numeric columns can/should be integers; exception was bathrooms which I left as a float
    intcolumns = ['bedrooms', 'has_pool', 'squarefeet', 'lotsize_sqft', 'year', 'county'
                  , 'cityid', 'zip', 'censustract_block', 'property_value']
    for col in intcolumns:
        df[col] = df[col].astype(int)
    
    # county really should be a categorical, since it represents a county in California
    df.county = df.county.map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})

    # if these are even useful, they will be useful in the format degrees.decimal_degrees
    df.lat = df.lat / 1_000_000
    df.long = df.long / 1_000_000
    
    # for now, I will keep only these columns. Maybe the others will have use later
    keep = ['property_value', 'bathrooms', 'bedrooms', 'has_pool', 'squarefeet', 'lotsize_sqft', 'year', 'county']
    df = df[keep]
    
    # After univariate visualization, I want to purge a few outliers
    # in total, 4371 rows were cut out of 52320; ~8.4% of the data
    df = df[df.property_value <= 1_500_000]
    df = df[df.lotsize_sqft <= 60_000]
    
    # make dummy columns for the categorical column, 'county'
    dummy_df = pd.get_dummies(df[['county']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df


# generic function to deal with missing values (nulls) in a more systematic way
def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - calculates the minimum number of non-missing values required for each column/row to be retained
    - drops columns/rows with a high proportion of missing values.
    - returns the new df
    """
    
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    
    return df


# generic function to remove rows and columns that have a high proportiion of missing values
def data_prep(df, col_to_remove=[], prop_required_columns=0.5, prop_required_rows=0.75):
    """
    This function will:
    - take in: 
        - a dataframe
        - list of columns
        - column threshold (defaulted to 0.5)
        - row threshold (defaulted to 0.75)
    - removes unwanted columns
    - remove rows and columns that contain a high proportion of missing values
    - returns cleaned df
    """
    df = df.drop(columns=cols_to_remove)
    df = handle_missing_values(df, prop_required_columns, prop_required_rows)
    return df



def split_function(df, target_var=''):
    """
    This function will
    - take in a dataframe (df)
    - option to accept a target_var in string format
    - split the dataframe into 3 data frames: train (60%), validate (20%), test (20%)
    -   while stratifying on the target_var (if present)
    - And finally return the three dataframes in order: train, validate, test
    """
    if len(target_var)>0:
        train, test = train_test_split(df, random_state=42, test_size=.2, stratify=df[target_var])
        train, validate = train_test_split(train, random_state=42, test_size=.25, stratify=train[target_var])
    else:
        train, test = train_test_split(df, random_state=42, test_size=.2)
        train, validate = train_test_split(train, random_state=42, test_size=.25)        
        
    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test

# copying this function to assist with choosing what type of scaler to use in the future
def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    """
    This function will
    - plot some charts of data before scaling next to data after scaling
    - returns nothing
    - example function call:
    
        # call function with minmax
        visualize_scaler(scaler=MinMaxScaler(), 
                         df=train, 
                         columns_to_scale=to_scale, 
                         bins=50)
    """
    #create subplot structure
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(12,12))

    #copy the df for scaling
    df_scaled = df.copy()
    
    #fit and transform the df
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    #plot the pre-scaled data next to the post-scaled data in one row of a subplot
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()

# defining a function to get scaled data using MinMaxScaler
def get_minmax_scaled (train, validate, test, columns_to_scale):
    """ 
    This function will
    - accept train, validate, test, and which columns are to be scaled
    - makes minmax scaler, fits scaler on train columns
    - returns 3 scaled dataframes; one for train/validate/test
    """
    # make copies for scaling
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # make and fit minmax scaler
    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])

    # use the thing
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    return train_scaled, validate_scaled, test_scaled