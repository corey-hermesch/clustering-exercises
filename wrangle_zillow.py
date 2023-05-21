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
def get_zillow_data(sql_query= """
    SELECT *
    FROM (
            SELECT temp.parcelid as parcelid, p.logerror as logerror, temp.transactiondate as transactiondate
            FROM (
                SELECT parcelid, MAX(transactiondate) as transactiondate
                FROM predictions_2017
                WHERE (transactiondate >= '2017-01-01') and (transactiondate < '2018-01-01')
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
        df = pd.read_csv(filename, low_memory=False)
        print ("csv file found and read")
        return df
    else:
        url = get_db_url('zillow')
        
        df = pd.read_sql(sql_query, url)
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
def nulls_by_row(df):
    """
    """
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': pct_miss})

    # rows_missing = df.merge(rows_missing,
    #                     left_index=True,
    #                     right_index=True).reset_index()[['num_cols_missing', 'percent_cols_missing']]
    
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
        
    # fig, axes = plt.subplots(1, len(get_numeric_cols(df)), figsize=(15, 5))
    
    for i, col in enumerate(get_numeric_cols(df)):
        sns.histplot(df[col]) # , ax = axes[i])
        plt.title(f'Histogram of {col}') # axes[i].set_title(f'Histogram of {col}')
        plt.show()

# This function will remove rows containing outliers in any of the outlier_columns based on 
# being too far below Q1 or above Q3
def remove_outliers(df, outlier_columns, IQR_mult = 1.5):
    """
    This function will remove rows with outliers in any of the outlier_columns:
    - accept a dataframe with multiple columns and column types
    - accept a list of columns to check for outliers; all columns in outlier_columns should be numeric
    - accept an IQR multiplier (default value 3); 
        - all rows containing a value in one of the numeric columns that is (< (Q1 - 1.5*IQR)) or (> (Q3 + 1.5*IQR)) will be removed
    """
    
    # make new dataframe with just the outlier columns
    outlier_df = df[outlier_columns]
    
    # make a dataframe with the other columns to merge back in later
    non_out_cols = [col for col in df.columns if col not in outlier_columns]
    non_outlier_df = df[non_out_cols]
    
    # calculate Q1, Q3, IQR
    Q1 = outlier_df.quantile(0.25)
    Q3 = outlier_df.quantile(0.75)
    IQR = Q3 - Q1
    
    # remove rows from the outlier_df that are too far outside the IQR
    outlier_df_out = outlier_df[ ~ ((outlier_df < (Q1 - IQR_mult * IQR)) | (outlier_df > (Q3 + IQR_mult * IQR))).any(axis=1)]
    
    # merge df's back together keeping only the rows in the outlier_df_out
    return_df  = outlier_df_out.merge(non_outlier_df, how='inner', left_index=True, right_index=True)
    
    print (f'Incoming dataframe to remove_outliers had {df.shape[0]} rows.')
    print (f'Returned dataframe from remove_outliers has {return_df.shape[0]} rows.')
    
    return return_df

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

###### defining overall function to acquire and clean up zillow data
def wrangle_zillow():
    """
    This function will acquire and prep specific column data from zillow database. 
    Specifically it will:
    - 1. get data from database or csv if cached
    - 2. check for duplicate rows
    - 3. drop nulls in latitude and longitude columns if any
    - 4. drop all rows that aren't 'Single Family Residential'
    - 5. fill nulls in taxdelinquencyyear, poolcnt, hashottuborspa with 0
    - 6. fill nulls in airconditioningdesc with None
    - 7. fill nulls in heatingorsystemdesc with Other
    - 8. make a month_sold column from transactiondate
    - 9. make a has_halfbath column: 1 if bathrooms are x.5, 0 if bathrooms are x.0
    -10. remove columns with more than 60% missing values or rows with > 75% missing values
    -11. drop several remaining columns that are unhelpful or repeated information
    -12. drop remaining rows with nulls except for the column, buildingqualitytypeid (will impute later)
    -13. remove rows with bedrooms or bathrooms < 1
    -14. rename the columns to something more useful
    -15. 'fips' becomes 'county' with 3 values (LA, Orange, Ventura)
    -16. change the datatype to int where appropriate
    -17. drop outliers (using IQR and testing if too far below Q1 or above Q3)
    
    - returns a df that looks like this:
        - property_value - int (property_value is the target)
        - 20 other columns describing the property
        - dummy columns
    """
    # first get the data from csv or sql
    df = get_zillow_data()
    
    # check for duplicate rows
    if df.duplicated().sum() != 0: 
        print('Found duplicated rows. Check SQL query')
        return
    
    # check for and drop rows with nulls in latitude or longitude
    df = df.loc[~(df.latitude.isnull())]
    df = df.loc[~(df.longitude.isnull())]
    
    # Remove any properties that are not single unit properties; 
    df = df[df.propertylandusedesc=='Single Family Residential']
    
    # replace null values with 0 in taxdelinquencyyear
    df.taxdelinquencyyear = df.taxdelinquencyyear.fillna(0)
    
    # made an assumption that if poolcnt was null, it should be 0 (no pool)
    df.poolcnt = df.poolcnt.fillna(0)
    
    # made an assumption that if hashottubeorspa was null, it should be 0 (no hottub)
    df.hashottuborspa = df.hashottuborspa.fillna(0)
    
    # some google searching said less than half of homes in Los Angeles have air conditioning. 
    # So, I'm replacing the nulls with None. Since Wall Unit is only 16 rows, I'm changing those to 'Yes'
    df.airconditioningdesc = df.airconditioningdesc.fillna('None')
    df.airconditioningdesc = df.airconditioningdesc.map({'None': 'None', 'Central': 'Central'
                                                         , 'Yes': 'Yes', 'Wall Unit': 'Yes'})

    # heatingorsystemdesc has a good amount of non-nulls, so I'm just going to add an "Other" entry for the nulls
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('Other')
    
    # transactiondate is a string like '2017-01-01'; 
    # I am binning all these value into the month, i.e. a value 1-12
    df['month_sold'] = df.transactiondate.str[5:7].astype(int)
    
    # Drop columns/rows that had more than 60% nulls in the column or 75% nulls in the row
    df = handle_missing_values(df, .6, .75)
    
    # Drop any remain columns that won't help, i.e. the 'id' columns and a few others listed here:
    drop_cols = ['propertylandusetypeid', 'heatingorsystemtypeid', 'parcelid', 'id', 'propertycountylandusecode'
                 , 'propertyzoningdesc', 'rawcensustractandblock', 'structuretaxvaluedollarcnt', 'assessmentyear'
                 , 'landtaxvaluedollarcnt', 'taxamount', 'finishedsquarefeet12', 'calculatedbathnbr'
                 , 'fullbathcnt', 'unitcnt', 'propertylandusedesc', 'regionidcounty', 'censustractandblock'
                 , 'transactiondate']
    df = df.drop(columns=drop_cols)
    
    # for buildingqualitytypeid, there are enough non-nulls, I will keep it, but I'll need to impute 
    # it after train/val/test split
    # For the rest, I want to remove the rows that have null values as there won't be that many
    df = df.dropna(subset=df.columns.difference(['buildingqualitytypeid']))
    
    # outliers: 1. remove any rows with bathroomcnt < 1
    df = df[df.bathroomcnt >= 1]
    
    # outliers: 2. remove any rows with bedroomcnt < 1
    df = df[df.bedroomcnt >= 1]
    
    # feature engineering: make a has_halfbath
    # and since it is related to bathroomcnt, make bathroomcnt an integer 
    #   i.e. 3.5 becomes 3, with has_halfbath = 1 (True)
    df['has_halfbath'] = ((df.bathroomcnt % 1) == .5)
    df.has_halfbath = df.has_halfbath.astype(int)
    df.bathroomcnt = df.bathroomcnt.astype(int)

    # change the column names to something more meaningful and easier to use
    columns = ['logerror', 'bathrooms', 'bedrooms', 'bldg_quality_score', 'square_feet'
               , 'county', 'has_hottub', 'lat', 'long', 'lot_size_sqft'
               , 'has_pool', 'city_id', 'zip_code', 'rooms', 'year_built'
               , 'property_value', 'years_tax_delinquent'
               , 'ac_type', 'heating_type', 'month_sold', 'has_halfbath']
    df.columns = columns
    
    # change county (formerly fips) to map to the correct county
    df.county = df.county.map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    
    # change lat/long to the format degrees.decimal_degrees
    df.lat = df.lat / 1_000_000
    df.long = df.long / 1_000_000
    
    # outliers 2a. one stray zip code that didn't make any sense
    df = df[df.zip_code <= 99999]
    
    # outliers 2b. one value for years_tax_delinquent that was way out
    df = df[df.years_tax_delinquent < 20]
    
    # fill nulls with -1 for imputing later
    df.bldg_quality_score = df.bldg_quality_score.fillna(-1)
    
    # change many float columns to ints
    intcolumns = ['bedrooms', 'square_feet', 'has_hottub', 'lot_size_sqft'
                  , 'has_pool', 'city_id', 'zip_code', 'rooms', 'year_built', 'property_value'
                  , 'years_tax_delinquent']
    df[intcolumns] = df[intcolumns].astype(int)
    
    # outliers: 3. remove rows with values outside of 1.5*IQR above Q3 or below Q1 in these columns
    outlier_columns = ['bathrooms', 'bedrooms', 'square_feet', 'lot_size_sqft', 'property_value']
    df = remove_outliers(df, outlier_columns)
    
    # reorder the columns
    ordered_columns = ['property_value', 'logerror', 'square_feet', 'lot_size_sqft', 'bathrooms'
                   , 'has_halfbath', 'bedrooms', 'rooms', 'has_hottub', 'has_pool', 'ac_type'
                   , 'heating_type', 'month_sold', 'year_built'
                   , 'years_tax_delinquent', 'bldg_quality_score', 'lat', 'long', 'county', 'city_id'
                   , 'zip_code']
    df = df[ordered_columns]
        
    # make dummy columns for one categorical column, 'county'
    # city_id and zip_code may need dummies as well, but there are a lot of them
    dummy_df = pd.get_dummies(df[['county']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
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