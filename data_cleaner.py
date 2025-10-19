import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


"""
This function will intake a data frame and remove all rows and columns 
that are duplicates within the data frame. 
Cleaning the data this way improves processing time by cutting out unnecessary work.
"""

def drop_duplicate_rows_cols(df):


    # remove duplicated rows
    df_1 = df.drop_duplicates()

    # remove duplicated columns
    df_2_transpose = df_1.T.drop_duplicates()
    df_2 = df_2_transpose.T

    df_3 = df_2.dropna(axis = 0)

    return df_3


"""
This functin intakes a dataframe consisting of all the features, and encodes 
the necessary columns using pythons one hot encoder function. This allows the model
to identify boolean conditions for each string-column, without asigning weights to that
column based off of the alphabetical "weighting" of the string.
Numerical columns do not need to be encoded.
"""

def ohe(df_x):


    ohe = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)

    cat_cols = []
    num_cols = []

    for col in df_x:
        if df_x[col].dtype == 'object' or df_x[col].dtype == 'bool':
            # We only need the column name here as opposed to the column
            cat_cols.append(col)
        
        else:
            num_cols.append(col)

    x_cat = ohe.fit_transform(df_x[cat_cols])
    x_cat_cols = ohe.get_feature_names_out(cat_cols)
    df_x_cat = pd.DataFrame(x_cat, columns = x_cat_cols, index = df_x.index)

    x_encoded = pd.concat([df_x_cat, df_x[num_cols]], axis = 1)

    return x_encoded


"""
This function allows the model to identify the cyclic behaviour of years. Consiquently,
months are joined up (dates like 1/2/25 and 27/12/24 woud otherwise be far apart), and 
seasons with similar traits paired, like Oct and Jan.
In order to use timestamp, python required "year", "month", "day", hence the re-naming.  
"""

def model_seasonal_cycles(df):


    df['day'] = df['Date']
    df['month'] = df['Month']
    df['year'] = df['Year']

    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')


    # convert these dates into an intager to allow for numeric operations
    df['timestamp'] = df['datetime'].apply(lambda x: x.timestamp())

    # engneers cyclic patterns
    df['dayofyear'] = df['datetime'].dt.day_of_year
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

    return df['sin_day'], df['cos_day']

"""
This definition finds the weighting of every element class in the importances list
and sums them in order to return a single weighting of impoartance for each catagory.
the function itterates through, and finds every instance of each catagory, then adds 
their values to a list. These are then summed in order to return a single class-importance.
"""

def return_importance(x, feature_name, importances):


    features = list(x.columns)

    idx_list = []

    for col in features:
        if feature_name in col.lower():
            idx_list.append(features.index(col))
    
    importance_weights = importances[idx_list]

    total = 0
    for weight in importance_weights:
        total += weight

    return total

