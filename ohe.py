"""
this is the function that will one hot encode wind speed and any other
column into binary 1 or 0 columns in order to remove numerical bias
for columns that contain numerical values.
"""

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

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