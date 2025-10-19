import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from data_cleaner import drop_duplicate_rows_cols, model_seasonal_cycles, ohe, return_importance


df = pd.read_csv('MDM2/csv_files/interactions.csv')

focused = ['Pollinator Species', 'Caste', 'Plant Species', 'Latitude', 
            'Longitude', 'Habitat', 'Month', 'Year', 'Date', 'Interactions']

df = df[focused]

df = drop_duplicate_rows_cols(df)

df['sin_day'], df['cos_day'] = model_seasonal_cycles(df)

columns = df.columns
# print(columns)

features = ['Pollinator Species', 'Caste', 'Plant Species', 'Latitude', 
            'Longitude', 'Habitat', 'sin_day', 'cos_day']

x_raw = df[features]
y = df['Interactions']

x = ohe(x_raw)
print(x.columns)

model = RandomForestRegressor(n_estimators = 100, random_state = 0)
model.fit(x, y)

importances = model.feature_importances_
print(importances)

# total = 0

# for element in importances:
#     total += element

# print(f'\n Total is: {total}')

latitude_importance = return_importance(x, 'latitude', importances)
longitude_importance = return_importance(x, 'longitude', importances)
caste_importance = return_importance(x, 'caste', importances)
habitat_importance = return_importance(x, 'habitat', importances)
seasonal_importance = return_importance(x, 'day', importances)

print(f'\n Latitude importance is: {latitude_importance}')
print(f'\n Longitde importance is: {longitude_importance}')
print(f'\n Caste importnace is: {caste_importance}')
print(f'\n Habitat type importance is: {habitat_importance}')
print(f'\n Sesonal imporance is: {seasonal_importance}')