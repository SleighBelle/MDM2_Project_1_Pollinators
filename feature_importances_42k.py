import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import time

from data_cleaner_42k import (
    ohe,
    model_seasonal_cycles,
    get_pair_interaction_weight,
    identify_upper_lower
)

path = 'MDM2/csv_files/42k_flowers.csv'

df = pd.read_csv(path, encoding='latin-1', header=0)

for c in ['TotalCount', 'temperature', 'wind_speed', 'sunshine']:
    df[c] = pd.to_numeric(df[c], errors = 'coerce')

df['sin_week'] = model_seasonal_cycles(df)

popular_plant_list = identify_upper_lower(df)
print(f'Top plants: {popular_plant_list}')

df_top = df[df['flower_visited'].isin(popular_plant_list)]

'''
This will deal with multiple rows of the same instance of plants and pollinators.
If there are two rows seperately recording interactions between bees and daffodils, 
we combine the two so that we can gather all that relevent data. df will then get 
shorter as well to improve computatiponal efficiency. We are also reducing the 
reduncant 'year' column that gives no actionable data.
'''
agg_functions = {
    'TotalCount': 'sum',
    'temperature': 'mean',
    'wind_speed': 'mean',
    'sunshine': 'mean',
    'Year': 'first',
    'sin_week': 'first'
}
df_agg = df_top.groupby(['latin', 'flower_visited', 'Week']).agg(agg_functions).reset_index()

features = ['sunshine', 'wind_speed', 'temperature', 'sin_week', 'latin', 'flower_visited']

x_raw = df_agg[features]
y = df_agg['TotalCount']

x = ohe(x_raw)
y = y.loc[x.index]

model = RandomForestRegressor(n_estimators = 100, random_state = 0)
model.fit(x, y)

# print(len(x))

# We can use the line below for any models that are too big
#x_shap = shap.sample(x, 1000, random_state=0)

explainer = shap.TreeExplainer(
    model,
    feature_perturbation = "tree_path_dependent" # This is a chat GPT line. if it's not here I get a "warning" msg
)

'''
2D shap values - this will give us the extent to which each pair - pollinator 
pair is effected by changes in specific features at a time, e.g. Temprature,
or Wind-Speed or Sucshine.
'''
print("⏱️ Computing 2D SHAP values (main effects) on subsample...\n")
t0 = time.time()
shap_values_2d = explainer.shap_values(x)
t1 = time.time()
print("Elapsed for shap_values:", t1 - t0, "seconds\n")


'''
3D Shap values - This is the average of the value of all possible values of

'''
print("⏱️ Computing 3D SHAP interaction values on subsample...\n")
t0 = time.time()
shap_interactions_3d = explainer.shap_interaction_values(x)
t1 = time.time()
print("Elapsed for shap_interactions:", t1 - t0, "seconds\n")

df_agg_shap = df_agg.loc[x.index].copy()

all_pair_stats = []

print("⏱️ Computing relationship strengths for top plants...\n")

for plant in popular_plant_list:
    mask_plant = df_agg_shap['flower_visited'] == plant
    pollinators_for_plant = df_agg_shap.loc[mask_plant, 'latin'].unique()
    
    for pol in pollinators_for_plant:
        stats = get_pair_interaction_weight(
            shap_values_2d,
            shap_interactions_3d,
            x,
            y,
            pol,
            plant
        )
        
        if stats is None:
            continue
        
        all_pair_stats.append(stats)

pair_strengths = pd.DataFrame(all_pair_stats)

delta_C = {
    'temp': 2.0,   # +2°C by 2040s (UKCP18)
    'wind': 0.0,   # no change
    'sun':  0.056   # +12% hot-day chance on 1-5 sunshine scale → 0.12 * 4
}

# PFIS = Per Feature Importance Sore, this is ** NOT ** robustness
print("\n📝 Calculating per-feature PFIS and overall robustness...")

total_count = pair_strengths['mean_total_count_for_pair'] + 1e-6  # avoid deviding by 0

pair_strengths['PFIS_temp'] = (
    delta_C['temp'] * pair_strengths['mean_directional_shap_temp'] / total_count
)
pair_strengths['PFIS_wind'] = (
    delta_C['wind'] * pair_strengths['mean_directional_shap_wind'] / total_count
)
pair_strengths['PFIS_sun'] = (
    delta_C['sun'] * pair_strengths['mean_directional_shap_sun'] / total_count
)

pair_strengths['total_vulnerability'] = (
    pair_strengths['PFIS_temp'].abs()
    + pair_strengths['PFIS_wind'].abs()
    + pair_strengths['PFIS_sun'].abs()
)

pair_strengths['robustness_score'] = 1.0 / (pair_strengths['total_vulnerability'] + 1e-6)

pair_strengths = pair_strengths.sort_values(
    by='robustness_score',
    ascending=False
).reset_index(drop=True)

pair_strengths.to_csv('MDM2/outputs/strengths.csv', index = False)

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# import shap
# import time

# from data_cleaner_42k import (
#     ohe,
#     model_seasonal_cycles,
#     get_pair_interaction_weight,
#     identify_upper_lower,
# )

# path = 'MDM2/csv_files/42k_flowers.csv'

# df = pd.read_csv(path, encoding='latin-1', header=0)

# features_to_check = ['TotalCount', 'flower_visited', 'temperature',
#                      'wind_speed', 'sunshine', 'Year', 'Week']

# # Ensure numeric types for environmental variables and counts
# for c in ['TotalCount', 'temperature', 'wind_speed', 'sunshine']:
#     df[c] = pd.to_numeric(df[c], errors='coerce')

# # Add seasonal term (sin of week)
# df['sin_week'] = model_seasonal_cycles(df.copy())

# # 1) Identify the top 10 plants (from the full dataset)
# popular_plant_list = identify_upper_lower(df)
# print("Top plants:", popular_plant_list)

# # 2) Restrict the dataframe to ONLY those top plants
# df_top = df[df['flower_visited'].isin(popular_plant_list)].copy()

# # 3) Aggregate to plant–pollinator–week level, but ONLY within this subnetwork
# agg_functions = {
#     'TotalCount': 'sum',          # total visits for this pollinator–plant–week
#     'temperature': 'mean',
#     'wind_speed': 'mean',
#     'sunshine': 'mean',
#     'Year': 'first',              # keep a representative year (if needed)
#     'sin_week': 'first'           # same sin_week for all rows in this group
# }

# df_agg = df_top.groupby(['latin', 'flower_visited', 'Week']).agg(agg_functions).reset_index()

# # Features: environment + seasonal term + categorical pollinator/plant
# features = ['sunshine', 'wind_speed', 'temperature', 'sin_week', 'latin', 'flower_visited']

# x_raw = df_agg[features]
# y = df_agg['TotalCount']

# # One-hot encode categorical variables (latin, flower_visited)
# x = ohe(x_raw)
# y = y.loc[x.index]

# # 4) Fit Random Forest on ONLY the top-plant subnetwork
# model = RandomForestRegressor(n_estimators = 100, random_state = 0)
# model.fit(x, y)

# # 5) SHAP on this restricted network
# # Optional but recommended: subsample for speed
# x_shap = shap.sample(x, 1000, random_state=0)  # tweak 1000 if needed

# explainer = shap.TreeExplainer(
#     model,
#     feature_perturbation="tree_path_dependent"   # faster; or "interventional" if you prefer
# )

# print("⏱️ Computing SHAP interaction values on subsample...")
# t0 = time.time()
# shap_interactions = explainer.shap_interaction_values(x_shap)
# t1 = time.time()
# print("Elapsed for shap_interactions:", t1 - t0, "seconds")

# # Align the aggregated dataframe to the SHAP subset using index
# df_agg_shap = df_agg.loc[x_shap.index].copy()

# all_pair_stats = []

# print("⏱️ Computing relationship strengths for top plants...")

# for plant in popular_plant_list:
#     # Pollinators that actually interacted with this plant in the SHAP subset
#     mask_plant = df_agg_shap['flower_visited'] == plant
#     pollinators_for_plant = df_agg_shap.loc[mask_plant, 'latin'].unique()

#     for pol in pollinators_for_plant:
#         stats = get_pair_interaction_weight(shap_interactions, x_shap, pol, plant)

#         if stats is None:
#             continue  # skip pairs with no signal / missing columns

#         all_pair_stats.append(stats)

# # Turn into a nice DataFrame
# pair_strengths = pd.DataFrame(all_pair_stats)

# # Sort by strongest relationships first
# pair_strengths = pair_strengths.sort_values(
#     by = 'mean_abs_interaction',
#     ascending = True 
# ).reset_index(drop=True)

# # print("\nTop plant–pollinator relationships among the top plants:")
# # print(pair_strengths.head(30))

# df_strengths = pair_strengths.to_csv('MDM2/outputs/strengths.csv', index = False)

# ---------------------------------------------------------------------------------------------------------------------

#importances = model.feature_importances_

# temp_importance = return_importance(x, 'temperature', importances)
# windspeed_importance = return_importance(x, 'wind_speed', importances)
# sunshine_importance = return_importance(x, 'sunshine', importances)
# seasonal_importance = return_importance(x, 'sin_week', importances) # This gives us the seasonal importance, not the acc day

# print(f'\n --- Aggregated Importance Results (Focusing on Environment) ---')
# print(f' Temperature importance is: {temp_importance:.4f}')
# print(f' Windspeed importance is: {windspeed_importance:.4f}')
# print(f' Sunshine importance is: {sunshine_importance:.4f}')
# print(f' Seasonal (date) importance is: {seasonal_importance:.4f}')

# Good for tree models when you want interactions
# Build explainer
# explainer = shap.TreeExplainer(
#     model,
#     feature_perturbation="interventional"
# )

# x_shap = x

# print("⏱️ Computing SHAP interaction values on subsample...")
# shap_interactions = explainer.shap_interaction_values(x_shap)

# # Align the aggregated dataframe to the SHAP subset using index
# df_agg_shap = df_agg.loc[x_shap.index].copy()

# # Get all distinct pollinator-plant pairs that appear in these rows
# pairs = df_agg_shap[['latin', 'flower_visited']].drop_duplicates()

# # Collect relationship strengths here
# pair_stats_list = []

# print("⏱️ Computing pair-wise interaction strengths...")
# for _, row in pairs.iterrows():
#     pollinator = row['latin']
#     plant = row['flower_visited']

#     stats = get_pair_interaction_weight(shap_interactions, x_shap, pollinator, plant)

#     if stats is None:
#         continue

#     pair_stats_list.append(stats)

# # Convert to a nice DataFrame: one row = one plant-pollinator pair
# pair_strengths = pd.DataFrame(pair_stats_list)

# # Sort by relationship strength (descending)
# pair_strengths = pair_strengths.sort_values(
#     by='mean_abs_interaction',
#     ascending=False
# ).reset_index(drop=True)

# print("\nTop plant-pollinator relationships by SHAP interaction strength:")
# print(pair_strengths.head(20))