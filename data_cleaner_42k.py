import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def ohe(df_x):
    # (No changes to this function)
    ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_cols = [col for col in df_x.columns if df_x[col].dtype in ['object', 'bool']]
    num_cols = [col for col in df_x.columns if col not in cat_cols]
    df_x_copy = df_x.copy()
    if cat_cols:
        x_cat = ohe_encoder.fit_transform(df_x_copy[cat_cols])
        x_cat_cols = ohe_encoder.get_feature_names_out(cat_cols)
        df_x_cat = pd.DataFrame(x_cat, columns=x_cat_cols, index=df_x_copy.index)
    else:
        df_x_cat = pd.DataFrame(index=df_x_copy.index)
    x_encoded = pd.concat([df_x_cat, df_x_copy[num_cols]], axis=1)
    return x_encoded

def model_seasonal_cycles(df):
    # (No changes to this function)
    df_copy = df.copy()
    df_copy['sin_week'] = np.sin(2 * np.pi * df_copy['Week'] / 52)
    return df_copy['sin_week']

def return_importance(x, feature_name, importances):
    # (No changes to this function, though it is not used in your main script)
    features = list(x.columns)
    idx_list = []
    for col in features:
        if feature_name.lower() in col.lower():
            idx_list.append(features.index(col))
    importance_weights = importances[idx_list]
    total = 0
    for weight in importance_weights:
        total += weight
    return total

def print_shap_pct(shap_values, x):
    # (No changes to this function, though it is not used in your main script)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp_df = (
        pd.DataFrame({'feature': x.columns, 'mean_abs_shap': mean_abs})
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    total = mean_abs.sum()
    for f, v in zip(x.columns, mean_abs):
        print(f"{f}: {100 * v / total:.2f}%")
    return 


def get_pair_interaction_weight(
    shap_values,         # ### NEW: Pass 2D main SHAP values ###
    shap_interactions,   # 3D interaction SHAP values
    x,                   # x_shap (features)
    y,                   # ### NEW: Pass y_shap (target) ###
    pollinator_name, 
    plant_name
):
    """
    Returns a dict with SHAP stats for a specific pollinator–plant pair.
    
    Includes:
    1. Network Strength (mean_abs_interaction)
    2. Temp Susceptibility (temp_susceptibility_total)
    3. Directional SHAP effects (mean_directional_shap_...)
    4. Baseline Count (mean_total_count_for_pair)
    """
    cols = x.columns.tolist()
    
    # --- Get feature columns and indices ---
    pollinator_cols = [c for c in cols if 'latin' in c and pollinator_name in c]
    plant_cols      = [c for c in cols if 'flower_visited' in c and plant_name in c]
    
    # Get numeric feature columns
    temp_cols   = [c for c in cols if c == 'temperature']
    wind_cols   = [c for c in cols if c == 'wind_speed']
    sun_cols    = [c for c in cols if c == 'sunshine']
    
    # Check if all required features are present
    if not all([pollinator_cols, plant_cols, temp_cols, wind_cols, sun_cols]):
        return None # Missing one of the key features
    
    poll_col = pollinator_cols[0]
    plant_col = plant_cols[0]
    temp_col = temp_cols[0]
    wind_col = wind_cols[0]
    sun_col = sun_cols[0]
    
    # Get indices for all features
    i = cols.index(poll_col)
    j = cols.index(plant_col)
    k_temp = cols.index(temp_col)
    k_wind = cols.index(wind_col)
    k_sun = cols.index(sun_col)
    
    # --- Create mask for this pair ---
    mask = (x[poll_col] == 1) & (x[plant_col] == 1)
    if mask.sum() == 0:
        return None
    
    # --- 1. Network Strength (from 3D interactions) ---
    pair_int_values = shap_interactions[mask][:, i, j]
    mean_abs_int = np.mean(np.abs(pair_int_values))

    # --- 2. Environmental Robustness (from 3D interactions) ---
    pol_temp_interactions = shap_interactions[mask][:, i, k_temp]
    plant_temp_interactions = shap_interactions[mask][:, j, k_temp]
    temp_susceptibility = np.mean(np.abs(pol_temp_interactions)) + np.mean(np.abs(plant_temp_interactions))
    
    # --- 3. NEW: Directional SHAP Effects (from 2D values) ---
    # This is the (Mean Directional SHAP) part of your formula
    shap_values_masked = shap_values[mask]
    
    mean_directional_shap_temp = shap_values_masked[:, k_temp].mean()
    mean_directional_shap_wind = shap_values_masked[:, k_wind].mean()
    mean_directional_shap_sun  = shap_values_masked[:, k_sun].mean()
    
    # --- 4. NEW: Baseline TotalCount (from y) ---
    # This is the (Baseline_TotalCount) part of your formula
    mean_total_count = y.loc[mask].mean()
    
    return {
        'pollinator_name': pollinator_name,
        'plant_name': plant_name,
        'n_rows_for_pair': int(mask.sum()),
        
        # Original metrics
        'mean_abs_interaction': mean_abs_int, 
        'temp_susceptibility_total': temp_susceptibility,
        
        # --- NEW OUTPUT COLUMNS ---
        'mean_total_count_for_pair': mean_total_count,
        'mean_directional_shap_temp': mean_directional_shap_temp,
        'mean_directional_shap_wind': mean_directional_shap_wind,
        'mean_directional_shap_sun': mean_directional_shap_sun
    }


def identify_upper_lower(df):
    # (No changes to this function)
    df_2 = df.copy()
    plant_totals = (
        df_2.groupby('flower_visited', as_index=False)['TotalCount']
            .sum()
            .rename(columns={'TotalCount': 'TotalCount_sum'})
    )
    top_10 = (
        plant_totals.sort_values('TotalCount_sum', ascending=False)
                    .head(5)
    )
    plant_list = top_10['flower_visited'].tolist()
    return plant_list