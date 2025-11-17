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
    shap_values_2d,
    # shap_interactions_3d,   # UNUSED – commented out
    x,
    y,
    pollinator_name,
    plant_name
):
    cols = x.columns.tolist()

    # Identify the OHE columns for the pollinator and plant
    pollinator_cols = [c for c in cols if ("latin_" in c) and (pollinator_name in c)]
    plant_cols      = [c for c in cols if ("flower_visited_" in c) and (plant_name in c)]

    # Environmental feature columns
    temp_cols = [c for c in cols if c == "temperature"]
    wind_cols = [c for c in cols if c == "wind_speed"]
    sun_cols  = [c for c in cols if c == "sunshine"]

    if not all([pollinator_cols, plant_cols, temp_cols, wind_cols, sun_cols]):
        return None

    poll_col = pollinator_cols[0]
    plant_col = plant_cols[0]
    temp_col = temp_cols[0]
    wind_col = wind_cols[0]
    sun_col  = sun_cols[0]

    # mask for rows representing this plant–pollinator interaction
    mask = (x[poll_col] == 1) & (x[plant_col] == 1)
    if mask.sum() == 0:
        return None

    # 2D SHAP values (main effects)
    shap_values_masked = shap_values_2d[mask]

    # Mean directional SHAP values (2D only)
    mean_directional_shap_temp = shap_values_masked[:, x.columns.get_loc(temp_col)].mean()
    mean_directional_shap_wind = shap_values_masked[:, x.columns.get_loc(wind_col)].mean()
    mean_directional_shap_sun  = shap_values_masked[:, x.columns.get_loc(sun_col)].mean()

    # Mean interaction count
    mean_total_count = y.loc[mask].mean()

    # ----------------------------------------------------------------------
    # 3D SHAP interaction values – UNUSED (commented out)
    #
    # pair_int_values = shap_interactions_3d[mask][:, i, j]
    # mean_abs_interaction = np.mean(np.abs(pair_int_values))
    #
    # climate interactions:
    # pol_temp_interactions   = shap_interactions_3d[mask][:, i, k_temp]
    # plant_temp_interactions = shap_interactions_3d[mask][:, j, k_temp]
    # temp_susceptibility_total = (
    #     np.mean(np.abs(pol_temp_interactions)) 
    #     + np.mean(np.abs(plant_temp_interactions))
    # )
    # ----------------------------------------------------------------------

    return {
        "pollinator_name": pollinator_name,
        "plant_name": plant_name,
        "n_rows_for_pair": int(mask.sum()),
        "mean_directional_shap_temp": mean_directional_shap_temp,
        "mean_directional_shap_wind": mean_directional_shap_wind,
        "mean_directional_shap_sun":  mean_directional_shap_sun,
        "mean_total_count_for_pair": float(mean_total_count),

        # "mean_abs_interaction": mean_abs_interaction,   # UNUSED
        # "temp_susceptibility_total": temp_susceptibility_total,  # UNUSED
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
