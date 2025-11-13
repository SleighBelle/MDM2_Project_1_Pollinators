import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------------- Global plotting style ------------------------- #
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)

# Color used to replace the default Matplotlib C0 blue for "observed" series
SKY = "skyblue"

# Default file paths (can be changed if needed)
EXCEL_BEEWALK_PATH = "(42k flowers) BeeWalk data 2008-23 31012024.xlsx"
CSV_BEEWALK_PATH = "50K_Beewalk.csv"


# ======================================================================== #
#                         Helper functions: Excel part                     #
# ======================================================================== #
def add_day_of_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the DataFrame has a 'day_of_year' column.
    Priority: existing 'day_of_year' > 'Date' > ('Year', 'Month', 'Day') > 'Week'.
    """
    if "day_of_year" in df.columns:
        return df

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["day_of_year"] = df["Date"].dt.dayofyear
    elif all(col in df.columns for col in ["Year", "Month", "Day"]):
        df["day_of_year"] = pd.to_datetime(
            dict(year=df["Year"], month=df["Month"], day=df["Day"])
        ).dt.dayofyear
    elif "Week" in df.columns:
        df["day_of_year"] = (df["Week"].astype(float) - 0.5) * 7.0
    else:
        raise KeyError(
            "Cannot construct 'day_of_year'. Provide 'day_of_year', 'Date', "
            "('Year', 'Month', 'Day'), or 'Week'."
        )

    return df


def add_circular_day_features(df: pd.DataFrame, day_col: str = "day_of_year") -> pd.DataFrame:
    """
    Add sine and cosine seasonal encodings for the given day-of-year column.
    """
    if day_col not in df.columns:
        raise KeyError(f"Column '{day_col}' not found for circular date features.")

    df["sin_day"] = np.sin(2.0 * np.pi * df[day_col] / 365.25)
    df["cos_day"] = np.cos(2.0 * np.pi * df[day_col] / 365.25)
    return df


def prepare_rubus_from_excel(
    file_path: str, rubus_name_candidates: list[str]
) -> pd.DataFrame:
    """
    Read the BeeWalk Excel file and extract the subset corresponding to Rubus fruticosus.
    """
    df_xl = pd.read_excel(file_path, sheet_name=0)

    if "flower_visited" not in df_xl.columns:
        raise KeyError("Column 'flower_visited' not found in the dataset.")

    df_rubus = df_xl[df_xl["flower_visited"].isin(rubus_name_candidates)].copy()
    if df_rubus.empty:
        raise ValueError("No records found for Rubus fruticosus in the dataset.")

    if "TotalCount" not in df_rubus.columns:
        raise KeyError("Column 'TotalCount' not found for interaction counts.")

    df_rubus["Interactions"] = df_rubus["TotalCount"]

    if "Year" not in df_rubus.columns:
        raise KeyError("Column 'Year' not found in the dataset.")

    df_rubus = add_day_of_year(df_rubus)
    df_rubus = add_circular_day_features(df_rubus, day_col="day_of_year")

    return df_rubus


def compute_annual_interactions(df_rubus: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate yearly total interactions for Rubus fruticosus.
    """
    annual = (
        df_rubus.groupby("Year", as_index=False)["Interactions"]
        .sum()
        .sort_values("Year")
    )
    return annual


def plot_annual_interactions(annual: pd.DataFrame) -> None:
    """
    Plot historical yearly interactions (observed only, in skyblue).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(
        annual["Year"],
        annual["Interactions"],
        marker="o",
        linestyle="-",
        color=SKY,
        label="Historical yearly interactions",
    )
    plt.xlabel("Year")
    plt.ylabel("Total pollination interactions")
    plt.title("Rubus fruticosus - Historical yearly pollination interactions")
    plt.legend()
    plt.tight_layout()
    plt.show()


def select_candidate_features(df_rubus: pd.DataFrame) -> list[str]:
    """
    Select candidate environmental and seasonal features for importance analysis.
    Excludes 'Week' and 'Year'.
    """
    candidate_features: list[str] = []

    if "temperature" in df_rubus.columns:
        candidate_features.append("temperature")

    wind_col = None
    if "wind_speed" in df_rubus.columns:
        wind_col = "wind_speed"
    elif "windspeed" in df_rubus.columns:
        wind_col = "windspeed"
    if wind_col is not None:
        candidate_features.append(wind_col)

    if "sunshine" in df_rubus.columns:
        candidate_features.append("sunshine")

    if "sin_day" in df_rubus.columns:
        candidate_features.append("sin_day")
    if "cos_day" in df_rubus.columns:
        candidate_features.append("cos_day")

    candidate_features = [c for c in candidate_features if c in df_rubus.columns]
    if not candidate_features:
        raise ValueError("No valid candidate features found for importance analysis.")

    print("\nUsing the following features for importance analysis (no Week, no Year):")
    print(candidate_features)

    return candidate_features


def compute_and_plot_rf_importance(
    df_rubus: pd.DataFrame, feature_cols: list[str], target_col: str = "Interactions"
) -> pd.Series:
    """
    Fit a RandomForestRegressor on the restricted feature set and plot feature importances.
    """
    df_model = df_rubus[feature_cols + [target_col]].dropna()
    if df_model.empty:
        raise ValueError("No valid rows left after dropping NaNs for selected features.")

    X = df_model[feature_cols]
    y = df_model[target_col].values

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=3,
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )

    print(
        "\nFeature importance for Rubus fruticosus interactions "
        "(restricted feature set):"
    )
    for feat, imp in importances.items():
        print(f"{feat}: {imp:.10f}")

    plt.figure(figsize=(10, 5))
    importances.plot(kind="bar", color=SKY)
    plt.ylabel("Relative importance")
    plt.title("Feature importance")
    plt.tight_layout()
    plt.show()

    return importances


def run_gbr_projection(
    df_rubus: pd.DataFrame, annual: pd.DataFrame, top_feature: str
) -> None:
    """
    Use GradientBoostingRegressor with the single most important feature
    to project yearly interactions 10 years into the future.
    """
    print("\nTop feature based on restricted Random Forest importances:", top_feature)

    yearly_top = (
        df_rubus.groupby("Year")
        .agg(
            total_interactions=("Interactions", "sum"),
            top_feature_mean=(top_feature, "mean"),
        )
        .dropna()
        .reset_index()
    )

    if yearly_top.empty:
        raise ValueError("No valid yearly data for top feature after aggregation.")

    print("\nYearly data used for Gradient Boosting (head5):")
    print(yearly_top.head())

    X_gb = yearly_top[["top_feature_mean"]]
    y_gb = yearly_top["total_interactions"].values

    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=1.0,
        random_state=42,
    )
    gbr.fit(X_gb, y_gb)

    trend_model = LinearRegression()
    trend_model.fit(yearly_top[["Year"]], yearly_top[["top_feature_mean"]])

    last_year = int(yearly_top["Year"].max())
    future_years = np.arange(last_year + 1, last_year + 11)
    future_years_df = pd.DataFrame({"Year": future_years})
    future_top_feature_mean = trend_model.predict(future_years_df).ravel()

    X_future = pd.DataFrame({"top_feature_mean": future_top_feature_mean})
    future_interactions = np.clip(gbr.predict(X_future), 0, None)

    plt.figure(figsize=(10, 5))
    plt.plot(
        annual["Year"],
        annual["Interactions"],
        marker="o",
        linestyle="-",
        color=SKY,
        label="Historical yearly interactions",
    )
    plt.plot(
        future_years,
        future_interactions,
        marker="o",
        linestyle="--",
        label=(
            "Predicted yearly interactions\n"
            f"(driven by {top_feature} trend, Gradient Boosting)"
        ),
    )
    plt.xlabel("Year")
    plt.ylabel("Total pollination interactions")
    plt.title(
        "Rubus fruticosus - Historical and projected interactions\n"
        f"(with respect to top feature: {top_feature})"
    )
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    print(
        "\nPredicted yearly interactions for the next 10 years "
        "(driven by top feature trend):"
    )
    for year, value in zip(future_years, future_interactions):
        print(f"Year {year}: predicted interactions ≈ {value:.1f}")

    print("\nYearly_top (used for GB training):")
    print(yearly_top)
    print("\nNaN counts in yearly_top:")
    print(yearly_top.isna().sum())


# ======================================================================== #
#                      Helper functions: CSV / yearly part                 #
# ======================================================================== #
def fill_env_by_year(group: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing environmental variables within each year using mode (or mean as fallback).
    """
    for col in ["sunshine_num", "wind_num"]:
        mode = group[col].mode(dropna=True)
        fill_val = mode.iloc[0] if len(mode) > 0 else group[col].mean()
        group[col] = group[col].fillna(fill_val)
    group["temperature"] = group["temperature"].fillna(group["temperature"].mean())
    return group


def prepare_yearly_aggregates(csv_path: str) -> pd.DataFrame:
    """
    Read CSV, filter Rubus fruticosus sp, encode sun/wind, fill missing values
    and aggregate yearly environmental variables and total interactions.
    """
    df_csv = pd.read_csv(csv_path)

    rubus = df_csv[df_csv["flower_visited"] == "Rubus fruticosus sp"].copy()

    # Encode categorical weather variables
    sun_map = {"Cloudy": 0, "Sunny/Cloudy": 1, "Sunny": 2}
    wind_map = {
        "None": 0,
        "0. Smoke rises vertically": 0,
        "1. Slight smoke drift": 1,
        "2. Wind felt on face, leaves rustle": 2,
        "3. Leaves and twigs in slight motion": 3,
        "4. Dust raised and small branches move": 4,
        "5. Small trees in leaf begin to sway": 5,
        "Light": 2,
        "Breezy": 3,
    }

    rubus["sunshine_num"] = rubus["sunshine"].map(sun_map)
    rubus["wind_num"] = rubus["wind_speed"].map(wind_map)

    # Fill missing environmental values year-wise
    rubus = rubus.groupby("Year", group_keys=False).apply(fill_env_by_year)

    yearly = (
        rubus.groupby("Year")
        .agg(
            yearly_total=("TotalCount", "sum"),
            mean_sunshine=("sunshine_num", "mean"),
            mean_wind=("wind_num", "mean"),
            mean_temp=("temperature", "mean"),
            n_rows=("TotalCount", "size"),
        )
        .reset_index()
        .sort_values("Year")
    )

    print("\nYearly aggregated data (CSV):")
    print(yearly)

    return yearly


def build_model_and_forecast(
    yearly_df: pd.DataFrame,
    year_start: int,
    year_end: int,
    horizon: int = 5,
    verbose_name: str | None = None,
):
    """
    Train a standardized Linear Regression model on a given window [year_start, year_end]
    using mean_sunshine, mean_wind, mean_temp as features, then extrapolate those
    features linearly and forecast yearly_total for the next `horizon` years.
    """
    df_train = yearly_df[
        (yearly_df["Year"] >= year_start) & (yearly_df["Year"] <= year_end)
    ].copy()
    if df_train.empty:
        raise ValueError(f"No data between {year_start} and {year_end}")

    features = ["mean_sunshine", "mean_wind", "mean_temp"]
    X_train = df_train[features].values
    y_train = df_train["yearly_total"].values

    model = Pipeline(steps=[("scaler", StandardScaler()), ("reg", LinearRegression())])
    model.fit(X_train, y_train)

    # Standardized coefficients and relative weights
    coefs = model.named_steps["reg"].coef_
    abs_sum = np.abs(coefs).sum()
    rel_weights = np.zeros_like(coefs) if abs_sum == 0 else np.abs(coefs) / abs_sum
    weights_df = pd.DataFrame(
        {"feature": features, "coef": coefs, "relative_weight": rel_weights}
    )

    # Linear extrapolation of features into the future
    years_train = df_train["Year"].values.reshape(-1, 1)
    last_year = df_train["Year"].max()
    future_years = np.arange(last_year + 1, last_year + 1 + horizon)

    future_feats = {}
    for fname in features:
        f_reg = LinearRegression()
        f_reg.fit(years_train, df_train[fname].values)
        future_feats[fname] = f_reg.predict(future_years.reshape(-1, 1))

    X_future = np.column_stack([future_feats[f] for f in features])
    y_future_pred = model.predict(X_future)

    pred_df = pd.DataFrame(
        {
            "Year": future_years,
            "predicted_interactions": np.round(y_future_pred).astype(int),
        }
    )

    label = verbose_name or f"{year_start}-{year_end}"
    print("=" * 70)
    print(f"Training window: {label}")
    print("\nFeature weights (standardized):")
    print(weights_df.round({"coef": 2, "relative_weight": 4}))
    print("\nNext", horizon, "years prediction:")
    print(pred_df)

    return weights_df, pred_df


def plot_lr_forecasts_and_weights(
    yearly: pd.DataFrame,
    weights_1: pd.DataFrame,
    pred_1: pd.DataFrame,
    weights_2: pd.DataFrame,
    pred_2: pd.DataFrame,
    year_start_1: int,
    year_end_1: int,
    year_start_2: int,
    year_end_2: int,
) -> None:
    """
    Plot observed vs. Linear Regression forecasts and their feature weights.
    """
    # Observed vs predicted yearly interactions
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        yearly["Year"],
        yearly["yearly_total"],
        marker="o",
        linestyle="-",
        color=SKY,
        label="Observed",
    )
    ax.plot(
        pred_1["Year"],
        pred_1["predicted_interactions"],
        marker="^",
        linestyle="--",
        label=f"Forecast (train {year_start_1}-{year_end_1})",
    )
    ax.plot(
        pred_2["Year"],
        pred_2["predicted_interactions"],
        marker="s",
        linestyle="--",
        label=f"Forecast (train {year_start_2}-{year_end_2})",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Rubus fruticosus sp yearly interactions")
    ax.set_title("Observed vs. 5-year forecasts (Linear Regression)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Relative weights
    features_lr = weights_1["feature"].tolist()
    x = np.arange(len(features_lr))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        weights_1["relative_weight"],
        width,
        color=SKY,
        label=f"Train {year_start_1}-{year_end_1}",
    )
    ax.bar(
        x + width / 2,
        weights_2["relative_weight"],
        width,
        label=f"Train {year_start_2}-{year_end_2}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(features_lr)
    ax.set_ylabel("Relative weight (|coef| normalized)")
    ax.set_title("Relative importance of sunshine, wind_speed, temperature")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Standardized coefficients
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        weights_1["coef"],
        width,
        color=SKY,
        label=f"Train {year_start_1}-{year_end_1}",
    )
    ax.bar(
        x + width / 2,
        weights_2["coef"],
        width,
        label=f"Train {year_start_2}-{year_end_2}",
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(features_lr)
    ax.set_ylabel("Standardized coefficient")
    ax.set_title("Standardized coefficients (direction and magnitude)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def build_gb_and_forecast(
    yearly_df: pd.DataFrame,
    year_start: int,
    year_end: int,
    horizon: int = 5,
    verbose_name: str | None = None,
):
    """
    Train a GradientBoostingRegressor on [year_start, year_end] using
    mean_sunshine, mean_wind, mean_temp as features and yearly_total as target.
    Then extrapolate features by linear trend over Year and forecast the next
    `horizon` years.
    """
    df_train = yearly_df[
        (yearly_df["Year"] >= year_start) & (yearly_df["Year"] <= year_end)
    ].copy()
    if df_train.empty:
        raise ValueError(f"No data between {year_start} and {year_end}")

    features = ["mean_sunshine", "mean_wind", "mean_temp"]
    X_train = df_train[features].values
    y_train = df_train["yearly_total"].values

    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=0,
    )
    gb.fit(X_train, y_train)

    importances = gb.feature_importances_
    imp_sum = importances.sum()
    rel_weights = importances if imp_sum == 0 else importances / imp_sum

    weights_df = pd.DataFrame(
        {"feature": features, "gb_importance": importances, "relative_weight": rel_weights}
    )

    years_train = df_train["Year"].values.reshape(-1, 1)
    last_year = df_train["Year"].max()
    future_years = np.arange(last_year + 1, last_year + 1 + horizon)

    future_feats = {}
    for fname in features:
        lr_f = LinearRegression()
        lr_f.fit(years_train, df_train[fname].values)
        future_feats[fname] = lr_f.predict(future_years.reshape(-1, 1))

    X_future = np.column_stack([future_feats[f] for f in features])
    y_future_pred = gb.predict(X_future)

    pred_df = pd.DataFrame(
        {"Year": future_years, "predicted_interactions": np.round(y_future_pred).astype(int)}
    )

    label = verbose_name or f"{year_start}-{year_end}"
    print("=" * 70)
    print(f"Gradient Boosting model training window: {label}")
    print("\nGradient Boosting feature importances (raw and relative):")
    print(weights_df.round({"gb_importance": 4, "relative_weight": 4}))
    print("\nGradient Boosting forecast for next", horizon, "years:")
    print(pred_df)

    return weights_df, pred_df


def plot_gb_forecasts_and_importance(
    yearly: pd.DataFrame,
    gb_weights_1: pd.DataFrame,
    gb_pred_1: pd.DataFrame,
    gb_weights_2: pd.DataFrame,
    gb_pred_2: pd.DataFrame,
    year_start_1: int,
    year_end_1: int,
    year_start_2: int,
    year_end_2: int,
) -> None:
    """
    Plot observed vs. Gradient Boosting forecasts and their feature importances.
    """
    # Observed vs predicted yearly interactions
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        yearly["Year"],
        yearly["yearly_total"],
        marker="o",
        linestyle="-",
        color=SKY,
        label="Observed",
    )
    ax.plot(
        gb_pred_1["Year"],
        gb_pred_1["predicted_interactions"],
        marker="^",
        linestyle="--",
        label=f"GB forecast (train {year_start_1}-{year_end_1})",
    )
    ax.plot(
        gb_pred_2["Year"],
        gb_pred_2["predicted_interactions"],
        marker="s",
        linestyle="--",
        label=f"GB forecast (train {year_start_2}-{year_end_2})",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Rubus fruticosus sp yearly interactions")
    ax.set_title("Observed vs. Gradient Boosting 5-year forecasts")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Relative importances
    features_gb = gb_weights_1["feature"].tolist()
    x = np.arange(len(features_gb))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        gb_weights_1["relative_weight"],
        width,
        color=SKY,
        label=f"GB train {year_start_1}-{year_end_1}",
    )
    ax.bar(
        x + width / 2,
        gb_weights_2["relative_weight"],
        width,
        label=f"GB train {year_start_2}-{year_end_2}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(features_gb)
    ax.set_ylabel("Relative importance (normalized)")
    ax.set_title("Gradient Boosting relative feature importance")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Raw importances
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        gb_weights_1["gb_importance"],
        width,
        color=SKY,
        label=f"GB train {year_start_1}-{year_end_1}",
    )
    ax.bar(
        x + width / 2,
        gb_weights_2["gb_importance"],
        width,
        label=f"GB train {year_start_2}-{year_end_2}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(features_gb)
    ax.set_ylabel("GB feature importance")
    ax.set_title("Gradient Boosting feature importances (raw)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ======================================================================== #
#                           High-level orchestration                       #
# ======================================================================== #
def run_excel_based_analysis(excel_path: str) -> None:
    """
    Full pipeline for the Excel-based Rubus analysis:
    - Read Rubus data
    - Compute yearly totals and plot
    - Compute RF feature importance and plot
    - Project future interactions with Gradient Boosting
    """
    rubus_name_candidates = [
        "Rubus fruticosus sp",
        "Rubus fruticosus",
        "Rubus fructicosus",  # include common misspelling
    ]

    df_rubus = prepare_rubus_from_excel(excel_path, rubus_name_candidates)

    # Annual interactions (historical)
    annual = compute_annual_interactions(df_rubus)
    plot_annual_interactions(annual)

    print("\nYearly total interactions for Rubus fruticosus:")
    print(annual)

    # Restricted feature set and RF importance
    candidate_features = select_candidate_features(df_rubus)
    importances = compute_and_plot_rf_importance(
        df_rubus, candidate_features, target_col="Interactions"
    )
    top_feature = importances.idxmax()

    # Gradient Boosting projection based on top feature
    run_gbr_projection(df_rubus, annual, top_feature)


def run_csv_based_analysis(csv_path: str) -> None:
    """
    Full pipeline for the CSV-based Rubus analysis:
    - Aggregate yearly data
    - Linear Regression forecasts for two windows + plots
    - Gradient Boosting forecasts for two windows + plots
    """
    yearly = prepare_yearly_aggregates(csv_path)

    # Linear Regression: two training windows, 5-year forecasts
    max_year = int(yearly["Year"].max())

    year_start_1, requested_end_1 = 2012, 2025
    year_end_1 = min(requested_end_1, max_year)
    weights_1, pred_1 = build_model_and_forecast(
        yearly_df=yearly,
        year_start=year_start_1,
        year_end=year_end_1,
        horizon=5,
        verbose_name=f"2012–{year_end_1} (requested 2012–2025)",
    )

    year_start_2, requested_end_2 = 2019, 2025
    year_end_2 = min(requested_end_2, max_year)
    weights_2, pred_2 = build_model_and_forecast(
        yearly_df=yearly,
        year_start=year_start_2,
        year_end=year_end_2,
        horizon=5,
        verbose_name=f"2019–{year_end_2} (requested 2019–2025)",
    )

    plot_lr_forecasts_and_weights(
        yearly=yearly,
        weights_1=weights_1,
        pred_1=pred_1,
        weights_2=weights_2,
        pred_2=pred_2,
        year_start_1=year_start_1,
        year_end_1=year_end_1,
        year_start_2=year_start_2,
        year_end_2=year_end_2,
    )

    # Gradient Boosting: two training windows, 5-year forecasts
    gb_weights_1, gb_pred_1 = build_gb_and_forecast(
        yearly_df=yearly,
        year_start=year_start_1,
        year_end=year_end_1,
        horizon=5,
        verbose_name=f"2012–{year_end_1} (requested 2012–2025)",
    )

    gb_weights_2, gb_pred_2 = build_gb_and_forecast(
        yearly_df=yearly,
        year_start=year_start_2,
        year_end=year_end_2,
        horizon=5,
        verbose_name=f"2019–{year_end_2} (requested 2019–2025)",
    )

    plot_gb_forecasts_and_importance(
        yearly=yearly,
        gb_weights_1=gb_weights_1,
        gb_pred_1=gb_pred_1,
        gb_weights_2=gb_weights_2,
        gb_pred_2=gb_pred_2,
        year_start_1=year_start_1,
        year_end_1=year_end_1,
        year_start_2=year_start_2,
        year_end_2=year_end_2,
    )


def main() -> None:
    run_excel_based_analysis(EXCEL_BEEWALK_PATH)
    run_csv_based_analysis(CSV_BEEWALK_PATH)


if __name__ == "__main__":
    main()
