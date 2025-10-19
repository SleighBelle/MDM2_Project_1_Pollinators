import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import shap
# Data pre-process
df = pd.read_csv('interactions.csv')

focused = ['Pollinator Species', 'Caste', 'Plant Species', 'Latitude',
            'Longitude', 'Habitat', 'Month', 'Year', 'Date', 'Interactions']
df = df[focused]

from data_cleaner import drop_duplicate_rows_cols, model_seasonal_cycles
df = drop_duplicate_rows_cols(df)
df['sin_day'], df['cos_day'] = model_seasonal_cycles(df)

features = ['Pollinator Species', 'Caste', 'Plant Species', 'Latitude',
            'Longitude', 'Habitat', 'sin_day', 'cos_day']
x_raw = df[features]
y = df['Interactions']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
x_array = encoder.fit_transform(x_raw)
feature_names = encoder.get_feature_names_out(x_raw.columns)
x = pd.DataFrame(x_array, columns=feature_names)

# Model Definition
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
    "Linear Regression": LinearRegression(),
    "Lasso (L1)": LassoCV(cv=5, random_state=0),
    "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=0)
}

#Trianing
num_models = len(models)
cols = 3
rows = (num_models + cols - 1) // cols  # 自动计算行数
fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 5))
axes = axes.flatten()

top_n = 20

for idx, (name, model) in enumerate(models.items()):
    print(f"\n Training {name} ...")
    model.fit(x, y)
    y_pred = model.predict(x)
    score = r2_score(y, y_pred)

    # Abstract Feature Importance
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = model.coef_
    else:
        importance = abs(model.coefs_[0]).mean(axis=1)  # 适用于 MLP

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).head(top_n)

    sns.barplot(
        data=importance_df,
        x='Importance',
        y='Feature',
        palette='YlOrRd',
        ax=axes[idx]
    )

    axes[idx].set_title(f"{name}\nTop {top_n} Features\nR²={score:.2f}", fontsize=12)
    axes[idx].set_xlabel('Importance')
    axes[idx].set_ylabel('Feature')

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# SHAP analysis with Random Forest Model
shap_model = models["Random Forest"]
explainer = shap.Explainer(shap_model, x)
shap_values = explainer(x)

# Plot bar chart
shap.summary_plot(shap_values, x, plot_type="bar", max_display=20, show=False)
axes[idx + 1].imshow(plt.gcf().canvas.buffer_rgba())
axes[idx + 1].axis('off')
axes[idx + 1].set_title("Random Tree SHAP\nTop 20 Feature Importance (Bar)")

# Plot summary graph
plt.clf()
shap.summary_plot(shap_values, x, max_display=20, show=False)
axes[idx + 2].imshow(plt.gcf().canvas.buffer_rgba())
axes[idx + 2].axis('off')
axes[idx + 2].set_title("Random Tree SHAP\nTop 20 Feature Impact (Scatter)")

# Adjustment
for j in range(idx + 3, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


