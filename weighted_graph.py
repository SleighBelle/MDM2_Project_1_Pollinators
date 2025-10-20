import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\archr\Downloads\MDM2_Project_1_Pollinators-main\MDM2_Project_1_Pollinators-main\(50k flowers) BeeWalk data 2008-23 31012024.csv'
data = pd.read_csv(file_path)

# Check for required columns
required_columns = ['latin', 'flower_visited', 'sunshine', 'wind_speed', 'H1', 'start_time']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# Clean the sunshine and wind_speed columns to ensure they are integers
data['sunshine'] = pd.to_numeric(data['sunshine'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce').fillna(0).astype(int)
data['wind_speed'] = pd.to_numeric(data['wind_speed'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce').fillna(0).astype(int)

# Calculate the weight of each edge
edge_weights = data.groupby(['latin', 'flower_visited']).size().reset_index(name='weight')

# Create a network graph
G = nx.from_pandas_edgelist(edge_weights, 'latin', 'flower_visited', ['weight'])

# Get top 10 plants and top 10 pollinators
top_pollinators = edge_weights.groupby('latin')['weight'].sum().nlargest(10).index
top_plants = edge_weights.groupby('flower_visited')['weight'].sum().nlargest(10).index

# Filter the graph for top pollinators and plants
H = G.subgraph(top_pollinators.union(top_plants))

# Draw the network graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(H)
nx.draw(H, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
edge_labels = nx.get_edge_attributes(H, 'weight')
nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)
plt.title('Network Graph of Top 10 Pollinators and Plants')
plt.show()

# Print an array containing each weighting of each edge
weights_array = edge_weights['weight'].to_numpy()
print(weights_array)

# Save the weights array to a CSV file
weights_df = pd.DataFrame(weights_array, columns=['weight'])
weights_df.to_csv(r'C:\Users\archr\Downloads\weights_array.csv', index=False)  # Specify path

# Prepare the features and labels
# Select features and drop rows with NaN values
features = data[['sunshine', 'wind_speed', 'H1', 'start_time']].dropna()

# Create labels based on the original data
labels = []
for index, row in features.iterrows():
    latin = data.loc[row.name, 'latin']  # Get the corresponding 'latin' value
    flower_visited = data.loc[row.name, 'flower_visited']  # Get the corresponding 'flower_visited' value
    weight = edge_weights.loc[(edge_weights['latin'] == latin) & (edge_weights['flower_visited'] == flower_visited), 'weight']
    labels.append(weight.values[0] if not weight.empty else 0)  # Default to 0 if not found

# Ensure features and labels are aligned
if len(features) != len(labels):
    raise ValueError("Features and labels are not aligned in length.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Get feature importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))