import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data cleaning
df = pd.read_csv("50K_Beewalk.csv", encoding="latin1", low_memory=False)

cols = ['Transect.lat', 'Transect.long', 'latin', 'flower_visited']
df_map = df[cols].dropna(subset=cols)

df_map['Transect.lat'] = pd.to_numeric(df_map['Transect.lat'], errors='coerce')
df_map['Transect.long'] = pd.to_numeric(df_map['Transect.long'], errors='coerce')
df_map = df_map.dropna(subset=['Transect.lat', 'Transect.long'])

# Calculate plant_visited number
pollination_summary = (
    df_map.groupby(['Transect.lat', 'Transect.long', 'latin'])
    ['flower_visited']
    .nunique()
    .reset_index(name='flower_diversity')
)

# Plot 2D Graph

plt.figure(figsize=(30, 20))
sns.scatterplot(
    data=pollination_summary,
    x='Transect.long',
    y='Transect.lat',
    hue='latin',              # Divide by bee species
    size='flower_diversity',  # Adjust the size of Dot by the number of plant_visited
    sizes=(30, 300),
    alpha=0.7,
    palette='tab10'
)

plt.title("Bee Species and Flower Diversity across Locations", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
