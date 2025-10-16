import pandas as pd
import networkx as nx
from matplotlib.offsetbox import AnchoredText

import matplotlib.pyplot as plt
import random
import community as community_louvain

# Load the CSV file
file_path = r"C:\Users\archr\Downloads\6d8d5cb5-bd54-4da7-903a-15bd4bbd531b\data\plant_pollinator_interactions_for_potential_networks_2018.csv"
df = pd.read_csv(file_path)

# Replace these with the actual column names for pollinator and plant
pollinator_col = 'POLLINATOR_NAME'  # Example column name
plant_col = 'PLANT_NAME'            # Example column name

# Create a bipartite graph
G = nx.Graph()

# Add edges between pollinators and plants
for _, row in df.iterrows():
    pollinator = row[pollinator_col]
    plant = row[plant_col]
    G.add_edge(pollinator, plant)

# Draw the network
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=500, font_size=8)
plt.title("Pollinator-Plant Interaction Network")
plt.show()
# Improve visualization: color nodes by type, reduce label clutter

# Identify pollinator and plant nodes
pollinators = set(df[pollinator_col])
plants = set(df[plant_col])

# Assign colors
node_colors = []
for node in G.nodes():
    if node in pollinators:
        node_colors.append('skyblue')
    else:
        node_colors.append('lightgreen')

# Draw with better separation and smaller labels
plt.figure(figsize=(16, 10))
pos = nx.spring_layout(G, k=1.2, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=pollinators, node_color='skyblue', node_size=400, label='Pollinators')
nx.draw_networkx_nodes(G, pos, nodelist=plants, node_color='lightgreen', node_size=400, label='Plants')

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.5)

# Draw only a subset of labels to reduce clutter
labels = {}
for node in G.nodes():
    # Show label only if degree > 2 (connected to more than 2 nodes)
    if G.degree(node) > 2:
        labels[node] = node
nx.draw_networkx_labels(G, pos, labels, font_size=8)

plt.title("Pollinator-Plant Interaction Network (Improved Visualization)")
plt.legend(scatterpoints=1)
plt.axis('off')
plt.tight_layout()
# Option 1: Show only top N pollinators and plants by degree (most connected)
N = 10  # Number of top nodes to show

# Get top N pollinators and plants by degree
pollinator_degrees = sorted([(node, G.degree(node)) for node in pollinators], key=lambda x: x[1], reverse=True)[:N]
plant_degrees = sorted([(node, G.degree(node)) for node in plants], key=lambda x: x[1], reverse=True)[:N]

top_pollinators = set([node for node, _ in pollinator_degrees])
top_plants = set([node for node, _ in plant_degrees])

# Create subgraph with only top pollinators and plants
top_nodes = top_pollinators | top_plants
G_sub = G.subgraph(top_nodes)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_sub, k=1.2, seed=42)
nx.draw_networkx_nodes(G_sub, pos, nodelist=top_pollinators, node_color='skyblue', node_size=600, label='Pollinators')
nx.draw_networkx_nodes(G_sub, pos, nodelist=top_plants, node_color='lightgreen', node_size=600, label='Plants')
nx.draw_networkx_edges(G_sub, pos, alpha=0.7)
nx.draw_networkx_labels(G_sub, pos, font_size=10)
plt.title(f"Top {N} Pollinators and Plants (by degree)")
plt.legend(scatterpoints=1)
plt.axis('off')
plt.tight_layout()
plt.show()

# Option 2: Show only a random sample of edges (for a quick overview)
sample_size = 50  # Number of edges to show
edges_sample = random.sample(list(G.edges()), min(sample_size, G.number_of_edges()))
G_sample = nx.Graph()
G_sample.add_edges_from(edges_sample)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G_sample, k=1.2, seed=42)
nx.draw(G_sample, pos, with_labels=True, node_size=500, font_size=8)
plt.title(f"Random Sample of {sample_size} Edges")
plt.axis('off')
plt.tight_layout()
plt.show()
# Change N to 7 for top nodes
N = 10

# Get top N pollinators and plants by degree
pollinator_degrees = sorted([(node, G.degree(node)) for node in pollinators], key=lambda x: x[1], reverse=True)[:N]
plant_degrees = sorted([(node, G.degree(node)) for node in plants], key=lambda x: x[1], reverse=True)[:N]

top_pollinators = set([node for node, _ in pollinator_degrees])
top_plants = set([node for node, _ in plant_degrees])

# Create subgraph with only top pollinators and plants
top_nodes = top_pollinators | top_plants
G_sub = G.subgraph(top_nodes)

plt.figure(figsize=(14, 8))
pos = nx.spring_layout(G_sub, k=1.2, seed=42)
nx.draw_networkx_nodes(G_sub, pos, nodelist=top_pollinators, node_color='skyblue', node_size=600, label='Pollinators')
nx.draw_networkx_nodes(G_sub, pos, nodelist=top_plants, node_color='lightgreen', node_size=600, label='Plants')
nx.draw_networkx_edges(G_sub, pos, alpha=0.7)
nx.draw_networkx_labels(G_sub, pos, font_size=10)
plt.title(f"Top {N} Pollinators and Plants (by degree)")
plt.axis('off')

# Add ranking legend on the right

ranking_text = "Top Pollinators:\n" + "\n".join([f"{i+1}. {node} ({deg})" for i, (node, deg) in enumerate(pollinator_degrees)])
ranking_text += "\n\nTop Plants:\n" + "\n".join([f"{i+1}. {node} ({deg})" for i, (node, deg) in enumerate(plant_degrees)])

anchored_text = AnchoredText(ranking_text, loc='upper right', prop={'size': 10}, frameon=True)
plt.gca().add_artist(anchored_text)

plt.tight_layout()
plt.show()

# Bar chart for top pollinators and plants
fig, ax = plt.subplots(figsize=(10, 6))
names = [node for node, _ in pollinator_degrees] + [node for node, _ in plant_degrees]
degrees = [deg for _, deg in pollinator_degrees] + [deg for _, deg in plant_degrees]
colors = ['skyblue'] * N + ['lightgreen'] * N

bars = ax.barh(names, degrees, color=colors)
ax.set_xlabel('Degree (Number of Connections)')
ax.set_title(f'Top {N} Pollinators and Plants by Degree')
ax.invert_yaxis()
plt.tight_layout()
plt.show()
# Show bottom 10 pollinators and plants by degree
N_bottom = 10

# Get bottom N pollinators and plants by degree
pollinator_degrees_bottom = sorted([(node, G.degree(node)) for node in pollinators], key=lambda x: x[1])[:N_bottom]
plant_degrees_bottom = sorted([(node, G.degree(node)) for node in plants], key=lambda x: x[1])[:N_bottom]

bottom_pollinators = set([node for node, _ in pollinator_degrees_bottom])
bottom_plants = set([node for node, _ in plant_degrees_bottom])

# Create subgraph with only bottom pollinators and plants
bottom_nodes = bottom_pollinators | bottom_plants
G_sub_bottom = G.subgraph(bottom_nodes)

plt.figure(figsize=(14, 8))
pos_bottom = nx.spring_layout(G_sub_bottom, k=1.2, seed=42)
nx.draw_networkx_nodes(G_sub_bottom, pos_bottom, nodelist=bottom_pollinators, node_color='skyblue', node_size=600, label='Pollinators')
nx.draw_networkx_nodes(G_sub_bottom, pos_bottom, nodelist=bottom_plants, node_color='lightgreen', node_size=600, label='Plants')
nx.draw_networkx_edges(G_sub_bottom, pos_bottom, alpha=0.7)
nx.draw_networkx_labels(G_sub_bottom, pos_bottom, font_size=10)
plt.title(f"Bottom {N_bottom} Pollinators and Plants (by degree)")
plt.axis('off')

# Add ranking legend on the right
ranking_text_bottom = "Bottom Pollinators:\n" + "\n".join([f"{i+1}. {node} ({deg})" for i, (node, deg) in enumerate(pollinator_degrees_bottom)])
ranking_text_bottom += "\n\nBottom Plants:\n" + "\n".join([f"{i+1}. {node} ({deg})" for i, (node, deg) in enumerate(plant_degrees_bottom)])

anchored_text_bottom = AnchoredText(ranking_text_bottom, loc='upper right', prop={'size': 10}, frameon=True)
plt.gca().add_artist(anchored_text_bottom)

plt.tight_layout()
plt.show()

# Bar chart for bottom pollinators and plants
fig, ax = plt.subplots(figsize=(10, 6))
names_bottom = [node for node, _ in pollinator_degrees_bottom] + [node for node, _ in plant_degrees_bottom]
degrees_bottom = [deg for _, deg in pollinator_degrees_bottom] + [deg for _, deg in plant_degrees_bottom]
colors_bottom = ['skyblue'] * N_bottom + ['lightgreen'] * N_bottom

bars_bottom = ax.barh(names_bottom, degrees_bottom, color=colors_bottom)
ax.set_xlabel('Degree (Number of Connections)')
ax.set_title(f'Bottom {N_bottom} Pollinators and Plants by Degree')
ax.invert_yaxis()
plt.tight_layout()
plt.show()
# Check for modularity using the Louvain method (community detection)

# Project bipartite graph to pollinator-only unipartite graph for Louvain
pollinator_projection = nx.bipartite.projected_graph(G, pollinators)

# Compute the best partition on the projected graph
partition = community_louvain.best_partition(pollinator_projection)

# Get unique modules
modules = set(partition.values())
print(f"Number of modules detected: {len(modules)}")

# Draw each module as a separate figure
for module_id in modules:
    module_nodes = [node for node, mod in partition.items() if mod == module_id]
    G_module = G.subgraph(module_nodes)
    plt.figure(figsize=(10, 6))
    pos_module = nx.spring_layout(G_module, k=1.2, seed=42)
    # Color pollinators and plants differently
    pollinators_module = [node for node in module_nodes if node in pollinators]
    plants_module = [node for node in module_nodes if node in plants]
    nx.draw_networkx_nodes(G_module, pos_module, nodelist=pollinators_module, node_color='skyblue', node_size=500, label='Pollinators')
    nx.draw_networkx_nodes(G_module, pos_module, nodelist=plants_module, node_color='lightgreen', node_size=500, label='Plants')
    nx.draw_networkx_edges(G_module, pos_module, alpha=0.7)
    nx.draw_networkx_labels(G_module, pos_module, font_size=9)
    plt.title(f"Module {module_id} (size: {len(module_nodes)})")
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()